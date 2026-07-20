#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_name="shirt-fold-hf-embodiment-adapter-5k-v1-local-cpu"
run_root="$repo_root/adapter_cache/$run_name"
cache_path="$run_root/frozen_predictions.npz"
split_path="$repo_root/adapter_cache/shirt-fold-hf-embodiment-adapter-5k-v1/split_info.json"
manifest="$repo_root/configs/embodiment_adapters/hf_folding_to_anvil_openarm2.json"
dataset="$repo_root/datasets/shirt-fold/lerobot-hf-phase-aligned"
base_policy="$repo_root/model_zoo/hf-folding-final"
output="$repo_root/model_zoo/openarm2-shirt-fold-phase-aligned-v1/$run_name/checkpoints/005000"
log_path="$run_root/local_run.log"

mkdir -p "$run_root" "$(dirname "$output")"
exec > >(tee -a "$log_path") 2>&1

export PYTHONPATH="$repo_root/packages/anvil_embodiment/src:$repo_root/packages/anvil_shared/src:$repo_root/packages/anvil_eval/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

cloud_env="$HOME/dev/openarm2-cloud-runner/.env"
if [[ ! -f "$cloud_env" ]]; then
  echo "missing Hugging Face credential-path environment: $cloud_env" >&2
  exit 1
fi
set -a
source "$cloud_env"
set +a
if [[ -z "${HF_TOKEN_PATH:-}" || ! -f "$HF_TOKEN_PATH" ]]; then
  echo "HF_TOKEN_PATH does not resolve to a token file" >&2
  exit 1
fi
export HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_PATH")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

write_completion() {
  local status="$1"
  RUN_STATUS="$status" RUN_ROOT="$run_root" OUTPUT="$output" .venv/bin/python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "status": int(os.environ["RUN_STATUS"]),
    "device": "cpu",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "output": os.environ["OUTPUT"],
}
Path(os.environ["RUN_ROOT"], "completion.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
}
job_completed=0
finalize() {
  local status="$1"
  trap - EXIT
  if [[ "$job_completed" != 1 && "$status" -eq 0 ]]; then
    status=125
  fi
  write_completion "$status"
  exit "$status"
}
trap 'finalize "$?"' EXIT

echo "event=local_adapter_start device=cpu run_name=$run_name timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
.venv/bin/python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("datasets/shirt-fold/lerobot-hf-phase-aligned/meta/trim_manifest.json").read_text()
)
assert manifest["source"]["episodes"] == 33
assert manifest["summary"]["input_frames"] == 43_625
assert manifest["summary"]["output_frames"] == 34_850
assert len(manifest["episodes"]) == 33
assert all(episode["removed_start_frames"] > 0 for episode in manifest["episodes"])
assert sum(episode["removed_end_frames"] for episode in manifest["episodes"]) > 0
end_trimmed = sum(episode["removed_end_frames"] > 0 for episode in manifest["episodes"])
print(
    "event=trim_contract_verified episodes=33 input_frames=43625 "
    f"output_frames=34850 start_trimmed_episodes=33 end_trimmed_episodes={end_trimmed}"
)
PY

.venv/bin/python -m anvil_embodiment.cli validate \
  --manifest "$manifest" \
  --base-policy "$base_policy" \
  --dataset "$dataset" \
  --stride 400 \
  | tee "$run_root/validation.json"

if [[ ! -f "$cache_path" ]]; then
  .venv/bin/python -m anvil_embodiment.cli cache \
    --manifest "$manifest" \
    --base-policy "$base_policy" \
    --dataset "$dataset" \
    --split-info "$split_path" \
    --output "$cache_path" \
    --task "Fold the T-shirt properly" \
    --device cpu \
    --stride 10 \
    --seed 42
else
  echo "event=cache_resume path=$cache_path"
fi

.venv/bin/python -m anvil_embodiment.cli train \
  --manifest "$manifest" \
  --cache "$cache_path" \
  --output "$output" \
  --device cpu \
  --steps 5000 \
  --batch-size 64 \
  --eval-every 100 \
  --seed 42 \
  --wandb-project openarm2-shirt-folding \
  --wandb-run-name "$run_name" \
  --wandb-mode online

echo "event=local_adapter_complete timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ) output=$output"
job_completed=1
