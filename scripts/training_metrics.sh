#!/usr/bin/env bash
# Quick training benchmark: runs ACT / Diffusion / Pi0.5 for 100 steps each
# and reports wall-clock time per model.
#
# Usage:
#   ./scripts/benchmark_training.sh [--models act,diffusion,pi05] [--steps 100]
#
# Hyperparameters mirror the model_zoo training runs for this dataset.
# Wandb is disabled; no checkpoints are saved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STEPS=100
MODELS="act,diffusion,pi05"

for arg in "$@"; do
    case "$arg" in
        --steps=*)   STEPS="${arg#*=}" ;;
        --models=*)  MODELS="${arg#*=}" ;;
    esac
done

DATASET_ROOT="data/datasets/placing-one-block-r4-fisheye"
OUT_DIR="/tmp/anvil_train_benchmark"

COMMON=(
    "--dataset.root=${DATASET_ROOT}"
    "--dataset.repo_id=local"
    "--steps=${STEPS}"
    "--eval_freq=0"
    "--save_freq=999999"
    "--log_freq=10"
    "--wandb.enable=false"
)

declare -A RESULTS

run_model() {
    local name="$1"
    shift
    local model_out="${OUT_DIR}/${name}"
    rm -rf "$model_out"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Training: ${name} (${STEPS} steps)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local t0 t1 elapsed
    t0=$(date +%s%3N)

    uv run --directory "$REPO_ROOT" anvil-trainer \
        "${COMMON[@]}" \
        "--output_dir=${model_out}" \
        "--job_name=${name}_bench" \
        "$@"

    t1=$(date +%s%3N)
    elapsed=$(( t1 - t0 ))

    local secs=$(( elapsed / 1000 ))
    local ms=$(( elapsed % 1000 ))
    local per_step
    per_step=$(python3 -c "print(f'{${elapsed} / ${STEPS} / 1000:.2f}')")

    RESULTS[$name]="${secs}.${ms}s total | ${per_step}s/step"
    echo "  Done: ${secs}.${ms}s total (${per_step}s/step)"
}

mkdir -p "$OUT_DIR"

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
for model in "${MODEL_LIST[@]}"; do
    case "$model" in
        act)
            run_model "act" \
                "--policy.type=act" \
                "--policy.normalization_mapping={\"ACTION\":\"MEAN_STD\",\"STATE\":\"MEAN_STD\",\"VISUAL\":\"IDENTITY\"}" \
                "--policy.vision_backbone=resnet18" \
                "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1" \
                "--batch_size=16"
            ;;
        diffusion)
            run_model "diffusion" \
                "--policy.type=diffusion" \
                "--policy.normalization_mapping={\"ACTION\":\"MIN_MAX\",\"STATE\":\"MEAN_STD\",\"VISUAL\":\"IDENTITY\"}" \
                "--policy.n_action_steps=16" \
                "--policy.horizon=24" \
                "--policy.down_dims=[256,512,1024]" \
                "--policy.vision_backbone=resnet18" \
                "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1" \
                "--policy.use_group_norm=false" \
                "--batch_size=8"
            ;;
        pi05)
            run_model "pi05" \
                "--policy.type=pi05" \
                "--policy.pretrained_path=lerobot/pi05_base" \
                "--policy.dtype=bfloat16" \
                "--policy.gradient_checkpointing=true" \
                "--policy.compile_model=true" \
                "--policy.train_expert_only=true" \
                "--policy.normalization_mapping={\"ACTION\":\"MEAN_STD\",\"STATE\":\"MEAN_STD\",\"VISUAL\":\"IDENTITY\"}" \
                "--policy.chunk_size=50" \
                "--policy.n_action_steps=50" \
                "--batch_size=4" \
                "--num_workers=0"
            ;;
        *)
            echo "Unknown model: $model (valid: act, diffusion, pi05)"
            exit 1
            ;;
    esac
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Benchmark Results (${STEPS} steps each)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for model in "${MODEL_LIST[@]}"; do
    printf "  %-12s %s\n" "${model}:" "${RESULTS[$model]:-SKIPPED}"
done
echo ""
