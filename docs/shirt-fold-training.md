[← Back to README](../README.md)

# OpenArm 2 Shirt-fold Training

This runbook defines a controlled 5,000-step A/B test for the Anvil OpenArm 2
shirt-fold dataset. Run these two fine-tunes before spending a larger training
budget or training the embodiment adapter:

| Run | Initialization | Recipe |
|---|---|---|
| HF prior | Hugging Face OpenArm shirt-fold policy | `configs/training/shirt_fold_pi05_hf_phase_aligned.yaml` |
| Base prior | Upstream Pi0.5 base policy | `configs/training/shirt_fold_pi05_base_phase_aligned.yaml` |

The recipes intentionally differ only in the pretrained model/revision and run
identity. This isolates whether Hugging Face's folding prior helps despite its
modified OpenArm Mini embodiment.

## Dataset contract

Both runs use the generated
`datasets/shirt-fold/lerobot-hf-phase-aligned` dataset:

- 33 episodes and 34,850 frames at 30 Hz;
- three video observations:
  `left_wrist`, `right_wrist`, and `base`, each 3×270×480;
- 16D state/action vectors in radians, ordered right arm first, then left arm;
- seven arm joints plus one gripper value per side;
- `q01` and `q99` statistics required by Pi0.5's `QUANTILES`
  normalization.

The trim is reproducible from
`lerobot-hf-phase-aligned.trim-plan.json`; generated datasets and adapter
caches remain local or in artifact storage and are not committed to Git.
Do not concatenate the Hugging Face OpenArm Mini demonstrations with the Anvil
episodes. Their left-first ordering, degree units, and different link geometry
make them a reference dataset, not compatible training rows.

Trimming removes the homing/setup phase and final idle phase. Consequently,
inference must begin near a demonstrated `start_state` from
`meta/trim_manifest.json`, not from the mechanical home pose.

## Why these settings are paired

- The policy's native relative-action processor is enabled, excluding grippers.
  Pass `--action-type=absolute` to Anvil so a second delta transform is not
  applied.
- The explicit action names lock the right-first joint mapping.
- The explicit 16D features override Pi0.5 base's incompatible 32D defaults.
- The 30-action horizon matches the HF folding checkpoint and covers one second
  at 30 Hz.
- Both runs train the action expert for 5,000 steps at `1e-5`, log every 100
  steps, and checkpoint every 500 steps.
- W&B records train loss every 100 steps, validation loss every 500 steps, and
  test loss at every checkpoint.

## Local commands

```bash
uv run anvil-trainer \
  --config_path=configs/training/shirt_fold_pi05_hf_phase_aligned.yaml \
  --task-description="Fold the T-shirt properly" \
  --action-type=absolute \
  --split-ratio=8,1,1 \
  --note="HF folding initialization; phase-aligned Anvil OpenArm 2 demos"
```

```bash
uv run anvil-trainer \
  --config_path=configs/training/shirt_fold_pi05_base_phase_aligned.yaml \
  --task-description="Fold the T-shirt properly" \
  --action-type=absolute \
  --split-ratio=8,1,1 \
  --note="Pi0.5 base initialization; phase-aligned Anvil OpenArm 2 demos"
```

With seed 1000, the episode split is fixed:

- train: 0, 1, 3–5, 7–10, 13, 15–26, 28–32;
- validation: 2, 11, 14;
- test: 6, 12, 27.

## Vast launch through openarm2-cloud-runner

Upload the complete phase-aligned dataset—including `data/`, `meta/`, and
`videos/`—to a dedicated Hugging Face dataset repository and pin its commit
revision. Do not point Vast at the untrimmed upload or the HF reference dataset.

Use direct Anvil mode in two separate cloud-runner configs:

```yaml
secrets:
  wandb_api_key_env: WANDB_API_KEY

repos:
  anvil-embodied-ai:
    url: https://github.com/bensonlee5/anvil-embodied-ai.git
    default_ref: <COMMIT_CONTAINING_THE_RECIPES>

job:
  kind: train_policy
  policy_type: pi05
  task_description: Fold the T-shirt properly
  job_name: <UNIQUE_RUN_NAME>
  dataset_repo: <HF_USER>/openarm2-shirt-fold-phase-aligned-v1
  dataset_revision: <PINNED_DATASET_COMMIT>
  checkpoint_repo: <HF_USER>/<UNIQUE_CHECKPOINT_REPO>
  run_smoke_test: false
  hub_push_interval_seconds: 120
  extra_args:
    - --config_path=configs/training/<RECIPE>.yaml
    - --action-type=absolute
    - --split-ratio=8,1,1
    - --note=<RUN_DESCRIPTION>
```

Use an 80 GB or larger GPU and enough disk for ten multi-gigabyte checkpoints.
Launch each config separately:

```bash
./scripts/vast/launch --config configs/<RUN_CONFIG>.yaml
```

The checkpoint repositories and job names must be distinct. Keep the VM after
the first checkpoint is verified so a bootstrap, synchronization, or upload
problem can be inspected without losing its local logs.

## Quality gates

Treat a live process as necessary but insufficient. For both runs:

1. Confirm the resolved log prints the expected dataset, model revision, 16D
   feature schema, right-first action names, and relative-action processor.
2. Confirm loss and gradient norm are finite from the first logged step.
3. At step 500, require a numeric checkpoint, a W&B validation point, and a
   successful Hugging Face checkpoint push.
4. Compare the A/B runs at matched steps. Prefer validation/test loss over train
   loss; a lower train loss with a flat or worsening validation curve is not an
   improvement.
5. Investigate NaN/Inf immediately, repeated gradient spikes, loss increasing
   across several logging intervals, or a widening train/validation gap.
6. Before live control, run offline dataset replay and inspect per-joint error,
   especially shoulder joints 1–3, predicted motion magnitude, and temporal
   alignment.

The HF-initialized run is the first deployment candidate if its validation and
replay metrics are at least as good as the base-initialized run. Train the
embodiment adapter next only if the folding prior is useful but joint-space
replay still shows a systematic embodiment-dependent error.

## Three-stage quality-priority experiment

The next HF-prior experiment uses all 33 successful, phase-aligned target
episodes while changing only how training frames are selected. The reviewed
task structure is:

1. fold the first side;
2. fold the other side;
3. fold the shirt from bottom to top.

The full untrimmed videos were used for annotation so setup, reset, and the
complete manipulation context remained visible. Training still uses only the
34,850-frame phase-aligned trim. The immutable annotation and weighting artifact
is
`configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json`;
`scripts/training/build_openarm2_shirt_fold_priority_manifest.py --check`
detects drift.

Each stage has an independent 1–5 quality score. All labels describe successful
folds, so a score of 1 is a poor-quality success rather than task failure. The
artifact separately records 28 medium-confidence repeated-grasp attempts,
defined as a short close/reopen followed by another close from the same gripper
within 1.5 seconds. Recovery frames remain ordinary demonstrations. Smoothing
review windows are also separate: slow surface or edge refinement is deliberate
and receives exactly zero quality adjustment.

The sampler follows the data-selection mechanism in [Larchenko's LeHome 2026
write-up](https://arxiv.org/html/2606.27163), not its later on-policy learning
claim. It samples with replacement in proportion to an exponential priority;
the Pi0.5 flow-matching action loss is unchanged. Manual quality scores are
mapped to clipped log-priorities `[-1, -0.5, 0, 0.5, 1]`, repeated-attempt
approach windows receive `-0.5`, and the three stages receive equal aggregate
probability mass. These annotations are not learned advantages and this run
must not be called AWR. If auxiliary heads are added later, their losses require
inverse-sampling correction; the current action-only run does not.

Run the frozen experiment with:

```bash
uv run anvil-trainer \
  --config_path=configs/training/shirt_fold_pi05_hf_phase_aligned_priority_v1.yaml \
  --priority-sampling-manifest=configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json \
  --task-description="Fold the T-shirt properly" \
  --action-type=absolute \
  --split-ratio=8,1,1 \
  --note="HF prior; three-stage quality-priority sampler v1; all 33 trimmed successes"
```

Only training uses the priority sampler. Validation and test remain exhaustive,
unweighted episode splits. Every checkpoint copies the exact manifest and its
SHA-256 into `pretrained_model/`, allowing a resumed run to inherit it without
depending on a machine-local path.

## Native SARM and RA-BC experiment

The reward-aware experiment is a separate follow-up to the manual priority
sampler. It uses LeRobot's native SARM `dense_only` model to learn progress over
the same three stages, then uses native RA-BC to weight the ordinary Pi0.5
action loss. Do not combine manual priority sampling and RA-BC in the first
controlled run: doing so would confound two different sampling/weighting
mechanisms.

The immutable contract is
`configs/training/sarm_manifests/openarm2_shirt_fold_sarm_v1.json`. It preserves
the policy experiment's seed-1000 27/3/3 episode split and derives SARM temporal
proportions from the 27 training episodes only. The converter translates the
review manifest's exclusive stage ends to the inclusive ends consumed by
LeRobot SARM. It never changes the 34,850-frame phase-aligned source dataset.

Materialize and independently validate the derived dataset with:

```bash
uv run python scripts/training/materialize_openarm2_shirt_fold_sarm_dataset.py
uv run python scripts/training/materialize_openarm2_shirt_fold_sarm_dataset.py --check
```

Train the reward model with
`configs/training/shirt_fold_sarm_dense_v1.yaml`. SARM's bidirectional image
window is appropriate for this offline data-selection experiment, but the
reward model is not an online production component. After training, compute
every-frame dense progress without interpolation:

```bash
python -m lerobot.rewards.sarm.compute_rabc_weights \
  --dataset-repo-id=bohlt/openarm2-shirt-fold-phase-aligned-sarm-v1 \
  --reward-model-path=bohlt/openarm2-shirt-fold-sarm-v1 \
  --head-mode=dense \
  --stride=1 \
  --num-visualizations=6 \
  --output-path=datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1/sarm_progress.parquet

uv run python scripts/training/audit_openarm2_sarm_progress.py
uv run python scripts/training/resolve_openarm2_sarm_rabc_recipe.py
```

The audit requires one finite `[0, 1]` value for every frame, emits a separately
pinned train-only progress parquet so native RA-BC cannot compute normalization
statistics from policy holdouts, reports progress
agreement by split and stage, checks monotonicity, and separately reports the
reviewed quality, repeated-grasp, and coarse smoothing strata. Repeated grasps
remain evaluation signals rather than task-failure labels; smoothing windows
remain diagnostic because their current boundaries are review windows, not
precise action masks. The checked-in RA-BC recipe is deliberately fail-closed
with zero provenance hashes. Only the generated recipe may be launched after
the audit freezes the full and train-only progress SHA-256 values, audit SHA-256,
and train-only kappa.

## Deployment preflight

Use the shadow inference config before enabling commands. The shirt-fold live
and shadow configs deliberately apply no local absolute or delta action limiter;
the only requested enforcement is the downstream `anvil-loader` safety layer.
Stage the arms near a demonstrated trimmed start pose and verify radians,
right-first ordering, camera routing, and processor metadata before live motion.
