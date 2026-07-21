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
34,850-frame phase-aligned trim. The first holistic annotation artifact,
`configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json`, is
frozen history. The current candidate is the criterion-based blinded rescore in
`configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json`.
`scripts/training/build_openarm2_shirt_fold_priority_manifest_v2.py --check`
detects drift in the v2 review and generated manifest.

Each stage has four stage-specific 0–2 criteria whose total maps mechanically
to a 1–5 quality score. All labels describe successful folds, so a score of 1
is a poor-quality success rather than task failure. The rubric, untouched blind
responses, token mapping, temporal adjudication, and agreement audit are under
`configs/training/quality_annotations/openarm2_shirt_fold_3stage_v2/`.

The side-fold rescore has weak agreement with the old holistic labels, so v2
does not treat either set as ground truth or apply an aggressive weight ratio.
The 28 telemetry retry candidates are neutral until video review distinguishes
confirmed misgrasps from deliberate regrasping. Slow smoothing and edge
refinement remain deliberate, quality-neutral behaviors.

The sampler follows the data-selection mechanism in [Larchenko's LeHome 2026
write-up](https://arxiv.org/html/2606.27163), not its later on-policy learning
claim. It samples with replacement in proportion to an exponential priority;
the Pi0.5 flow-matching action loss is unchanged. Manual quality scores are
mapped to clipped log-priorities `[-0.4, -0.2, 0, 0.2, 0.4]`, for a maximum
2.23x sampling ratio. Retry candidates receive no adjustment, and aggregate
stage mass follows the seed-1000 training split's observed 6,174 / 14,262 /
8,798 frame distribution instead of forcing each stage to one third. These
annotations are not learned advantages and this run must not be called AWR. If
auxiliary heads are added later, their losses require inverse-sampling
correction; the current action-only run does not.

The priority-only v2 recipe remains available as a diagnostic, but it is not
the next run. Invoke it with:

```bash
uv run anvil-trainer \
  --config_path=configs/training/shirt_fold_pi05_hf_phase_aligned_priority_v2.yaml \
  --priority-sampling-manifest=configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json \
  --task-description="Fold the T-shirt properly" \
  --action-type=absolute \
  --split-ratio=8,1,1 \
  --note="HF prior; conservative blinded-rubric priority sampler v2; all 33 trimmed successes"
```

Only training uses the priority sampler. Validation and test remain exhaustive,
unweighted episode splits. Every checkpoint copies the exact manifest and its
SHA-256 into `pretrained_model/`, allowing a resumed run to inherit it without
depending on a machine-local path.

The next experiment is not another plain-BC control. Existing plain fine-tunes
are already too far from the acceptable behavior threshold to justify spending
another run on that objective. The production-candidate screen combines this
conservative sampler with audited dense SARM RA-BC, as specified below. This
choice deliberately answers whether the intended integrated training system is
promising; it does not isolate the causal contribution of either component.

This screen does not require new demonstrations. Compare its exhaustive
validation/test loss, per-actuator loss, stage-stratified holdout loss, and
robot folding quality with the already completed runs as historical context,
not as a new randomized control. More steps are justified only if both train
and holdout curves are still improving at step 5,000. Collect new episodes only
after evaluation identifies a specific underrepresented garment state or
failure mode that the 33 successful demonstrations cannot cover.

## Integrated quality sampling and dense SARM RA-BC

Here, **SARM v2 means experiment/data-contract revision 2 using the released
single-task SARM architecture**. It does not mean the distinct multi-task
`SARM2` architecture. The reward checkpoint is reused because the v2 blind
rescore changed quality sampling only; it changed no SARM target, stage
boundary, frame, or episode split.

The production-candidate screen uses LeRobot's native SARM `dense_only` model
to learn progress over the same three stages, then combines two deliberately
distinct mechanisms:

- blinded stage quality changes only the probability that a training frame is
  sampled; and
- the SARM progress change over the policy's next 30 frames changes only that
  sample's native RA-BC action-loss weight.

Validation and test remain exhaustive and unweighted. Slow deliberate
smoothing and unreviewed retry candidates remain neutral under both mechanisms.
This is an integrated product experiment, not a component ablation, and must
not be described as proving that either quality sampling or SARM caused any
observed change.

The original immutable contract is
`configs/training/sarm_manifests/openarm2_shirt_fold_sarm_v1.json`. The
criterion-based quality rescore did not change any stage boundary, dense target,
frame, or split assignment, so the completed SARM reward checkpoint remains
valid. `configs/training/sarm_manifests/openarm2_shirt_fold_sarm_v2.json` binds
those unchanged targets to the v2 priority-manifest SHA rather than pretending
the quality labels used to train SARM. Both contracts preserve the policy
experiment's seed-1000 27/3/3 split and derive temporal proportions from the 27
training episodes only.

Materialize and independently validate the derived dataset with:

```bash
uv run python scripts/training/materialize_openarm2_shirt_fold_sarm_dataset.py
uv run python scripts/training/materialize_openarm2_shirt_fold_sarm_dataset.py --check
```

The completed reward run used
`configs/training/shirt_fold_sarm_dense_v1.yaml`: checkpoint 1,200 from
`train_reward_shirt_20260720_sarm_dense_v3`, W&B run `kttuwuef`, and Hub
revision `108048371c101e77299b8b60ae5f214d30b295f2`. Its final train/eval
losses were finite (0.0080 / 0.1053), and its wrapper, final sync, and Hub push
all exited zero. SARM's bidirectional image window is appropriate for this
offline training signal, but it is not a causal online production reward. The
deployed policy does not require the SARM model.

The completed run computed every-frame dense progress without interpolation:

```bash
python -m lerobot.rewards.sarm.compute_rabc_weights \
  --dataset-repo-id=bohlt/openarm2-shirt-fold-phase-aligned-sarm-v1 \
  --reward-model-path=bohlt/openarm2-shirt-fold-sarm-v1 \
  --head-mode=dense \
  --stride=1 \
  --num-visualizations=6 \
  --output-path=datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1/sarm_progress.parquet

# V2 changed no dense target; retain the source bytes under a versioned name.
cp datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1/sarm_progress.parquet \
  datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1/sarm_progress_v2.parquet

uv run python scripts/training/audit_openarm2_sarm_progress.py \
  --priority-manifest=configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json \
  --contract=configs/training/sarm_manifests/openarm2_shirt_fold_sarm_v2.json \
  --progress=datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1/sarm_progress_v2.parquet \
  --training-progress=datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1/sarm_progress_train_v2.parquet \
  --output-json=datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1/sarm_progress_audit_v2.json

PYTHONPATH=packages/anvil_trainer/src uv run python \
  scripts/training/audit_openarm2_quality_sarm_integration.py --check
```

The audit requires one finite `[0, 1]` value for every frame, emits a separately
pinned train-only progress parquet so native RA-BC cannot compute normalization
statistics from policy holdouts, reports progress
agreement by split and stage, checks monotonicity, and separately reports the
reviewed quality and coarse smoothing strata. Repeated grasps remain evaluation
signals rather than task-failure labels; smoothing windows remain diagnostic
because their current boundaries are review windows, not precise action masks.

The v2 audit freezes kappa at `0.05090876221656798`. The integration audit at
`configs/training/quality_sarm_audits/openarm2_shirt_fold_quality_sarm_v2.json`
mirrors native RA-BC and evaluates its interaction with the sampler. Manual
sampling retains 95.8% effective sample size; the expected combined objective
retains 79.1%; 97.65% of training frames have nonzero RA-BC weight; and combined
stage mass remains 22.2% / 49.5% / 28.4%. Quality score and RA-BC weight have
only -0.048 correlation, evidence that the two mechanisms are not simply
duplicating the same ranking.

Launch the single 5,000-step candidate with:

```bash
uv run anvil-trainer \
  --config_path=configs/training/shirt_fold_pi05_hf_phase_aligned_quality_sarm_rabc_v2.yaml \
  --priority-sampling-manifest=configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json \
  --task-description="Fold the T-shirt properly" \
  --action-type=absolute \
  --split-ratio=8,1,1 \
  --note="HF prior; blind-quality sampling plus audited dense SARM RA-BC v2; 33 trimmed successes"
```

Before remote launch, copy or upload the exact v2 train-only progress parquet
and audit JSON named by the recipe and verify their SHA-256 values. The trainer
fails closed if the progress, audit, kappa, priority manifest, SARM contract, or
30-frame policy chunk differs. Do not substitute a freshly regenerated artifact
without updating the versioned contract and integration audit.

## Released reward-model comparison: SARM v2 and Robometer

The next reward comparison has exactly two arms, both backed by public released
implementations:

- single-task SARM with the v2 quality/sampling contract described above; and
- Robometer at official Git revision
  `5b815254bf31ee1bea3753c3a2da9f9033736d9a`, initialized from
  `robometer/Robometer-4B@beef63bc914c5c189329d49c6d712d96d632aa34`.

RARM is excluded because its authors have not released the implementation. No
local approximation, similarly named manifest, or custom architecture may be
presented as a RARM result.

The Robometer data contract is
`configs/training/robometer_manifests/openarm2_shirt_fold_robometer_v1.json`.
It uses only the same trimmed base-camera stream and keeps the seed-1000
27/3/3 episode split. Every split contains one full trajectory and three stage
clips per original episode: 108 train, 12 validation, and 12 test trajectories.
No clip can cross its source episode or split.

All 33 demonstrations remain labeled successful. Full trajectories use
`partial_success=1.0` for temporal progress. Each exact stage clip uses the
blind stage-quality score divided by five as its partial-progress target and
same-task preference rank. Released Robometer's
`predict_last_frame_partial_progress=true` mask applies that target only to the
clip's final frame, so intermediate stage frames are not incorrectly assigned
zero progress. Smoothing remains inside the clips and is quality-neutral.

Validate the conversion contract locally with:

```bash
python scripts/training/materialize_openarm2_robometer_dataset.py \
  --dataset-root=datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1 \
  --output-root=/tmp/openarm2-robometer-v1 \
  --check
```

The remote materialization must render and frame-count every exact clip, push
three private Hub configs (`openarm2_train`, `openarm2_validation`, and
`openarm2_test`), and record every clip SHA-256 in
`robometer_dataset_audit_v1.json`. Robometer then runs the released 4B LoRA
recipe for 1,000 reward-model steps with progress and preference heads enabled,
the success head frozen, and holdout evaluation at 100-step intervals. A 5,000
step policy run is allowed only after its full-frame progress artifact passes
the same episode-isolation, finiteness, monotonicity, quality-stratification,
and train-only RA-BC audits as the SARM arm.

## Deployment preflight

Use the shadow inference config before enabling commands. The shirt-fold live
and shadow configs deliberately apply no local absolute or delta action limiter;
the only requested enforcement is the downstream `anvil-loader` safety layer.
Stage the arms near a demonstrated trimmed start pose and verify radians,
right-first ordering, camera routing, and processor metadata before live motion.
