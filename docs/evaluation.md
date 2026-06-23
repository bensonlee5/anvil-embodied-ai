[← Back to README](../README.md)

# Offline Evaluation

Validate model performance before deploying to a robot. Two complementary modes:

| Mode | Command | What it tests |
|------|---------|---------------|
| **Dataset replay** | `anvil-eval` | Feeds dataset observations into the model — fast, no ROS2 needed |
| **ROS2 MCAP replay** | `anvil-eval-ros` | Replays raw MCAP through the full Docker inference stack — mirrors real deployment |

> **Both modes are open-loop.** The model's predicted actions are never executed — recorded observations are fed in regardless of what the model outputs. This means evaluation scores (MAE, cosine similarity, etc.) measure how closely the model tracks demonstrated trajectories, not whether it can successfully complete a task on a real robot. A low MAE is a necessary signal for good training, but not sufficient evidence of real-world performance. True closed-loop evaluation requires running the model on the actual robot.

Results are written to:
```
eval_results/{dataset}/{job}/{checkpoint}/
├── raw/     ← anvil-eval output
└── ros/     ← anvil-eval-ros output
```

## Dataset Replay (`anvil-eval`)

```bash
uv run anvil-eval \
  --checkpoint model_zoo/my-task/checkpoints/last \
  --dataset data/datasets/my-task \
  --num-eps 5 \
  --device cuda
```

Produces per-joint trajectory plots and a summary box plot.

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | required | Checkpoint directory |
| `--dataset PATH` | required | LeRobot dataset directory |
| `--num-eps N` | `3` | Episodes to sample from the selected split |
| `--split SPLIT` | `all` | Which split to evaluate: `train` · `val` · `test` · `all` |
| `--episodes "0,3,5"` | — | Manually specify episode indices (overrides `--split`) |
| `--output-dir PATH` | auto | Output dir (default: `eval_results/{dataset}/{job}/{checkpoint}/raw`) |
| `--device DEVICE` | `cuda` | Inference device: `cuda` or `cpu` |
| `--task-description TEXT` | — | VLA task prompt — overrides `anvil_config.json` (SmolVLA / Pi0.5 only) |
| `--seed N` | `42` | Random seed for episode sampling |
| `--eval-type LIST` | `trajectory` | Comma-separated analysis modes: `trajectory` · `horizon` |
| `--phases SOURCE` | `gripper` | Gripper-phase overlay (lines on trajectory plots + per-arm MAE timeline): `gripper` · `none` |

Use `--split all` to sample from across the full dataset:

```bash
uv run anvil-eval \
  --checkpoint model_zoo/my-task/checkpoints/last \
  --dataset data/datasets/my-task \
  --split all \
  --num-eps 5 \
  --device cuda
```

### Key Metrics

| Metric | What it measures |
|---|---|
| **MAE** | Mean Absolute Error — average per-joint position error across all frames |
| **MSE** | Mean Squared Error — penalizes large errors more heavily than MAE |
| **Cosine Similarity** | Direction alignment between predicted and ground-truth action vectors |
| **Smoothness** | L2 norm of consecutive action deltas — high variance indicates jittery output |

### Analyzing Plots

**Episode plots** (`plots/episode_NNNN_<split>.png`) — one subplot per joint showing predicted (red) vs ground-truth (blue) trajectories for the full episode. Gripper joints (`finger_joint1`) are moved to the end of the grid for easier comparison.

**Summary box plot** (`plots/summary_per_joint_mae.png`) — distribution of MAE across all evaluated episodes for each joint. Use this to identify which joints the model is most/least accurate on.

### Evaluation Strategies

- `--split all --num-eps 5` gives a representative cross-section from train/val/test rather than a single split.
- **High error on `train` split** → underfitting — model needs more training or more capacity.
- **Low `train` error but high `val`/`test` error** → overfitting — add augmentation, reduce steps, or collect more diverse data.

### Diagnostic Modes (`--eval-type`, `--phases`)

Break model behavior down along two independent axes. All modes write their full underlying data to `substrate.csv` + `metrics_summary.json`, so any plot can be regenerated with your own tools.

- **`trajectory`** (default) — per-joint predicted-vs-ground-truth trajectory across the episode (the plots above).
- **`horizon`** — per-joint error vs. how many steps ahead the model predicts (chunk offset). Shows how far a single prediction stays trustworthy before re-planning: a flat curve means you can safely raise `n_action_steps`; a steep one points to retraining. (Adds an extra inference per chunk, so only runs when requested.)
- **`--phases gripper`** (default on) — segments each episode into per-arm task phases at gripper open↔close transitions (from ground truth). Adds **grasp (green) / release (red) dashed lines** to the trajectory plots, and emits a **per-arm MAE-over-time** plot (`episode_*_phase_mae.png`): `frame vs mean|error|` across each arm's joints (fingers excluded), so you can see which task phases drive error. Use `--phases none` to disable.

```bash
uv run anvil-eval \
  --checkpoint model_zoo/my-task/checkpoints/last \
  --dataset data/datasets/my-task \
  --eval-type trajectory,horizon \
  --phases gripper
```

## ROS2 MCAP Replay (`anvil-eval-ros`)

Replays raw MCAP recordings through the same inference node that runs on the real robot. Catches integration issues (topic remapping, timing, action chunking) that dataset replay cannot.

```bash
uv run anvil-eval-ros \
  --checkpoint model_zoo/my-task/checkpoints/last \
  --mcap-root data/raw/my-task \
  --num-eps 3
```

**How it works:**
```
Host: anvil-eval-ros
  │  generates eval_plan.json → launches docker compose
  │
  ├─ [inference]      model on GPU, publishes to /eval/* topics
  ├─ [mcap-player]    replays one MCAP per episode
  └─ [eval-recorder]  records GT + predicted actions → metrics + plots
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | required | Checkpoint directory |
| `--mcap-root PATH` | required | Raw MCAP directory (e.g. `data/raw/my-task`) |
| `--num-eps N` | `3` | Episodes to sample per split |
| `--split SPLIT` | `all` | Which split: `train` · `val` · `test` · `all` |
| `--episodes "0,3,5"` | — | Manually specify episode indices (overrides `--split`) |
| `--seed N` | `42` | Random seed for episode sampling |
| `--base-inference-config PATH` | — | Override default `configs/lerobot_control/inference_eval.yaml` |
| `--dataset-dir PATH` | — | Explicit path to converted LeRobot dataset — used to locate `conversion_config.yaml` when raw and dataset dirs are not co-located |
| `--output-dir PATH` | auto | Output dir (default: `eval_results/{dataset}/{job}/{checkpoint}/ros`) |
| `--monitor` | off | Record per-step CSV + PNG report via inference monitor |
| `--image-tag TAG` | `latest` | Docker image tag |
| `--no-docker` | off | Dry-run: print `eval_plan.json` path and compose command without running |
| `--warmup-sec N` | `5.0` | Seconds to wait for inference warmup before first episode |
| `--inference-drain-sec N` | `1.5` | Seconds to wait after MCAP ends for inference pipeline to drain |
| `--inter-episode-sec N` | `1.0` | Sleep between episodes |
| `--silence-timeout-sec N` | `1.0` | Seconds of GT topic silence before declaring episode done |
| `--ack-timeout-sec N` | `20.0` | Max seconds mcap-player waits for episode acknowledgement |

**Inference Monitor (`--monitor`)** — records `/monitor/*` topics and writes:
```
ros/
├── monitor/
│   ├── inference_data.csv       ← per-step obs_state / raw_output / control_cmd
│   └── inference_report.png    ← joint-level overlay plot
└── plots/
    └── episode_NNNN_*.png      ← GT (blue) / Pred (red) / Raw output (orange)
```

The orange "Raw" line shows model output **before** postprocessing — useful for diagnosing whether the policy or postprocessor is responsible for a tracking error.

> Requires Docker with NVIDIA GPU support. Set `LEROBOT_EXTRAS` if your model needs extra dependencies (e.g. `pi`, `smolvla`).

> For running inference on a live robot, see [Run Inference](inference.md).

---

## Design Notes (`--eval-type` / `--phases` internals)

For reviewers and contributors working on the diagnostic modes.

**Horizon capture.** The model predicts a full trajectory of length `horizon` per inference but
normally executes only `n_action_steps` of it; that full trajectory is identical regardless of
`n_action_steps`. For `horizon` mode the evaluator captures the whole chunk at each inference anchor
via `predict_action_chunk` (for diffusion, temporarily raising `n_action_steps` to
`horizon - n_obs_steps + 1`, the model's cap, since `generate_actions` slices to `n_action_steps`).
The executed trajectory (`trajectory` mode) is left on the untouched `select_action` path. This extra
capture is **gated on `--eval-type horizon`** so trajectory/phase-only runs pay no extra inference.

**Substrate.** Horizon is a pure function of one long-form table (`substrate.csv`), keyed by
`(episode, anchor_frame, horizon_offset, joint)` with `predicted`, `ground_truth`, `error`,
`obs_state`, and `phase_left`/`phase_right` columns. Written only for `horizon` runs.

**Phase labeler.** Per arm, binarize the **ground-truth** `*_finger_joint1` at the midpoint of its
range, debounce with hysteresis + a minimum-segment length, and cut at **every** transition (both
open→close and close→open). Segments get generic labels (`left:closed#1`, …) — no hardcoded task
semantics. GT-sourced so boundaries are identical across checkpoints. Phases drive two cheap outputs
(no horizon capture needed): grasp/release **lines** on the trajectory plots, and a **per-arm
MAE-over-time** plot scored on each arm's own joints with fingers excluded (MAE takes `|·|` per joint
before averaging, so positive/negative per-joint errors don't cancel).

**Scope.** Targets queue-based chunked models (diffusion, ACT without temporal ensembling). ACT with
`temporal_ensemble_coeff` re-infers every step and does not follow chunk-slicing semantics — detected
and restricted to `trajectory` for now. Closed-loop `rollout` (feeding predictions back) is reserved
as a future `--eval-type` value; see the open-loop caveat at the top.

**Touch points.** New: `substrate.py`, `horizon.py`, `phases.py`. Changed: `evaluator.py` (gated
full-chunk capture), `cli.py` (flags + dispatch), `plotting.py` (`plot_horizon_curve`,
`plot_phase_mae_timeline`, phase lines in `plot_episode_joints`).

[← Back to README](../README.md)
