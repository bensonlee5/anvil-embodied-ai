<p align="center">
  <a href="https://anvil.bot/">
    <img src="material/anvil.png" alt="Anvil" width="120" />
  </a>
</p>

<h1 align="center">Anvil-Embodied-AI</h1>

<p align="center">
  <a href="https://anvil.bot/"><img src="https://img.shields.io/badge/Website-anvil.bot-blue?style=for-the-badge" alt="Website" /></a>
  <a href="https://docs.anvil.bot/"><img src="https://img.shields.io/badge/Documentation-docs.anvil.bot-green?style=for-the-badge" alt="Docs" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-orange?style=for-the-badge" alt="License" /></a>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12+-yellow?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://docs.ros.org/en/jazzy/"><img src="https://img.shields.io/badge/ROS2-Jazzy-22314E?style=flat-square&logo=ros&logoColor=white" alt="ROS2" /></a>
  <a href="https://github.com/huggingface/lerobot"><img src="https://img.shields.io/badge/LeRobot-v0.5.1-ff69b4?style=flat-square&logo=huggingface&logoColor=white" alt="LeRobot" /></a>
</p>

---

## 📢 News

- **2026-05-08** — Upgraded to LeRobot v0.5.1.

_See full history in [CHANGELOG.md](CHANGELOG.md)._

---

## Overview

This repository is the embodied AI stack for the Anvil platform — data conversion, model training, and real-time inference for robot manipulation policies.

```
  Anvil Devbox (Data collection)          This repo (anvil-embodied-ai)
┌──────────────────────────────┐    ┌──────────────────────────────────────────────────────────┐
│  Teleoperation + Recording   │───>│  Convert      ───>  Train         ───>  Run Inference    │
│  MCAP files                  │    │  mcap-convert       anvil-trainer       ROS2 CycloneDDS  │
└──────────────────────────────┘    └──────────────────────────────────────────────────────────┘
```

| Stage | Description |
|-------|-------------|
| **0. Data Collection** | Record teleoperation demos as MCAP files via [Anvil Devbox](https://shop.anvil.bot/products/anvil-devbox) |
| **1. Data Conversion** | Convert MCAP recordings to LeRobot v3.0 datasets → [docs/data-conversion.md](docs/data-conversion.md) |
| **2. Model Training** | Train ACT, Diffusion, SmolVLA, Pi0, or Pi0.5 policies → [docs/training.md](docs/training.md) |
| **3. Offline Evaluation** | Validate model performance against ground-truth before deploying → [docs/evaluation.md](docs/evaluation.md) |
| **4. Run Inference** | Deploy trained models on a GPU PC via ROS2 CycloneDDS → [docs/inference.md](docs/inference.md) |

> **Don't have data yet?** The [Anvil OpenARM Quest Teleop Kit](https://shop.anvil.bot/products/openarm-quest-teleop-kit) gives you everything you need to start collecting demonstrations out of the box. See the [data collection guide](https://docs.anvil.bot/software/collecting-data).

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker (for inference and ROS2 eval)

```bash
git clone https://github.com/anvil-robotics/anvil-embodied-ai.git
cd anvil-embodied-ai
uv sync --all-packages
```

ACT and Diffusion are included in the base install. For other policies:

| Extra | Policy |
|-------|--------|
| `smolvla` | SmolVLA |
| `pi` | Pi0 / Pi0.5 |

```bash
uv sync --all-packages --extra smolvla
uv sync --all-packages --extra smolvla --extra pi   # multiple
uv sync --all-packages --extra all                  # all policies
```

> **GPU / CUDA note:** The root `pyproject.toml` pins torch to the `cu128` index. If your machine uses a different CUDA driver, change `pytorch-cu128` → `pytorch-cu126` (or `cu124`) in `pyproject.toml` before syncing.

---

## Usage

### 0. Data Collection

Record teleoperation demonstrations as ROS2 MCAP files through an [Anvil Devbox](https://shop.anvil.bot/products/anvil-devbox). See the [data collection guide](https://docs.anvil.bot/software/collecting-data) for details.

### 1. Data Conversion ([doc](docs/data-conversion.md))

Convert MCAP recordings into LeRobot v3.0 datasets. Pick the config that matches your recording setup and run `mcap-convert`.

### 2. Model Training ([doc](docs/training.md))

Train ACT, Diffusion, SmolVLA, Pi0, or Pi0.5 policies. Checkpoints saved to `model_zoo/<space>-space/<dataset>/<job_name>/` (`ee-space/` or `joint-space/` based on `--action-type`).

| Config | Data space | Teleop mode | Arms | `observation.state` | `action` |
|--------|-----------|-------------|------|---------------------|---------|
| `openarm_ee_bimanual.yaml` | **EE Cartesian** | Quest VR | Bimanual | `(16,)` xyz+quat+gripper × 2 | `(20,)` xyz+rot6d+gripper × 2 |
| `openarm_ee_left.yaml` | **EE Cartesian** | Quest VR | Left only | `(8,)` xyz+quat+gripper | `(10,)` xyz+rot6d+gripper |
| `openarm_joint_bimanual.yaml` | Joint | Quest VR | Bimanual | `(16,)` joint positions | `(16,)` from command topics |
| `openarm_bimanual.yaml` | Joint | Leader-follower | Bimanual | `(16,)` joint positions | `(16,)` leader joints |
| `openarm_bimanual_quest.yaml` | Joint | Quest VR | Bimanual | `(16,)` joint positions | `(16,)` from command topics |
| `openarm_single_quest.yaml` | Joint | Quest VR | Single (right) | `(8,)` joint positions | `(8,)` from command topic |
| `openarm_single_quest_afo.yaml` | Joint | Quest VR | Single (right) | `(8,)` joint positions | `(8,)` = obs[t] (future window at train time) |

```bash
# EE Cartesian bimanual (recommended for EE-space diffusion policy)
# Output: data/datasets/ee-space/my-sessions/
uv run mcap-convert \
  --input-dir data/raw/my-sessions \
  --config configs/mcap_converter/openarm_ee_bimanual.yaml

# Joint space
# Output: data/datasets/joint-space/my-sessions/
uv run mcap-convert \
  --input-dir data/raw/my-sessions \
  --config configs/mcap_converter/openarm_joint_bimanual.yaml
```

### 3. Offline Evaluation ([doc](docs/evaluation.md))

Validate model performance before deploying. Two modes: dataset replay (`anvil-eval`) and ROS2 MCAP replay (`anvil-eval-ros`).

### 4. Run Inference ([doc](docs/inference.md))

### EE Cartesian mode

EE configs read from `/ee_pose_left` / `/ee_pose_right` (`anvil_msgs/msg/CommandedEEPose`):

```
observation.state per arm (8 dims): [x, y, z, qx, qy, qz, qw, gripper]
action         per arm (10 dims): [x, y, z, r0, r1, r2, r3, r4, r5, gripper]
```

The action uses 6D rotation representation (Zhou et al. 2019) for regression stability.
`action[t] = observation.state[t]` in the converter — the future prediction window
(`action[t] = ee_pose[t+k]`) is handled by LeRobot's `delta_timestamps` at train time.

### `--act-from-obs` flag

Force `action[t] = observation.state[t]` even when `action_topics` are configured in joint mode:

```bash
uv run mcap-convert \
  --input-dir data/raw/my-sessions \
  --config configs/mcap_converter/openarm_joint_bimanual.yaml \
  --act-from-obs
```

The future window is applied by `delta_timestamps` at train time. EE mode is always act-from-obs.

**Common flags:**

| Flag | Description |
|------|-------------|
| `--resume` | Skip already-converted episodes — safe to re-run after interruption |
| `--max-episodes N` | Convert only the first N episodes |
| `--fps N` | Override output FPS (auto-detected by default) |
| `--vcodec` | `h264` (default) · `hevc` · `libsvtav1` |
| `--robot-type` | `anvil_openarm` (default) · `anvil_yam` |
| `--act-from-obs` | Force `action[t] = obs[t]` regardless of configured action topics |

Then validate:

```bash
uv run dataset-validate --root data/datasets/my-sessions
```

Expected: 5 checks all showing `[OK]`.

---

## 2. Model Training

### Supported Policies

| Policy | `--policy.type` | Notes |
|--------|----------------|-------|
| ACT | `act` | Action Chunking Transformer — fast, reliable |
| Diffusion | `diffusion` | Diffusion Policy — smooth, handles multimodal distributions |
| SmolVLA | `smolvla` | Language-conditioned VLA; requires task description |
| Pi0 | `pi0` | Flow-matching VLA; PaliGemma-3B backbone |
| Pi0.5 | `pi05` | Larger Pi0 variant (~4B params); higher VRAM |

Checkpoints are saved to `model_zoo/<space>-space/<dataset>/<job_name>/` (`ee-space/` for EE action types, `joint-space/` for joint). Run `anvil-trainer --help` for the full flag reference.

---

### Common Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset.root=PATH` | — | Path to converted LeRobot dataset |
| `--policy.type=TYPE` | — | Policy type (see table above) |
| `--job_name=NAME` | `<policy>_<timestamp>` | Checkpoint directory name |
| `--steps=N` | `100000` | Total training steps |
| `--batch_size=N` | `8` | Reduce if GPU OOM |
| `--save_freq=N` | `10000` | Checkpoint save interval |
| `--split-ratio=T,V,S` | `8,1,1` | Train/val/test episode split. Two values = no test set. Val loss logged every `log_freq×5` steps; test loss at every checkpoint |
| `--max-episodes=N` | — | Subsample N episodes before splitting (reproducible with training seed) |
| `--exclude-observation=K1,K2` | — | Drop observations by suffix after `observation.` — e.g. `images.chest`, `velocity`, `effort` |
| `--backbone=NAME` | `resnet18` | Vision backbone for ACT/Diffusion: `resnet18` · `resnet34` · `resnet50` |
| `--resume=PATH` | — | Resume from job root or specific checkpoint (e.g. `model_zoo/my-task/checkpoints/020000`) |

---

### Action Types

| `--action-type` | Formula | When to use |
|-----------------|---------|-------------|
| `absolute` (default) | Raw joint positions | Simplest; works well for ACT and Diffusion |
| `delta_obs_t` | `Δ[k] = action[k] − obs_state[t]` | All steps share the same obs reference |
| `delta_sequential` | `Δ[0] = action[0] − obs_state[t]`; `Δ[k] = action[k] − action[k−1]` | Encodes velocity; smoother trajectories since consecutive deltas are small |

Action type is persisted to `anvil_config.json` — inference applies the inverse automatically.

Additional delta flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--delta-exclude-joints=J1,J2` | — | Keep these joints absolute (e.g. `finger_joint1` for grippers) |
| `--delta-stats-n-steps=N` | `1` | Look-ahead steps for delta normalizer stats. Increase to cover multi-step displacement range |

---

### Normalization Mapping

`--policy.normalization_mapping='{"ACTION":"...","STATE":"...","VISUAL":"..."}'`

| Value | Description |
|-------|-------------|
| `MEAN_STD` | Normalize by μ/σ |
| `MIN_MAX` | Normalize to [−1, 1] by observed min/max |
| `IDENTITY` | Passthrough — always use for `VISUAL` |

**Guidance by policy:**
- **Diffusion** → `ACTION: MIN_MAX`. Diffusion clips denoised actions to ±1 at every step (`clip_sample=True`); `MEAN_STD` silently truncates extreme actions.
- **ACT / SmolVLA / Pi0 / Pi0.5** → `ACTION: MEAN_STD`

---

### Weights & Biases

```bash
uv run wandb login   # one-time setup
```

| Flag | Description |
|------|-------------|
| `--wandb.enable=true` | Enable W&B logging |
| `--wandb.project=NAME` | Project name (auto-set to dataset folder name) |

Key metrics to watch: `train/loss` (should decrease steadily), `train/grad_norm` (spikes → lower LR), `eval/val_loss`, `eval/test_loss`.

---

### ACT

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --wandb.enable=false
```

**Tips:**
- Match `chunk_size` and `n_action_steps` to task speed: 50 for precise tasks, 100 for sweeping motions
- Enable temporal ensemble at inference for smoother execution — no retraining needed
- 100k steps / batch 16 is a solid default; 50k for small datasets

---

### Data Augmentation

Two built-in augmentation layers, both disabled by default. Can be combined with any policy.

#### Layer 1 — Color Augmentation (all policies)

Randomly applies up to `max_num_transforms` color transforms per image at training time. Pre-configured with conservative strengths:

| Transform | Range |
|-----------|-------|
| Brightness | [0.8, 1.2] |
| Contrast | [0.8, 1.2] |
| Saturation | [0.5, 1.5] |
| Hue | [−0.05, 0.05] |
| Sharpness | [0.5, 1.5] |
| Affine | ±5° rotation, 5% translation |

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3
```

#### Layer 2 — Random Crop (Diffusion only)

Diffusion's `DiffusionRgbEncoder` applies `RandomCrop` during training and `CenterCrop` during inference — the switch is automatic, no inference-time config needed.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=diffusion \
  --policy.crop_is_random=true \
  --policy.crop_ratio=0.9
```

`crop_ratio=0.9` crops to 90% of the original image size. Combine both layers for best generalization:

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=diffusion \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.crop_is_random=true \
  --policy.crop_ratio=0.9
```

---

### Diffusion

Good for tasks with multimodal action distributions (multiple valid ways to complete the task). Produces smooth motions; inference is slower than ACT due to the denoising loop.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=diffusion \
  --policy.normalization_mapping='{"ACTION":"MIN_MAX","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.horizon=24 \
  --policy.down_dims='[256,512,1024]' \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.use_group_norm=false \
  --wandb.enable=false
```

**Hyperparameters — for datasets under ~500 episodes:**

| Flag | Default | Recommended | Why |
|------|---------|-------------|-----|
| `--policy.horizon` | `16` | `24` | Longer horizon gives UNet more temporal context; must satisfy `n_obs_steps(2) + n_action_steps + drop_frames` |
| `--policy.down_dims` | `[512,1024,2048]` | `[256,512,1024]` | Smaller UNet reduces overfitting on small datasets |
| `--policy.use_group_norm` | `true` | `false` | Required when using pretrained ImageNet backbone (preserves BatchNorm) |

> **Inference-only flags** — set these in `inference_tuning.diffusion` in the YAML config, not at training time:
> - `n_action_steps: 16` — steps to execute per chunk before re-planning (default from checkpoint: 8)
> - `num_inference_steps: 10` — denoising iterations; reduces from 100 steps (~300ms) to 10 steps (~30ms) without retraining

---

### SmolVLA

Language-conditioned — always pass `--task-description` and `--policy.pretrained_path`.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --task-description="Grab the gray doll and put it in the bucket" \
  --wandb.enable=false
```

**Tips:** 30k–50k steps is usually enough from a pretrained base. The task description is saved to `anvil_config.json` in the checkpoint and auto-loaded at inference — no manual copy needed.

---

### Pi0 / Pi0.5

Flow-matching VLA policies from [Physical Intelligence](https://github.com/Physical-Intelligence/openpi). Both require HuggingFace access to [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) — request access on the model page, then run `huggingface-hub login` once.

**Pi0:**

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=pi0 \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --task-description="Grab the gray doll and put it in the bucket" \
  --wandb.enable=false
```

**Pi0.5** — same as Pi0 but ~4B params. Add `--num_workers=0` (prevents CPU RAM OOM from forked workers) and `--batch_size=16`:

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --batch_size=16 \
  --num_workers=0 \
  --task-description="Grab the gray doll and put it in the bucket" \
  --wandb.enable=false
```

> Pi0.5 requires quantile stats (`q01`/`q99`) which `mcap-convert` does not produce. Use `MEAN_STD` for `ACTION` (recommended), or see [Pi0.5 normalization](docs/training-tips.md#normalization-mapping) for how to compute them.

**Key flags for Pi series:**

| Flag | Recommendation |
|------|----------------|
| `--policy.train_expert_only=true` | Freeze backbone, train only action expert — lower memory, faster convergence |
| `--policy.compile_model=true` | `torch.compile` — ~10–20% throughput gain |
| `--policy.gradient_checkpointing=true` | Reduces VRAM — always enable |
| `--policy.dtype=bfloat16` | Halves VRAM — required for Pi0.5 on 24 GB GPU |

---

### Fine-tune from a Checkpoint

Start a new run from a previously trained checkpoint (step counter resets, new output directory):

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.path=model_zoo/my-task/checkpoints/last/pretrained_model
```

`--policy.type` is not needed — it is read from the checkpoint automatically.

> **`--policy.path` vs `--resume`:** `--policy.path` starts fresh from a checkpoint's weights (new output dir, step 0). `--resume` continues a stopped run in-place (same output dir, step counter carries over).

---

### Resume a Run

```bash
# Resume from the latest checkpoint
uv run anvil-trainer --resume=model_zoo/pick-and-place

# Resume from a specific step
uv run anvil-trainer --resume=model_zoo/pick-and-place/checkpoints/020000
```

Only pass `--resume` — all other settings are restored from the checkpoint's `train_config.json`. Action type settings are inherited from `anvil_config.json` automatically.

---

### Checkpoint Output Structure

```
model_zoo/
├── ee-space/               # EE action types (ee_abs, ee_rel)
│   └── <dataset>/
│       └── <job_name>/
│           ├── checkpoints/
│           │   ├── last -> 100000/          # symlink to latest checkpoint
│           │   ├── 010000/
│           │   │   └── pretrained_model/
│           │   │       ├── config.json              # LeRobot policy config
│           │   │       ├── anvil_config.json        # action_type, note, task_description
│           │   │       ├── split_info.json          # train/val/test episode lists
│           │   │       ├── policy_preprocessor.json # normalizer + resize config
│           │   │       └── policy_postprocessor.json
│           │   └── 100000/
│           └── wandb/
└── joint-space/            # Joint action type (joint_abs)
    └── <dataset>/
        └── <job_name>/
            └── checkpoints/
                └── ...
```

---

## 3. Offline Evaluation

Validate model performance before deploying to a robot. Two complementary modes:

| Mode | Command | What it tests |
|------|---------|---------------|
| **Dataset replay** | `anvil-eval` | Feeds dataset observations into the model — fast, no ROS2 needed |
| **ROS2 MCAP replay** | `anvil-eval-ros` | Replays raw MCAP through the full Docker inference stack — mirrors real deployment |

Results are written to:
```
eval_results/{dataset}/{job}/{checkpoint}/
├── raw/     ← anvil-eval output
└── ros/     ← anvil-eval-ros output
```

### Dataset Replay (`anvil-eval`)

```bash
uv run anvil-eval \
  --checkpoint model_zoo/my-task/checkpoints/last \
  --dataset data/datasets/my-task \
  --num-eps 5 \
  --device cuda
```

Produces per-joint trajectory plots (predicted vs ground-truth) and summary box plots. Evaluates across train/val/test splits.

### ROS2 MCAP Replay (`anvil-eval-ros`)

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

**Common flags:**

| Flag | Description |
|------|-------------|
| `--checkpoint PATH` | Checkpoint directory |
| `--mcap-root PATH` | Raw MCAP directory (e.g. `data/raw/my-task`) |
| `--num-eps N` | Episodes per split (train/val/test) |
| `--episodes "0,3,5"` | Manually specify episode indices |
| `--seed N` | Random seed for episode sampling (default: 42) |
| `--base-inference-config PATH` | Override default `configs/lerobot_control/inference_eval.yaml` |
| `--monitor` | Record per-step CSV + PNG report via inference monitor |

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

---

## 4. Run Inference

All inference scenarios go through `scripts/run_inference.sh`:

```bash
./scripts/run_inference.sh [--fake-hardware] [--monitor] [--echo-topic-only] [COMPOSE_ARGS...]
```

| Flag | Description |
|------|-------------|
| `--fake-hardware` | Simulate 2-PC setup locally (bridge network + CycloneDDS, no real robot) |
| `--monitor` | Enable real-time monitor: records CSV + plots PNG to `./monitor_output/` on exit |
| `--echo-topic-only` | Subscribe and log FPS only — verify DDS connectivity without a model |

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | Host path to checkpoint (**required** for production inference) |
| `CONFIG_FILE` | Custom inference config YAML (default: `./configs/lerobot_control/inference_default.yaml`) |
| `MONITOR_OUTPUT_DIR` | Host dir for monitor output (default: `./monitor_output`) |
| `LEROBOT_EXTRAS` | Policy extras to install in the image (e.g. `pi,smolvla`). Rebuild after changing. |

### Test with Fake Hardware First (Recommended)

```bash
# 1. Verify DDS connectivity + camera FPS (no model, no GPU needed)
./scripts/run_inference.sh --fake-hardware --monitor up --build

# 2. Validate full pipeline with your model (GPU required)
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --fake-hardware up --build --profile inference
```

If `Control Loop` hits 30 Hz, the setup is ready for real hardware.

### Production (Real Robot)

```bash
# Standard inference
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh up --build

# With inference monitor
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --monitor up --build

# Verify DDS connectivity without a checkpoint
./scripts/run_inference.sh --echo-topic-only up --build
```

> **`MODEL_PATH` must be absolute or start with `./`.** Bare relative paths are treated as named Docker volumes.
> ```bash
> MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last   # recommended
> MODEL_PATH=./model_zoo/my-task/checkpoints/last        # also valid
> ```

### Inference Config (`configs/lerobot_control/inference_default.yaml`)

Before running, review this file:

**Model**
```yaml
model:
  task_description: null
  # VLA-only (SmolVLA / Pi0 / Pi0.5): task prompt the model was trained on.
  # null = auto-read from anvil_config.json in the checkpoint (recommended).
```

**Per-model inference tuning** — override checkpoint defaults without retraining:
```yaml
inference_tuning:

  act:
    n_action_steps: null
    # Steps to execute per chunk before re-running inference.
    # null = use training value. Jittery? → raise. Hesitates? → lower.
    temporal_ensemble_coeff: null
    # Re-infers every step with exponentially weighted overlapping predictions.
    # Use 0.01 (paper default). Forces n_action_steps=1.

  diffusion:
    n_action_steps: null
    # Steps to execute per chunk. null = use training value.
    num_inference_steps: 10
    # Denoising iterations at inference time.
    # null = num_train_timesteps (100 steps, ~300ms on GPU).
    # 10   = ~30ms on GPU — recommended for real-time deployment.

  rtc:
    # VLA models only (SmolVLA / Pi0 / Pi0.5)
    inference_delay: 10
    # Fallback step-count before LatencyTracker auto-calibrates.
    # Rule of thumb: ceil(first_inference_ms × control_freq / 1000)
    queue_trigger_threshold: 50
    # Re-trigger inference when ActionQueue depth ≤ this.
    execution_horizon: 12
    # Steps consumed per chunk before the next inference fires.
    max_guidance_weight: 10.0
    prefix_attention_schedule: EXP
```

**Safety limits:**
```yaml
# safety:
#   max_position_delta: 0.1
#   # Hard limit on joint position change per control step (radians).
#   min_position_delta: 0.05
#   # Minimum cumulative change before publishing a new command.
#   # Holds the last command until threshold is crossed — useful for
#   # overcoming motor dead zones / friction. Default: disabled (null).
```

### Distributed Inference Architecture

```
  Anvil Devbox (anvil-loader)             CycloneDDS              GPU PC (anvil-embodied-ai)
┌─────────────────────────────┐    ┌────────────────────┐    ┌─────────────────────────────┐
│  ros2_control               │    │                    │    │  lerobot_control            │
│  joint_states (500 Hz)      │◄───┤  Gigabit Switch    ├───►│  inference_node (30 Hz)     │
│  cameras (4× 30 Hz)         │    │                    │    │  action commands            │
└─────────────────────────────┘    └────────────────────┘    └─────────────────────────────┘
```

The Anvil Devbox streams joint states and camera feeds over CycloneDDS. The GPU PC subscribes to those streams, runs the policy, and publishes action commands back. See the [full documentation](https://docs.anvil.bot/) for network setup.

---

## Project Structure

```
anvil-embodied-ai/
├── packages/
│   ├── mcap_converter/            # MCAP → LeRobot dataset conversion
│   ├── anvil_trainer/             # Training wrapper: transforms, splits, val loss
│   ├── anvil_eval/                # Offline evaluation: dataset replay
│   ├── anvil_eval_ros/            # Offline evaluation: ROS2 MCAP replay
│   └── anvil_shared/              # Shared utilities (pure-Python, no ML deps)
├── ros2/
│   └── src/lerobot_control/       # ROS2 inference node (Jazzy)
├── configs/
│   ├── cyclonedds/                # CycloneDDS peer configs
│   ├── lerobot_control/           # Inference YAML configs (cameras, joints, arms)
│   └── mcap_converter/            # Data conversion configs
├── docs/
│   ├── data-conversion.md         # Data conversion guide
│   ├── training.md                # Model training guide
│   ├── evaluation.md              # Offline evaluation guide
│   └── inference.md               # Inference deployment guide
├── docker/
│   └── inference/                 # Dockerfile + entrypoint
├── scripts/
│   ├── run_inference.sh           # Entry point for all inference scenarios
│   └── plot_monitor_csv.py        # Plot obs.state / raw_output / control_cmd from CSV
├── tests/
│   ├── smoke/                     # End-to-end smoke tests
│   └── unit/                      # Unit tests per package
├── docker-compose.yml                    # Production inference
├── docker-compose.fake-hardware.yml      # Simulate 2-PC setup locally
├── docker-compose.eval.yml               # ROS2 MCAP replay eval stack
├── docker-compose.monitor-smoke-test.yml # Monitor smoke test
├── .env.example                          # Environment variable template
└── model_zoo/                            # Trained checkpoints (gitignored)
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
