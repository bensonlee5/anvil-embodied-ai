# Inference Guide

End-to-end reference for running the Anvil inference stack — how it's architected, how to deploy it on a single GPU PC or across two PCs, how to choose and configure the DDS middleware, and how to run, tune, and debug it.

For a quick start, see the [Run Inference](../README.md#4-run-inference) section of the README. For training, see [training-tips.md](training-tips.md).

## Table of Contents

- [1. Overview](#1-overview)
- [2. Architecture & Design](#2-architecture--design)
- [3. DDS Middleware & Deployment Modes](#3-dds-middleware--deployment-modes)
- [4. Distributed (Two-PC) Network Setup](#4-distributed-two-pc-network-setup)
- [5. Running Inference](#5-running-inference)
- [6. Configuration Reference](#6-configuration-reference)
- [7. Monitoring & Debugging](#7-monitoring--debugging)
- [8. Troubleshooting](#8-troubleshooting)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Docker + Docker Compose** | Docker Engine with the Compose plugin (V2). |
| **NVIDIA driver** | Must be installed on the host. Verified at `docker compose up` via the GPU reservation in `docker-compose.yml`. |
| **NVIDIA Container Toolkit** | Enables GPU passthrough to containers (`deploy.resources.reservations.devices` in compose). Install via the [NVIDIA Container Toolkit guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). |
| **Matching `ROS_DOMAIN_ID`** | Must be the same integer on every machine and container in the ROS2 graph (both `anvil-loader` and inference). This stack defaults to `1`. |
| **Gigabit LAN (two-PC only)** | A direct cable or dedicated switch between the robot PC and GPU PC is strongly recommended for streaming 4× camera feeds. |

> **VLA models (Pi0/Pi0.5/SmolVLA)** additionally require a HuggingFace account and a populated `HF_CACHE` — see the `HF_CACHE` entry in Section 5.

---

## 1. Overview

The inference stack runs a trained manipulation policy in real time. It:

1. **Subscribes** to robot state and camera feeds published by `anvil-loader` (the workcell control stack):
   - `/joint_states` — `sensor_msgs/JointState`, ≈500 Hz
   - camera topics — `sensor_msgs/CompressedImage`, 4× 30 Hz (waist, chest, both wrists)
2. **Runs** the policy at a fixed control rate (`control_frequency`, default **30 Hz**), assembling observations and producing joint-position action chunks.
3. **Publishes** commands back to the workcell's controllers:
   - `/follower_l_forward_position_controller/commands` and `/follower_r_forward_position_controller/commands` — `std_msgs/Float64MultiArray`

Everything runs in Docker (`ghcr.io/anvil-robotics/lerobot-inference`) and communicates over ROS2 (DDS). There are two deployment topologies:

| Topology | Where it runs | Transport |
|----------|---------------|-----------|
| **Single-GPU-PC** | `anvil-loader` + inference on the **same machine** | DDS over the host's loopback/local interface |
| **Distributed (two-PC)** | robot PC runs `anvil-loader`; a separate **GPU PC** runs inference | DDS over the LAN (Gigabit switch) |

Both topologies use the same inference node and config; only the DDS middleware setup differs (see [Section 3](#3-dds-middleware--deployment-modes)).

---

## 2. Architecture & Design

### System data flow

```
   anvil-loader (workcell)                DDS                inference (GPU)
┌───────────────────────────┐                       ┌─────────────────────────────┐
│  ros2_control             │   /joint_states 500Hz │  LeRobotInferenceNode       │
│  joint_states publisher   │ ────────────────────► │   _obs_update  (producer)   │
│  4× camera publishers     │   /cam_*/compressed   │   _publish_loop (consumer)  │
│                           │ ────────────────────► │                             │
│  forward_position         │   /follower_*/        │   policy (ACT/Diffusion/VLA)│
│  controllers              │ ◄──────────────────── │   action commands  30Hz     │
└───────────────────────────┘  commands             └─────────────────────────────┘
```

_Shorthand: `/cam_*/compressed` represents the four topics listed in `camera_mapping` — see Section 6 for the exact names._

The node (`ros2/src/lerobot_control/lerobot_control/inference_node.py`, class `LeRobotInferenceNode`) is the heart of the stack.

### Split-timer design

The node separates **producing** actions from **publishing** them, using two ROS2 timers that both fire at `1 / control_frequency`, each in its own `MutuallyExclusiveCallbackGroup`, driven by a `MultiThreadedExecutor(num_threads=4)`:

- **`_obs_update`** (producer) — assembles the latest observation and feeds the policy.
- **`_publish_loop`** (consumer) — pops one action from a buffer and publishes it to the controllers.

Decoupling the two means publishing stays at a steady control rate even when a single inference call is slow or jittery — the consumer just drains whatever the producer has buffered.

### Multi-process image decoding

JPEG decompression for four cameras is CPU-heavy and would contend on the Python GIL inside the inference process. The node delegates this to `MultiProcessStrategy` (`strategies/multi_process.py`):

- One **spawned OS process per camera** subscribes to its `CompressedImage` topic, decodes JPEG (`cv2.imdecode`), converts BGR→RGB, and resizes (aspect-preserving, with padding) to the model's input resolution.
- Decoded frames are written into a **`SharedImageBuffer`** — zero-copy `multiprocessing.shared_memory`, one block per camera holding pixel bytes + timestamp + frame counter.
- The main process reads all camera blocks via `read_all_if_ready()`, which returns an observation **only when every camera has a fresh frame**. If a camera is stale/missing, `get_observation()` returns `None` and records which camera was missing (`_last_incomplete_reason`).
- Joint state is lightweight, so it stays in the **main process** (a plain subscription callback).

This gives true parallelism for decode, process isolation (a crashing decoder doesn't take down inference), and a clean "all cameras synced" gate on every observation.

### Observation assembly & action publishing

- **Observation**: images become `observation.images.{name}` tensors (normalized, CHW, batched); joint values become `observation.state` ordered by `model_joint_order` × `arm_mapping` from the config.
- **Publishing** (`_publish_action`): the action vector is sliced per arm using each arm's `action_start:action_end`, passed through `ActionLimiter` (safety clamping), packed into a `Float64MultiArray`, and published to that arm's `command_topic`. When `monitor_enable` is set, the node also publishes `/monitor/{obs_state,raw_output,control_cmd}` for offline plotting.

> **Image resolution is auto-detected** from the checkpoint's `config.json` (the first VISUAL `input_features` entry), not set at launch. Inference resolution therefore always matches training — there's no way to accidentally feed the model the wrong size.

### Two execution paths (the key concept)

The split-timer skeleton is identical for every model, but **how `_obs_update` produces actions differs by model family**:

**ACT / Diffusion — synchronous, per-chunk**
- The policy runs **inline** in `_obs_update` via `model.select_action()` (under `torch.inference_mode()`).
- Output actions land in `_classic_action_deque`; `_publish_loop` pops from it.
- Latency is bounded by how fast `select_action()` returns. No background thread.

**VLA (SmolVLA / Pi0 / Pi0.5) — background RTC thread**
- `_obs_update` only **preprocesses and snapshots** the observation into `_latest_obs`; it does *not* run the model.
- A dedicated **daemon thread** (`_inference_loop`) runs `model.predict_action_chunk()` continuously, refilling an `ActionQueue` whenever its depth falls to/below `queue_trigger_threshold`.
- This is **RTC (real-time chunking)**: the thread measures inference latency, computes an adaptive `inference_delay` (`ceil(latency × control_freq)`), and merges raw + post-processed chunks back into the queue with guidance so consecutive chunks blend smoothly.
- `_publish_loop` drains the `ActionQueue` at the control rate (skipping, and counting `_vla_skip_count`, if the queue is momentarily empty).

The operator takeaway: **ACT/Diffusion compute on the control thread; VLA decouples slow chunk inference into a background thread** so the publish rate stays smooth at `control_frequency` even when one VLA forward pass spans many control periods.

### Checkpoint loading

`ModelLoader` (`model_loader.py`) auto-detects the model type from `config.json`, loads the matching LeRobot policy class, applies any `inference_tuning` overrides onto the model config post-load, and loads the `policy_preprocessor.json` / `policy_postprocessor.json` pipelines (normalization, resize). Action-type handling (`absolute` vs `delta_obs_t` vs `delta_sequential`) and `delta_exclude_joints` are read from the checkpoint's `anvil_config.json` and applied as a chunk-level delta-restore so per-step publishing never re-enters normalized space.

---

## 3. DDS Middleware & Deployment Modes

ROS2 communicates over a DDS implementation selected by the `RMW_IMPLEMENTATION` environment variable. Two are supported:

- **CycloneDDS** (`rmw_cyclonedds_cpp`) — the **default**; faster and more reliable in our testing, and required for cross-machine discovery.
- **Fast DDS** (`rmw_fastrtps_cpp`) — ROS2's built-in default; available as an opt-in for single-PC use.

> ⚠ **Both sides of the link must use the same RMW.** If `anvil-loader` runs Fast DDS and inference runs CycloneDDS (or vice versa), they will **not discover each other at all** — no error, just silence.

### Selection table

| Deployment | `RMW_IMPLEMENTATION` | `CYCLONEDDS_URI` | anvil-loader `.env.config` |
|-----------|----------------------|------------------|----------------------------|
| **Single-PC · CycloneDDS** *(default)* | `rmw_cyclonedds_cpp` | `file:///workspace/configs/cyclonedds/single_pc.xml` | `ENABLE_CYCLONEDDS=true`<br>`CYCLONEDDS_PEER_IP=127.0.0.1` |
| Single-PC · Fast DDS | `rmw_fastrtps_cpp` | *(ignored)* | `ENABLE_CYCLONEDDS=false` |
| Two-PC · CycloneDDS | `rmw_cyclonedds_cpp` | `file:///workspace/configs/cyclonedds/gpu_pc.xml` | `ENABLE_CYCLONEDDS=true`<br>`CYCLONEDDS_PEER_IP=<gpu_pc_ip>`<br>(generates `robot_pc.xml` at runtime — see Section 4) |

All CycloneDDS configs live in `configs/cyclonedds/`. `ROS_DOMAIN_ID` must match on both sides (this stack defaults to `1`; the ROS 2 native default is `0` — any other ROS 2 tooling on the same subnet must use the same value to participate).

### How config flows

```
.env  ──►  docker-compose.yml env  ──►  docker/inference/entrypoint.sh  ──►  ROS2
```

- `docker-compose.yml` defaults both inference services to CycloneDDS + `single_pc.xml`:
  ```yaml
  - RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
  - CYCLONEDDS_URI=${CYCLONEDDS_URI:-file:///workspace/configs/cyclonedds/single_pc.xml}
  ```
- `entrypoint.sh` **unsets** `RMW_IMPLEMENTATION` / `CYCLONEDDS_URI` if they arrive empty, so ROS2 falls back cleanly to its Fast DDS default.
- Override any variable in `.env` (copied from `.env.example`) to switch modes.

### Single-PC: why no peers

`single_pc.xml` deliberately has **no `<Peers>` section**. Because `anvil-loader` and inference share the host network (`network_mode: host`), CycloneDDS multicast on the autodetermined interface discovers the other participant on the same machine automatically. Setting `CYCLONEDDS_PEER_IP=127.0.0.1` on the loader side just seeds unicast discovery as a belt-and-suspenders.

To switch single-PC to **Fast DDS**: set `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` in `.env` and `ENABLE_CYCLONEDDS=false` in the loader's `.env.config`. `CYCLONEDDS_URI` is ignored under Fast DDS, so it's harmless to leave set.

---

## 4. Distributed (Two-PC) Network Setup

In the two-PC topology, the robot PC runs `anvil-loader` and a separate GPU PC runs inference, connected over a LAN (ideally a dedicated Gigabit switch). Each machine needs a CycloneDDS config that names the **other** machine as a peer.

```
  Robot PC (anvil-loader)             CycloneDDS              GPU PC (anvil-embodied-ai)
┌─────────────────────────────┐    ┌────────────────────┐    ┌─────────────────────────────┐
│  robot_pc.xml               │    │                    │    │  gpu_pc.xml                 │
│  Peer = <gpu_pc_ip>         │◄───┤  Gigabit Switch    ├───►│  Peer = <robot_pc_ip>       │
│  joint_states + cameras     │    │                    │    │  inference_node             │
└─────────────────────────────┘    └────────────────────┘    └─────────────────────────────┘
```

### Steps

1. **Assign static IPs** to both machines on the shared subnet (e.g. `192.168.0.146` GPU, `192.168.0.128` robot).
2. **Edit `configs/cyclonedds/gpu_pc.xml`** on the GPU PC:
   - Set `<NetworkInterface name="eno1"/>` to the GPU PC's actual interface (or use `autodetermine="true"`).
   - Set `<Peer address="..."/>` to the **robot PC's** IP.
3. **`configs/cyclonedds/robot_pc.xml`** on the robot PC is generated by the loader at runtime from `CYCLONEDDS_PEER_IP` in `.env.config` — set the GPU PC's IP there rather than hand-editing the XML (manual edits are overwritten on regeneration). Edit the XML directly only when running the loader outside the generated flow.
4. **Match `ROS_DOMAIN_ID`** on both sides.
5. On the GPU PC `.env`: `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, `CYCLONEDDS_URI=file:///workspace/configs/cyclonedds/gpu_pc.xml`.
6. On the robot PC `.env.config`: `ENABLE_CYCLONEDDS=true`, `CYCLONEDDS_PEER_IP=<gpu_pc_ip>`.
7. **Verify** with `--echo-topic-only` (see [Section 5](#5-running-inference)) before loading a model.

### Performance tuning (already in the configs)

The two-PC configs set transport parameters sized for streaming four camera feeds:

| Setting | Value | Purpose |
|---------|-------|---------|
| `MaxMessageSize` | `65535B` | Max single DDS message |
| `FragmentSize` | `4000B` | Fragmentation unit for large samples |
| `SocketReceiveBufferSize` | `min 128MB` | Absorbs camera bursts without drops |
| `WhcHigh` | `8MB` | Writer history cache high-water mark |

> The 128 MB receive buffer requires the OS to allow it. If you see CycloneDDS warnings about the socket buffer, raise the kernel limit:
> ```bash
> sudo sysctl -w net.core.rmem_max=134217728
> ```
> `-w` is not persistent across reboots. To persist:
> ```bash
> echo 'net.core.rmem_max=134217728' | sudo tee /etc/sysctl.d/99-cyclonedds.conf
> sudo sysctl -p /etc/sysctl.d/99-cyclonedds.conf
> ```

### Gotchas

- **Multicast must be allowed** on the subnet/switch for discovery; if it's blocked, the explicit `<Peer>` unicast entries still establish the link.
- **Firewalls** (`ufw`, corporate) commonly block DDS's UDP discovery/data ports — open them or disable the firewall on a trusted LAN to test.
- **Interface mismatch** is the most common failure: a wrong `<NetworkInterface name>` makes CycloneDDS bind to the wrong NIC. Use `autodetermine="true"` if unsure.

---

## 5. Running Inference

All scenarios go through `scripts/run_inference.sh`, a thin wrapper over `docker compose`:

```bash
./scripts/run_inference.sh [--fake-hardware] [--monitor] [--echo-topic-only] [COMPOSE_ARGS...]
```

| Flag | Effect |
|------|--------|
| `--fake-hardware` | Use `docker-compose.fake-hardware.yml` — a simulated 2-PC bridge network with a mock robot publisher; no real hardware or GPU needed for the connectivity check |
| `--monitor` | Enable the monitor profile; records `/monitor/*` to CSV and auto-plots `inference_report.png` on exit |
| `--echo-topic-only` | Subscribe and log topic FPS **without loading a model** — verifies DDS connectivity and camera/joint throughput |

All other arguments (`up --build`, `down`, `logs`, `--profile inference`, …) pass straight through to `docker compose`.

### Environment variables

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | **Host path** to the checkpoint dir. Must be absolute or start with `./` (bare relative paths become Docker named volumes). Required for production inference. |
| `CONFIG_FILE` | Inference YAML (default: `./configs/lerobot_control/inference_default.yaml`) |
| `HF_CACHE` | HuggingFace cache dir — required for VLA models, which download their VLM backbone (tokenizer + weights) at runtime: PaliGemma for Pi0/Pi0.5, SmolVLM-2 for SmolVLA. |
| `IMAGE_TAG` | Docker image tag (default `latest`) |
| `LEROBOT_EXTRAS` | Policy extras to bake into the image (e.g. `pi`, `smolvla`); rebuild after changing |
| `MONITOR_OUTPUT_DIR` | Host dir for monitor output (default `./monitor_output`) |
| `DEBUG` | Set to `true` to enable debug metrics inside the container: action smoothness, queue-depth stats, Action FPS, and pre-model input frame dumps (see Section 7). |

> **`--monitor` vs `monitor_enable`:** The `--monitor` flag in `run_inference.sh` activates the Docker Compose `monitor` profile (which starts the `inference-monitor` service) **and** sets `MONITOR_ENABLE=true` inside the main container. Setting `MONITOR_ENABLE=true` in `.env` alone enables `/monitor/*` topic publishing *without* starting the separate monitor service — useful when you have a custom subscriber.

`run_inference.sh` auto-detects `ACTION_TYPE` from the checkpoint's `anvil_config.json` (checking `pretrained_model/`, the root, then HF `snapshots/`), so you rarely need to set it manually.

### Recommended progression

Work up from cheapest-to-verify to full deployment:

```bash
# 1. Connectivity check — no model, confirms DDS + topic rates
./scripts/run_inference.sh --echo-topic-only up --build

# 2. Simulated 2-PC pipeline with a mock robot (no real hardware)
./scripts/run_inference.sh --fake-hardware --monitor up --build

# 3. Production inference on the real robot (GPU required)
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
  ./scripts/run_inference.sh up --build

# 3b. …with the real-time monitor + auto-plot
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
  ./scripts/run_inference.sh --monitor up --build
```

If the `Control Loop` reaches **30 Hz** in the logs, the setup is ready.

In `--echo-topic-only` mode, confirm the middleware in the container logs:

```
[entrypoint] RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
[entrypoint] CYCLONEDDS_URI=file:///workspace/configs/cyclonedds/<your-config>.xml
```

### Shutdown behaviour

When the inference container stops (Ctrl-C, `docker compose down`, or SIGINT):

1. `_shutting_down` is set, blocking new callback executions.
2. All ROS2 timers (`_obs_timer`, `_publish_timer`, `_stats_timer`) are cancelled.
3. The VLA background inference thread is signalled to stop and joined (2 s timeout).
4. If the node published **at least one** action during this session, `_publish_hold_position()` fires: it reads the current joint positions and publishes them once to every arm controller — the arm will **hold its last commanded position**.
5. If the node never published (e.g. crashed before the first action), no hold command is sent. Arm behaviour then reverts to whatever `anvil-loader`'s `forward_position_controller` does on loss-of-input.

> **On SIGKILL or OOM kill:** no hold command is sent. Prefer SIGINT (`Ctrl-C` / `docker compose stop`) to ensure the hold fires.

---

## 6. Configuration Reference

The inference YAML (`configs/lerobot_control/inference_default.yaml`) controls everything that isn't a launch arg. Model type and training flags are read from the checkpoint — only set fields here to override or to provide runtime wiring.

### `model`
```yaml
model:
  task_description: null   # VLA prompt (SmolVLA/Pi0/Pi0.5); null = auto-read from anvil_config.json
  dtype: null              # optional: "bfloat16" | "float16" | "float32". null = checkpoint default.
                           # Setting "bfloat16" halves VRAM for VLA models. Validate on target GPU first.
```

### `inference_tuning`
Only the block matching your model type is used.

```yaml
inference_tuning:
  act:
    n_action_steps: null          # steps executed per chunk before re-inferring. Jittery → raise, hesitant → lower
    temporal_ensemble_coeff: null # 0.01 = paper default; ⚠ setting this forces n_action_steps=1

  diffusion:
    n_action_steps: 10            # shipped default (checkpoint default is 8)
    num_inference_steps: 10       # denoising steps; 10 ≈ 78ms vs 100 ≈ 711ms (TODO: annotate GPU model). Big real-time win.

  rtc:                            # VLA only (SmolVLA / Pi0 / Pi0.5) — always enabled
    # How the three control parameters interact:
    #   inference_delay: fallback step-count used BEFORE LatencyTracker collects enough
    #     samples. Set to ceil(first_inference_ms × control_freq / 1000). Auto-adjusts
    #     from measured latency after the first inference completes.
    #   queue_trigger_threshold: the ActionQueue depth at which the background thread
    #     fires a new predict_action_chunk() call. Must be high enough that the queue
    #     doesn't run dry between triggers. Rule of thumb: set ≥ execution_horizon.
    #   execution_horizon: how many steps from each chunk the publish loop consumes
    #     before the background thread re-infers. Lower = faster reaction, higher GPU load.
    #     Higher = fewer re-inferences, slower adaptation. chunk_size is the hard ceiling.
    inference_delay: 10           # steps; ~ceil(330ms × 30Hz); auto-calibrates after first inference
    queue_trigger_threshold: 50   # re-trigger when ActionQueue depth ≤ this (set ≥ execution_horizon)
    execution_horizon: 12         # steps consumed per chunk before next re-inference
    max_guidance_weight: 10.0     # RTC guidance blend weight
    prefix_attention_schedule: EXP  # attention schedule for prefix tokens (EXP or LINEAR)
```

### Joints, arms, cameras
```yaml
joint_state_topic: "/joint_states"
joint_names:
  model_joint_order: [...]        # order the model outputs (must match training)
  controller_joint_order: [...]   # order the ROS2 controller expects
  arm_mapping: {l: left, r: right}
  state_features: [position]

arms:
  left:  { ros_prefix: follower_l, command_topic: /follower_l_forward_position_controller/commands, action_start: 0, action_end: 8 }
  right: { ros_prefix: follower_r, command_topic: /follower_r_forward_position_controller/commands, action_start: 8, action_end: 16 }

camera_mapping:
  "/cam_waist/image_raw/compressed":  "waist"
  "/cam_wrist_r/image_raw/compressed": "wrist_r"
  "/cam_chest/image_raw/compressed":  "chest"
  "/cam_wrist_l/image_raw/compressed": "wrist_l"
```

The `camera_mapping` keys are the ROS topics; the values are the model's camera names (must match training). The `arms` slices (`action_start:action_end`) split the flat action vector across controllers.

### Safety
```yaml
# safety:
#   max_position_delta: 0.1   # hard cap on joint change per step (rad)
#   min_position_delta: 0.05  # min cumulative change before publishing — overcomes motor dead zones
```
Disabled by default. `min_position_delta` holds the last command until the model's cumulative delta crosses the threshold — useful when delta outputs are too small to overcome friction.

### Config variants
- `inference_single_arm.yaml` — single-arm (8-DOF) setups.
- `inference_eval.yaml` — used by the offline ROS2 MCAP replay eval stack.

Point `CONFIG_FILE` at any of these (or your own copy) to override the default.

### Launch parameters

The following are set as **ROS2 launch arguments** (declared in `launch/inference.launch.py`) — they are **not** read from the YAML config file:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `control_frequency` | `30.0` Hz | Master control rate. Drives both `_obs_update` (producer) and `_publish_loop` (consumer) timers at `1 / control_frequency`. All step-based parameters (`n_action_steps`, `execution_horizon`, `inference_delay`) scale with this rate. |
| `device` | `cuda` | Inference device (`cuda` or `cpu`). |
| `echo_topic_only` | `false` | Subscribe and log FPS without loading a model. |
| `debug` | `false` | Enable debug metrics (action smoothness, queue-depth stats, Action FPS, frame dumps). |
| `monitor_enable` | `false` | Publish `/monitor/*` topics (set automatically by `--monitor` flag in `run_inference.sh`). |
| `deterministic` | `false` | Enable deterministic mode (fixes seeds, disables cuDNN benchmarking). |
| `deterministic_seed` | `42` | Seed used when `deterministic=true`. |

These are passed through `docker-compose.yml`'s `command:` line and cannot be set in the YAML config.

---

## 7. Monitoring & Debugging

| Tool | How | What you get |
|------|-----|--------------|
| **Real-time monitor** | `--monitor` | Records `/monitor/*` to `monitor_output/inference_data.csv`; on exit auto-plots `inference_report.png` (per-joint obs/raw-output/control-cmd overlay) via `scripts/plot_monitor_csv.py` |
| **Topic echo** | `--echo-topic-only` | Per-topic FPS for joints and each camera, with no model loaded — isolates DDS/connectivity from model issues |
| **Debug mode** | `DEBUG=true` | Adds action-smoothness, queue-depth stats, Action FPS; dumps pre-model input frames to `debug_images/` for training/inference comparison |

The `inference_report.png` "Raw" trace shows model output **before** postprocessing — useful for telling whether a tracking error comes from the policy itself or from the postprocessor/delta-restore.

---

## 8. Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| No topics discovered at all | **RMW mismatch** — both sides must use the same `RMW_IMPLEMENTATION`. Or **`ROS_DOMAIN_ID` mismatch**. Or firewall blocking multicast. |
| Two-PC: still no discovery | Wrong `<NetworkInterface name>` or `<Peer address>` in the XML; firewall on the DDS UDP ports; multicast disabled on the switch. |
| CycloneDDS socket-buffer warning | Raise the kernel limit: `sudo sysctl -w net.core.rmem_max=134217728`. |
| Low / zero camera FPS | A camera topic name in `camera_mapping` doesn't match what `anvil-loader` publishes; `get_observation()` waits for **all** cameras (`_last_incomplete_reason` names the missing one). |
| Control loop below 30 Hz | Can be **GPU compute** *or* **CPU/decode contention** — the 4× JPEG decode + resize runs in worker processes and competes for CPU/memory, so a slow loop is not always the GPU. Use `DEBUG=true` to compare Action FPS vs Control FPS and isolate which. **If GPU-bound:** Diffusion → lower `num_inference_steps` (try 10) and/or `n_action_steps`; ACT → lower `n_action_steps`. **If CPU/decode-bound:** reduce camera count/resolution, check decode-worker CPU affinity, or use a faster JPEG decode path. |
| Jittery / oscillating motion | Raise `n_action_steps` (execute more of each chunk before re-planning). |
| Hesitant / freezing motion | Lower `n_action_steps`. |
| VLA skips actions (`_vla_skip_count` rising) | Queue starvation — raise `queue_trigger_threshold` so the background thread refills sooner, or lower `execution_horizon`. |
| `MODEL_PATH` errors / empty mount | Use an absolute path or one starting with `./`; bare relative paths become Docker named volumes. Point at the checkpoint dir (auto-detects `pretrained_model/` and HF `snapshots/`). |
| Motors don't move on small deltas | Set `safety.min_position_delta` (start at `0.05`) to accumulate change past the motor dead zone. |

### Pi0 / Pi0.5 memory footprint (known upstream limitation)

Pi-series policies in LeRobot use noticeably more RAM/VRAM than their weight size implies. The main causes are the attention implementation falling back to the inefficient `eager` path instead of `flex`/`fa2`, full-precision loading, and LeRobot's low-VRAM evaluation path still being in progress upstream (see [huggingface/lerobot#3134](https://github.com/huggingface/lerobot/issues/3134) and [#862](https://github.com/huggingface/lerobot/issues/862)). Anvil treats this as a known constraint and provisions GPU memory accordingly rather than patching upstream. To minimise the footprint, confirm `attention_implementation` resolves to `flex` or `fa2` (not `eager`) and that the model is loaded in `bfloat16`.

---

## See also

- [README — Run Inference](../README.md#4-run-inference) — quick start
- [training-tips.md](training-tips.md) — training defaults, checkpoint layout, W&B
- [docs.anvil.bot](https://docs.anvil.bot/) — full platform documentation and network setup
