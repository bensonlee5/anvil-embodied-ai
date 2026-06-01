# Inference Guide Revision Summary

Changes applied to `docs/inference-guide.md`, `ros2/src/lerobot_control/lerobot_control/inference_node.py`,
`ros2/src/lerobot_control/lerobot_control/model_loader.py`,
`ros2/src/lerobot_control/lerobot_control/strategies/multi_process.py`, and
`configs/lerobot_control/inference_default.yaml`.

---

## Changed

| Item | What was done |
|------|---------------|
| **A1** | `HF_CACHE` description corrected: PaliGemma tokenizer for Pi0/Pi0.5, SmolVLM-2 for SmolVLA |
| **A2** | "Control loop below 30 Hz" troubleshooting row updated: cause is GPU *or* CPU/decode contention (4× JPEG worker processes); use `DEBUG=true` to isolate which |
| **A3** | Pi0/Pi0.5 memory footprint callout added (before "See also"): `eager` attention path, full-precision loading, upstream lerobot#3134 / #862 |
| **A4** | Section 4 step 3: clarified that `robot_pc.xml` is generated at runtime from `CYCLONEDDS_PEER_IP` — manual edits are overwritten on regeneration |
| **A5** | New "### Launch parameters" table in Section 6 documenting `control_frequency`, `device`, `echo_topic_only`, `debug`, `monitor_enable`, `deterministic`, `deterministic_seed` as ROS2 launch args (not YAML fields) |
| **A6** | Added `DEBUG` row to the Section 5 env-var table; added `--monitor` vs `MONITOR_ENABLE` interaction note |
| **A7** | New "## Prerequisites" section: Docker + Compose V2, NVIDIA driver, NVIDIA Container Toolkit, matching `ROS_DOMAIN_ID`, Gigabit LAN (two-PC only) |
| **A8** | New "### Shutdown behaviour" subsection: hold-position fires if node published ≥1 action; steps 1-5 listed; SIGKILL caveat documented |
| **A9** | `rtc:` YAML block in Section 6 expanded with inline explanation of how `inference_delay`, `queue_trigger_threshold`, and `execution_horizon` interact |
| **A10** | 8 polish items: `ROS_DOMAIN_ID` default note (stack uses 1, ROS 2 native default is 0); `sysctl` persistence via `/etc/sysctl.d/`; `SmolVLA/Pi0/Pi0.5` naming consistency; `⚠ setting temporal_ensemble_coeff forces n_action_steps=1`; diagram topics note pointing to Section 6 `camera_mapping`; selection-table Two-PC row notes `robot_pc.xml` generation; benchmark timings annotated with hardware TODO; echo-log genericised (removed `single_pc.xml` hardcode) |
| **B2** | `dtype` opt-in knob added to `ModelLoader` (`model.dtype` in YAML, default `null` = no cast, behaviour unchanged); effective dtype always logged at load; `inference_default.yaml` updated with the field |
| **B3** | `attention_implementation` logged for VLA models at load time; `warn` emitted when value is `"eager"` with a pointer to the Pi memory note; TODO comment left for override injection |
| **B5** | `_move_to_device` now uses `non_blocking=True` for CPU→GPU tensor transfers; TODO comment left for pinned SharedImageBuffer |
| **B9** | VLA skip count always logged (no longer DEBUG-only) with actionable tuning hint; resets every stats window |

---

## Skipped

| Item | Reason |
|------|--------|
| **B1** | Already correct — VLA `_inference_loop` intentionally omits `torch.inference_mode()`: `RTCProcessor` calls `torch.enable_grad()` internally for guidance gradients; wrapping with `inference_mode()` would silently zero them |

---

## Needs human decision / TODOs left in code

| Item | Location | What's needed |
|------|----------|---------------|
| **B2 default** | `model_loader.py` | Consider defaulting `dtype` to `"bfloat16"` for VLA models once validated on real GPU hardware (`# TODO(inference-opt)`) |
| **B3 override** | `model_loader.py` | `attention_implementation` override is model-specific; safe values differ between Pi0, Pi0.5, SmolVLA — TODO left for human to evaluate per-model |
| **B4** | `inference_node.py` | Startup warmup forward pass — correct dummy-obs shapes/dtypes vary across 5 model families; must be validated on GPU (`# TODO(inference-opt B4)`) |
| **B6** | `multi_process.py` | Decode path: (1) confirm libjpeg-turbo linkage in Docker image; (2) evaluate CPU affinity for worker processes; (3) measure camera Hz vs target before concluding. TODO with instructions in `_start_workers` |
| **B7** | `multi_process.py` | `torch.compile` opt-in for ACT/Diffusion — TODO alongside B6; unverifiable offline; must pair with B4 warmup |
| **B8** | `multi_process.py` | Per-camera max-staleness tolerance — semantics/safety change that must default off; TODO with design spec in `get_observation()` |
| **Branch discrepancy** | `docs/inference-guide.md` | Guide documents `single_pc.xml` as the CycloneDDS default, but that file exists only on `patrick/single-pc-inference`. Recommend merging the guide onto that branch, or cherry-picking `single_pc.xml` + compose defaults to `main` before merging this branch |
