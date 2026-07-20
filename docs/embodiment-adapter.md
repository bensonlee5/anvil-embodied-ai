# Shirt-fold embodiment adapter

The Hugging Face folding policy and Anvil OpenArm 2.0 are not the same action
embodiment even though both expose 16 values named as seven joints plus a
gripper per arm. The frozen Hugging Face policy uses the modified OpenArm v1
geometry, including a 50 mm longer upper-arm link and custom jaws. Its processor
state is in degrees. The Anvil demonstrations use OpenArm 2.0 kinematics and arm
joints in radians. Copying slots by name therefore preserves neither units nor
TCP motion.

The adapter keeps the folding VLA frozen and composes four explicit stages:

1. Convert an OpenArm 2.0 observation from radians into a reference OpenArm v1
   state by FK, fixed TCP-frame alignment, bounded multi-start IK, and calibrated
   gripper open fraction.
2. Run the original Hugging Face preprocessor, frozen policy, and postprocessor
   without replacing its normalization state.
3. Convert the predicted reference action chunk back to OpenArm 2.0 through the
   inverse kinematic bridge.
4. Apply a small learned, bounded residual in OpenArm 2.0 joint coordinates.

Reference IK continuity is retained between action chunks and cleared only at
an episode reset. Both the IK result and learned residual are clamped to the
same buffered target command envelope.

The artifact remains `offline_only`. It is not connected to the live ROS command
path by this change.

## Production promotion gates

The current state-only residual is an offline transfer ablation, not a complete
deployment embodiment layer. It may establish whether the frozen folding prior
survives kinematic transfer, but it cannot correct a camera-domain mistake,
choose a different grasp, or make an unreachable source trajectory safe.

Do not change `deployment_status` until all of these blockers are closed:

1. Bind the artifact to independently measured base/TCP and camera intrinsics /
   extrinsics, an independently generated robot model, joint-specific command
   margins, and the complete source model-weight hash.
2. Count every bridge rejection in the evaluation denominator and report it by
   episode, task phase, direction, and reason. A rejected sample is a failed
   prediction, not missing data.
3. Replace pointwise inverse IK at the command boundary with a constrained
   bimanual trajectory layer enforcing position/orientation objectives, joint
   position/velocity/acceleration bounds, executed-state continuity, collision
   constraints, and a defined short-horizon fallback.
4. Validate physical gripper aperture, latency, force/contact and grasp timing;
   add a separate bounded monotonic gripper adapter if endpoint calibration is
   insufficient.
5. Fingerprint the camera contract, trim manifest, episode split, full policy
   weights and generated cache. Refuse candidate publication when absolute,
   rejection-adjusted validation/test gates fail.
6. Pass shadow and low-speed closed-loop rollouts from the demonstrated start
   distribution before any live task-speed evaluation.

After those safety/integrity gates, evaluate a small visual/embodiment adapter,
multiple frozen-policy noise samples per observation, chunk-boundary smoothness,
and data collected from states reached by the adapted policy. The long-term
mixed-embodiment path should supervise a canonical TCP trajectory and gripper
intent with an embodiment token; source and target joint labels must never be
concatenated as if they described the same mechanism.

## Why joint-angle matching is still useful

Direct v1-to-v2 angle equality is not the primary target because different link
lengths produce different TCP positions. Joint matching is useful after the TCP
constraint as a posture/null-space objective: a 7-DoF arm can reach the same TCP
pose with several shoulder/elbow configurations. The bridge uses the current or
previous solution as that posture seed, and the learned loss includes target-side
joint and velocity terms. It never compares a Hugging Face degree value directly
to an OpenArm 2 radian target.

The deterministic bridge is evaluated before training. Samples outside the
overlapping workspace fail closed rather than being silently clipped into an
unrelated motion. Record the accepted/rejected counts from the exact pinned
manifest and dataset in the run metadata; do not carry counts forward after a
kinematic model or joint-limit change.

## Target limits and gripper calibration

The target arm uses these controller-coordinate nominal limits:

| Joint | Right | Left |
|---|---:|---:|
| J1 | −80°…+200° | −200°…+80° |
| J2 | −10°…+190° | −190°…+10° |
| J3 | −90°…+90° | −90°…+90° |
| J4 | 0°…+140° | 0°…+140° |
| J5 | −90°…+90° | −90°…+90° |
| J6 | −70°…+45° | −45°…+70° |
| J7 | −90°…+90° | −90°…+90° |

J1–J5/J7 come from the pinned upstream OpenArm 2.0 description with the
bimanual side convention. Anvil documents the wider v2 J6 qualitatively but
does not publish numeric controller limits; the mirrored J6 range is resolved
from all 33 sessions in the pinned target dataset. Target follower-state extrema
are −70.30°/+44.34° on right J6 and −43.94°/+63.46° on left J6. Small excursions
past a nominal endpoint are treated as encoder/controller tolerance, not as a
reason to expand the mechanical model.

Arm commands remain 0.005 rad (0.286°) inside every nominal endpoint. The
target gripper mapping deliberately uses command endpoints rather than follower
state extrema: −0.003 rad is closed and +0.050 rad is open. The source policy's
corresponding endpoints are 0° and −65°. This retains the small negative close
command present in the target demonstrations instead of weakening it to zero.

## Loss

For bridge output `q_bridge` and bounded residual network `r_theta`, the command
is

```text
q_hat = q_bridge + b * tanh(r_theta)
```

where `b` is at most 0.15 rad and at most 10% of each target joint range. Gripper
corrections are always zero. Training minimizes

```text
L = L_joint + 0.25 L_pose + 0.05 L_velocity
    + lambda_motion L_motion + lambda_residual L_residual
```

- `L_joint`: Smooth L1 on OpenArm 2 joint errors normalized by each joint range.
- `L_pose`: differentiable OpenArm 2 FK position error scaled by 5 cm plus TCP
  geodesic orientation error scaled by 0.25 rad.
- `L_velocity`: Smooth L1 on consecutive target-side joint deltas.
- `L_motion`: Smooth L1 on commanded displacement magnitude from the measured
  current state. This guards against a residual that lowers pointwise error by
  damping the action chunk.
- `L_residual`: squared residual as a fraction of its safety bound.

The pose term handles link-length geometry. The joint and velocity terms select
the demonstrator's shoulder/elbow posture among redundant IK solutions and keep
chunks temporally coherent. The motion term is enabled explicitly for transfer
runs after validation establishes that source and target motion intensity differ;
its weight and the residual weight are recorded in training provenance.

For a production policy, loss weighting is not a substitute for an explicit
temporal contract. Calibrate source-specific control rates before chunking,
normalize action deltas per horizon timestep, execute only a short prefix before
re-planning, and overlap adjacent chunks with a soft continuity anchor. These
principles follow the transfer and inference findings in Larchenko's LeHome 2026
folding system; they must be validated on this embodiment rather than copied as
fixed constants.

## Pinned inputs

The manifest is
`configs/embodiment_adapters/hf_folding_to_anvil_openarm2.json`. It pins:

- `lerobot-data-collection/folding_final` at revision
  `695abe40dbf3aac04efda59c1501d748681fa0fb`;
- both processor JSON files and both normalization-state files by SHA-256;
- the 16-D right-then-left vector contract and degree/radian units;
- the mesh-free modified-v1 and Anvil-v2 kinematic models by SHA-256;
- TCP-frame alignment, IK tolerances/restarts, gripper endpoints, and residual
  bounds.

The normalization-state hash matters: the policy's observed arm features are in
degrees, and swapping in a processor trained on OpenArm 2 radians recreates the
same scale failure the adapter is intended to prevent.

## Offline workflow

Validate the manifest and sampled OpenArm 2 IK coverage without model weights:

```bash
uv run anvil-adapter validate \
  --manifest configs/embodiment_adapters/hf_folding_to_anvil_openarm2.json \
  --dataset datasets/shirt-fold/lerobot \
  --stride 500
```

Obtain the exact frozen policy revision once. This is a large download:

```bash
hf download lerobot-data-collection/folding_final \
  --revision 695abe40dbf3aac04efda59c1501d748681fa0fb \
  --local-dir model_zoo/hf-folding-final
```

Then validate all processor content hashes:

```bash
uv run anvil-adapter validate \
  --manifest configs/embodiment_adapters/hf_folding_to_anvil_openarm2.json \
  --base-policy model_zoo/hf-folding-final \
  --dataset datasets/shirt-fold/lerobot
```

Cache the frozen policy every ten frames. The optional baseline is the existing
5k OpenArm 2 fine-tune and is run directly, without the embodiment bridge:

```bash
uv run anvil-adapter cache \
  --manifest configs/embodiment_adapters/hf_folding_to_anvil_openarm2.json \
  --base-policy model_zoo/hf-folding-final \
  --baseline-policy model_zoo/openarm2-real-topshort-teleop-v1/pi05_openarm2_flat_shirt_expert_v1/checkpoints/005000/pretrained_model \
  --dataset datasets/shirt-fold/lerobot \
  --split-info model_zoo/openarm2-real-topshort-teleop-v1/pi05_openarm2_flat_shirt_expert_v1/checkpoints/005000/pretrained_model/split_info.json \
  --output adapter_cache/shirt-fold-frozen-hf.npz \
  --task "Fold the T-shirt properly" \
  --stride 10 \
  --device cuda
```

Cache generation uses the checkpoint's exact episode split and deterministic
noise seeds. Rejected IK samples are listed next to the cache in a JSON report.
The bridge column is produced with a zero-initialized residual, so it is the
required bridge-only sanity baseline.

The residual is supervised only by target OpenArm 2.0 episodes. The much larger
and more diverse source corpus is not discarded: it is already represented in
the frozen `folding_final` policy weights and pinned processors. Mixing
source-embodiment joint targets into the residual loss would reintroduce the
unit/geometry mismatch; source-only data can instead be used for frozen-policy
regression tests of the reference side of the bridge.

The target rows must come from the pinned phase-aligned trim, not the full raw
33 sessions. Its `meta/trim_manifest.json` reduces 43,625 source frames to
34,850 by removing 8,775 setup/final-idle frames. All 33 episodes have a non-zero
start trim; 27 have a non-zero final-idle trim, while six have no detected final
idle window to remove. The saved train/validation/test assignment must come from
that same dataset revision.

Train only the residual:

```bash
uv run anvil-adapter train \
  --manifest configs/embodiment_adapters/hf_folding_to_anvil_openarm2.json \
  --cache adapter_cache/shirt-fold-frozen-hf.npz \
  --output model_zoo/adapters/hf-folding-to-openarm2-v1 \
  --steps 5000 \
  --device cuda \
  --wandb-project shirt-fold \
  --wandb-run-name hf-folding-to-openarm2-v1-seed-42 \
  --wandb-mode online
```

Authenticate first with `uv run wandb login`. If credentials are unavailable during
training, use `--wandb-mode offline`; the complete run can later be uploaded with
`uv run wandb sync wandb/offline-run-*`.

Every validation interval logs loss terms plus bridge-relative normalized joint,
shoulder, TCP position/orientation, commanded-motion, and residual-bound metrics.
The quality gate passes only when joint, shoulder, and TCP position errors improve,
motion does not collapse, and fewer than 5% of residual values reach 95% of their
safety bound. The pinned manifest/cache and final adapter/evaluation are stored as
W&B artifacts.

Compare hold, deterministic bridge, learned adapter, and the optional current
5k baseline on train/validation/test episodes:

```bash
uv run anvil-adapter evaluate \
  --adapter model_zoo/adapters/hf-folding-to-openarm2-v1 \
  --cache adapter_cache/shirt-fold-frozen-hf.npz \
  --output eval_results/shirt-fold/embodiment-adapter.json \
  --device cuda
```

Reported metrics include range-normalized joint MAE, shoulder J1/J2 MAE, TCP
position/orientation error, commanded motion, and motion ratio. A residual
training run is justified only if the bridge-only result is kinematically sane
and the learned adapter improves validation/test metrics rather than only the
training split.

## Live gate

Do not change `deployment_status` to `live_approved` based only on offline loss.
Promotion requires the devbox connected to the arms, low-speed shadow output,
per-joint direction checks, reachable-pose checks, command-rate measurement,
and a hardware e-stop test. Live artifact loading refuses an `offline_only`
manifest.
