# OpenArm task-space action representation

`anvil-openarm2-bimanual-tcp-delta-outward-elbow-v1` is an offline
representation experiment for the 33-session shirt-fold dataset. It replaces
the 16-D joint-command target with a 14-D target:

```text
right: delta TCP translation (3), delta TCP rotation vector (3), gripper (1)
left:  delta TCP translation (3), delta TCP rotation vector (3), gripper (1)
```

Every horizon is relative to the observation at time `t`; future targets are
not recursively differenced. Translation and rotation vectors are expressed
in the pinned robot-base frame. Grippers retain the demonstrated absolute
command contract.

This collapses the important kinematic symmetry: many 7-DoF joint
configurations represent the same 6-DoF tool pose. The policy does not need to
memorize which redundant elbow branch happened to appear in each demonstration.
It predicts the task motion, and a deterministic solver selects a joint
trajectory.

## Outward elbows

“Elbow out” is geometric, not a fixed preferred joint vector. For each arm the
solver takes the shoulder-to-TCP line as the swivel axis, projects both the
shoulder-to-elbow vector and the side-specific outward base-frame axis onto the
plane perpendicular to that line, and maximizes their cosine alignment.

The right-arm outward axis is negative base-frame Y; the left-arm axis is
positive Y. The objective is projected through the TCP Jacobian nullspace:

```text
dq = J# e_tcp
   + (I - J#J) (
       w_elbow grad(elbow_alignment)
       - w_continuity normalized(q - q_previous)
       + w_center normalized(q_midpoint - q)
     )
```

TCP tracking remains primary. The elbow objective is active only below the
pinned target alignment, so it cannot redefine the task pose. Continuity and
joint centering resolve the remaining ambiguity and discourage branch flips.

## Hard trajectory constraints

The decoder solves the chunk sequentially at 30 Hz. Each waypoint is restricted
to the intersection of:

- the pinned OpenArm 2 joint range with a 0.005 rad inward margin;
- the previous command plus/minus the joint-specific velocity allowance;
- the previous velocity plus/minus the joint-specific acceleration allowance.

No sigmoid or per-joint distance-to-limit warp is used. Bounds are properties
of the trajectory solver, not the learned task coordinate. An infeasible TCP
waypoint fails closed rather than returning an unrelated clipped joint pose.

The task-space targets use train-only robust, per-horizon normalization. A
two-pass cubic B-spline smooths pose deltas within gripper-event segments before
trajectory solving. Grippers are copied unchanged and segment endpoints are
preserved.

## Versioned surfaces

- Contract:
  `configs/training/action_contracts/openarm2_shirt_fold_task_space_outward_v1.json`
- Matched five-stage raw-SARM recipe:
  `configs/training/shirt_fold_pi05_hf_task_space_outward_5stage_sarm_raw_v4.yaml`
- Codec and serialized processors:
  `packages/anvil_trainer/src/anvil_trainer/task_space_actions.py`
- Constrained solver:
  `packages/anvil_embodiment/src/anvil_embodiment/trajectory.py`

The old bounded-joint v2 contract remains immutable so existing checkpoints
stay reproducible.

## Offline audit

Run the representation/solver audit without decoding video:

```bash
uv run python scripts/training/audit_openarm2_task_space_actions.py \
  --dataset datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-5stage-v1 \
  --contract configs/training/action_contracts/openarm2_shirt_fold_task_space_outward_v1.json \
  --output artifacts/task_space_actions/openarm2_shirt_fold_task_space_outward_v1_stride300.json \
  --stride 300
```

The current stride-300 audit selects 114 complete chunks (3,420 waypoints per
arm). It reports zero joint-position, velocity, or acceleration violations and
99.18% pose-tolerance convergence. The 56 rejected arm-waypoints are retained
as failures; they are concentrated in fast demonstration segments whose command
trajectory outruns the pinned 30 Hz velocity envelope. The train-only target
clip fraction is 0.000429%.

## Production boundary

The v1 contract is deliberately `offline_only`. It enforces joint position,
velocity, and acceleration constraints and an outward-elbow posture, but its
mesh-free kinematic model does not contain self-collision, arm-arm collision,
table, garment, or scene geometry. Production promotion therefore still
requires:

1. full-geometry bimanual collision constraints and a short-horizon fallback;
2. measured base, TCP, and camera calibration hashes;
3. parity between the pinned solver limits and the deployed hardware
   controller;
4. executed-state feedback, stale-observation handling, and re-planning;
5. shadow, low-speed, and task-speed rollout gates.

Offline loss must be reported in the 14-D task-action order. It is not directly
comparable per dimension with a 16-D joint-action checkpoint. Candidate
selection should include FK pose error, solver rejection rate, hard-limit
incidence, outward-alignment distribution, gripper-event accuracy, and
closed-loop rollout success.
