"""SE(3) EE action transforms shared by trainer, anvil_eval, and ROS inference.

Layout conventions
------------------
State per arm  (8 dims): [x, y, z, qx, qy, qz, qw, gripper]
Action per arm (10 dims): [x, y, z, r0, r1, r2, r3, r4, r5, gripper]

  - Rotation in action is 6D (rot6d, Zhou et al. 2019): first two columns of the
    3×3 rotation matrix, stacked column-major as [R[:,0], R[:,1]].
  - Rotation in state is quaternion [qx, qy, qz, qw] (ROS / TF2 convention).
  - Gripper is in metres; training keeps it in absolute space (no delta).

Bimanual: state (16,), action (20,) — left arm first, right arm second.

Public API
----------
n_arms_from_dims(state_dim, action_dim)      → int
ee_rel_forward(action_abs, state)            → np.ndarray   abs → rel (training)
ee_rel_inverse(action_rel, state)            → np.ndarray   rel → abs (inference/eval)
ee_action_to_poses(action_abs, n_arms)       → list[dict]   for CommandedEEPose
ee_rot6d_to_quat_layout(actions_10)         → np.ndarray   (T,10n) rot6d → (T,8n) quat
ee_quat_layout_names(rot6d_names)            → list[str]    feature name conversion
"""
from __future__ import annotations

import numpy as np

from anvil_shared.rotation import (
    matrices_to_quats,
    matrices_to_rot6d,
    matrix_to_quat,
    quat_to_matrix,
    quats_to_matrices,
    rot6d_to_matrix,
    rot6ds_to_matrices,
)

EE_STATE_DIM_PER_ARM = 8   # [x, y, z, qx, qy, qz, qw, gripper]
EE_ACTION_DIM_PER_ARM = 10  # [x, y, z, r0..r5, gripper]


def n_arms_from_dims(state_dim: int, action_dim: int) -> int:
    """Validate EE layout dimensions and return the number of arms.

    Raises
    ------
    ValueError
        If ``state_dim`` is not a positive multiple of 8, or if
        ``action_dim != 10 * (state_dim // 8)``.
    """
    if state_dim <= 0 or state_dim % EE_STATE_DIM_PER_ARM != 0:
        raise ValueError(
            f"EE observation.state dim {state_dim} is not a positive multiple of "
            f"{EE_STATE_DIM_PER_ARM}; expected 8 * n_arms (bimanual=16, single=8)."
        )
    n = state_dim // EE_STATE_DIM_PER_ARM
    expected_action = EE_ACTION_DIM_PER_ARM * n
    if action_dim != expected_action:
        raise ValueError(
            f"EE action dim {action_dim} != {expected_action} ({EE_ACTION_DIM_PER_ARM} * {n} arms). "
            f"State suggests {n} arm(s)."
        )
    return n


def ee_rel_forward(
    action_abs: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Convert absolute EE actions to SE(3)-relative representation.

    This is the forward transform applied at training time (and used for
    computing stats and GT in evaluation).

    Per arm:
        delta_xyz   = act_xyz - state_xyz
        delta_rot6d = matrices_to_rot6d(R_state.T @ R_action)
        gripper     = act_gripper  (kept absolute)

    Parameters
    ----------
    action_abs:
        Absolute EE actions in rot6d encoding.
        Shape ``(..., 10 * n_arms)``.  A 1-D input ``(10 * n_arms,)`` is
        also accepted.
    state:
        EE observation state.
        Either ``(8 * n_arms,)`` — a **single** reference state broadcast
        over all time steps; or ``(..., 8 * n_arms)`` — **per-sample**
        states with the same leading dims as ``action_abs`` (used for
        dataset-wide stats computation where every frame has its own state).

    Returns
    -------
    np.ndarray
        Relative actions with the same shape as ``action_abs``.
    """
    action_abs = np.asarray(action_abs, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    action_dim = action_abs.shape[-1]
    state_dim = state.shape[-1]
    n_arms = n_arms_from_dims(state_dim, action_dim)

    result = action_abs.copy()
    per_sample_state = state.ndim > 1  # True when state has same batch leading dims

    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        a0 = arm * EE_ACTION_DIM_PER_ARM

        state_xyz = state[..., s0:s0 + 3]    # (3,) or (..., 3)
        state_quat = state[..., s0 + 3:s0 + 7]  # (4,) or (..., 4)

        # xyz: simple broadcast subtraction
        result[..., a0:a0 + 3] = action_abs[..., a0:a0 + 3] - state_xyz

        # rot6d: R_rel = R_state.T @ R_action — vectorised over time/batch dims
        act_r6d = action_abs[..., a0 + 3:a0 + 9]  # (..., 6)
        Rs_action = rot6ds_to_matrices(act_r6d)      # (..., 3, 3)

        if per_sample_state:
            # Per-sample states: quats_to_matrices handles batch dims
            Rs_state = quats_to_matrices(state_quat)          # (..., 3, 3)
            Rs_state_T = Rs_state.swapaxes(-2, -1)            # (..., 3, 3)
            Rs_rel = Rs_state_T @ Rs_action                   # (..., 3, 3)
        else:
            # Single reference state: R_state is (3, 3), broadcasts over batch
            R_state = quat_to_matrix(state_quat)              # (3, 3)
            Rs_rel = R_state.T @ Rs_action                    # (3,3) @ (...,3,3) → (...,3,3)

        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_rel)  # (..., 6)
        # gripper unchanged (already copied via .copy())

    return result


def ee_rel_inverse(
    action_rel: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Restore SE(3)-relative EE actions to absolute representation.

    Inverse of :func:`ee_rel_forward`.  Used at inference time to convert
    model outputs back to absolute EE poses before publishing.

    Per arm:
        abs_xyz     = state_xyz + delta_xyz
        R_abs       = R_state @ rot6ds_to_matrices(delta_rot6d)
        abs_rot6d   = matrices_to_rot6d(R_abs)
        gripper     = delta_gripper  (kept absolute during training)

    Parameters
    ----------
    action_rel:
        Relative EE actions.  Shape ``(..., 10 * n_arms)``.
    state:
        EE observation state used as the restore reference.
        Either ``(8 * n_arms,)`` (single reference, broadcasts) or
        ``(..., 8 * n_arms)`` (per-sample, same leading dims).

    Returns
    -------
    np.ndarray
        Absolute EE actions (rot6d encoded) with the same shape as
        ``action_rel``.
    """
    action_rel = np.asarray(action_rel, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    action_dim = action_rel.shape[-1]
    state_dim = state.shape[-1]
    n_arms = n_arms_from_dims(state_dim, action_dim)

    result = action_rel.copy()
    per_sample_state = state.ndim > 1

    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        a0 = arm * EE_ACTION_DIM_PER_ARM

        state_xyz = state[..., s0:s0 + 3]
        state_quat = state[..., s0 + 3:s0 + 7]

        # xyz: abs = state + delta
        result[..., a0:a0 + 3] = state_xyz + action_rel[..., a0:a0 + 3]

        # rot6d: R_abs = R_state @ R_rel
        rel_r6d = action_rel[..., a0 + 3:a0 + 9]  # (..., 6)
        Rs_rel = rot6ds_to_matrices(rel_r6d)          # (..., 3, 3)

        if per_sample_state:
            Rs_state = quats_to_matrices(state_quat)  # (..., 3, 3)
            Rs_abs = Rs_state @ Rs_rel                # (..., 3, 3)
        else:
            R_state = quat_to_matrix(state_quat)      # (3, 3)
            Rs_abs = R_state @ Rs_rel                 # (3,3) @ (...,3,3) → (...,3,3)

        result[..., a0 + 3:a0 + 9] = matrices_to_rot6d(Rs_abs)
        # gripper unchanged

    return result


def ee_rot6d_to_quat_layout(actions_10: np.ndarray) -> np.ndarray:
    """Convert EE actions from rot6d layout to quaternion layout.

    Parameters
    ----------
    actions_10:
        ``(T, 10 * n_arms)`` absolute EE actions in rot6d encoding.
        Per arm: [x, y, z, r0..r5, gripper].

    Returns
    -------
    np.ndarray
        ``(T, 8 * n_arms)`` with per-arm layout [x, y, z, qx, qy, qz, qw, gripper].
        Uses vectorised ``rot6ds_to_matrices`` → ``matrices_to_quats``.
    """
    arr = np.asarray(actions_10, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"ee_rot6d_to_quat_layout: expected 2D (T, 10*n), got {arr.shape}"
        )
    _T, D = arr.shape
    if D % EE_ACTION_DIM_PER_ARM != 0:
        raise ValueError(
            f"ee_rot6d_to_quat_layout: dim {D} not divisible by {EE_ACTION_DIM_PER_ARM}"
        )
    n_arms = D // EE_ACTION_DIM_PER_ARM

    out_cols: list[np.ndarray] = []
    for arm_idx in range(n_arms):
        a0 = arm_idx * EE_ACTION_DIM_PER_ARM
        xyz  = arr[:, a0:a0 + 3]           # (T, 3)
        r6d  = arr[:, a0 + 3:a0 + 9]       # (T, 6)
        grip = arr[:, a0 + 9:a0 + 10]      # (T, 1)
        R    = rot6ds_to_matrices(r6d)      # (T, 3, 3)
        quat = matrices_to_quats(R)         # (T, 4) [qx, qy, qz, qw]
        out_cols.extend([xyz, quat, grip])

    return np.concatenate(out_cols, axis=1)  # (T, 8*n_arms)


def ee_quat_layout_names(rot6d_names: list[str]) -> list[str]:
    """Convert EE feature names from rot6d layout (10/arm) to quaternion layout (8/arm).

    Example::

        ["right_x","right_y","right_z","right_r0",...,"right_r5","right_gripper"]
        → ["right_x","right_y","right_z","right_qx","right_qy","right_qz","right_qw",
           "right_gripper"]
    """
    n = len(rot6d_names)
    if n % EE_ACTION_DIM_PER_ARM != 0:
        raise ValueError(
            f"ee_quat_layout_names: expected multiple of {EE_ACTION_DIM_PER_ARM} names, got {n}"
        )
    n_arms = n // EE_ACTION_DIM_PER_ARM
    out: list[str] = []
    for arm_idx in range(n_arms):
        prefix = rot6d_names[arm_idx * EE_ACTION_DIM_PER_ARM].rsplit("_", 1)[0]
        out += [
            f"{prefix}_x", f"{prefix}_y", f"{prefix}_z",
            f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz", f"{prefix}_qw",
            f"{prefix}_gripper",
        ]
    return out


def ee_action_to_poses(
    action_abs: np.ndarray,
    n_arms: int | None = None,
) -> list[dict]:
    """Convert a chunk of absolute rot6d EE actions to per-step per-arm pose dicts.

    Replaces the old ``rot6d_chunk_to_quat`` in ``delta_restore.py``.

    Parameters
    ----------
    action_abs:
        Absolute EE actions, shape ``(chunk_size, 10 * n_arms)`` or
        ``(10 * n_arms,)`` for a single step.
    n_arms:
        Number of arms.  Derived from ``action_abs.shape[1] // 10`` when
        ``None`` (default).

    Returns
    -------
    list of dict
        One dict per time step.  Each dict maps ``arm_index (int)`` →
        ``{"pos": np.ndarray (3,), "quat_xyzw": np.ndarray (4,), "gripper": float}``.
    """
    action_abs = np.asarray(action_abs, dtype=np.float64)
    if action_abs.ndim == 1:
        action_abs = action_abs[np.newaxis, :]  # (1, D)

    chunk_size, D = action_abs.shape
    if n_arms is None:
        n_arms = D // EE_ACTION_DIM_PER_ARM

    result: list[dict] = []
    for k in range(chunk_size):
        step: dict = {}
        for arm in range(n_arms):
            a0 = arm * EE_ACTION_DIM_PER_ARM
            pos = action_abs[k, a0:a0 + 3].copy()
            r6d = action_abs[k, a0 + 3:a0 + 9]
            grip = float(action_abs[k, a0 + 9])
            R = rot6d_to_matrix(r6d)
            quat = matrix_to_quat(R)  # [qx, qy, qz, qw]
            step[arm] = {"pos": pos, "quat_xyzw": quat, "gripper": grip}
        result.append(step)
    return result
