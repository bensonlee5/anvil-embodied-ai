"""Shared delta action restore utilities.

restore_delta_chunk(chunk_np, obs_t, action_type, exclude_indices)
    Converts a chunk of raw model delta outputs to absolute joint positions.
    action_type: "delta_obs_t" | "delta_sequential"

restore_ee_delta_chunk(chunk_np, obs_t)
    Restores EE Cartesian delta actions (ee_delta) to absolute EE poses.
    chunk_np : (chunk_size, 10*n_arms) in delta space
                 per arm: [delta_xyz(3), delta_rot6d(6), gripper_abs(1)]
    obs_t    : (8*n_arms,) in state space
                 per arm: [xyz(3), quat_xyzw(4), gripper(1)]
    returns  : (chunk_size, 10*n_arms) absolute EE actions (rot6d encoded)

For ee_absolute, the model output is already absolute rot6d — no restore needed.
Call rot6d_chunk_to_quat(chunk_np) to convert for robot publishing.

resolve_action_type(cfg) normalises the action_type field from a checkpoint anvil_config dict,
applying backward-compat mapping from the old use_delta_actions boolean.
"""
from __future__ import annotations

import numpy as np


def resolve_action_type(cfg: dict) -> str:
    """Return normalised action_type string from an anvil_config dict.

    Handles backward compat: old checkpoints without action_type but with
    use_delta_actions=True are mapped to "delta_obs_t".
    """
    action_type = cfg.get("action_type", "absolute")
    if action_type == "absolute" and cfg.get("use_delta_actions", False):
        return "delta_obs_t"
    return action_type


def restore_delta_chunk(
    chunk_np: np.ndarray,
    obs_t: np.ndarray,
    action_type: str,
    exclude_indices: set[int] | list[int],
) -> np.ndarray:
    """Restore a chunk of delta actions to absolute joint positions.

    Args:
        chunk_np: Raw model output, shape (chunk_size, D).
        obs_t: Observation state at chunk generation time, shape (D,).
        action_type: "delta_obs_t" or "delta_sequential".
        exclude_indices: Joint indices kept in absolute space during training.
            Pass a pre-computed set to avoid repeated conversion.

    Returns:
        Absolute actions, shape (chunk_size, D).
    """
    chunk_np = np.asarray(chunk_np, dtype=np.float64)
    obs_t = np.asarray(obs_t, dtype=np.float64)

    if chunk_np.ndim == 1:
        chunk_np = chunk_np[np.newaxis, :]

    chunk_size, D = chunk_np.shape
    excl = exclude_indices if isinstance(exclude_indices, set) else set(exclude_indices)
    non_excl = [i for i in range(D) if i not in excl]

    abs_chunk = chunk_np.copy()  # excluded joints: keep raw value (already absolute)

    if action_type == "delta_obs_t":
        # All non-excluded joints: absolute = obs_t + delta (same reference for all k)
        if non_excl:
            abs_chunk[:, non_excl] = obs_t[non_excl] + chunk_np[:, non_excl]

    elif action_type == "delta_sequential":
        # Non-excluded joints: cumulative sum along time axis, then + obs_t
        if non_excl:
            abs_chunk[:, non_excl] = obs_t[non_excl] + np.cumsum(chunk_np[:, non_excl], axis=0)

    return abs_chunk


def restore_ee_delta_chunk(
    chunk_np: np.ndarray,
    obs_t: np.ndarray,
) -> np.ndarray:
    """Restore EE Cartesian delta actions to absolute EE poses.

    Inverse of ``EEDeltaTransform`` applied at training time.

    Per arm (10 action dims, 8 state dims):
      abs_xyz   = obs_xyz + delta_xyz
      R_abs     = R_state @ rot6d_to_matrix(delta_rot6d)
      abs_r6d   = matrix_to_rot6d(R_abs)
      gripper   = delta_gripper  (kept absolute during training)

    Args:
        chunk_np: (chunk_size, 10*n_arms) delta-space model output.
        obs_t:    (8*n_arms,) observation state at chunk generation time.

    Returns:
        (chunk_size, 10*n_arms) absolute EE actions (rot6d encoded, same layout as action).
    """
    try:
        import sys
        import os
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
        _shared_src = os.path.join(_repo_root, "packages", "anvil_shared", "src")
        if _shared_src not in sys.path:
            sys.path.insert(0, _shared_src)
        from anvil_shared.rotation import matrix_to_rot6d, quat_to_matrix, rot6d_to_matrix
    except ImportError as e:
        raise ImportError(
            "restore_ee_delta_chunk requires anvil_shared.rotation. "
            "Ensure packages/anvil_shared is on PYTHONPATH."
        ) from e

    chunk_np = np.asarray(chunk_np, dtype=np.float64)
    obs_t    = np.asarray(obs_t, dtype=np.float64)

    if chunk_np.ndim == 1:
        chunk_np = chunk_np[np.newaxis, :]

    chunk_size = chunk_np.shape[0]
    n_arms = obs_t.shape[-1] // 8

    abs_chunk = chunk_np.copy()

    for arm in range(n_arms):
        s0, a0 = arm * 8, arm * 10
        obs_xyz  = obs_t[s0:s0+3]
        obs_quat = obs_t[s0+3:s0+7]  # [qx,qy,qz,qw]
        R_state  = quat_to_matrix(obs_quat)

        for k in range(chunk_size):
            delta_xyz  = chunk_np[k, a0:a0+3]
            delta_r6d  = chunk_np[k, a0+3:a0+9]
            grip_abs   = chunk_np[k, a0+9]

            abs_xyz = obs_xyz + delta_xyz
            R_rel   = rot6d_to_matrix(delta_r6d)
            R_abs   = R_state @ R_rel
            abs_r6d = matrix_to_rot6d(R_abs)

            abs_chunk[k, a0:a0+3]   = abs_xyz
            abs_chunk[k, a0+3:a0+9] = abs_r6d
            abs_chunk[k, a0+9]      = grip_abs

    return abs_chunk


def rot6d_chunk_to_quat(chunk_np: np.ndarray, n_arms: int = 1) -> list[dict]:
    """Convert a chunk of absolute rot6d EE actions to per-step per-arm pose dicts.

    Returns a list of chunk_size dicts, each with key ``{arm_idx}``:
      {"pos": (3,), "quat": (4,) [x,y,z,w], "gripper": float}

    Useful for publishing ``CommandedEEPose`` in the inference node.
    """
    try:
        import sys, os
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
        _shared_src = os.path.join(_repo_root, "packages", "anvil_shared", "src")
        if _shared_src not in sys.path:
            sys.path.insert(0, _shared_src)
        from anvil_shared.rotation import matrix_to_quat, rot6d_to_matrix
    except ImportError as e:
        raise ImportError("rot6d_chunk_to_quat requires anvil_shared.rotation.") from e

    chunk_np = np.asarray(chunk_np, dtype=np.float64)
    if chunk_np.ndim == 1:
        chunk_np = chunk_np[np.newaxis, :]

    result = []
    for k in range(chunk_np.shape[0]):
        step: dict = {}
        for arm in range(n_arms):
            a0 = arm * 10
            pos  = chunk_np[k, a0:a0+3]
            r6d  = chunk_np[k, a0+3:a0+9]
            grip = float(chunk_np[k, a0+9])
            R    = rot6d_to_matrix(r6d)
            quat = matrix_to_quat(R)  # [x,y,z,w]
            step[arm] = {"pos": pos, "quat": quat, "gripper": grip}
        result.append(step)
    return result
