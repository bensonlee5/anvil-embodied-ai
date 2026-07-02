"""EE Cartesian runtime utilities for inference.

``resolve_action_type(cfg)``
    Normalises the action_type field from a checkpoint anvil_config dict.
    Accepts the three canonical types: joint_abs, ee_abs, ee_rel.

``read_checkpoint_anvil_config(model_path)``
    Resolves a checkpoint path (bare / pretrained_model/ / HF-cache snapshot)
    and reads its anvil_config.json, if present.

``ee_rel_restore_chunk(chunk_np, obs_t)``
    Restores EE relative actions (ee_rel) to absolute EE poses.
    Thin wrapper around ``anvil_shared.ee_transform.ee_rel_inverse``.

``ee_poses_from_chunk(chunk_np, n_arms)``
    Converts a chunk of absolute rot6d EE actions to per-step per-arm
    pose dicts suitable for building ``CommandedEEPose`` messages.
    Thin wrapper around ``anvil_shared.ee_transform.ee_action_to_poses``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _ensure_anvil_shared() -> None:
    """Add packages/anvil_shared/src to sys.path so ee_transform helpers are importable.

    Called lazily inside EE functions so the import overhead is paid only when
    those functions are actually used.
    """
    import os
    import sys

    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    _shared_src = os.path.join(_repo_root, "packages", "anvil_shared", "src")
    if _shared_src not in sys.path:
        sys.path.insert(0, _shared_src)


def resolve_action_type(cfg: dict) -> str:
    """Return the normalised action_type string from an anvil_config dict.

    Accepts the three canonical types: "joint_abs", "ee_abs", "ee_rel".
    Old checkpoints that pre-date the three-type scheme will have
    ``action_type="joint_abs"`` (or absent, defaulting to "joint_abs").
    """
    return cfg.get("action_type", "joint_abs")


def read_checkpoint_anvil_config(model_path: str) -> dict:
    """Resolve *model_path* to a checkpoint dir and read its anvil_config.json.

    Mirrors the path resolution in ``inference_node._read_checkpoint_metadata``
    (bare checkpoint dir / ``pretrained_model/`` subdir / HF-cache
    ``snapshots/<hash>/`` layout) so callers get the same answer regardless
    of which convention *model_path* uses.

    Returns ``{}`` if *model_path* is falsy or no anvil_config.json is found —
    callers should fall back to their own default (e.g. a ROS param) in that case.
    """
    if not model_path:
        return {}

    checkpoint = Path(model_path)

    pretrained = checkpoint / "pretrained_model"
    if pretrained.exists() and (pretrained / "config.json").exists():
        checkpoint = pretrained

    if not (checkpoint / "config.json").exists():
        snapshots = checkpoint / "snapshots"
        if snapshots.is_dir():
            for snap in sorted(snapshots.iterdir(), reverse=True):
                if (snap / "config.json").exists():
                    checkpoint = snap
                    break

    anvil_path = checkpoint / "anvil_config.json"
    if not anvil_path.exists():
        return {}
    return json.loads(anvil_path.read_text())


def ee_rel_restore_chunk(
    chunk_np: np.ndarray,
    obs_t: np.ndarray,
) -> np.ndarray:
    """Restore EE relative actions (ee_rel) to absolute EE poses.

    Inverse of the SE(3) forward transform applied at training time.

    Per arm (10 action dims, 8 state dims):
        abs_xyz   = obs_xyz + delta_xyz
        R_abs     = R_state @ rot6ds_to_matrices(delta_rot6d)
        abs_rot6d = matrices_to_rot6d(R_abs)
        gripper   = delta_gripper  (kept absolute during training)

    Args:
        chunk_np: (chunk_size, 10*n_arms) relative-space model output.
        obs_t:    (8*n_arms,) or (n_obs_steps, 8*n_arms) observation state
                  at chunk generation time. If 2-D, the last row is used.

    Returns:
        (chunk_size, 10*n_arms) absolute EE actions (rot6d encoded).
    """
    try:
        _ensure_anvil_shared()
        from anvil_shared.ee_transform import ee_rel_inverse
    except ImportError as e:
        raise ImportError(
            "ee_rel_restore_chunk requires anvil_shared.ee_transform. "
            "Ensure packages/anvil_shared is on PYTHONPATH."
        ) from e

    chunk_np = np.asarray(chunk_np, dtype=np.float64)
    obs_t = np.asarray(obs_t, dtype=np.float64)

    if chunk_np.ndim == 1:
        chunk_np = chunk_np[np.newaxis, :]

    # Accept stacked multi-step obs (e.g. shape (n_obs_steps, 8*n_arms)); use last row.
    if obs_t.ndim > 1:
        obs_t = obs_t[-1]

    return ee_rel_inverse(chunk_np, obs_t)


def ee_poses_from_chunk(
    chunk_np: np.ndarray,
    n_arms: int | None = None,
) -> list[dict]:
    """Convert a chunk of absolute rot6d EE actions to per-step per-arm pose dicts.

    Args:
        chunk_np: (chunk_size, 10*n_arms) absolute rot6d actions,
                  or (10*n_arms,) for a single step.
        n_arms:   Number of arms. Derived from chunk_np.shape[1] // 10 when None.

    Returns:
        List of chunk_size dicts. Each dict maps arm_index (int) to:
          {"pos": np.ndarray (3,), "quat_xyzw": np.ndarray (4,), "gripper": float}
        where quat_xyzw = [qx, qy, qz, qw] (ROS convention).
    """
    try:
        _ensure_anvil_shared()
        from anvil_shared.ee_transform import ee_action_to_poses
    except ImportError as e:
        raise ImportError(
            "ee_poses_from_chunk requires anvil_shared.ee_transform. "
            "Ensure packages/anvil_shared is on PYTHONPATH."
        ) from e

    chunk_np = np.asarray(chunk_np, dtype=np.float64)
    return ee_action_to_poses(chunk_np, n_arms=n_arms)
