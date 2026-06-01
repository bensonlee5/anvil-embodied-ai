"""Episode-level and summary metrics computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EEMetrics:
    """EE Cartesian-specific metrics for a single evaluated episode.

    All values are per-arm averages over the episode unless noted.
    Pass/fail thresholds: position < 0.02 m, orientation < 5° (0.0873 rad).
    """

    # Per-arm position error (metres), averaged over frames
    position_error_m: dict[str, float] = field(default_factory=dict)
    # Per-arm orientation geodesic error (radians), averaged over frames
    orientation_error_rad: dict[str, float] = field(default_factory=dict)
    # Per-arm gripper absolute error (metres), averaged over frames
    gripper_error_m: dict[str, float] = field(default_factory=dict)
    # Per-arm, per-step position error curves: {arm: (T,)}
    position_error_per_step: dict[str, list[float]] = field(default_factory=dict)
    # Per-arm, per-step orientation error curves: {arm: (T,)}
    orientation_error_per_step: dict[str, list[float]] = field(default_factory=dict)

    @property
    def position_pass(self, threshold: float = 0.02) -> bool:
        return all(v < threshold for v in self.position_error_m.values())

    @property
    def orientation_pass(self, threshold: float = 0.0873) -> bool:  # 5 degrees
        return all(v < threshold for v in self.orientation_error_rad.values())


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluated episode."""

    episode_idx: int
    split_label: str
    num_frames: int
    mse: float
    mae: float
    rmse: float
    max_abs_error: float
    max_abs_error_joint: str
    per_joint_mse: dict[str, float]
    per_joint_mae: dict[str, float]
    per_joint_rmse: dict[str, float]
    cosine_similarity: float
    pred_smoothness_mean: float
    pred_smoothness_std: float
    gt_smoothness_mean: float
    gt_smoothness_std: float
    # EE Cartesian metrics — populated only for ee_absolute / ee_delta action types
    ee: Optional[EEMetrics] = None


def compute_ee_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    action_names: list[str],
) -> EEMetrics:
    """Compute EE Cartesian metrics: position (m), orientation geodesic (rad), gripper (m).

    Expects action layout per arm: [x, y, z, r0..r5, gripper] (10 dims).
    Arms are inferred from action_names by the prefix before the first underscore.

    Args:
        predicted:     (T, 10*n_arms) predicted actions
        ground_truth:  (T, 10*n_arms) ground-truth actions
        action_names:  list of 10*n_arms feature names, e.g. ["left_x", ..., "right_gripper"]
    """
    from anvil_shared.rotation import rot6d_to_matrix

    # Collect arm prefix ordering from action_names
    seen: list[str] = []
    for name in action_names:
        prefix = name.rsplit("_", 1)[0] if "_" in name else name
        if prefix not in seen:
            seen.append(prefix)
    # Each arm has 10 dims; detect by grouping consecutive same-prefix names
    arm_names = [p for p in seen if action_names.count(p + "_x") == 1 or True]
    # Simpler: n_arms = total_dims // 10
    n_arms = predicted.shape[1] // 10
    # Derive arm labels from names (first name per 10-dim block)
    arm_labels = []
    for arm_idx in range(n_arms):
        a0 = arm_idx * 10
        label = action_names[a0].rsplit("_", 1)[0] if a0 < len(action_names) else f"arm{arm_idx}"
        arm_labels.append(label)

    pos_error:     dict[str, float] = {}
    ori_error:     dict[str, float] = {}
    grip_error:    dict[str, float] = {}
    pos_per_step:  dict[str, list[float]] = {}
    ori_per_step:  dict[str, list[float]] = {}

    for arm_idx, label in enumerate(arm_labels):
        a0 = arm_idx * 10
        pred_xyz  = predicted[:, a0:a0+3]
        gt_xyz    = ground_truth[:, a0:a0+3]
        pred_r6d  = predicted[:, a0+3:a0+9]
        gt_r6d    = ground_truth[:, a0+3:a0+9]
        pred_grip = predicted[:, a0+9]
        gt_grip   = ground_truth[:, a0+9]

        # Position error: per-frame Euclidean distance
        pos_err_steps = np.linalg.norm(pred_xyz - gt_xyz, axis=1)  # (T,)
        pos_error[label] = float(np.mean(pos_err_steps))
        pos_per_step[label] = pos_err_steps.tolist()

        # Orientation error: geodesic angle arccos((trace(R_pred.T @ R_gt) - 1) / 2)
        ori_err_steps = np.zeros(predicted.shape[0])
        for t in range(predicted.shape[0]):
            try:
                R_pred = rot6d_to_matrix(pred_r6d[t])
                R_gt   = rot6d_to_matrix(gt_r6d[t])
                trace  = np.clip((np.trace(R_pred.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
                ori_err_steps[t] = float(np.arccos(trace))
            except Exception:
                ori_err_steps[t] = 0.0
        ori_error[label] = float(np.mean(ori_err_steps))
        ori_per_step[label] = ori_err_steps.tolist()

        grip_error[label] = float(np.mean(np.abs(pred_grip - gt_grip)))

    return EEMetrics(
        position_error_m=pos_error,
        orientation_error_rad=ori_error,
        gripper_error_m=grip_error,
        position_error_per_step=pos_per_step,
        orientation_error_per_step=ori_per_step,
    )


def compute_episode_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    joint_names: list[str],
    episode_idx: int,
    split_label: str,
    action_type: str = "absolute",
) -> EpisodeMetrics:
    """Compute all metrics for a single episode.

    Args:
        predicted: (T, D) predicted actions
        ground_truth: (T, D) ground-truth actions
        joint_names: list of D joint names
        episode_idx: episode index
        split_label: split label (train/val/test/manual)
    """
    error = predicted - ground_truth  # (T, D)

    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))

    abs_error = np.abs(error)
    max_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
    max_abs_error = float(abs_error[max_idx])
    max_abs_error_joint = joint_names[max_idx[1]] if len(joint_names) > max_idx[1] else f"joint_{max_idx[1]}"

    per_joint_mse = {
        name: float(np.mean(error[:, j] ** 2)) for j, name in enumerate(joint_names)
    }
    per_joint_mae = {
        name: float(np.mean(np.abs(error[:, j]))) for j, name in enumerate(joint_names)
    }
    per_joint_rmse = {
        name: float(np.sqrt(per_joint_mse[name])) for name in joint_names
    }

    # Cosine similarity (per-frame, then average)
    dot = np.sum(predicted * ground_truth, axis=1)
    norm_p = np.linalg.norm(predicted, axis=1)
    norm_g = np.linalg.norm(ground_truth, axis=1)
    denom = norm_p * norm_g
    cos_sim = np.where(denom > 1e-8, dot / denom, 0.0)
    cosine_similarity = float(np.mean(cos_sim))

    # Smoothness: L2 norm of consecutive action deltas
    if predicted.shape[0] > 1:
        pred_deltas = np.linalg.norm(np.diff(predicted, axis=0), axis=1)
        gt_deltas = np.linalg.norm(np.diff(ground_truth, axis=0), axis=1)
        pred_smooth_mean = float(np.mean(pred_deltas))
        pred_smooth_std = float(np.std(pred_deltas))
        gt_smooth_mean = float(np.mean(gt_deltas))
        gt_smooth_std = float(np.std(gt_deltas))
    else:
        pred_smooth_mean = pred_smooth_std = gt_smooth_mean = gt_smooth_std = 0.0

    ee_metrics: Optional[EEMetrics] = None
    if action_type in ("ee_absolute", "ee_delta") and predicted.shape[1] % 10 == 0:
        ee_metrics = compute_ee_metrics(predicted, ground_truth, joint_names)

    return EpisodeMetrics(
        episode_idx=episode_idx,
        split_label=split_label,
        num_frames=predicted.shape[0],
        mse=mse,
        mae=mae,
        rmse=rmse,
        max_abs_error=max_abs_error,
        max_abs_error_joint=max_abs_error_joint,
        per_joint_mse=per_joint_mse,
        per_joint_mae=per_joint_mae,
        per_joint_rmse=per_joint_rmse,
        cosine_similarity=cosine_similarity,
        pred_smoothness_mean=pred_smooth_mean,
        pred_smoothness_std=pred_smooth_std,
        gt_smoothness_mean=gt_smooth_mean,
        gt_smoothness_std=gt_smooth_std,
        ee=ee_metrics,
    )


def compute_summary_metrics(episode_metrics: list[EpisodeMetrics]) -> dict:
    """Aggregate metrics across episodes, grouped by split."""
    by_split: dict[str, list[EpisodeMetrics]] = {}
    for m in episode_metrics:
        by_split.setdefault(m.split_label, []).append(m)

    summary: dict = {}
    for split_name, metrics_list in by_split.items():
        n = len(metrics_list)
        mses = [m.mse for m in metrics_list]
        maes = [m.mae for m in metrics_list]
        rmses = [m.rmse for m in metrics_list]
        cos_sims = [m.cosine_similarity for m in metrics_list]

        # Per-joint aggregation
        joint_names = list(metrics_list[0].per_joint_mae.keys())
        per_joint_mae_mean = {}
        per_joint_mae_std = {}
        for jn in joint_names:
            vals = [m.per_joint_mae[jn] for m in metrics_list]
            per_joint_mae_mean[jn] = float(np.mean(vals))
            per_joint_mae_std[jn] = float(np.std(vals))

        split_summary: dict = {
            "num_episodes": n,
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "cosine_similarity_mean": float(np.mean(cos_sims)),
            "cosine_similarity_std": float(np.std(cos_sims)),
            "per_joint_mae_mean": per_joint_mae_mean,
            "per_joint_mae_std": per_joint_mae_std,
        }

        # EE Cartesian metrics (only when present)
        ee_eps = [m for m in metrics_list if m.ee is not None]
        if ee_eps:
            arm_names = list(ee_eps[0].ee.position_error_m.keys())
            pos_thresh, ori_thresh = 0.02, 0.0873  # 2 cm, 5 degrees
            ee_summary: dict = {}
            for arm in arm_names:
                pos_vals  = [m.ee.position_error_m.get(arm, float("nan")) for m in ee_eps]
                ori_vals  = [m.ee.orientation_error_rad.get(arm, float("nan")) for m in ee_eps]
                grip_vals = [m.ee.gripper_error_m.get(arm, float("nan")) for m in ee_eps]
                ee_summary[arm] = {
                    "position_error_m_mean":      float(np.nanmean(pos_vals)),
                    "position_error_m_std":       float(np.nanstd(pos_vals)),
                    "orientation_error_deg_mean": float(np.degrees(np.nanmean(ori_vals))),
                    "orientation_error_deg_std":  float(np.degrees(np.nanstd(ori_vals))),
                    "gripper_error_m_mean":       float(np.nanmean(grip_vals)),
                    "pass_position":  bool(np.nanmean(pos_vals) < pos_thresh),
                    "pass_orientation": bool(np.degrees(np.nanmean(ori_vals)) < 5.0),
                }
            split_summary["ee"] = ee_summary

        summary[split_name] = split_summary

    return summary
