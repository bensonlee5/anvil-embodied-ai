"""Horizon analysis — error as a function of prediction horizon offset.

Pure functions over the captured :class:`~anvil_eval.substrate.EpisodeSubstrate` list.
For each horizon offset ``h`` (steps ahead of the inference anchor) we aggregate the
absolute error across every anchor and episode in a split, yielding the
error-vs-horizon curve that shows how far a single prediction stays trustworthy.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .substrate import EpisodeSubstrate


def aggregate_horizon(episodes: list[EpisodeSubstrate]) -> dict:
    """Aggregate absolute error by horizon offset, grouped by split.

    Returns ``{split: {offsets, mae_mean, mae_std, count, executed_len,
    per_joint_mae_mean}}`` where each list is indexed by horizon offset.
    """
    by_split: dict[str, list[EpisodeSubstrate]] = {}
    for ep in episodes:
        by_split.setdefault(ep.split_label, []).append(ep)

    result: dict = {}
    for split, eps in by_split.items():
        joint_names = eps[0].joint_names
        per_offset: dict[int, list[np.ndarray]] = {}  # h -> list of (D,) abs-error vectors
        executed_len = None
        for ep in eps:
            for ch in ep.anchors:
                if executed_len is None:
                    executed_len = ch.executed_len
                ae = np.abs(ch.abs_pred - ch.abs_gt)  # (H, D)
                for h in range(ae.shape[0]):
                    per_offset.setdefault(h, []).append(ae[h])

        offsets = sorted(per_offset)
        mae_mean, mae_std, count = [], [], []
        per_joint_mae_mean: dict[str, list[float]] = {jn: [] for jn in joint_names}
        for h in offsets:
            stack = np.stack(per_offset[h])           # (N, D)
            mae_mean.append(float(stack.mean()))
            mae_std.append(float(stack.mean(axis=1).std()))  # spread across anchors
            count.append(int(stack.shape[0]))
            joint_means = stack.mean(axis=0)          # (D,)
            for j, jn in enumerate(joint_names):
                per_joint_mae_mean[jn].append(float(joint_means[j]))

        result[split] = {
            "offsets": offsets,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "count": count,
            "executed_len": executed_len,
            "per_joint_mae_mean": per_joint_mae_mean,
        }
    return result


def write_horizon_csv(agg: dict, path: Path) -> None:
    """Write per-(split, offset) horizon stats, including per-joint MAE columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joint_names: list[str] = []
    for split_data in agg.values():
        joint_names = list(split_data["per_joint_mae_mean"].keys())
        break

    fieldnames = ["split", "horizon_offset", "executed", "count", "mae_mean", "mae_std"]
    fieldnames += [f"mae_{jn}" for jn in joint_names]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split, d in agg.items():
            exec_len = d["executed_len"] or 0
            for i, h in enumerate(d["offsets"]):
                row = {
                    "split": split,
                    "horizon_offset": h,
                    "executed": int(h < exec_len),
                    "count": d["count"][i],
                    "mae_mean": d["mae_mean"][i],
                    "mae_std": d["mae_std"][i],
                }
                for jn in joint_names:
                    row[f"mae_{jn}"] = d["per_joint_mae_mean"][jn][i]
                writer.writerow(row)
