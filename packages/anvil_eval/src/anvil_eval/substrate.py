"""Unified evaluation substrate — the source-of-truth data captured per replay.

A single replay pass captures, at every inference anchor, the full predicted action
chunk plus the ground truth it should match. Both diagnostic modes are pure functions
of this substrate:

  - ``trajectory`` : the executed prefix (``horizon_offset < executed_len``) of each
    chunk, stitched by ``target_frame`` — reproduces the legacy per-frame view.
  - ``horizon``    : error aggregated by ``horizon_offset`` across all anchors.

The substrate serializes to a long-form CSV (one row per
``(episode, anchor_frame, horizon_offset, joint)``) so any plot can be regenerated with
external tools, plus a small JSON of run metadata.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AnchorChunk:
    """One inference's full predicted chunk and the GT it is compared against.

    All arrays are trimmed to the frames that actually exist in the episode, so
    ``abs_pred`` / ``abs_gt`` share the same leading length ``H_avail`` which may be
    shorter than the model's full horizon near the end of an episode.
    """

    anchor_frame: int            # frame index where this chunk was predicted
    executed_len: int            # native n_action_steps — how many offsets the robot runs
    abs_pred: np.ndarray         # (H_avail, D) absolute predicted (post delta-restore)
    abs_gt: np.ndarray           # (H_avail, D) absolute ground truth at target frames
    obs_ref: np.ndarray | None = None    # (D,) observation.state at the anchor
    raw_pred: np.ndarray | None = None   # (H_avail, D) raw model output (delta space)


@dataclass
class EpisodeSubstrate:
    """All captured anchors for one episode, plus optional per-frame phase labels."""

    episode_idx: int
    split_label: str
    joint_names: list[str]
    anchors: list[AnchorChunk] = field(default_factory=list)
    # phase label per frame index, per arm; filled by the phase labeler (later step).
    phase_left: dict[int, str] = field(default_factory=dict)
    phase_right: dict[int, str] = field(default_factory=dict)


# ── Long-form records ────────────────────────────────────────────────────────

# Column order for substrate.csv — stable contract for downstream tooling.
SUBSTRATE_COLUMNS = [
    "episode_idx",
    "split_label",
    "anchor_frame",
    "horizon_offset",
    "target_frame",
    "executed",
    "joint",
    "predicted",
    "ground_truth",
    "error",
    "obs_state",
    "phase_left",
    "phase_right",
]


def iter_records(ep: EpisodeSubstrate):
    """Yield one dict row per (anchor, horizon_offset, joint) for an episode."""
    for chunk in ep.anchors:
        h_avail = chunk.abs_pred.shape[0]
        for h in range(h_avail):
            target_frame = chunk.anchor_frame + h
            phase_l = ep.phase_left.get(target_frame, "")
            phase_r = ep.phase_right.get(target_frame, "")
            for j, joint in enumerate(ep.joint_names):
                pred = float(chunk.abs_pred[h, j])
                gt = float(chunk.abs_gt[h, j])
                obs = (
                    float(chunk.obs_ref[j])
                    if chunk.obs_ref is not None and j < len(chunk.obs_ref)
                    else ""
                )
                yield {
                    "episode_idx": ep.episode_idx,
                    "split_label": ep.split_label,
                    "anchor_frame": chunk.anchor_frame,
                    "horizon_offset": h,
                    "target_frame": target_frame,
                    "executed": int(h < chunk.executed_len),
                    "joint": joint,
                    "predicted": pred,
                    "ground_truth": gt,
                    "error": pred - gt,
                    "obs_state": obs,
                    "phase_left": phase_l,
                    "phase_right": phase_r,
                }


def write_substrate_csv(episodes: list[EpisodeSubstrate], path: Path) -> None:
    """Write the full long-form substrate for all episodes to one CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBSTRATE_COLUMNS)
        writer.writeheader()
        for ep in episodes:
            for row in iter_records(ep):
                writer.writerow(row)


def write_run_meta_json(
    episodes: list[EpisodeSubstrate],
    path: Path,
    extra: dict | None = None,
) -> None:
    """Write run metadata describing what the substrate contains."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "episodes": [
            {
                "episode_idx": ep.episode_idx,
                "split_label": ep.split_label,
                "num_anchors": len(ep.anchors),
                "executed_len": ep.anchors[0].executed_len if ep.anchors else None,
                "max_horizon": max((c.abs_pred.shape[0] for c in ep.anchors), default=0),
            }
            for ep in episodes
        ],
        "joint_names": episodes[0].joint_names if episodes else [],
        "columns": SUBSTRATE_COLUMNS,
    }
    if extra:
        meta.update(extra)
    path.write_text(json.dumps(meta, indent=2))
