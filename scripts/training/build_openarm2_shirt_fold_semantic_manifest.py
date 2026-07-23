#!/usr/bin/env python3
"""Build the five-stage OpenARM2 shirt-fold semantic segmentation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = (
    ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json"
)
DEFAULT_DATA = ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned/data/chunk-000/file-000.parquet"
DEFAULT_OUTPUT = ROOT / "configs/training/semantic_manifests/openarm2_shirt_fold_5stage_v1.json"

STAGE_ORDER = (
    "side_one",
    "recenter_pull",
    "side_two",
    "strip_refinement",
    "bottom_to_top",
)
OPTIONAL_STAGES = ("recenter_pull", "strip_refinement")
GRIPPER_INDICES = (7, 15)
GRIPPER_CLOSED_THRESHOLD = 0.02
MIN_BIMANUAL_RUN_FRAMES = 5
BRIDGE_OPEN_GAP_FRAMES = 6
RECENTER_FIRST_RUN_START_FRACTION_MAX = 0.25
RECENTER_FIRST_GROUP_END_FRACTION_MAX = 0.55
RECENTER_FOLD_RUN_START_FRACTION_MIN = 0.45
REFINEMENT_FOLD_CYCLE_FRACTION = 0.75


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _bridge_short_open_gaps(closed: np.ndarray) -> np.ndarray:
    bridged = closed.copy()
    cursor = 0
    while cursor < len(bridged):
        if bridged[cursor]:
            cursor += 1
            continue
        end = cursor
        while end < len(bridged) and not bridged[end]:
            end += 1
        if cursor > 0 and end < len(bridged) and end - cursor <= BRIDGE_OPEN_GAP_FRAMES:
            bridged[cursor:end] = True
        cursor = end
    return bridged


def _closed_runs(actions: np.ndarray) -> list[tuple[int, int]]:
    bimanual_closed = np.logical_and(
        actions[:, GRIPPER_INDICES[0]] < GRIPPER_CLOSED_THRESHOLD,
        actions[:, GRIPPER_INDICES[1]] < GRIPPER_CLOSED_THRESHOLD,
    )
    bridged = _bridge_short_open_gaps(bimanual_closed)
    padded = np.concatenate(([False], bridged, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end - start >= MIN_BIMANUAL_RUN_FRAMES
    ]


def _semantic_boundaries(
    actions: np.ndarray,
    *,
    old_start: int,
    old_end: int,
    smoothing_label: str,
) -> dict[str, Any]:
    stage_actions = actions[old_start:old_end]
    runs = _closed_runs(stage_actions)
    if not runs:
        raise ValueError(f"old side_two interval {old_start}:{old_end} has no bimanual grasp")

    interval_length = old_end - old_start
    gaps = [runs[index + 1][0] - runs[index][1] for index in range(len(runs) - 1)]
    largest_gap_index = int(np.argmax(gaps)) if gaps else None
    recenter_present = False
    if largest_gap_index is not None:
        early_start = runs[0][0] / interval_length
        early_end = runs[largest_gap_index][1] / interval_length
        fold_start = runs[largest_gap_index + 1][0] / interval_length
        recenter_present = (
            early_start < RECENTER_FIRST_RUN_START_FRACTION_MAX
            and early_end < RECENTER_FIRST_GROUP_END_FRACTION_MAX
            and fold_start > RECENTER_FOLD_RUN_START_FRACTION_MIN
        )

    if recenter_present:
        assert largest_gap_index is not None
        recenter_end_local = round(
            (runs[largest_gap_index][1] + runs[largest_gap_index + 1][0]) / 2
        )
        fold_runs = runs[largest_gap_index + 1 :]
        recenter_gap = gaps[largest_gap_index]
        recenter_confidence = "high" if recenter_gap >= 45 else "medium"
        recenter_source = "largest_stable_dual-release_gap"
    else:
        recenter_end_local = 0
        fold_runs = runs
        recenter_confidence = "medium"
        recenter_source = "no_separated_early_bimanual_cycle"

    fold_cycle_start = fold_runs[0][0]
    fold_cycle_end = fold_runs[-1][1]
    refinement_present = smoothing_label != "absent"
    if refinement_present:
        refinement_start_local = round(
            fold_cycle_start + REFINEMENT_FOLD_CYCLE_FRACTION * (fold_cycle_end - fold_cycle_start)
        )
        refinement_start_local = max(recenter_end_local + 1, refinement_start_local)
        refinement_start_local = min(interval_length - 1, refinement_start_local)
    else:
        refinement_start_local = interval_length

    return {
        "recenter_present": recenter_present,
        "recenter_end": old_start + recenter_end_local,
        "recenter_confidence": recenter_confidence,
        "recenter_source": recenter_source,
        "recenter_gap_frames": recenter_gap if recenter_present else None,
        "refinement_present": refinement_present,
        "refinement_start": old_start + refinement_start_local,
        "refinement_confidence": "low",
        "refinement_source": "kinematic_proposal_at_75pct_of_final_bimanual_cycle",
        "bimanual_runs": [
            {"start_frame": old_start + start, "end_frame": old_start + end} for start, end in runs
        ],
    }


def _stage(
    name: str,
    start: int,
    end: int,
    *,
    present: bool,
    confidence: str,
    source: str,
) -> dict[str, Any]:
    if present != (end > start):
        raise ValueError(f"stage {name} presence does not match {start}:{end}")
    return {
        "name": name,
        "present": present,
        "start_frame": start,
        "end_frame": end,
        "segmentation_confidence": confidence,
        "boundary_source": source,
    }


def build_manifest(source_manifest_path: Path, data_path: Path) -> dict[str, Any]:
    source = json.loads(source_manifest_path.read_text())
    if source["stage_order"] != ["side_one", "side_two", "bottom_to_top"]:
        raise ValueError("source manifest must use the frozen three-stage order")

    table = pq.read_table(data_path, columns=["action", "episode_index", "frame_index"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    episode_indices = np.asarray(table["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(table["frame_index"], dtype=np.int64)

    episodes: list[dict[str, Any]] = []
    for episode in source["episodes"]:
        episode_index = int(episode["episode_index"])
        frame_count = int(episode["frame_count"])
        mask = episode_indices == episode_index
        episode_actions = actions[mask][np.argsort(frame_indices[mask])]
        if len(episode_actions) != frame_count:
            raise ValueError(
                f"episode {episode_index} has {len(episode_actions)} rows, expected {frame_count}"
            )

        side_one, old_side_two, bottom_to_top = episode["stages"]
        proposal = _semantic_boundaries(
            episode_actions,
            old_start=int(old_side_two["start_frame"]),
            old_end=int(old_side_two["end_frame"]),
            smoothing_label=str(episode["smoothing"]["label"]),
        )
        recenter_start = int(side_one["end_frame"])
        recenter_end = int(proposal["recenter_end"])
        refinement_start = int(proposal["refinement_start"])
        side_two_end = refinement_start
        old_side_two_end = int(old_side_two["end_frame"])

        stages = [
            _stage(
                "side_one",
                0,
                recenter_start,
                present=True,
                confidence="high",
                source="frozen_three_stage_v2_boundary",
            ),
            _stage(
                "recenter_pull",
                recenter_start,
                recenter_end,
                present=bool(proposal["recenter_present"]),
                confidence=str(proposal["recenter_confidence"]),
                source=str(proposal["recenter_source"]),
            ),
            _stage(
                "side_two",
                recenter_end,
                side_two_end,
                present=True,
                confidence="medium",
                source="between_recenter_and_refinement_proposals",
            ),
            _stage(
                "strip_refinement",
                refinement_start,
                old_side_two_end,
                present=bool(proposal["refinement_present"]),
                confidence=str(proposal["refinement_confidence"]),
                source=str(proposal["refinement_source"]),
            ),
            _stage(
                "bottom_to_top",
                old_side_two_end,
                frame_count,
                present=True,
                confidence="high",
                source="frozen_three_stage_v2_boundary",
            ),
        ]

        outcomes = [
            {
                "name": "side_one",
                "observation_frame": int(side_one["end_frame"]) - 1,
                "observed_after_stage": "side_one",
                "quality_score": int(side_one["quality_score"]),
                "quality_confidence": str(side_one["quality_confidence"]),
            },
            {
                "name": "side_two",
                "observation_frame": old_side_two_end - 1,
                "observed_after_stage": "strip_refinement",
                "quality_score": int(old_side_two["quality_score"]),
                "quality_confidence": str(old_side_two["quality_confidence"]),
            },
            {
                "name": "bottom_to_top",
                "observation_frame": frame_count - 1,
                "observed_after_stage": "bottom_to_top",
                "quality_score": int(bottom_to_top["quality_score"]),
                "quality_confidence": str(bottom_to_top["quality_confidence"]),
            },
        ]
        episodes.append(
            {
                "episode_index": episode_index,
                "frame_count": frame_count,
                "stages": stages,
                "outcomes": outcomes,
                "boundary_evidence": {
                    "bimanual_closed_runs": proposal["bimanual_runs"],
                    "recenter_gap_frames": proposal["recenter_gap_frames"],
                },
            }
        )

    return {
        "schema_version": "openarm2.shirt-fold-semantic-segmentation.v1",
        "description": (
            "Five-stage semantic segmentation proposals for the 33 trimmed successful "
            "OpenARM2 shirt-fold demonstrations. Optional recentering and refinement are "
            "separated from fold motions; the three frozen v2 outcome-quality observations "
            "remain distinct from motion-stage labels."
        ),
        "dataset": source["dataset"],
        "stage_order": list(STAGE_ORDER),
        "optional_stages": list(OPTIONAL_STAGES),
        "outcome_order": ["side_one", "side_two", "bottom_to_top"],
        "generation_contract": {
            "status": "proposed_for_human_review",
            "action_feature": "action",
            "right_gripper_index": GRIPPER_INDICES[0],
            "left_gripper_index": GRIPPER_INDICES[1],
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "minimum_bimanual_run_frames": MIN_BIMANUAL_RUN_FRAMES,
            "bridge_open_gap_frames": BRIDGE_OPEN_GAP_FRAMES,
            "recenter_detection": {
                "split_rule": "midpoint_of_largest_stable_dual-release_gap",
                "first_run_start_fraction_max": RECENTER_FIRST_RUN_START_FRACTION_MAX,
                "first_group_end_fraction_max": RECENTER_FIRST_GROUP_END_FRACTION_MAX,
                "fold_run_start_fraction_min": RECENTER_FOLD_RUN_START_FRACTION_MIN,
            },
            "refinement_detection": {
                "rule": "75pct_of_final_bimanual_cycle_to_old_side_two_end",
                "fold_cycle_fraction": REFINEMENT_FOLD_CYCLE_FRACTION,
                "requires_human_review": True,
            },
            "absent_stage_encoding": "zero_length_half_open_interval",
            "quality_contract": (
                "Outcome quality remains attached to three observation frames and must not "
                "be interpreted as quality of recenter_pull, side_two motion, or "
                "strip_refinement in isolation."
            ),
        },
        "provenance": {
            "source_manifest": _portable_path(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "action_data": _portable_path(data_path),
            "action_data_sha256": _sha256(data_path),
        },
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_manifest = args.source_manifest.expanduser().resolve()
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = build_manifest(source_manifest, data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    present_counts = {
        stage: sum(
            item["present"]
            for episode in manifest["episodes"]
            for item in episode["stages"]
            if item["name"] == stage
        )
        for stage in STAGE_ORDER
    }
    print(json.dumps({"output": str(output), "present_counts": present_counts}, indent=2))


if __name__ == "__main__":
    main()
