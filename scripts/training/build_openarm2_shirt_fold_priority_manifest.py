#!/usr/bin/env python3
"""Build the frozen three-stage OpenARM2 shirt-fold priority manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EPISODE_LENGTHS = [
    892, 1194, 833, 1215, 1105, 1241, 1046, 1022, 1026, 1298, 1032,
    875, 936, 1160, 909, 900, 1061, 1197, 1020, 1082, 973, 1095, 963,
    1255, 991, 1255, 910, 1017, 1112, 1010, 1124, 1003, 1098,
]

# End-exclusive retained-frame boundaries.  Boundaries were reviewed against
# the base-camera state after each stable gripper release/next-stage grasp.
SIDE_ONE_END = [
    132, 269, 166, 241, 176, 235, 157, 138, 179, 157, 172,
    169, 199, 249, 153, 181, 262, 304, 204, 244, 202, 272,
    247, 249, 248, 296, 191, 239, 276, 246, 273, 228, 303,
]
SIDE_TWO_END = [
    681, 961, 522, 903, 726, 652, 696, 721, 712, 796, 669,
    631, 682, 766, 682, 645, 798, 948, 704, 799, 462, 829,
    571, 893, 753, 894, 654, 628, 866, 708, 797, 687, 841,
]

# Quality describes the garment state produced by each fold stage.  Every
# demonstration remains a task success; 1 means a poor-quality success, not a
# failure.  These are single-annotator visual labels and therefore medium
# confidence even when the garment is clearly visible.
SIDE_ONE_QUALITY = [
    3, 4, 3, 2, 3, 5, 4, 5, 3, 3, 4, 4, 4, 5, 4, 4, 5,
    4, 5, 3, 4, 3, 4, 3, 5, 4, 4, 4, 3, 4, 3, 5, 4,
]
SIDE_TWO_QUALITY = [
    2, 4, 4, 3, 2, 3, 3, 3, 2, 2, 3, 2, 4, 4, 3, 3, 4,
    3, 4, 4, 4, 4, 5, 2, 4, 3, 4, 3, 2, 3, 2, 3, 4,
]
BOTTOM_TO_TOP_QUALITY = [
    1, 2, 1, 3, 1, 5, 5, 4, 1, 3, 3, 5, 1, 3, 3, 4, 5,
    1, 5, 3, 4, 1, 5, 2, 5, 1, 3, 2, 3, 5, 2, 1, 1,
]

# (episode, gripper, close, reopen, retry).  A candidate requires a short close
# followed by a reopen and another close by the same gripper within 1.5 s.  The
# command trace and base-camera window were reviewed together.  The penalty
# window includes 0.5 s of approach and ends at the retry; the recovery itself
# is retained at the stage's ordinary priority.
REPEATED_GRASPS = [
    (0, "right", 433, 461, 463),
    (0, "right", 463, 465, 487),
    (0, "right", 487, 503, 523),
    (1, "right", 553, 564, 606),
    (1, "left", 561, 568, 605),
    (2, "left", 435, 479, 522),
    (3, "left", 328, 355, 372),
    (4, "right", 476, 505, 550),
    (5, "right", 910, 932, 964),
    (6, "right", 461, 486, 491),
    (6, "right", 491, 493, 506),
    (6, "right", 820, 835, 866),
    (7, "right", 484, 495, 508),
    (7, "right", 823, 853, 876),
    (9, "right", 939, 978, 1008),
    (9, "left", 193, 208, 234),
    (11, "right", 422, 466, 486),
    (13, "right", 887, 927, 963),
    (14, "right", 480, 487, 505),
    (15, "right", 573, 574, 586),
    (16, "left", 44, 46, 71),
    (18, "left", 671, 672, 704),
    (23, "right", 675, 715, 735),
    (25, "right", 646, 674, 692),
    (27, "right", 410, 428, 447),
    (27, "right", 862, 869, 875),
    (30, "right", 291, 298, 336),
    (32, "left", 782, 826, 841),
]

# Coarse retained-task windows from the independent smoothing review.  They
# intentionally remain review windows, not frame-accurate masks.  Smoothing is
# kept quality-neutral because it can be slow, targeted, and deliberate.
SMOOTHING_SECONDS = [
    ("present", "high", 7, 20),
    ("present", "high", 12, 22),
    ("present", "medium", 7, 16),
    ("present", "high", 9, 24),
    ("uncertain", "low", 12, 20),
    ("present", "high", 9, 23),
    ("present", "high", 7, 18),
    ("present", "high", 9, 19),
    ("present", "high", 9, 20),
    ("present", "high", 9, 21),
    ("present", "medium", 7, 17),
    ("present", "high", 7, 17),
    ("present", "medium", 8, 18),
    ("present", "high", 8, 22),
    ("present", "high", 8, 19),
    ("present", "high", 7, 17),
    ("present", "high", 10, 22),
    ("present", "high", 9, 21),
    ("present", "high", 8, 20),
    ("present", "high", 11, 21),
    ("present", "medium", 7, 18),
    ("present", "high", 12, 23),
    ("present", "medium", 7, 16),
    ("present", "high", 10, 24),
    ("present", "high", 9, 21),
    ("uncertain", "low", 12, 24),
    ("present", "high", 10, 18),
    ("present", "high", 7, 17),
    ("present", "high", 10, 23),
    ("present", "high", 9, 20),
    ("present", "medium", 9, 21),
    ("present", "medium", 10, 19),
    ("present", "high", 9, 22),
]


def stage_for_frame(episode_index: int, frame: int) -> str:
    if frame < SIDE_ONE_END[episode_index]:
        return "side_one"
    if frame < SIDE_TWO_END[episode_index]:
        return "side_two"
    return "bottom_to_top"


def smoothing_context(episode_index: int, start: int, end: int) -> list[str]:
    ranges = [
        ("side_one", 0, SIDE_ONE_END[episode_index]),
        ("side_two", SIDE_ONE_END[episode_index], SIDE_TWO_END[episode_index]),
        ("bottom_to_top", SIDE_TWO_END[episode_index], EPISODE_LENGTHS[episode_index]),
    ]
    return [name for name, stage_start, stage_end in ranges if start < stage_end and end > stage_start]


def build_manifest() -> dict:
    repeated_by_episode: dict[int, list[dict]] = {index: [] for index in range(33)}
    for episode, gripper, close, reopen, retry in REPEATED_GRASPS:
        repeated_by_episode[episode].append(
            {
                "gripper": gripper,
                "stage": stage_for_frame(episode, close),
                "close_frame": close,
                "reopen_frame": reopen,
                "retry_frame": retry,
                "penalty_start_frame": max(0, close - 15),
                "penalty_end_frame": retry,
                "confidence": "medium",
            }
        )

    episodes = []
    for episode_index, frame_count in enumerate(EPISODE_LENGTHS):
        smoothing_label, smoothing_confidence, start_second, end_second = SMOOTHING_SECONDS[
            episode_index
        ]
        smoothing_start = min(start_second * 30, frame_count - 1)
        smoothing_end = min(end_second * 30, frame_count)
        episodes.append(
            {
                "episode_index": episode_index,
                "frame_count": frame_count,
                "stages": [
                    {
                        "name": "side_one",
                        "start_frame": 0,
                        "end_frame": SIDE_ONE_END[episode_index],
                        "quality_score": SIDE_ONE_QUALITY[episode_index],
                        "quality_confidence": "medium",
                    },
                    {
                        "name": "side_two",
                        "start_frame": SIDE_ONE_END[episode_index],
                        "end_frame": SIDE_TWO_END[episode_index],
                        "quality_score": SIDE_TWO_QUALITY[episode_index],
                        "quality_confidence": "medium",
                    },
                    {
                        "name": "bottom_to_top",
                        "start_frame": SIDE_TWO_END[episode_index],
                        "end_frame": frame_count,
                        "quality_score": BOTTOM_TO_TOP_QUALITY[episode_index],
                        "quality_confidence": "medium",
                    },
                ],
                "repeated_grasps": repeated_by_episode[episode_index],
                "smoothing": {
                    "label": smoothing_label,
                    "confidence": smoothing_confidence,
                    "review_start_frame": smoothing_start,
                    "review_end_frame": smoothing_end,
                    "stage_context": smoothing_context(
                        episode_index, smoothing_start, smoothing_end
                    ),
                    "priority_adjustment": 0.0,
                },
            }
        )

    return {
        "schema_version": "openarm2.priority-sampling.v1",
        "description": (
            "Three-stage quality and behavior annotations for the 33 trimmed successful "
            "OpenARM2 shirt-fold demonstrations, resolved into Larchenko-inspired frame priorities."
        ),
        "dataset": {
            "repo_id": "bohlt/openarm2-shirt-fold-phase-aligned-v1",
            "revision": "8411e3e85eaf3e482b4ccb1cac9d4fc02891305e",
            "episodes": 33,
            "frames": 34850,
            "fps": 30,
            "fingerprints": {
                "meta/info.json": "6022ac2f297aa46503934b3710be54004db78c93a1c3892e8382a1e24d590285",
                "data/chunk-000/file-000.parquet": "2d8a6e2a851df08c4a4d1173d89b07aeefd886ee38c3505644a233c39453d867",
                "meta/trim_manifest.json": "97019524cda0d85347979b00db56c6e19e26b0d962e1a17fee35c22832a0821f",
            },
        },
        "stage_order": ["side_one", "side_two", "bottom_to_top"],
        "annotation_contract": {
            "quality_scale": [1, 5],
            "repeated_grasp_definition": (
                "A short close/reopen followed by another close from the same gripper within 1.5 s; "
                "the event is quality evidence, not a task-failure label."
            ),
            "smoothing_definition": (
                "A deliberate contact pass over an already placed layer or edge; review windows are "
                "coarse and independent of motion speed."
            ),
            "smoothing_affects_priority": False,
            "single_annotator": True,
        },
        "weighting": {
            "method": "exponential_priority_sampler",
            "loss_reweighting": False,
            "sampling_replacement": True,
            "quality_log_priority": {"1": -1.0, "2": -0.5, "3": 0.0, "4": 0.5, "5": 1.0},
            "repeated_grasp_log_penalty": -0.5,
            "smoothing_log_adjustment": 0.0,
            "stage_probability_mass": {
                "side_one": 1.0,
                "side_two": 1.0,
                "bottom_to_top": 1.0,
            },
        },
        "episodes": episodes,
    }


def write_review_artifacts(
    output_dir: Path,
    manifest: dict,
    *,
    trim_manifest_path: Path,
) -> None:
    """Write human-readable source/retained annotation tables."""
    trim_manifest = json.loads(trim_manifest_path.read_text())
    trims = {item["episode_index"]: item for item in trim_manifest["episodes"]}
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "episode_stage_quality_v2.csv").open("w", newline="") as stream:
        fields = [
            "episode_index",
            "stage",
            "quality_score",
            "quality_confidence",
            "retained_start_frame",
            "retained_end_frame_exclusive",
            "source_start_frame",
            "source_end_frame_exclusive",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for episode in manifest["episodes"]:
            trim_start = trims[episode["episode_index"]]["start"]
            for stage in episode["stages"]:
                writer.writerow(
                    {
                        "episode_index": episode["episode_index"],
                        "stage": stage["name"],
                        "quality_score": stage["quality_score"],
                        "quality_confidence": stage["quality_confidence"],
                        "retained_start_frame": stage["start_frame"],
                        "retained_end_frame_exclusive": stage["end_frame"],
                        "source_start_frame": trim_start + stage["start_frame"],
                        "source_end_frame_exclusive": trim_start + stage["end_frame"],
                    }
                )

    with (output_dir / "repeated_grasps_v2.csv").open("w", newline="") as stream:
        fields = [
            "episode_index",
            "stage",
            "gripper",
            "confidence",
            "retained_close_frame",
            "retained_reopen_frame",
            "retained_retry_frame",
            "source_close_frame",
            "source_reopen_frame",
            "source_retry_frame",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for episode in manifest["episodes"]:
            trim_start = trims[episode["episode_index"]]["start"]
            for event in episode["repeated_grasps"]:
                writer.writerow(
                    {
                        "episode_index": episode["episode_index"],
                        "stage": event["stage"],
                        "gripper": event["gripper"],
                        "confidence": event["confidence"],
                        "retained_close_frame": event["close_frame"],
                        "retained_reopen_frame": event["reopen_frame"],
                        "retained_retry_frame": event["retry_frame"],
                        "source_close_frame": trim_start + event["close_frame"],
                        "source_reopen_frame": trim_start + event["reopen_frame"],
                        "source_retry_frame": trim_start + event["retry_frame"],
                    }
                )

    with (output_dir / "episode_smoothing_v2.csv").open("w", newline="") as stream:
        fields = [
            "episode_index",
            "label",
            "confidence",
            "stage_context",
            "priority_adjustment",
            "retained_review_start_frame",
            "retained_review_end_frame_exclusive",
            "source_review_start_frame",
            "source_review_end_frame_exclusive",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for episode in manifest["episodes"]:
            trim_start = trims[episode["episode_index"]]["start"]
            smoothing = episode["smoothing"]
            writer.writerow(
                {
                    "episode_index": episode["episode_index"],
                    "label": smoothing["label"],
                    "confidence": smoothing["confidence"],
                    "stage_context": "+".join(smoothing["stage_context"]),
                    "priority_adjustment": smoothing["priority_adjustment"],
                    "retained_review_start_frame": smoothing["review_start_frame"],
                    "retained_review_end_frame_exclusive": smoothing["review_end_frame"],
                    "source_review_start_frame": trim_start + smoothing["review_start_frame"],
                    "source_review_end_frame_exclusive": trim_start
                    + smoothing["review_end_frame"],
                }
            )

    (output_dir / "README.md").write_text(
        """# Untrimmed 33-episode three-stage review v2

These tables map the reviewed full source videos to the retained training cut.
The task stages are `side_one`, `side_two`, then `bottom_to_top`. Quality is
scored independently for each produced garment state. Repeated-grasp events and
smoothing are separate annotations; smoothing is deliberately priority-neutral.

The training source of truth is
`configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json`.
Source-frame columns add the immutable trim manifest's per-episode start offset;
end columns are exclusive. The review is single-annotator and should receive a
second blinded pass before being treated as a production-quality reward model.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json"
        ),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--review-output-dir", type=Path)
    parser.add_argument(
        "--trim-manifest",
        type=Path,
        default=Path("datasets/shirt-fold/lerobot-hf-phase-aligned/meta/trim_manifest.json"),
    )
    args = parser.parse_args()
    rendered = json.dumps(build_manifest(), indent=2) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text() != rendered:
            raise SystemExit(f"Generated priority manifest is stale: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    if args.review_output_dir is not None:
        write_review_artifacts(
            args.review_output_dir,
            build_manifest(),
            trim_manifest_path=args.trim_manifest,
        )


if __name__ == "__main__":
    main()
