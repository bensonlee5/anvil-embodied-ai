#!/usr/bin/env python3
"""Build the v2 shirt-fold quality manifest from the blinded rubric review."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = ROOT / "configs/training/quality_annotations/openarm2_shirt_fold_3stage_v2"
DEFAULT_RAW_SCORES = ANNOTATION_DIR / "blind_stage_quality_raw_v2.csv"
DEFAULT_MAPPING = ANNOTATION_DIR / "blind_stage_mapping_v2.json"
DEFAULT_TRIM_MANIFEST = (
    ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned/meta/trim_manifest.json"
)
DEFAULT_V1_MANIFEST = (
    ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json"
)
DEFAULT_OUTPUT_MANIFEST = (
    ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json"
)

STAGE_CRITERIA = {
    "side_one": (
        "lateral_placement",
        "sleeve_side_containment",
        "longitudinal_alignment",
        "flatness_body_preservation",
    ),
    "side_two": (
        "second_side_placement",
        "second_sleeve_containment",
        "preservation_bilateral_alignment",
        "flatness_straightness",
    ),
    "bottom_to_top": (
        "fold_placement_coverage",
        "rectangularity_compactness",
        "layer_edge_containment",
        "flatness_stability",
    ),
}
SPLIT = {
    "seed": 1000,
    "ratio": [8, 1, 1],
    "train": [
        0,
        1,
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        13,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        28,
        29,
        30,
        31,
        32,
    ],
    "validation": [2, 11, 14],
    "test": [6, 12, 27],
}

# The raw blind pass is immutable. These six changes were made only after the
# anonymous tokens were resolved and a five-second, three-camera temporal panel
# was reviewed. Each override changes criterion scores, never just the 1-5 sum.
ADJUDICATION_OVERRIDES: dict[tuple[int, str], dict[str, Any]] = {
    (3, "side_one"): {
        "criteria": (1, 1, 1, 1),
        "reason": (
            "Temporal wrist views exposed broad bunching and uneven placement that "
            "the final base snapshots made look like a localized loose edge."
        ),
    },
    (23, "side_one"): {
        "criteria": (2, 2, 2, 1),
        "reason": (
            "Temporal wrist views showed a persistent layered ridge, so flatness is "
            "acceptable rather than strong."
        ),
    },
    (4, "side_two"): {
        "criteria": (2, 1, 2, 1),
        "reason": (
            "The temporal views showed a rolled sleeve/edge and a non-flat end that "
            "were understated in the final base snapshots."
        ),
    },
    (9, "side_two"): {
        "criteria": (2, 1, 2, 1),
        "reason": (
            "The settled strip retained a protruding layered end and localized roll; "
            "containment and flatness are acceptable rather than strong."
        ),
    },
    (10, "side_two"): {
        "criteria": (2, 1, 2, 1),
        "reason": (
            "Temporal views showed a persistent protruding layered flap and localized "
            "unevenness at the strip end."
        ),
    },
    (25, "side_two"): {
        "criteria": (2, 1, 2, 1),
        "reason": (
            "Temporal views showed a protruding layered end and edge waviness not "
            "visible enough in the four-frame base panel."
        ),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality_score(criteria: tuple[int, int, int, int]) -> int:
    total = sum(criteria)
    if total <= 1:
        return 1
    if total <= 3:
        return 2
    if total <= 5:
        return 3
    if total <= 7:
        return 4
    return 5


def _rank(values: list[int]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else 0.0


def _quadratic_weighted_kappa(left: list[int], right: list[int]) -> float:
    observed = [[0.0] * 5 for _ in range(5)]
    for a, b in zip(left, right):
        observed[a - 1][b - 1] += 1
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = [
        [left_counts[i + 1] * right_counts[j + 1] / len(left) for j in range(5)] for i in range(5)
    ]
    weighted_observed = 0.0
    weighted_expected = 0.0
    for i in range(5):
        for j in range(5):
            weight = ((i - j) / 4) ** 2
            weighted_observed += weight * observed[i][j]
            weighted_expected += weight * expected[i][j]
    return 1 - weighted_observed / weighted_expected if weighted_expected else 0.0


def _distribution(values: list[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(score): counts[score] for score in range(1, 6) if counts[score]}


def _comparison(old: list[int], new: list[int]) -> dict[str, Any]:
    differences = [abs(a - b) for a, b in zip(old, new)]
    return {
        "old_distribution": _distribution(old),
        "new_distribution": _distribution(new),
        "old_mean": sum(old) / len(old),
        "new_mean": sum(new) / len(new),
        "mean_delta": sum(b - a for a, b in zip(old, new)) / len(old),
        "exact_agreement": sum(a == b for a, b in zip(old, new)) / len(old),
        "within_one_agreement": sum(value <= 1 for value in differences) / len(old),
        "mean_absolute_difference": sum(differences) / len(old),
        "spearman_rank_correlation": _correlation(_rank(old), _rank(new)),
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(old, new),
    }


def _load_records(
    raw_scores_path: Path,
    mapping_path: Path,
    v1_manifest_path: Path,
    trim_manifest_path: Path,
) -> list[dict[str, Any]]:
    raw_rows = list(csv.DictReader(raw_scores_path.open()))
    if len(raw_rows) != 99 or len({row["token"] for row in raw_rows}) != 99:
        raise ValueError("blind score file must contain 99 unique tokens")
    raw_by_token = {row["token"]: row for row in raw_rows}

    mapping = json.loads(mapping_path.read_text())
    if mapping["schema_version"] != "openarm2.blind-stage-review.v1":
        raise ValueError("unsupported blind-stage mapping schema")
    if len(mapping["items"]) != 99:
        raise ValueError("blind mapping must contain 99 items")

    v1_manifest = json.loads(v1_manifest_path.read_text())
    old_quality = {
        (episode["episode_index"], stage["name"]): stage["quality_score"]
        for episode in v1_manifest["episodes"]
        for stage in episode["stages"]
    }
    trims = {
        int(item["episode_index"]): item
        for item in json.loads(trim_manifest_path.read_text())["episodes"]
    }

    records: list[dict[str, Any]] = []
    for item in mapping["items"]:
        raw = raw_by_token[item["token"]]
        episode_index = int(item["episode_index"])
        stage = str(item["stage"])
        if raw["stage"] != stage:
            raise ValueError(f"stage mismatch for token {item['token']}")
        raw_criteria = tuple(int(raw[f"criterion_{index}"]) for index in range(1, 5))
        if any(value not in {0, 1, 2} for value in raw_criteria):
            raise ValueError(f"invalid criterion score for token {item['token']}")
        raw_quality = int(raw["quality_score"])
        if raw_quality != _quality_score(raw_criteria):
            raise ValueError(f"quality total mismatch for token {item['token']}")

        override = ADJUDICATION_OVERRIDES.get((episode_index, stage))
        final_criteria = tuple(override["criteria"]) if override else raw_criteria
        final_quality = _quality_score(final_criteria)
        old_score = int(old_quality[(episode_index, stage)])
        temporal_reviewed = (
            abs(old_score - raw_quality) > 1 or raw["visibility_confidence"] != "high"
        )
        trim = trims[episode_index]
        source_offset = int(trim["source_from_index"]) + int(trim["start"])
        retained_start = int(item["retained_start_frame"])
        retained_end = int(item["retained_end_frame_exclusive"])
        records.append(
            {
                "episode_index": episode_index,
                "stage": stage,
                "token": item["token"],
                "old_v1_quality_score": old_score,
                "raw_criteria": raw_criteria,
                "raw_quality_score": raw_quality,
                "raw_visibility_confidence": raw["visibility_confidence"],
                "raw_review_note": raw["review_note"],
                "temporal_reviewed": temporal_reviewed,
                "final_criteria": final_criteria,
                "final_quality_score": final_quality,
                "final_label_confidence": "medium",
                "adjudication_changed_score": final_quality != raw_quality,
                "adjudication_reason": (
                    override["reason"]
                    if override
                    else (
                        "Temporal three-camera review confirmed the raw criterion scores."
                        if temporal_reviewed
                        else ""
                    )
                ),
                "retained_start_frame": retained_start,
                "retained_end_frame_exclusive": retained_end,
                "source_start_frame": source_offset + retained_start,
                "source_end_frame_exclusive": source_offset + retained_end,
            }
        )

    stage_index = {stage: index for index, stage in enumerate(STAGE_CRITERIA)}
    records.sort(key=lambda row: (row["episode_index"], stage_index[row["stage"]]))
    if Counter(row["stage"] for row in records) != Counter(dict.fromkeys(STAGE_CRITERIA, 33)):
        raise ValueError("every stage must contain exactly 33 labels")
    return records


def _build_manifest(v1_manifest_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = copy.deepcopy(json.loads(v1_manifest_path.read_text()))
    final_scores = {
        (row["episode_index"], row["stage"]): row["final_quality_score"] for row in records
    }
    manifest["description"] = (
        "Conservative three-stage priority sampler for the 33 trimmed successful OpenARM2 "
        "shirt-fold demonstrations, using the blinded criterion-based v2 quality review and "
        "the seed-1000 27/3/3 split."
    )
    manifest["annotation_contract"]["repeated_grasp_definition"] = (
        "Telemetry retry candidates are excluded from v2 priorities until video review can "
        "distinguish confirmed misgrasps from deliberate regrasping."
    )
    manifest["annotation_contract"]["smoothing_definition"] = (
        "Deliberate contact refinement is quality-neutral; slow targeted smoothing is not an "
        "error signal."
    )
    manifest["weighting"]["quality_log_priority"] = {
        "1": -0.4,
        "2": -0.2,
        "3": 0.0,
        "4": 0.2,
        "5": 0.4,
    }
    manifest["weighting"]["repeated_grasp_log_penalty"] = 0.0
    manifest["weighting"]["stage_probability_mass"] = {
        stage: sum(
            annotation["end_frame"] - annotation["start_frame"]
            for episode in manifest["episodes"]
            if episode["episode_index"] in SPLIT["train"]
            for annotation in episode["stages"]
            if annotation["name"] == stage
        )
        for stage in manifest["stage_order"]
    }
    for episode in manifest["episodes"]:
        for stage in episode["stages"]:
            stage["quality_score"] = final_scores[(episode["episode_index"], stage["name"])]
            stage["quality_confidence"] = "medium"
        episode["repeated_grasps"] = []
    return manifest


def _quality_csv(records: list[dict[str, Any]]) -> str:
    fields = [
        "episode_index",
        "stage",
        "token",
        "old_v1_quality_score",
        "criterion_1_name",
        "raw_criterion_1_score",
        "final_criterion_1_score",
        "criterion_2_name",
        "raw_criterion_2_score",
        "final_criterion_2_score",
        "criterion_3_name",
        "raw_criterion_3_score",
        "final_criterion_3_score",
        "criterion_4_name",
        "raw_criterion_4_score",
        "final_criterion_4_score",
        "raw_quality_score",
        "raw_visibility_confidence",
        "raw_review_note",
        "temporal_reviewed",
        "final_quality_score",
        "final_label_confidence",
        "adjudication_changed_score",
        "adjudication_reason",
        "retained_start_frame",
        "retained_end_frame_exclusive",
        "source_start_frame",
        "source_end_frame_exclusive",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    criterion_fields = {
        f"{prefix}_criterion_{index}_score" for index in range(1, 5) for prefix in ("raw", "final")
    } | {f"criterion_{index}_name" for index in range(1, 5)}
    for row in records:
        output: dict[str, Any] = {key: row[key] for key in fields if key not in criterion_fields}
        for index, (name, raw_score, final_score) in enumerate(
            zip(
                STAGE_CRITERIA[row["stage"]],
                row["raw_criteria"],
                row["final_criteria"],
            ),
            start=1,
        ):
            output[f"criterion_{index}_name"] = name
            output[f"raw_criterion_{index}_score"] = raw_score
            output[f"final_criterion_{index}_score"] = final_score
        writer.writerow(output)
    return stream.getvalue()


def _build_report(
    records: list[dict[str, Any]],
    *,
    raw_scores_path: Path,
    mapping_path: Path,
    v1_manifest_path: Path,
    trim_manifest_path: Path,
) -> dict[str, Any]:
    raw_comparisons = {}
    final_comparisons = {}
    for stage in STAGE_CRITERIA:
        stage_rows = [row for row in records if row["stage"] == stage]
        old = [row["old_v1_quality_score"] for row in stage_rows]
        raw = [row["raw_quality_score"] for row in stage_rows]
        final = [row["final_quality_score"] for row in stage_rows]
        raw_comparisons[stage] = _comparison(old, raw)
        final_comparisons[stage] = _comparison(old, final)
    return {
        "schema_version": "openarm2.stage-quality-review.v2",
        "rubric": "docs/shirt-fold-labeling-v2.md",
        "review_scope": {
            "episodes": 33,
            "stages_per_episode": 3,
            "blind_items": 99,
            "temporal_items_reviewed": sum(row["temporal_reviewed"] for row in records),
            "score_changes_after_temporal_review": sum(
                row["adjudication_changed_score"] for row in records
            ),
        },
        "provenance": {
            "raw_scores_sha256": _sha256(raw_scores_path),
            "mapping_sha256": _sha256(mapping_path),
            "v1_manifest_sha256": _sha256(v1_manifest_path),
            "trim_manifest_sha256": _sha256(trim_manifest_path),
            "blind_seed": 20260720,
        },
        "policy_episode_split": SPLIT,
        "raw_blind_vs_v1": raw_comparisons,
        "adjudicated_vs_v1": final_comparisons,
        "adjudication_changes": [
            {
                "episode_index": row["episode_index"],
                "stage": row["stage"],
                "raw_quality_score": row["raw_quality_score"],
                "final_quality_score": row["final_quality_score"],
                "raw_criteria": list(row["raw_criteria"]),
                "final_criteria": list(row["final_criteria"]),
                "reason": row["adjudication_reason"],
            }
            for row in records
            if row["adjudication_changed_score"]
        ],
        "limitations": [
            "The rescore is blind to v1 numeric labels but is not an independent second-rater study.",
            "The same AI-assisted reviewer performed the blind and temporal passes.",
            "All labels describe successful outcomes; score 1 is not a failure label.",
            "Side-fold v1 agreement is too weak to treat either label set as ground truth.",
            "Smoothing speed and retry count are excluded from outcome quality.",
        ],
    }


def _render_outputs(
    *,
    raw_scores_path: Path,
    mapping_path: Path,
    v1_manifest_path: Path,
    trim_manifest_path: Path,
) -> dict[Path, str]:
    records = _load_records(
        raw_scores_path,
        mapping_path,
        v1_manifest_path,
        trim_manifest_path,
    )
    manifest = _build_manifest(v1_manifest_path, records)
    report = _build_report(
        records,
        raw_scores_path=raw_scores_path,
        mapping_path=mapping_path,
        v1_manifest_path=v1_manifest_path,
        trim_manifest_path=trim_manifest_path,
    )

    return {
        ANNOTATION_DIR / "episode_stage_quality_v2.csv": _quality_csv(records),
        ANNOTATION_DIR / "adjudication_report_v2.json": json.dumps(report, indent=2) + "\n",
        DEFAULT_OUTPUT_MANIFEST: json.dumps(manifest, indent=2) + "\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-scores", type=Path, default=DEFAULT_RAW_SCORES)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--v1-manifest", type=Path, default=DEFAULT_V1_MANIFEST)
    parser.add_argument("--trim-manifest", type=Path, default=DEFAULT_TRIM_MANIFEST)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    outputs = _render_outputs(
        raw_scores_path=args.raw_scores,
        mapping_path=args.mapping,
        v1_manifest_path=args.v1_manifest,
        trim_manifest_path=args.trim_manifest,
    )
    if args.check:
        stale = [
            str(path)
            for path, text in outputs.items()
            if not path.exists() or path.read_text() != text
        ]
        if stale:
            raise SystemExit("stale generated files: " + ", ".join(stale))
        print(json.dumps({"status": "current", "files": len(outputs)}, indent=2))
        return
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(json.dumps({"status": "written", "files": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
