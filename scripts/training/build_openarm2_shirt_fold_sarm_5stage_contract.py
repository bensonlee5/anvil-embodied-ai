#!/usr/bin/env python3
"""Generate the released-SARM contract for five-stage shirt-fold semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anvil_trainer.semantic_sarm_annotations import (
    compute_temporal_proportions,
    load_semantic_review,
)
from anvil_trainer.semantic_stages import SemanticStageManifest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/training/semantic_manifests/openarm2_shirt_fold_5stage_v1.json"
DEFAULT_REVIEW = (
    ROOT / "configs/training/semantic_reviews/openarm2_shirt_fold_5stage_review_v1.json"
)
DEFAULT_OUTPUT = ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_5stage_v1.json"

TRAIN_EPISODES = [
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
]
VALIDATION_EPISODES = [2, 11, 14]
TEST_EPISODES = [6, 12, 27]


def build_contract(manifest_path: Path, review_path: Path) -> dict:
    manifest = SemanticStageManifest.load(manifest_path)
    _, review_sha, _ = load_semantic_review(review_path, manifest=manifest)
    return {
        "schema_version": "openarm2.sarm-semantic-dataset.v1",
        "description": (
            "Released single-task SARM dense-only screen over the reviewed five-stage "
            "semantic contract for all 33 trimmed successful shirt folds."
        ),
        "semantic_manifest_sha256": manifest.sha256,
        "semantic_review_sha256": review_sha,
        "annotation_mode": "dense_only",
        "sparse_task": "Fold the T-shirt properly",
        "dense_stage_order": list(manifest.stage_order),
        "image_key": "observation.images.base",
        "state_key": "observation.state",
        "frame_semantics": {
            "semantic_manifest_end": "exclusive",
            "lerobot_sarm_end": "inclusive",
            "conversion": "dense_end_frame=manifest_end_frame-1",
        },
        "split": {
            "seed": 1000,
            "ratio": [8, 1, 1],
            "train": TRAIN_EPISODES,
            "validation": VALIDATION_EPISODES,
            "test": TEST_EPISODES,
        },
        "temporal_proportions": compute_temporal_proportions(manifest, TRAIN_EPISODES),
        "optional_stage_policy": {
            "annotation_encoding": "zero_length_interval_with_inclusive_end_before_start",
            "raw_reward_behavior": "released_sarm_global_stage_skip",
            "training_progress_correction": (
                "remove_absent_stage_mass_then_renormalize_per_episode"
            ),
        },
        "behavior_labels": {
            "quality": "three_external_outcomes_not_sarm_targets",
            "repeated_grasps": "evaluation_only_not_failure_labels",
            "smoothing": "included_in_strip_refinement_not_speed_penalized",
        },
        "progress_gate": {
            "minimum_holdout_spearman": 0.6,
            "maximum_holdout_mae": 0.2,
            "maximum_stage_monotonicity_violation_rate": 0.25,
            "maximum_corrected_optional_skip_jump": 0.1,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = build_contract(args.manifest, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
