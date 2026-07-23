#!/usr/bin/env python3
"""Materialize or verify the exact five-stage native-LeRobot SARM dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anvil_trainer.semantic_sarm_annotations import (
    SemanticSARMContract,
    materialize_semantic_sarm_dataset,
    validate_semantic_sarm_dataset,
)
from anvil_trainer.semantic_stages import SemanticStageManifest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned"
DEFAULT_OUTPUT = ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-5stage-v1"
DEFAULT_MANIFEST = ROOT / "configs/training/semantic_manifests/openarm2_shirt_fold_5stage_v1.json"
DEFAULT_REVIEW = (
    ROOT / "configs/training/semantic_reviews/openarm2_shirt_fold_5stage_review_v1.json"
)
DEFAULT_CONTRACT = ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_5stage_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    manifest = SemanticStageManifest.load(args.manifest)
    contract = SemanticSARMContract.load(
        args.contract,
        manifest=manifest,
        review_path=args.review,
    )
    if args.check:
        result = validate_semantic_sarm_dataset(
            args.output_root,
            manifest=manifest,
            review_path=args.review,
            contract=contract,
        )
    else:
        result = materialize_semantic_sarm_dataset(
            args.source_root,
            args.output_root,
            manifest=manifest,
            review_path=args.review,
            contract=contract,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
