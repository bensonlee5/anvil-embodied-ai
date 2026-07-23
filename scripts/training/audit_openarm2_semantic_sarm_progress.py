#!/usr/bin/env python3
"""Gate five-stage SARM progress and emit corrected train-only RA-BC input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anvil_trainer.semantic_sarm_annotations import SemanticSARMContract
from anvil_trainer.semantic_sarm_progress import (
    audit_semantic_sarm_progress,
    write_semantic_progress_audit,
)
from anvil_trainer.semantic_stages import SemanticStageManifest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-5stage-v1"
DEFAULT_MANIFEST = ROOT / "configs/training/semantic_manifests/openarm2_shirt_fold_5stage_v1.json"
DEFAULT_REVIEW = (
    ROOT / "configs/training/semantic_reviews/openarm2_shirt_fold_5stage_review_v1.json"
)
DEFAULT_CONTRACT = ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_5stage_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--progress", type=Path, default=DEFAULT_DATASET / "sarm_progress_5stage_v1.parquet"
    )
    parser.add_argument(
        "--training-progress",
        type=Path,
        default=DEFAULT_DATASET / "sarm_progress_train_5stage_v1.parquet",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_DATASET / "sarm_progress_audit_5stage_v1.json",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero after writing the audit if the policy-training gate fails",
    )
    args = parser.parse_args()

    manifest = SemanticStageManifest.load(args.manifest)
    contract = SemanticSARMContract.load(
        args.contract,
        manifest=manifest,
        review_path=args.review,
    )
    result = audit_semantic_sarm_progress(
        args.progress,
        manifest=manifest,
        contract=contract,
        chunk_size=args.chunk_size,
        training_progress_path=args.training_progress,
    )
    write_semantic_progress_audit(result, args.output_json)
    print(json.dumps(result, indent=2))
    if args.require_pass and not result["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
