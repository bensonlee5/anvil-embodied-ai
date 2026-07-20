#!/usr/bin/env python3
"""Audit full-frame SARM progress and resolve the train-only RA-BC kappa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anvil_trainer.priority_sampling import PriorityManifest
from anvil_trainer.sarm_annotations import SARMAnnotationContract
from anvil_trainer.sarm_progress import audit_sarm_progress, write_progress_audit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1"
DEFAULT_PRIORITY = (
    ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json"
)
DEFAULT_CONTRACT = (
    ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--progress",
        type=Path,
        default=DEFAULT_DATASET / "sarm_progress.parquet",
    )
    parser.add_argument(
        "--training-progress",
        type=Path,
        default=DEFAULT_DATASET / "sarm_progress_train.parquet",
        help="Train-only parquet used by native RA-BC to avoid holdout-statistics leakage",
    )
    parser.add_argument("--priority-manifest", type=Path, default=DEFAULT_PRIORITY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_DATASET / "sarm_progress_audit.json",
    )
    args = parser.parse_args()

    manifest = PriorityManifest.load(args.priority_manifest)
    contract = SARMAnnotationContract.load(args.contract, priority_manifest=manifest)
    result = audit_sarm_progress(
        args.progress,
        manifest=manifest,
        contract=contract,
        chunk_size=args.chunk_size,
        training_progress_path=args.training_progress,
    )
    write_progress_audit(result, args.output_json)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
