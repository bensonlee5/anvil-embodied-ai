#!/usr/bin/env python3
"""Materialize or verify the exact native-LeRobot SARM dataset derivative."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anvil_trainer.priority_sampling import PriorityManifest
from anvil_trainer.sarm_annotations import (
    SARMAnnotationContract,
    materialize_sarm_dataset,
    validate_sarm_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned"
DEFAULT_OUTPUT = ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1"
DEFAULT_PRIORITY = (
    ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json"
)
DEFAULT_CONTRACT = (
    ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--priority-manifest", type=Path, default=DEFAULT_PRIORITY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify an existing output without modifying either dataset tree.",
    )
    args = parser.parse_args()

    manifest = PriorityManifest.load(args.priority_manifest)
    contract = SARMAnnotationContract.load(args.contract, priority_manifest=manifest)
    if args.check:
        result = validate_sarm_dataset(
            args.output_root,
            manifest=manifest,
            contract=contract,
        )
    else:
        result = materialize_sarm_dataset(
            args.source_root,
            args.output_root,
            manifest=manifest,
            contract=contract,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
