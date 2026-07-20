#!/usr/bin/env python3
"""Freeze an audited SARM progress artifact into a runnable RA-BC recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = (
    ROOT / "configs/training/shirt_fold_pi05_hf_phase_aligned_sarm_rabc_v1.yaml"
)
DEFAULT_AUDIT = (
    ROOT
    / "datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1/sarm_progress_audit.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/training/generated/shirt_fold_pi05_hf_phase_aligned_sarm_rabc_v1.yaml"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    template = yaml.safe_load(args.template.read_text())
    audit = json.loads(args.audit.read_text())
    if audit.get("schema_version") != "openarm2.sarm-progress-audit.v1":
        raise ValueError("Audit does not use openarm2.sarm-progress-audit.v1")
    weighting = template["sample_weighting"]
    extra = weighting["extra_params"]
    for field in ("priority_manifest_sha256", "sarm_contract_sha256"):
        if extra[field] != audit[field]:
            raise ValueError(f"Template {field} does not match the progress audit")
    weighting["kappa"] = audit["rabc"]["recommended_kappa"]
    extra["audit_sha256"] = sha256(args.audit)
    extra["source_progress_sha256"] = audit["progress_sha256"]
    training_progress = audit.get("training_progress")
    if not isinstance(training_progress, dict) or not training_progress.get("sha256"):
        raise ValueError("Audit does not pin a train-only progress artifact")
    extra["training_progress_sha256"] = training_progress["sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite resolved recipe: {args.output}")
    args.output.write_text(yaml.safe_dump(template, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()
