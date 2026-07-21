"""Fail-closed provenance verification for learned RA-BC sample weights."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from lerobot.utils.sample_weighting import SampleWeightingConfig


class RABCAuditError(ValueError):
    """Raised when an RA-BC recipe is not pinned to its audited progress artifact."""


_AUDIT_KEYS = {
    "audit_path",
    "audit_sha256",
    "source_progress_sha256",
    "training_progress_sha256",
    "priority_manifest_sha256",
    "sarm_contract_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact_path(raw: str, dataset_root: str | None) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute() and dataset_root:
        path = Path(dataset_root).expanduser() / path
    return path.resolve()


def make_audit_verified_sample_weighter(
    config: SampleWeightingConfig | None,
    policy: Any,
    device: Any,
    dataset_root: str | None,
    dataset_repo_id: str | None,
    *,
    original_factory: Callable[..., Any],
) -> Any:
    """Verify SARM audit provenance, then delegate to LeRobot's native factory."""
    if config is None or config.type != "rabc" or "audit_path" not in config.extra_params:
        return original_factory(
            config,
            policy,
            device,
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
        )
    missing = _AUDIT_KEYS - set(config.extra_params)
    if missing:
        raise RABCAuditError(f"RA-BC audit parameters are missing: {sorted(missing)}")
    audit_path = _resolve_artifact_path(
        str(config.extra_params["audit_path"]), dataset_root
    )
    if not audit_path.is_file():
        raise RABCAuditError(f"RA-BC progress audit not found: {audit_path}")
    expected_audit_hash = str(config.extra_params["audit_sha256"])
    actual_audit_hash = _sha256(audit_path)
    if actual_audit_hash != expected_audit_hash:
        raise RABCAuditError(
            f"RA-BC audit hash mismatch: expected {expected_audit_hash}, got {actual_audit_hash}"
        )
    audit = json.loads(audit_path.read_text())
    if audit.get("schema_version") != "openarm2.sarm-progress-audit.v1":
        raise RABCAuditError("Unsupported or missing RA-BC progress-audit schema")
    exact_fields = {
        "progress_sha256": config.extra_params["source_progress_sha256"],
        "priority_manifest_sha256": config.extra_params["priority_manifest_sha256"],
        "sarm_contract_sha256": config.extra_params["sarm_contract_sha256"],
    }
    for field, expected in exact_fields.items():
        if audit.get(field) != expected:
            raise RABCAuditError(
                f"RA-BC audit {field} mismatch: expected {expected}, got {audit.get(field)}"
            )
    progress_path = _resolve_artifact_path(config.progress_path or "", dataset_root)
    if not progress_path.is_file():
        raise RABCAuditError(f"RA-BC progress parquet not found: {progress_path}")
    training_progress = audit.get("training_progress")
    if not isinstance(training_progress, dict):
        raise RABCAuditError("RA-BC audit does not identify a train-only progress artifact")
    if training_progress.get("sha256") != config.extra_params["training_progress_sha256"]:
        raise RABCAuditError("RA-BC train-only progress hash does not match its audit")
    if training_progress.get("episodes") != audit.get("splits", {}).get("train", {}).get("episodes"):
        raise RABCAuditError("RA-BC train-only progress episodes do not match the audited split")
    if _sha256(progress_path) != training_progress["sha256"]:
        raise RABCAuditError("RA-BC train-only progress parquet hash does not match its audit")
    policy_chunk = getattr(policy.config, "chunk_size", None)
    if audit.get("chunk_size") != policy_chunk:
        raise RABCAuditError(
            f"RA-BC audit chunk size {audit.get('chunk_size')} != policy chunk {policy_chunk}"
        )
    recommended = audit.get("rabc", {}).get("recommended_kappa")
    if not isinstance(recommended, (int, float)) or not math.isfinite(float(recommended)):
        raise RABCAuditError("RA-BC audit does not contain a finite recommended kappa")
    if not math.isclose(config.kappa, float(recommended), rel_tol=0.0, abs_tol=1e-12):
        raise RABCAuditError(
            f"RA-BC config kappa {config.kappa} != audited train-only kappa {recommended}"
        )

    native_extra = {
        key: value for key, value in config.extra_params.items() if key not in _AUDIT_KEYS
    }
    verified_config = replace(
        config,
        progress_path=str(progress_path),
        extra_params=native_extra,
    )
    return original_factory(
        verified_config,
        policy,
        device,
        dataset_root=dataset_root,
        dataset_repo_id=dataset_repo_id,
    )
