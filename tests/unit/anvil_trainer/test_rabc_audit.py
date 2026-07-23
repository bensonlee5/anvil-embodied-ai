"""RA-BC must be pinned to the exact audited SARM progress artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from lerobot.utils.sample_weighting import SampleWeightingConfig

from anvil_trainer.rabc_audit import (
    RABCAuditError,
    make_audit_verified_sample_weighter,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[SampleWeightingConfig, SimpleNamespace, dict]:
    progress = tmp_path / "sarm_progress.parquet"
    progress.write_bytes(b"progress")
    audit = {
        "schema_version": "openarm2.sarm-semantic-progress-audit.v2",
        "progress_sha256": _sha256(progress),
        "semantic_manifest_sha256": "1" * 64,
        "semantic_sarm_contract_sha256": "2" * 64,
        "progress_calibration_contract_sha256": "3" * 64,
        "calibration_scope": "offline_train_weighting_only",
        "training_progress": {
            "path": str(progress),
            "sha256": _sha256(progress),
            "frames": 1,
            "episodes": [0],
        },
        "splits": {"train": {"episodes": [0]}},
        "chunk_size": 30,
        "rabc": {"recommended_kappa": 0.03125},
        "gate": {"passed": True},
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    config = SampleWeightingConfig(
        type="rabc",
        progress_path=str(progress),
        head_mode="dense",
        kappa=0.03125,
        extra_params={
            "audit_path": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "source_progress_sha256": _sha256(progress),
            "training_progress_sha256": _sha256(progress),
            "semantic_manifest_sha256": "1" * 64,
            "semantic_sarm_contract_sha256": "2" * 64,
            "progress_calibration_contract_sha256": "3" * 64,
            "fallback_weight": 1.0,
        },
    )
    return config, SimpleNamespace(config=SimpleNamespace(chunk_size=30)), audit


def _raw_fixture(tmp_path: Path) -> tuple[SampleWeightingConfig, SimpleNamespace, dict]:
    progress = tmp_path / "sarm_progress_raw.parquet"
    progress.write_bytes(b"raw-progress")
    episodes = [0]
    audit = {
        "schema_version": "openarm2.sarm-semantic-progress-audit.v1",
        "progress_sha256": _sha256(progress),
        "semantic_manifest_sha256": "4" * 64,
        "semantic_sarm_contract_sha256": "5" * 64,
        "training_progress": {
            "path": str(progress),
            "sha256": _sha256(progress),
            "frames": 1,
            "episodes": episodes,
            "correction": "remove_absent_stage_mass_then_renormalize_per_episode",
        },
        "optional_stage_corrected": {
            "splits": {"train": {"episodes": episodes, "frames": 1}}
        },
        "chunk_size": 30,
        "rabc": {"recommended_kappa": 0.05},
        "gate": {
            "checks": {
                "validation_spearman": True,
                "test_spearman": True,
                "validation_mae": True,
                "test_mae": True,
                "stage_monotonicity": False,
                "corrected_optional_skip_jump": False,
                "positive_finite_kappa": True,
                "train_only_rows": True,
            },
            "passed": False,
        },
    }
    audit_path = tmp_path / "raw_audit.json"
    audit_path.write_text(json.dumps(audit))
    config = SampleWeightingConfig(
        type="rabc",
        progress_path=str(progress),
        head_mode="dense",
        kappa=0.05,
        extra_params={
            "audit_path": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "source_progress_sha256": _sha256(progress),
            "training_progress_sha256": _sha256(progress),
            "semantic_manifest_sha256": "4" * 64,
            "semantic_sarm_contract_sha256": "5" * 64,
            "reward_calibration": "none",
            "negative_delta_audit_tolerance": 0.05,
            "fallback_weight": 1.0,
        },
    )
    return config, SimpleNamespace(config=SimpleNamespace(chunk_size=30)), audit


def test_verified_factory_strips_only_audit_parameters(tmp_path: Path) -> None:
    config, policy, _audit = _fixture(tmp_path)
    captured = {}

    def original_factory(native_config, *_args, **_kwargs):
        captured["config"] = native_config
        return "verified"

    result = make_audit_verified_sample_weighter(
        config,
        policy,
        "cpu",
        None,
        None,
        original_factory=original_factory,
    )
    assert result == "verified"
    assert captured["config"].extra_params == {"fallback_weight": 1.0}


def test_verified_factory_accepts_raw_progress_without_fitted_calibration(
    tmp_path: Path,
) -> None:
    config, policy, _audit = _raw_fixture(tmp_path)
    captured = {}

    def original_factory(native_config, *_args, **_kwargs):
        captured["config"] = native_config
        return "verified"

    result = make_audit_verified_sample_weighter(
        config,
        policy,
        "cpu",
        None,
        None,
        original_factory=original_factory,
    )

    assert result == "verified"
    assert captured["config"].extra_params == {"fallback_weight": 1.0}


def test_verified_factory_rejects_fitted_contract_in_raw_mode(tmp_path: Path) -> None:
    config, policy, _audit = _raw_fixture(tmp_path)
    config.extra_params["progress_calibration_contract_sha256"] = "6" * 64

    with pytest.raises(RABCAuditError, match="must not specify a fitted"):
        make_audit_verified_sample_weighter(
            config,
            policy,
            "cpu",
            None,
            None,
            original_factory=lambda *_args, **_kwargs: None,
        )


def test_verified_factory_rejects_failed_required_raw_check(tmp_path: Path) -> None:
    config, policy, audit = _raw_fixture(tmp_path)
    audit["gate"]["checks"]["validation_spearman"] = False
    audit_path = Path(config.extra_params["audit_path"])
    audit_path.write_text(json.dumps(audit))
    config.extra_params["audit_sha256"] = _sha256(audit_path)

    with pytest.raises(RABCAuditError, match="validation_spearman"):
        make_audit_verified_sample_weighter(
            config,
            policy,
            "cpu",
            None,
            None,
            original_factory=lambda *_args, **_kwargs: None,
        )


def test_verified_factory_rejects_invalid_raw_tolerance(tmp_path: Path) -> None:
    config, policy, _audit = _raw_fixture(tmp_path)
    config.extra_params["negative_delta_audit_tolerance"] = float("nan")

    with pytest.raises(RABCAuditError, match="finite and nonnegative"):
        make_audit_verified_sample_weighter(
            config,
            policy,
            "cpu",
            None,
            None,
            original_factory=lambda *_args, **_kwargs: None,
        )


def test_verified_factory_rejects_unresolved_or_changed_kappa(tmp_path: Path) -> None:
    config, policy, _audit = _fixture(tmp_path)
    config.kappa = 0.01
    with pytest.raises(RABCAuditError, match="audited train-only kappa"):
        make_audit_verified_sample_weighter(
            config,
            policy,
            "cpu",
            None,
            None,
            original_factory=lambda *_args, **_kwargs: None,
        )


def test_verified_factory_rejects_progress_drift(tmp_path: Path) -> None:
    config, policy, _audit = _fixture(tmp_path)
    Path(config.progress_path).write_bytes(b"changed")
    with pytest.raises(RABCAuditError, match="progress parquet hash"):
        make_audit_verified_sample_weighter(
            config,
            policy,
            "cpu",
            None,
            None,
            original_factory=lambda *_args, **_kwargs: None,
        )


def test_verified_factory_resolves_artifacts_from_dataset_root(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    config, policy, _audit = _fixture(dataset_root)
    config.progress_path = Path(config.progress_path).name
    config.extra_params["audit_path"] = Path(config.extra_params["audit_path"]).name
    captured = {}

    def original_factory(native_config, *_args, **_kwargs):
        captured["config"] = native_config
        return "verified"

    result = make_audit_verified_sample_weighter(
        config,
        policy,
        "cpu",
        str(dataset_root),
        "example/dataset",
        original_factory=original_factory,
    )

    assert result == "verified"
    assert captured["config"].progress_path == str(
        (dataset_root / "sarm_progress.parquet").resolve()
    )
