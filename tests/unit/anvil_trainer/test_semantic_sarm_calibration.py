"""Tests for the offline-only five-stage SARM calibration contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from anvil_trainer.semantic_sarm_annotations import SemanticSARMContract
from anvil_trainer.semantic_sarm_calibration import (
    SemanticSARMCalibrationContract,
    audit_calibrated_semantic_sarm_progress,
    isotonic_projection,
    monotone_pchip,
)
from anvil_trainer.semantic_sarm_progress import expected_raw_progress
from anvil_trainer.semantic_stages import SemanticStageManifest

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "configs/training/semantic_manifests/openarm2_shirt_fold_5stage_v1.json"
REVIEW_PATH = ROOT / "configs/training/semantic_reviews/openarm2_shirt_fold_5stage_review_v1.json"
SARM_CONTRACT_PATH = (
    ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_5stage_v1.json"
)
CALIBRATION_PATH = (
    ROOT / "configs/training/progress_calibrations/openarm2_shirt_fold_sarm_isotonic_v1.json"
)


def _contracts() -> tuple[
    SemanticStageManifest,
    SemanticSARMContract,
    SemanticSARMCalibrationContract,
]:
    manifest = SemanticStageManifest.load(MANIFEST_PATH)
    source = SemanticSARMContract.load(
        SARM_CONTRACT_PATH,
        manifest=manifest,
        review_path=REVIEW_PATH,
    )
    calibration = SemanticSARMCalibrationContract.load(
        CALIBRATION_PATH,
        source_contract=source,
    )
    return manifest, source, calibration


def test_isotonic_pchip_is_finite_monotone_and_smooth() -> None:
    noisy = np.asarray([0.0, 0.2, 0.1, 0.4, 0.39, 0.8, 0.7, 1.0])
    projected = isotonic_projection(noisy)
    smoothed = monotone_pchip(projected, knot_stride_frames=3)

    assert np.isfinite(smoothed).all()
    assert np.all(np.diff(projected) >= 0)
    assert np.all(np.diff(smoothed) >= -1e-12)
    assert smoothed.min() >= 0 and smoothed.max() <= 1


def test_calibration_passes_ideal_curve_and_writes_train_only_lineage(
    tmp_path: Path,
) -> None:
    manifest, source, calibration = _contracts()
    frame = expected_raw_progress(manifest, source).drop(columns="stage_name")
    raw_path = tmp_path / "raw.parquet"
    train_path = tmp_path / "train.parquet"
    frame.to_parquet(raw_path, index=False)

    result = audit_calibrated_semantic_sarm_progress(
        raw_path,
        manifest=manifest,
        source_contract=source,
        calibration_contract=calibration,
        training_progress_path=train_path,
    )

    assert result["schema_version"] == "openarm2.sarm-semantic-progress-audit.v2"
    assert result["gate"]["passed"] is True
    assert result["training_progress"]["frames"] == 29_234
    assert result["rabc"]["train_negative_delta_fraction"] == 0
    assert result["optional_stage_calibrated"]["maximum_absolute_optional_skip_jump"] < 0.1
    metadata = pq.read_metadata(train_path).metadata
    assert metadata[b"noncausal_offline_only"] == b"true"
    assert metadata[b"progress_calibration"] == (
        b"absent_stage_mass_then_episodewise_isotonic_pchip"
    )


def test_small_raw_regressions_are_diagnostic_not_gating(tmp_path: Path) -> None:
    manifest, source, calibration = _contracts()
    frame = expected_raw_progress(manifest, source).drop(columns="stage_name")
    perturb = np.resize(np.asarray([0.0, -0.01, 0.0, 0.01]), len(frame))
    frame["progress_dense"] = np.clip(frame["progress_dense"] + perturb, 0, 1)
    raw_path = tmp_path / "raw.parquet"
    frame.to_parquet(raw_path, index=False)

    result = audit_calibrated_semantic_sarm_progress(
        raw_path,
        manifest=manifest,
        source_contract=source,
        calibration_contract=calibration,
    )

    assert result["gate"]["checks"]["stage_severe_drop_fraction"] is True
    assert result["gate"]["passed"] is True
