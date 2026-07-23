"""Offline monotone calibration for five-stage released-SARM progress.

The released reward artifact is intentionally left untouched. This module
produces a separately versioned, train-only progress artifact for RA-BC.
Calibration is episode-local and non-causal, so it is suitable for weighting
an existing offline dataset but is not an online reward estimator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator

from anvil_trainer.semantic_sarm_annotations import SemanticSARMContract
from anvil_trainer.semantic_sarm_progress import (
    correct_optional_stage_progress,
    expected_raw_progress,
)
from anvil_trainer.semantic_stages import (
    SemanticStageError,
    SemanticStageManifest,
    file_sha256,
)

CALIBRATION_SCHEMA = "openarm2.sarm-progress-calibration.v1"
CALIBRATED_AUDIT_SCHEMA = "openarm2.sarm-semantic-progress-audit.v2"
CALIBRATION_METHOD = "episodewise_isotonic_pchip"
TRAINING_CORRECTION = "absent_stage_mass_then_episodewise_isotonic_pchip"


@dataclass(frozen=True)
class SemanticSARMCalibrationContract:
    """Immutable offline calibration and gate contract."""

    path: Path
    sha256: str
    calibration_id: str
    source_semantic_sarm_contract_sha256: str
    knot_stride_frames: int
    noncausal_offline_only: bool
    progress_gate: Mapping[str, float]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        source_contract: SemanticSARMContract,
    ) -> SemanticSARMCalibrationContract:
        source = Path(path).expanduser().resolve()
        payload = source.read_bytes()
        raw = json.loads(payload)
        if raw.get("schema_version") != CALIBRATION_SCHEMA:
            raise SemanticStageError("Unsupported SARM progress-calibration schema")
        if raw.get("source_semantic_sarm_contract_sha256") != source_contract.sha256:
            raise SemanticStageError("Calibration contract targets a different SARM contract")
        method = raw.get("method", {})
        if method != {
            "kind": CALIBRATION_METHOD,
            "knot_stride_frames": method.get("knot_stride_frames"),
            "noncausal_offline_only": True,
            "preserve_raw_artifact": True,
        }:
            raise SemanticStageError("Calibration method is not the frozen offline method")
        stride = int(method["knot_stride_frames"])
        if stride < 2:
            raise SemanticStageError("Calibration knot stride must be at least two frames")
        gate = {key: float(value) for key, value in raw.get("progress_gate", {}).items()}
        required_gate = {
            "minimum_holdout_spearman",
            "maximum_holdout_mae",
            "severe_negative_delta_tolerance",
            "maximum_stage_severe_drop_fraction",
            "maximum_calibrated_optional_skip_jump",
        }
        if set(gate) != required_gate:
            raise SemanticStageError("Calibration progress-gate keys are invalid")
        if gate["severe_negative_delta_tolerance"] <= 0:
            raise SemanticStageError("Severe negative-delta tolerance must be positive")
        if not 0 <= gate["maximum_stage_severe_drop_fraction"] <= 1:
            raise SemanticStageError("Maximum severe-drop fraction must be in [0, 1]")
        calibration_id = str(raw.get("calibration_id", ""))
        if not calibration_id:
            raise SemanticStageError("Calibration id is required")
        return cls(
            path=source,
            sha256=hashlib.sha256(payload).hexdigest(),
            calibration_id=calibration_id,
            source_semantic_sarm_contract_sha256=source_contract.sha256,
            knot_stride_frames=stride,
            noncausal_offline_only=True,
            progress_gate=gate,
        )


def isotonic_projection(values: np.ndarray) -> np.ndarray:
    """Return the equal-weight least-squares nondecreasing projection."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or not len(source):
        raise SemanticStageError("Isotonic projection requires a non-empty vector")
    if not np.isfinite(source).all():
        raise SemanticStageError("Isotonic projection input contains non-finite values")

    sums: list[float] = []
    counts: list[int] = []
    starts: list[int] = []
    for index, value in enumerate(source):
        sums.append(float(value))
        counts.append(1)
        starts.append(index)
        while len(sums) >= 2 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
            right_sum = sums.pop()
            right_count = counts.pop()
            starts.pop()
            sums[-1] += right_sum
            counts[-1] += right_count

    result = np.empty_like(source)
    for block, start in enumerate(starts):
        end = starts[block + 1] if block + 1 < len(starts) else len(source)
        result[start:end] = sums[block] / counts[block]
    return result


def monotone_pchip(values: np.ndarray, *, knot_stride_frames: int) -> np.ndarray:
    """Smooth a nondecreasing vector on a fixed uniform knot grid."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or not len(source):
        raise SemanticStageError("Monotone PCHIP requires a non-empty vector")
    if np.any(np.diff(source) < -1e-12):
        raise SemanticStageError("Monotone PCHIP input must be nondecreasing")
    if len(source) < 3:
        return source.copy()
    knots = np.arange(0, len(source), knot_stride_frames, dtype=np.int64)
    if knots[-1] != len(source) - 1:
        knots = np.append(knots, len(source) - 1)
    knot_values = isotonic_projection(source[knots])
    if len(knots) == 2:
        result = np.interp(np.arange(len(source)), knots, knot_values)
    else:
        result = PchipInterpolator(knots, knot_values, extrapolate=False)(np.arange(len(source)))
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(np.diff(result) < -1e-10):
        raise SemanticStageError("Monotone PCHIP produced an invalid curve")
    return np.clip(result, 0.0, 1.0)


def calibrate_episode_progress(
    corrected_progress: np.ndarray,
    *,
    knot_stride_frames: int,
) -> np.ndarray:
    """Project and smooth one complete episode without changing episode order."""
    return monotone_pchip(
        isotonic_projection(corrected_progress),
        knot_stride_frames=knot_stride_frames,
    )


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=np.float64)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _split_metrics(
    progress: np.ndarray,
    target: np.ndarray,
    episode_ids: np.ndarray,
    episodes: tuple[int, ...],
) -> dict[str, Any]:
    mask = np.isin(episode_ids, np.asarray(episodes, dtype=np.int64))
    return {
        "episodes": list(episodes),
        "frames": int(mask.sum()),
        "progress_mae_vs_temporal_target": float(np.abs(progress[mask] - target[mask]).mean()),
        "progress_spearman_vs_temporal_target": _rank_correlation(progress[mask], target[mask]),
    }


def _write_training_progress(
    frame: pd.DataFrame,
    path: Path,
    *,
    source_progress_sha256: str,
    manifest: SemanticStageManifest,
    source_contract: SemanticSARMContract,
    calibration_contract: SemanticSARMCalibrationContract,
) -> None:
    metadata = {
        b"source_progress_sha256": source_progress_sha256.encode(),
        b"semantic_manifest_sha256": manifest.sha256.encode(),
        b"semantic_sarm_contract_sha256": source_contract.sha256.encode(),
        b"progress_calibration_contract_sha256": calibration_contract.sha256.encode(),
        b"split": b"train",
        b"progress_calibration": TRAINING_CORRECTION.encode(),
        b"noncausal_offline_only": b"true",
    }
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        pq.write_table(table, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_calibrated_semantic_sarm_progress(
    progress_path: str | Path,
    *,
    manifest: SemanticStageManifest,
    source_contract: SemanticSARMContract,
    calibration_contract: SemanticSARMCalibrationContract,
    chunk_size: int = 30,
    training_progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calibrate full progress and emit a provenance-locked train-only artifact."""
    if chunk_size <= 0:
        raise SemanticStageError("chunk_size must be positive")
    path = Path(progress_path).expanduser().resolve()
    frame = pd.read_parquet(path).sort_values("index", kind="stable").reset_index(drop=True)
    expected = expected_raw_progress(manifest, source_contract)
    if len(frame) != len(expected):
        raise SemanticStageError("SARM progress does not cover every expected frame")
    for column in ("index", "episode_index", "frame_index"):
        if not np.array_equal(
            frame[column].to_numpy(dtype=np.int64),
            expected[column].to_numpy(dtype=np.int64),
        ):
            raise SemanticStageError(f"SARM progress {column} does not match the dataset")

    progress = frame["progress_dense"].to_numpy(dtype=np.float64)
    if not np.isfinite(progress).all() or np.any(progress < 0) or np.any(progress > 1):
        raise SemanticStageError("SARM progress must be finite and within [0, 1]")
    episode_ids = expected["episode_index"].to_numpy(dtype=np.int64)
    local_frames = expected["frame_index"].to_numpy(dtype=np.int64)
    stage_names = expected["stage_name"].to_numpy(dtype=object)
    raw_target = expected["progress_dense"].to_numpy(dtype=np.float64)
    corrected = correct_optional_stage_progress(
        progress,
        episode_ids,
        stage_names,
        manifest=manifest,
        contract=source_contract,
    )
    corrected_target = correct_optional_stage_progress(
        raw_target,
        episode_ids,
        stage_names,
        manifest=manifest,
        contract=source_contract,
    )
    calibrated = np.empty_like(corrected)
    for episode_index in range(33):
        mask = episode_ids == episode_index
        calibrated[mask] = calibrate_episode_progress(
            corrected[mask],
            knot_stride_frames=calibration_contract.knot_stride_frames,
        )

    split_episodes = {
        "train": source_contract.train_episodes,
        "validation": source_contract.validation_episodes,
        "test": source_contract.test_episodes,
    }
    splits = {
        name: _split_metrics(calibrated, corrected_target, episode_ids, episodes)
        for name, episodes in split_episodes.items()
    }
    tolerance = calibration_contract.progress_gate["severe_negative_delta_tolerance"]
    stage_metrics: dict[str, Any] = {}
    for stage_name in source_contract.dense_stage_order:
        mask = stage_names == stage_name
        corrected_deltas: list[float] = []
        calibrated_deltas: list[float] = []
        for episode_index in range(33):
            episode_mask = mask & (episode_ids == episode_index)
            corrected_deltas.extend(np.diff(corrected[episode_mask]).tolist())
            calibrated_deltas.extend(np.diff(calibrated[episode_mask]).tolist())
        raw_delta = np.asarray(corrected_deltas, dtype=np.float64)
        delta = np.asarray(calibrated_deltas, dtype=np.float64)
        severe = int((delta < -tolerance).sum())
        stage_metrics[stage_name] = {
            "frames": int(mask.sum()),
            "corrected_raw_negative_delta_fraction": float(np.mean(raw_delta < 0)),
            "corrected_raw_drop_below_negative_0_05_fraction": float(np.mean(raw_delta < -0.05)),
            "calibrated_mean_frame_delta": float(delta.mean()),
            "calibrated_severe_drop_tolerance": tolerance,
            "calibrated_severe_drops": severe,
            "calibrated_comparisons": len(delta),
            "calibrated_severe_drop_fraction": severe / max(len(delta), 1),
        }

    calibrated_lookup = {
        (int(episode), int(local)): float(value)
        for episode, local, value in zip(episode_ids, local_frames, calibrated, strict=True)
    }
    optional_boundaries: list[dict[str, Any]] = []
    for episode in manifest.episodes:
        for stage in episode.stages:
            if stage.name not in manifest.optional_stages or stage.present:
                continue
            boundary = stage.start_frame
            before = calibrated_lookup[(episode.episode_index, boundary - 1)]
            after = calibrated_lookup[(episode.episode_index, boundary)]
            optional_boundaries.append(
                {
                    "episode_index": episode.episode_index,
                    "absent_stage": stage.name,
                    "boundary_frame": boundary,
                    "calibrated_jump": after - before,
                    "absolute_calibrated_jump": abs(after - before),
                }
            )
    max_optional_jump = max(
        (item["absolute_calibrated_jump"] for item in optional_boundaries),
        default=0.0,
    )

    train_mask = np.isin(episode_ids, np.asarray(source_contract.train_episodes, dtype=np.int64))
    chunk_deltas: list[float] = []
    for episode_index in source_contract.train_episodes:
        values = calibrated[episode_ids == episode_index]
        for local_index in range(len(values)):
            future = min(local_index + chunk_size, len(values) - 1)
            chunk_deltas.append(float(values[future] - values[local_index]))
    delta_array = np.asarray(chunk_deltas, dtype=np.float64)
    nonnegative = delta_array[delta_array >= 0]
    if not len(nonnegative):
        raise SemanticStageError("Calibrated train progress has no nonnegative deltas")
    recommended_kappa = float(np.quantile(nonnegative, 0.95))
    if not math.isfinite(recommended_kappa) or recommended_kappa <= 0:
        raise SemanticStageError("Calibrated RA-BC kappa is not positive")

    training_progress: dict[str, Any] | None = None
    if training_progress_path is not None:
        training_path = Path(training_progress_path).expanduser().resolve()
        training_frame = frame.loc[train_mask].copy().reset_index(drop=True)
        training_frame["progress_dense"] = calibrated[train_mask]
        if len(training_frame) != 29_234:
            raise SemanticStageError("Calibrated training progress split is incomplete")
        _write_training_progress(
            training_frame,
            training_path,
            source_progress_sha256=file_sha256(path),
            manifest=manifest,
            source_contract=source_contract,
            calibration_contract=calibration_contract,
        )
        training_progress = {
            "path": str(training_path),
            "sha256": file_sha256(training_path),
            "frames": len(training_frame),
            "episodes": list(source_contract.train_episodes),
            "calibration": TRAINING_CORRECTION,
        }

    gate = calibration_contract.progress_gate
    checks = {
        "validation_spearman": splits["validation"]["progress_spearman_vs_temporal_target"]
        >= gate["minimum_holdout_spearman"],
        "test_spearman": splits["test"]["progress_spearman_vs_temporal_target"]
        >= gate["minimum_holdout_spearman"],
        "validation_mae": splits["validation"]["progress_mae_vs_temporal_target"]
        <= gate["maximum_holdout_mae"],
        "test_mae": splits["test"]["progress_mae_vs_temporal_target"]
        <= gate["maximum_holdout_mae"],
        "stage_severe_drop_fraction": max(
            value["calibrated_severe_drop_fraction"] for value in stage_metrics.values()
        )
        <= gate["maximum_stage_severe_drop_fraction"],
        "calibrated_optional_skip_jump": max_optional_jump
        <= gate["maximum_calibrated_optional_skip_jump"],
        "positive_finite_kappa": math.isfinite(recommended_kappa) and recommended_kappa > 0,
        "train_only_rows": int(train_mask.sum()) == 29_234,
    }
    return {
        "schema_version": CALIBRATED_AUDIT_SCHEMA,
        "progress_path": str(path),
        "progress_sha256": file_sha256(path),
        "semantic_manifest_sha256": manifest.sha256,
        "semantic_sarm_contract_sha256": source_contract.sha256,
        "progress_calibration_contract_sha256": calibration_contract.sha256,
        "calibration_id": calibration_contract.calibration_id,
        "calibration_scope": "offline_train_weighting_only",
        "chunk_size": chunk_size,
        "splits": splits,
        "training_progress": training_progress,
        "calibration": {
            "method": CALIBRATION_METHOD,
            "knot_stride_frames": calibration_contract.knot_stride_frames,
            "noncausal_offline_only": True,
            "mean_absolute_adjustment_vs_optional_corrected_raw": float(
                np.abs(calibrated - corrected).mean()
            ),
        },
        "stages": stage_metrics,
        "optional_stage_calibrated": {
            "boundaries": optional_boundaries,
            "maximum_absolute_optional_skip_jump": max_optional_jump,
        },
        "rabc": {
            "kappa_rule": "95th percentile of nonnegative calibrated train deltas",
            "recommended_kappa": recommended_kappa,
            "train_delta_mean": float(delta_array.mean()),
            "train_delta_std": float(delta_array.std()),
            "train_negative_delta_fraction": float(np.mean(delta_array < 0)),
        },
        "gate": {
            "thresholds": dict(gate),
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def write_calibrated_progress_audit(result: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
