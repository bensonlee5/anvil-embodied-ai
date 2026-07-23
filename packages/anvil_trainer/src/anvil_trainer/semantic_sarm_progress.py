"""Audit five-stage released-SARM progress before bounded RA-BC training."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from anvil_trainer.semantic_sarm_annotations import SemanticSARMContract
from anvil_trainer.semantic_stages import (
    SemanticStageError,
    SemanticStageManifest,
    file_sha256,
)

PROGRESS_AUDIT_SCHEMA = "openarm2.sarm-semantic-progress-audit.v1"
TRAINING_CORRECTION = "remove_absent_stage_mass_then_renormalize_per_episode"


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=np.float64)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def expected_raw_progress(
    manifest: SemanticStageManifest,
    contract: SemanticSARMContract,
) -> pd.DataFrame:
    """Return the global-stage temporal target emitted by released SARM."""
    rows: list[dict[str, Any]] = []
    global_index = 0
    offsets: dict[str, float] = {}
    cumulative = 0.0
    for stage_name in contract.dense_stage_order:
        offsets[stage_name] = cumulative
        cumulative += contract.temporal_proportions[stage_name]

    for episode in manifest.episodes:
        for stage in episode.stages:
            length = stage.end_frame - stage.start_frame
            if length == 0:
                continue
            denominator = max(length - 1, 1)
            mass = contract.temporal_proportions[stage.name]
            for position, frame_index in enumerate(range(stage.start_frame, stage.end_frame)):
                rows.append(
                    {
                        "index": global_index + frame_index,
                        "episode_index": episode.episode_index,
                        "frame_index": frame_index,
                        "stage_name": stage.name,
                        "progress_dense": offsets[stage.name] + mass * position / denominator,
                    }
                )
        global_index += episode.frame_count
    frame = pd.DataFrame(rows).sort_values("index", kind="stable").reset_index(drop=True)
    expected_indices = np.arange(int(manifest.dataset["frames"]), dtype=np.int64)
    if not np.array_equal(frame["index"].to_numpy(dtype=np.int64), expected_indices):
        raise SemanticStageError("Semantic stages do not assign every dataset frame exactly once")
    return frame


def _episode_correction(
    manifest: SemanticStageManifest,
    contract: SemanticSARMContract,
    episode_index: int,
) -> tuple[float, dict[str, float]]:
    episode = manifest.episode(episode_index)
    absent = {stage.name for stage in episode.stages if not stage.present}
    absent_mass = sum(contract.temporal_proportions[name] for name in absent)
    denominator = 1.0 - absent_mass
    if not math.isfinite(denominator) or denominator <= 0:
        raise SemanticStageError(f"Episode {episode_index} has no usable reward mass")
    removed_before: dict[str, float] = {}
    removed = 0.0
    for name in contract.dense_stage_order:
        removed_before[name] = removed
        if name in absent:
            removed += contract.temporal_proportions[name]
    return denominator, removed_before


def correct_optional_stage_progress(
    progress: np.ndarray,
    episode_ids: np.ndarray,
    stage_names: np.ndarray,
    *,
    manifest: SemanticStageManifest,
    contract: SemanticSARMContract,
) -> np.ndarray:
    """Remove absent-stage mass without rewriting the released-SARM artifact."""
    corrected = np.empty_like(progress, dtype=np.float64)
    for episode_index in range(33):
        episode_mask = episode_ids == episode_index
        denominator, removed_before = _episode_correction(manifest, contract, episode_index)
        for stage_name in contract.dense_stage_order:
            mask = episode_mask & (stage_names == stage_name)
            corrected[mask] = (progress[mask] - removed_before[stage_name]) / denominator
    return np.clip(corrected, 0.0, 1.0)


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
    contract: SemanticSARMContract,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"source_progress_sha256": source_progress_sha256.encode(),
        b"semantic_manifest_sha256": manifest.sha256.encode(),
        b"semantic_sarm_contract_sha256": contract.sha256.encode(),
        b"split": b"train",
        b"optional_stage_correction": TRAINING_CORRECTION.encode(),
    }
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(metadata)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        pq.write_table(table, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_semantic_sarm_progress(
    progress_path: str | Path,
    *,
    manifest: SemanticStageManifest,
    contract: SemanticSARMContract,
    chunk_size: int = 30,
    training_progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit raw released-SARM output and write corrected train-only progress."""
    if chunk_size <= 0:
        raise SemanticStageError("chunk_size must be positive")
    path = Path(progress_path).expanduser().resolve()
    try:
        frame = pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise SemanticStageError(f"SARM progress parquet not found: {path}") from exc
    required = {"index", "episode_index", "frame_index", "progress_dense"}
    missing = required - set(frame.columns)
    if missing:
        raise SemanticStageError(f"SARM progress is missing columns: {sorted(missing)}")
    frame = frame.sort_values("index", kind="stable").reset_index(drop=True)
    expected = expected_raw_progress(manifest, contract)
    if len(frame) != len(expected):
        raise SemanticStageError(
            f"SARM progress rows {len(frame)} != dataset frames {len(expected)}"
        )
    for column in ("index", "episode_index", "frame_index"):
        if not np.array_equal(
            frame[column].to_numpy(dtype=np.int64),
            expected[column].to_numpy(dtype=np.int64),
        ):
            raise SemanticStageError(f"SARM progress {column} does not match the dataset")

    progress = frame["progress_dense"].to_numpy(dtype=np.float64)
    if not np.isfinite(progress).all():
        raise SemanticStageError("SARM progress contains non-finite values")
    if np.any(progress < -1e-6) or np.any(progress > 1.0 + 1e-6):
        raise SemanticStageError("SARM progress must remain within [0, 1]")
    episode_ids = expected["episode_index"].to_numpy(dtype=np.int64)
    local_frames = expected["frame_index"].to_numpy(dtype=np.int64)
    stage_names = expected["stage_name"].to_numpy(dtype=object)
    raw_target = expected["progress_dense"].to_numpy(dtype=np.float64)
    corrected = correct_optional_stage_progress(
        progress,
        episode_ids,
        stage_names,
        manifest=manifest,
        contract=contract,
    )
    corrected_target = correct_optional_stage_progress(
        raw_target,
        episode_ids,
        stage_names,
        manifest=manifest,
        contract=contract,
    )

    split_episodes = {
        "train": contract.train_episodes,
        "validation": contract.validation_episodes,
        "test": contract.test_episodes,
    }
    raw_splits = {
        name: _split_metrics(progress, raw_target, episode_ids, episodes)
        for name, episodes in split_episodes.items()
    }
    corrected_splits = {
        name: _split_metrics(corrected, corrected_target, episode_ids, episodes)
        for name, episodes in split_episodes.items()
    }

    stage_metrics: dict[str, Any] = {}
    for stage_name in contract.dense_stage_order:
        mask = stage_names == stage_name
        violations = 0
        comparisons = 0
        deltas: list[float] = []
        for episode_index in range(33):
            values = corrected[mask & (episode_ids == episode_index)]
            episode_deltas = np.diff(values)
            violations += int((episode_deltas < -1e-6).sum())
            comparisons += len(episode_deltas)
            deltas.extend(episode_deltas.tolist())
        stage_metrics[stage_name] = {
            "present_episodes": sum(
                int(next(s for s in episode.stages if s.name == stage_name).present)
                for episode in manifest.episodes
            ),
            "frames": int(mask.sum()),
            "corrected_progress_mae_vs_temporal_target": float(
                np.abs(corrected[mask] - corrected_target[mask]).mean()
            ),
            "corrected_progress_spearman_vs_temporal_target": _rank_correlation(
                corrected[mask], corrected_target[mask]
            ),
            "mean_frame_delta": float(np.mean(deltas)) if deltas else None,
            "monotonicity_violations": violations,
            "monotonicity_comparisons": comparisons,
            "monotonicity_violation_rate": violations / max(comparisons, 1),
        }

    corrected_lookup = {
        (int(episode), int(local)): float(value)
        for episode, local, value in zip(episode_ids, local_frames, corrected, strict=True)
    }
    optional_boundaries: list[dict[str, Any]] = []
    for episode in manifest.episodes:
        for stage in episode.stages:
            if stage.name not in manifest.optional_stages or stage.present:
                continue
            boundary = stage.start_frame
            if boundary <= 0 or boundary >= episode.frame_count:
                raise SemanticStageError(
                    f"Episode {episode.episode_index} optional boundary is not internal"
                )
            before = corrected_lookup[(episode.episode_index, boundary - 1)]
            after = corrected_lookup[(episode.episode_index, boundary)]
            optional_boundaries.append(
                {
                    "episode_index": episode.episode_index,
                    "absent_stage": stage.name,
                    "boundary_frame": boundary,
                    "corrected_jump": after - before,
                    "absolute_corrected_jump": abs(after - before),
                }
            )
    max_optional_jump = max(
        (item["absolute_corrected_jump"] for item in optional_boundaries), default=0.0
    )

    train_mask = np.isin(episode_ids, np.asarray(contract.train_episodes, dtype=np.int64))
    chunk_deltas: list[float] = []
    for episode_index in contract.train_episodes:
        values = corrected[episode_ids == episode_index]
        for local_index in range(len(values)):
            future = min(local_index + chunk_size, len(values) - 1)
            chunk_deltas.append(float(values[future] - values[local_index]))
    delta_array = np.asarray(chunk_deltas, dtype=np.float64)
    nonnegative = delta_array[delta_array >= 0]
    if not np.isfinite(delta_array).all() or not len(nonnegative):
        raise SemanticStageError("Corrected train progress has no finite nonnegative deltas")
    recommended_kappa = float(np.quantile(nonnegative, 0.95))
    if not math.isfinite(recommended_kappa) or recommended_kappa <= 0:
        raise SemanticStageError("Corrected RA-BC kappa is not positive")

    training_progress: dict[str, Any] | None = None
    if training_progress_path is not None:
        training_path = Path(training_progress_path).expanduser().resolve()
        if training_path == path:
            raise SemanticStageError("Training progress path must differ from raw progress path")
        training_frame = frame.loc[train_mask].copy().reset_index(drop=True)
        training_frame["progress_dense"] = corrected[train_mask]
        expected_train_frames = sum(
            manifest.episode(index).frame_count for index in contract.train_episodes
        )
        if len(training_frame) != expected_train_frames:
            raise SemanticStageError("Corrected training progress split is incomplete")
        _write_training_progress(
            training_frame,
            training_path,
            source_progress_sha256=file_sha256(path),
            manifest=manifest,
            contract=contract,
        )
        training_progress = {
            "path": str(training_path),
            "sha256": file_sha256(training_path),
            "frames": len(training_frame),
            "episodes": list(contract.train_episodes),
            "correction": TRAINING_CORRECTION,
        }

    checks = {
        "validation_spearman": corrected_splits["validation"][
            "progress_spearman_vs_temporal_target"
        ]
        >= contract.progress_gate["minimum_holdout_spearman"],
        "test_spearman": corrected_splits["test"]["progress_spearman_vs_temporal_target"]
        >= contract.progress_gate["minimum_holdout_spearman"],
        "validation_mae": corrected_splits["validation"]["progress_mae_vs_temporal_target"]
        <= contract.progress_gate["maximum_holdout_mae"],
        "test_mae": corrected_splits["test"]["progress_mae_vs_temporal_target"]
        <= contract.progress_gate["maximum_holdout_mae"],
        "stage_monotonicity": max(
            value["monotonicity_violation_rate"] for value in stage_metrics.values()
        )
        <= contract.progress_gate["maximum_stage_monotonicity_violation_rate"],
        "corrected_optional_skip_jump": max_optional_jump
        <= contract.progress_gate["maximum_corrected_optional_skip_jump"],
        "positive_finite_kappa": math.isfinite(recommended_kappa) and recommended_kappa > 0,
        "train_only_rows": int(train_mask.sum()) == 29_234,
    }

    return {
        "schema_version": PROGRESS_AUDIT_SCHEMA,
        "progress_path": str(path),
        "progress_sha256": file_sha256(path),
        "training_progress": training_progress,
        "semantic_manifest_sha256": manifest.sha256,
        "semantic_sarm_contract_sha256": contract.sha256,
        "chunk_size": chunk_size,
        "raw_released_sarm": {
            "progress_min": float(progress.min()),
            "progress_max": float(progress.max()),
            "splits": raw_splits,
        },
        "optional_stage_corrected": {
            "correction": TRAINING_CORRECTION,
            "progress_min": float(corrected.min()),
            "progress_max": float(corrected.max()),
            "splits": corrected_splits,
            "optional_skip_boundaries": optional_boundaries,
            "maximum_absolute_optional_skip_jump": max_optional_jump,
        },
        "stages": stage_metrics,
        "rabc": {
            "kappa_rule": "95th percentile of nonnegative corrected train deltas",
            "recommended_kappa": recommended_kappa,
            "train_delta_mean": float(delta_array.mean()),
            "train_delta_std": float(delta_array.std()),
            "train_negative_delta_fraction": float(np.mean(delta_array < 0)),
        },
        "gate": {
            "thresholds": dict(contract.progress_gate),
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def write_semantic_progress_audit(result: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
