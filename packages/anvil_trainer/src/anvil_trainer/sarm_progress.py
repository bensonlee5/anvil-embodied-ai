"""Audit native SARM progress labels against the frozen OpenARM2 contract."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from anvil_trainer.priority_sampling import PriorityManifest
from anvil_trainer.sarm_annotations import SARMAnnotationContract, SARMAnnotationError


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=np.float64)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _expected_progress(
    manifest: PriorityManifest,
    contract: SARMAnnotationContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = int(manifest.dataset["frames"])
    target = np.empty(frame_count, dtype=np.float64)
    episode_ids = np.empty(frame_count, dtype=np.int64)
    local_frames = np.empty(frame_count, dtype=np.int64)
    stage_names = np.empty(frame_count, dtype=object)
    cursor = 0
    cumulative = 0.0
    stage_offsets: dict[str, float] = {}
    for stage in contract.dense_stage_order:
        stage_offsets[stage] = cumulative
        cumulative += contract.temporal_proportions[stage]
    for episode in manifest.episodes:
        for stage in episode.stages:
            length = stage.end_frame - stage.start_frame
            denominator = max(length - 1, 1)
            local = np.arange(length, dtype=np.float64)
            values = stage_offsets[stage.name] + contract.temporal_proportions[stage.name] * (
                local / denominator
            )
            start = cursor + stage.start_frame
            end = cursor + stage.end_frame
            target[start:end] = values
            episode_ids[start:end] = episode.episode_index
            local_frames[start:end] = np.arange(stage.start_frame, stage.end_frame)
            stage_names[start:end] = stage.name
        cursor += episode.frame_count
    return target, episode_ids, local_frames, stage_names


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def audit_sarm_progress(
    progress_path: str | Path,
    *,
    manifest: PriorityManifest,
    contract: SARMAnnotationContract,
    chunk_size: int = 30,
    training_progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a full-frame progress parquet and summarize reward behavior."""
    if chunk_size <= 0:
        raise SARMAnnotationError("chunk_size must be positive")
    path = Path(progress_path).expanduser().resolve()
    try:
        frame = pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise SARMAnnotationError(f"SARM progress parquet not found: {path}") from exc
    required = {"index", "episode_index", "frame_index", "progress_dense"}
    missing = required - set(frame.columns)
    if missing:
        raise SARMAnnotationError(f"SARM progress parquet is missing columns: {sorted(missing)}")
    frame = frame.sort_values("index", kind="stable").reset_index(drop=True)
    expected_frames = int(manifest.dataset["frames"])
    if len(frame) != expected_frames:
        raise SARMAnnotationError(
            f"SARM progress rows {len(frame)} != dataset frames {expected_frames}"
        )
    indices = frame["index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(expected_frames, dtype=np.int64)):
        raise SARMAnnotationError("SARM progress indices must cover every global frame exactly once")
    progress = frame["progress_dense"].to_numpy(dtype=np.float64)
    if not np.isfinite(progress).all():
        raise SARMAnnotationError("SARM progress contains non-finite values")
    if np.any(progress < -1e-6) or np.any(progress > 1.0 + 1e-6):
        raise SARMAnnotationError("SARM progress must remain within [0, 1]")

    target, episode_ids, local_frames, stage_names = _expected_progress(manifest, contract)
    actual_episode_ids = frame["episode_index"].to_numpy(dtype=np.int64)
    actual_local_frames = frame["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_episode_ids, episode_ids):
        raise SARMAnnotationError("SARM progress episode_index order does not match the dataset")
    if not np.array_equal(actual_local_frames, local_frames):
        raise SARMAnnotationError("SARM progress frame_index order does not match the dataset")

    absolute_error = np.abs(progress - target)
    splits = {
        "train": contract.train_episodes,
        "validation": contract.validation_episodes,
        "test": contract.test_episodes,
    }
    split_metrics: dict[str, Any] = {}
    for name, episodes in splits.items():
        mask = np.isin(episode_ids, np.asarray(episodes))
        split_metrics[name] = {
            "episodes": list(episodes),
            "frames": int(mask.sum()),
            "progress_mae_vs_temporal_target": float(absolute_error[mask].mean()),
            "progress_spearman_vs_temporal_target": _rank_correlation(
                progress[mask], target[mask]
            ),
        }

    stage_metrics: dict[str, Any] = {}
    for stage in contract.dense_stage_order:
        mask = stage_names == stage
        diffs: list[float] = []
        violations = 0
        comparisons = 0
        for episode in manifest.episodes:
            episode_mask = mask & (episode_ids == episode.episode_index)
            episode_progress = progress[episode_mask]
            episode_diffs = np.diff(episode_progress)
            diffs.extend(episode_diffs.tolist())
            violations += int((episode_diffs < -1e-6).sum())
            comparisons += len(episode_diffs)
        stage_metrics[stage] = {
            "frames": int(mask.sum()),
            "progress_mae_vs_temporal_target": float(absolute_error[mask].mean()),
            "progress_spearman_vs_temporal_target": _rank_correlation(
                progress[mask], target[mask]
            ),
            "mean_frame_delta": _mean(diffs),
            "monotonicity_violations": violations,
            "monotonicity_violation_rate": violations / max(comparisons, 1),
        }

    lookup = {
        (int(row.episode_index), int(row.frame_index)): float(row.progress_dense)
        for row in frame.itertuples(index=False)
    }
    repeated_events: list[dict[str, Any]] = []
    for episode in manifest.episodes:
        for event in episode.repeated_grasps:
            close = lookup[(episode.episode_index, event.close_frame)]
            reopen = lookup[(episode.episode_index, event.reopen_frame)]
            retry = lookup[(episode.episode_index, min(event.retry_frame, episode.frame_count - 1))]
            future_frame = min(event.retry_frame + chunk_size, episode.frame_count - 1)
            repeated_events.append(
                {
                    "episode_index": episode.episode_index,
                    "stage": event.stage,
                    "gripper": event.gripper,
                    "close_to_reopen_delta": reopen - close,
                    "reopen_to_retry_delta": retry - reopen,
                    "post_retry_chunk_delta": lookup[
                        (episode.episode_index, future_frame)
                    ]
                    - retry,
                }
            )

    quality_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cursor = 0
    for episode in manifest.episodes:
        for stage in episode.stages:
            start = cursor + stage.start_frame
            end = cursor + stage.end_frame
            quality_errors[stage.name][str(stage.quality_score)].extend(
                absolute_error[start:end].tolist()
            )
        cursor += episode.frame_count
    quality_metrics = {
        stage: {
            score: {"frames": len(values), "progress_mae": float(np.mean(values))}
            for score, values in sorted(scores.items())
        }
        for stage, scores in quality_errors.items()
    }

    chunk_deltas: list[float] = []
    for episode_index in contract.train_episodes:
        episode = manifest.episodes[episode_index]
        for local_frame in range(episode.frame_count):
            future = min(local_frame + chunk_size, episode.frame_count - 1)
            chunk_deltas.append(
                lookup[(episode_index, future)] - lookup[(episode_index, local_frame)]
            )
    delta_array = np.asarray(chunk_deltas, dtype=np.float64)
    if not np.isfinite(delta_array).all():
        raise SARMAnnotationError("Training progress deltas contain non-finite values")
    nonnegative = delta_array[delta_array >= 0]
    if len(nonnegative) == 0:
        raise SARMAnnotationError("No nonnegative training progress deltas are available")
    recommended_kappa = float(np.quantile(nonnegative, 0.95))
    if not math.isfinite(recommended_kappa) or recommended_kappa <= 0:
        raise SARMAnnotationError("The 95th-percentile RA-BC kappa is not positive")

    training_progress: dict[str, Any] | None = None
    if training_progress_path is not None:
        training_path = Path(training_progress_path).expanduser().resolve()
        if training_path == path:
            raise SARMAnnotationError("Training progress path must differ from full progress path")
        train_mask = frame["episode_index"].isin(contract.train_episodes)
        training_frame = frame.loc[train_mask].reset_index(drop=True)
        expected_training_frames = sum(
            manifest.episodes[index].frame_count for index in contract.train_episodes
        )
        if len(training_frame) != expected_training_frames:
            raise SARMAnnotationError(
                f"Training progress rows {len(training_frame)} != expected {expected_training_frames}"
            )
        _write_training_progress(
            training_frame,
            training_path,
            source_progress_sha256=_file_sha256(path),
            manifest=manifest,
            contract=contract,
        )
        training_progress = {
            "path": str(training_path),
            "sha256": _file_sha256(training_path),
            "frames": len(training_frame),
            "episodes": list(contract.train_episodes),
        }

    smoothing_diagnostics: list[float] = []
    for episode in manifest.episodes:
        start = int(episode.smoothing["review_start_frame"])
        end = int(episode.smoothing["review_end_frame"])
        for local_frame in range(start, end):
            future = min(local_frame + chunk_size, episode.frame_count - 1)
            smoothing_diagnostics.append(
                lookup[(episode.episode_index, future)]
                - lookup[(episode.episode_index, local_frame)]
            )

    return {
        "schema_version": "openarm2.sarm-progress-audit.v1",
        "progress_path": str(path),
        "progress_sha256": _file_sha256(path),
        "training_progress": training_progress,
        "priority_manifest_sha256": manifest.sha256,
        "sarm_contract_sha256": contract.sha256,
        "chunk_size": chunk_size,
        "global": {
            "frames": expected_frames,
            "progress_min": float(progress.min()),
            "progress_max": float(progress.max()),
            "progress_mae_vs_temporal_target": float(absolute_error.mean()),
            "progress_spearman_vs_temporal_target": _rank_correlation(progress, target),
        },
        "splits": split_metrics,
        "stages": stage_metrics,
        "quality_strata": quality_metrics,
        "repeated_grasps": {
            "events": repeated_events,
            "count": len(repeated_events),
            "mean_close_to_reopen_delta": _mean(
                [event["close_to_reopen_delta"] for event in repeated_events]
            ),
            "mean_reopen_to_retry_delta": _mean(
                [event["reopen_to_retry_delta"] for event in repeated_events]
            ),
            "mean_post_retry_chunk_delta": _mean(
                [event["post_retry_chunk_delta"] for event in repeated_events]
            ),
        },
        "smoothing_review_windows": {
            "warning": "coarse diagnostic windows; not frame-accurate masks or reward overrides",
            "chunk_deltas": len(smoothing_diagnostics),
            "mean_chunk_delta": _mean(smoothing_diagnostics),
            "negative_chunk_delta_fraction": float(
                np.mean(np.asarray(smoothing_diagnostics) < 0)
            ),
        },
        "rabc": {
            "kappa_rule": "95th percentile of nonnegative train-episode progress deltas",
            "recommended_kappa": recommended_kappa,
            "train_delta_mean": float(delta_array.mean()),
            "train_delta_std": float(delta_array.std()),
            "train_negative_delta_fraction": float(np.mean(delta_array < 0)),
        },
    }


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_training_progress(
    frame: pd.DataFrame,
    path: Path,
    *,
    source_progress_sha256: str,
    manifest: PriorityManifest,
    contract: SARMAnnotationContract,
) -> None:
    """Write the exact train split consumed by native RA-BC statistics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"source_progress_sha256": source_progress_sha256.encode(),
        b"priority_manifest_sha256": manifest.sha256.encode(),
        b"sarm_contract_sha256": contract.sha256.encode(),
        b"split": b"train",
    }
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(metadata)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        pq.write_table(table, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_progress_audit(result: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
