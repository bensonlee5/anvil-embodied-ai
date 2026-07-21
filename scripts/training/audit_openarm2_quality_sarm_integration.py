#!/usr/bin/env python3
"""Audit the combined blind-quality sampler and native SARM RA-BC objective."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anvil_trainer.priority_sampling import PriorityManifest
from anvil_trainer.sarm_annotations import SARMAnnotationContract

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json"
)
DEFAULT_CONTRACT = ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_v2.json"
DEFAULT_DATASET = ROOT / "datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-v1"
DEFAULT_PROGRESS = DEFAULT_DATASET / "sarm_progress_train_v2.parquet"
DEFAULT_PROGRESS_AUDIT = DEFAULT_DATASET / "sarm_progress_audit_v2.json"
DEFAULT_OUTPUT = (
    ROOT
    / "configs/training/quality_sarm_audits/openarm2_shirt_fold_quality_sarm_v2.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_sample_size(probabilities: np.ndarray) -> float:
    return float(1.0 / np.square(probabilities).sum())


def _rabc_weights(
    deltas: np.ndarray,
    *,
    kappa: float,
    epsilon: float,
) -> tuple[np.ndarray, float, float]:
    """Mirror LeRobot RABCWeights._compute_global_stats/_compute_weights."""
    delta_mean = max(float(deltas.mean()), 0.0)
    delta_std = max(float(deltas.std()), epsilon)
    lower_bound = delta_mean - 2 * delta_std
    soft = np.clip((deltas - lower_bound) / (4 * delta_std + epsilon), 0.0, 1.0)
    weights = np.zeros_like(deltas, dtype=np.float64)
    weights[deltas > kappa] = 1.0
    moderate = (deltas >= 0) & (deltas <= kappa)
    weights[moderate] = soft[moderate]
    return weights, delta_mean, delta_std


def _stage_summary(
    stage_names: np.ndarray,
    quality_scores: np.ndarray,
    priority_probability: np.ndarray,
    combined_probability: np.ndarray,
    rabc_weights: np.ndarray,
    stage_order: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in stage_order:
        stage_mask = stage_names == stage
        quality: dict[str, Any] = {}
        for score in sorted(set(quality_scores[stage_mask].tolist())):
            mask = stage_mask & (quality_scores == score)
            quality[str(score)] = {
                "frames": int(mask.sum()),
                "manual_sampling_mass": float(priority_probability[mask].sum()),
                "combined_expected_mass": float(combined_probability[mask].sum()),
                "mean_raw_rabc_weight": float(rabc_weights[mask].mean()),
            }
        result[stage] = {
            "frames": int(stage_mask.sum()),
            "manual_sampling_mass": float(priority_probability[stage_mask].sum()),
            "combined_expected_mass": float(combined_probability[stage_mask].sum()),
            "mean_raw_rabc_weight": float(rabc_weights[stage_mask].mean()),
            "zero_rabc_weight_fraction": float(np.mean(rabc_weights[stage_mask] == 0)),
            "quality_strata": quality,
        }
    return result


def audit_integration(
    *,
    manifest_path: Path,
    contract_path: Path,
    progress_path: Path,
    progress_audit_path: Path,
) -> dict[str, Any]:
    manifest = PriorityManifest.load(manifest_path)
    contract = SARMAnnotationContract.load(contract_path, priority_manifest=manifest)
    progress_audit = json.loads(progress_audit_path.read_text())
    if progress_audit.get("schema_version") != "openarm2.sarm-progress-audit.v1":
        raise ValueError("Progress audit does not use openarm2.sarm-progress-audit.v1")
    exact = {
        "priority_manifest_sha256": manifest.sha256,
        "sarm_contract_sha256": contract.sha256,
    }
    for field, expected in exact.items():
        if progress_audit.get(field) != expected:
            raise ValueError(
                f"Progress audit {field}={progress_audit.get(field)!r}, expected {expected!r}"
            )
    training_progress = progress_audit.get("training_progress")
    if not isinstance(training_progress, dict):
        raise ValueError("Progress audit is missing its train-only artifact")
    if _sha256(progress_path) != training_progress.get("sha256"):
        raise ValueError("Train-only progress parquet does not match the progress audit")
    if list(contract.train_episodes) != training_progress.get("episodes"):
        raise ValueError("Train-only progress episodes do not match the frozen split")

    progress = pd.read_parquet(progress_path).sort_values("index", kind="stable")
    required = {"index", "episode_index", "frame_index", "progress_dense"}
    if not required <= set(progress.columns):
        raise ValueError(f"Train-only progress is missing {sorted(required - set(progress.columns))}")
    actual_episodes = sorted(int(value) for value in progress["episode_index"].unique())
    if actual_episodes != sorted(contract.train_episodes):
        raise ValueError("Train-only progress contains a holdout episode or omits a train episode")
    expected_frames = sum(
        manifest.episodes[index].frame_count for index in contract.train_episodes
    )
    if len(progress) != expected_frames:
        raise ValueError(f"Train-only progress has {len(progress)} rows, expected {expected_frames}")

    lookup = {
        int(row.index): float(row.progress_dense)
        for row in progress.itertuples(index=False)
    }
    episode_bounds = {
        int(episode): (int(frame["index"].min()), int(frame["index"].max()) + 1)
        for episode, frame in progress.groupby("episode_index")
    }

    rows: list[tuple[int, int, str, int, float]] = []
    cursor = 0
    train_set = set(contract.train_episodes)
    for episode in manifest.episodes:
        if episode.episode_index in train_set:
            for stage in episode.stages:
                log_priority = manifest.quality_log_priority[stage.quality_score]
                rows.extend(
                    (
                        cursor + local_frame,
                        episode.episode_index,
                        stage.name,
                        stage.quality_score,
                        log_priority,
                    )
                    for local_frame in range(stage.start_frame, stage.end_frame)
                )
        cursor += episode.frame_count

    global_indices = np.asarray([row[0] for row in rows], dtype=np.int64)
    episode_indices = np.asarray([row[1] for row in rows], dtype=np.int64)
    stage_names = np.asarray([row[2] for row in rows], dtype=object)
    quality_scores = np.asarray([row[3] for row in rows], dtype=np.int64)
    log_priorities = np.asarray([row[4] for row in rows], dtype=np.float64)
    if set(global_indices) != set(lookup):
        raise ValueError("Priority sampler frames and train-only progress indices differ")

    raw_priority = np.exp(log_priorities)
    priority_probability = raw_priority.copy()
    total_stage_mass = sum(manifest.stage_probability_mass.values())
    for stage in manifest.stage_order:
        mask = stage_names == stage
        target_mass = manifest.stage_probability_mass[stage] / total_stage_mass
        priority_probability[mask] *= target_mass / raw_priority[mask].sum()
    priority_probability /= priority_probability.sum()

    chunk_size = int(progress_audit["chunk_size"])
    deltas = np.empty(len(rows), dtype=np.float64)
    for position, (global_index, episode_index) in enumerate(
        zip(global_indices, episode_indices, strict=True)
    ):
        future = min(global_index + chunk_size, episode_bounds[int(episode_index)][1] - 1)
        deltas[position] = lookup[int(future)] - lookup[int(global_index)]
    if not np.isfinite(deltas).all():
        raise ValueError("SARM progress deltas are not finite")

    kappa = float(progress_audit["rabc"]["recommended_kappa"])
    epsilon = 1e-6
    rabc_weights, delta_mean, delta_std = _rabc_weights(
        deltas,
        kappa=kappa,
        epsilon=epsilon,
    )
    combined_probability = priority_probability * rabc_weights
    combined_sum = float(combined_probability.sum())
    if not math.isfinite(combined_sum) or combined_sum <= 0:
        raise ValueError("Combined expected contribution has no positive mass")
    combined_probability /= combined_sum

    frame_count = len(rows)
    priority_ess = _effective_sample_size(priority_probability)
    combined_ess = _effective_sample_size(combined_probability)
    positive_combined = combined_probability[combined_probability > 0]
    quality_delta_correlation = float(np.corrcoef(quality_scores, deltas)[0, 1])
    quality_weight_correlation = float(np.corrcoef(quality_scores, rabc_weights)[0, 1])
    stages = _stage_summary(
        stage_names,
        quality_scores,
        priority_probability,
        combined_probability,
        rabc_weights,
        manifest.stage_order,
    )

    gates = {
        "only_frozen_train_episodes": actual_episodes == sorted(contract.train_episodes),
        "manual_sampling_ess_fraction_at_least_0_90": priority_ess / frame_count >= 0.90,
        "rabc_nonzero_fraction_at_least_0_95": float(np.mean(rabc_weights > 0)) >= 0.95,
        "combined_ess_fraction_at_least_0_75": combined_ess / frame_count >= 0.75,
        "each_stage_combined_mass_between_0_15_and_0_60": all(
            0.15 <= values["combined_expected_mass"] <= 0.60
            for values in stages.values()
        ),
    }

    return {
        "schema_version": "openarm2.quality-sarm-integration.v1",
        "description": (
            "Pre-launch audit for one integrated candidate: conservative blind-quality "
            "sampling plus native dense SARM RA-BC. No new plain-BC control is required."
        ),
        "roles": {
            "manual_quality": "training-frame sampling probability only",
            "sarm_dense_progress": "30-frame native RA-BC action-loss weight only",
            "validation_and_test": "exhaustive unweighted episode holdouts",
            "smoothing": "neutral",
            "retry_candidates": "neutral",
        },
        "provenance": {
            "priority_manifest_sha256": manifest.sha256,
            "sarm_contract_sha256": contract.sha256,
            "progress_audit_sha256": _sha256(progress_audit_path),
            "source_progress_sha256": progress_audit["progress_sha256"],
            "training_progress_sha256": _sha256(progress_path),
            "reward_model": {
                "repo_id": "bohlt/openarm2-shirt-fold-sarm-v1",
                "revision": "108048371c101e77299b8b60ae5f214d30b295f2",
                "training_run_id": "train_reward_shirt_20260720_sarm_dense_v3",
                "wandb_run_id": "kttuwuef",
                "checkpoint_step": 1200,
                "reuse_justification": (
                    "v2 changed only blinded quality labels; stage boundaries, dense targets, "
                    "all 34,850 frames, and the 27/3/3 split are byte-for-byte equivalent"
                ),
            },
        },
        "split": {
            "seed": contract.split_seed,
            "train": list(contract.train_episodes),
            "validation": list(contract.validation_episodes),
            "test": list(contract.test_episodes),
            "training_frames": frame_count,
        },
        "manual_sampling": {
            "quality_log_priority": {
                str(score): value
                for score, value in sorted(manifest.quality_log_priority.items())
            },
            "maximum_quality_probability_ratio": float(
                priority_probability.max() / priority_probability.min()
            ),
            "effective_sample_size": priority_ess,
            "effective_sample_size_fraction": priority_ess / frame_count,
        },
        "rabc": {
            "chunk_size": chunk_size,
            "kappa": kappa,
            "epsilon": epsilon,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "negative_or_zero_weight_fraction": float(np.mean(rabc_weights == 0)),
            "nonzero_weight_fraction": float(np.mean(rabc_weights > 0)),
            "full_weight_fraction": float(np.mean(rabc_weights == 1)),
            "mean_raw_weight": float(rabc_weights.mean()),
            "minimum_nonzero_raw_weight": float(rabc_weights[rabc_weights > 0].min()),
        },
        "combined_expected_contribution": {
            "definition": (
                "normalized manual sampling probability multiplied by raw native RA-BC "
                "weight; this is a large-batch expectation because RA-BC renormalizes each batch"
            ),
            "effective_sample_size": combined_ess,
            "effective_sample_size_fraction": combined_ess / frame_count,
            "maximum_to_minimum_nonzero_ratio": float(
                combined_probability.max() / positive_combined.min()
            ),
            "quality_score_progress_delta_correlation": quality_delta_correlation,
            "quality_score_rabc_weight_correlation": quality_weight_correlation,
            "stages": stages,
        },
        "gates": {**gates, "pass": all(gates.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--progress-audit", type=Path, default=DEFAULT_PROGRESS_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = audit_integration(
        manifest_path=args.manifest,
        contract_path=args.contract,
        progress_path=args.progress,
        progress_audit_path=args.progress_audit,
    )
    if args.check:
        if not args.output.is_file():
            raise FileNotFoundError(f"Integration audit is missing: {args.output}")
        if json.loads(args.output.read_text()) != result:
            raise ValueError(f"Integration audit is stale: {args.output}")
        print(args.output)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite integration audit: {args.output}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
