"""Contracts for exact native-LeRobot SARM annotation materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from anvil_trainer.priority_sampling import PriorityManifest
from anvil_trainer.sarm_annotations import (
    SARMAnnotationContract,
    SARMAnnotationError,
    compute_temporal_proportions,
    materialize_sarm_dataset,
    validate_sarm_dataset,
)

ROOT = Path(__file__).resolve().parents[3]
PRIORITY_PATH = (
    ROOT / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json"
)
CONTRACT_PATH = ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, PriorityManifest, SARMAnnotationContract]:
    source = tmp_path / "source"
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "meta/info.json").write_text('{"fps": 30}')
    (source / "data/chunk-000/file-000.parquet").write_bytes(b"data")
    (source / "meta/trim_manifest.json").write_text("{}")
    pd.DataFrame(
        {
            "episode_index": [0, 1],
            "length": [6, 6],
            "dataset_from_index": [0, 6],
            "dataset_to_index": [6, 12],
        }
    ).to_parquet(source / "meta/episodes/chunk-000/file-000.parquet", index=False)

    episodes = []
    for episode_index in range(2):
        episodes.append(
            {
                "episode_index": episode_index,
                "frame_count": 6,
                "stages": [
                    {
                        "name": name,
                        "start_frame": start,
                        "end_frame": end,
                        "quality_score": 3,
                        "quality_confidence": "medium",
                    }
                    for name, start, end in (
                        ("side_one", 0, 2),
                        ("side_two", 2, 4),
                        ("bottom_to_top", 4, 6),
                    )
                ],
                "repeated_grasps": [],
                "smoothing": {
                    "label": "present",
                    "confidence": "medium",
                    "review_start_frame": 1,
                    "review_end_frame": 5,
                    "stage_context": ["side_one", "side_two", "bottom_to_top"],
                    "priority_adjustment": 0.0,
                },
            }
        )
    manifest_data = {
        "schema_version": "openarm2.priority-sampling.v1",
        "description": "fixture",
        "dataset": {
            "repo_id": "test/dataset",
            "revision": "0" * 40,
            "episodes": 2,
            "frames": 12,
            "fps": 30,
            "fingerprints": {
                relative: _sha256(source / relative)
                for relative in (
                    "meta/info.json",
                    "data/chunk-000/file-000.parquet",
                    "meta/trim_manifest.json",
                )
            },
        },
        "stage_order": ["side_one", "side_two", "bottom_to_top"],
        "annotation_contract": {
            "quality_scale": [1, 5],
            "repeated_grasp_definition": "fixture",
            "smoothing_definition": "fixture",
            "smoothing_affects_priority": False,
            "single_annotator": True,
        },
        "weighting": {
            "method": "exponential_priority_sampler",
            "loss_reweighting": False,
            "sampling_replacement": True,
            "quality_log_priority": {
                "1": -1.0,
                "2": -0.5,
                "3": 0.0,
                "4": 0.5,
                "5": 1.0,
            },
            "repeated_grasp_log_penalty": -0.5,
            "smoothing_log_adjustment": 0.0,
            "stage_probability_mass": {
                "side_one": 1.0,
                "side_two": 1.0,
                "bottom_to_top": 1.0,
            },
        },
        "episodes": episodes,
    }
    manifest_path = tmp_path / "priority.json"
    manifest_path.write_text(json.dumps(manifest_data))
    manifest = PriorityManifest.load(manifest_path)
    contract_data = {
        "schema_version": "openarm2.sarm-dataset.v1",
        "description": "fixture",
        "priority_manifest_sha256": manifest.sha256,
        "annotation_mode": "dense_only",
        "sparse_task": "fold_t_shirt",
        "dense_stage_order": ["side_one", "side_two", "bottom_to_top"],
        "image_key": "observation.images.base",
        "state_key": "observation.state",
        "frame_semantics": {
            "priority_manifest_end": "exclusive",
            "lerobot_sarm_end": "inclusive",
            "conversion": "dense_end_frame=manifest_end_frame-1",
        },
        "split": {
            "seed": 1000,
            "ratio": [8, 1, 1],
            "train": [0],
            "validation": [1],
            "test": [],
        },
        "temporal_proportions": {
            "side_one": 1 / 3,
            "side_two": 1 / 3,
            "bottom_to_top": 1 / 3,
        },
        "behavior_labels": {
            "quality": "external_stage_prior_not_sarm_target",
            "repeated_grasps": "evaluation_only_not_failure_labels",
            "smoothing": "coarse_review_windows_priority_neutral",
        },
    }
    contract_path = tmp_path / "sarm.json"
    contract_path.write_text(json.dumps(contract_data))
    return source, manifest, SARMAnnotationContract.load(
        contract_path, priority_manifest=manifest
    )


def test_checked_in_sarm_contract_matches_priority_annotations() -> None:
    manifest = PriorityManifest.load(PRIORITY_PATH)
    contract = SARMAnnotationContract.load(CONTRACT_PATH, priority_manifest=manifest)
    assert contract.train_episodes == (
        0,
        1,
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        13,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        28,
        29,
        30,
        31,
        32,
    )
    assert compute_temporal_proportions(manifest, contract.train_episodes) == pytest.approx(
        contract.temporal_proportions,
        abs=1e-12,
    )


def test_materialization_preserves_source_and_converts_exclusive_ends(tmp_path: Path) -> None:
    source, manifest, contract = _write_fixture(tmp_path)
    source_meta = source / "meta/episodes/chunk-000/file-000.parquet"
    source_hash = _sha256(source_meta)
    output = tmp_path / "derived"
    summary = materialize_sarm_dataset(
        source,
        output,
        manifest=manifest,
        contract=contract,
    )
    assert summary["episodes"] == 2
    assert _sha256(source_meta) == source_hash
    derived = pd.read_parquet(output / "meta/episodes/chunk-000/file-000.parquet")
    assert list(derived.loc[0, "dense_subtask_start_frames"]) == [0, 2, 4]
    assert list(derived.loc[0, "dense_subtask_end_frames"]) == [1, 3, 5]
    assert list(derived.loc[0, "dense_subtask_end_times"]) == pytest.approx(
        [1 / 30, 3 / 30, 5 / 30]
    )
    validate_sarm_dataset(output, manifest=manifest, contract=contract)


def test_validation_detects_annotation_drift(tmp_path: Path) -> None:
    source, manifest, contract = _write_fixture(tmp_path)
    output = tmp_path / "derived"
    materialize_sarm_dataset(source, output, manifest=manifest, contract=contract)
    path = output / "meta/episodes/chunk-000/file-000.parquet"
    frame = pd.read_parquet(path)
    frame.at[0, "dense_subtask_end_frames"] = [2, 3, 5]
    frame.to_parquet(path, index=False)
    with pytest.raises(SARMAnnotationError, match="dense_subtask_end_frames mismatch"):
        validate_sarm_dataset(output, manifest=manifest, contract=contract)
