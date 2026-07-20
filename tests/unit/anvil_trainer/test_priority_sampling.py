"""Contracts for annotation-driven frame-priority sampling."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from anvil_trainer.config import TrainingConfig
from anvil_trainer.priority_sampling import (
    PriorityEpisodeAwareSampler,
    PriorityManifest,
    PriorityManifestError,
)


def _write_manifest(path: Path, *, repeated_grasp: bool = False) -> PriorityManifest:
    episodes = []
    for episode_index, score in enumerate((1, 5)):
        repeated = []
        if repeated_grasp and episode_index == 1:
            repeated = [
                {
                    "gripper": "right",
                    "stage": "side_one",
                    "close_frame": 0,
                    "reopen_frame": 1,
                    "retry_frame": 2,
                    "penalty_start_frame": 0,
                    "penalty_end_frame": 1,
                    "confidence": "medium",
                }
            ]
        episodes.append(
            {
                "episode_index": episode_index,
                "frame_count": 6,
                "stages": [
                    {
                        "name": "side_one",
                        "start_frame": 0,
                        "end_frame": 2,
                        "quality_score": score,
                        "quality_confidence": "medium",
                    },
                    {
                        "name": "side_two",
                        "start_frame": 2,
                        "end_frame": 4,
                        "quality_score": score,
                        "quality_confidence": "medium",
                    },
                    {
                        "name": "bottom_to_top",
                        "start_frame": 4,
                        "end_frame": 6,
                        "quality_score": score,
                        "quality_confidence": "medium",
                    },
                ],
                "repeated_grasps": repeated,
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
    data = {
        "schema_version": "openarm2.priority-sampling.v1",
        "description": "test",
        "dataset": {
            "repo_id": "test/dataset",
            "revision": "0" * 40,
            "episodes": 2,
            "frames": 12,
            "fps": 30,
            "fingerprints": {"meta/info.json": "0" * 64},
        },
        "stage_order": ["side_one", "side_two", "bottom_to_top"],
        "annotation_contract": {
            "quality_scale": [1, 5],
            "repeated_grasp_definition": "test",
            "smoothing_definition": "test",
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
    path.write_text(json.dumps(data))
    return PriorityManifest.load(path)


def _sampler(manifest: PriorityManifest, **kwargs) -> PriorityEpisodeAwareSampler:
    return PriorityEpisodeAwareSampler(
        [0, 6],
        [6, 12],
        manifest=manifest,
        shuffle=True,
        seed=1000,
        **kwargs,
    )


def test_manifest_is_strict_about_unknown_keys(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json")
    data = json.loads(manifest.path.read_text())
    data["unexpected"] = True
    manifest.path.write_text(json.dumps(data))
    with pytest.raises(PriorityManifestError, match=r"extra=\['unexpected'\]"):
        PriorityManifest.load(manifest.path)


def test_sampler_equalizes_stage_mass_and_preserves_quality_ratio(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json")
    sampler = _sampler(manifest)
    probabilities = sampler.probabilities.numpy()
    assert probabilities[0:2].sum() + probabilities[6:8].sum() == pytest.approx(1 / 3)
    assert probabilities[2:4].sum() + probabilities[8:10].sum() == pytest.approx(1 / 3)
    assert probabilities[4:6].sum() + probabilities[10:12].sum() == pytest.approx(1 / 3)
    assert probabilities[6] / probabilities[0] == pytest.approx(math.exp(2.0))


def test_repeated_grasp_penalty_is_local_and_smoothing_is_neutral(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json", repeated_grasp=True)
    sampler = _sampler(manifest, episode_indices_to_use=[1])
    probabilities = sampler.probabilities.numpy()
    assert probabilities[1] / probabilities[0] == pytest.approx(math.exp(0.5))
    # The broad smoothing window crosses frames 1..4 but has no discontinuity.
    assert probabilities[2] == pytest.approx(probabilities[3])
    assert probabilities[4] == pytest.approx(probabilities[5])


def test_sampler_maps_filtered_absolute_indices_once_and_resumes_exactly(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json")
    absolute_to_relative = {absolute: absolute - 6 for absolute in range(6, 12)}
    sampler = _sampler(
        manifest,
        episode_indices_to_use=[1],
        absolute_to_relative_idx=absolute_to_relative,
    )
    first_epoch = list(iter(sampler))
    assert len(first_epoch) == 6
    assert set(first_epoch) <= set(range(6))

    resumed = _sampler(
        manifest,
        episode_indices_to_use=[1],
        absolute_to_relative_idx=absolute_to_relative,
    )
    resumed.load_state_dict({"epoch": 0, "start_index": 3})
    assert list(iter(resumed)) == first_epoch[3:]


def test_checked_in_manifest_is_current_and_complete() -> None:
    root = Path(__file__).resolve().parents[3]
    manifest = PriorityManifest.load(
        root
        / "configs/training/priority_manifests/openarm2_shirt_fold_3stage_v1.json"
    )
    assert len(manifest.episodes) == 33
    assert sum(episode.frame_count for episode in manifest.episodes) == 34850
    assert sum(len(episode.repeated_grasps) for episode in manifest.episodes) == 28


def test_priority_manifest_cli_flag_is_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "anvil-trainer",
            "--priority-sampling-manifest=/tmp/priority.json",
            "--dataset.root=/tmp/dataset",
            "--policy.type=pi05",
        ],
    )
    config = TrainingConfig.from_env_and_args()
    assert config.priority_sampling_manifest == "/tmp/priority.json"
    assert not any(argument.startswith("--priority-sampling-manifest=") for argument in sys.argv)
