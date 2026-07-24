"""Regression tests for Pi0.5 embodiment and normalization contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from lerobot.configs.types import FeatureType, PolicyFeature

from anvil_trainer.config import TrainingConfig
from anvil_trainer.patches import TransformRunner
from anvil_trainer.transforms import DataIntegrityError

NAMES = ["right_joint_1.pos", "right_gripper.pos"]
CAMERAS = [
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.base",
]


def _dataset(*, camera_keys: list[str] | None = None) -> SimpleNamespace:
    actions = np.array(
        [
            [0.0, 0.10],
            [0.2, 0.20],
            [0.5, 0.30],
            [0.9, 0.40],
            [1.4, 0.50],
            [2.0, 0.60],
        ],
        dtype=np.float32,
    )
    states = np.array(
        [
            [-0.1, 0.10],
            [0.0, 0.20],
            [0.2, 0.30],
            [0.5, 0.40],
            [0.9, 0.50],
            [1.4, 0.60],
        ],
        dtype=np.float32,
    )
    features = {
        "action": {"shape": [2], "names": NAMES},
        "observation.state": {"shape": [2], "names": NAMES},
        **{key: {"shape": [3, 270, 480], "dtype": "video"} for key in CAMERAS},
    }
    raw_stats = {
        "action": {
            "mean": actions.mean(axis=0),
            "std": actions.std(axis=0),
            "min": actions.min(axis=0),
            "max": actions.max(axis=0),
            "q01": np.quantile(actions, 0.01, axis=0),
            "q99": np.quantile(actions, 0.99, axis=0),
        }
    }
    return SimpleNamespace(
        hf_dataset={
            "action": actions,
            "observation.state": states,
            "episode_index": np.zeros(len(actions), dtype=np.int64),
        },
        meta=SimpleNamespace(
            features=features,
            camera_keys=list(camera_keys or CAMERAS),
            stats=raw_stats,
        ),
    )


def _policy(*, cameras: list[str] | None = None) -> SimpleNamespace:
    camera_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 270, 480))
        for key in (cameras or CAMERAS)
    }
    return SimpleNamespace(
        type="pi05",
        use_relative_actions=True,
        chunk_size=3,
        relative_exclude_joints=["gripper"],
        action_feature_names=NAMES,
        input_features={
            **camera_features,
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        image_features=camera_features,
    )


def test_pi05_contract_accepts_exact_dataset_features() -> None:
    TransformRunner._validate_policy_dataset_contract(_policy(), _dataset())


def test_pi05_contract_rejects_inherited_checkpoint_camera() -> None:
    policy = _policy(cameras=[*CAMERAS, "observation.images.base_0_rgb"])

    with pytest.raises(DataIntegrityError, match="camera keys differ"):
        TransformRunner._validate_policy_dataset_contract(policy, _dataset())


def test_native_relative_stats_replace_absolute_dataset_stats() -> None:
    dataset = _dataset()
    original_action_mean = dataset.meta.stats["action"]["mean"].copy()
    runner = TransformRunner(TrainingConfig())

    stats = runner._compute_native_relative_action_stats(
        dataset,
        _policy(),
        num_workers=0,
    )

    assert stats is dataset.meta.stats["action"]
    assert not np.allclose(stats["mean"], original_action_mean)
    # Joint 1 becomes action - current state; the excluded gripper stays absolute.
    assert stats["mean"][0] < 0.8
    assert stats["mean"][1] > 0.2
    assert int(stats["count"][0]) == 12  # four valid starts * chunk size three
    assert runner._normalization_contract == {
        "action_space": "relative_to_observation_state",
        "chunk_size": 3,
        "exclude_joints": ["gripper"],
        "stats_source": "all_valid_dataset_chunks",
        "stats_sample_count": 12,
    }


def test_native_relative_stats_fail_closed_with_legacy_delta_transform() -> None:
    runner = TransformRunner(TrainingConfig(action_type="delta_obs_t"))

    with pytest.raises(DataIntegrityError, match="cannot both be enabled"):
        runner._compute_native_relative_action_stats(
            _dataset(),
            _policy(),
            num_workers=0,
        )
