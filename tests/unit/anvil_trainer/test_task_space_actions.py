"""Contracts for task-space targets and outward-elbow trajectory decoding."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from anvil_embodiment.kinematics import get_model_spec
from anvil_embodiment.trajectory import ConstrainedBimanualTrajectorySolver
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)

from anvil_trainer.config import TrainingConfig
from anvil_trainer.patches import (
    TransformRunner,
    _act_loss_per_dim,
    _restore_task_space_policy_surface,
    _task_space_dataset_metadata,
)
from anvil_trainer.task_space_actions import (
    TaskSpaceActionContract,
    TaskSpaceRelativeActionsProcessorStep,
    decode_task_space_targets,
    encode_task_space_actions,
    make_task_space_processor_steps,
    smooth_task_space_chunk,
)
from anvil_trainer.transforms import DataIntegrityError

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    REPO_ROOT
    / "configs"
    / "training"
    / "action_contracts"
    / "openarm2_shirt_fold_task_space_outward_v1.json"
)


def _contract() -> TaskSpaceActionContract:
    return TaskSpaceActionContract.load(CONTRACT_PATH)


def test_task_space_contract_pins_model_and_stays_offline_only() -> None:
    contract = _contract()

    assert contract.model_id == "anvil_openarm_v2"
    assert len(contract.model_sha256) == 64
    assert contract.deployment_status == "offline_only"
    assert len(contract.task_action_names) == 14
    assert contract.source_soft_lower.shape == (16,)
    assert np.all(contract.source_soft_lower < contract.source_soft_upper)


def test_task_space_recipe_keeps_matched_raw_sarm_surface() -> None:
    recipe = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "training"
            / "shirt_fold_pi05_hf_task_space_outward_5stage_sarm_raw_v4.yaml"
        ).read_text()
    )
    contract = _contract()

    assert recipe["policy"]["action_feature_names"] == list(contract.task_action_names)
    assert json.loads(recipe["policy"]["output_features"])["action"]["shape"] == [14]
    assert recipe["policy"]["use_relative_actions"] is False
    assert recipe["sample_weighting"]["extra_params"]["reward_calibration"] == "none"
    assert recipe["sample_weighting"]["progress_path"] == ("sarm_progress_train_5stage_v1.parquet")
    assert recipe["batch_size"] == 16
    assert recipe["steps"] == 5000


def test_act_task_space_recipe_changes_only_the_policy_baseline_surface() -> None:
    pi05 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "training"
            / "shirt_fold_pi05_hf_task_space_outward_5stage_sarm_raw_v4.yaml"
        ).read_text()
    )
    act = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "training"
            / "shirt_fold_act_hf_task_space_outward_5stage_sarm_raw_v1.yaml"
        ).read_text()
    )

    assert act["dataset"] == pi05["dataset"]
    assert act["sample_weighting"] == pi05["sample_weighting"]
    for key in (
        "seed",
        "num_workers",
        "persistent_workers",
        "batch_size",
        "steps",
        "log_freq",
        "save_freq",
    ):
        assert act[key] == pi05[key]
    assert act["policy"]["chunk_size"] == 30
    assert act["policy"]["n_action_steps"] == 30
    assert act["policy"]["output_features"]["action"]["shape"] == [14]
    assert act["policy"]["normalization_mapping"] == {
        "VISUAL": "MEAN_STD",
        "STATE": "MEAN_STD",
        "ACTION": "IDENTITY",
    }
    assert "path" not in act["policy"]
    assert act["policy"]["vision_backbone"] == "resnet18"
    assert act["policy"]["use_vae"] is True
    assert act["policy"]["kl_weight"] == 10.0


def test_task_space_policy_surface_is_restored_after_dataset_inference() -> None:
    contract = _contract()
    task_feature = PolicyFeature(type=FeatureType.ACTION, shape=(14,))
    joint_feature = PolicyFeature(type=FeatureType.ACTION, shape=(16,))
    policy_cfg = SimpleNamespace(
        output_features={"action": joint_feature},
        action_feature_names=list(contract.source_action_names),
    )
    runtime_cfg = SimpleNamespace(
        output_features={"action": joint_feature},
        action_feature_names=list(contract.source_action_names),
    )
    policy = SimpleNamespace(config=runtime_cfg)

    _restore_task_space_policy_surface(
        policy_cfg,
        policy,
        contract,
        {"action": task_feature},
    )

    for config in (policy_cfg, runtime_cfg):
        assert config.output_features["action"].shape == (14,)
        assert config.action_feature_names == list(contract.task_action_names)


def test_task_space_metadata_proxy_does_not_mutate_joint_source() -> None:
    contract = _contract()
    source_features = {
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": list(contract.source_action_names),
        }
    }

    class Metadata:
        def __init__(self, features):
            self.info = SimpleNamespace(features=features)

        @property
        def features(self):
            return self.info.features

    source = Metadata(source_features)

    proxy = _task_space_dataset_metadata(source, contract)

    assert proxy.features["action"]["shape"] == (14,)
    assert proxy.features["action"]["names"] == list(contract.task_action_names)
    assert source.features["action"]["shape"] == (16,)
    assert source.features["action"]["names"] == list(contract.source_action_names)


def test_task_space_collapses_joint_configuration_when_tcp_motion_is_zero() -> None:
    contract = _contract()
    center = torch.zeros((contract.chunk_size, 14), dtype=torch.float64)
    scale = torch.ones_like(center)
    first = torch.zeros(16, dtype=torch.float64)
    first[[7, 15]] = 0.02
    second = torch.tensor(
        [
            -0.1,
            0.2,
            -0.15,
            0.7,
            0.1,
            -0.2,
            0.2,
            0.02,
            0.1,
            -0.2,
            0.15,
            0.7,
            -0.1,
            0.2,
            -0.2,
            0.02,
        ],
        dtype=torch.float64,
    )

    encoded_first = encode_task_space_actions(
        first, first, contract=contract, center=center, scale=scale
    )
    encoded_second = encode_task_space_actions(
        second, second, contract=contract, center=center, scale=scale
    )

    torch.testing.assert_close(encoded_first, encoded_second, atol=1.0e-10, rtol=0)
    torch.testing.assert_close(encoded_first[:6], torch.zeros(6, dtype=torch.float64))
    torch.testing.assert_close(encoded_first[7:13], torch.zeros(6, dtype=torch.float64))


def test_task_space_target_decode_recovers_fk_pose_and_gripper() -> None:
    contract = _contract()
    center = torch.zeros((contract.chunk_size, 14), dtype=torch.float64)
    scale = torch.ones_like(center)
    state = torch.zeros(16, dtype=torch.float64)
    target = state.repeat(3, 1)
    target[:, 0] = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
    target[:, 8] = -target[:, 0]
    target[:, 7] = 0.01
    target[:, 15] = 0.04

    encoded = encode_task_space_actions(
        target.unsqueeze(0),
        state.unsqueeze(0),
        contract=contract,
        center=center,
        scale=scale,
    )
    positions, rotations, grippers = decode_task_space_targets(
        encoded,
        state.unsqueeze(0),
        contract=contract,
        center=center,
        scale=scale,
    )
    model = get_model_spec(contract.model_id)
    from anvil_embodiment.kinematics import torch_forward_kinematics

    expected_right = torch_forward_kinematics(model, "right", target[:, :7])
    expected_left = torch_forward_kinematics(model, "left", target[:, 8:15])
    torch.testing.assert_close(positions[0, :, 0], expected_right[0], atol=1.0e-8, rtol=0)
    torch.testing.assert_close(positions[0, :, 1], expected_left[0], atol=1.0e-8, rtol=0)
    torch.testing.assert_close(rotations[0, :, 0], expected_right[1], atol=1.0e-8, rtol=0)
    torch.testing.assert_close(rotations[0, :, 1], expected_left[1], atol=1.0e-8, rtol=0)
    torch.testing.assert_close(
        grippers[0],
        torch.tensor([[0.01, 0.04]] * 3, dtype=torch.float64),
    )


def test_solver_preserves_tcp_while_moving_elbows_outward_with_hard_bounds() -> None:
    contract = _contract()
    solver = ConstrainedBimanualTrajectorySolver(get_model_spec(contract.model_id), contract.solver)
    current = np.zeros(16, dtype=np.float64)
    current[[7, 15]] = 0.02
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    for side in ("right", "left"):
        position, rotation = solver.arms[side].pose(np.zeros(7))
        positions.append(position)
        rotations.append(rotation)

    result = solver.solve(
        positions=np.asarray([positions]),
        rotations=np.asarray([rotations]),
        grippers=np.asarray([[0.02, 0.02]]),
        current_state=current,
    )

    assert result.valid
    assert all(item.outward_alignment >= 0.85 for item in result.diagnostics)
    assert all(
        item.position_error_m <= contract.solver.position_tolerance_m for item in result.diagnostics
    )
    lower = contract.source_soft_lower
    upper = contract.source_soft_upper
    assert np.all(result.values >= lower - 1.0e-12)
    assert np.all(result.values <= upper + 1.0e-12)
    for start in (0, 8):
        delta = np.abs(result.values[0, start : start + 7] - current[start : start + 7])
        assert np.all(
            delta
            <= np.asarray(contract.solver.max_velocity_rad_s) * contract.solver.dt_seconds + 1.0e-12
        )


def test_task_space_smoothing_preserves_gripper_events_and_segment_endpoints() -> None:
    action = torch.zeros((8, 14), dtype=torch.float64)
    action[:, 0] = torch.tensor([0, 1, -1, 0, 1, -1, 1, 0], dtype=torch.float64)
    action[:4, 6] = -1
    action[4:, 6] = 1
    original = action.clone()

    smoothed = smooth_task_space_chunk(
        action,
        kernel=(1 / 6, 2 / 3, 1 / 6),
        passes=2,
        gripper_event_threshold_normalized=0.1,
    )

    torch.testing.assert_close(smoothed[:, [6, 13]], original[:, [6, 13]])
    torch.testing.assert_close(smoothed[[0, 3, 4, 7], 0], original[[0, 3, 4, 7], 0])
    assert (
        torch.diff(smoothed[:, 0], n=2).abs().mean() < torch.diff(original[:, 0], n=2).abs().mean()
    )


def _dataset(*, perturb_holdout: bool) -> SimpleNamespace:
    contract = _contract()
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    episodes: list[int] = []
    holdout = set(range(33)) - set(contract.training_episode_indices)
    for episode in range(33):
        phase = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        state = np.zeros((len(phase), 16), dtype=np.float64)
        state[:, 3] = 0.7
        state[:, 11] = 0.7
        state[:, 0] = 0.05 * np.sin(phase)
        state[:, 8] = -state[:, 0]
        state[:, [7, 15]] = 0.02
        action = state.copy()
        action[:, 0] += 0.02 * np.cos(phase)
        action[:, 8] -= 0.02 * np.cos(phase)
        if perturb_holdout and episode in holdout:
            action[:, 0] += 0.4
            action[:, 8] -= 0.4
        actions.append(action)
        states.append(state)
        episodes.extend([episode] * len(phase))
    names = list(contract.source_action_names)
    return SimpleNamespace(
        hf_dataset={
            "action": np.concatenate(actions),
            "observation.state": np.concatenate(states),
            "episode_index": np.asarray(episodes),
        },
        meta=SimpleNamespace(
            features={
                "action": {"shape": [16], "names": names},
                "observation.state": {"shape": [16], "names": names},
                **{
                    key: {"shape": [3, 270, 480], "dtype": "video"}
                    for key in (
                        "observation.images.left_wrist",
                        "observation.images.right_wrist",
                        "observation.images.base",
                    )
                },
            },
            camera_keys=[
                "observation.images.left_wrist",
                "observation.images.right_wrist",
                "observation.images.base",
            ],
        ),
    )


def _policy(*, policy_type: str = "pi05") -> SimpleNamespace:
    contract = _contract()
    cameras = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 270, 480))
        for key in (
            "observation.images.left_wrist",
            "observation.images.right_wrist",
            "observation.images.base",
        )
    }
    return SimpleNamespace(
        type=policy_type,
        use_relative_actions=False,
        chunk_size=contract.chunk_size,
        action_feature_names=list(contract.task_action_names),
        normalization_mapping={FeatureType.ACTION: NormalizationMode.IDENTITY},
        input_features={
            **cameras,
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        image_features=cameras,
    )


def test_task_space_statistics_are_train_only_and_install_processors() -> None:
    contract = _contract()
    config = TrainingConfig(task_space_action_contract=str(CONTRACT_PATH))
    first = TransformRunner(config)
    second = TransformRunner(config)
    train = list(contract.training_episode_indices)

    first_stats = first._fit_task_space_action_statistics(
        _dataset(perturb_holdout=False), _policy(), train
    )
    second_stats = second._fit_task_space_action_statistics(
        _dataset(perturb_holdout=True), _policy(), train
    )

    assert first_stats is not None and second_stats is not None
    np.testing.assert_allclose(first_stats[0], second_stats[0])
    np.testing.assert_allclose(first_stats[1], second_stats[1])
    assert first_stats[0].shape == (30, 14)
    assert first._normalization_contract["stats_source"] == "frozen_training_episodes_only"

    relative = RelativeActionsProcessorStep(enabled=False)
    preprocessor = SimpleNamespace(steps=[relative])
    postprocessor = SimpleNamespace(
        steps=[AbsoluteActionsProcessorStep(enabled=False, relative_step=relative)]
    )
    first._install_task_space_action_processors(_policy(), preprocessor, postprocessor)
    assert preprocessor.steps[0].__class__._registry_name == (
        "task_space_relative_actions_processor"
    )
    assert postprocessor.steps[0].relative_step is preprocessor.steps[0]
    features = {
        PipelineFeatureType.ACTION: {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        PipelineFeatureType.OBSERVATION: {},
    }
    assert preprocessor.steps[0].transform_features(features)[PipelineFeatureType.ACTION][
        "action"
    ].shape == (14,)


def test_task_space_act_processors_wrap_identity_action_normalization() -> None:
    contract = _contract()
    config = TrainingConfig(task_space_action_contract=str(CONTRACT_PATH))
    runner = TransformRunner(config)
    train = list(contract.training_episode_indices)
    stats = runner._fit_task_space_action_statistics(
        _dataset(perturb_holdout=False),
        _policy(policy_type="act"),
        train,
    )
    assert stats is not None

    policy = ACTConfig(
        device="cpu",
        chunk_size=30,
        n_action_steps=30,
        input_features=_policy(policy_type="act").input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        normalization_mapping={
            FeatureType.VISUAL: NormalizationMode.MEAN_STD,
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.IDENTITY,
        },
    )
    policy.action_feature_names = list(contract.task_action_names)
    preprocessor, postprocessor = make_act_pre_post_processors(policy)

    runner._install_task_space_action_processors(policy, preprocessor, postprocessor)

    assert [step.__class__._registry_name for step in preprocessor.steps].count(
        "task_space_relative_actions_processor"
    ) == 1
    assert [step.__class__._registry_name for step in postprocessor.steps].count(
        "task_space_absolute_actions_processor"
    ) == 1


def test_act_per_output_loss_ignores_padded_actions() -> None:
    targets = torch.tensor([[[1.0, 2.0], [10.0, 20.0]], [[3.0, 4.0], [5.0, 6.0]]])
    predictions = torch.zeros_like(targets)
    batch = {
        "action": targets,
        "action_is_pad": torch.tensor([[False, True], [False, False]]),
    }

    assert _act_loss_per_dim(batch, predictions) == pytest.approx([3.0, 4.0])


def test_act_forward_supports_rabc_per_sample_reduction() -> None:
    runner = TransformRunner(TrainingConfig())
    runner.apply_act_per_output_loss_patch()
    try:
        policy = ACTPolicy(
            ACTConfig(
                device="cpu",
                chunk_size=3,
                n_action_steps=3,
                input_features={
                    "observation.environment_state": PolicyFeature(
                        type=FeatureType.ENV, shape=(3,)
                    ),
                    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                },
                output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
                dim_model=32,
                n_heads=4,
                dim_feedforward=64,
                n_encoder_layers=1,
                n_decoder_layers=1,
                latent_dim=4,
                n_vae_encoder_layers=1,
            )
        )
        batch = {
            "observation.environment_state": torch.randn(2, 3),
            "observation.state": torch.randn(2, 2),
            "action": torch.randn(2, 3, 2),
            "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
        }

        per_sample, per_sample_metrics = policy.forward(batch, reduction="none")
        mean, mean_metrics = policy.forward(batch, reduction="mean")

        assert per_sample.shape == (2,)
        assert torch.isfinite(per_sample).all()
        assert torch.isfinite(mean)
        assert len(per_sample_metrics["loss_per_dim"]) == 2
        assert len(mean_metrics["loss_per_dim"]) == 2
    finally:
        runner.restore_all_patches()


def test_task_space_policy_contract_rejects_joint_output_surface() -> None:
    contract = _contract()
    policy = _policy()
    policy.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(16,))

    with pytest.raises(DataIntegrityError, match="effective action shapes differ"):
        TransformRunner._validate_policy_dataset_contract(
            policy, _dataset(perturb_holdout=False), contract
        )


def test_task_space_cli_flag_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["anvil-trainer", f"--task-space-action-contract={CONTRACT_PATH}"],
    )

    config = TrainingConfig.from_env_and_args()

    assert config.task_space_action_contract == str(CONTRACT_PATH)
    assert not any("task-space-action-contract" in item for item in sys.argv)


def test_task_space_and_bounded_contracts_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingConfig(
            bounded_action_contract="bounded.json",
            task_space_action_contract="task.json",
        )


def test_task_processor_factory_pins_contract_hash() -> None:
    contract = _contract()
    center = np.zeros((contract.chunk_size, 14))
    scale = np.ones_like(center)

    relative, absolute = make_task_space_processor_steps(contract, center=center, scale=scale)

    assert relative.contract_sha256 == contract.sha256
    assert relative.source_action_names == list(contract.source_action_names)
    assert relative.task_action_names == list(contract.task_action_names)
    assert relative.load_contract().sha256 == contract.sha256
    restored = TaskSpaceRelativeActionsProcessorStep(**relative.get_config())
    assert restored.load_contract().sha256 == contract.sha256
    assert absolute.relative_step is relative
