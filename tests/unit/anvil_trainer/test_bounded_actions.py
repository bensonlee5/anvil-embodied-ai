"""Contracts for bounded Pi0.5 actions and Larchenko-derived train-only stats."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)

from anvil_trainer.bounded_actions import (
    BoundedActionContract,
    decode_bounded_actions,
    encode_bounded_actions,
)
from anvil_trainer.config import TrainingConfig
from anvil_trainer.patches import TransformRunner
from anvil_trainer.transforms import BoundedRobustnessTransform, DataIntegrityError

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    REPO_ROOT / "configs" / "training" / "action_contracts" / "openarm2_shirt_fold_bounded_v1.json"
)


def _contract() -> BoundedActionContract:
    return BoundedActionContract.load(CONTRACT_PATH)


def test_bounded_codec_round_trips_physical_targets() -> None:
    contract = _contract()
    lower = torch.tensor(contract.soft_lower, dtype=torch.float64)
    upper = torch.tensor(contract.soft_upper, dtype=torch.float64)
    state = (lower + upper).unsqueeze(0) / 2
    action = torch.stack(
        [
            state[0],
            lower * 0.75 + upper * 0.25,
            lower * 0.25 + upper * 0.75,
        ]
    ).unsqueeze(0)
    center = torch.zeros((contract.chunk_size, len(lower)), dtype=torch.float64)
    scale = torch.ones_like(center)

    encoded = encode_bounded_actions(
        action,
        state,
        lower=lower,
        upper=upper,
        arm_indices=contract.arm_indices,
        absolute_indices=contract.absolute_indices,
        center=center,
        scale=scale,
        clip_value=1.0,
    )
    decoded = decode_bounded_actions(
        encoded,
        state,
        lower=lower,
        upper=upper,
        arm_indices=contract.arm_indices,
        absolute_indices=contract.absolute_indices,
        center=center,
        scale=scale,
        clip_value=1.0,
    )

    torch.testing.assert_close(decoded, action)
    assert torch.all(encoded >= -1) and torch.all(encoded <= 1)


def test_bounded_decoder_guarantees_soft_limits_for_any_finite_output() -> None:
    contract = _contract()
    lower = torch.tensor(contract.soft_lower)
    upper = torch.tensor(contract.soft_upper)
    state = torch.linspace(-10, 10, len(lower)).unsqueeze(0)
    output = torch.full((1, contract.chunk_size, len(lower)), 1.0e6)
    output[:, 1::2] *= -1
    center = torch.zeros((contract.chunk_size, len(lower)))
    scale = torch.full_like(center, 0.25)

    decoded = decode_bounded_actions(
        output,
        state,
        lower=lower,
        upper=upper,
        arm_indices=contract.arm_indices,
        absolute_indices=contract.absolute_indices,
        center=center,
        scale=scale,
        clip_value=1.0,
    )

    assert torch.isfinite(decoded).all()
    assert torch.all(decoded >= lower.to(decoded.dtype))
    assert torch.all(decoded <= upper.to(decoded.dtype))


def _dataset(*, perturb_holdout: bool) -> SimpleNamespace:
    contract = _contract()
    lower = contract.soft_lower
    upper = contract.soft_upper
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    episodes: list[int] = []
    holdout = set(range(33)) - set(contract.training_episode_indices)
    for episode in range(33):
        phase = np.linspace(0, 2 * np.pi, 35, endpoint=False)
        midpoint = 0.5 * (lower + upper)
        amplitude = 0.04 * (upper - lower)
        state = midpoint + np.sin(phase[:, None] + np.arange(len(lower))[None, :]) * amplitude
        action = state + np.cos(phase[:, None]) * 0.01 * (upper - lower)
        action = np.clip(action, lower, upper)
        if perturb_holdout and episode in holdout:
            action[:, list(contract.arm_indices)] = upper[list(contract.arm_indices)]
        actions.append(action)
        states.append(state)
        episodes.extend([episode] * len(state))
    return SimpleNamespace(
        hf_dataset={
            "action": np.concatenate(actions),
            "observation.state": np.concatenate(states),
            "episode_index": np.asarray(episodes),
        }
    )


def _policy() -> SimpleNamespace:
    contract = _contract()
    return SimpleNamespace(
        type="pi05",
        use_relative_actions=False,
        chunk_size=contract.chunk_size,
        action_feature_names=list(contract.action_names),
        normalization_mapping={FeatureType.ACTION: NormalizationMode.IDENTITY},
    )


def test_per_horizon_statistics_ignore_holdout_rows() -> None:
    contract = _contract()
    config = TrainingConfig(bounded_action_contract=str(CONTRACT_PATH))
    first = TransformRunner(config)
    second = TransformRunner(config)

    first_stats = first._fit_bounded_action_statistics(
        _dataset(perturb_holdout=False), _policy(), list(contract.training_episode_indices)
    )
    second_stats = second._fit_bounded_action_statistics(
        _dataset(perturb_holdout=True), _policy(), list(contract.training_episode_indices)
    )

    assert first_stats is not None and second_stats is not None
    np.testing.assert_allclose(first_stats[0], second_stats[0])
    np.testing.assert_allclose(first_stats[1], second_stats[1])
    assert first_stats[0].shape == (30, 16)
    assert first._normalization_contract["stats_source"] == "frozen_training_episodes_only"
    assert first._normalization_contract["fit_episode_indices"] == list(
        contract.training_episode_indices
    )


def test_bounded_statistics_reject_a_different_resolved_split() -> None:
    runner = TransformRunner(TrainingConfig(bounded_action_contract=str(CONTRACT_PATH)))
    with pytest.raises(DataIntegrityError, match="resolved training episodes differ"):
        runner._fit_bounded_action_statistics(_dataset(perturb_holdout=False), _policy(), [0])


def test_processor_install_replaces_inherited_relative_pair() -> None:
    contract = _contract()
    runner = TransformRunner(TrainingConfig(bounded_action_contract=str(CONTRACT_PATH)))
    runner._fit_bounded_action_statistics(
        _dataset(perturb_holdout=False), _policy(), list(contract.training_episode_indices)
    )
    relative = RelativeActionsProcessorStep(enabled=False)
    preprocessor = SimpleNamespace(steps=[relative])
    postprocessor = SimpleNamespace(
        steps=[AbsoluteActionsProcessorStep(enabled=False, relative_step=relative)]
    )

    runner._install_bounded_action_processors(_policy(), preprocessor, postprocessor)

    assert preprocessor.steps[0].__class__._registry_name == "bounded_relative_actions_processor"
    assert postprocessor.steps[0].__class__._registry_name == "bounded_absolute_actions_processor"
    assert postprocessor.steps[0].relative_step is preprocessor.steps[0]
    assert preprocessor.steps[0].contract_sha256 == contract.sha256


def test_camera_dropout_never_drops_every_view(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TrainingConfig(
        bounded_action_contract=str(CONTRACT_PATH),
        camera_dropout_probability=0.5,
    )
    transform = BoundedRobustnessTransform(config)
    monkeypatch.setattr(torch, "rand", lambda count: torch.zeros(count))
    monkeypatch.setattr(torch, "randint", lambda *_args: torch.tensor([1]))
    item = {
        "observation.images.base": torch.ones(3, 4, 4),
        "observation.images.left_wrist": torch.ones(3, 4, 4),
        "observation.images.right_wrist": torch.ones(3, 4, 4),
    }

    transformed = transform.apply(item, config)

    live = sum(bool(value.any()) for value in transformed.values())
    assert live == 1
    assert transform.training_only is True


def test_state_noise_never_changes_gripper_observations() -> None:
    contract = _contract()
    config = TrainingConfig(
        bounded_action_contract=str(CONTRACT_PATH),
        state_noise_std_fraction=0.01,
    )
    transform = BoundedRobustnessTransform(config)
    state = torch.as_tensor(0.5 * (contract.soft_lower + contract.soft_upper)).float()
    state[7] = 0.052
    state[15] = 0.051

    transformed = transform.apply({"observation.state": state.clone()}, config)

    assert transformed["observation.state"][7] == state[7]
    assert transformed["observation.state"][15] == state[15]
    assert not torch.equal(
        transformed["observation.state"][list(contract.arm_indices)],
        state[list(contract.arm_indices)],
    )


def test_bounded_robustness_cli_flags_are_stripped_before_lerobot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "anvil-trainer",
            f"--bounded-action-contract={CONTRACT_PATH}",
            "--camera-dropout-probability=0.1",
            "--state-noise-std-fraction=0.002",
        ],
    )

    config = TrainingConfig.from_env_and_args()

    assert config.bounded_action_contract == str(CONTRACT_PATH)
    assert config.camera_dropout_probability == 0.1
    assert config.state_noise_std_fraction == 0.002
    assert not any("bounded-action" in arg for arg in sys.argv)
    assert not any("camera-dropout" in arg for arg in sys.argv)
    assert not any("state-noise" in arg for arg in sys.argv)
