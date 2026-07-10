from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from lerobot.configs.parser import _flatten_to_cli_args

from anvil_trainer.patches import _vla_jepa_current_state

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs" / "training"
VLA_JEPA_REVISION = "e946c3e5b538d760f4b4ff239d1b1c12090c041d"
SMOLVLA_REVISION = "967623a0f38c7e1236c66b3893c830398d793ff7"


@pytest.mark.parametrize(
    ("filename", "expected_horizon", "expected_steps"),
    (
        ("lego_in_cup_vla_jepa_world_model.yaml", 32, 10000),
        ("lego_in_cup_smolvla.yaml", 50, 20000),
    ),
)
def test_lego_finetuning_configs_use_policy_specific_eight_dimensional_chunks(
    filename: str, expected_horizon: int, expected_steps: int
) -> None:
    config = yaml.safe_load((CONFIG_ROOT / filename).read_text())
    policy = config["policy"]

    assert policy["chunk_size"] == expected_horizon
    assert policy["n_action_steps"] == expected_horizon
    assert config["steps"] == expected_steps
    assert config["log_freq"] == 500

    input_features = json.loads(policy["input_features"])
    output_features = json.loads(policy["output_features"])
    assert input_features["observation.state"]["shape"] == [8]
    assert output_features["action"]["shape"] == [8]


def test_vla_jepa_world_model_reinitializes_only_dimension_dependent_heads() -> None:
    config = yaml.safe_load(
        (CONFIG_ROOT / "lego_in_cup_vla_jepa_world_model.yaml").read_text()
    )
    policy = config["policy"]

    assert policy["enable_world_model"] is True
    assert policy["pretrained_revision"] == VLA_JEPA_REVISION
    assert policy["freeze_qwen"] is False
    assert policy["action_dim"] == 8
    assert policy["state_dim"] == 8
    assert set(json.loads(policy["reinit_modules"])) == {
        "model.action_model.action_encoder",
        "model.action_model.action_decoder",
        "model.action_model.state_encoder",
    }
    overrides = _flatten_to_cli_args(
        {key: value for key, value in policy.items() if key != "path"}
    )
    assert f'--reinit_modules={policy["reinit_modules"]}' in overrides


def test_smolvla_config_preserves_pretrained_expert_and_freezes_vision() -> None:
    config = yaml.safe_load(
        (CONFIG_ROOT / "lego_in_cup_smolvla.yaml").read_text()
    )
    policy = config["policy"]

    assert policy["path"] == "lerobot/smolvla_robotwin"
    assert policy["pretrained_revision"] == SMOLVLA_REVISION
    assert policy["load_vlm_weights"] is True
    assert policy["freeze_vision_encoder"] is True
    assert policy["train_expert_only"] is True
    assert policy["train_state_proj"] is True


def test_vla_jepa_inference_continuously_prefetches_and_replaces_stale_tail() -> None:
    config = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "lerobot_control"
            / "inference_lego_in_cup_vla_jepa.yaml"
        ).read_text()
    )
    sync = config["inference_tuning"]["sync"]

    assert sync["n_action_steps"] is None
    assert sync["async_prefetch"] is True
    assert sync["prefetch_threshold"] == 32
    assert sync["replace_pending_actions"] is True


def test_vla_jepa_stacked_state_uses_current_observation_not_future_state() -> None:
    state = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 8, 8)

    selected = _vla_jepa_current_state(state)

    assert selected.shape == (2, 1, 8)
    assert torch.equal(selected[:, 0], state[:, 0])
    assert not torch.equal(selected[:, 0], state[:, -1])
