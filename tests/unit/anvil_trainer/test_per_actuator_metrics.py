"""Per-actuator loss logging contracts."""

from __future__ import annotations

import math

import pytest

from anvil_trainer.patches import _action_batch_size, _PerActuatorLossMeter
from anvil_trainer.transforms import DataIntegrityError

ACTION_NAMES = (
    "right_joint_1.pos",
    "right_gripper.pos",
    "left_joint_1.pos",
    "left_gripper.pos",
)


def test_per_actuator_meter_emits_weighted_scalar_metrics() -> None:
    meter = _PerActuatorLossMeter(ACTION_NAMES)

    cleaned = meter.update(
        {"loss": 0.25, "loss_per_dim": [0.1, 0.2, 0.3, 0.4]},
        weight=2,
    )
    meter.update({"loss_per_dim": [0.4, 0.5, 0.6, 0.7]}, weight=1)
    metrics = meter.pop_metrics("eval/val_loss_per_actuator")

    assert cleaned == {"loss": 0.25}
    assert metrics == pytest.approx(
        {
            "eval/val_loss_per_actuator/right_joint_1.pos": 0.2,
            "eval/val_loss_per_actuator/right_gripper.pos": 0.3,
            "eval/val_loss_per_actuator/left_joint_1.pos": 0.4,
            "eval/val_loss_per_actuator/left_gripper.pos": 0.5,
        }
    )
    assert all(isinstance(value, float) for value in metrics.values())
    assert meter.pop_metrics("unused") == {}


@pytest.mark.parametrize(
    ("action_names", "losses", "match"),
    [
        ((), [0.1], "non-empty action feature names"),
        (("joint", "joint"), [0.1, 0.2], "unique action feature names"),
        (("joint",), [0.1, 0.2], "length mismatch"),
        (("joint",), [math.nan], "non-finite"),
        (("joint",), [math.inf], "non-finite"),
    ],
)
def test_per_actuator_meter_rejects_invalid_contracts(
    action_names: tuple[str, ...], losses: list[float], match: str
) -> None:
    if not action_names or len(set(action_names)) != len(action_names):
        with pytest.raises(DataIntegrityError, match=match):
            _PerActuatorLossMeter(action_names)
        return

    meter = _PerActuatorLossMeter(action_names)
    with pytest.raises(DataIntegrityError, match=match):
        meter.update({"loss_per_dim": losses})


def test_per_actuator_meter_requires_sequence_and_positive_weight() -> None:
    meter = _PerActuatorLossMeter(("joint",))

    with pytest.raises(DataIntegrityError, match="list or tuple"):
        meter.update({"loss_per_dim": 0.1})
    with pytest.raises(DataIntegrityError, match="positive integer"):
        meter.update({"loss_per_dim": [0.1]}, weight=0)
    with pytest.raises(DataIntegrityError, match="positive integer"):
        meter.update({"loss_per_dim": [0.1]}, weight=True)
    with pytest.raises(DataIntegrityError, match="non-numeric"):
        meter.update({"loss_per_dim": ["bad"]})


def test_action_batch_size_uses_action_tensor() -> None:
    torch = pytest.importorskip("torch")

    assert _action_batch_size({"action": torch.zeros(7, 30, 16)}) == 7
    with pytest.raises(DataIntegrityError, match="positive batch size"):
        _action_batch_size({})
