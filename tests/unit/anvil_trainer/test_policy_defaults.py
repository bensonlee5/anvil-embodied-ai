from __future__ import annotations

import os
import sys

from anvil_trainer.config import TrainingConfig


def _parse_with_policy(policy_type: str) -> list[str]:
    original_argv = sys.argv[:]
    original_env = os.environ.copy()
    try:
        sys.argv = [
            "anvil-trainer",
            "--dataset.root=/tmp/fake",
            f"--policy.type={policy_type}",
        ]
        for key in (
            "LEROBOT_EXCLUDE_OBSERVS",
            "LEROBOT_EXCLUDE_OBSERVATION",
            "LEROBOT_TASK_OVERRIDE",
        ):
            os.environ.pop(key, None)

        TrainingConfig.from_env_and_args()
        return sys.argv[:]
    finally:
        sys.argv = original_argv
        os.environ.clear()
        os.environ.update(original_env)


def test_act_gets_default_backbone_flags():
    argv = _parse_with_policy("act")

    assert "--policy.vision_backbone=resnet18" in argv
    assert "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1" in argv


def test_selected_foundation_policy_skips_backbone_flags():
    argv = _parse_with_policy("molmoact2")

    assert not any(arg.startswith("--policy.vision_backbone=") for arg in argv)
    assert not any(arg.startswith("--policy.pretrained_backbone_weights=") for arg in argv)


def test_sync_foundation_policy_skips_backbone_flags():
    argv = _parse_with_policy("multi_task_dit")

    assert not any(arg.startswith("--policy.vision_backbone=") for arg in argv)
    assert not any(arg.startswith("--policy.pretrained_backbone_weights=") for arg in argv)
