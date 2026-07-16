from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    REPO_ROOT / "configs" / "lerobot_control" / "inference_shirt_fold_pi05_shadow.yaml"
)
STRATEGY_PATH = (
    REPO_ROOT
    / "ros2"
    / "src"
    / "lerobot_control"
    / "lerobot_control"
    / "strategies"
    / "multi_process.py"
)


def test_shirt_fold_pi05_shadow_contract_is_right_first_and_disconnected() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["model"]["task_description"] == "Fold the T-shirt properly"
    assert list(config["joint_names"]["arm_mapping"]) == ["r", "l"]
    assert config["joint_names"]["model_joint_order"] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
    ]
    assert config["arms"]["right"]["action_start"] == 0
    assert config["arms"]["right"]["action_end"] == 8
    assert config["arms"]["left"]["action_start"] == 8
    assert config["arms"]["left"]["action_end"] == 16
    assert all(
        arm["command_topic"].startswith("/eval/")
        for arm in config["arms"].values()
    )
    assert set(config["cameras"]["mapping"].values()) == {
        "base",
        "left_wrist",
        "right_wrist",
    }
    assert config["inference_tuning"]["rtc"]["queue_trigger_threshold"] == 30
    assert config["inference_tuning"]["rtc"]["execution_horizon"] == 20
    assert config["safety"]["max_position_delta"] == 0.02


def test_multi_process_strategy_honors_yaml_arm_insertion_order() -> None:
    source = STRATEGY_PATH.read_text()
    assert "for arm_key in arm_mapping:" in source
    assert "sorted(arm_mapping.keys())" not in source
