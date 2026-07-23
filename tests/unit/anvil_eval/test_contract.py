import json
from pathlib import Path

import yaml

from anvil_eval.contract import audit_policy_contract


def _write_contract_fixture(root: Path, arm_mapping: dict[str, str]) -> tuple[Path, Path, Path]:
    checkpoint = root / "checkpoint" / "pretrained_model"
    dataset = root / "dataset"
    checkpoint.mkdir(parents=True)
    (dataset / "meta").mkdir(parents=True)
    names = ["right_joint_1.pos", "left_joint_1.pos"]
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "input_features": {"observation.state": {"shape": [2]}},
                "output_features": {"action": {"shape": [2]}},
                "action_feature_names": names,
                "use_relative_actions": True,
                "chunk_size": 30,
                "n_action_steps": 30,
            }
        )
    )
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "relative_actions_processor",
                        "config": {"enabled": True, "action_names": names, "exclude_joints": []},
                    }
                ]
            }
        )
    )
    (checkpoint / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "absolute_actions_processor",
                        "config": {"enabled": True},
                    }
                ]
            }
        )
    )
    (checkpoint / "anvil_config.json").write_text(
        json.dumps(
            {
                "normalization_contract": {
                    "action_space": "relative_to_observation_state",
                    "chunk_size": 30,
                    "exclude_joints": [],
                    "stats_source": "all_valid_dataset_chunks",
                    "stats_sample_count": 100,
                }
            }
        )
    )
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {"shape": [2], "names": names},
                    "action": {"shape": [2], "names": names},
                }
            }
        )
    )
    config = root / "inference.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "joint_names": {
                    "arm_mapping": arm_mapping,
                    "model_joint_order": ["joint1"],
                },
                "arms": {
                    "right": {"action_start": 0, "action_end": 1},
                    "left": {"action_start": 1, "action_end": 2},
                },
                "cameras": {"mapping": {}},
                "inference_tuning": {
                    "sync": {
                        "async_prefetch": True,
                        "prefetch_threshold": 20,
                        "replace_pending_actions": True,
                    },
                    "rtc": {"enabled": False},
                },
                "safety": {"max_position_delta": 0.05},
            },
            sort_keys=False,
        )
    )
    return checkpoint.parent, dataset, config


def test_contract_audit_accepts_matching_right_first_vector(tmp_path: Path) -> None:
    checkpoint, dataset, config = _write_contract_fixture(tmp_path, {"r": "right", "l": "left"})

    report = audit_policy_contract(checkpoint, dataset, config)

    assert report["errors"] == []
    assert report["checks"]["runtime_arm_order"] == ["right", "left"]


def test_contract_audit_rejects_left_first_runtime_vector(tmp_path: Path) -> None:
    checkpoint, dataset, config = _write_contract_fixture(tmp_path, {"l": "left", "r": "right"})

    report = audit_policy_contract(checkpoint, dataset, config)

    assert any("runtime/checkpoint arm order" in error for error in report["errors"])


def test_contract_audit_rejects_missing_relative_normalization_contract(
    tmp_path: Path,
) -> None:
    checkpoint, dataset, config = _write_contract_fixture(tmp_path, {"r": "right", "l": "left"})
    (checkpoint / "pretrained_model" / "anvil_config.json").write_text("{}")

    report = audit_policy_contract(checkpoint, dataset, config)

    assert any("missing its normalization contract" in error for error in report["errors"])


def test_contract_audit_rejects_relative_normalization_chunk_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint, dataset, config = _write_contract_fixture(tmp_path, {"r": "right", "l": "left"})
    anvil_config_path = checkpoint / "pretrained_model" / "anvil_config.json"
    anvil_config = json.loads(anvil_config_path.read_text())
    anvil_config["normalization_contract"]["chunk_size"] = 50
    anvil_config_path.write_text(json.dumps(anvil_config))

    report = audit_policy_contract(checkpoint, dataset, config)

    assert any("normalization/checkpoint chunk size" in error for error in report["errors"])


def test_contract_audit_accepts_train_only_bounded_representation(tmp_path: Path) -> None:
    checkpoint, dataset, config = _write_contract_fixture(tmp_path, {"r": "right", "l": "left"})
    model_dir = checkpoint / "pretrained_model"
    model_config = json.loads((model_dir / "config.json").read_text())
    model_config["use_relative_actions"] = False
    (model_dir / "config.json").write_text(json.dumps(model_config))
    digest = "a" * 64
    split = "b" * 64
    names = model_config["action_feature_names"]
    (model_dir / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "bounded_relative_actions_processor",
                        "config": {
                            "enabled": True,
                            "action_names": names,
                            "representation_id": "bounded-v1",
                            "contract_sha256": digest,
                            "split_sha256": split,
                            "inference_smoothing_kernel": [
                                1.0 / 6.0,
                                2.0 / 3.0,
                                1.0 / 6.0,
                            ],
                            "inference_smoothing_passes": 2,
                            "gripper_event_threshold": 0.005,
                        },
                    }
                ]
            }
        )
    )
    (model_dir / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "bounded_absolute_actions_processor",
                        "config": {"enabled": True},
                    }
                ]
            }
        )
    )
    (model_dir / "anvil_config.json").write_text(
        json.dumps(
            {
                "normalization_contract": {
                    "action_space": "state_relative_soft_limit_fraction",
                    "chunk_size": 30,
                    "contract_sha256": digest,
                    "split_sha256": split,
                    "stats_source": "frozen_training_episodes_only",
                    "horizon_sample_counts": [100] * 30,
                    "inference_smoothing": {
                        "method": "uniform_cubic_bspline",
                        "kernel": [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                        "passes": 2,
                        "gripper_mode": "absolute_passthrough",
                        "gripper_event_threshold": 0.005,
                    },
                }
            }
        )
    )

    report = audit_policy_contract(checkpoint, dataset, config)

    assert report["errors"] == []
    assert report["checks"]["bounded_actions"] == {
        "preprocessor_enabled": True,
        "postprocessor_enabled": True,
        "representation_id": "bounded-v1",
        "contract_sha256": digest,
        "inference_smoothing": {
            "method": "uniform_cubic_bspline",
            "kernel": [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            "passes": 2,
            "gripper_mode": "absolute_passthrough",
            "gripper_event_threshold": 0.005,
        },
        "normalization_contract": {
            "action_space": "state_relative_soft_limit_fraction",
            "chunk_size": 30,
            "contract_sha256": digest,
            "split_sha256": split,
            "stats_source": "frozen_training_episodes_only",
            "horizon_sample_counts": [100] * 30,
            "inference_smoothing": {
                "method": "uniform_cubic_bspline",
                "kernel": [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                "passes": 2,
                "gripper_mode": "absolute_passthrough",
                "gripper_event_threshold": 0.005,
            },
        },
    }
