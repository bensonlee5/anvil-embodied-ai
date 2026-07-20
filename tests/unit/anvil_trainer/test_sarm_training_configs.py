"""Frozen SARM reward-model and isolated RA-BC recipe contracts."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "configs/training"


def test_sarm_reward_recipe_transcribes_the_frozen_contract() -> None:
    contract = json.loads(
        (CONFIG_ROOT / "sarm_manifests/openarm2_shirt_fold_sarm_v1.json").read_text()
    )
    config = yaml.safe_load((CONFIG_ROOT / "shirt_fold_sarm_dense_v1.yaml").read_text())
    reward = config["reward_model"]
    assert reward["type"] == "sarm"
    assert reward["annotation_mode"] == "dense_only"
    assert reward["image_key"] == contract["image_key"]
    assert reward["state_key"] == contract["state_key"]
    assert reward["dense_subtask_names"] == contract["dense_stage_order"]
    assert reward["dense_temporal_proportions"] == list(
        contract["temporal_proportions"].values()
    )
    split = contract["split"]
    assert config["dataset"]["revision"] == "a631469960ec5b983eb43e430c39ffc621f7c23b"
    assert config["dataset"]["episodes"] == split["train"] + split["validation"] + split["test"]
    assert config["dataset"]["eval_split"] == 0.18
    assert config["steps"] == 1200


def test_rabc_recipe_is_isolated_and_fails_closed_until_audit_resolution() -> None:
    control = yaml.safe_load(
        (CONFIG_ROOT / "shirt_fold_pi05_hf_phase_aligned.yaml").read_text()
    )
    rabc = yaml.safe_load(
        (CONFIG_ROOT / "shirt_fold_pi05_hf_phase_aligned_sarm_rabc_v1.yaml").read_text()
    )
    assert "sample_weighting" not in control
    weighting = rabc["sample_weighting"]
    assert weighting["type"] == "rabc"
    assert weighting["head_mode"] == "dense"
    assert weighting["extra_params"]["audit_sha256"] == 0
    assert weighting["extra_params"]["source_progress_sha256"] == 0
    assert weighting["extra_params"]["training_progress_sha256"] == 0
    assert weighting["progress_path"].endswith("sarm_progress_train.parquet")
    assert rabc["dataset"]["root"].endswith("-sarm-v1")
    assert "priority_sampling_manifest" not in rabc

    for config in (control, rabc):
        config.pop("output_dir")
        config.pop("job_name")
        config.pop("dataset")
        config.pop("wandb")
    rabc.pop("sample_weighting")
    assert rabc == control
