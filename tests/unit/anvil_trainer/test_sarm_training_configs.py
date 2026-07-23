"""Frozen SARM reward-model and isolated RA-BC recipe contracts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "configs/training"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert config["dataset"]["video_backend"] == "torchcodec"
    assert config["steps"] == 1200


def test_five_stage_sarm_screen_transcribes_reviewed_semantics() -> None:
    contract = json.loads(
        (
            CONFIG_ROOT
            / "sarm_manifests/openarm2_shirt_fold_sarm_5stage_v1.json"
        ).read_text()
    )
    config = yaml.safe_load(
        (CONFIG_ROOT / "shirt_fold_sarm_5stage_dense_v1.yaml").read_text()
    )
    reward = config["reward_model"]
    split = contract["split"]

    assert config["dataset"] == {
        "repo_id": "bohlt/openarm2-shirt-fold-phase-aligned-sarm-5stage-v1",
        "revision": "1cc0cd37f070bbd34f22f4d821130842e08ae698",
        "root": "datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-5stage-v1",
        "episodes": split["train"] + split["validation"] + split["test"],
        "eval_split": 0.18,
        "video_backend": "torchcodec",
        "use_imagenet_stats": False,
        "image_transforms": {"enable": False},
    }
    assert reward["type"] == "sarm"
    assert reward["annotation_mode"] == "dense_only"
    assert reward["num_dense_stages"] == 5
    assert reward["dense_subtask_names"] == contract["dense_stage_order"]
    assert reward["dense_temporal_proportions"] == list(
        contract["temporal_proportions"].values()
    )
    assert reward["repo_id"] == "bohlt/openarm2-shirt-fold-sarm-5stage-v1"
    assert config["steps"] == 1200
    assert config["batch_size"] == 64
    assert config["save_freq"] == 200


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


def test_integrated_quality_sarm_candidate_is_fully_resolved() -> None:
    config = yaml.safe_load(
        (
            CONFIG_ROOT
            / "shirt_fold_pi05_hf_phase_aligned_quality_sarm_rabc_v2.yaml"
        ).read_text()
    )
    manifest_path = (
        CONFIG_ROOT / "priority_manifests/openarm2_shirt_fold_3stage_v2.json"
    )
    contract_path = CONFIG_ROOT / "sarm_manifests/openarm2_shirt_fold_sarm_v2.json"
    integration = json.loads(
        (
            CONFIG_ROOT
            / "quality_sarm_audits/openarm2_shirt_fold_quality_sarm_v2.json"
        ).read_text()
    )
    provenance = integration["provenance"]
    weighting = config["sample_weighting"]
    extra = weighting["extra_params"]

    assert config["dataset"]["root"].endswith("-sarm-v1")
    assert config["steps"] == 5000
    assert config["batch_size"] == 16
    assert config["seed"] == 1000
    assert weighting["type"] == "rabc"
    assert weighting["head_mode"] == "dense"
    assert weighting["kappa"] == 0.05090876221656798
    assert weighting["progress_path"].endswith("sarm_progress_train_v2.parquet")
    assert extra == {
        "audit_path": "sarm_progress_audit_v2.json",
        "audit_sha256": provenance["progress_audit_sha256"],
        "source_progress_sha256": provenance["source_progress_sha256"],
        "training_progress_sha256": provenance["training_progress_sha256"],
        "priority_manifest_sha256": _sha256(manifest_path),
        "sarm_contract_sha256": _sha256(contract_path),
    }
    assert provenance["reward_model"] == {
        "repo_id": "bohlt/openarm2-shirt-fold-sarm-v1",
        "revision": "108048371c101e77299b8b60ae5f214d30b295f2",
        "training_run_id": "train_reward_shirt_20260720_sarm_dense_v3",
        "wandb_run_id": "kttuwuef",
        "checkpoint_step": 1200,
        "reuse_justification": (
            "v2 changed only blinded quality labels; stage boundaries, dense targets, "
            "all 34,850 frames, and the 27/3/3 split are byte-for-byte equivalent"
        ),
    }
    assert integration["gates"]["pass"] is True
    assert integration["manual_sampling"]["effective_sample_size_fraction"] > 0.95
    assert integration["combined_expected_contribution"][
        "effective_sample_size_fraction"
    ] > 0.79
    assert abs(
        integration["combined_expected_contribution"][
            "quality_score_rabc_weight_correlation"
        ]
    ) < 0.05

    # Priority sampling is an Anvil CLI flag rather than a LeRobot YAML field.
    assert "priority_sampling_manifest" not in config

    control = yaml.safe_load(
        (CONFIG_ROOT / "shirt_fold_pi05_hf_phase_aligned.yaml").read_text()
    )
    candidate = deepcopy(config)
    for recipe in (control, candidate):
        recipe.pop("dataset")
        recipe.pop("output_dir")
        recipe.pop("job_name")
    candidate.pop("sample_weighting")
    assert candidate == control
