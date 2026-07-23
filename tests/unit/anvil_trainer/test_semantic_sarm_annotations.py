"""Contracts for the reviewed five-stage released-SARM derivative."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil_trainer.semantic_sarm_annotations import (
    SemanticSARMContract,
    _episode_rows,
    compute_temporal_proportions,
)
from anvil_trainer.semantic_stages import SemanticStageError, SemanticStageManifest

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "configs/training/semantic_manifests/openarm2_shirt_fold_5stage_v1.json"
REVIEW_PATH = ROOT / "configs/training/semantic_reviews/openarm2_shirt_fold_5stage_review_v1.json"
CONTRACT_PATH = ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_5stage_v1.json"


def test_checked_in_contract_pins_review_split_and_duration_masses() -> None:
    manifest = SemanticStageManifest.load(MANIFEST_PATH)
    contract = SemanticSARMContract.load(
        CONTRACT_PATH,
        manifest=manifest,
        review_path=REVIEW_PATH,
    )

    assert contract.dense_stage_order == manifest.stage_order
    assert len(contract.train_episodes) == 27
    assert len(contract.validation_episodes) == 3
    assert len(contract.test_episodes) == 3
    assert compute_temporal_proportions(manifest, contract.train_episodes) == pytest.approx(
        contract.temporal_proportions, abs=1e-12
    )
    assert sum(contract.temporal_proportions.values()) == pytest.approx(1.0)


def test_absent_optional_stage_uses_reversed_inclusive_interval() -> None:
    manifest = SemanticStageManifest.load(MANIFEST_PATH)
    rows = _episode_rows(manifest)
    absent = [2, 5, 10, 20, 22, 27]
    for episode_index in absent:
        names = rows[episode_index]["dense_subtask_names"]
        position = names.index("recenter_pull")
        starts = rows[episode_index]["dense_subtask_start_frames"]
        ends = rows[episode_index]["dense_subtask_end_frames"]
        assert ends[position] == starts[position] - 1


def test_contract_rejects_unreviewed_semantic_manifest(tmp_path: Path) -> None:
    review = json.loads(REVIEW_PATH.read_text())
    review["reviewed_episode_ids"] = list(range(32))
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    manifest = SemanticStageManifest.load(MANIFEST_PATH)

    with pytest.raises(SemanticStageError, match="cover every episode"):
        SemanticSARMContract.load(
            CONTRACT_PATH,
            manifest=manifest,
            review_path=review_path,
        )
