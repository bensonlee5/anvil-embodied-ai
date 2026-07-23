"""Tests for the five-stage optional-stage reward gate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from anvil_trainer.semantic_sarm_annotations import SemanticSARMContract
from anvil_trainer.semantic_sarm_progress import (
    audit_semantic_sarm_progress,
    expected_raw_progress,
)
from anvil_trainer.semantic_stages import SemanticStageManifest

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "configs/training/semantic_manifests/openarm2_shirt_fold_5stage_v1.json"
REVIEW_PATH = ROOT / "configs/training/semantic_reviews/openarm2_shirt_fold_5stage_review_v1.json"
CONTRACT_PATH = ROOT / "configs/training/sarm_manifests/openarm2_shirt_fold_sarm_5stage_v1.json"


def _contract() -> tuple[SemanticStageManifest, SemanticSARMContract]:
    manifest = SemanticStageManifest.load(MANIFEST_PATH)
    contract = SemanticSARMContract.load(
        CONTRACT_PATH,
        manifest=manifest,
        review_path=REVIEW_PATH,
    )
    return manifest, contract


def test_ideal_progress_passes_gate_and_writes_exact_train_split(tmp_path: Path) -> None:
    manifest, contract = _contract()
    frame = expected_raw_progress(manifest, contract).drop(columns="stage_name")
    raw_path = tmp_path / "raw.parquet"
    train_path = tmp_path / "train.parquet"
    frame.to_parquet(raw_path, index=False)

    result = audit_semantic_sarm_progress(
        raw_path,
        manifest=manifest,
        contract=contract,
        training_progress_path=train_path,
    )

    assert result["gate"]["passed"] is True
    assert result["training_progress"]["frames"] == 29_234
    assert len(result["optional_stage_corrected"]["optional_skip_boundaries"]) == 6
    assert result["optional_stage_corrected"]["maximum_absolute_optional_skip_jump"] < 1e-12
    metadata = pq.read_metadata(train_path).metadata
    assert metadata[b"optional_stage_correction"] == (
        b"remove_absent_stage_mass_then_renormalize_per_episode"
    )
    train = pd.read_parquet(train_path)
    assert sorted(train["episode_index"].unique()) == list(contract.train_episodes)


def test_optional_boundary_spike_fails_gate(tmp_path: Path) -> None:
    manifest, contract = _contract()
    frame = expected_raw_progress(manifest, contract).drop(columns="stage_name")
    episode = manifest.episode(2)
    recenter = next(stage for stage in episode.stages if stage.name == "recenter_pull")
    row = frame.index[
        (frame["episode_index"] == 2) & (frame["frame_index"] == recenter.start_frame)
    ][0]
    frame.loc[row, "progress_dense"] = 1.0
    raw_path = tmp_path / "raw.parquet"
    frame.to_parquet(raw_path, index=False)

    result = audit_semantic_sarm_progress(
        raw_path,
        manifest=manifest,
        contract=contract,
    )

    assert result["gate"]["checks"]["corrected_optional_skip_jump"] is False
    assert result["gate"]["passed"] is False
