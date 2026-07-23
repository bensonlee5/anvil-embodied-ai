"""Five-stage semantic annotations for released single-task SARM."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from anvil_trainer.semantic_stages import (
    SemanticStageError,
    SemanticStageManifest,
    file_sha256,
)

SEMANTIC_SARM_SCHEMA = "openarm2.sarm-semantic-dataset.v1"
SEMANTIC_SARM_METADATA = "semantic_sarm_annotation_contract.json"
SEMANTIC_MANIFEST_COPY = "openarm2_semantic_manifest.json"
SEMANTIC_REVIEW_COPY = "openarm2_semantic_review.json"
SEMANTIC_SARM_SPLIT = "semantic_sarm_split_info.json"
SEMANTIC_SARM_PROPORTIONS = "temporal_proportions_dense.json"

_SARM_COLUMNS = (
    "dense_subtask_names",
    "dense_subtask_start_times",
    "dense_subtask_end_times",
    "dense_subtask_start_frames",
    "dense_subtask_end_frames",
)


def _require_keys(data: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = required - set(data)
    extra = set(data) - required
    if missing or extra:
        raise SemanticStageError(
            f"{context} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def load_semantic_review(
    path: str | Path,
    *,
    manifest: SemanticStageManifest,
) -> tuple[Path, str, Mapping[str, Any]]:
    review_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(review_path.read_text())
    except FileNotFoundError as exc:
        raise SemanticStageError(f"Semantic review not found: {review_path}") from exc
    if data.get("schema_version") != "openarm2.shirt-fold-semantic-review.v1":
        raise SemanticStageError("Semantic review schema is unsupported")
    if data.get("semantic_manifest_sha256") != manifest.sha256:
        raise SemanticStageError("Semantic review targets a different manifest")
    if data.get("reviewed_episode_ids") != list(range(33)):
        raise SemanticStageError("Semantic review must cover every episode")
    gate = data.get("gate")
    if not isinstance(gate, dict) or gate.get("status") != "pass_for_reward_model_screen_only":
        raise SemanticStageError("Semantic review has not passed the reward-model screen gate")
    expected_absent = {
        stage: [
            episode.episode_index
            for episode in manifest.episodes
            if not next(item for item in episode.stages if item.name == stage).present
        ]
        for stage in manifest.optional_stages
    }
    review_absent = data.get("optional_stage_audit", {})
    if review_absent != {
        "recenter_pull_absent_episode_ids": expected_absent["recenter_pull"],
        "strip_refinement_absent_episode_ids": expected_absent["strip_refinement"],
    }:
        raise SemanticStageError("Semantic review optional-stage audit is stale")
    return review_path, file_sha256(review_path), data


def compute_temporal_proportions(
    manifest: SemanticStageManifest,
    episode_indices: Sequence[int],
) -> dict[str, float]:
    """Compute released-SARM duration proportions over reward training only."""
    if not episode_indices or len(episode_indices) != len(set(episode_indices)):
        raise SemanticStageError("episode_indices must be unique and non-empty")
    totals = dict.fromkeys(manifest.stage_order, 0.0)
    for episode_index in episode_indices:
        episode = manifest.episode(episode_index)
        for stage in episode.stages:
            totals[stage.name] += (stage.end_frame - stage.start_frame) / episode.frame_count
    count = len(episode_indices)
    proportions = {name: totals[name] / count for name in manifest.stage_order}
    if not math.isclose(sum(proportions.values()), 1.0, abs_tol=1e-12):
        raise SemanticStageError("Five-stage temporal proportions do not sum to one")
    if not all(math.isfinite(value) and value > 0 for value in proportions.values()):
        raise SemanticStageError("Every global semantic stage needs positive train support")
    return proportions


@dataclass(frozen=True)
class SemanticSARMContract:
    path: Path
    sha256: str
    semantic_manifest_sha256: str
    semantic_review_sha256: str
    annotation_mode: str
    sparse_task: str
    dense_stage_order: tuple[str, ...]
    image_key: str
    state_key: str
    split_seed: int
    split_ratio: tuple[int, int, int]
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]
    test_episodes: tuple[int, ...]
    temporal_proportions: Mapping[str, float]
    optional_stage_policy: Mapping[str, str]
    progress_gate: Mapping[str, float]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        manifest: SemanticStageManifest,
        review_path: str | Path,
    ) -> SemanticSARMContract:
        contract_path = Path(path).expanduser().resolve()
        data = json.loads(contract_path.read_text())
        _require_keys(
            data,
            {
                "schema_version",
                "description",
                "semantic_manifest_sha256",
                "semantic_review_sha256",
                "annotation_mode",
                "sparse_task",
                "dense_stage_order",
                "image_key",
                "state_key",
                "frame_semantics",
                "split",
                "temporal_proportions",
                "optional_stage_policy",
                "behavior_labels",
                "progress_gate",
            },
            "semantic SARM contract",
        )
        if data["schema_version"] != SEMANTIC_SARM_SCHEMA:
            raise SemanticStageError(f"Unsupported semantic SARM schema: {data['schema_version']}")
        review_file, review_sha, _ = load_semantic_review(review_path, manifest=manifest)
        del review_file
        if data["semantic_manifest_sha256"] != manifest.sha256:
            raise SemanticStageError("SARM contract semantic-manifest hash is stale")
        if data["semantic_review_sha256"] != review_sha:
            raise SemanticStageError("SARM contract semantic-review hash is stale")
        if data["annotation_mode"] != "dense_only":
            raise SemanticStageError("Semantic SARM requires dense_only released SARM")
        stage_order = tuple(data["dense_stage_order"])
        if stage_order != manifest.stage_order:
            raise SemanticStageError("Semantic SARM stage order differs from the manifest")
        if data["frame_semantics"] != {
            "semantic_manifest_end": "exclusive",
            "lerobot_sarm_end": "inclusive",
            "conversion": "dense_end_frame=manifest_end_frame-1",
        }:
            raise SemanticStageError("Semantic SARM frame semantics are invalid")

        split = data["split"]
        _require_keys(split, {"seed", "ratio", "train", "validation", "test"}, "split")
        split_lists = {
            name: tuple(int(value) for value in split[name])
            for name in ("train", "validation", "test")
        }
        flattened = [
            episode for name in ("train", "validation", "test") for episode in split_lists[name]
        ]
        if sorted(flattened) != list(range(33)) or len(flattened) != len(set(flattened)):
            raise SemanticStageError("Semantic SARM split must partition episodes 0..32")
        ratio = tuple(split["ratio"])
        if split["seed"] != 1000 or ratio != (8, 1, 1):
            raise SemanticStageError("Semantic SARM must retain the seed-1000 8/1/1 split")

        proportions = {name: float(data["temporal_proportions"][name]) for name in stage_order}
        computed = compute_temporal_proportions(manifest, split_lists["train"])
        if set(data["temporal_proportions"]) != set(stage_order):
            raise SemanticStageError("Temporal-proportion keys do not match five stages")
        for name in stage_order:
            if not math.isclose(proportions[name], computed[name], abs_tol=1e-12):
                raise SemanticStageError(f"Temporal proportion for {name} is stale")

        optional_policy = data["optional_stage_policy"]
        if optional_policy != {
            "annotation_encoding": "zero_length_interval_with_inclusive_end_before_start",
            "raw_reward_behavior": "released_sarm_global_stage_skip",
            "training_progress_correction": "remove_absent_stage_mass_then_renormalize_per_episode",
        }:
            raise SemanticStageError("Optional-stage policy is not the audited five-stage policy")
        if data["behavior_labels"] != {
            "quality": "three_external_outcomes_not_sarm_targets",
            "repeated_grasps": "evaluation_only_not_failure_labels",
            "smoothing": "included_in_strip_refinement_not_speed_penalized",
        }:
            raise SemanticStageError("Semantic SARM behavior-label semantics are invalid")
        gate = {key: float(value) for key, value in data["progress_gate"].items()}
        required_gate = {
            "minimum_holdout_spearman",
            "maximum_holdout_mae",
            "maximum_stage_monotonicity_violation_rate",
            "maximum_corrected_optional_skip_jump",
        }
        if set(gate) != required_gate:
            raise SemanticStageError("Semantic SARM progress-gate keys are invalid")
        for field in ("sparse_task", "image_key", "state_key"):
            if not isinstance(data[field], str) or not data[field]:
                raise SemanticStageError(f"{field} must be non-empty")
        return cls(
            path=contract_path,
            sha256=file_sha256(contract_path),
            semantic_manifest_sha256=manifest.sha256,
            semantic_review_sha256=review_sha,
            annotation_mode=data["annotation_mode"],
            sparse_task=data["sparse_task"],
            dense_stage_order=stage_order,
            image_key=data["image_key"],
            state_key=data["state_key"],
            split_seed=1000,
            split_ratio=ratio,
            train_episodes=split_lists["train"],
            validation_episodes=split_lists["validation"],
            test_episodes=split_lists["test"],
            temporal_proportions=proportions,
            optional_stage_policy=optional_policy,
            progress_gate=gate,
        )

    @property
    def ordered_episodes(self) -> tuple[int, ...]:
        return self.train_episodes + self.validation_episodes + self.test_episodes

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": SEMANTIC_SARM_SCHEMA,
            "contract_sha256": self.sha256,
            "semantic_manifest_sha256": self.semantic_manifest_sha256,
            "semantic_review_sha256": self.semantic_review_sha256,
            "annotation_mode": self.annotation_mode,
            "sparse_task": self.sparse_task,
            "dense_stage_order": list(self.dense_stage_order),
            "image_key": self.image_key,
            "state_key": self.state_key,
            "frame_semantics": {
                "semantic_manifest_end": "exclusive",
                "lerobot_sarm_end": "inclusive",
                "conversion": "dense_end_frame=manifest_end_frame-1",
            },
            "split": {
                "seed": self.split_seed,
                "ratio": list(self.split_ratio),
                "train": list(self.train_episodes),
                "validation": list(self.validation_episodes),
                "test": list(self.test_episodes),
            },
            "temporal_proportions": dict(self.temporal_proportions),
            "optional_stage_policy": dict(self.optional_stage_policy),
            "behavior_labels": {
                "quality": "three_external_outcomes_not_sarm_targets",
                "repeated_grasps": "evaluation_only_not_failure_labels",
                "smoothing": "included_in_strip_refinement_not_speed_penalized",
            },
            "progress_gate": dict(self.progress_gate),
        }


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _episode_rows(manifest: SemanticStageManifest) -> dict[int, dict[str, list[Any]]]:
    fps = float(manifest.dataset["fps"])
    rows: dict[int, dict[str, list[Any]]] = {}
    for episode in manifest.episodes:
        inclusive_ends = [stage.end_frame - 1 for stage in episode.stages]
        rows[episode.episode_index] = {
            "dense_subtask_names": [stage.name for stage in episode.stages],
            "dense_subtask_start_times": [stage.start_frame / fps for stage in episode.stages],
            "dense_subtask_end_times": [frame / fps for frame in inclusive_ends],
            "dense_subtask_start_frames": [stage.start_frame for stage in episode.stages],
            "dense_subtask_end_frames": inclusive_ends,
        }
    return rows


def _annotate_episode_metadata(root: Path, manifest: SemanticStageManifest) -> Path:
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        raise SemanticStageError(f"No LeRobot episode metadata found under {root}")
    rows = _episode_rows(manifest)
    seen: set[int] = set()
    for episode_path in episode_files:
        frame = pd.read_parquet(episode_path)
        for column in _SARM_COLUMNS:
            frame[column] = None
        for row_index, episode_index in enumerate(frame["episode_index"].tolist()):
            episode_index = int(episode_index)
            expected = rows.get(episode_index)
            if expected is None:
                raise SemanticStageError(f"Dataset contains unexpected episode {episode_index}")
            for column, value in expected.items():
                frame.at[row_index, column] = value
            seen.add(episode_index)
        temporary = episode_path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, engine="pyarrow", compression="snappy", index=False)
        temporary.replace(episode_path)
    if seen != set(range(33)):
        raise SemanticStageError("Semantic SARM annotations do not cover all episodes")
    return episode_files[0]


def materialize_semantic_sarm_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    manifest: SemanticStageManifest,
    review_path: str | Path,
    contract: SemanticSARMContract,
) -> dict[str, Any]:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if source == output or output.exists():
        raise SemanticStageError("Semantic SARM output must be a new dataset path")
    manifest.verify_dataset(source)
    review_file, _, _ = load_semantic_review(review_path, manifest=manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary, copy_function=_hardlink_or_copy)
        episode_path = _annotate_episode_metadata(temporary, manifest)
        meta = temporary / "meta"
        (meta / SEMANTIC_SARM_PROPORTIONS).write_text(
            json.dumps(dict(contract.temporal_proportions), indent=2) + "\n"
        )
        (meta / SEMANTIC_MANIFEST_COPY).write_bytes(manifest.path.read_bytes())
        (meta / SEMANTIC_REVIEW_COPY).write_bytes(review_file.read_bytes())
        (meta / SEMANTIC_SARM_SPLIT).write_text(
            json.dumps(contract.provenance()["split"], indent=2) + "\n"
        )
        metadata = contract.provenance()
        metadata["source_dataset"] = dict(manifest.dataset)
        metadata["annotated_episode_metadata_sha256"] = file_sha256(episode_path)
        (meta / SEMANTIC_SARM_METADATA).write_text(json.dumps(metadata, indent=2) + "\n")
        validate_semantic_sarm_dataset(
            temporary,
            manifest=manifest,
            review_path=review_path,
            contract=contract,
        )
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_semantic_sarm_dataset(
        output,
        manifest=manifest,
        review_path=review_path,
        contract=contract,
    )


def validate_semantic_sarm_dataset(
    root: str | Path,
    *,
    manifest: SemanticStageManifest,
    review_path: str | Path,
    contract: SemanticSARMContract,
) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    manifest.verify_dataset(dataset_root)
    review_file, review_sha, _ = load_semantic_review(review_path, manifest=manifest)
    meta = dataset_root / "meta"
    required = {
        SEMANTIC_SARM_METADATA,
        SEMANTIC_MANIFEST_COPY,
        SEMANTIC_REVIEW_COPY,
        SEMANTIC_SARM_SPLIT,
        SEMANTIC_SARM_PROPORTIONS,
    }
    for name in required:
        if not (meta / name).is_file():
            raise SemanticStageError(f"Semantic SARM metadata is missing: {name}")
    if file_sha256(meta / SEMANTIC_MANIFEST_COPY) != manifest.sha256:
        raise SemanticStageError("Copied semantic manifest hash is stale")
    if file_sha256(meta / SEMANTIC_REVIEW_COPY) != review_sha:
        raise SemanticStageError("Copied semantic review hash is stale")
    if json.loads((meta / SEMANTIC_SARM_SPLIT).read_text()) != contract.provenance()["split"]:
        raise SemanticStageError("Semantic SARM split metadata is stale")
    if json.loads((meta / SEMANTIC_SARM_PROPORTIONS).read_text()) != dict(
        contract.temporal_proportions
    ):
        raise SemanticStageError("Semantic SARM temporal proportions are stale")

    rows = _episode_rows(manifest)
    seen: set[int] = set()
    episode_files = sorted((meta / "episodes").glob("chunk-*/file-*.parquet"))
    for episode_path in episode_files:
        frame = pd.read_parquet(episode_path)
        for row_index, episode_index in enumerate(frame["episode_index"].tolist()):
            episode_index = int(episode_index)
            expected = rows[episode_index]
            for column, value in expected.items():
                if list(frame.at[row_index, column]) != value:
                    raise SemanticStageError(f"Episode {episode_index} {column} is stale")
            seen.add(episode_index)
    if seen != set(range(33)):
        raise SemanticStageError("Semantic SARM episode annotations are incomplete")
    metadata = json.loads((meta / SEMANTIC_SARM_METADATA).read_text())
    for key, expected in contract.provenance().items():
        if metadata.get(key) != expected:
            raise SemanticStageError(f"Semantic SARM metadata field {key} is stale")
    if metadata.get("source_dataset") != dict(manifest.dataset):
        raise SemanticStageError("Semantic SARM source dataset provenance is stale")
    first_episode = episode_files[0]
    if metadata.get("annotated_episode_metadata_sha256") != file_sha256(first_episode):
        raise SemanticStageError("Semantic SARM annotated episode hash is stale")
    return {
        "root": str(dataset_root),
        "episodes": len(seen),
        "frames": int(manifest.dataset["frames"]),
        "semantic_manifest_sha256": manifest.sha256,
        "semantic_review_sha256": file_sha256(review_file),
        "semantic_sarm_contract_sha256": contract.sha256,
        "temporal_proportions": dict(contract.temporal_proportions),
        "split": contract.provenance()["split"],
        "annotated_episode_metadata_sha256": file_sha256(first_episode),
    }
