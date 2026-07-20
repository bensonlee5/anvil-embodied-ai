"""Immutable OpenARM2 stage annotations for native LeRobot SARM training.

The reviewed priority manifest uses ordinary Python slice semantics: stage end
frames are exclusive.  LeRobot SARM uses inclusive stage end frames.  This
module owns that one-way conversion and verifies the derived dataset before it
can be used for reward-model training.
"""

from __future__ import annotations

import hashlib
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

from anvil_trainer.priority_sampling import PriorityManifest

SARM_CONTRACT_SCHEMA = "openarm2.sarm-dataset.v1"
SARM_METADATA_NAME = "sarm_annotation_contract.json"
SARM_PRIORITY_MANIFEST_NAME = "openarm2_priority_manifest.json"
SARM_SPLIT_NAME = "sarm_split_info.json"
SARM_TEMPORAL_PROPORTIONS_NAME = "temporal_proportions_dense.json"

_SARM_COLUMNS = (
    "dense_subtask_names",
    "dense_subtask_start_times",
    "dense_subtask_end_times",
    "dense_subtask_start_frames",
    "dense_subtask_end_frames",
)


class SARMAnnotationError(ValueError):
    """Raised when the frozen SARM annotation contract is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_keys(data: Mapping[str, Any], required: set[str], context: str) -> None:
    actual = set(data)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        raise SARMAnnotationError(
            f"{context} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


@dataclass(frozen=True)
class SARMAnnotationContract:
    """Frozen training/evaluation split and native SARM transcription rules."""

    path: Path
    sha256: str
    priority_manifest_sha256: str
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

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        priority_manifest: PriorityManifest,
    ) -> SARMAnnotationContract:
        contract_path = Path(path).expanduser().resolve()
        try:
            data = json.loads(contract_path.read_text())
        except FileNotFoundError as exc:
            raise SARMAnnotationError(f"SARM contract not found: {contract_path}") from exc
        except json.JSONDecodeError as exc:
            raise SARMAnnotationError(f"SARM contract is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise SARMAnnotationError("SARM contract root must be an object")
        _require_keys(
            data,
            {
                "schema_version",
                "description",
                "priority_manifest_sha256",
                "annotation_mode",
                "sparse_task",
                "dense_stage_order",
                "image_key",
                "state_key",
                "frame_semantics",
                "split",
                "temporal_proportions",
                "behavior_labels",
            },
            "SARM contract",
        )
        if data["schema_version"] != SARM_CONTRACT_SCHEMA:
            raise SARMAnnotationError(
                f"Unsupported SARM contract schema: {data['schema_version']!r}"
            )
        if data["priority_manifest_sha256"] != priority_manifest.sha256:
            raise SARMAnnotationError(
                "SARM contract priority-manifest hash does not match the loaded manifest"
            )
        if data["annotation_mode"] != "dense_only":
            raise SARMAnnotationError("OpenARM2 SARM v1 requires annotation_mode=dense_only")
        stage_order = tuple(data["dense_stage_order"])
        if stage_order != priority_manifest.stage_order:
            raise SARMAnnotationError(
                f"SARM stage order {stage_order} != priority manifest {priority_manifest.stage_order}"
            )
        _require_keys(
            data["frame_semantics"],
            {"priority_manifest_end", "lerobot_sarm_end", "conversion"},
            "frame_semantics",
        )
        if data["frame_semantics"] != {
            "priority_manifest_end": "exclusive",
            "lerobot_sarm_end": "inclusive",
            "conversion": "dense_end_frame=manifest_end_frame-1",
        }:
            raise SARMAnnotationError("SARM frame semantics do not declare the exact v1 conversion")

        split = data["split"]
        _require_keys(
            split,
            {"seed", "ratio", "train", "validation", "test"},
            "split",
        )
        split_lists = {
            name: tuple(int(value) for value in split[name])
            for name in ("train", "validation", "test")
        }
        flattened = [
            episode
            for name in ("train", "validation", "test")
            for episode in split_lists[name]
        ]
        expected = list(range(len(priority_manifest.episodes)))
        if sorted(flattened) != expected or len(flattened) != len(set(flattened)):
            raise SARMAnnotationError("SARM split must partition every episode exactly once")
        ratio = tuple(split["ratio"])
        if ratio != (8, 1, 1):
            raise SARMAnnotationError(f"SARM v1 split ratio must be [8, 1, 1], got {ratio}")

        proportions = data["temporal_proportions"]
        if not isinstance(proportions, dict) or tuple(proportions) != stage_order:
            raise SARMAnnotationError(
                "temporal_proportions keys must exactly follow dense_stage_order"
            )
        parsed_proportions = {name: float(proportions[name]) for name in stage_order}
        if not all(math.isfinite(value) and value > 0 for value in parsed_proportions.values()):
            raise SARMAnnotationError("temporal proportions must be finite and positive")
        if not math.isclose(sum(parsed_proportions.values()), 1.0, abs_tol=1e-12):
            raise SARMAnnotationError("temporal proportions must sum to one")
        computed = compute_temporal_proportions(
            priority_manifest,
            split_lists["train"],
        )
        for name in stage_order:
            if not math.isclose(
                parsed_proportions[name], computed[name], rel_tol=0.0, abs_tol=1e-12
            ):
                raise SARMAnnotationError(
                    f"temporal proportion for {name} is stale: "
                    f"contract={parsed_proportions[name]}, computed={computed[name]}"
                )

        behavior = data["behavior_labels"]
        _require_keys(
            behavior,
            {"quality", "repeated_grasps", "smoothing"},
            "behavior_labels",
        )
        if behavior != {
            "quality": "external_stage_prior_not_sarm_target",
            "repeated_grasps": "evaluation_only_not_failure_labels",
            "smoothing": "coarse_review_windows_priority_neutral",
        }:
            raise SARMAnnotationError("behavior label semantics do not match SARM v1")

        for field in ("sparse_task", "image_key", "state_key"):
            if not isinstance(data[field], str) or not data[field]:
                raise SARMAnnotationError(f"{field} must be a non-empty string")

        return cls(
            path=contract_path,
            sha256=_sha256(contract_path),
            priority_manifest_sha256=priority_manifest.sha256,
            annotation_mode=data["annotation_mode"],
            sparse_task=data["sparse_task"],
            dense_stage_order=stage_order,
            image_key=data["image_key"],
            state_key=data["state_key"],
            split_seed=int(split["seed"]),
            split_ratio=ratio,
            train_episodes=split_lists["train"],
            validation_episodes=split_lists["validation"],
            test_episodes=split_lists["test"],
            temporal_proportions=parsed_proportions,
        )

    @property
    def ordered_episodes(self) -> tuple[int, ...]:
        """Episode order that makes LeRobot's tail eval split exact."""
        return self.train_episodes + self.validation_episodes + self.test_episodes

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": SARM_CONTRACT_SCHEMA,
            "contract_sha256": self.sha256,
            "priority_manifest_sha256": self.priority_manifest_sha256,
            "annotation_mode": self.annotation_mode,
            "sparse_task": self.sparse_task,
            "dense_stage_order": list(self.dense_stage_order),
            "image_key": self.image_key,
            "state_key": self.state_key,
            "frame_semantics": {
                "priority_manifest_end": "exclusive",
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
            "behavior_labels": {
                "quality": "external_stage_prior_not_sarm_target",
                "repeated_grasps": "evaluation_only_not_failure_labels",
                "smoothing": "coarse_review_windows_priority_neutral",
            },
        }


def compute_temporal_proportions(
    manifest: PriorityManifest,
    episode_indices: Sequence[int],
) -> dict[str, float]:
    """Compute SARM formula (1) over only the reward-model training episodes."""
    if not episode_indices or len(episode_indices) != len(set(episode_indices)):
        raise SARMAnnotationError("episode_indices must contain unique episodes")
    if any(index < 0 or index >= len(manifest.episodes) for index in episode_indices):
        raise SARMAnnotationError("episode_indices contains an out-of-range episode")
    totals = dict.fromkeys(manifest.stage_order, 0.0)
    for episode_index in episode_indices:
        episode = manifest.episodes[episode_index]
        for stage in episode.stages:
            totals[stage.name] += (stage.end_frame - stage.start_frame) / episode.frame_count
    count = len(episode_indices)
    return {name: totals[name] / count for name in manifest.stage_order}


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _episode_rows(manifest: PriorityManifest) -> dict[int, dict[str, list[Any]]]:
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


def _annotate_episode_metadata(root: Path, manifest: PriorityManifest) -> Path:
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        raise SARMAnnotationError(f"No LeRobot episode metadata found under {root}")
    rows = _episode_rows(manifest)
    seen: set[int] = set()
    for episode_path in episode_files:
        frame = pd.read_parquet(episode_path)
        for column in _SARM_COLUMNS:
            frame[column] = None
        for row_index, episode_index in enumerate(frame["episode_index"].tolist()):
            episode_index = int(episode_index)
            if episode_index not in rows:
                raise SARMAnnotationError(
                    f"Dataset contains unannotated episode {episode_index} in {episode_path}"
                )
            for column, value in rows[episode_index].items():
                frame.at[row_index, column] = value
            seen.add(episode_index)
        temporary = episode_path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, engine="pyarrow", compression="snappy", index=False)
        temporary.replace(episode_path)
    expected = set(range(len(manifest.episodes)))
    if seen != expected:
        raise SARMAnnotationError(
            f"Episode metadata coverage mismatch: missing={sorted(expected-seen)}"
        )
    return episode_files[0]


def materialize_sarm_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    manifest: PriorityManifest,
    contract: SARMAnnotationContract,
) -> dict[str, Any]:
    """Create a revision-safe annotated dataset without mutating the source tree."""
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if source == output:
        raise SARMAnnotationError("SARM output_root must differ from source_root")
    manifest.verify_dataset(source)
    if output.exists():
        raise SARMAnnotationError(
            f"SARM output already exists: {output}; use --check instead of overwriting"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary, copy_function=_hardlink_or_copy)
        episode_path = _annotate_episode_metadata(temporary, manifest)
        meta_root = temporary / "meta"
        (meta_root / SARM_TEMPORAL_PROPORTIONS_NAME).write_text(
            json.dumps(dict(contract.temporal_proportions), indent=2) + "\n"
        )
        (meta_root / SARM_PRIORITY_MANIFEST_NAME).write_bytes(manifest.path.read_bytes())
        (meta_root / SARM_SPLIT_NAME).write_text(
            json.dumps(contract.provenance()["split"], indent=2) + "\n"
        )
        metadata = contract.provenance()
        metadata["source_dataset"] = dict(manifest.dataset)
        metadata["annotated_episode_metadata_sha256"] = _sha256(episode_path)
        (meta_root / SARM_METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n")
        validate_sarm_dataset(temporary, manifest=manifest, contract=contract)
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_sarm_dataset(output, manifest=manifest, contract=contract)


def validate_sarm_dataset(
    root: str | Path,
    *,
    manifest: PriorityManifest,
    contract: SARMAnnotationContract,
) -> dict[str, Any]:
    """Fail closed unless every native SARM annotation and provenance field matches."""
    dataset_root = Path(root).expanduser().resolve()
    manifest.verify_dataset(dataset_root)
    metadata_path = dataset_root / "meta" / SARM_METADATA_NAME
    priority_copy = dataset_root / "meta" / SARM_PRIORITY_MANIFEST_NAME
    split_path = dataset_root / "meta" / SARM_SPLIT_NAME
    proportions_path = dataset_root / "meta" / SARM_TEMPORAL_PROPORTIONS_NAME
    for path in (metadata_path, priority_copy, split_path, proportions_path):
        if not path.is_file():
            raise SARMAnnotationError(f"Required SARM metadata is missing: {path}")
    if _sha256(priority_copy) != manifest.sha256:
        raise SARMAnnotationError("Copied priority manifest hash does not match")
    if json.loads(split_path.read_text()) != contract.provenance()["split"]:
        raise SARMAnnotationError("Derived SARM split metadata is stale")
    stored_proportions = json.loads(proportions_path.read_text())
    if stored_proportions != dict(contract.temporal_proportions):
        raise SARMAnnotationError("Derived SARM temporal proportions are stale")

    rows = _episode_rows(manifest)
    seen: set[int] = set()
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    for episode_path in episode_files:
        frame = pd.read_parquet(episode_path)
        missing = [column for column in _SARM_COLUMNS if column not in frame.columns]
        if missing:
            raise SARMAnnotationError(
                f"SARM columns missing from {episode_path}: {missing}"
            )
        for row_index, episode_index in enumerate(frame["episode_index"].tolist()):
            episode_index = int(episode_index)
            expected = rows.get(episode_index)
            if expected is None:
                raise SARMAnnotationError(f"Unexpected episode {episode_index}")
            for column, value in expected.items():
                actual = frame.at[row_index, column]
                if list(actual) != value:
                    raise SARMAnnotationError(
                        f"Episode {episode_index} {column} mismatch: {list(actual)} != {value}"
                    )
            seen.add(episode_index)
    if seen != set(rows):
        raise SARMAnnotationError("Derived SARM annotations do not cover every episode")

    metadata = json.loads(metadata_path.read_text())
    expected_metadata = contract.provenance()
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise SARMAnnotationError(f"SARM metadata field {key!r} is stale")
    if metadata.get("source_dataset") != dict(manifest.dataset):
        raise SARMAnnotationError("SARM source dataset provenance is stale")
    first_episode_file = episode_files[0]
    if metadata.get("annotated_episode_metadata_sha256") != _sha256(first_episode_file):
        raise SARMAnnotationError("Annotated episode metadata hash is stale")

    return {
        "root": str(dataset_root),
        "episodes": len(seen),
        "frames": int(manifest.dataset["frames"]),
        "priority_manifest_sha256": manifest.sha256,
        "sarm_contract_sha256": contract.sha256,
        "temporal_proportions": dict(contract.temporal_proportions),
        "split": contract.provenance()["split"],
        "annotated_episode_metadata_sha256": _sha256(first_episode_file),
    }
