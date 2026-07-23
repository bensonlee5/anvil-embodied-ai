"""Validated five-stage OpenARM2 shirt-fold semantic annotations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEMANTIC_MANIFEST_SCHEMA = "openarm2.shirt-fold-semantic-segmentation.v1"


class SemanticStageError(ValueError):
    """Raised when the semantic-stage contract is incomplete or stale."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_keys(data: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = required - set(data)
    extra = set(data) - required
    if missing or extra:
        raise SemanticStageError(
            f"{context} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


@dataclass(frozen=True)
class SemanticStage:
    name: str
    present: bool
    start_frame: int
    end_frame: int
    segmentation_confidence: str
    boundary_source: str


@dataclass(frozen=True)
class SemanticEpisode:
    episode_index: int
    frame_count: int
    stages: tuple[SemanticStage, ...]


@dataclass(frozen=True)
class SemanticStageManifest:
    """Immutable semantic-stage source used by the five-stage SARM derivative."""

    path: Path
    sha256: str
    dataset: Mapping[str, Any]
    stage_order: tuple[str, ...]
    optional_stages: tuple[str, ...]
    episodes: tuple[SemanticEpisode, ...]

    @classmethod
    def load(cls, path: str | Path) -> SemanticStageManifest:
        source = Path(path).expanduser().resolve()
        try:
            data = json.loads(source.read_text())
        except FileNotFoundError as exc:
            raise SemanticStageError(f"Semantic manifest not found: {source}") from exc
        except json.JSONDecodeError as exc:
            raise SemanticStageError(f"Semantic manifest is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise SemanticStageError("Semantic manifest root must be an object")
        _require_keys(
            data,
            {
                "schema_version",
                "description",
                "dataset",
                "stage_order",
                "optional_stages",
                "outcome_order",
                "generation_contract",
                "episodes",
                "provenance",
            },
            "semantic manifest",
        )
        if data["schema_version"] != SEMANTIC_MANIFEST_SCHEMA:
            raise SemanticStageError(
                f"Unsupported semantic manifest schema: {data['schema_version']!r}"
            )
        stage_order = tuple(data["stage_order"])
        if stage_order != (
            "side_one",
            "recenter_pull",
            "side_two",
            "strip_refinement",
            "bottom_to_top",
        ):
            raise SemanticStageError(f"Unexpected semantic stage order: {stage_order}")
        optional_stages = tuple(data["optional_stages"])
        if optional_stages != ("recenter_pull", "strip_refinement"):
            raise SemanticStageError(f"Unexpected optional stages: {optional_stages}")
        if data["outcome_order"] != ["side_one", "side_two", "bottom_to_top"]:
            raise SemanticStageError("Quality outcomes must remain separate from motion stages")

        dataset = data["dataset"]
        _require_keys(
            dataset,
            {"repo_id", "revision", "episodes", "frames", "fps", "fingerprints"},
            "semantic dataset",
        )
        if dataset["episodes"] != 33 or dataset["frames"] != 34_850 or dataset["fps"] != 30:
            raise SemanticStageError("Semantic dataset identity is not the 33 trimmed episodes")
        fingerprints = dataset["fingerprints"]
        if not isinstance(fingerprints, dict) or not fingerprints:
            raise SemanticStageError("Semantic dataset fingerprints are required")

        episodes = tuple(
            cls._parse_episode(raw, index=index, stage_order=stage_order, optional=optional_stages)
            for index, raw in enumerate(data["episodes"])
        )
        if [episode.episode_index for episode in episodes] != list(range(33)):
            raise SemanticStageError("Semantic episodes must be complete and ordered 0..32")
        if sum(episode.frame_count for episode in episodes) != dataset["frames"]:
            raise SemanticStageError("Semantic episode frame count does not match dataset")
        return cls(
            path=source,
            sha256=file_sha256(source),
            dataset=dataset,
            stage_order=stage_order,
            optional_stages=optional_stages,
            episodes=episodes,
        )

    @staticmethod
    def _parse_episode(
        raw: Any,
        *,
        index: int,
        stage_order: tuple[str, ...],
        optional: tuple[str, ...],
    ) -> SemanticEpisode:
        if not isinstance(raw, dict):
            raise SemanticStageError(f"episodes[{index}] must be an object")
        _require_keys(
            raw,
            {"episode_index", "frame_count", "stages", "outcomes", "boundary_evidence"},
            f"episodes[{index}]",
        )
        if raw["episode_index"] != index:
            raise SemanticStageError(f"episodes[{index}] has the wrong episode_index")
        frame_count = raw["frame_count"]
        if not isinstance(frame_count, int) or frame_count <= 0:
            raise SemanticStageError(f"episode {index} frame_count must be positive")
        if len(raw["stages"]) != len(stage_order):
            raise SemanticStageError(f"episode {index} must contain five semantic stages")
        cursor = 0
        stages: list[SemanticStage] = []
        for position, stage_raw in enumerate(raw["stages"]):
            _require_keys(
                stage_raw,
                {
                    "name",
                    "present",
                    "start_frame",
                    "end_frame",
                    "segmentation_confidence",
                    "boundary_source",
                },
                f"episode {index} stage {position}",
            )
            name = stage_raw["name"]
            start = stage_raw["start_frame"]
            end = stage_raw["end_frame"]
            present = stage_raw["present"]
            if name != stage_order[position] or start != cursor:
                raise SemanticStageError(
                    f"episode {index} stages have a gap, overlap, or order error"
                )
            if not isinstance(start, int) or not isinstance(end, int) or end < start:
                raise SemanticStageError(f"episode {index} stage {name} has invalid bounds")
            if not isinstance(present, bool) or present != (end > start):
                raise SemanticStageError(
                    f"episode {index} stage {name} presence disagrees with bounds"
                )
            if not present and name not in optional:
                raise SemanticStageError(f"episode {index} required stage {name} is absent")
            confidence = stage_raw["segmentation_confidence"]
            if confidence not in {"low", "medium", "high"}:
                raise SemanticStageError(f"episode {index} stage {name} confidence is invalid")
            source = stage_raw["boundary_source"]
            if not isinstance(source, str) or not source:
                raise SemanticStageError(f"episode {index} stage {name} boundary source is empty")
            stages.append(
                SemanticStage(
                    name=name,
                    present=present,
                    start_frame=start,
                    end_frame=end,
                    segmentation_confidence=confidence,
                    boundary_source=source,
                )
            )
            cursor = end
        if cursor != frame_count:
            raise SemanticStageError(f"episode {index} stages do not cover every frame")
        return SemanticEpisode(index, frame_count, tuple(stages))

    def verify_dataset(self, root: str | Path) -> None:
        dataset_root = Path(root).expanduser().resolve()
        for relative, expected in self.dataset["fingerprints"].items():
            path = dataset_root / relative
            if not path.is_file():
                raise SemanticStageError(f"Semantic dataset file is missing: {path}")
            actual = file_sha256(path)
            if actual != expected:
                raise SemanticStageError(
                    f"Semantic dataset fingerprint mismatch for {relative}: {actual} != {expected}"
                )

    def episode(self, episode_index: int) -> SemanticEpisode:
        return self.episodes[episode_index]
