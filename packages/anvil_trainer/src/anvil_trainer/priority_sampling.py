"""Deterministic frame-priority sampling for offline imitation learning.

The sampler implements the data-selection part of Larchenko's laundry system:
sample frames in proportion to an exponential priority while leaving the
behavior-cloning action loss unchanged.  The manifest is deliberately strict
and immutable so annotations cannot silently drift between runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


class PriorityManifestError(ValueError):
    """Raised when a priority-sampling manifest violates its contract."""


def _require_keys(data: Mapping[str, Any], *, required: set[str], context: str) -> None:
    actual = set(data)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        raise PriorityManifestError(
            f"{context} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PriorityManifestError(f"{context} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise PriorityManifestError(f"{context} must be finite, got {value!r}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StageAnnotation:
    name: str
    start_frame: int
    end_frame: int
    quality_score: int
    quality_confidence: str


@dataclass(frozen=True)
class RepeatedGraspAnnotation:
    gripper: str
    stage: str
    close_frame: int
    reopen_frame: int
    retry_frame: int
    penalty_start_frame: int
    penalty_end_frame: int
    confidence: str


@dataclass(frozen=True)
class EpisodeAnnotation:
    episode_index: int
    frame_count: int
    stages: tuple[StageAnnotation, ...]
    repeated_grasps: tuple[RepeatedGraspAnnotation, ...]
    smoothing: Mapping[str, Any]


@dataclass(frozen=True)
class PriorityManifest:
    """Validated, versioned source of frame priorities."""

    path: Path
    sha256: str
    schema_version: str
    dataset: Mapping[str, Any]
    stage_order: tuple[str, ...]
    quality_log_priority: Mapping[int, float]
    repeated_grasp_log_penalty: float
    smoothing_log_adjustment: float
    stage_probability_mass: Mapping[str, float]
    episodes: tuple[EpisodeAnnotation, ...]

    @classmethod
    def load(cls, path: str | Path) -> PriorityManifest:
        source_path = Path(path).expanduser().resolve()
        try:
            data = json.loads(source_path.read_text())
        except FileNotFoundError as exc:
            raise PriorityManifestError(f"Priority manifest not found: {source_path}") from exc
        except json.JSONDecodeError as exc:
            raise PriorityManifestError(
                f"Priority manifest is not valid JSON: {source_path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise PriorityManifestError("Priority manifest root must be a JSON object")
        _require_keys(
            data,
            required={
                "schema_version",
                "description",
                "dataset",
                "stage_order",
                "annotation_contract",
                "weighting",
                "episodes",
            },
            context="manifest",
        )
        if data["schema_version"] != "openarm2.priority-sampling.v1":
            raise PriorityManifestError(
                f"Unsupported priority manifest schema: {data['schema_version']!r}"
            )
        if not isinstance(data["description"], str) or not data["description"].strip():
            raise PriorityManifestError("description must be a non-empty string")

        dataset = data["dataset"]
        if not isinstance(dataset, dict):
            raise PriorityManifestError("dataset must be an object")
        _require_keys(
            dataset,
            required={
                "repo_id",
                "revision",
                "episodes",
                "frames",
                "fps",
                "fingerprints",
            },
            context="dataset",
        )
        fingerprints = dataset["fingerprints"]
        for field in ("repo_id", "revision"):
            if not isinstance(dataset[field], str) or not dataset[field]:
                raise PriorityManifestError(f"dataset.{field} must be a non-empty string")
        for field in ("episodes", "frames"):
            if isinstance(dataset[field], bool) or not isinstance(dataset[field], int):
                raise PriorityManifestError(f"dataset.{field} must be an integer")
            if dataset[field] <= 0:
                raise PriorityManifestError(f"dataset.{field} must be positive")
        if _finite_number(dataset["fps"], context="dataset.fps") <= 0:
            raise PriorityManifestError("dataset.fps must be positive")
        if not isinstance(fingerprints, dict) or not fingerprints:
            raise PriorityManifestError("dataset.fingerprints must be a non-empty object")
        for relative_path, expected_hash in fingerprints.items():
            if not isinstance(relative_path, str) or not relative_path:
                raise PriorityManifestError("dataset fingerprint paths must be non-empty strings")
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(char not in "0123456789abcdef" for char in expected_hash)
            ):
                raise PriorityManifestError(
                    f"dataset fingerprint for {relative_path!r} is not a lowercase SHA-256"
                )

        stage_order_raw = data["stage_order"]
        if (
            not isinstance(stage_order_raw, list)
            or not stage_order_raw
            or any(not isinstance(item, str) or not item for item in stage_order_raw)
            or len(stage_order_raw) != len(set(stage_order_raw))
        ):
            raise PriorityManifestError("stage_order must contain unique non-empty strings")
        stage_order = tuple(stage_order_raw)

        annotation_contract = data["annotation_contract"]
        if not isinstance(annotation_contract, dict):
            raise PriorityManifestError("annotation_contract must be an object")
        _require_keys(
            annotation_contract,
            required={
                "quality_scale",
                "repeated_grasp_definition",
                "smoothing_definition",
                "smoothing_affects_priority",
                "single_annotator",
            },
            context="annotation_contract",
        )
        if annotation_contract["quality_scale"] != [1, 5]:
            raise PriorityManifestError("annotation_contract.quality_scale must be [1, 5]")
        if annotation_contract["smoothing_affects_priority"] is not False:
            raise PriorityManifestError(
                "v1 requires smoothing_affects_priority=false; deliberate speed is quality-neutral"
            )

        weighting = data["weighting"]
        if not isinstance(weighting, dict):
            raise PriorityManifestError("weighting must be an object")
        _require_keys(
            weighting,
            required={
                "method",
                "loss_reweighting",
                "sampling_replacement",
                "quality_log_priority",
                "repeated_grasp_log_penalty",
                "smoothing_log_adjustment",
                "stage_probability_mass",
            },
            context="weighting",
        )
        if weighting["method"] != "exponential_priority_sampler":
            raise PriorityManifestError("weighting.method must be exponential_priority_sampler")
        if weighting["loss_reweighting"] is not False:
            raise PriorityManifestError("Behavior-cloning action loss must remain unweighted")
        if weighting["sampling_replacement"] is not True:
            raise PriorityManifestError("Priority sampling must use replacement")

        raw_quality = weighting["quality_log_priority"]
        if not isinstance(raw_quality, dict) or set(raw_quality) != {"1", "2", "3", "4", "5"}:
            raise PriorityManifestError("quality_log_priority must define scores 1 through 5")
        quality_log_priority = {
            int(score): _finite_number(value, context=f"quality_log_priority[{score}]")
            for score, value in raw_quality.items()
        }
        ordered_quality = [quality_log_priority[score] for score in range(1, 6)]
        if any(value < -2 or value > 2 for value in ordered_quality):
            raise PriorityManifestError("quality log-priorities must stay inside [-2, 2]")
        if any(left >= right for left, right in pairwise(ordered_quality)):
            raise PriorityManifestError("quality log-priorities must increase strictly with score")
        repeated_grasp_log_penalty = _finite_number(
            weighting["repeated_grasp_log_penalty"],
            context="repeated_grasp_log_penalty",
        )
        if not -2 <= repeated_grasp_log_penalty <= 0:
            raise PriorityManifestError("repeated_grasp_log_penalty must be inside [-2, 0]")
        smoothing_log_adjustment = _finite_number(
            weighting["smoothing_log_adjustment"],
            context="smoothing_log_adjustment",
        )
        if smoothing_log_adjustment != 0:
            raise PriorityManifestError("v1 smoothing_log_adjustment must be exactly 0")

        raw_stage_mass = weighting["stage_probability_mass"]
        if not isinstance(raw_stage_mass, dict) or set(raw_stage_mass) != set(stage_order):
            raise PriorityManifestError(
                "stage_probability_mass keys must exactly match stage_order"
            )
        stage_probability_mass = {
            stage: _finite_number(raw_stage_mass[stage], context=f"stage mass {stage}")
            for stage in stage_order
        }
        if any(value <= 0 for value in stage_probability_mass.values()):
            raise PriorityManifestError("Every stage probability mass must be > 0")

        raw_episodes = data["episodes"]
        if not isinstance(raw_episodes, list) or not raw_episodes:
            raise PriorityManifestError("episodes must be a non-empty list")
        episodes = tuple(
            cls._parse_episode(item, stage_order=stage_order, index=position)
            for position, item in enumerate(raw_episodes)
        )
        episode_indices = [item.episode_index for item in episodes]
        if episode_indices != list(range(len(episodes))):
            raise PriorityManifestError(
                "episodes must be complete and ordered from episode_index 0"
            )
        if dataset["episodes"] != len(episodes):
            raise PriorityManifestError(
                f"dataset episode count {dataset['episodes']} != annotations {len(episodes)}"
            )
        if dataset["frames"] != sum(item.frame_count for item in episodes):
            raise PriorityManifestError("dataset frame count does not match episode annotations")

        return cls(
            path=source_path,
            sha256=_sha256(source_path),
            schema_version=data["schema_version"],
            dataset=dataset,
            stage_order=stage_order,
            quality_log_priority=quality_log_priority,
            repeated_grasp_log_penalty=repeated_grasp_log_penalty,
            smoothing_log_adjustment=smoothing_log_adjustment,
            stage_probability_mass=stage_probability_mass,
            episodes=episodes,
        )

    @staticmethod
    def _parse_episode(
        raw: Any,
        *,
        stage_order: tuple[str, ...],
        index: int,
    ) -> EpisodeAnnotation:
        if not isinstance(raw, dict):
            raise PriorityManifestError(f"episodes[{index}] must be an object")
        _require_keys(
            raw,
            required={
                "episode_index",
                "frame_count",
                "stages",
                "repeated_grasps",
                "smoothing",
            },
            context=f"episodes[{index}]",
        )
        episode_index = raw["episode_index"]
        frame_count = raw["frame_count"]
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise PriorityManifestError(f"episodes[{index}].episode_index must be an integer")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise PriorityManifestError(f"episodes[{index}].frame_count must be positive")

        raw_stages = raw["stages"]
        if not isinstance(raw_stages, list) or len(raw_stages) != len(stage_order):
            raise PriorityManifestError(
                f"episode {episode_index} must contain exactly {len(stage_order)} stages"
            )
        stages: list[StageAnnotation] = []
        cursor = 0
        for stage_index, stage_raw in enumerate(raw_stages):
            if not isinstance(stage_raw, dict):
                raise PriorityManifestError(
                    f"episode {episode_index} stage {stage_index} must be an object"
                )
            _require_keys(
                stage_raw,
                required={
                    "name",
                    "start_frame",
                    "end_frame",
                    "quality_score",
                    "quality_confidence",
                },
                context=f"episode {episode_index} stage {stage_index}",
            )
            name = stage_raw["name"]
            start_frame = stage_raw["start_frame"]
            end_frame = stage_raw["end_frame"]
            quality_score = stage_raw["quality_score"]
            quality_confidence = stage_raw["quality_confidence"]
            if name != stage_order[stage_index]:
                raise PriorityManifestError(
                    f"episode {episode_index} stages are not in declared order"
                )
            if start_frame != cursor or not isinstance(end_frame, int) or end_frame <= start_frame:
                raise PriorityManifestError(
                    f"episode {episode_index} stage {name} has a gap, overlap, or empty range"
                )
            if quality_score not in {1, 2, 3, 4, 5}:
                raise PriorityManifestError(
                    f"episode {episode_index} stage {name} quality must be 1..5"
                )
            if quality_confidence not in {"low", "medium", "high"}:
                raise PriorityManifestError(
                    f"episode {episode_index} stage {name} confidence is invalid"
                )
            stages.append(
                StageAnnotation(
                    name=name,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    quality_score=quality_score,
                    quality_confidence=quality_confidence,
                )
            )
            cursor = end_frame
        if cursor != frame_count:
            raise PriorityManifestError(
                f"episode {episode_index} stages end at {cursor}, expected {frame_count}"
            )

        stage_for_frame = np.empty(frame_count, dtype=object)
        for stage in stages:
            stage_for_frame[stage.start_frame : stage.end_frame] = stage.name

        raw_repeated = raw["repeated_grasps"]
        if not isinstance(raw_repeated, list):
            raise PriorityManifestError(
                f"episode {episode_index}.repeated_grasps must be a list"
            )
        repeated: list[RepeatedGraspAnnotation] = []
        for event_index, event in enumerate(raw_repeated):
            if not isinstance(event, dict):
                raise PriorityManifestError(
                    f"episode {episode_index} repeated_grasps[{event_index}] must be an object"
                )
            _require_keys(
                event,
                required={
                    "gripper",
                    "stage",
                    "close_frame",
                    "reopen_frame",
                    "retry_frame",
                    "penalty_start_frame",
                    "penalty_end_frame",
                    "confidence",
                },
                context=f"episode {episode_index} repeated_grasps[{event_index}]",
            )
            frames = [
                event["close_frame"],
                event["reopen_frame"],
                event["retry_frame"],
                event["penalty_start_frame"],
                event["penalty_end_frame"],
            ]
            if any(isinstance(value, bool) or not isinstance(value, int) for value in frames):
                raise PriorityManifestError(
                    f"episode {episode_index} repeated-grasp frames must be integers"
                )
            close_frame, reopen_frame, retry_frame, penalty_start, penalty_end = frames
            if not (
                0 <= penalty_start <= close_frame < reopen_frame <= retry_frame <= frame_count
                and penalty_start < penalty_end <= retry_frame
            ):
                raise PriorityManifestError(
                    f"episode {episode_index} repeated-grasp frame order is invalid"
                )
            stage = event["stage"]
            if stage not in stage_order or stage_for_frame[close_frame] != stage:
                raise PriorityManifestError(
                    f"episode {episode_index} repeated grasp has incorrect stage {stage!r}"
                )
            if event["gripper"] not in {"left", "right"}:
                raise PriorityManifestError(
                    f"episode {episode_index} repeated grasp has invalid gripper"
                )
            if event["confidence"] not in {"low", "medium", "high"}:
                raise PriorityManifestError(
                    f"episode {episode_index} repeated grasp confidence is invalid"
                )
            repeated.append(
                RepeatedGraspAnnotation(
                    gripper=event["gripper"],
                    stage=stage,
                    close_frame=close_frame,
                    reopen_frame=reopen_frame,
                    retry_frame=retry_frame,
                    penalty_start_frame=penalty_start,
                    penalty_end_frame=penalty_end,
                    confidence=event["confidence"],
                )
            )

        smoothing = raw["smoothing"]
        if not isinstance(smoothing, dict):
            raise PriorityManifestError(f"episode {episode_index}.smoothing must be an object")
        _require_keys(
            smoothing,
            required={
                "label",
                "confidence",
                "review_start_frame",
                "review_end_frame",
                "stage_context",
                "priority_adjustment",
            },
            context=f"episode {episode_index}.smoothing",
        )
        if smoothing["label"] not in {"present", "uncertain", "absent"}:
            raise PriorityManifestError(f"episode {episode_index} smoothing label is invalid")
        if smoothing["confidence"] not in {"low", "medium", "high"}:
            raise PriorityManifestError(f"episode {episode_index} smoothing confidence is invalid")
        smoothing_start = smoothing["review_start_frame"]
        smoothing_end = smoothing["review_end_frame"]
        if not (
            isinstance(smoothing_start, int)
            and isinstance(smoothing_end, int)
            and 0 <= smoothing_start < smoothing_end <= frame_count
        ):
            raise PriorityManifestError(
                f"episode {episode_index} smoothing review window is invalid"
            )
        if (
            not isinstance(smoothing["stage_context"], list)
            or not smoothing["stage_context"]
            or any(stage not in stage_order for stage in smoothing["stage_context"])
        ):
            raise PriorityManifestError(
                f"episode {episode_index} smoothing stage_context is invalid"
            )
        if smoothing["priority_adjustment"] != 0.0:
            raise PriorityManifestError(
                f"episode {episode_index} smoothing must be priority-neutral"
            )

        return EpisodeAnnotation(
            episode_index=episode_index,
            frame_count=frame_count,
            stages=tuple(stages),
            repeated_grasps=tuple(repeated),
            smoothing=smoothing,
        )

    def verify_dataset(self, root: str | Path) -> None:
        dataset_root = Path(root).expanduser().resolve()
        for relative_path, expected_hash in self.dataset["fingerprints"].items():
            path = dataset_root / relative_path
            if not path.is_file():
                raise PriorityManifestError(
                    f"Priority manifest dataset file is missing: {path}"
                )
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise PriorityManifestError(
                    f"Priority manifest fingerprint mismatch for {path}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_path": str(self.path),
            "manifest_sha256": self.sha256,
            "method": "exponential_priority_sampler",
            "loss_reweighting": False,
            "sampling_replacement": True,
            "stage_order": list(self.stage_order),
        }


class PriorityEpisodeAwareSampler:
    """Episode-aware, deterministic weighted sampler with sample-exact resume."""

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        *,
        manifest: PriorityManifest,
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        if not shuffle:
            raise ValueError("PriorityEpisodeAwareSampler requires shuffle=True")
        if drop_n_first_frames < 0 or drop_n_last_frames < 0:
            raise ValueError("drop_n_first_frames and drop_n_last_frames must be >= 0")
        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError("dataset_from_indices and dataset_to_indices must have equal length")
        if len(from_indices) != len(manifest.episodes):
            raise PriorityManifestError(
                f"Sampler has {len(from_indices)} episodes; manifest has {len(manifest.episodes)}"
            )
        for episode, start, end in zip(manifest.episodes, from_indices, to_indices, strict=True):
            if int(end - start) != episode.frame_count:
                raise PriorityManifestError(
                    f"Episode {episode.episode_index} length mismatch: "
                    f"dataset={int(end-start)}, manifest={episode.frame_count}"
                )

        used_episodes = (
            list(range(len(from_indices)))
            if episode_indices_to_use is None
            else [int(value) for value in episode_indices_to_use]
        )
        if not used_episodes or len(used_episodes) != len(set(used_episodes)):
            raise ValueError("episode_indices_to_use must contain unique episode indices")
        if any(value < 0 or value >= len(from_indices) for value in used_episodes):
            raise ValueError("episode_indices_to_use contains an out-of-range episode")

        frame_indices: list[int] = []
        log_priorities: list[float] = []
        frame_stages: list[str] = []
        for episode_index in used_episodes:
            episode = manifest.episodes[episode_index]
            keep_start = drop_n_first_frames
            keep_end = episode.frame_count - drop_n_last_frames
            if keep_end <= keep_start:
                log.warning(
                    "Skipping episode %d because frame dropping removed every frame",
                    episode_index,
                )
                continue
            episode_log_priorities = np.empty(episode.frame_count, dtype=np.float64)
            episode_stages = np.empty(episode.frame_count, dtype=object)
            for stage in episode.stages:
                episode_log_priorities[stage.start_frame : stage.end_frame] = (
                    manifest.quality_log_priority[stage.quality_score]
                )
                episode_stages[stage.start_frame : stage.end_frame] = stage.name
            for event in episode.repeated_grasps:
                episode_log_priorities[event.penalty_start_frame : event.penalty_end_frame] += (
                    manifest.repeated_grasp_log_penalty
                )
            absolute_start = int(from_indices[episode_index]) + keep_start
            absolute_end = int(from_indices[episode_index]) + keep_end
            frame_indices.extend(range(absolute_start, absolute_end))
            log_priorities.extend(episode_log_priorities[keep_start:keep_end].tolist())
            frame_stages.extend(episode_stages[keep_start:keep_end].tolist())

        if not frame_indices:
            raise ValueError("No valid frames remain for priority sampling")
        raw_priorities = np.exp(np.asarray(log_priorities, dtype=np.float64))
        frame_stage_array = np.asarray(frame_stages, dtype=object)
        priorities = raw_priorities.copy()
        total_stage_mass = sum(manifest.stage_probability_mass.values())
        for stage in manifest.stage_order:
            mask = frame_stage_array == stage
            stage_sum = float(raw_priorities[mask].sum())
            if stage_sum <= 0:
                raise PriorityManifestError(f"Selected episodes contain no frames for stage {stage}")
            target_mass = manifest.stage_probability_mass[stage] / total_stage_mass
            priorities[mask] *= target_mass / stage_sum
        priorities /= priorities.sum()
        if not np.isfinite(priorities).all() or np.any(priorities <= 0):
            raise PriorityManifestError("Resolved frame priorities are not finite and positive")

        self._absolute_indices = np.asarray(frame_indices, dtype=np.int64)
        self._probabilities = torch.as_tensor(priorities, dtype=torch.double)
        self._absolute_to_relative = absolute_to_relative_idx
        if self._absolute_to_relative is not None:
            missing = [
                int(index)
                for index in self._absolute_indices
                if int(index) not in self._absolute_to_relative
            ]
            if missing:
                raise PriorityManifestError(
                    f"absolute_to_relative_idx is missing sampled frame {missing[0]}"
                )
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        log.info(
            "[priority-sampling] Resolved %d frames from %d episodes; manifest=%s",
            len(self._absolute_indices),
            len(used_episodes),
            manifest.sha256,
        )

    @property
    def indices(self) -> list[int]:
        return [self._map_index(int(index)) for index in self._absolute_indices]

    @property
    def probabilities(self) -> torch.Tensor:
        return self._probabilities.clone()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self._epoch = int(state["epoch"])
        self._start_index = int(state["start_index"])

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        epoch_seed = int(
            np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0]
        )
        return torch.Generator().manual_seed(epoch_seed)

    def _map_index(self, absolute_index: int) -> int:
        if self._absolute_to_relative is None:
            return absolute_index
        return self._absolute_to_relative[absolute_index]

    def __iter__(self) -> Iterator[int]:
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        order = torch.multinomial(
            self._probabilities,
            len(self._absolute_indices),
            replacement=True,
            generator=self._epoch_generator(epoch),
        )
        return (
            self._map_index(int(self._absolute_indices[int(position)]))
            for position in order[start:]
        )

    def __len__(self) -> int:
        return len(self._absolute_indices)
