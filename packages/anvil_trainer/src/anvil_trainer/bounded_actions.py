"""State-relative action codec with hard OpenArm position guarantees.

The native Pi0.5 relative-action processor predicts unconstrained joint deltas.
This module instead represents each arm target as a signed fraction of the
remaining motion between the current state and a buffered joint endpoint.  A
decoded arm command is therefore inside the configured soft limits for every
finite model output.  Grippers remain absolute bounded state transitions.

Arm fractions are robustly normalized per actuator *and horizon position* from
the frozen training episodes only.  This is the useful temporal-normalization
piece of the Larchenko recipe without importing embodiment-specific constants.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStepRegistry
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE


class BoundedActionError(ValueError):
    """Raised when a bounded-action contract or fitted statistic is invalid."""


@dataclass(frozen=True)
class BoundedActionContract:
    """Immutable physical and statistical contract for the bounded codec."""

    path: Path
    sha256: str
    representation_id: str
    action_names: tuple[str, ...]
    chunk_size: int
    training_episode_indices: tuple[int, ...]
    split_sha256: str
    arm_indices: tuple[int, ...]
    absolute_indices: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    arm_margin: float
    quantile_low: float
    quantile_high: float
    minimum_scale: float
    clip_value: float
    smoothing_kernel: tuple[float, ...]
    max_training_clip_fraction: float

    @classmethod
    def load(cls, path: str | Path) -> BoundedActionContract:
        source = Path(path).expanduser().resolve()
        payload = source.read_bytes()
        raw = json.loads(payload)
        if raw.get("schema_version") != 1:
            raise BoundedActionError("bounded action contract schema_version must be 1")
        fit = raw.get("train_only_fit", {})
        result = cls(
            path=source,
            sha256=hashlib.sha256(payload).hexdigest(),
            representation_id=str(raw.get("representation_id", "")),
            action_names=tuple(str(value) for value in raw.get("action_feature_names", [])),
            chunk_size=int(raw.get("chunk_size", 0)),
            training_episode_indices=tuple(int(value) for value in fit.get("episode_indices", [])),
            split_sha256=str(fit.get("split_sha256", "")),
            arm_indices=tuple(int(value) for value in raw.get("arm_indices", [])),
            absolute_indices=tuple(int(value) for value in raw.get("absolute_indices", [])),
            lower=tuple(float(value) for value in raw.get("lower", [])),
            upper=tuple(float(value) for value in raw.get("upper", [])),
            arm_margin=float(raw.get("arm_margin_rad", 0.0)),
            quantile_low=float(fit.get("quantile_low", 0.01)),
            quantile_high=float(fit.get("quantile_high", 0.99)),
            minimum_scale=float(fit.get("minimum_scale", 0.02)),
            clip_value=float(fit.get("clip_value", 1.0)),
            smoothing_kernel=tuple(float(value) for value in fit.get("smoothing_kernel", [])),
            max_training_clip_fraction=float(fit.get("max_training_clip_fraction", 0.01)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        dimension = len(self.action_names)
        if not self.representation_id:
            raise BoundedActionError("representation_id is required")
        if dimension == 0 or len(set(self.action_names)) != dimension:
            raise BoundedActionError("action_feature_names must be non-empty and unique")
        if self.chunk_size < 1:
            raise BoundedActionError("chunk_size must be positive")
        if not self.training_episode_indices or len(set(self.training_episode_indices)) != len(
            self.training_episode_indices
        ):
            raise BoundedActionError("train_only_fit.episode_indices must be non-empty and unique")
        if len(self.split_sha256) != 64:
            raise BoundedActionError("train_only_fit.split_sha256 must be a SHA-256 digest")
        if len(self.lower) != dimension or len(self.upper) != dimension:
            raise BoundedActionError("lower/upper must match action_feature_names")
        if any(not np.isfinite(value) for value in (*self.lower, *self.upper)):
            raise BoundedActionError("lower/upper contain non-finite values")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise BoundedActionError("every lower endpoint must be below its upper endpoint")
        arm = set(self.arm_indices)
        absolute = set(self.absolute_indices)
        if arm & absolute or arm | absolute != set(range(dimension)):
            raise BoundedActionError(
                "arm_indices and absolute_indices must partition the action vector"
            )
        if self.arm_margin < 0 or any(
            self.upper[index] - self.lower[index] <= 2 * self.arm_margin
            for index in self.arm_indices
        ):
            raise BoundedActionError("arm_margin_rad leaves an empty soft range")
        if not 0 <= self.quantile_low < self.quantile_high <= 1:
            raise BoundedActionError("fit quantiles must satisfy 0 <= low < high <= 1")
        if self.minimum_scale <= 0 or self.clip_value <= 0:
            raise BoundedActionError("minimum_scale and clip_value must be positive")
        if not self.smoothing_kernel or any(value < 0 for value in self.smoothing_kernel):
            raise BoundedActionError("smoothing_kernel must contain non-negative weights")
        if sum(self.smoothing_kernel) <= 0:
            raise BoundedActionError("smoothing_kernel must have positive mass")
        if not 0 <= self.max_training_clip_fraction <= 1:
            raise BoundedActionError("max_training_clip_fraction must be in [0, 1]")

    @property
    def soft_lower(self) -> np.ndarray:
        values = np.asarray(self.lower, dtype=np.float64).copy()
        values[list(self.arm_indices)] += self.arm_margin
        return values

    @property
    def soft_upper(self) -> np.ndarray:
        values = np.asarray(self.upper, dtype=np.float64).copy()
        values[list(self.arm_indices)] -= self.arm_margin
        return values


def _horizon_view(values: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Return [1,T,D] or [T,D] statistics matching an action tensor."""
    if action.ndim == 3:
        return values[: action.shape[-2]].unsqueeze(0)
    if action.ndim == 2:
        return values[: action.shape[-2]]
    if action.ndim == 1:
        return values[0]
    raise BoundedActionError(f"action tensor must have 1, 2, or 3 dimensions, got {action.ndim}")


def _state_for_action(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    state = state.to(device=action.device, dtype=action.dtype)
    if state.ndim > 2:
        state = state[..., -1, :]
    if action.ndim == 3 and state.ndim == 2:
        return state.unsqueeze(-2)
    return state


def encode_bounded_actions(
    actions: torch.Tensor,
    state: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    arm_indices: tuple[int, ...],
    absolute_indices: tuple[int, ...],
    center: torch.Tensor,
    scale: torch.Tensor,
    clip_value: float,
) -> torch.Tensor:
    """Encode physical commands into bounded, per-horizon model targets."""
    result = actions.clone()
    lower = lower.to(device=actions.device, dtype=actions.dtype)
    upper = upper.to(device=actions.device, dtype=actions.dtype)
    reference = _state_for_action(state, actions).clamp(lower, upper)
    eps = torch.finfo(actions.dtype).eps

    arm = list(arm_indices)
    target = actions[..., arm].clamp(lower[arm], upper[arm])
    ref = reference[..., arm]
    delta = target - ref
    positive = (upper[arm] - ref).clamp_min(eps)
    negative = (ref - lower[arm]).clamp_min(eps)
    fraction = torch.where(delta >= 0, delta / positive, delta / negative)
    result[..., arm] = fraction

    absolute = list(absolute_indices)
    absolute_target = actions[..., absolute].clamp(lower[absolute], upper[absolute])
    result[..., absolute] = (
        2.0 * ((absolute_target - lower[absolute]) / (upper[absolute] - lower[absolute])) - 1.0
    )

    center_view = _horizon_view(center.to(actions.device, actions.dtype), result)
    scale_view = _horizon_view(scale.to(actions.device, actions.dtype), result)
    return ((result - center_view) / scale_view).clamp(-clip_value, clip_value)


def decode_bounded_actions(
    encoded: torch.Tensor,
    state: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    arm_indices: tuple[int, ...],
    absolute_indices: tuple[int, ...],
    center: torch.Tensor,
    scale: torch.Tensor,
    clip_value: float,
) -> torch.Tensor:
    """Decode model outputs; every finite result is inside the soft bounds."""
    lower = lower.to(device=encoded.device, dtype=encoded.dtype)
    upper = upper.to(device=encoded.device, dtype=encoded.dtype)
    center_view = _horizon_view(center.to(encoded.device, encoded.dtype), encoded)
    scale_view = _horizon_view(scale.to(encoded.device, encoded.dtype), encoded)
    normalized = encoded.nan_to_num().clamp(-clip_value, clip_value)
    fraction = (normalized * scale_view + center_view).clamp(-1.0, 1.0)
    reference = _state_for_action(state, encoded).clamp(lower, upper)
    result = fraction.clone()

    arm = list(arm_indices)
    arm_fraction = fraction[..., arm]
    ref = reference[..., arm]
    positive = upper[arm] - ref
    negative = ref - lower[arm]
    result[..., arm] = ref + torch.where(
        arm_fraction >= 0,
        arm_fraction * positive,
        arm_fraction * negative,
    )

    absolute = list(absolute_indices)
    result[..., absolute] = lower[absolute] + 0.5 * (fraction[..., absolute] + 1.0) * (
        upper[absolute] - lower[absolute]
    )
    return result.clamp(lower, upper)


@ProcessorStepRegistry.register("bounded_relative_actions_processor")
@dataclass
class BoundedRelativeActionsProcessorStep(RelativeActionsProcessorStep):
    """Preprocessor step for the state-relative soft-limit representation."""

    representation_id: str = ""
    contract_sha256: str = ""
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    arm_indices: list[int] = field(default_factory=list)
    absolute_indices: list[int] = field(default_factory=list)
    horizon_center: list[list[float]] = field(default_factory=list)
    horizon_scale: list[list[float]] = field(default_factory=list)
    clip_value: float = 1.0
    split_sha256: str = ""
    fit_episode_indices: list[int] = field(default_factory=list)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            self._last_state = state
        if not self.enabled:
            return transition
        result = transition.copy()
        action = result.get(TransitionKey.ACTION)
        if action is None or state is None:
            return result
        result[TransitionKey.ACTION] = encode_bounded_actions(
            action,
            state,
            lower=torch.as_tensor(self.lower),
            upper=torch.as_tensor(self.upper),
            arm_indices=tuple(self.arm_indices),
            absolute_indices=tuple(self.absolute_indices),
            center=torch.as_tensor(self.horizon_center),
            scale=torch.as_tensor(self.horizon_scale),
            clip_value=self.clip_value,
        )
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
            "representation_id": self.representation_id,
            "contract_sha256": self.contract_sha256,
            "lower": self.lower,
            "upper": self.upper,
            "arm_indices": self.arm_indices,
            "absolute_indices": self.absolute_indices,
            "horizon_center": self.horizon_center,
            "horizon_scale": self.horizon_scale,
            "clip_value": self.clip_value,
            "split_sha256": self.split_sha256,
            "fit_episode_indices": self.fit_episode_indices,
        }


@ProcessorStepRegistry.register("bounded_absolute_actions_processor")
@dataclass
class BoundedAbsoluteActionsProcessorStep(AbsoluteActionsProcessorStep):
    """Postprocessor paired with :class:`BoundedRelativeActionsProcessorStep`."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if not isinstance(self.relative_step, BoundedRelativeActionsProcessorStep):
            raise RuntimeError("bounded absolute processor is missing its bounded relative step")
        state = self.relative_step.get_cached_state()
        if state is None:
            raise RuntimeError("bounded absolute processor has no cached observation state")
        result = transition.copy()
        action = result.get(TransitionKey.ACTION)
        if action is None:
            return result
        step = self.relative_step
        result[TransitionKey.ACTION] = decode_bounded_actions(
            action,
            state,
            lower=torch.as_tensor(step.lower),
            upper=torch.as_tensor(step.upper),
            arm_indices=tuple(step.arm_indices),
            absolute_indices=tuple(step.absolute_indices),
            center=torch.as_tensor(step.horizon_center),
            scale=torch.as_tensor(step.horizon_scale),
            clip_value=step.clip_value,
        )
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def smooth_horizon(values: np.ndarray, kernel: tuple[float, ...]) -> np.ndarray:
    """Smooth [H,D] values along H with edge renormalization."""
    weights = np.asarray(kernel, dtype=np.float64)
    weights /= weights.sum()
    radius = len(weights) // 2
    result = np.empty_like(values, dtype=np.float64)
    for horizon in range(len(values)):
        start = max(0, horizon - radius)
        end = min(len(values), horizon + len(weights) - radius)
        w_start = radius - (horizon - start)
        selected = weights[w_start : w_start + (end - start)]
        selected = selected / selected.sum()
        result[horizon] = np.sum(values[start:end] * selected[:, None], axis=0)
    return result


def make_processor_steps(
    contract: BoundedActionContract,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[BoundedRelativeActionsProcessorStep, BoundedAbsoluteActionsProcessorStep]:
    """Construct a serialized processor pair from fitted train-only statistics."""
    lower = contract.soft_lower.tolist()
    upper = contract.soft_upper.tolist()
    relative = BoundedRelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=[contract.action_names[index] for index in contract.absolute_indices],
        action_names=list(contract.action_names),
        representation_id=contract.representation_id,
        contract_sha256=contract.sha256,
        lower=lower,
        upper=upper,
        arm_indices=list(contract.arm_indices),
        absolute_indices=list(contract.absolute_indices),
        horizon_center=center.tolist(),
        horizon_scale=scale.tolist(),
        clip_value=contract.clip_value,
        split_sha256=contract.split_sha256,
        fit_episode_indices=list(contract.training_episode_indices),
    )
    return relative, BoundedAbsoluteActionsProcessorStep(enabled=True, relative_step=relative)
