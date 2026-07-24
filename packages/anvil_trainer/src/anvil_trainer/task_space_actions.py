"""Task-space action targets with deterministic outward-elbow decoding.

Each arm contributes a base-frame TCP translation delta, a base-frame rotation
vector delta, and an absolute gripper command.  The policy therefore does not
learn one of the infinitely many redundant 7-DoF joint configurations for the
same tool pose.  The paired postprocessor delegates that redundancy to the
hard-bounded trajectory solver in :mod:`anvil_embodiment.trajectory`.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anvil_embodiment.kinematics import (
    get_model_spec,
    model_spec_hash,
    torch_forward_kinematics,
)
from anvil_embodiment.trajectory import (
    ConstrainedBimanualTrajectorySolver,
    OutwardElbowConfig,
    TrajectorySolverConfig,
)
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.configs.types import FeatureType
from lerobot.processor.pipeline import ProcessorStepRegistry
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE


class TaskSpaceActionError(ValueError):
    """A task-space contract or action tensor is invalid."""


@dataclass(frozen=True)
class TaskSpaceActionContract:
    """Immutable representation, physical-solver, and train-fit contract."""

    path: Path
    sha256: str
    representation_id: str
    deployment_status: str
    model_id: str
    model_sha256: str
    source_action_names: tuple[str, ...]
    task_action_names: tuple[str, ...]
    chunk_size: int
    training_episode_indices: tuple[int, ...]
    split_sha256: str
    quantile_low: float
    quantile_high: float
    minimum_scale: float
    clip_value: float
    right_joint_indices: tuple[int, ...]
    left_joint_indices: tuple[int, ...]
    gripper_indices: tuple[int, ...]
    gripper_lower: tuple[float, ...]
    gripper_upper: tuple[float, ...]
    smoothing_kernel: tuple[float, ...]
    smoothing_passes: int
    gripper_event_threshold: float
    solver: TrajectorySolverConfig

    @classmethod
    def load(cls, path: str | Path) -> TaskSpaceActionContract:
        source = Path(path).expanduser().resolve()
        payload = source.read_bytes()
        return cls._parse(source, payload)

    @classmethod
    def from_serialized(
        cls,
        payload_text: str,
        *,
        expected_sha256: str,
    ) -> TaskSpaceActionContract:
        """Load a checkpoint-embedded contract without relying on a repo path."""
        payload = payload_text.encode()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_sha256:
            raise TaskSpaceActionError(
                f"serialized task-space contract hash mismatch: "
                f"expected {expected_sha256}, got {actual}"
            )
        return cls._parse(Path("task_space_action_contract.json"), payload)

    @classmethod
    def _parse(cls, source: Path, payload: bytes) -> TaskSpaceActionContract:
        raw = json.loads(payload)
        if raw.get("schema_version") != 1:
            raise TaskSpaceActionError("task-space action contract schema_version must be 1")
        fit = raw.get("train_only_fit", {})
        indices = raw.get("source_indices", {})
        grippers = raw.get("grippers", {})
        smoothing = raw.get("task_space_smoothing", {})
        solver_raw = raw.get("trajectory_solver", {})
        elbow_raw = solver_raw.get("outward_elbow", {})

        def elbow(side: str) -> OutwardElbowConfig:
            value = elbow_raw.get(side, {})
            return OutwardElbowConfig(
                shoulder_body=str(value.get("shoulder_body", "")),
                elbow_body=str(value.get("elbow_body", "")),
                outward_axis=tuple(float(item) for item in value.get("outward_axis", [])),
                weight=float(value.get("weight", 0.0)),
                finite_difference_rad=float(value.get("finite_difference_rad", 0.0)),
                target_alignment=float(value.get("target_alignment", 0.0)),
            )

        solver = TrajectorySolverConfig(
            dt_seconds=float(solver_raw.get("dt_seconds", 0.0)),
            joint_limit_margin_rad=float(solver_raw.get("joint_limit_margin_rad", 0.0)),
            max_velocity_rad_s=tuple(
                float(item) for item in solver_raw.get("max_velocity_rad_s", [])
            ),
            max_acceleration_rad_s2=tuple(
                float(item) for item in solver_raw.get("max_acceleration_rad_s2", [])
            ),
            max_iterations=int(solver_raw.get("max_iterations", 0)),
            damping=float(solver_raw.get("damping", 0.0)),
            max_iteration_step_rad=float(solver_raw.get("max_iteration_step_rad", 0.0)),
            position_tolerance_m=float(solver_raw.get("position_tolerance_m", 0.0)),
            orientation_tolerance_rad=float(
                solver_raw.get("orientation_tolerance_rad", 0.0)
            ),
            continuity_weight=float(solver_raw.get("continuity_weight", 0.0)),
            joint_center_weight=float(solver_raw.get("joint_center_weight", 0.0)),
            right_elbow=elbow("right"),
            left_elbow=elbow("left"),
        )
        result = cls(
            path=source,
            sha256=hashlib.sha256(payload).hexdigest(),
            representation_id=str(raw.get("representation_id", "")),
            deployment_status=str(raw.get("deployment_status", "")),
            model_id=str(raw.get("kinematic_model", {}).get("model_id", "")),
            model_sha256=str(raw.get("kinematic_model", {}).get("sha256", "")),
            source_action_names=tuple(
                str(item) for item in raw.get("source_action_feature_names", [])
            ),
            task_action_names=tuple(
                str(item) for item in raw.get("task_action_feature_names", [])
            ),
            chunk_size=int(raw.get("chunk_size", 0)),
            training_episode_indices=tuple(
                int(item) for item in fit.get("episode_indices", [])
            ),
            split_sha256=str(fit.get("split_sha256", "")),
            quantile_low=float(fit.get("quantile_low", 0.0)),
            quantile_high=float(fit.get("quantile_high", 0.0)),
            minimum_scale=float(fit.get("minimum_scale", 0.0)),
            clip_value=float(fit.get("clip_value", 0.0)),
            right_joint_indices=tuple(int(item) for item in indices.get("right_arm", [])),
            left_joint_indices=tuple(int(item) for item in indices.get("left_arm", [])),
            gripper_indices=tuple(int(item) for item in indices.get("grippers", [])),
            gripper_lower=tuple(float(item) for item in grippers.get("lower", [])),
            gripper_upper=tuple(float(item) for item in grippers.get("upper", [])),
            smoothing_kernel=tuple(float(item) for item in smoothing.get("kernel", [])),
            smoothing_passes=int(smoothing.get("passes", 0)),
            gripper_event_threshold=float(
                smoothing.get("gripper_event_threshold", 0.0)
            ),
            solver=solver,
        )
        result.validate(raw)
        return result

    def validate(self, raw: dict[str, Any] | None = None) -> None:
        if not self.representation_id:
            raise TaskSpaceActionError("representation_id is required")
        if self.deployment_status != "offline_only":
            raise TaskSpaceActionError(
                "v1 task-space contracts must remain offline_only until collision auditing exists"
            )
        if len(self.source_action_names) != 16 or len(set(self.source_action_names)) != 16:
            raise TaskSpaceActionError("source action names must be 16 unique right-first names")
        if len(self.task_action_names) != 14 or len(set(self.task_action_names)) != 14:
            raise TaskSpaceActionError("task action names must be 14 unique right-first names")
        if self.chunk_size < 1:
            raise TaskSpaceActionError("chunk_size must be positive")
        if not self.training_episode_indices or len(set(self.training_episode_indices)) != len(
            self.training_episode_indices
        ):
            raise TaskSpaceActionError("train-only episode indices must be non-empty and unique")
        if len(self.split_sha256) != 64:
            raise TaskSpaceActionError("split_sha256 must be a SHA-256 digest")
        if self.right_joint_indices != tuple(range(7)):
            raise TaskSpaceActionError("right arm must occupy source indices 0..6")
        if self.left_joint_indices != tuple(range(8, 15)):
            raise TaskSpaceActionError("left arm must occupy source indices 8..14")
        if self.gripper_indices != (7, 15):
            raise TaskSpaceActionError("grippers must occupy source indices 7 and 15")
        if len(self.gripper_lower) != 2 or len(self.gripper_upper) != 2:
            raise TaskSpaceActionError("two gripper endpoint pairs are required")
        if any(
            lower >= upper
            for lower, upper in zip(
                self.gripper_lower, self.gripper_upper, strict=True
            )
        ):
            raise TaskSpaceActionError("gripper lower endpoints must be below upper endpoints")
        if not 0 <= self.quantile_low < self.quantile_high <= 1:
            raise TaskSpaceActionError("invalid train-only fit quantiles")
        if self.minimum_scale <= 0 or self.clip_value <= 0:
            raise TaskSpaceActionError("fit scale and clip value must be positive")
        expected_kernel = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
        if not np.allclose(self.smoothing_kernel, expected_kernel, atol=1.0e-15, rtol=0):
            raise TaskSpaceActionError("task-space smoothing must use a cubic B-spline kernel")
        if self.smoothing_passes < 1 or self.gripper_event_threshold <= 0:
            raise TaskSpaceActionError("invalid task-space smoothing settings")
        self.solver.validate()
        try:
            spec = get_model_spec(self.model_id)
        except KeyError as error:
            raise TaskSpaceActionError(str(error)) from error
        if model_spec_hash(spec) != self.model_sha256:
            raise TaskSpaceActionError("kinematic model hash does not match the built-in model")
        if raw is not None:
            frame = raw.get("task_frame", {})
            if frame.get("name") != "robot_base" or frame.get("source_to_task") != {
                "translation_m": [0.0, 0.0, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }:
                raise TaskSpaceActionError(
                    "v1 supports only an explicitly pinned robot-base task frame"
                )

    @property
    def source_soft_lower(self) -> np.ndarray:
        spec = get_model_spec(self.model_id)
        result = np.zeros(16, dtype=np.float64)
        result[:7] = np.asarray(
            [link.joint_range[0] for link in spec.arms["right"].links]
        ) + self.solver.joint_limit_margin_rad
        result[8:15] = np.asarray(
            [link.joint_range[0] for link in spec.arms["left"].links]
        ) + self.solver.joint_limit_margin_rad
        result[list(self.gripper_indices)] = self.gripper_lower
        return result

    @property
    def source_soft_upper(self) -> np.ndarray:
        spec = get_model_spec(self.model_id)
        result = np.zeros(16, dtype=np.float64)
        result[:7] = np.asarray(
            [link.joint_range[1] for link in spec.arms["right"].links]
        ) - self.solver.joint_limit_margin_rad
        result[8:15] = np.asarray(
            [link.joint_range[1] for link in spec.arms["left"].links]
        ) - self.solver.joint_limit_margin_rad
        result[list(self.gripper_indices)] = self.gripper_upper
        return result


def _rotation_matrix_to_vector(rotation: torch.Tensor) -> torch.Tensor:
    """Vectorized SO(3) log map for ordinary (<pi) trajectory deltas."""
    trace = torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.sin(angle)
    ordinary = angle / (2.0 * sine.clamp_min(1.0e-7))
    scale = torch.where(angle < 1.0e-5, torch.full_like(angle, 0.5), ordinary)
    result = vee * scale.unsqueeze(-1)
    if not torch.isfinite(result).all():
        raise TaskSpaceActionError("SO(3) log produced a non-finite rotation vector")
    return result


def _rotation_vector_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Vectorized Rodrigues map."""
    angle = torch.linalg.vector_norm(vector, dim=-1)
    safe_angle = angle.clamp_min(1.0e-12)
    axis = vector / safe_angle.unsqueeze(-1)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(
        vector.shape[:-1] + (3, 3)
    )
    outer = axis.unsqueeze(-1) @ axis.unsqueeze(-2)
    result = (
        torch.cos(angle)[..., None, None] * eye
        + (1.0 - torch.cos(angle))[..., None, None] * outer
        + torch.sin(angle)[..., None, None] * skew
    )
    small_skew = torch.stack(
        (
            torch.stack((zero, -vector[..., 2], vector[..., 1]), dim=-1),
            torch.stack((vector[..., 2], zero, -vector[..., 0]), dim=-1),
            torch.stack((-vector[..., 1], vector[..., 0], zero), dim=-1),
        ),
        dim=-2,
    )
    return torch.where((angle < 1.0e-7)[..., None, None], eye + small_skew, result)


def _reference_state(state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    state = state.to(device=actions.device, dtype=actions.dtype)
    if state.ndim > 2:
        state = state[..., 0, :]
    if actions.ndim == 3 and state.ndim == 2:
        state = state.unsqueeze(-2)
    return state


def _horizon_view(values: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    if action.ndim == 3:
        return values[: action.shape[-2]].unsqueeze(0)
    if action.ndim == 2:
        return values[: action.shape[-2]]
    if action.ndim == 1:
        return values[0]
    raise TaskSpaceActionError("task action tensor must have 1, 2, or 3 dimensions")


def encode_task_space_actions(
    actions: torch.Tensor,
    state: torch.Tensor,
    *,
    contract: TaskSpaceActionContract,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Encode 16-D joint targets as normalized 14-D task-space targets."""
    if actions.shape[-1] != 16:
        raise TaskSpaceActionError("source action tensor must have 16 dimensions")
    reference = _reference_state(state, actions)
    if reference.shape[-1] != 16:
        raise TaskSpaceActionError("observation state must have 16 dimensions")
    model = get_model_spec(contract.model_id)
    chunks: list[torch.Tensor] = []
    for side_index, side in enumerate(("right", "left")):
        joint_indices = (
            contract.right_joint_indices
            if side == "right"
            else contract.left_joint_indices
        )
        target_position, target_rotation = torch_forward_kinematics(
            model, side, actions[..., list(joint_indices)]
        )
        current_position, current_rotation = torch_forward_kinematics(
            model, side, reference[..., list(joint_indices)]
        )
        relative_rotation = target_rotation @ current_rotation.transpose(-1, -2)
        rotation_vector = _rotation_matrix_to_vector(relative_rotation)
        gripper_index = contract.gripper_indices[side_index]
        lower = contract.gripper_lower[side_index]
        upper = contract.gripper_upper[side_index]
        gripper = (
            2.0 * (actions[..., gripper_index].clamp(lower, upper) - lower) / (upper - lower)
            - 1.0
        ).unsqueeze(-1)
        chunks.append(
            torch.cat((target_position - current_position, rotation_vector, gripper), dim=-1)
        )
    task = torch.cat(chunks, dim=-1)
    center_view = _horizon_view(center.to(task.device, task.dtype), task)
    scale_view = _horizon_view(scale.to(task.device, task.dtype), task)
    return ((task - center_view) / scale_view).clamp(
        -contract.clip_value, contract.clip_value
    )


def decode_task_space_targets(
    encoded: torch.Tensor,
    state: torch.Tensor,
    *,
    contract: TaskSpaceActionContract,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode normalized output into absolute TCP poses and physical grippers."""
    task = denormalize_task_space_actions(
        encoded,
        contract=contract,
        center=center,
        scale=scale,
    )
    return task_space_values_to_targets(task, state, contract=contract)


def denormalize_task_space_actions(
    encoded: torch.Tensor,
    *,
    contract: TaskSpaceActionContract,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Map model coordinates back to metric/radian task deltas."""
    if encoded.shape[-1] != 14:
        raise TaskSpaceActionError("task action tensor must have 14 dimensions")
    center_view = _horizon_view(center.to(encoded.device, encoded.dtype), encoded)
    scale_view = _horizon_view(scale.to(encoded.device, encoded.dtype), encoded)
    return encoded.nan_to_num().clamp(
        -contract.clip_value, contract.clip_value
    ) * scale_view + center_view


def task_space_values_to_targets(
    task: torch.Tensor,
    state: torch.Tensor,
    *,
    contract: TaskSpaceActionContract,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose raw task deltas with the current observation FK."""
    if task.shape[-1] != 14:
        raise TaskSpaceActionError("task action tensor must have 14 dimensions")
    reference = _reference_state(state, task)
    if reference.shape[-1] != 16:
        raise TaskSpaceActionError("observation state must have 16 dimensions")
    model = get_model_spec(contract.model_id)
    positions: list[torch.Tensor] = []
    rotations: list[torch.Tensor] = []
    grippers: list[torch.Tensor] = []
    for side_index, side in enumerate(("right", "left")):
        offset = side_index * 7
        joint_indices = (
            contract.right_joint_indices
            if side == "right"
            else contract.left_joint_indices
        )
        current_position, current_rotation = torch_forward_kinematics(
            model, side, reference[..., list(joint_indices)]
        )
        positions.append(current_position + task[..., offset : offset + 3])
        delta_rotation = _rotation_vector_to_matrix(task[..., offset + 3 : offset + 6])
        rotations.append(delta_rotation @ current_rotation)
        lower = contract.gripper_lower[side_index]
        upper = contract.gripper_upper[side_index]
        grippers.append(
            lower + 0.5 * (task[..., offset + 6].clamp(-1.0, 1.0) + 1.0) * (upper - lower)
        )
    return (
        torch.stack(positions, dim=-2),
        torch.stack(rotations, dim=-3),
        torch.stack(grippers, dim=-1),
    )


def smooth_task_space_chunk(
    encoded: torch.Tensor,
    *,
    kernel: tuple[float, ...],
    passes: int,
    gripper_event_threshold_normalized: float,
) -> torch.Tensor:
    """Cubic B-spline smoothing of TCP deltas without blurring gripper events."""
    if encoded.ndim == 1 or encoded.shape[-2] < 3:
        return encoded.clone()
    if encoded.ndim not in {2, 3}:
        raise TaskSpaceActionError("task action chunk must have shape [T,14] or [B,T,14]")
    source = encoded.unsqueeze(0) if encoded.ndim == 2 else encoded
    result = source.clone()
    pose_indices = [*range(6), *range(7, 13)]
    gripper_indices = [6, 13]
    weights = torch.as_tensor(kernel, dtype=source.dtype, device=source.device)
    for batch_index in range(source.shape[0]):
        grippers = source[batch_index, :, gripper_indices]
        events = torch.any(
            torch.abs(grippers[1:] - grippers[:-1]) >= gripper_event_threshold_normalized,
            dim=-1,
        )
        cuts = [0]
        cuts.extend((torch.nonzero(events, as_tuple=False).flatten() + 1).tolist())
        cuts.append(source.shape[1])
        for start, end in zip(cuts[:-1], cuts[1:], strict=True):
            if end - start < 3:
                continue
            segment = result[batch_index, start:end, pose_indices]
            for _ in range(passes):
                smoothed = segment.clone()
                smoothed[1:-1] = (
                    weights[0] * segment[:-2]
                    + weights[1] * segment[1:-1]
                    + weights[2] * segment[2:]
                )
                segment = smoothed
            result[batch_index, start:end, pose_indices] = segment
    result[..., gripper_indices] = source[..., gripper_indices]
    return result[0] if encoded.ndim == 2 else result


@ProcessorStepRegistry.register("task_space_relative_actions_processor")
@dataclass
class TaskSpaceRelativeActionsProcessorStep(RelativeActionsProcessorStep):
    """Serialized preprocessor for joint-to-task-space conversion."""

    contract_path: str = ""
    contract_json: str = ""
    contract_sha256: str = ""
    representation_id: str = ""
    horizon_center: list[list[float]] = field(default_factory=list)
    horizon_scale: list[list[float]] = field(default_factory=list)
    source_action_names: list[str] = field(default_factory=list)
    task_action_names: list[str] = field(default_factory=list)
    _contract: TaskSpaceActionContract | None = field(default=None, init=False, repr=False)

    def load_contract(self) -> TaskSpaceActionContract:
        if not self.contract_json:
            raise TaskSpaceActionError("serialized task-space processor has no embedded contract")
        if self._contract is None:
            self._contract = TaskSpaceActionContract.from_serialized(
                self.contract_json,
                expected_sha256=self.contract_sha256,
            )
        return self._contract

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            self._last_state = state
        result = transition.copy()
        action = result.get(TransitionKey.ACTION)
        if not self.enabled or action is None or state is None:
            return result
        contract = self.load_contract()
        result[TransitionKey.ACTION] = encode_task_space_actions(
            action,
            state,
            contract=contract,
            center=torch.as_tensor(self.horizon_center),
            scale=torch.as_tensor(self.horizon_scale),
        )
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
            "contract_path": self.contract_path,
            "contract_json": self.contract_json,
            "contract_sha256": self.contract_sha256,
            "representation_id": self.representation_id,
            "horizon_center": self.horizon_center,
            "horizon_scale": self.horizon_scale,
            "source_action_names": self.source_action_names,
            "task_action_names": self.task_action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = deepcopy(features)
        action = transformed.get(PipelineFeatureType.ACTION, {})
        if "action" in action:
            action["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(len(self.task_action_names),),
            )
        return transformed


@ProcessorStepRegistry.register("task_space_absolute_actions_processor")
@dataclass
class TaskSpaceAbsoluteActionsProcessorStep(AbsoluteActionsProcessorStep):
    """Decode task-space predictions through the constrained trajectory solver."""

    _solver: ConstrainedBimanualTrajectorySolver | None = field(
        default=None, init=False, repr=False
    )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if not isinstance(self.relative_step, TaskSpaceRelativeActionsProcessorStep):
            raise RuntimeError("task-space postprocessor is missing its paired preprocessor")
        state = self.relative_step.get_cached_state()
        if state is None:
            raise RuntimeError("task-space postprocessor has no cached observation state")
        result = transition.copy()
        action = result.get(TransitionKey.ACTION)
        if action is None:
            return result
        contract = self.relative_step.load_contract()
        task_values = denormalize_task_space_actions(
            action,
            contract=contract,
            center=torch.as_tensor(self.relative_step.horizon_center),
            scale=torch.as_tensor(self.relative_step.horizon_scale),
        )
        smoothed = smooth_task_space_chunk(
            task_values,
            kernel=contract.smoothing_kernel,
            passes=contract.smoothing_passes,
            gripper_event_threshold_normalized=(
                2.0
                * contract.gripper_event_threshold
                / min(
                    upper - lower
                    for lower, upper in zip(
                        contract.gripper_lower, contract.gripper_upper, strict=True
                    )
                )
            ),
        )
        positions, rotations, grippers = task_space_values_to_targets(
            smoothed,
            state,
            contract=contract,
        )
        if self._solver is None:
            self._solver = ConstrainedBimanualTrajectorySolver(
                get_model_spec(contract.model_id), contract.solver
            )
        solver = self._solver
        batched = action.ndim == 3
        source = smoothed if batched else smoothed.unsqueeze(0)
        position_batch = positions if batched else positions.unsqueeze(0)
        rotation_batch = rotations if batched else rotations.unsqueeze(0)
        gripper_batch = grippers if batched else grippers.unsqueeze(0)
        solver_state = state[..., 0, :] if state.ndim > 2 else state
        state_batch = (
            solver_state if solver_state.ndim == 2 else solver_state.unsqueeze(0)
        )
        decoded: list[torch.Tensor] = []
        for index in range(source.shape[0]):
            trajectory = solver.solve(
                positions=position_batch[index].detach().cpu().double().numpy(),
                rotations=rotation_batch[index].detach().cpu().double().numpy(),
                grippers=gripper_batch[index].detach().cpu().double().numpy(),
                current_state=state_batch[index].detach().cpu().double().numpy(),
                require_convergence=True,
            )
            decoded.append(
                torch.as_tensor(
                    trajectory.values,
                    dtype=action.dtype,
                    device=action.device,
                )
            )
        stacked = torch.stack(decoded)
        result[TransitionKey.ACTION] = stacked if batched else stacked[0]
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = deepcopy(features)
        if isinstance(self.relative_step, TaskSpaceRelativeActionsProcessorStep):
            action = transformed.get(PipelineFeatureType.ACTION, {})
            if "action" in action:
                action["action"] = PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(len(self.relative_step.source_action_names),),
                )
        return transformed


def make_task_space_processor_steps(
    contract: TaskSpaceActionContract,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[
    TaskSpaceRelativeActionsProcessorStep,
    TaskSpaceAbsoluteActionsProcessorStep,
]:
    """Construct the serialized pre/postprocessor pair."""
    relative = TaskSpaceRelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=[],
        action_names=list(contract.source_action_names),
        contract_path=str(contract.path),
        contract_json=contract.path.read_text(),
        contract_sha256=contract.sha256,
        representation_id=contract.representation_id,
        horizon_center=center.tolist(),
        horizon_scale=scale.tolist(),
        source_action_names=list(contract.source_action_names),
        task_action_names=list(contract.task_action_names),
    )
    return relative, TaskSpaceAbsoluteActionsProcessorStep(
        enabled=True, relative_step=relative
    )
