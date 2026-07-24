"""Constrained task-space trajectory solving for bimanual OpenArm policies.

The policy owns the two TCP trajectories.  Redundant joint configuration is a
deterministic controller concern: TCP tracking is primary, elbow-out posture is
secondary in the Jacobian nullspace, and continuity/joint centering are tertiary.
Position, velocity, and acceleration bounds are enforced as hard waypoint boxes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import MujocoArmKinematics, RobotModelSpec


class TrajectorySolveError(RuntimeError):
    """A task-space chunk could not be mapped to a safe joint trajectory."""


def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the base-frame rotation vector taking ``current`` to ``desired``."""
    relative = np.asarray(desired, dtype=np.float64) @ np.asarray(current, dtype=np.float64).T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1.0e-8:
        return 0.5 * vee
    sine = float(np.sin(angle))
    if abs(sine) >= 1.0e-8:
        return angle * vee / (2.0 * sine)

    # The skew term vanishes at pi.  Recover a stable axis from R + I.
    symmetric = 0.5 * (relative + np.eye(3, dtype=np.float64))
    axis = symmetric[:, int(np.argmax(np.diag(symmetric)))]
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-10:
        raise TrajectorySolveError("could not recover the axis of a pi rotation")
    return angle * axis / norm


def rotation_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    """Convert a finite 3-D rotation vector to a rotation matrix."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation vector must be finite with shape (3,)")
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        skew = np.asarray(
            [
                [0.0, -vector[2], vector[1]],
                [vector[2], 0.0, -vector[0]],
                [-vector[1], vector[0], 0.0],
            ]
        )
        return np.eye(3, dtype=np.float64) + skew
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.cos(angle) * np.eye(3, dtype=np.float64)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * skew
    )


@dataclass(frozen=True)
class OutwardElbowConfig:
    """Geometric elbow-swivel objective independent of joint conventions."""

    shoulder_body: str
    elbow_body: str
    outward_axis: tuple[float, float, float]
    weight: float
    finite_difference_rad: float
    target_alignment: float


@dataclass(frozen=True)
class TrajectorySolverConfig:
    """Pinned numerical and physical constraints for task-space decoding."""

    dt_seconds: float
    joint_limit_margin_rad: float
    max_velocity_rad_s: tuple[float, ...]
    max_acceleration_rad_s2: tuple[float, ...]
    max_iterations: int
    damping: float
    max_iteration_step_rad: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    continuity_weight: float
    joint_center_weight: float
    right_elbow: OutwardElbowConfig
    left_elbow: OutwardElbowConfig

    def validate(self) -> None:
        if self.dt_seconds <= 0 or self.joint_limit_margin_rad < 0:
            raise ValueError("solver timestep must be positive and joint margin non-negative")
        if len(self.max_velocity_rad_s) != 7 or any(
            not np.isfinite(value) or value <= 0 for value in self.max_velocity_rad_s
        ):
            raise ValueError("solver requires seven finite positive velocity limits")
        if len(self.max_acceleration_rad_s2) != 7 or any(
            not np.isfinite(value) or value <= 0 for value in self.max_acceleration_rad_s2
        ):
            raise ValueError("solver requires seven finite positive acceleration limits")
        if self.max_iterations < 1 or self.damping <= 0 or self.max_iteration_step_rad <= 0:
            raise ValueError("invalid trajectory solver iteration settings")
        if self.position_tolerance_m <= 0 or self.orientation_tolerance_rad <= 0:
            raise ValueError("trajectory solver tolerances must be positive")
        if self.continuity_weight < 0 or self.joint_center_weight < 0:
            raise ValueError("secondary objective weights must be non-negative")
        for elbow in (self.right_elbow, self.left_elbow):
            axis = np.asarray(elbow.outward_axis, dtype=np.float64)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0:
                raise ValueError("outward elbow axes must be finite non-zero 3-D vectors")
            if (
                elbow.weight < 0
                or elbow.finite_difference_rad <= 0
                or not -1.0 <= elbow.target_alignment <= 1.0
            ):
                raise ValueError("invalid outward elbow objective settings")


@dataclass(frozen=True)
class WaypointDiagnostic:
    side: str
    waypoint: int
    converged: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    outward_alignment: float
    min_joint_margin_rad: float
    velocity_saturated: bool
    acceleration_saturated: bool


@dataclass(frozen=True)
class TrajectoryResult:
    values: np.ndarray
    diagnostics: tuple[WaypointDiagnostic, ...]

    @property
    def valid(self) -> bool:
        return all(item.converged for item in self.diagnostics)


class ConstrainedBimanualTrajectorySolver:
    """Decode bimanual TCP waypoints into a deterministic, bounded joint chunk."""

    SIDES = ("right", "left")

    def __init__(self, model_spec: RobotModelSpec, config: TrajectorySolverConfig):
        config.validate()
        self.model_spec = model_spec
        self.config = config
        self.arms = {
            side: MujocoArmKinematics(model_spec, side) for side in self.SIDES
        }

    def _elbow_config(self, side: str) -> OutwardElbowConfig:
        return self.config.right_elbow if side == "right" else self.config.left_elbow

    def outward_alignment(self, side: str, joints: np.ndarray) -> float:
        """Cosine alignment of elbow swivel with the side's outward direction."""
        arm = self.arms[side]
        elbow_config = self._elbow_config(side)
        shoulder = arm.body_position(joints, elbow_config.shoulder_body)
        elbow = arm.body_position(joints, elbow_config.elbow_body)
        tcp, _ = arm.pose(joints)
        axis = tcp - shoulder
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1.0e-9:
            return -1.0
        axis /= axis_norm
        elbow_vector = elbow - shoulder
        elbow_plane = elbow_vector - np.dot(elbow_vector, axis) * axis
        outward = np.asarray(elbow_config.outward_axis, dtype=np.float64)
        outward /= np.linalg.norm(outward)
        outward_plane = outward - np.dot(outward, axis) * axis
        denominator = float(np.linalg.norm(elbow_plane) * np.linalg.norm(outward_plane))
        if denominator < 1.0e-10:
            return -1.0
        return float(np.clip(np.dot(elbow_plane, outward_plane) / denominator, -1.0, 1.0))

    def _outward_gradient(self, side: str, joints: np.ndarray) -> np.ndarray:
        elbow_config = self._elbow_config(side)
        epsilon = elbow_config.finite_difference_rad
        gradient = np.zeros(7, dtype=np.float64)
        for index in range(7):
            plus = joints.copy()
            minus = joints.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            gradient[index] = (
                self.outward_alignment(side, plus) - self.outward_alignment(side, minus)
            ) / (2.0 * epsilon)
        norm = float(np.linalg.norm(gradient))
        return gradient / norm if norm > 1.0e-10 else gradient

    def _dynamic_bounds(
        self,
        arm: MujocoArmKinematics,
        previous: np.ndarray,
        previous_delta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        config = self.config
        hard_lower = arm.limits[:, 0] + config.joint_limit_margin_rad
        hard_upper = arm.limits[:, 1] - config.joint_limit_margin_rad
        if np.any(hard_lower >= hard_upper):
            raise TrajectorySolveError("joint limit margin leaves an empty command range")

        velocity_delta = np.asarray(config.max_velocity_rad_s) * config.dt_seconds
        lower = np.maximum(hard_lower, previous - velocity_delta)
        upper = np.minimum(hard_upper, previous + velocity_delta)
        velocity_saturated = bool(
            np.any(lower > hard_lower + 1.0e-12) or np.any(upper < hard_upper - 1.0e-12)
        )

        acceleration_delta = (
            np.asarray(config.max_acceleration_rad_s2) * config.dt_seconds**2
        )
        lower_acceleration = previous + previous_delta - acceleration_delta
        upper_acceleration = previous + previous_delta + acceleration_delta
        acceleration_saturated = bool(
            np.any(lower_acceleration > lower + 1.0e-12)
            or np.any(upper_acceleration < upper - 1.0e-12)
        )
        lower = np.maximum(lower, lower_acceleration)
        upper = np.minimum(upper, upper_acceleration)
        if np.any(lower > upper):
            raise TrajectorySolveError("position/velocity/acceleration constraints are inconsistent")
        return lower, upper, velocity_saturated, acceleration_saturated

    def _solve_waypoint(
        self,
        *,
        side: str,
        waypoint: int,
        desired_position: np.ndarray,
        desired_rotation: np.ndarray,
        previous: np.ndarray,
        previous_delta: np.ndarray,
    ) -> tuple[np.ndarray, WaypointDiagnostic]:
        arm = self.arms[side]
        config = self.config
        lower, upper, velocity_saturated, acceleration_saturated = self._dynamic_bounds(
            arm, previous, previous_delta
        )
        q = np.clip(previous.copy(), lower, upper)
        width = np.maximum(arm.limits[:, 1] - arm.limits[:, 0], 1.0e-6)
        midpoint = 0.5 * (arm.limits[:, 0] + arm.limits[:, 1])
        elbow_config = self._elbow_config(side)
        position_error = float("inf")
        rotation_error = float("inf")
        converged = False
        iterations = 0

        for iteration in range(1, config.max_iterations + 1):
            iterations = iteration
            position, rotation = arm.pose(q)
            translation_delta = desired_position - position
            rotation_delta = orientation_error(desired_rotation, rotation)
            position_error = float(np.linalg.norm(translation_delta))
            rotation_error = float(np.linalg.norm(rotation_delta))
            error = np.concatenate([translation_delta, rotation_delta])
            jacobian = arm.jacobian(q)
            regularized = jacobian @ jacobian.T + config.damping**2 * np.eye(6)
            try:
                jacobian_inverse = jacobian.T @ np.linalg.solve(
                    regularized, np.eye(6, dtype=np.float64)
                )
            except np.linalg.LinAlgError:
                break
            task_step = jacobian_inverse @ error
            nullspace = np.eye(7, dtype=np.float64) - jacobian_inverse @ jacobian
            alignment = self.outward_alignment(side, q)
            outward_weight = (
                elbow_config.weight if alignment < elbow_config.target_alignment else 0.0
            )
            secondary = (
                outward_weight * self._outward_gradient(side, q)
                - config.continuity_weight * (q - previous) / width
                + config.joint_center_weight * (midpoint - q) / width
            )
            step = task_step + nullspace @ secondary
            step = np.clip(
                step,
                -config.max_iteration_step_rad,
                config.max_iteration_step_rad,
            )
            candidate = np.clip(q + step, lower, upper)
            if np.linalg.norm(candidate - q) < 1.0e-10:
                break
            if (
                position_error <= config.position_tolerance_m
                and rotation_error <= config.orientation_tolerance_rad
                and alignment >= elbow_config.target_alignment
            ):
                converged = True
                break
            q = candidate

        # Recompute diagnostics for the returned, hard-bounded configuration.
        position, rotation = arm.pose(q)
        position_error = float(np.linalg.norm(desired_position - position))
        rotation_error = float(np.linalg.norm(orientation_error(desired_rotation, rotation)))
        converged = (
            position_error <= config.position_tolerance_m
            and rotation_error <= config.orientation_tolerance_rad
        )
        margin = float(
            np.min(np.minimum(q - arm.limits[:, 0], arm.limits[:, 1] - q))
        )
        diagnostic = WaypointDiagnostic(
            side=side,
            waypoint=waypoint,
            converged=converged,
            iterations=iterations,
            position_error_m=position_error,
            orientation_error_rad=rotation_error,
            outward_alignment=self.outward_alignment(side, q),
            min_joint_margin_rad=margin,
            velocity_saturated=velocity_saturated,
            acceleration_saturated=acceleration_saturated,
        )
        return q, diagnostic

    def solve(
        self,
        *,
        positions: np.ndarray,
        rotations: np.ndarray,
        grippers: np.ndarray,
        current_state: np.ndarray,
        require_convergence: bool = True,
    ) -> TrajectoryResult:
        """Solve arrays shaped ``[T,2,3]``, ``[T,2,3,3]``, and ``[T,2]``.

        Output order is right arm joints, right gripper, left arm joints, left
        gripper.  No unsafe fallback is returned when ``require_convergence`` is
        true.
        """
        positions = np.asarray(positions, dtype=np.float64)
        rotations = np.asarray(rotations, dtype=np.float64)
        grippers = np.asarray(grippers, dtype=np.float64)
        current_state = np.asarray(current_state, dtype=np.float64)
        horizon = positions.shape[0] if positions.ndim else 0
        if positions.shape != (horizon, 2, 3):
            raise ValueError("positions must have shape [T,2,3]")
        if rotations.shape != (horizon, 2, 3, 3):
            raise ValueError("rotations must have shape [T,2,3,3]")
        if grippers.shape != (horizon, 2):
            raise ValueError("grippers must have shape [T,2]")
        if current_state.shape != (16,):
            raise ValueError("current_state must have shape [16]")
        if not all(
            np.all(np.isfinite(value))
            for value in (positions, rotations, grippers, current_state)
        ):
            raise ValueError("trajectory inputs must all be finite")

        output = np.zeros((horizon, 16), dtype=np.float64)
        previous = {
            "right": current_state[:7].copy(),
            "left": current_state[8:15].copy(),
        }
        previous_delta = {side: np.zeros(7, dtype=np.float64) for side in self.SIDES}
        diagnostics: list[WaypointDiagnostic] = []
        for waypoint in range(horizon):
            for side_index, side in enumerate(self.SIDES):
                solved, diagnostic = self._solve_waypoint(
                    side=side,
                    waypoint=waypoint,
                    desired_position=positions[waypoint, side_index],
                    desired_rotation=rotations[waypoint, side_index],
                    previous=previous[side],
                    previous_delta=previous_delta[side],
                )
                start = side_index * 8
                output[waypoint, start : start + 7] = solved
                output[waypoint, start + 7] = grippers[waypoint, side_index]
                previous_delta[side] = solved - previous[side]
                previous[side] = solved
                diagnostics.append(diagnostic)

        result = TrajectoryResult(output, tuple(diagnostics))
        if require_convergence and not result.valid:
            first = next(item for item in result.diagnostics if not item.converged)
            raise TrajectorySolveError(
                f"{first.side} waypoint {first.waypoint} is infeasible under hard bounds: "
                f"position_error={first.position_error_m:.4f}m, "
                f"orientation_error={first.orientation_error_rad:.4f}rad"
            )
        return result
