#!/usr/bin/env python3
"""Audit task-space encoding and constrained decoding on a pinned LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
from anvil_embodiment.kinematics import get_model_spec
from anvil_embodiment.trajectory import ConstrainedBimanualTrajectorySolver

from anvil_trainer.config import TrainingConfig
from anvil_trainer.patches import TransformRunner
from anvil_trainer.task_space_actions import (
    TaskSpaceActionContract,
    denormalize_task_space_actions,
    encode_task_space_actions,
    smooth_task_space_chunk,
    task_space_values_to_targets,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet files under {root}")
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    for path in files:
        table = pq.read_table(
            path,
            columns=["action", "observation.state", "episode_index"],
        )
        actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float64))
        states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float64))
        episodes.append(np.asarray(table["episode_index"], dtype=np.int64))
    return (
        np.concatenate(actions),
        np.concatenate(states),
        np.concatenate(episodes),
        files,
    )


def _fit_statistics(
    contract: TaskSpaceActionContract,
    actions: np.ndarray,
    states: np.ndarray,
    episodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    runner = TransformRunner(
        TrainingConfig(task_space_action_contract=str(contract.path))
    )
    dataset = SimpleNamespace(
        hf_dataset={
            "action": actions,
            "observation.state": states,
            "episode_index": episodes,
        }
    )
    policy = SimpleNamespace(
        type="pi05",
        use_relative_actions=False,
        chunk_size=contract.chunk_size,
        action_feature_names=list(contract.task_action_names),
    )
    fitted = runner._fit_task_space_action_statistics(
        dataset,
        policy,
        list(contract.training_episode_indices),
    )
    if fitted is None:
        raise RuntimeError("task-space statistics were not fit")
    return fitted[0], fitted[1], runner._normalization_contract


def audit(
    *,
    dataset_root: Path,
    contract_path: Path,
    stride: int,
    max_chunks: int | None,
) -> dict:
    contract = TaskSpaceActionContract.load(contract_path)
    actions, states, episodes, parquet_files = _load_dataset(dataset_root)
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    action_names = tuple(info["features"]["action"]["names"])
    state_names = tuple(info["features"]["observation.state"]["names"])
    if action_names != contract.source_action_names or state_names != action_names:
        raise ValueError("dataset action/state names do not match the source contract")
    centers, scales, normalization = _fit_statistics(
        contract, actions, states, episodes
    )

    solver = ConstrainedBimanualTrajectorySolver(
        get_model_spec(contract.model_id), contract.solver
    )
    center_tensor = torch.as_tensor(centers, dtype=torch.float64)
    scale_tensor = torch.as_tensor(scales, dtype=torch.float64)
    horizon = contract.chunk_size
    starts = [
        index
        for index in range(0, len(actions) - horizon + 1, stride)
        if episodes[index] == episodes[index + horizon - 1]
    ]
    if max_chunks is not None:
        starts = starts[:max_chunks]
    if not starts:
        raise ValueError("no complete same-episode chunks selected")

    pose_position_errors: list[float] = []
    pose_rotation_errors: list[float] = []
    waypoint_diagnostics = []
    hard_bound_violations = 0
    velocity_violations = 0
    acceleration_violations = 0
    soft_lower = contract.source_soft_lower
    soft_upper = contract.source_soft_upper
    velocity_delta = (
        np.asarray(contract.solver.max_velocity_rad_s) * contract.solver.dt_seconds
    )
    acceleration_delta = (
        np.asarray(contract.solver.max_acceleration_rad_s2)
        * contract.solver.dt_seconds**2
    )

    for start in starts:
        target = torch.as_tensor(
            actions[start : start + horizon], dtype=torch.float64
        ).unsqueeze(0)
        state = torch.as_tensor(states[start], dtype=torch.float64).unsqueeze(0)
        encoded = encode_task_space_actions(
            target,
            state,
            contract=contract,
            center=center_tensor,
            scale=scale_tensor,
        )
        task_values = denormalize_task_space_actions(
            encoded,
            contract=contract,
            center=center_tensor,
            scale=scale_tensor,
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
        result = solver.solve(
            positions=positions[0].numpy(),
            rotations=rotations[0].numpy(),
            grippers=grippers[0].numpy(),
            current_state=states[start],
            require_convergence=False,
        )
        waypoint_diagnostics.extend(result.diagnostics)
        hard_bound_violations += int(
            np.count_nonzero(
                (result.values < soft_lower - 1.0e-10)
                | (result.values > soft_upper + 1.0e-10)
            )
        )
        for arm_start in (0, 8):
            solved = result.values[:, arm_start : arm_start + 7]
            previous = np.concatenate(
                (states[start, arm_start : arm_start + 7][None, :], solved[:-1]),
                axis=0,
            )
            deltas = solved - previous
            velocity_violations += int(
                np.count_nonzero(np.abs(deltas) > velocity_delta + 1.0e-10)
            )
            previous_deltas = np.concatenate(
                (np.zeros((1, 7)), deltas[:-1]),
                axis=0,
            )
            acceleration_violations += int(
                np.count_nonzero(
                    np.abs(deltas - previous_deltas)
                    > acceleration_delta + 1.0e-10
                )
            )
            side = "right" if arm_start == 0 else "left"
            for waypoint in range(horizon):
                actual_position, actual_rotation = solver.arms[side].pose(solved[waypoint])
                desired_position = positions[0, waypoint, 0 if side == "right" else 1].numpy()
                desired_rotation = rotations[0, waypoint, 0 if side == "right" else 1].numpy()
                pose_position_errors.append(
                    float(np.linalg.norm(actual_position - desired_position))
                )
                relative = desired_rotation @ actual_rotation.T
                pose_rotation_errors.append(
                    float(
                        np.arccos(
                            np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
                        )
                    )
                )

    converged = np.asarray([item.converged for item in waypoint_diagnostics])
    alignments = np.asarray(
        [item.outward_alignment for item in waypoint_diagnostics], dtype=np.float64
    )
    position_errors = np.asarray(pose_position_errors, dtype=np.float64)
    rotation_errors = np.asarray(pose_rotation_errors, dtype=np.float64)
    return {
        "schema_version": 1,
        "representation_id": contract.representation_id,
        "contract_path": str(contract.path),
        "contract_sha256": contract.sha256,
        "deployment_status": contract.deployment_status,
        "dataset_root": str(dataset_root.resolve()),
        "dataset_info_sha256": _sha256(dataset_root / "meta" / "info.json"),
        "data_parquet": [
            {"path": str(path), "sha256": _sha256(path)} for path in parquet_files
        ],
        "rows": int(len(actions)),
        "episodes": int(len(np.unique(episodes))),
        "fit_episode_indices": list(contract.training_episode_indices),
        "normalization_contract": normalization,
        "sampled_chunk_stride": stride,
        "sampled_chunks": len(starts),
        "sampled_waypoints_per_arm": len(starts) * horizon,
        "solver": {
            "converged_fraction": float(converged.mean()),
            "rejected_waypoints": int(np.count_nonzero(~converged)),
            "position_error_m": {
                "mean": float(position_errors.mean()),
                "p95": float(np.quantile(position_errors, 0.95)),
                "max": float(position_errors.max()),
            },
            "orientation_error_rad": {
                "mean": float(rotation_errors.mean()),
                "p95": float(np.quantile(rotation_errors, 0.95)),
                "max": float(rotation_errors.max()),
            },
            "outward_alignment": {
                "mean": float(alignments.mean()),
                "p05": float(np.quantile(alignments, 0.05)),
                "min": float(alignments.min()),
                "target": contract.solver.right_elbow.target_alignment,
            },
            "hard_bound_violations": hard_bound_violations,
            "velocity_violations": velocity_violations,
            "acceleration_violations": acceleration_violations,
        },
        "production_gate": {
            "passed": False,
            "reason": (
                "contract is offline_only; full collision geometry and deployed "
                "controller parity are not implemented"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=300)
    parser.add_argument("--max-chunks", type=int)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    report = audit(
        dataset_root=args.dataset,
        contract_path=args.contract,
        stride=args.stride,
        max_chunks=args.max_chunks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["solver"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
