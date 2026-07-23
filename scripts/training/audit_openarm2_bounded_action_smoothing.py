#!/usr/bin/env python3
"""Audit bounded cubic B-spline action smoothing on the frozen train split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anvil_trainer.bounded_actions import (
    BoundedActionContract,
    smooth_bounded_action_chunk,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    ROOT
    / "datasets/shirt-fold/lerobot-hf-phase-aligned-sarm-5stage-v1/data/chunk-000/file-000.parquet"
)
DEFAULT_CONTRACT = ROOT / "configs/training/action_contracts/openarm2_shirt_fold_bounded_v2.json"
DEFAULT_OUTPUT = (
    ROOT / "configs/training/action_contracts/openarm2_shirt_fold_bounded_v2_audit.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_action_smoothing(
    data_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    data = Path(data_path).expanduser().resolve()
    contract = BoundedActionContract.load(contract_path)
    frame = pd.read_parquet(data, columns=["action", "episode_index"])
    actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
    episodes = frame["episode_index"].to_numpy(dtype=np.int64)
    starts: list[int] = []
    for episode_index in contract.training_episode_indices:
        indices = np.flatnonzero(episodes == episode_index)
        starts.extend(indices[: -contract.chunk_size + 1].tolist())
    start_array = np.asarray(starts, dtype=np.int64)
    if len(start_array) != 28_451:
        raise ValueError(f"Unexpected number of train chunks: {len(start_array)}")

    lower = torch.as_tensor(contract.soft_lower, dtype=torch.float64)
    upper = torch.as_tensor(contract.soft_upper, dtype=torch.float64)
    arm = list(contract.arm_indices)
    absolute = list(contract.absolute_indices)
    adjustments: list[np.ndarray] = []
    derivative_before: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    derivative_after: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    grippers_exact = True
    endpoints_exact = True
    event_boundaries_exact = True
    finite = True
    arm_inside_limits = True

    for offset in range(0, len(start_array), 256):
        batch_starts = start_array[offset : offset + 256]
        chunks = np.stack([actions[start : start + contract.chunk_size] for start in batch_starts])
        source = torch.from_numpy(chunks)
        smoothed_tensor = smooth_bounded_action_chunk(
            source,
            lower=lower,
            upper=upper,
            arm_indices=contract.arm_indices,
            absolute_indices=contract.absolute_indices,
            kernel=contract.inference_smoothing_kernel,
            passes=contract.inference_smoothing_passes,
            gripper_event_threshold=contract.gripper_event_threshold,
        )
        smoothed = smoothed_tensor.numpy()
        finite &= bool(np.isfinite(smoothed).all())
        grippers_exact &= bool(np.array_equal(smoothed[..., absolute], chunks[..., absolute]))
        endpoints_exact &= bool(
            np.array_equal(
                smoothed[:, [0, -1]][..., arm],
                chunks[:, [0, -1]][..., arm],
            )
        )
        arm_inside_limits &= bool(
            np.all(smoothed[..., arm] >= contract.soft_lower[arm] - 1e-6)
            and np.all(smoothed[..., arm] <= contract.soft_upper[arm] + 1e-6)
        )
        adjustment = np.abs(smoothed[..., arm] - chunks[..., arm])
        adjustments.append(adjustment.reshape(-1))
        for order in (1, 2, 3):
            derivative_before[order].append(
                np.abs(np.diff(chunks[..., arm], n=order, axis=1)).reshape(-1)
            )
            derivative_after[order].append(
                np.abs(np.diff(smoothed[..., arm], n=order, axis=1)).reshape(-1)
            )
        changes = np.any(
            np.abs(chunks[:, 1:, absolute] - chunks[:, :-1, absolute])
            >= contract.gripper_event_threshold,
            axis=-1,
        )
        for event in np.argwhere(changes):
            batch_index = int(event[0])
            boundary = int(event[1]) + 1
            event_boundaries_exact &= bool(
                np.array_equal(
                    smoothed[batch_index, [boundary - 1, boundary]][:, arm],
                    chunks[batch_index, [boundary - 1, boundary]][:, arm],
                )
            )

    adjustment = np.concatenate(adjustments)
    ratios = {}
    for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
        before = np.concatenate(derivative_before[order]).mean()
        after = np.concatenate(derivative_after[order]).mean()
        ratios[f"{name}_absolute_difference_ratio"] = float(after / before)
    checks = {
        "finite": finite,
        "arm_inside_soft_limits": arm_inside_limits,
        "grippers_exact_passthrough": grippers_exact,
        "chunk_endpoints_exact": endpoints_exact,
        "gripper_event_boundaries_exact": event_boundaries_exact,
        "p99_arm_adjustment_at_most_0_01_rad": float(np.quantile(adjustment, 0.99)) <= 0.01,
        "acceleration_not_increased": ratios["acceleration_absolute_difference_ratio"] <= 1.0,
        "jerk_not_increased": ratios["jerk_absolute_difference_ratio"] <= 1.0,
    }
    return {
        "schema_version": "openarm2.bounded-action-smoothing-audit.v1",
        "contract_sha256": contract.sha256,
        "data_parquet_sha256": _sha256(data),
        "split": "train",
        "episode_indices": list(contract.training_episode_indices),
        "chunks": len(start_array),
        "chunk_size": contract.chunk_size,
        "smoothing": {
            "method": "uniform_cubic_bspline",
            "kernel": list(contract.inference_smoothing_kernel),
            "passes": contract.inference_smoothing_passes,
            "gripper_event_threshold": contract.gripper_event_threshold,
        },
        "metrics": {
            "arm_adjustment_mean_rad": float(adjustment.mean()),
            "arm_adjustment_p99_rad": float(np.quantile(adjustment, 0.99)),
            "arm_adjustment_max_rad": float(adjustment.max()),
            **ratios,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = audit_action_smoothing(args.data, args.contract)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.require_pass and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
