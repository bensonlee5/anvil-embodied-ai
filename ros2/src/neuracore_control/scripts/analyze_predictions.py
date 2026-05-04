#!/usr/bin/env python3
"""Run the trained policy on observations from data/episode_<idx>/ and
plot predicted action chunks against the demo's commanded targets
from joint_targets.csv. Observations come from joint_observations.csv
(plus images) — those go into the policy as inputs.

Run download_data.py first.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# Make `neuracore_control.common` importable when this script is run directly
# (`python scripts/analyze_predictions.py`) without `colcon build` / `pip
# install -e .`. The inner package sits at <pkg_root>/neuracore_control/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import neuracore as nc
import numpy as np
import torch
from PIL import Image
from neuracore_types import DataType

from neuracore_control.common import (
    CAMERAS,
    DEFAULT_TRAIN_RUN_NAME,
    LEFT_ARM,
    LEFT_GRIPPER,
    get_model_embodiments,
)


def chunk_from_prediction(preds) -> np.ndarray:
    """Decode policy.predict() output to (horizon, n_arm + 1) array."""
    j = preds[DataType.JOINT_TARGET_POSITIONS]
    g = preds[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS][LEFT_GRIPPER]
    arm = torch.cat([j[n].value for n in LEFT_ARM], dim=2)[0]
    out = torch.cat([arm, g.open_amount[0]], dim=1)
    return out.detach().cpu().numpy().astype(np.float64)


def detect_motion_onset(rows: list[dict], threshold: float) -> int:
    """First frame where any joint has moved > threshold rad from rest."""
    base = np.array([float(rows[0][n]) for n in LEFT_ARM])
    for i, r in enumerate(rows):
        cur = np.array([float(r[n]) for n in LEFT_ARM])
        if np.max(np.abs(cur - base)) > threshold:
            return i
    return 0


def log_observation(data_dir: Path, row: dict, frame: int) -> None:
    """Log joints, gripper, and images via nc.log_*; predict() reads from there."""
    nc.log_joint_positions({n: float(row[n]) for n in LEFT_ARM})
    nc.log_parallel_gripper_open_amounts({LEFT_GRIPPER: float(row[LEFT_GRIPPER])})
    for cam_name in CAMERAS:
        img = np.asarray(Image.open(
            data_dir / "images" / cam_name / f"frame_{frame:04d}.jpg"
        ).convert("RGB"))
        nc.log_rgb(cam_name, img)


def gt_array(rows: list[dict], start: int, length: int) -> np.ndarray:
    """(length, n_arm + 1) GT joint+gripper from rows[start:start+length]."""
    out = np.zeros((length, len(LEFT_ARM) + 1))
    for i in range(length):
        r = rows[start + i]
        out[i, : len(LEFT_ARM)] = [float(r[n]) for n in LEFT_ARM]
        out[i, -1] = float(r[LEFT_GRIPPER])
    return out


def plot(
    predictions: list[tuple[int, np.ndarray]],
    gt: np.ndarray,
    gt_start: int,
    out_path: Path,
    model_label: str,
) -> None:
    labels = LEFT_ARM + ["gripper"]
    gt_x = np.arange(gt_start, gt_start + len(gt))
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(gt_x, gt[:, j], color="C0", label="ground truth")
        for k, (obs_idx, pred) in enumerate(predictions):
            x = np.arange(obs_idx + 1, obs_idx + 1 + pred.shape[0])
            label = f"pred @ frame {obs_idx}" if j == 0 else None
            ax.plot(x, pred[:, j], color=f"C{k + 1}", label=label)
        ax.set_title(labels[j])
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("frame")
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=min(len(lbls), 5))
    n = len(predictions)
    fig.suptitle(
        f"{model_label}: policy predictions "
        f"({n} chunk{'s' if n != 1 else ''}) vs ground truth"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=120)
    print(f"saved plot to {out_path}")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-file", type=Path, default=None)
    ap.add_argument("--train-run-name", default=DEFAULT_TRAIN_RUN_NAME)
    ap.add_argument("--robot-name", default="anvil_openarm")
    ap.add_argument("--episode-idx", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-chunks", type=int, default=3)
    ap.add_argument(
        "--start-frame", default="auto", help="Integer or 'auto' (motion-onset detection)"
    )
    ap.add_argument("--motion-threshold", type=float, default=0.02)
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parent / "data" / f"episode_{args.episode_idx:03d}"
    with (data_dir / "joint_observations.csv").open() as f:
        obs_rows = list(csv.DictReader(f))
    with (data_dir / "joint_targets.csv").open() as f:
        tgt_rows = list(csv.DictReader(f))

    if args.start_frame == "auto":
        start_frame = detect_motion_onset(tgt_rows, args.motion_threshold)
        print(f"motion onset (in commanded targets) at frame {start_frame}")
    else:
        start_frame = int(args.start_frame)

    embodiment = get_model_embodiments(args.train_run_name)

    nc.login()
    nc.connect_robot(args.robot_name)
    src = (
        f"model_file={args.model_file}"
        if args.model_file
        else f"train_run_name={args.train_run_name}"
    )
    print(f"loading policy ({src}) on {args.device}")
    t0 = time.perf_counter()
    policy = nc.policy(
        input_embodiment_description=embodiment["input"],
        output_embodiment_description=embodiment["output"],
        model_file=str(args.model_file) if args.model_file else None,
        train_run_name=args.train_run_name,
        device=args.device,
    )
    print(f"policy loaded in {time.perf_counter() - t0:.2f}s")

    predictions: list[tuple[int, np.ndarray]] = []
    obs_idx = start_frame
    for k in range(args.num_chunks):
        if obs_idx >= len(obs_rows) - 1:
            break
        log_observation(data_dir, obs_rows[obs_idx], obs_idx)
        tp = time.perf_counter()
        pred = chunk_from_prediction(policy.predict(timeout=10.0))
        print(
            f"  chunk {k} obs_frame={obs_idx:4d} "
            f"predict {1e3 * (time.perf_counter() - tp):.1f}ms "
            f"horizon={pred.shape[0]}"
        )
        predictions.append((obs_idx, pred))
        obs_idx += pred.shape[0]

    gt_start = predictions[0][0] + 1
    gt_length = min(
        predictions[-1][0] + predictions[-1][1].shape[0] - gt_start + 1,
        len(tgt_rows) - gt_start,
    )
    gt = gt_array(tgt_rows, gt_start, gt_length)

    model_label = (
        Path(args.model_file).stem if args.model_file else args.train_run_name
    )
    out_path = (
        data_dir
        / f"predictions_{model_label}_f{start_frame:04d}_n{len(predictions)}.png"
    )
    plot(predictions, gt, gt_start, out_path, model_label)


if __name__ == "__main__":
    main()
