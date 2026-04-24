#!/usr/bin/env python3
"""Replay a Neuracore dataset recording through a trained policy.

Pulls a synchronized recording from a Neuracore dataset, feeds each sync
point through the policy in chunk-playback mode (predict() once per chunk,
pop one action per step), and writes a CSV in the same schema the live
ROS2 inference node produces — so `plot_predictions.py` works on it
unchanged.

This is offline reanalysis: no robot, no ROS, no live cameras. Useful for
sanity-checking that the embodiment description matches training data and
for visualizing chunk-boundary jumps without running the robot.

Usage:
    NEURACORE_API_KEY=... python3 replay_dataset_through_policy.py \
        --dataset kit-block \
        --train-run-name kit-block-dp \
        --recording-idx 0 \
        --num-points 250 \
        --device cuda \
        --out /tmp/replay.csv

Then:
    python3 plot_predictions.py /tmp/replay.csv --chunks 2 --show
(or pass --plot to do it automatically)
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, cast

import neuracore as nc
import numpy as np
import torch
from neuracore.core.data.recording import Recording
from neuracore_types import DataType, SynchronizedPoint
from neuracore_types.nc_data.camera_data import RGBCameraData
from neuracore_types.nc_data.joint_data import JointData
from neuracore_types.nc_data.parallel_gripper_open_amount_data import (
    ParallelGripperOpenAmountData,
)


LEFT_ARM = [f"follower_l_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER = "follower_l_finger_joint1"

GRIPPER_LO = 0.0
GRIPPER_HI = 0.05


def gripper_normalize(value: float) -> float:
    return max(0.0, min(1.0, (value - GRIPPER_LO) / (GRIPPER_HI - GRIPPER_LO)))


# Must match the live inference node — anyone retraining must update both.
INPUT_DESC = {
    DataType.JOINT_POSITIONS: {
        2:  "follower_l_joint2",
        4:  "follower_l_joint7",
        5:  "follower_l_joint1",
        6:  "follower_l_joint4",
        7:  "follower_l_joint3",
        11: "follower_l_joint5",
        13: "follower_l_joint6",
    },
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {
        0: "follower_l_finger_joint1",
    },
    DataType.RGB_IMAGES: {
        0: "cam_wrist_l",
        1: "cam_waist",
        2: "cam_chest",
    },
}
OUTPUT_DESC = {
    DataType.JOINT_TARGET_POSITIONS: {
        0: "follower_l_joint2",
        1: "follower_l_joint1",
        2: "follower_l_joint7",
        3: "follower_l_joint4",
        4: "follower_l_joint3",
        5: "follower_l_joint5",
        6: "follower_l_joint6",
    },
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: {
        0: "follower_l_finger_joint1",
    },
}


def cross_embodiment_union_for(robot_name: str):
    return {
        robot_name: {
            DataType.JOINT_POSITIONS: LEFT_ARM,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: [LEFT_GRIPPER],
            DataType.RGB_IMAGES: ["cam_wrist_l", "cam_waist", "cam_chest"],
        }
    }


def obs_arm_from_sync_point(sync_point) -> Optional[np.ndarray]:
    joint_data = sync_point.data.get(DataType.JOINT_POSITIONS)
    if not joint_data:
        return None
    try:
        return np.array(
            [float(joint_data[n].value) for n in LEFT_ARM], dtype=np.float64
        )
    except KeyError:
        return None


def obs_grip_from_sync_point(sync_point) -> float:
    grip = sync_point.data.get(DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS, {}).get(
        LEFT_GRIPPER
    )
    if grip is None:
        return float("nan")
    # Stored amounts come in [0, 1] already; clamp defensively.
    return max(0.0, min(1.0, float(grip.open_amount)))


def make_synthetic_sync_point(step: int) -> SynchronizedPoint:
    """Build a zero-filled SynchronizedPoint with the structure the policy expects.

    Joints + gripper at 0; cameras as 480x640 black RGB frames. Useful for
    smoke-testing predict() when neuracore's per-recording sync is failing.
    """
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    return SynchronizedPoint(
        timestamp=float(step),
        data={
            DataType.JOINT_POSITIONS: {
                name: JointData(value=0.0) for name in LEFT_ARM
            },
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {
                LEFT_GRIPPER: ParallelGripperOpenAmountData(open_amount=0.0),
            },
            DataType.RGB_IMAGES: {
                cam: RGBCameraData(frame=black, frame_idx=step)
                for cam in ("cam_wrist_l", "cam_waist", "cam_chest")
            },
        },
    )


def chunk_from_prediction(preds) -> np.ndarray:
    joint_preds = preds[DataType.JOINT_TARGET_POSITIONS]
    arm = torch.cat([joint_preds[n].value for n in LEFT_ARM], dim=2)[0]
    grip_preds = preds.get(DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS, {})
    if grip_preds:
        grip = grip_preds[LEFT_GRIPPER].open_amount[0]  # (horizon, 1)
        out = torch.cat([arm, grip], dim=1)
    else:
        out = arm
    return out.detach().cpu().numpy().astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None,
                    help="Neuracore dataset name (the one used to train). "
                         "Required unless --synthetic is set.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Skip neuracore data fetch entirely; feed zero-filled sync points "
                         "(joints=0, black images) into the policy. Just exercises the "
                         "predict() path — outputs are meaningless robot-wise.")
    ap.add_argument("--train-run-name", default=os.environ.get("NEURACORE_TRAIN_RUN_NAME", ""),
                    help="Training run to load (or pass --model-file)")
    ap.add_argument("--model-file", default=os.environ.get("NEURACORE_MODEL_FILE", ""),
                    help="Path to local .nc.zip (overrides --train-run-name)")
    ap.add_argument("--robot-name", default=os.environ.get("NEURACORE_ROBOT_NAME", "anvil_openarm"))
    ap.add_argument("--recording-idx", type=int, default=0,
                    help="Which recording in the dataset to replay")
    ap.add_argument("--frequency", type=int, default=50,
                    help="Synchronization frequency Hz (must match training run)")
    ap.add_argument("--num-points", type=int, default=250,
                    help="How many sync points to feed (default 250 = 2.5 chunks @ horizon 100)")
    ap.add_argument("--predict-every", type=int, default=100,
                    help="Steps between predict() calls (default 100 = chunk size; "
                         "lower values simulate receding-horizon)")
    ap.add_argument("--device", default=os.environ.get("NEURACORE_DEVICE", "cuda"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/replay.csv"),
                    help="CSV output path (same schema as inference_node)")
    ap.add_argument("--plot", action="store_true",
                    help="After writing CSV, invoke plot_predictions.py on it")
    ap.add_argument("--plot-chunks", type=int, default=2,
                    help="Chunks to render when --plot is set")
    args = ap.parse_args()

    if not args.train_run_name and not args.model_file:
        print("ERROR: pass --train-run-name or --model-file (or set NEURACORE_TRAIN_RUN_NAME / NEURACORE_MODEL_FILE)",
              file=sys.stderr)
        return 2

    if not args.synthetic and not args.dataset:
        print("ERROR: --dataset is required unless --synthetic is set", file=sys.stderr)
        return 2

    api_key = os.environ.get("NEURACORE_API_KEY", "")
    if not api_key:
        print("ERROR: NEURACORE_API_KEY not set", file=sys.stderr)
        return 2

    print(f"[replay] login")
    nc.login(api_key=api_key)

    print(f"[replay] connect_robot('{args.robot_name}')")
    nc.connect_robot(args.robot_name)

    recording = None
    if not args.synthetic:
        print(f"[replay] get_dataset(name='{args.dataset}')")
        dataset = nc.get_dataset(name=args.dataset)

        union = cross_embodiment_union_for(args.robot_name)

        if args.recording_idx >= len(dataset):
            print(f"ERROR: recording_idx {args.recording_idx} >= {len(dataset)} recordings",
                  file=sys.stderr)
            return 2
        raw_recording = cast(Recording, dataset[args.recording_idx])
        print(f"[replay] selected recording '{raw_recording.name}' (id={raw_recording.id})")
        print(f"[replay] synchronize(frequency={args.frequency}) — this recording only")
        recording = raw_recording.synchronize(
            frequency=args.frequency,
            cross_embodiment_union=union,
        )
        n_total = len(recording)
        n_use = min(args.num_points, n_total)
        print(f"[replay] recording has {n_total} sync points; using first {n_use}")
    else:
        n_use = args.num_points
        print(f"[replay] synthetic mode: feeding {n_use} zero-filled sync points")

    src = f"model_file={args.model_file}" if args.model_file else f"train_run_name={args.train_run_name}"
    print(f"[replay] loading policy ({src}) on device={args.device}")
    t0 = time.perf_counter()
    policy = nc.policy(
        input_embodiment_description=INPUT_DESC,
        output_embodiment_description=OUTPUT_DESC,
        model_file=args.model_file or None,
        train_run_name=args.train_run_name or None,
        device=args.device,
    )
    print(f"[replay] policy loaded in {time.perf_counter() - t0:.2f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_arm = len(LEFT_ARM)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["t", "chunk_id", "chunk_idx"]
        header += [f"obs_{n}" for n in LEFT_ARM]
        header += ["obs_grip"]
        header += [f"out_{n}" for n in LEFT_ARM]
        header += ["out_grip"]
        writer.writerow(header)

        chunk: Optional[np.ndarray] = None
        chunk_idx = 0
        chunk_id = -1
        chunk_obs_arm: Optional[np.ndarray] = None
        chunk_obs_grip: float = float("nan")

        for step in range(n_use):
            if args.synthetic:
                sync_point = make_synthetic_sync_point(step)
            else:
                assert recording is not None
                sync_point = cast(SynchronizedPoint, recording[step])

            if chunk is None or chunk_idx >= args.predict_every or chunk_idx >= len(chunk):
                chunk_obs_arm = obs_arm_from_sync_point(sync_point)
                chunk_obs_grip = obs_grip_from_sync_point(sync_point)
                chunk_id += 1

                tp = time.perf_counter()
                preds = policy.predict(sync_point=sync_point, timeout=10.0)
                chunk = chunk_from_prediction(preds)
                chunk_idx = 0
                print(f"[replay] step={step:4d} chunk_id={chunk_id} predict={1e3*(time.perf_counter()-tp):.1f}ms "
                      f"horizon={chunk.shape[0]}")

            action = chunk[chunk_idx]
            target_arm = action[:n_arm].tolist()
            out_grip = float(action[n_arm]) if action.shape[0] >= n_arm + 1 else float("nan")

            row = [time.time(), chunk_id, chunk_idx]
            row += (chunk_obs_arm.tolist() if chunk_obs_arm is not None
                    else [float("nan")] * n_arm)
            row += [chunk_obs_grip]
            row += target_arm
            row += [out_grip]
            writer.writerow(row)

            chunk_idx += 1

    print(f"[replay] wrote {args.out}")

    if args.plot:
        plot_script = Path(__file__).resolve().parent / "plot_predictions.py"
        cmd = [sys.executable, str(plot_script), str(args.out),
               "--chunks", str(args.plot_chunks)]
        print(f"[replay] running: {' '.join(cmd)}")
        subprocess.check_call(cmd)

    try:
        policy.disconnect()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
