#!/usr/bin/env python3
"""Generate synthetic EE-space fixture MCAPs for the smoke test.

Creates tests/smoke/fixtures/ee-session/{0001..0005}/ with:
  - <episode>_0.mcap  — /ee_pose_right + 3 camera topics at 30 Hz, ~4 s
  - metadata.json     — version/status/duration
  - metadata.yaml     — rosbag2 bagfile info

Run from repo root:
  uv run python tests/smoke/fixtures/scripts/generate_ee_fixtures.py
"""
from __future__ import annotations

import io
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

OUT_DIR = Path(__file__).resolve().parents[1] / "ee-session"

# ── Recording parameters ──────────────────────────────────────────────────────
N_EPISODES = 5
FPS        = 30
N_FRAMES   = 120                           # 4 s at 30 Hz
DT_NS      = int(1e9 / FPS)               # 33_333_333 ns

CAMERA_TOPICS = [
    "/cam_waist/image_raw/compressed",
    "/cam_wrist_r/image_raw/compressed",
    "/cam_chest/image_raw/compressed",
]
EE_TOPIC = "/ee_pose_right"

# ── Message definitions ───────────────────────────────────────────────────────

EE_POSE_MSGDEF = """\
std_msgs/Header header
geometry_msgs/Pose pose
float64 gripper

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w"""

COMPRESSED_IMAGE_MSGDEF = """\
std_msgs/Header header
string format
uint8[] data

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg(r: int, g: int, b: int, w: int = 4, h: int = 4) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (r, g, b)).save(buf, "JPEG")
    return buf.getvalue()


def _ns_to_stamp(ns: int) -> dict:
    sec, nanosec = divmod(ns, int(1e9))
    return {"sec": int(sec), "nanosec": int(nanosec)}


def _ee_msg(ts_ns: int, pos: np.ndarray, quat_xyzw: np.ndarray, gripper: float) -> dict:
    stamp = _ns_to_stamp(ts_ns)
    return {
        "header": {"stamp": stamp, "frame_id": "world"},
        "pose": {
            "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "orientation": {
                "x": float(quat_xyzw[0]),
                "y": float(quat_xyzw[1]),
                "z": float(quat_xyzw[2]),
                "w": float(quat_xyzw[3]),
            },
        },
        "gripper": float(gripper),
    }


def _cam_msg(ts_ns: int, jpeg_data: bytes, frame_id: str) -> dict:
    stamp = _ns_to_stamp(ts_ns)
    return {
        "header": {"stamp": stamp, "frame_id": frame_id},
        "format": "jpeg",
        "data": jpeg_data,
    }


# ── Trajectory ────────────────────────────────────────────────────────────────

_HOME = np.array([0.30, -0.20, 0.10])  # nominal right-arm EE home (metres)

def _ee_trajectory(ep_idx: int, frame_idx: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (pos, quat_xyzw, gripper) for a smooth sinusoidal trajectory.

    Each episode has a slightly different phase offset so episodes are distinct.
    Rotation stays near identity (small oscillation around z).
    """
    t = frame_idx / FPS                    # time in seconds
    phase = ep_idx * 0.4                   # episode-specific offset

    # Small sinusoidal motion (±2 cm amplitude)
    pos = _HOME + np.array([
        0.02 * math.sin(2 * math.pi * 0.5 * t + phase),        # 0.5 Hz x
        0.015 * math.cos(2 * math.pi * 0.3 * t + phase),       # 0.3 Hz y
        0.01 * math.sin(2 * math.pi * 0.7 * t + phase + 1.0),  # 0.7 Hz z
    ])

    # Small rotation around z axis (±5°)
    angle = math.radians(5) * math.sin(2 * math.pi * 0.2 * t + phase)
    quat_xyzw = np.array([
        0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)
    ])

    # Gripper: slightly open/close cycle
    gripper = 0.02 + 0.005 * math.sin(2 * math.pi * 0.1 * t)

    return pos, quat_xyzw, gripper


# ── Episode writer ────────────────────────────────────────────────────────────

def write_episode(ep_idx: int, out_dir: Path) -> None:
    """Write one episode MCAP + metadata files."""
    from mcap_ros2.writer import Writer

    ep_num = ep_idx + 1
    ep_dir = out_dir / f"{ep_num:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    mcap_path = ep_dir / f"{ep_num:04d}_0.mcap"

    # Pre-generate JPEG bytes (one per camera, slightly different colour per episode)
    base_colour = [(180, 100, 100), (100, 180, 100), (100, 100, 180)]
    jpegs = [_make_jpeg(*(c[0]+ep_idx*10, c[1], c[2])) for c in base_colour]

    total_duration_ns = (N_FRAMES - 1) * DT_NS

    with open(mcap_path, "wb") as f, Writer(f) as w:
        ee_schema  = w.register_msgdef("anvil_msgs/msg/CommandedEEPose", EE_POSE_MSGDEF)
        cam_schema = w.register_msgdef("sensor_msgs/msg/CompressedImage", COMPRESSED_IMAGE_MSGDEF)

        for frame in range(N_FRAMES):
            ts_ns = frame * DT_NS

            # EE pose
            pos, quat, grip = _ee_trajectory(ep_idx, frame)
            ee_m = _ee_msg(ts_ns, pos, quat, grip)
            w.write_message(EE_TOPIC, ee_schema, ee_m, log_time=ts_ns)

            # Cameras
            for cam_i, cam_topic in enumerate(CAMERA_TOPICS):
                cam_m = _cam_msg(ts_ns, jpegs[cam_i], cam_topic.split("/")[1])
                w.write_message(cam_topic, cam_schema, cam_m, log_time=ts_ns)

    # metadata.json
    meta_json = {"version": 1, "status": "success", "note": None, "duration": 4}
    (ep_dir / "metadata.json").write_text(json.dumps(meta_json, indent=2))

    # metadata.yaml
    msg_count_per_topic = N_FRAMES
    topic_entries = [
        {
            "name": EE_TOPIC,
            "type": "anvil_msgs/msg/CommandedEEPose",
            "count": msg_count_per_topic,
        },
        *[
            {
                "name": t,
                "type": "sensor_msgs/msg/CompressedImage",
                "count": msg_count_per_topic,
            }
            for t in CAMERA_TOPICS
        ],
    ]
    total_msgs = msg_count_per_topic * (1 + len(CAMERA_TOPICS))

    topics_yaml = "\n".join(
        f"""\
  - message_count: {e['count']}
    topic_metadata:
      name: {e['name']}
      offered_qos_profiles: []
      serialization_format: cdr
      type: {e['type']}
      type_description_hash: ''"""
        for e in topic_entries
    )

    yaml_text = f"""\
rosbag2_bagfile_information:
  compression_format: ''
  compression_mode: ''
  custom_data: null
  duration:
    nanoseconds: {total_duration_ns}
  files:
  - duration:
      nanoseconds: {total_duration_ns}
    message_count: {total_msgs}
    path: {ep_num:04d}_0.mcap
    starting_time:
      nanoseconds_since_epoch: 0
  message_count: {total_msgs}
  relative_file_paths:
  - {ep_num:04d}_0.mcap
  ros_distro: jazzy
  starting_time:
    nanoseconds_since_epoch: 0
  storage_identifier: mcap
  topics_with_message_count:
{topics_yaml}
  version: 9
"""
    (ep_dir / "metadata.yaml").write_text(yaml_text)

    print(f"  Written {mcap_path.relative_to(REPO)} ({total_msgs} msgs)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Generating EE fixture MCAPs → {OUT_DIR.relative_to(REPO)}")
    for ep_idx in range(N_EPISODES):
        write_episode(ep_idx, OUT_DIR)
    print(f"Done — {N_EPISODES} episodes written.")


if __name__ == "__main__":
    main()
