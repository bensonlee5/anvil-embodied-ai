"""Shared constants and helpers for neuracore_control.

Copied verbatim from anvil-workcell/ros2/src/neuracore_bridge/neuracore_bridge/
common.py so joint naming + gripper normalization stays consistent between
the data-collection side (on the robot PC) and the inference side (here).
If you edit one, edit the other.
"""

import time

import cv2
import numpy as np
from neuracore_types import DataType

LEFT_ARM = [f"follower_l_joint{i}" for i in range(1, 8)]
RIGHT_ARM = [f"follower_r_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER = "follower_l_finger_joint1"
RIGHT_GRIPPER = "follower_r_finger_joint1"
FINGER_JOINTS = {LEFT_GRIPPER, RIGHT_GRIPPER}
CAMERAS = ["cam_wrist_l", "cam_waist", "cam_chest"]

# URDF hard limits for finger_joint1 (prismatic, meters).
GRIPPER_LO = 0.0
GRIPPER_HI = 0.05

CMD_L_TOPIC = "/follower_l_forward_position_controller/commands"
CMD_R_TOPIC = "/follower_r_forward_position_controller/commands"

DEFAULT_CAMERA_TOPICS = [
    "/cam_wrist_l/image_raw/compressed",
    "/cam_waist/image_raw/compressed",
    "/cam_chest/image_raw/compressed",
]


def header_time(msg) -> float:
    """Publisher stamp if set, else wall clock."""
    try:
        s, n = msg.header.stamp.sec, msg.header.stamp.nanosec
        if s or n:
            return s + n * 1e-9
    except AttributeError:
        pass
    return time.time()


def gripper_normalize(value: float) -> float:
    """Raw finger_joint1 position (m) → [0, 1]."""
    return max(0.0, min(1.0, (value - GRIPPER_LO) / (GRIPPER_HI - GRIPPER_LO)))


def gripper_denormalize(value: float) -> float:
    """[0, 1] → raw finger_joint1 position (m), clamped to URDF limits."""
    clamped = max(0.0, min(1.0, value))
    return clamped * (GRIPPER_HI - GRIPPER_LO) + GRIPPER_LO


def decode_compressed_image(
    msg, size: tuple[int, int] = (640, 480)
) -> np.ndarray:
    """CompressedImage (MJPEG/JPEG) → uint8 HxWx3 RGB, resized to (W, H)."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None")
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def camera_name_from_topic(topic: str) -> str:
    """'/cam_wrist_l/image_raw/compressed' → 'cam_wrist_l'."""
    return topic.lstrip("/").split("/")[0]


# Per-model embodiment descriptions — MUST match the training run that
# produced the loaded checkpoint. Source for each entry: the
# input_cross_embodiment_description / output_cross_embodiment_description
# fields from `neuracore training inspect <name>` (the kit-block-dp*.txt
# snapshots in this package). Inputs sort by int key to set observation
# order; outputs use int keys as absolute tensor positions in the model's
# prediction. Shared between the live ROS2 inference node and the offline
# analysis script.
#
# To add a new trained run: copy its inspect output into a <name>.txt file,
# transcribe the in/out descriptions into a new MODEL_EMBODIMENTS entry,
# and pass that name as train_run_name to the inference node / scripts.


MODEL_EMBODIMENTS: dict[str, dict] = {
    "kit-block-dp": {
        "input": {
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
                0: LEFT_GRIPPER,
            },
            DataType.RGB_IMAGES: {
                0: "cam_wrist_l",
                1: "cam_waist",
                2: "cam_chest",
            },
        },
        "output": {
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
                0: LEFT_GRIPPER,
            },
        },
    },
    "kit-block-dp-attempt2": {
        "input": {
            DataType.JOINT_POSITIONS: {
                0: "follower_l_joint1",
                1: "follower_l_joint2",
                2: "follower_l_joint3",
                3: "follower_l_joint4",
                4: "follower_l_joint5",
                5: "follower_l_joint6",
                6: "follower_l_joint7",
            },
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {
                0: LEFT_GRIPPER,
            },
            DataType.RGB_IMAGES: {
                0: "cam_waist",
                1: "cam_chest",
                2: "cam_wrist_l",
            },
        },
        "output": {
            DataType.JOINT_TARGET_POSITIONS: {
                0: "follower_l_joint1",
                1: "follower_l_joint2",
                2: "follower_l_joint3",
                3: "follower_l_joint4",
                4: "follower_l_joint5",
                5: "follower_l_joint6",
                6: "follower_l_joint7",
            },
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: {
                0: LEFT_GRIPPER,
            },
        },
    },
}

DEFAULT_TRAIN_RUN_NAME = "kit-block-dp"


def get_model_embodiment(train_run_name: str) -> dict:
    """Return {'input': ..., 'output': ...} descriptions for a known training run."""
    try:
        return MODEL_EMBODIMENTS[train_run_name]
    except KeyError as e:
        known = ", ".join(sorted(MODEL_EMBODIMENTS))
        raise ValueError(
            f"Unknown train_run_name {train_run_name!r}. Known: {known}. "
            f"Add a MODEL_EMBODIMENTS entry in {__file__} after running "
            f"`neuracore training inspect <name>`."
        ) from e
