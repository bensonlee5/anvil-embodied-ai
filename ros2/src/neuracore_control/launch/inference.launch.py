"""Launch file for Neuracore local inference node."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_file_arg = DeclareLaunchArgument(
        "model_file",
        default_value=os.environ.get("NEURACORE_MODEL_FILE", ""),
        description=(
            "Path to local .nc.zip (takes precedence over train_run_name). "
            "Defaults to $NEURACORE_MODEL_FILE."
        ),
    )
    train_run_name_arg = DeclareLaunchArgument(
        "train_run_name",
        default_value=os.environ.get("NEURACORE_TRAIN_RUN_NAME", ""),
        description=(
            "Neuracore training run name (downloaded on first use). "
            "Defaults to $NEURACORE_TRAIN_RUN_NAME."
        ),
    )
    robot_name_arg = DeclareLaunchArgument(
        "robot_name",
        default_value=os.environ.get("NEURACORE_ROBOT_NAME", "anvil_openarm"),
        description=(
            "Neuracore robot name — must match the trained embodiment. "
            "Defaults to $NEURACORE_ROBOT_NAME or 'anvil_openarm'."
        ),
    )
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value="",
        description="Optional URDF path passed to nc.connect_robot",
    )
    inference_rate_arg = DeclareLaunchArgument(
        "inference_rate_hz",
        default_value=os.environ.get("NEURACORE_INFERENCE_RATE_HZ", "50.0"),
        description=(
            "Control loop rate (Hz). Should match the training run's "
            "synchronization_details.frequency so each predicted action chunk "
            "plays out at the cadence it was trained on. "
            "Defaults to $NEURACORE_INFERENCE_RATE_HZ or 50.0."
        ),
    )
    debug_arg = DeclareLaunchArgument(
        "debug",
        default_value="false",
        description="If true, log predicted actions instead of publishing commands",
    )
    max_joint_delta_arg = DeclareLaunchArgument(
        "max_joint_delta",
        default_value=os.environ.get("NEURACORE_MAX_JOINT_DELTA", "0.05"),
        description=(
            "Per-tick arm joint target is clamped to current ± this "
            "(radians). At 50 Hz: 0.05 ≈ 2.5 rad/s, 0.2 ≈ 10 rad/s, "
            "999 effectively disables. Defaults to $NEURACORE_MAX_JOINT_DELTA "
            "or 0.05."
        ),
    )
    device_arg = DeclareLaunchArgument(
        "device",
        default_value=os.environ.get("NEURACORE_DEVICE", "cuda"),
        description=(
            "Torch device for inference (cuda, cuda:0, cpu, mps). "
            "Defaults to $NEURACORE_DEVICE or 'cuda'."
        ),
    )
    predictions_log_arg = DeclareLaunchArgument(
        "predictions_log",
        default_value="",
        description="If set, write per-tick predictions to this CSV path (raw policy targets + current state)",
    )
    image_log_chunks_arg = DeclareLaunchArgument(
        "image_log_chunks",
        default_value=os.environ.get("NEURACORE_IMAGE_LOG_CHUNKS", "10"),
        description=(
            "Save the camera frames that fed predict() for the first N chunks "
            "to disk (one folder per chunk). 0 disables. Defaults to "
            "$NEURACORE_IMAGE_LOG_CHUNKS or 10."
        ),
    )
    image_log_dir_arg = DeclareLaunchArgument(
        "image_log_dir",
        default_value=os.environ.get("NEURACORE_IMAGE_LOG_DIR", ""),
        description=(
            "Directory to write per-chunk image folders into. If empty and "
            "predictions_log is set, derives <predictions_log without .csv>_images/."
        ),
    )

    inference_node = Node(
        package="neuracore_control",
        executable="inference_node",
        name="neuracore_inference",
        output="screen",
        parameters=[
            {
                "model_file": LaunchConfiguration("model_file"),
                "train_run_name": LaunchConfiguration("train_run_name"),
                "robot_name": LaunchConfiguration("robot_name"),
                "urdf_path": LaunchConfiguration("urdf_path"),
                "inference_rate_hz": LaunchConfiguration("inference_rate_hz"),
                "debug": LaunchConfiguration("debug"),
                "max_joint_delta": LaunchConfiguration("max_joint_delta"),
                "predictions_log": LaunchConfiguration("predictions_log"),
                "device": LaunchConfiguration("device"),
                "image_log_chunks": LaunchConfiguration("image_log_chunks"),
                "image_log_dir": LaunchConfiguration("image_log_dir"),
            }
        ],
    )

    return LaunchDescription(
        [
            model_file_arg,
            train_run_name_arg,
            robot_name_arg,
            urdf_path_arg,
            inference_rate_arg,
            debug_arg,
            max_joint_delta_arg,
            predictions_log_arg,
            device_arg,
            image_log_chunks_arg,
            image_log_dir_arg,
            inference_node,
        ]
    )
