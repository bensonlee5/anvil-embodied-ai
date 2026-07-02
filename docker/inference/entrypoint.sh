#!/bin/bash
set -e

# Source ROS2
source /opt/ros/jazzy/setup.bash
source /workspace/install/setup.bash

# --- CycloneDDS / distributed discovery ---
# If ROS_DOMAIN_ID is set and non-empty, configure for distributed mode.
# Otherwise, fall back to localhost-only communication.
if [ -n "${ROS_DOMAIN_ID}" ]; then
    export ROS_DOMAIN_ID
    echo "[entrypoint] ROS_DOMAIN_ID=${ROS_DOMAIN_ID} — distributed mode"
else
    export ROS_LOCALHOST_ONLY=1
    echo "[entrypoint] ROS_DOMAIN_ID not set — localhost-only mode"
fi

# Configure RMW middleware
if [ -n "${RMW_IMPLEMENTATION}" ]; then
    export RMW_IMPLEMENTATION
    echo "[entrypoint] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
else
    unset RMW_IMPLEMENTATION
    echo "[entrypoint] RMW_IMPLEMENTATION not set — using Fast DDS (default)"
fi

if [ -n "${CYCLONEDDS_URI}" ]; then
    export CYCLONEDDS_URI
    echo "[entrypoint] CYCLONEDDS_URI=${CYCLONEDDS_URI}"
else
    unset CYCLONEDDS_URI
fi

# When DEBUG=true, auto-configure debug image capture
DEBUG_IMAGE_ARG=""
if [ "${DEBUG:-false}" = "true" ]; then
    mkdir -p /workspace/debug_images
    DEBUG_IMAGE_ARG="debug_image_dir:=/workspace/debug_images"
    echo "[entrypoint] DEBUG=true — saving pre-model images to /workspace/debug_images"
fi

# When MONITOR_ENABLE=true, auto-configure full-episode camera video recording
MONITOR_VIDEO_ARG=""
if [ "${MONITOR_ENABLE:-false}" = "true" ]; then
    mkdir -p /workspace/monitor_output/videos
    MONITOR_VIDEO_ARG="monitor_video_dir:=/workspace/monitor_output/videos"
    echo "[entrypoint] MONITOR_ENABLE=true — recording cameras to /workspace/monitor_output/videos"
fi

exec "$@" ${DEBUG_IMAGE_ARG} ${MONITOR_VIDEO_ARG}
