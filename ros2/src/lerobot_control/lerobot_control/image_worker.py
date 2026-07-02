"""
Image Worker Process for Multi-Process Inference

Each image worker runs in a separate process, subscribing to a single camera topic,
decompressing JPEG images, and writing to shared memory. This eliminates GIL
contention with the main inference process.
"""

import signal
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from .shared_image_buffer import SharedImageBuffer


class ImageWorkerNode(Node):
    """
    ROS2 node that subscribes to a single camera and writes to shared memory.

    Runs in its own process for true parallelism (no GIL).
    """

    def __init__(
        self,
        camera_topic: str,
        camera_name: str,
        image_shape: tuple[int, int, int],
        buffer_name_prefix: str = "lerobot_img_",
        debug_dir: str | None = None,
        debug_max_frames: int = 10,
        video_dir: str | None = None,
    ):
        super().__init__(f"image_worker_{camera_name}")

        self.camera_name = camera_name
        self.camera_topic = camera_topic
        self.image_shape = image_shape

        self._debug_dir = Path(debug_dir) / camera_name if debug_dir else None
        self._debug_max_frames = debug_max_frames
        self._debug_saved = 0
        self._debug_last_save: float = 0.0

        # Full-episode recording (unlike the capped debug PNGs above) — enabled
        # alongside monitor_enable so a divergence anywhere in a long rollout can be
        # reviewed visually, not just the first `debug_max_frames` seconds.
        #
        # Deliberately NOT a cv2.VideoWriter: an mp4 container's index (moov atom)
        # is only written on a clean release(), so any abrupt process kill (SIGKILL,
        # or a slow shutdown racing a container's stop_grace_period) produces a
        # corrupted, unplayable file — this bit us in practice even after fixing
        # SIGTERM handling. Instead, dump one JPEG per frame (each write is complete
        # and independent; a kill mid-episode just means the tail frames are
        # missing, never corruption) and let run_inference.sh batch-convert the
        # sequence to mp4 with ffmpeg after the container has fully exited.
        self._frames_dir: Path | None = Path(video_dir) / f"{camera_name}_frames" if video_dir else None
        self._frame_index = 0
        if self._frames_dir is not None:
            self._frames_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"[video] Recording {camera_name} frames -> {self._frames_dir}")

        # Connect to shared memory (created by main process)
        self.shared_buffer = SharedImageBuffer(
            camera_names=[camera_name],
            image_shape=image_shape,
            create=False,
            buffer_name_prefix=buffer_name_prefix,
        )

        # Statistics
        self.frame_count = 0

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            CompressedImage, camera_topic, self._image_callback, qos_profile_sensor_data
        )

        self.get_logger().info(f"Image worker started: {camera_topic} -> {camera_name}")

    def _image_callback(self, msg: CompressedImage):
        """Process incoming compressed image."""
        try:
            # Decompress JPEG (CPU-intensive, but no GIL contention in separate process)
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                self.get_logger().warn(f"Failed to decode image from {self.camera_name}")
                return

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize with padding to preserve aspect ratio
            if image.shape[:2] != self.image_shape[:2]:
                target_h, target_w = self.image_shape[:2]
                src_h, src_w = image.shape[:2]
                scale = min(target_w / src_w, target_h / src_h)
                new_w = int(src_w * scale)
                new_h = int(src_h * scale)
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                offset_x = (target_w - new_w) // 2
                offset_y = (target_h - new_h) // 2
                canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
                image = canvas

            # Save debug frames at 1 Hz up to debug_max_frames (before model input, uint8 RGB)
            if self._debug_dir is not None and self._debug_saved < self._debug_max_frames:
                now = time.time()
                if now - self._debug_last_save >= 1.0:
                    self._debug_dir.mkdir(parents=True, exist_ok=True)
                    fname = self._debug_dir / f"frame_{self._debug_saved:04d}.png"
                    cv2.imwrite(str(fname), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    self._debug_last_save = now
                    self._debug_saved += 1
                    if self._debug_saved == self._debug_max_frames:
                        self.get_logger().info(
                            f"[debug] Saved {self._debug_max_frames} frames to {self._debug_dir}"
                        )

            # Write full-episode frame (same post-resize/pad image as model input).
            # Each imwrite() is a complete, independent file — no cross-frame state
            # that a mid-episode kill could corrupt (see __init__ docstring note).
            if self._frames_dir is not None:
                fname = self._frames_dir / f"frame_{self._frame_index:06d}.jpg"
                cv2.imwrite(str(fname), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                self._frame_index += 1

            # Get timestamp from message
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # Write to shared memory
            self.shared_buffer.write(self.camera_name, image, timestamp)

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def destroy_node(self):
        """Cleanup."""
        if self._frames_dir is not None:
            self.get_logger().info(
                f"[video] Saved {self._frame_index} frames to {self._frames_dir}"
            )
        self.shared_buffer.close()
        super().destroy_node()


def run_image_worker(
    camera_topic: str,
    camera_name: str,
    image_shape: tuple[int, int, int],
    buffer_name_prefix: str = "lerobot_img_",
    stop_event=None,
    debug_dir: str | None = None,
    debug_max_frames: int = 10,
    video_dir: str | None = None,
):
    """
    Entry point for running image worker in a separate process.

    Args:
        camera_topic: ROS2 topic to subscribe to
        camera_name: Name of the camera (e.g., 'waist')
        image_shape: Shape of images (H, W, C)
        buffer_name_prefix: Prefix for shared memory names
        stop_event: Optional multiprocessing.Event to signal shutdown
        debug_dir: If set, save the first debug_max_frames frames as PNGs (1 Hz)
        video_dir: If set, dump every frame as a JPEG under
            <video_dir>/<camera_name>_frames/ for run_inference.sh to batch-convert
            to <video_dir>/<camera_name>.mp4 with ffmpeg after shutdown.
    """
    rclpy.init(args=[])  # Empty args to avoid inheriting parent's --ros-args node name

    node = ImageWorkerNode(
        camera_topic=camera_topic,
        camera_name=camera_name,
        image_shape=image_shape,
        buffer_name_prefix=buffer_name_prefix,
        debug_dir=debug_dir,
        video_dir=video_dir,
        debug_max_frames=debug_max_frames,
    )

    # Defense in depth: if the parent's graceful stop_event handshake times out,
    # MultiProcessStrategy.cleanup() escalates to Process.terminate() (bare
    # SIGTERM). Call rclpy.shutdown() directly (matches inference_node.py /
    # inference_monitor_node.py) rather than raising KeyboardInterrupt — both
    # spin loops below check context validity every iteration, so invalidating
    # the context is enough to make them exit cleanly and reach the video
    # writer release in destroy_node().
    def _sigterm_handler(signum, frame):
        node.get_logger().info(f"[video] SIGTERM received for {camera_name}, shutting down...")
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        if stop_event is not None:
            # Spin with stop check
            while not stop_event.is_set() and rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
        else:
            # Spin forever
            executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class JointStateWorkerNode(Node):
    """
    ROS2 node that subscribes to joint states and writes to shared memory.

    Joint state processing is lightweight, but we run it in a worker for consistency.
    """

    def __init__(
        self, joint_topic: str, joint_names: list, buffer_name: str = "lerobot_joint_state"
    ):
        super().__init__("joint_state_worker")

        from sensor_msgs.msg import JointState

        from .shared_image_buffer import SharedJointStateBuffer

        self.joint_names = joint_names
        self.num_joints = len(joint_names)

        # Connect to shared memory
        self.shared_buffer = SharedJointStateBuffer(
            num_joints=self.num_joints, create=False, buffer_name=buffer_name
        )

        # Subscribe to joint states
        self.subscription = self.create_subscription(
            JointState, joint_topic, self._joint_callback, 10
        )

        self.frame_count = 0
        self.start_time = None

        self.get_logger().info(f"Joint state worker started: {joint_topic}")

    def _joint_callback(self, msg):
        """Process incoming joint state."""
        if self.start_time is None:
            self.start_time = time.time()

        try:
            # Extract positions in order
            positions = np.zeros(self.num_joints, dtype=np.float64)
            msg_names = list(msg.name)
            msg_positions = list(msg.position)

            for i, name in enumerate(self.joint_names):
                if name in msg_names:
                    idx = msg_names.index(name)
                    positions[i] = msg_positions[idx]

            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.shared_buffer.write(positions, timestamp)
            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing joint state: {e}")

    def destroy_node(self):
        self.shared_buffer.close()
        super().destroy_node()


def run_joint_state_worker(
    joint_topic: str, joint_names: list, buffer_name: str = "lerobot_joint_state", stop_event=None
):
    """Entry point for joint state worker process."""
    rclpy.init(args=[])  # Empty args to avoid inheriting parent's --ros-args node name

    node = JointStateWorkerNode(
        joint_topic=joint_topic, joint_names=joint_names, buffer_name=buffer_name
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        if stop_event is not None:
            while not stop_event.is_set() and rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
        else:
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
