#!/usr/bin/env python3

"""ROS2 node: local Neuracore policy inference → arm position commands.

Designed to run on a GPU PC, discover /joint_states + camera topics from the
Robot PC over CycloneDDS, and publish 8-element Float64MultiArray (7 arm +
1 gripper) commands back to the Robot PC's forward_position_controllers.

Mirrors the logging schema used by anvil-workcell's neuracore_bridge data
collector: joint names, gripper normalization (0..0.05m → [0, 1]), camera
naming ('cam_wrist_l', etc.) all match, so a model trained from that
collector's data runs here unchanged.
"""

import csv
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import neuracore as nc
import numpy as np
import rclpy
import torch
from neuracore_types import DataType
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64MultiArray

from .common import (
    CMD_L_TOPIC,
    DEFAULT_CAMERA_TOPICS,
    GRIPPER_HI,
    LEFT_ARM,
    LEFT_GRIPPER,
    camera_name_from_topic,
    decode_compressed_image,
    gripper_denormalize,
    gripper_normalize,
    header_time,
)


class NeuracoreInferenceNode(Node):
    """Runs a Neuracore policy locally and publishes arm commands."""

    def __init__(self):
        super().__init__("neuracore_inference_node")
        self.get_logger().info("[neura-infer] startup")

        self.declare_parameter("robot_name", "anvil_openarm")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("model_file", "")
        self.declare_parameter("train_run_name", "")
        self.declare_parameter("camera_topics", DEFAULT_CAMERA_TOPICS)
        self.declare_parameter("inference_rate_hz", 50.0)
        self.declare_parameter("debug", False)
        self.declare_parameter("max_joint_delta", 0.05)
        self.declare_parameter("predictions_log", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("image_log_chunks", 10)
        self.declare_parameter("image_log_dir", "")

        self._debug = (
            self.get_parameter("debug").get_parameter_value().bool_value
        )
        self._max_joint_delta = float(
            self.get_parameter("max_joint_delta")
            .get_parameter_value()
            .double_value
        )
        self._device = (
            self.get_parameter("device").get_parameter_value().string_value
        ) or "cuda"

        self._predictions_file = None
        self._predictions_writer = None
        log_path = (
            self.get_parameter("predictions_log")
            .get_parameter_value()
            .string_value
        )
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            self._predictions_file = open(log_path, "w", newline="")
            self._predictions_writer = csv.writer(self._predictions_file)
            header = ["t", "chunk_id", "chunk_idx"]
            header += [f"obs_{n}" for n in LEFT_ARM]
            header += ["obs_grip"]
            header += [f"out_{n}" for n in LEFT_ARM]
            header += ["out_grip"]
            self._predictions_writer.writerow(header)
            self._predictions_file.flush()
            self.get_logger().info(
                f"[neura-infer] writing predictions to {log_path}"
            )

        # Image logging — bounded to first N chunks so disk doesn't blow up.
        # Written synchronously between chunks (~tens of ms total); negligible
        # next to the multi-second predict() that just ran.
        self._image_log_chunks = int(
            self.get_parameter("image_log_chunks")
            .get_parameter_value()
            .integer_value
        )
        self._image_log_dir = (
            self.get_parameter("image_log_dir")
            .get_parameter_value()
            .string_value
        )
        if self._image_log_chunks > 0:
            if not self._image_log_dir and log_path:
                self._image_log_dir = os.path.splitext(log_path)[0] + "_images"
            if not self._image_log_dir:
                self.get_logger().warning(
                    "[neura-infer] image_log_chunks > 0 but no image_log_dir "
                    "or predictions_log set — disabling image logging"
                )
                self._image_log_chunks = 0
            else:
                os.makedirs(self._image_log_dir, exist_ok=True)
                self.get_logger().info(
                    f"[neura-infer] saving images for first "
                    f"{self._image_log_chunks} chunks to {self._image_log_dir}"
                )

        self._policy = None
        self._chunk_id = -1
        self._chunk_obs_arm: Optional[np.ndarray] = None
        self._chunk_obs_grip: float = float("nan")
        self._latest_joint_state: Optional[Tuple[float, JointState]] = None
        self._latest_frames: Dict[str, Tuple[float, np.ndarray]] = {}
        self._tick_count = 0

        self._camera_topics: List[str] = list(
            self.get_parameter("camera_topics")
            .get_parameter_value()
            .string_array_value
        )
        self._camera_names: List[str] = [
            camera_name_from_topic(t) for t in self._camera_topics
        ]

        self._arm_joints = LEFT_ARM
        self._gripper_joints = [LEFT_GRIPPER]

        # Embodiment descriptions — MUST match the training run that produced
        # the loaded checkpoint. Source: `neuracore training inspect <name>`,
        # input_cross_embodiment_description / output_cross_embodiment_description.
        # Inputs sort by int key to set observation order; outputs use int keys
        # as absolute tensor positions in the model's prediction.
        self._input_desc = {
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
        self._output_desc = {
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

        if not self._init_neuracore():
            raise RuntimeError("[neura-infer] neuracore init failed — aborting")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, sensor_qos
        )
        self.get_logger().info("[neura-infer] subscribed: /joint_states")
        for topic in self._camera_topics:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, t=topic: self._on_camera(msg, t),
                sensor_qos,
            )
            self.get_logger().info(f"[neura-infer] subscribed: {topic}")

        self._cmd_l_pub = self.create_publisher(Float64MultiArray, CMD_L_TOPIC, 10)

        model_file = (
            self.get_parameter("model_file").get_parameter_value().string_value
        )
        train_run_name = (
            self.get_parameter("train_run_name")
            .get_parameter_value()
            .string_value
        )
        err = self._load_policy(model_file, train_run_name)
        if err is not None:
            raise RuntimeError(f"[neura-infer] {err}")

        self._rate = float(
            self.get_parameter("inference_rate_hz")
            .get_parameter_value()
            .double_value
        )
        self.get_logger().info(
            f"[neura-infer] ready; will run inference loop @ {self._rate} Hz"
        )

    # ----------------------------------------------------------------- init

    def _init_neuracore(self) -> bool:
        api_key = os.environ.get("NEURACORE_API_KEY", "")
        if not api_key:
            self.get_logger().error("[neura-infer] NEURACORE_API_KEY not set")
            return False
        try:
            nc.login(api_key=api_key)
        except Exception as e:
            self.get_logger().error(f"[neura-infer] login failed: {e}")
            return False

        robot_name = (
            self.get_parameter("robot_name").get_parameter_value().string_value
        )
        urdf_path = (
            self.get_parameter("urdf_path").get_parameter_value().string_value
        )
        self.get_logger().info(
            f"[neura-infer] connecting robot '{robot_name}' (urdf='{urdf_path}')"
        )
        try:
            if urdf_path:
                nc.connect_robot(robot_name, urdf_path=urdf_path)
            else:
                nc.connect_robot(robot_name)
        except Exception as e:
            self.get_logger().error(f"[neura-infer] connect_robot failed: {e}")
            return False
        return True

    def _load_policy(self, model_file: str, train_run_name: str) -> Optional[str]:
        if not model_file and not train_run_name:
            return "no model_file or train_run_name configured"

        if model_file:
            src = f"model_file={model_file}"
        else:
            src = f"train_run_name={train_run_name}"

        self.get_logger().info(
            f"[neura-infer] loading policy ({src}) on device={self._device}"
        )
        t0 = time.perf_counter()
        try:
            self._policy = nc.policy(
                input_embodiment_description=self._input_desc,
                output_embodiment_description=self._output_desc,
                model_file=model_file or None,
                train_run_name=train_run_name or None,
                device=self._device,
            )
        except Exception as e:
            return f"policy load failed: {e}"
        self.get_logger().info(
            f"[neura-infer] policy loaded in {time.perf_counter() - t0:.2f}s"
        )
        return None

    # ------------------------------------------------------------ callbacks

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = (header_time(msg), msg)

    def _on_camera(self, msg: CompressedImage, topic: str) -> None:
        cam_name = camera_name_from_topic(topic)
        try:
            rgb = decode_compressed_image(msg, size=(640, 480))
        except Exception as e:
            self.get_logger().warning(
                f"[neura-infer] decode {cam_name} failed: {e}",
                throttle_duration_sec=10.0,
            )
            return
        self._latest_frames[cam_name] = (header_time(msg), rgb)

    # ----------------------------------------------------------- inference

    def run_inference(self) -> None:
        """Synchronous inference loop: predict → play 100 actions @ rate → repeat.

        Runs in the main thread; ROS callbacks (joint state, cameras) run on a
        separate spin thread so they keep flowing even while predict() blocks.
        """
        period = 1.0 / self._rate
        self.get_logger().info(
            f"[neura-infer] inference loop starting (period={period*1e3:.1f} ms)"
        )

        while rclpy.ok():
            if self._policy is None:
                time.sleep(0.1)
                continue

            if self._latest_joint_state is None:
                self.get_logger().warning(
                    "[neura-infer] waiting for /joint_states",
                    throttle_duration_sec=5.0,
                )
                time.sleep(0.1)
                continue
            missing = [n for n in self._camera_names if n not in self._latest_frames]
            if missing:
                self.get_logger().warning(
                    f"[neura-infer] waiting for cameras: {missing}",
                    throttle_duration_sec=5.0,
                )
                time.sleep(0.1)
                continue

            try:
                self._log_observations()
            except Exception as e:
                self.get_logger().warning(
                    f"[neura-infer] log observations failed: {e}",
                    throttle_duration_sec=5.0,
                )
                time.sleep(0.1)
                continue

            self._chunk_obs_arm = self._current_arm_positions()
            grip_raw = self._current_gripper_position()
            self._chunk_obs_grip = (
                gripper_normalize(grip_raw)
                if not np.isnan(grip_raw)
                else float("nan")
            )
            self._chunk_id += 1

            # Save the camera frames that fed this predict (bounded; sync).
            if (
                self._image_log_chunks > 0
                and self._chunk_id < self._image_log_chunks
            ):
                self._write_chunk_images(
                    self._chunk_id, dict(self._latest_frames)
                )

            try:
                chunk = self._predict_chunk()
            except Exception as e:
                self.get_logger().warning(
                    f"[neura-infer] predict failed: {e}",
                    throttle_duration_sec=5.0,
                )
                time.sleep(0.1)
                continue

            next_t = time.monotonic()
            for chunk_idx in range(len(chunk)):
                if not rclpy.ok():
                    return
                self._publish_commands(chunk[chunk_idx], chunk_idx)
                self._tick_count += 1

                next_t += period
                sleep_for = next_t - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_t = time.monotonic()  # we're behind; resync clock

            self.get_logger().info(
                f"[neura-infer] chunk {self._chunk_id} done "
                f"({len(chunk)} actions played, {self._tick_count} total ticks)"
            )

    def _log_observations(self) -> None:
        js_snapshot = self._latest_joint_state
        frames_snapshot = dict(self._latest_frames)
        if js_snapshot is None:
            return
        js_t, js = js_snapshot
        positions: Dict[str, float] = {}
        grippers: Dict[str, float] = {}
        for i, name in enumerate(js.name):
            if i >= len(js.position):
                continue
            if name in self._gripper_joints:
                grippers[name] = float(js.position[i])
            elif name in self._arm_joints:
                positions[name] = float(js.position[i])

        if positions:
            nc.log_joint_positions(positions, timestamp=js_t)
        for name, raw in grippers.items():
            nc.log_parallel_gripper_open_amount(
                name, gripper_normalize(raw), timestamp=js_t
            )
        for cam_name, (t, rgb) in frames_snapshot.items():
            nc.log_rgb(cam_name, rgb, timestamp=t)

    def _predict_chunk(self) -> np.ndarray:
        """Run policy and return (horizon, n_arm + n_grippers) array."""
        t0 = time.perf_counter()
        preds = self._policy.predict(timeout=5)
        predict_ms = (time.perf_counter() - t0) * 1e3

        joint_preds = preds[DataType.JOINT_TARGET_POSITIONS]
        arm_tensors = [joint_preds[n].value for n in self._arm_joints]
        arm = torch.cat(arm_tensors, dim=2)[0]  # (horizon, n_arm)

        grip_preds = preds.get(DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS, {})
        if grip_preds:
            grip_tensors = [grip_preds[n].open_amount for n in self._gripper_joints]
            grip = torch.cat(grip_tensors, dim=2)[0]
            out = torch.cat([arm, grip], dim=1)
        else:
            out = arm

        chunk = out.detach().cpu().numpy().astype(np.float64)
        self.get_logger().info(
            f"[neura-infer] predict ok: {predict_ms:.1f}ms, horizon={chunk.shape[0]}"
        )
        return chunk

    def _current_arm_positions(self) -> Optional[np.ndarray]:
        if self._latest_joint_state is None:
            return None
        _, js = self._latest_joint_state
        lookup = {n: p for n, p in zip(js.name, js.position)}
        try:
            return np.array([lookup[n] for n in self._arm_joints], dtype=np.float64)
        except KeyError:
            return None

    def _current_gripper_position(self) -> float:
        if self._latest_joint_state is None or not self._gripper_joints:
            return float("nan")
        _, js = self._latest_joint_state
        for name, pos in zip(js.name, js.position):
            if name == self._gripper_joints[0]:
                return float(pos)
        return float("nan")

    def _publish_commands(self, action: np.ndarray, chunk_idx: int) -> None:
        n_arm = len(self._arm_joints)

        # Raw policy output (before any clamp / denormalization).
        target_arm = action[:n_arm]
        if action.shape[0] >= n_arm + 1:
            left_grip_norm = float(action[n_arm])
            left_grip_raw = gripper_denormalize(left_grip_norm)
        else:
            left_grip_norm = float("nan")
            left_grip_raw = GRIPPER_HI

        # CSV: always the raw action, regardless of clamp / debug.
        if self._predictions_writer is not None and self._predictions_file is not None:
            obs_arm_row = (
                self._chunk_obs_arm.tolist()
                if self._chunk_obs_arm is not None
                else [float("nan")] * n_arm
            )
            row = [time.time(), self._chunk_id, chunk_idx]
            row += obs_arm_row
            row += [self._chunk_obs_grip]
            row += target_arm.tolist()
            row += [left_grip_norm]
            self._predictions_writer.writerow(row)
            self._predictions_file.flush()

        # Apply delta-clamp on top of the raw action to get the safe command.
        current_arm = self._current_arm_positions()
        if current_arm is not None and len(current_arm) == n_arm:
            raw_delta = target_arm - current_arm
            clamped = int(np.sum(np.abs(raw_delta) > self._max_joint_delta))
            limited = np.clip(
                raw_delta, -self._max_joint_delta, self._max_joint_delta
            )
            safe_arm = (current_arm + limited).tolist()
            if clamped:
                self.get_logger().warning(
                    f"[neura-infer] delta-clamped {clamped}/{n_arm} arm joints "
                    f"to ±{self._max_joint_delta:.3f} rad",
                    throttle_duration_sec=2.0,
                )
        else:
            safe_arm = target_arm.tolist()

        if self._debug:
            self.get_logger().info(
                f"[neura-infer] DEBUG raw[{chunk_idx}] "
                f"L_arm={['%.3f' % v for v in target_arm.tolist()]} "
                f"L_grip={left_grip_norm:.3f} "
                f"-> clamped L_arm={['%.3f' % v for v in safe_arm]}"
            )
            return  # debug: do everything except publish to the controller

        msg_l = Float64MultiArray()
        msg_l.data = safe_arm + [left_grip_raw]
        self._cmd_l_pub.publish(msg_l)

    # ------------------------------------------------------------- image log

    def _write_chunk_images(
        self, chunk_id: int, frames: Dict[str, Tuple[float, np.ndarray]]
    ) -> None:
        """Save the camera frames that fed predict() for this chunk."""
        chunk_dir = os.path.join(self._image_log_dir, f"chunk_{chunk_id:04d}")
        try:
            os.makedirs(chunk_dir, exist_ok=True)
            for cam_name, (_, rgb) in frames.items():
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(chunk_dir, f"{cam_name}.jpg"), bgr)
            self.get_logger().info(
                f"[neura-infer] wrote {len(frames)} images for "
                f"chunk {chunk_id} -> {chunk_dir}"
            )
        except Exception as e:
            self.get_logger().warning(
                f"[neura-infer] image write for chunk {chunk_id} failed: {e}"
            )

    # ------------------------------------------------------------ shutdown

    def destroy_node(self) -> None:
        self.get_logger().info("[neura-infer] destroy_node")
        if self._policy is not None:
            try:
                self._policy.disconnect()
            except Exception as e:
                self.get_logger().warning(f"[neura-infer] disconnect failed: {e}")
        if self._predictions_file is not None:
            self._predictions_file.close()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = NeuracoreInferenceNode()

    # ROS callbacks (joint state, cameras) on a daemon thread so they keep
    # firing while predict() blocks the main thread.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run_inference()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
