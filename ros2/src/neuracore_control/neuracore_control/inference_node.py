#!/usr/bin/env python3

"""ROS2 node: local Neuracore policy inference → arm position commands."""

import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path

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
    DEFAULT_TRAIN_RUN_NAME,
    LEFT_ARM,
    LEFT_GRIPPER,
    camera_name_from_topic,
    decode_compressed_image,
    get_model_embodiments,
    gripper_denormalize,
    gripper_normalize,
    header_time,
    stitch_frames,
)

DEFAULT_LOG_ROOT = Path(__file__).resolve().parents[1] / "logs"


class NeuracoreInferenceNode(Node):
    """Runs a Neuracore policy locally and publishes arm commands."""

    def __init__(self):
        super().__init__("neuracore_inference_node")
        self.get_logger().info("[neura-infer] startup")

        self.declare_parameter("robot_name", "anvil_openarm")
        self.declare_parameter("model_file", "")
        self.declare_parameter("train_run_name", "")
        self.declare_parameter("camera_topics", DEFAULT_CAMERA_TOPICS)
        self.declare_parameter("inference_rate_hz", 50.0)
        self.declare_parameter("debug", False)
        self.declare_parameter("max_joint_delta", 0.05)
        self.declare_parameter("predictions_log", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("image_log_chunks", 10)
        self.declare_parameter("image_log_individual", True)
        self.declare_parameter("image_log_stitched", True)

        self._debug = self.get_parameter("debug").get_parameter_value().bool_value
        self._max_joint_delta = float(
            self.get_parameter("max_joint_delta").get_parameter_value().double_value
        )
        self._device = (self.get_parameter("device").get_parameter_value().string_value) or "cuda"

        self._predictions_file = None
        self._predictions_writer = None
        log_param = self.get_parameter("predictions_log").get_parameter_value().string_value
        log_path = (
            Path(log_param)
            if log_param
            else (DEFAULT_LOG_ROOT / f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv")
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._predictions_file = log_path.open("w", newline="")
        self._predictions_writer = csv.writer(self._predictions_file)
        header = ["t", "chunk_id", "chunk_idx"]
        header += LEFT_ARM
        header += [LEFT_GRIPPER]
        header += [f"pred_target_{n}" for n in LEFT_ARM]
        header += [f"pred_target_{LEFT_GRIPPER}"]
        header += [f"current_{n}" for n in LEFT_ARM]
        header += [f"current_{LEFT_GRIPPER}"]
        header += [f"cmd_{n}" for n in LEFT_ARM]
        header += [f"cmd_{LEFT_GRIPPER}"]
        self._predictions_writer.writerow(header)
        self._predictions_file.flush()
        self.get_logger().info(f"[neura-infer] writing predictions to {log_path}")

        # Image logging — bounded to first N chunks so disk doesn't blow up.
        # Written synchronously between chunks (~tens of ms total); negligible
        # next to the multi-second predict() that just ran
        self._image_log_chunks = int(
            self.get_parameter("image_log_chunks").get_parameter_value().integer_value
        )
        self._image_log_individual = bool(
            self.get_parameter("image_log_individual").get_parameter_value().bool_value
        )
        self._image_log_stitched = bool(
            self.get_parameter("image_log_stitched").get_parameter_value().bool_value
        )
        self._image_log_dir = log_path.parent / "images"
        self._image_log_prefix = log_path.stem
        if self._image_log_chunks > 0 and (self._image_log_individual or self._image_log_stitched):
            self._image_log_dir.mkdir(parents=True, exist_ok=True)
            modes = []
            if self._image_log_individual:
                modes.append("per-cam")
            if self._image_log_stitched:
                modes.append("stitched")
            self.get_logger().info(
                f"[neura-infer] saving images ({'+'.join(modes)}) for first "
                f"{self._image_log_chunks} chunks to {self._image_log_dir}/"
                f"{self._image_log_prefix}_chunk_*.jpg"
            )

        self._policy = None
        self._chunk_id = -1
        self._obs_arm: np.ndarray | None = None
        self._obs_grip: float = float("nan")
        self._latest_joint_state_msg: tuple[float, JointState] | None = None
        self._latest_cam_msgs: dict[str, tuple[float, CompressedImage]] = {}
        self._tick_count = 0

        self._camera_topics: list[str] = list(
            self.get_parameter("camera_topics").get_parameter_value().string_array_value
        )
        self._camera_names: list[str] = [camera_name_from_topic(t) for t in self._camera_topics]

        self._arm_joint_namess = LEFT_ARM
        self._gripper_joint_names = [LEFT_GRIPPER]

        if not self._init_neuracore():
            raise RuntimeError("[neura-infer] neuracore init failed — aborting")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, sensor_qos)
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

        model_file = self.get_parameter("model_file").get_parameter_value().string_value
        train_run_name = self.get_parameter("train_run_name").get_parameter_value().string_value
        self._load_policy(model_file, train_run_name)

        self._rate = float(
            self.get_parameter("inference_rate_hz").get_parameter_value().double_value
        )
        self.get_logger().info(f"[neura-infer] ready; will run inference loop @ {self._rate} Hz")

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

        robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self.get_logger().info(f"[neura-infer] connecting robot '{robot_name}'")
        try:
            nc.connect_robot(robot_name)
        except Exception as e:
            self.get_logger().error(f"[neura-infer] connect_robot failed: {e}")
            return False
        return True

    def _load_policy(self, model_file: str, train_run_name: str) -> None:
        if not model_file and not train_run_name:
            raise ValueError("no model_file or train_run_name configured")

        descriptions_run = train_run_name or DEFAULT_TRAIN_RUN_NAME
        if model_file and not train_run_name:
            self.get_logger().warning(
                f"[neura-infer] train_run_name not set; using "
                f"'{DEFAULT_TRAIN_RUN_NAME}' embodiment descriptions for the "
                f"local model_file. Set train_run_name explicitly if the "
                f"checkpoint came from a different run."
            )
        embodiments = get_model_embodiments(descriptions_run)

        src = f"model_file={model_file}" if model_file else f"train_run_name={train_run_name}"
        self.get_logger().info(
            f"[neura-infer] loading policy ({src}, descriptions={descriptions_run}) "
            f"on device={self._device}"
        )
        t0 = time.perf_counter()
        self._policy = nc.policy(
            input_embodiment_description=embodiments["input"],
            output_embodiment_description=embodiments["output"],
            model_file=model_file or None,
            train_run_name=None if model_file else train_run_name,
            device=self._device,
        )
        self.get_logger().info(f"[neura-infer] policy loaded in {time.perf_counter() - t0:.2f}s")

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state_msg = (header_time(msg), msg)

    def _on_camera(self, msg: CompressedImage, topic: str) -> None:
        self._latest_cam_msgs[camera_name_from_topic(topic)] = (header_time(msg), msg)

    def _decode_frames(
        self, cam_msgs: dict[str, tuple[float, CompressedImage]]
    ) -> dict[str, tuple[float, np.ndarray]]:
        out: dict[str, tuple[float, np.ndarray]] = {}
        for cam_name, (t, msg) in cam_msgs.items():
            try:
                out[cam_name] = (t, decode_compressed_image(msg))
            except Exception as e:
                self.get_logger().warning(
                    f"[neura-infer] decode {cam_name} failed: {e}",
                    throttle_duration_sec=10.0,
                )
        return out

    def run_inference(self) -> None:
        """Synchronous inference loop: predict → play n actions @ rate → repeat."""
        period = 1.0 / self._rate
        self.get_logger().info(
            f"[neura-infer] inference loop starting (period={period * 1e3:.1f} ms)"
        )

        while rclpy.ok():
            if self._policy is None:
                self.get_logger().info("Policy not loaded yet, sleeping")
                time.sleep(0.1)
                continue

            js_snapshot = self._latest_joint_state_msg
            cam_snapshot = dict(self._latest_cam_msgs)

            if js_snapshot is None:
                self.get_logger().warning(
                    "[neura-infer] waiting for /joint_states",
                    throttle_duration_sec=5.0,
                )
                time.sleep(0.1)
                continue
            missing = [n for n in self._camera_names if n not in cam_snapshot]
            if missing:
                self.get_logger().warning(
                    f"[neura-infer] waiting for cameras: {missing}",
                    throttle_duration_sec=5.0,
                )
                time.sleep(0.1)
                continue

            decoded_frames = self._decode_frames(cam_snapshot)
            try:
                self._log_observations(js_snapshot, decoded_frames)
            except Exception as e:
                self.get_logger().warning(
                    f"[neura-infer] log observations failed: {e}",
                    throttle_duration_sec=5.0,
                )
                time.sleep(0.1)
                continue

            self._chunk_id += 1

            self._obs_arm = self._arm_positions_from(js_snapshot)
            grip_raw = self._gripper_position_from(js_snapshot)
            self._obs_grip = gripper_normalize(grip_raw) if not np.isnan(grip_raw) else float("nan")

            # Save the camera frames that fed this predict (bounded; sync).
            if (
                self._image_log_chunks > 0
                and self._chunk_id < self._image_log_chunks
                and (self._image_log_individual or self._image_log_stitched)
            ):
                self._write_chunk_images(self._chunk_id, decoded_frames)

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
                    self.get_logger().warning(
                        f"[neura-infer] tick overrun by {-sleep_for * 1e3:.1f}ms; resyncing",
                        throttle_duration_sec=2.0,
                    )
                    next_t = time.monotonic()  # we're behind; resync clock

            self.get_logger().info(
                f"[neura-infer] chunk {self._chunk_id} done "
                f"({len(chunk)} actions played, {self._tick_count} total ticks)"
            )

    def _log_observations(
        self,
        joint_state: tuple[float, JointState],
        decoded_frames: dict[str, tuple[float, np.ndarray]],
    ) -> None:
        js_t, js = joint_state
        positions: dict[str, float] = {}
        grippers: dict[str, float] = {}
        for i, name in enumerate(js.name):
            if i >= len(js.position):
                continue
            if name in self._gripper_joint_names:
                grippers[name] = float(js.position[i])
            elif name in self._arm_joint_namess:
                positions[name] = float(js.position[i])

        if positions:
            nc.log_joint_positions(positions, timestamp=js_t)
        if grippers:
            nc.log_parallel_gripper_open_amounts(
                {name: gripper_normalize(raw) for name, raw in grippers.items()},
                timestamp=js_t,
            )
        for cam_name, (t, rgb) in decoded_frames.items():
            nc.log_rgb(cam_name, rgb, timestamp=t)

    def _predict_chunk(self) -> np.ndarray:
        t0 = time.perf_counter()
        preds = self._policy.predict(timeout=5)
        predict_ms = (time.perf_counter() - t0) * 1e3

        j = preds[DataType.JOINT_TARGET_POSITIONS]
        g = preds[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
        arm = torch.cat([j[n].value for n in self._arm_joint_namess], dim=2)[0]
        grip = torch.cat([g[n].open_amount for n in self._gripper_joint_names], dim=2)[0]
        out = torch.cat([arm, grip], dim=1)
        chunk = out.detach().cpu().numpy().astype(np.float64)
        self.get_logger().info(
            f"[neura-infer] predict ok: {predict_ms:.1f}ms, horizon={chunk.shape[0]}"
        )
        return chunk

    def _arm_positions_from(
        self, joint_state: tuple[float, JointState] | None
    ) -> np.ndarray | None:
        if joint_state is None:
            return None
        _, js = joint_state
        lookup = {n: p for n, p in zip(js.name, js.position)}
        try:
            return np.array([lookup[n] for n in self._arm_joint_namess], dtype=np.float64)
        except KeyError:
            return None

    def _gripper_position_from(self, joint_state: tuple[float, JointState] | None) -> float:
        if joint_state is None or not self._gripper_joint_names:
            return float("nan")
        _, js = joint_state
        for name, pos in zip(js.name, js.position):
            if name == self._gripper_joint_names[0]:
                return float(pos)
        return float("nan")

    def _publish_commands(self, action: np.ndarray, chunk_idx: int) -> None:
        n_arm = len(self._arm_joint_namess)
        expected = n_arm + len(self._gripper_joint_names)
        if action.shape[0] != expected:
            raise ValueError(
                f"action length {action.shape[0]} != expected {expected} "
                f"(n_arm={n_arm}, n_grip={len(self._gripper_joint_names)})"
            )

        # Raw policy output (before any clamp / denormalization).
        target_arm = action[:n_arm]
        left_grip_norm = float(action[n_arm])
        left_grip_raw = gripper_denormalize(left_grip_norm)

        # Apply delta-clamp on top of the raw action to get the safe command.
        latest_js = self._latest_joint_state_msg
        current_arm = self._arm_positions_from(latest_js)
        if current_arm is not None and len(current_arm) == n_arm:
            raw_delta = target_arm - current_arm
            clamped = int(np.sum(np.abs(raw_delta) > self._max_joint_delta))
            limited = np.clip(raw_delta, -self._max_joint_delta, self._max_joint_delta)
            safe_arm = (current_arm + limited).tolist()
            if clamped:
                self.get_logger().warning(
                    f"[neura-infer] delta-clamped {clamped}/{n_arm} arm joints "
                    f"to ±{self._max_joint_delta:.3f} rad",
                    throttle_duration_sec=2.0,
                )
        else:
            safe_arm = target_arm.tolist()

        grip_raw = self._gripper_position_from(latest_js)
        current_grip_norm = gripper_normalize(grip_raw) if not np.isnan(grip_raw) else float("nan")

        # CSV: pred (raw model output) + current (this tick) + cmd (post-clamp).
        if self._predictions_writer is not None and self._predictions_file is not None:
            nan_arm = [float("nan")] * n_arm
            obs_arm_row = self._obs_arm.tolist() if self._obs_arm is not None else nan_arm
            current_arm_row = (
                current_arm.tolist()
                if current_arm is not None and len(current_arm) == n_arm
                else nan_arm
            )
            row = [time.time(), self._chunk_id, chunk_idx]
            row += obs_arm_row + [self._obs_grip]
            row += target_arm.tolist() + [left_grip_norm]
            row += current_arm_row + [current_grip_norm]
            row += safe_arm + [left_grip_norm]  # gripper has no clamp
            self._predictions_writer.writerow(row)
            self._predictions_file.flush()

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

    def _write_chunk_images(
        self, chunk_id: int, frames: dict[str, tuple[float, np.ndarray]]
    ) -> None:
        """Save the camera frames that fed predict() for this chunk."""
        try:
            written = 0
            if self._image_log_individual:
                for cam_name, (_, rgb) in frames.items():
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    fname = f"{self._image_log_prefix}_chunk_{chunk_id:04d}_{cam_name}.jpg"
                    cv2.imwrite(str(self._image_log_dir / fname), bgr)
                    written += 1
            if self._image_log_stitched:
                ordered: list[np.ndarray] = []
                for cam_name in self._camera_names:
                    entry = frames.get(cam_name)
                    if entry is None:
                        ordered.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    else:
                        ordered.append(entry[1])
                stitched = stitch_frames(ordered)
                bgr = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR)
                fname = f"{self._image_log_prefix}_chunk_{chunk_id:04d}_stitched.jpg"
                cv2.imwrite(str(self._image_log_dir / fname), bgr)
                written += 1
            self.get_logger().info(
                f"[neura-infer] wrote {written} image(s) for chunk {chunk_id} "
                f"-> {self._image_log_dir}/{self._image_log_prefix}_chunk_"
                f"{chunk_id:04d}_*.jpg"
            )
        except Exception as e:
            self.get_logger().warning(f"[neura-infer] image write for chunk {chunk_id} failed: {e}")

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
