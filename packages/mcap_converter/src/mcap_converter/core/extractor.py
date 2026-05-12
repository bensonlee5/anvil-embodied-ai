"""Data extraction from MCAP files"""

from collections import deque
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import numpy as np
from mcap.exceptions import McapError
from mcap.reader import make_reader as make_mcap_reader

from ..config import ActionSource, DataConfig
from ..exceptions import DataExtractionError
from ..utils.image_utils import decode_compressed_image, resize_image
from .constants import (
    ARM_PREFIX_TO_NAME,
    JOINT_NAME_SEPARATOR,
    LEADER_PREFIX,
    OBSERVATION_PREFIX,
    QUEST_JOINT_ORDER,
    quest_command_topic,
)
from .reader import McapReader

# Map source prefix -> role label used in joint_buffers keys.
_ROLE_BY_PREFIX = {OBSERVATION_PREFIX: "observation", LEADER_PREFIX: "action"}

# Canonical joint order for quest Float64MultiArray output (alphabetical).
_QUEST_CANONICAL_JOINT_NAMES: list[str] = sorted(QUEST_JOINT_ORDER)
_QUEST_REORDER: np.ndarray = np.array(
    [QUEST_JOINT_ORDER.index(name) for name in _QUEST_CANONICAL_JOINT_NAMES],
    dtype=np.intp,
)


def parse_joint_name(joint_name: str) -> tuple[str, str, str] | None:
    """
    Parse `{source}_{arm}_{joint_id}` into (role, arm, joint_id).

    `source` must be one of `follower` (-> role "observation") or `leader`
    (-> role "action"). `arm` must be `l` (-> "left") or `r` (-> "right").

    Returns None if the name doesn't follow the convention — callers should
    skip such joints rather than fail.

    Examples:
        "follower_r_joint1"        -> ("observation", "right", "joint1")
        "leader_l_finger_joint1"   -> ("action", "left", "finger_joint1")
    """
    parts = joint_name.split(JOINT_NAME_SEPARATOR, 2)
    if len(parts) < 3:
        return None
    source, arm_prefix, joint_id = parts
    role = _ROLE_BY_PREFIX.get(source)
    arm = ARM_PREFIX_TO_NAME.get(arm_prefix)
    if role is None or arm is None:
        return None
    return role, arm, joint_id


def message_timestamp(message) -> float:
    """
    POSIX seconds for a message, preferring the ROS `header.stamp` over MCAP log_time.

    `header.stamp` is closer to when the data was actually captured (camera shutter
    fired, joint state was sampled); MCAP `log_time` is when the recorder received
    the message, which can lag by 10-20 ms for JPEG-compressed images. We fall back
    to log_time for messages without a header (e.g. std_msgs/Float64MultiArray).
    """
    stamp = getattr(getattr(message.ros_msg, "header", None), "stamp", None)
    if stamp is not None:
        return stamp.sec + stamp.nanosec * 1e-9
    return message.log_time_ns * 1e-9


class BufferedStreamExtractor:
    """
    Memory-efficient streaming extractor for MCAP conversion.

    Uses a sliding window buffer for time alignment while keeping memory bounded.
    Extracts both images and joint states (full teleop data).

    Algorithm:
        1. Fill buffer to half_buffer, then start processing from cursor=0
        2. Buffer grows from half_buffer -> full_buffer (expanding lookahead)
        3. At full_buffer, switch to FIFO (steady state with +/-half_buffer lookahead)
        4. Flush remaining with shrinking lookahead

    Dual-Rate Buffering:
        - Camera buffer: ~30 Hz -> 150 frames in 5 sec
        - Joint state buffer: 100-250 Hz -> 500-1250 samples in 5 sec
        - Joint state samples aligned to camera timestamps via nearest-neighbor

    Example:
        extractor = BufferedStreamExtractor(config, buffer_seconds=5.0, fps=30)
        for frame in extractor.extract_frames("recording.mcap", task="teleop"):
            dataset.add_frame(frame)
    """

    def __init__(
        self,
        config: DataConfig,
        buffer_seconds: float = 5.0,
        fps: int = 60,
        quiet: bool = False,
        progress_callback: Callable[[int], None] | None = None,
    ):
        """
        Initialize buffered stream extractor.

        Args:
            config: Data configuration specifying topics and camera mapping
            buffer_seconds: Total buffer window size in seconds (default: 5.0)
            fps: Frame rate for buffer size calculation (default: 30)
            quiet: If True, suppress all print output (default: False)
            progress_callback: Called with frames_yielded count after each frame
        """
        self.config = config
        self.fps = fps
        self.frame_interval = 1.0 / fps  # seconds between output frames (for subsampling)
        self.buffer_seconds = buffer_seconds
        self.half_buffer = int(buffer_seconds * fps / 2)  # 75 frames for 5s @ 30fps
        self.full_buffer = self.half_buffer * 2  # 150 frames
        self.target_size = tuple(config.image_resolution)  # (width, height)
        self.quiet = quiet
        self.progress_callback = progress_callback

        # For quest_teleop: map each command topic to its arm.
        self._quest_topic_to_arm: dict[str, str] = (
            {quest_command_topic(arm): arm for arm in config.arms}
            if config.action_source is ActionSource.quest_teleop
            else {}
        )

    @staticmethod
    def _check_quest_topics_present(mcap_path: Path, expected: set[str]) -> None:
        """Raise DataExtractionError when none of the quest command topics are in the MCAP.

        Uses the MCAP footer index (O(1)) — does not scan all messages.
        """
        try:
            with open(mcap_path, "rb") as _f:
                _summary = make_mcap_reader(_f).get_summary()
        except (McapError, OSError):
            return  # Cannot read summary — let streaming fail naturally

        if _summary is None:
            return

        available = {ch.topic for ch in _summary.channels.values()}
        if expected and expected.isdisjoint(available):
            raise DataExtractionError(
                f"\n[ACTION SOURCE ERROR] action_source=quest_teleop, but none of the "
                f"expected command topics were found in this MCAP.\n"
                f"  Expected: {sorted(expected)}\n"
                f"  Available in MCAP: {sorted(available)}\n\n"
                "Fix options:\n"
                "  1. Set action_source: future_observations in config to derive actions "
                "from observations.\n"
                "  2. Record a new session that captures the action-command topics.\n"
                "  3. Check the `arms` field in your config matches what was recorded.\n"
            )

    def extract_frames(
        self,
        mcap_path: Path,
        task: str = "teleop",
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generator with bounded memory usage, yielding complete time-aligned frames.

        Args:
            mcap_path: Path to MCAP file
            task: Task name (default: "teleop")

        Yields:
            Dictionary with aligned frame data:
            {
                'observation.images.{camera}': np.ndarray (H, W, C),
                'observation.state': np.ndarray,        # or '{robot}.observation.state'
                'observation.velocity': np.ndarray,     # optional
                'observation.effort': np.ndarray,       # optional
                'action': np.ndarray,                   # or '{robot}.action'
                'task': str,
            }
        """
        if not self.quiet:
            print(f"[BufferedStream] Reading MCAP: {mcap_path}")
            print(f"[BufferedStream] Buffer: {self.full_buffer} frames ({self.half_buffer}x2)")

        reader = McapReader(str(mcap_path))

        topic_to_cam = dict(self.config.camera_topic_mapping)

        # Per-camera buffers: {cam_name: deque of (timestamp, image)}
        camera_buffers: dict[str, deque] = {
            cam: deque() for cam in self.config.camera_topic_mapping.values()
        }

        # Joint state buffers: {(role, arm): {'buffer': deque, 'joint_names': list}}
        joint_buffers: dict[tuple[str, str], dict] = {}

        # Build subscription topic list per action_source.
        all_topics = list(self.config.camera_topic_mapping.keys())
        all_topics.append(self.config.robot_state_topic)

        if self.config.action_source is ActionSource.quest_teleop:
            expected_cmd_topics = set(self._quest_topic_to_arm.keys())
            all_topics.extend(expected_cmd_topics)
            self._check_quest_topics_present(mcap_path, expected_cmd_topics)
        elif self.config.action_source is ActionSource.future_observations:
            n = self.config.action_n_step
            assert n is not None  # guaranteed by DataConfig validator
            ms = int(n / self.fps * 1000)
            if not self.quiet:
                print(
                    f"\n[ACTION SOURCE] action_source=future_observations — "
                    f"action[t] = observation[t + {n}] (~{ms} ms lookahead at {self.fps} fps)\n"
                )

        # We arbitrarily pick one of the cameras to designate the "main cam", which will key off observation frames
        main_cam = list(self.config.camera_topic_mapping.values())[0]

        cursor = 0  # Index of observation frame to process
        frames_yielded = 0
        frames_dropped = 0
        next_yield_ts = None  # Next target timestamp, used when subsampling

        for message in reader.read_messages(topics=all_topics):
            topic = message.channel.topic

            # Buffer joint state messages
            if topic == self.config.robot_state_topic:
                self._buffer_joint_state(message, joint_buffers)
                continue

            # Handle action command messages (quest teleop mode)
            if topic in self._quest_topic_to_arm:
                arm = self._quest_topic_to_arm[topic]
                self._buffer_action_command(message, arm, joint_buffers)
                continue

            # Handle camera messages
            if topic not in topic_to_cam:
                continue
            cam_name = topic_to_cam[topic]

            time_s = message_timestamp(message)

            ros_msg = message.ros_msg
            img = decode_compressed_image(ros_msg.data, ros_msg.format)

            # Add to buffer
            camera_buffers[cam_name].append((time_s, img))

            # Start processing when main camera buffer reaches threshold
            main_buffer_len = len(camera_buffers[main_cam])

            # Condition: buffer has at least half_buffer frames ahead of cursor
            if main_buffer_len >= self.half_buffer + cursor:
                frame_ts = camera_buffers[main_cam][cursor][0]

                # Initialize subsampling anchor on first frame
                if next_yield_ts is None:
                    next_yield_ts = frame_ts

                # Subsampling: only yield if this frame is at or past the next target timestamp
                if frame_ts >= next_yield_ts:
                    frame = self._align_frame_at_cursor(
                        camera_buffers, joint_buffers, cursor, main_cam, task, resize_image
                    )
                    if frame is not None:
                        yield frame
                        frames_yielded += 1
                        next_yield_ts += self.frame_interval
                    else:
                        frames_dropped += 1

                cursor += 1

                # Always report progress after each cursor advance (including skipped frames)
                if self.progress_callback:
                    self.progress_callback(frames_yielded)
                elif not self.quiet and frames_yielded % 100 == 0:
                    print(f"[BufferedStream] Processed {frames_yielded} frames...")

                # Once buffer reaches full size, remove oldest to maintain size
                if main_buffer_len > self.full_buffer:
                    for buffer in camera_buffers.values():
                        if len(buffer) > 0:
                            buffer.popleft()

                    # Sync joint buffers to remove old samples
                    if camera_buffers[main_cam]:
                        new_oldest_ts = camera_buffers[main_cam][0][0]
                        self._sync_joint_buffers(joint_buffers, new_oldest_ts)

                    cursor -= 1  # Adjust cursor after removal

        # Flush: process remaining frames in buffer
        if not self.quiet:
            print("[BufferedStream] Flushing remaining buffer...")
        while cursor < len(camera_buffers[main_cam]):
            frame_ts = camera_buffers[main_cam][cursor][0]
            if next_yield_ts is None or frame_ts >= next_yield_ts:
                frame = self._align_frame_at_cursor(
                    camera_buffers, joint_buffers, cursor, main_cam, task, resize_image
                )
                if frame is not None:
                    yield frame
                    frames_yielded += 1
                    if next_yield_ts is not None:
                        next_yield_ts += self.frame_interval
                    if self.progress_callback:
                        self.progress_callback(frames_yielded)
                else:
                    frames_dropped += 1
            cursor += 1

        if not self.quiet:
            if frames_dropped:
                print(
                    f"[BufferedStream] [OK] Extracted {frames_yielded} frames total "
                    f"({frames_dropped} dropped during alignment)"
                )
            else:
                print(f"[BufferedStream] [OK] Extracted {frames_yielded} frames total")

        if frames_yielded == 0:
            cam_counts = {cam: len(buf) for cam, buf in camera_buffers.items()}
            joint_keys = {
                f"{role}:{arm}": len(d["buffer"])
                for (role, arm), d in joint_buffers.items()
            }
            print(
                f"[BufferedStream] WARNING: 0 frames produced "
                f"({frames_dropped} dropped during alignment) — diagnostics:"
            )
            print(f"  Camera buffers: {cam_counts}")
            print(f"  Joint buffers:  {joint_keys or '(empty)'}")
            if not cam_counts or all(c == 0 for c in cam_counts.values()):
                print("  -> No camera images found. Check these topics exist in the MCAP:")
                for t in self.config.camera_topic_mapping:
                    print(f"       {t}")
            if not joint_keys:
                print(
                    f"  -> No joint state data. Check robot_state_topic: "
                    f"{self.config.robot_state_topic}"
                )
            elif self.config.action_source is ActionSource.quest_teleop and \
                 not any(k.startswith("action:") for k in joint_keys):
                print("  -> No action data from quest command topics:")
                for t in self._quest_topic_to_arm:
                    print(f"       {t}")
            elif self.config.action_source is ActionSource.leader and \
                 not any(k.startswith("action:") for k in joint_keys):
                print("  -> No 'leader_*' joints found in /joint_states.")

    def _align_frame_at_cursor(
        self,
        camera_buffers: dict[str, deque],
        joint_buffers: dict[tuple[str, str], dict],
        cursor: int,
        main_cam: str,
        task: str,
        resize_func,
    ) -> dict[str, Any] | None:
        """
        Align frame at cursor position using entire buffer for matching.

        Args:
            camera_buffers: Per-camera buffers
            joint_buffers: Joint state buffers (keyed by (role, robot))
            cursor: Index of frame to align in main camera buffer
            main_cam: Name of main camera
            task: Task name
            resize_func: Function to resize images

        Returns:
            Aligned frame dictionary, or None if not all data available
        """
        # Get main camera frame at cursor
        main_ts, main_img = camera_buffers[main_cam][cursor]

        # Resize main camera image
        resized_main = resize_func(main_img, self.target_size)
        frame = {f"observation.images.{main_cam}": resized_main}

        # Find nearest match for each other camera
        for cam_name, buffer in camera_buffers.items():
            if cam_name == main_cam:
                continue

            # If any camera has no data, skip this frame
            if len(buffer) == 0:
                return None

            # Search entire buffer for nearest timestamp match
            nearest_idx = self._find_nearest_in_buffer(buffer, main_ts)
            if nearest_idx is not None:
                _, img = buffer[nearest_idx]
                resized_img = resize_func(img, self.target_size)
                frame[f"observation.images.{cam_name}"] = resized_img
            else:
                return None

        # Align joint states
        if joint_buffers:
            if self.config.action_source is ActionSource.future_observations:
                assert self.config.action_n_step is not None
                action_ts = main_ts + self.frame_interval * self.config.action_n_step
            else:
                action_ts = None
            joint_aligned = self._align_joint_states(joint_buffers, main_ts, action_ts=action_ts)
            if joint_aligned is None:
                return None  # Skip frame if joint states not available
            frame.update(joint_aligned)

        frame["task"] = task
        return frame

    def _align_joint_states(
        self,
        joint_buffers: dict[tuple[str, str], dict],
        target_ts: float,
        action_ts: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Align joint states to target timestamp. Concatenates per-arm features in
        sorted arm order (left then right).

        - observation.state / observation.velocity / observation.effort: from `target_ts`
        - action: from action-role buffer at `target_ts`, OR from observation buffer at
          `action_ts` when `action_source == future_observations`.

        Returns None if any required arm is missing data.
        """
        obs_data: dict[str, dict[str, np.ndarray]] = {}
        action_data: dict[str, np.ndarray] = {}

        # Pull nearest sample per (role, arm).
        for (role, arm), data in joint_buffers.items():
            buffer = data["buffer"]
            if not buffer:
                return None
            nearest_idx = self._find_nearest_in_buffer(buffer, target_ts)
            if nearest_idx is None:
                return None
            _, pos, vel, eff = buffer[nearest_idx]
            if role == "observation":
                obs_data[arm] = {"pos": pos, "vel": vel, "eff": eff}
            else:
                action_data[arm] = pos

        # future_observations: derive action from a forward-in-time observation lookup.
        if self.config.action_source is ActionSource.future_observations:
            if action_ts is None:
                return None
            for (role, arm), data in joint_buffers.items():
                if role != "observation":
                    continue
                idx = self._find_nearest_in_buffer(data["buffer"], action_ts)
                if idx is None:
                    return None
                _, pos, _, _ = data["buffer"][idx]
                action_data[arm] = pos

        # Both observation and action must cover every configured arm.
        arms = sorted(self.config.arms)
        for arm in arms:
            if arm not in obs_data or arm not in action_data:
                return None

        return {
            "observation.state": np.concatenate([obs_data[a]["pos"] for a in arms]),
            "observation.velocity": np.concatenate([obs_data[a]["vel"] for a in arms]),
            "observation.effort": np.concatenate([obs_data[a]["eff"] for a in arms]),
            "action": np.concatenate([action_data[a] for a in arms]),
        }

    def _find_nearest_in_buffer(
        self,
        buffer: deque,
        target_ts: float,
    ) -> int | None:
        """
        Find index of frame with nearest timestamp in buffer.

        Args:
            buffer: Deque of (timestamp, ...) tuples (first element is timestamp)
            target_ts: Target timestamp to match

        Returns:
            Index of nearest frame, or None if buffer is empty
        """
        if not buffer:
            return None

        # Linear search for nearest (buffer is small, typically 150-1000 items)
        min_diff = float("inf")
        nearest_idx = 0

        for i, item in enumerate(buffer):
            ts = item[0]  # First element is always timestamp
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                nearest_idx = i

        return nearest_idx

    def _buffer_joint_state(
        self,
        message,
        joint_buffers: dict[tuple[str, str], dict],
    ) -> None:
        """
        Parse a JointState message and add per-(role, arm) samples to `joint_buffers`.
        Joint values within each group are sorted alphabetically by joint_id for a
        deterministic canonical order across messages.
        """
        ros_msg = message.ros_msg
        timestamp = message_timestamp(message)

        # Group by (role, arm).
        grouped: dict[tuple[str, str], dict] = {}
        for i, joint_name in enumerate(ros_msg.name):
            parsed = parse_joint_name(joint_name)
            if parsed is None:
                continue
            role, arm, joint_id = parsed

            key = (role, arm)
            g = grouped.setdefault(
                key, {"joint_ids": [], "position": [], "velocity": [], "effort": []}
            )
            g["joint_ids"].append(joint_id)
            if ros_msg.position and i < len(ros_msg.position):
                g["position"].append(ros_msg.position[i])
            if ros_msg.velocity and i < len(ros_msg.velocity):
                g["velocity"].append(ros_msg.velocity[i])
            if ros_msg.effort and i < len(ros_msg.effort):
                g["effort"].append(ros_msg.effort[i])

        # Sort each group's parallel arrays by joint_id.
        for g in grouped.values():
            order = sorted(range(len(g["joint_ids"])), key=lambda idx: g["joint_ids"][idx])
            for field in ("joint_ids", "position", "velocity", "effort"):
                g[field] = [g[field][idx] for idx in order] if g[field] else []

        for key, g in grouped.items():
            if key not in joint_buffers:
                joint_buffers[key] = {"buffer": deque(), "joint_names": g["joint_ids"]}
            pos = np.array(g["position"], dtype=np.float32)
            vel = np.array(g["velocity"], dtype=np.float32)
            eff = np.array(g["effort"], dtype=np.float32)
            joint_buffers[key]["buffer"].append((timestamp, pos, vel, eff))

    def _buffer_action_command(
        self,
        message,
        arm: str,
        joint_buffers: dict[tuple[str, str], dict],
    ) -> None:
        """
        Parse a Float64MultiArray action command and add it to `joint_buffers`.
        Values arrive in QUEST_JOINT_ORDER and are reordered to canonical
        (alphabetical) joint order to match observation grouping.
        """
        timestamp = message_timestamp(message)
        positions = np.array(message.ros_msg.data, dtype=np.float32)[_QUEST_REORDER]

        key = ("action", arm)
        if key not in joint_buffers:
            joint_buffers[key] = {
                "buffer": deque(),
                "joint_names": list(_QUEST_CANONICAL_JOINT_NAMES),
            }
        empty = np.array([], dtype=np.float32)
        joint_buffers[key]["buffer"].append((timestamp, positions, empty, empty))

    def _sync_joint_buffers(
        self,
        joint_buffers: dict[tuple[str, str], dict],
        oldest_camera_ts: float,
    ) -> None:
        """
        Remove joint state samples older than the oldest camera frame.

        This keeps joint state buffers synchronized with camera buffer time window.

        Args:
            joint_buffers: Joint state buffers to sync
            oldest_camera_ts: Timestamp of oldest remaining camera frame
        """
        for data in joint_buffers.values():
            buffer = data["buffer"]
            while buffer and buffer[0][0] < oldest_camera_ts:
                buffer.popleft()
