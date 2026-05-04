"""This example demonstrates how you can retrieve a dataset
from the Neuracore platform and visualize it."""

import argparse
import csv
import sys
from pathlib import Path

# Make `neuracore_control.common` importable when this script is run directly
# (`python scripts/download_data.py`) without `colcon build` / `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from neuracore_types import DataType, JointData, RGBCameraData

import neuracore as nc

from neuracore_control.common import CAMERAS, LEFT_ARM, LEFT_GRIPPER


def visualize_episode(
    joint_positions: list[dict[str, JointData]],
    camera_data: list[dict[str, RGBCameraData]],
    timestamps: list[float],
    episode_dir: Path,
    start_time: float = 0.0,
    end_time: float = 0.0,
):
    """Animate joint plot + first camera feed; save as episode.mp4 and show."""
    joint_names = list(joint_positions[0].keys())
    jps = np.array(
        [
            [joint_positions[t][name].value for name in joint_names]
            for t in range(len(joint_positions))
        ]
    )

    # Extract frames from the first camera in the dict at each timestep
    # Assumes we want to visualize the first camera
    first_camera_name = list(camera_data[0].keys())[0]
    images = np.array([camera_data[t][first_camera_name].frame for t in range(len(camera_data))])

    # Calculate relative times from timestamps
    relative_times = np.array([t - start_time for t in timestamps])

    # Add a "fake" point at the end using end_time
    jps = np.vstack([jps, jps[-1:]])
    images = np.vstack([images, images[-1:]])
    relative_times = np.append(relative_times, end_time - start_time)

    # Create a more compact figure
    fig = plt.figure(figsize=(12, 4))

    # Plot joint positions
    ax1 = plt.subplot(1, 2, 1)
    for joint_idx, joint_name in enumerate(joint_names):
        ax1.plot(relative_times, jps[:, joint_idx], label=joint_name)
    ax1.set_title("Joint Positions")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Camera feed
    ax2 = plt.subplot(1, 2, 2)
    img_display = ax2.imshow(images[0])
    ax2.set_title("Camera Feed")
    ax2.axis("off")

    # Time indicator and timestamp
    time_line = ax1.axvline(x=relative_times[0], color="r")
    timestamp_text = ax2.text(
        0.02,
        0.95,
        f"Time: {relative_times[0]:.2f}s / {relative_times[-1]:.2f}s",
        transform=ax2.transAxes,
        color="white",
        bbox=dict(facecolor="black", alpha=0.7),
    )

    plt.tight_layout()

    def update(frame):
        img_display.set_array(images[frame])
        time_line.set_xdata([relative_times[frame], relative_times[frame]])
        timestamp_text.set_text(f"Time: {relative_times[frame]:.2f}s / {relative_times[-1]:.2f}s")
        return [img_display, time_line, timestamp_text]

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(images), interval=20, blit=True, repeat=True
    )

    # Add play/pause button
    button_ax = plt.axes([0.45, 0.01, 0.1, 0.04])
    button = plt.Button(button_ax, "Play/Pause")

    def toggle_pause(event):
        if ani.running:
            ani.event_source.stop()
        else:
            ani.event_source.start()
        ani.running ^= True

    button.on_clicked(toggle_pause)
    ani.running = True

    episode_dir.mkdir(parents=True, exist_ok=True)
    out_path = episode_dir / "episode.mp4"
    ani.save(str(out_path), fps=50)
    print(f"saved animation to {out_path}")

    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="kit-block-4")
    ap.add_argument(
        "--episode-idx", type=int, default=0, help="Which episode in the dataset to download"
    )
    args = ap.parse_args()

    nc.login()

    dataset = nc.get_dataset(args.dataset)

    cross_embodiment_union: dict = {
        robot_id: {
            DataType.JOINT_POSITIONS: LEFT_ARM,
            DataType.JOINT_TARGET_POSITIONS: LEFT_ARM,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: [LEFT_GRIPPER],
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: [LEFT_GRIPPER],
            DataType.RGB_IMAGES: CAMERAS,
        }
        for robot_id in dataset.robot_ids
    }

    synced_dataset = dataset.synchronize(
        frequency=50,
        cross_embodiment_union=cross_embodiment_union,
    )
    print(f"Number of episodes: {len(dataset)}")


    data_dir = Path(__file__).resolve().parent / "data"
    episode_dir = data_dir / f"episode_{args.episode_idx:03d}"
    images_dir = episode_dir / "images"
    for cam in CAMERAS:
        (images_dir / cam).mkdir(parents=True, exist_ok=True)

    print(f"Streaming episode {args.episode_idx} from dataset '{args.dataset}'")
    episode = synced_dataset[args.episode_idx]
    obs_path = episode_dir / "joint_observations.csv"
    tgt_path = episode_dir / "joint_targets.csv"
    joint_positions = []
    camera_data = []
    timestamps = []
    with obs_path.open("w", newline="") as obs_f, tgt_path.open("w", newline="") as tgt_f:
        obs_writer = csv.writer(obs_f)
        tgt_writer = csv.writer(tgt_f)
        header = ["t"] + LEFT_ARM + [LEFT_GRIPPER]
        obs_writer.writerow(header)
        tgt_writer.writerow(header)

        for i, step in enumerate(episode):
            joints = step[DataType.JOINT_POSITIONS]
            grip = step[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
            tgt_joints = step[DataType.JOINT_TARGET_POSITIONS]
            tgt_grip = step[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
            obs_writer.writerow(
                [step.timestamp]
                + [joints[n].value for n in LEFT_ARM]
                + [grip[LEFT_GRIPPER].open_amount]
            )
            tgt_writer.writerow(
                [step.timestamp]
                + [tgt_joints[n].value for n in LEFT_ARM]
                + [tgt_grip[LEFT_GRIPPER].open_amount]
            )
            joint_positions.append(joints)
            camera_data.append(step[DataType.RGB_IMAGES])
            timestamps.append(step.timestamp)

            for cam, cam_data in step[DataType.RGB_IMAGES].items():
                cam_data.frame.save(
                    images_dir / cam / f"frame_{i:04d}.jpg", quality=95
                )

    print(f"saved {len(timestamps)} frames per camera to {images_dir}")
    print(f"saved observations to {obs_path}")
    print(f"saved targets to {tgt_path}")

    print(f"Episode length t: {episode.end_time - episode.start_time} seconds")
    visualize_episode(
        joint_positions,
        camera_data,
        timestamps,
        episode_dir=episode_dir,
        start_time=episode.start_time,
        end_time=episode.end_time,
    )


if __name__ == "__main__":
    main()
