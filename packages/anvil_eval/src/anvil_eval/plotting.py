"""Matplotlib plots for evaluation results."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .metrics import EpisodeMetrics


def reorder_joint_names(joint_names: list[str]) -> list[str]:
    """Reorder joints such that finger_joint1 is last (if present)."""
    finger_joints = [jn for jn in joint_names if "finger_joint1" in jn]
    other_joints = [jn for jn in joint_names if jn not in finger_joints]
    return other_joints + sorted(finger_joints)


def plot_episode_joints(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    joint_names: list[str],
    metrics: EpisodeMetrics,
    output_path: Path,
    raw_output: np.ndarray | None = None,
    obs_states: np.ndarray | None = None,
    action_type: str = "absolute",
    raw_ground_truth: np.ndarray | None = None,
    phase_boundaries: dict[str, list[tuple[int, str]]] | None = None,
) -> None:
    """Plot predicted vs ground-truth joint trajectories for one episode.

    Layout (per joint column):
    - Top block (absolute scale): GT, Pred, obs_state
    - Bottom block (delta scale, when action_type is delta and raw_ground_truth provided):
      raw model output and ΔGT taken directly from raw_ground_truth (pre-computed by evaluator)

    phase_boundaries: optional {arm: [(frame, entered_state)]} — draws a dashed vertical
        line at each gripper phase transition on that arm's joint subplots. ``entered_state``
        ("closed"=grasp, "open"=release) sets the line color.
    """
    import matplotlib.pyplot as plt

    _PHASE_COLOR = {"closed": "tab:green", "open": "tab:red"}

    def _draw_phase_lines(ax, arm: str, label: bool) -> None:
        if not phase_boundaries:
            return
        seen: set[str] = set()
        for frame, state in phase_boundaries.get(arm, []):
            lbl = None
            if label and state not in seen:
                lbl = "grasp" if state == "closed" else "release"
                seen.add(state)
            ax.axvline(frame, color=_PHASE_COLOR.get(state, "gray"),
                       linestyle="--", linewidth=0.9, alpha=0.6, label=lbl)

    new_names = reorder_joint_names(joint_names)
    idx_map = [joint_names.index(name) for name in new_names]
    n_joints = len(new_names)
    ncols = min(4, n_joints)
    nrows_abs = math.ceil(n_joints / ncols)

    show_delta = action_type in ("delta_obs_t", "delta_sequential") and raw_ground_truth is not None
    nrows_delta = math.ceil(n_joints / ncols) if show_delta else 0
    total_rows = nrows_abs + nrows_delta

    fig, axes = plt.subplots(
        total_rows, ncols,
        figsize=(4 * ncols, 3 * total_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Episode {metrics.episode_idx} [{metrics.split_label}] — MAE: {metrics.mae:.4f}",
        fontsize=14,
    )

    frames = np.arange(predicted.shape[0])

    for j, name in enumerate(new_names):
        orig_idx = idx_map[j]
        abs_row = j // ncols
        col = j % ncols
        arm = name.split("_")[0]  # "left"/"right" — selects this joint's phase boundaries

        # ── Top block: absolute signals ──
        ax = axes[abs_row][col]
        ax.plot(frames, ground_truth[:, orig_idx], "b-", linewidth=1.0, label="GT")
        ax.plot(frames, predicted[:, orig_idx], "r--", linewidth=1.0, label="Pred")
        if obs_states is not None:
            ax.plot(frames, obs_states[:, orig_idx], color="purple",
                    linewidth=0.9, alpha=0.7, label="Obs")
        _draw_phase_lines(ax, arm, label=(j == 0))
        joint_mae = metrics.per_joint_mae.get(name, 0.0)
        ax.set_title(f"{name} (MAE: {joint_mae:.4f})", fontsize=9)
        ax.set_xlabel("frame", fontsize=8)
        ax.set_ylabel("rad", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

        # ── Bottom block: delta signals ──
        if show_delta:
            delta_row = nrows_abs + (j // ncols)
            ax_d = axes[delta_row][col]
            if raw_output is not None and orig_idx < raw_output.shape[1]:
                ax_d.plot(frames, raw_output[:, orig_idx], color="darkorange",
                          linewidth=0.8, linestyle=":", label="Raw (delta)")
            if orig_idx < raw_ground_truth.shape[1]:
                ax_d.plot(frames, raw_ground_truth[:, orig_idx], color="green",
                          linewidth=0.8, linestyle="--", label="ΔGT")
            _draw_phase_lines(ax_d, arm, label=False)
            ax_d.set_title(f"{name} [delta]", fontsize=9)
            ax_d.set_xlabel("frame", fontsize=8)
            ax_d.set_ylabel("delta [rad]", fontsize=8)
            ax_d.tick_params(labelsize=7)
            if j == 0:
                ax_d.legend(fontsize=7)

    # Hide unused subplots in both blocks
    for block_start, nrows_block in [(0, nrows_abs), (nrows_abs, nrows_delta)]:
        for j in range(n_joints, nrows_block * ncols):
            r = block_start + j // ncols
            c = j % ncols
            if r < total_rows:
                axes[r][c].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_horizon_curve(agg: dict, output_path: Path) -> None:
    """Plot mean absolute error vs. horizon offset, one line per split.

    A vertical marker shows the executed prefix (native ``n_action_steps``): error to
    its left is what the robot runs; error to its right is the discarded tail. A flat
    curve means the model holds its plan well (safe to execute longer); a steep one
    points to retraining.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    exec_len = None
    for split, d in agg.items():
        offsets = np.asarray(d["offsets"])
        mean = np.asarray(d["mae_mean"])
        std = np.asarray(d["mae_std"])
        line, = ax.plot(offsets, mean, marker="o", markersize=3, label=f"{split} (n={d['count'][0] if d['count'] else 0})")
        ax.fill_between(offsets, mean - std, mean + std, alpha=0.15, color=line.get_color())
        exec_len = d.get("executed_len") or exec_len

    if exec_len:
        ax.axvline(exec_len - 0.5, color="gray", linestyle="--", linewidth=1.0,
                   label=f"executed prefix (n_action_steps={exec_len})")

    ax.set_xlabel("horizon offset (steps ahead of inference)")
    ax.set_ylabel("mean |error| [rad]")
    ax.set_title("Prediction error vs. horizon")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_phase_mae_timeline(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    joint_names: list[str],
    phase_boundaries: dict[str, list[tuple[int, str]]],
    episode_idx: int,
    split_label: str,
    output_path: Path,
) -> None:
    """Per-arm MAE-over-time with phase lines: one subplot per arm.

    Each subplot shows ``frame vs mean|error|`` averaged across that arm's joints (gripper
    finger joints excluded), with grasp (green) / release (red) phase boundaries overlaid.
    MAE (abs taken per joint before averaging) means positive/negative per-joint errors do
    not cancel.
    """
    import matplotlib.pyplot as plt

    arms = [a for a in phase_boundaries if _arm_score_indices(joint_names, a)]
    if not arms:
        return
    err = np.abs(predicted - ground_truth)  # (T, D)
    frames = np.arange(err.shape[0])

    fig, axes = plt.subplots(len(arms), 1, figsize=(10, 3 * len(arms)), squeeze=False)
    fig.suptitle(f"Episode {episode_idx} [{split_label}] — per-arm MAE over time", fontsize=13)
    color = {"closed": "tab:green", "open": "tab:red"}

    for row, arm in enumerate(arms):
        ax = axes[row][0]
        jidx = _arm_score_indices(joint_names, arm)
        arm_mae = err[:, jidx].mean(axis=1)
        ax.plot(frames, arm_mae, "b-", linewidth=1.0, label="MAE")
        seen: set[str] = set()
        for frame, state in phase_boundaries.get(arm, []):
            lbl = None
            if state not in seen:
                lbl = "grasp" if state == "closed" else "release"
                seen.add(state)
            ax.axvline(frame, color=color.get(state, "gray"), linestyle="--",
                       linewidth=0.9, alpha=0.6, label=lbl)
        ax.set_title(f"{arm} arm (mean over {len(jidx)} joints, fingers excluded) — "
                     f"MAE {arm_mae.mean():.4f}", fontsize=10)
        ax.set_xlabel("frame", fontsize=8)
        ax.set_ylabel("mean |error| [rad]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _arm_score_indices(joint_names: list[str], arm: str) -> list[int]:
    """Indices of an arm's joints used for MAE, excluding gripper finger joints."""
    return [
        i for i, n in enumerate(joint_names)
        if (n == arm or n.startswith(f"{arm}_")) and "finger_joint1" not in n
    ]


def plot_monitor_signals(
    obs: np.ndarray,
    cmd: np.ndarray,
    raw_output: np.ndarray | None,
    joint_names: list[str],
    title: str,
    output_path: Path,
    action_type: str = "absolute",
    ncols: int = 4,
    dpi: int = 120,
) -> None:
    """Plot inference monitor signals (obs/cmd/raw) for one session.

    Layout (per joint column):
    - Top block: obs_state, control_cmd
    - Bottom block (when action_type is delta): raw model output, delta_cmd

    delta_cmd is computed from cmd based on action_type:
    - delta_obs_t:      delta_cmd[t] = cmd[t] - obs[t]
    - delta_sequential: delta_cmd[0] = cmd[0] - obs[0]; delta_cmd[t] = cmd[t] - cmd[t-1]
    """
    import matplotlib.pyplot as plt

    n_joints = obs.shape[1]
    frames = np.arange(obs.shape[0])

    show_delta = action_type in ("delta_obs_t", "delta_sequential") and raw_output is not None
    ncols = min(ncols, n_joints)
    nrows_abs = math.ceil(n_joints / ncols)
    nrows_delta = math.ceil(n_joints / ncols) if show_delta else 0
    total_rows = nrows_abs + nrows_delta

    # Compute delta_cmd in model-output space based on action_type
    delta_cmd: np.ndarray | None = None
    if show_delta:
        d = min(raw_output.shape[1], cmd.shape[1], obs.shape[1])
        if action_type == "delta_sequential":
            delta_cmd = np.zeros((obs.shape[0], d), dtype=np.float32)
            delta_cmd[0] = cmd[0, :d] - obs[0, :d]
            delta_cmd[1:] = np.diff(cmd[:, :d], axis=0)
        else:  # delta_obs_t
            delta_cmd = cmd[:, :d] - obs[:, :d]

    def _joint_label(j: int) -> str:
        return joint_names[j] if j < len(joint_names) else f"joint[{j}]"

    fig, axes = plt.subplots(
        total_rows, ncols,
        figsize=(4 * ncols, 3 * total_rows),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=12)

    for j in range(n_joints):
        abs_row = j // ncols
        col = j % ncols

        ax = axes[abs_row][col]
        ax.plot(frames, obs[:, j], color="steelblue", linewidth=0.8, label="obs.state")
        ax.plot(frames, cmd[:, j], color="forestgreen", linewidth=0.8, label="control cmd")
        ax.set_title(_joint_label(j), fontsize=8)
        ax.set_xlabel("step", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right")

        if show_delta:
            delta_row = nrows_abs + (j // ncols)
            ax_d = axes[delta_row][col]
            if raw_output is not None and j < raw_output.shape[1]:
                ax_d.plot(frames, raw_output[:, j], color="darkorange", linewidth=0.8,
                          linestyle=":", label="raw output (delta)")
            if delta_cmd is not None and j < delta_cmd.shape[1]:
                ax_d.plot(frames, delta_cmd[:, j], color="crimson", linewidth=0.8,
                          linestyle="--", label="delta cmd")
            ax_d.set_title(f"{_joint_label(j)} [delta]", fontsize=8)
            ax_d.set_xlabel("step", fontsize=7)
            ax_d.set_ylabel("delta [rad]", fontsize=7)
            ax_d.tick_params(labelsize=6)
            if j == 0:
                ax_d.legend(fontsize=6, loc="upper right")

    for block_start, nrows_block in [(0, nrows_abs), (nrows_abs, nrows_delta)]:
        for j in range(n_joints, nrows_block * ncols):
            r = block_start + j // ncols
            c = j % ncols
            if r < total_rows:
                axes[r][c].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_summary_box_plot(
    all_metrics: list[EpisodeMetrics],
    joint_names: list[str],
    output_path: Path,
) -> None:
    """Plot per-joint MAE summary box plot, grouped by split."""
    import matplotlib.pyplot as plt

    by_split: dict[str, list[EpisodeMetrics]] = {}
    for m in all_metrics:
        by_split.setdefault(m.split_label, []).append(m)

    split_names = sorted(by_split.keys())
    n_splits = len(split_names)

    ordered_joint_names = reorder_joint_names(joint_names)
    n_joints = len(ordered_joint_names)

    if n_splits == 0 or n_joints == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, n_joints * 1.5), 6))

    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

    group_width = 0.8
    box_width = group_width / n_splits
    x = np.arange(n_joints)

    for i, split_name in enumerate(split_names):
        metrics_list = by_split[split_name]

        split_data = []
        for jn in ordered_joint_names:
            vals = [m.per_joint_mae[jn] for m in metrics_list]
            split_data.append(vals)

        offset = (i - n_splits / 2 + 0.5) * box_width
        pos = x + offset

        bp = ax.boxplot(
            split_data,
            positions=pos,
            widths=box_width * 0.8,
            patch_artist=True,
            showfliers=True,
            manage_ticks=False,
        )

        color = colors[i % len(colors)]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.plot([], [], color=color, label=split_name, linewidth=10, alpha=0.6)

    ax.set_xlabel("Joint")
    ax.set_ylabel("MAE")
    ax.set_title("Distribution of Per-Joint MAE by Split")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_joint_names, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(title="Split")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
