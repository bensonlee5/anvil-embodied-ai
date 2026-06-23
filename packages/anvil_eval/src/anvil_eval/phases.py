"""Heuristic task-phase segmentation from gripper open/close transitions.

Phases are derived from the **ground-truth** gripper signal (``*_finger_joint1``) per arm,
so the boundaries are identical across checkpoints (valid cross-model comparison). Each arm
is binarized at the midpoint of its observed range with hysteresis, debounced by a minimum
segment length, and cut at **every** transition (open→close and close→open). Segments get
generic labels like ``left:closed#1`` — no hardcoded task semantics.

These phases are an orthogonal overlay: any analysis mode can group by them. By convention an
arm's phases are scored on that arm's own joints (see ``arm_joint_indices``).
"""

from __future__ import annotations

import numpy as np

GRIPPER_SUFFIX = "finger_joint1"


def find_gripper_indices(joint_names: list[str]) -> dict[str, int]:
    """Map arm name -> index of its gripper joint, e.g. {"left": 0, "right": 8}.

    Arm name is the joint name with the gripper suffix stripped (``left_finger_joint1`` ->
    ``left``). Single-arm setups yield a single entry.
    """
    out: dict[str, int] = {}
    for i, name in enumerate(joint_names):
        if GRIPPER_SUFFIX in name:
            arm = name.split(f"_{GRIPPER_SUFFIX}")[0] or name.replace(GRIPPER_SUFFIX, "").strip("_") or "arm"
            out[arm] = i
    return out


def arm_joint_indices(joint_names: list[str], arm: str) -> list[int]:
    """Indices of joints belonging to ``arm`` (name starts with ``{arm}_`` or equals arm)."""
    return [i for i, n in enumerate(joint_names) if n == arm or n.startswith(f"{arm}_")]


def _segment_signal(
    signal: np.ndarray,
    arm: str,
    *,
    closed_is_low: bool = True,
    min_segment: int = 5,
    hysteresis_frac: float = 0.15,
    flat_range_eps: float = 0.02,
) -> list[tuple[int, int, str]]:
    """Segment a 1-D gripper signal into (start, end_exclusive, label) runs.

    closed_is_low: treat values below the midpoint as the "closed" state (convention;
        flip if your gripper encodes closed as the larger value).
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    T = signal.shape[0]
    if T == 0:
        return []

    lo, hi = float(signal.min()), float(signal.max())
    rng = hi - lo
    if rng < flat_range_eps:
        # Gripper never moves — one undifferentiated phase.
        return [(0, T, f"{arm}:nograsp#1")]

    mid = 0.5 * (lo + hi)
    band = hysteresis_frac * rng
    thr_lo, thr_hi = mid - band, mid + band

    # Hysteresis state machine: `low` = signal in the lower band region.
    low = signal[0] < mid
    is_low = np.empty(T, dtype=bool)
    for i in range(T):
        v = signal[i]
        if low and v > thr_hi:
            low = False
        elif not low and v < thr_lo:
            low = True
        is_low[i] = low

    runs = _coalesce(is_low)
    runs = _debounce(runs, min_segment)

    # Label runs: low state == closed iff closed_is_low.
    segments: list[tuple[int, int, str]] = []
    counts: dict[str, int] = {}
    for start, end, low_state in runs:
        closed = low_state if closed_is_low else not low_state
        state = "closed" if closed else "open"
        counts[state] = counts.get(state, 0) + 1
        segments.append((start, end, f"{arm}:{state}#{counts[state]}"))
    return segments


def _coalesce(flags: np.ndarray) -> list[tuple[int, int, bool]]:
    """Group a boolean array into maximal constant runs: [(start, end_excl, value)]."""
    runs: list[tuple[int, int, bool]] = []
    start = 0
    for i in range(1, len(flags)):
        if flags[i] != flags[start]:
            runs.append((start, i, bool(flags[start])))
            start = i
    runs.append((start, len(flags), bool(flags[start])))
    return runs


def _debounce(runs: list[tuple[int, int, bool]], min_segment: int) -> list[tuple[int, int, bool]]:
    """Drop runs shorter than min_segment by merging them into the previous run, then
    re-coalesce adjacent same-state runs. Iterates until stable."""
    if min_segment <= 1:
        return runs
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for idx in range(len(runs)):
            s, e, v = runs[idx]
            if e - s < min_segment:
                # Flip this short run to a neighbor's state (previous if any, else next).
                nb = runs[idx - 1][2] if idx > 0 else runs[idx + 1][2]
                runs[idx] = (s, e, nb)
                changed = True
        if changed:
            # Re-coalesce
            merged: list[tuple[int, int, bool]] = []
            for s, e, v in runs:
                if merged and merged[-1][2] == v:
                    ps, _, pv = merged[-1]
                    merged[-1] = (ps, e, pv)
                else:
                    merged.append((s, e, v))
            runs = merged
    return runs


def label_phases(
    gt_actions: np.ndarray,
    joint_names: list[str],
    **kwargs,
) -> dict[str, list[tuple[int, int, str]]]:
    """Segment each arm's GT gripper trajectory into labeled phase runs.

    Args:
        gt_actions: (T, D) absolute ground-truth actions for the episode.
        joint_names: D joint names.
    Returns:
        {arm: [(start_frame, end_frame_exclusive, label)]}.
    """
    gt = np.asarray(gt_actions, dtype=np.float64)
    out: dict[str, list[tuple[int, int, str]]] = {}
    for arm, idx in find_gripper_indices(joint_names).items():
        out[arm] = _segment_signal(gt[:, idx], arm, **kwargs)
    return out


def segments_to_frame_map(segments: list[tuple[int, int, str]]) -> dict[int, str]:
    """Expand [(start, end_excl, label)] into {frame_index: label}."""
    out: dict[int, str] = {}
    for start, end, label in segments:
        for f in range(start, end):
            out[f] = label
    return out


def segments_to_boundaries(segments: list[tuple[int, int, str]]) -> list[tuple[int, str]]:
    """Transition points as (frame, entered_state) for plotting vertical phase lines.

    Skips the first segment (frame 0 is not a transition) and any ``nograsp`` runs.
    State is parsed from the ``arm:state#n`` label (e.g. "closed", "open").
    """
    out: list[tuple[int, str]] = []
    for i, (start, _end, label) in enumerate(segments):
        if i == 0:
            continue
        state = label.split(":")[-1].split("#")[0]
        if state == "nograsp":
            continue
        out.append((start, state))
    return out
