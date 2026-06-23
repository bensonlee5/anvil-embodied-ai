"""Tests for horizon capture, substrate, and aggregation.

The capture path is exercised with a stub queue-based policy (no real model / GPU),
mirroring how diffusion fills an action queue and exposes ``predict_action_chunk``.
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from anvil_eval.evaluator import EpisodeEvaluator
from anvil_eval.horizon import aggregate_horizon, write_horizon_csv
from anvil_eval.metrics import compute_episode_metrics
from anvil_eval.phases import (
    arm_joint_indices,
    find_gripper_indices,
    label_phases,
    segments_to_boundaries,
    segments_to_frame_map,
)
from anvil_eval.plotting import plot_episode_joints, plot_phase_mae_timeline
from anvil_eval.substrate import (
    AnchorChunk,
    EpisodeSubstrate,
    iter_records,
    write_substrate_csv,
)

D = 2
JOINTS = ["j1", "j2"]


class _StubConfig:
    def __init__(self):
        self.n_action_steps = 4   # native executed steps
        self.n_obs_steps = 2
        self.horizon = 10
        self.temporal_ensemble_coeff = None


class _StubPolicy:
    """Queue-based policy mirroring diffusion: observations AND actions live in _queues,
    and predict_action_chunk rebuilds its input from the queue (not the passed batch) —
    matching how the evaluator calls it with a key-only dict.

    Predictions are deterministic in obs.state so the executed path and the captured
    chunk agree (real diffusion samples differ; here we test the plumbing, not RNG).
    """

    def __init__(self):
        self.config = _StubConfig()
        self.reset()

    def reset(self):
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }

    def _chunk_from_queue(self, n):
        base = self._queues["observation.state"][-1].flatten()[:D].to(torch.float64)
        return torch.stack([base + k * 0.1 for k in range(n)])  # (n, D)

    def select_action(self, batch):
        self._queues["observation.state"].append(batch["observation.state"])
        if len(self._queues["action"]) == 0:
            chunk = self._chunk_from_queue(self.config.n_action_steps)
            self._queues["action"].extend([chunk[k : k + 1] for k in range(chunk.shape[0])])
        return self._queues["action"].popleft()

    def predict_action_chunk(self, batch):
        # Pulls obs from the queue (ignores batch values); honors config.n_action_steps
        # like diffusion's generate_actions slice.
        return self._chunk_from_queue(self.config.n_action_steps).unsqueeze(0)  # (1, n, D)


class _StubDataset:
    def __init__(self, n_frames):
        self.n = n_frames

    def __getitem__(self, i):
        return {
            "action": torch.tensor([i * 0.5, i * 0.5], dtype=torch.float64),
            "observation.state": torch.tensor([float(i), float(i)], dtype=torch.float64),
        }


def _make_evaluator():
    anvil_cfg = {"action_type": "absolute"}
    return EpisodeEvaluator(
        model=_StubPolicy(),
        preprocessor=None,
        postprocessor=None,
        model_type="diffusion",
        device="cpu",
        anvil_cfg=anvil_cfg,
        task_description=None,
        joint_names=JOINTS,
    )


def test_capture_produces_anchors_at_native_cadence():
    ev = _make_evaluator()
    n = 16
    res = ev.evaluate_episode(_StubDataset(n), list(range(n)), episode_idx=0, split_label="val")
    sub = res.substrate
    assert sub is not None
    # native n_action_steps = 4 → anchors at frames 0, 4, 8, 12
    assert [c.anchor_frame for c in sub.anchors] == [0, 4, 8, 12]
    # diffusion Hmax = horizon - n_obs + 1 = 10 - 2 + 1 = 9 (trimmed near episode end)
    assert sub.anchors[0].abs_pred.shape == (9, D)
    assert sub.anchors[-1].abs_pred.shape[0] == n - 12  # trimmed


def test_executed_prefix_matches_trajectory():
    """The first executed_len rows of each captured chunk equal the executed trajectory."""
    ev = _make_evaluator()
    n = 16
    res = ev.evaluate_episode(_StubDataset(n), list(range(n)), episode_idx=0, split_label="val")
    for c in res.substrate.anchors:
        k = min(c.executed_len, c.abs_pred.shape[0])
        prefix = c.abs_pred[:k]
        traj = res.predicted[c.anchor_frame : c.anchor_frame + k]
        assert np.allclose(prefix, traj), f"parity mismatch at anchor {c.anchor_frame}"


def test_iter_records_count():
    chunk = AnchorChunk(
        anchor_frame=0, executed_len=2,
        abs_pred=np.zeros((5, D)), abs_gt=np.zeros((5, D)), obs_ref=np.zeros(D),
    )
    ep = EpisodeSubstrate(0, "train", JOINTS, [chunk])
    rows = list(iter_records(ep))
    assert len(rows) == 5 * D
    assert sum(r["executed"] for r in rows) == 2 * D  # offsets 0,1 executed


def test_aggregate_horizon_increases_with_offset(tmp_path: Path):
    # error grows linearly with horizon offset
    anchors = []
    for a in range(0, 12, 4):
        H = 6
        pred = np.zeros((H, D))
        gt = (np.arange(H)[:, None] * 0.01) * np.ones((1, D))
        anchors.append(AnchorChunk(anchor_frame=a, executed_len=4, abs_pred=pred, abs_gt=gt))
    ep = EpisodeSubstrate(0, "val", JOINTS, anchors)
    agg = aggregate_horizon([ep])
    means = agg["val"]["mae_mean"]
    assert means == sorted(means)  # monotonic non-decreasing
    assert agg["val"]["executed_len"] == 4
    write_substrate_csv([ep], tmp_path / "s.csv")
    write_horizon_csv(agg, tmp_path / "h.csv")
    assert (tmp_path / "s.csv").exists() and (tmp_path / "h.csv").exists()


# ── Phase labeler ────────────────────────────────────────────────────────────

PHASE_JOINTS = ["left_joint1", "left_finger_joint1", "right_joint1", "right_finger_joint1"]


def _gt_with_gripper(left_sig, right_sig):
    """Build (T, 4) GT where cols 1 and 3 are the two grippers."""
    T = len(left_sig)
    gt = np.zeros((T, 4))
    gt[:, 1] = left_sig
    gt[:, 3] = right_sig
    return gt


def test_find_gripper_and_arm_indices():
    assert find_gripper_indices(PHASE_JOINTS) == {"left": 1, "right": 3}
    assert arm_joint_indices(PHASE_JOINTS, "left") == [0, 1]
    assert arm_joint_indices(PHASE_JOINTS, "right") == [2, 3]


def test_phases_segment_open_close_both_directions():
    # left: open(0..9) -> closed(10..24) -> open(25..39); right stays open
    left = np.concatenate([np.ones(10) * 1.0, np.zeros(15), np.ones(15) * 1.0])
    right = np.ones(40) * 1.0
    seg = label_phases(_gt_with_gripper(left, right), PHASE_JOINTS, min_segment=3)
    labels = [s[2] for s in seg["left"]]
    # both directions cut → three segments: open#1, closed#1, open#2
    assert labels == ["left:open#1", "left:closed#1", "left:open#2"]
    # boundaries near the transitions
    bounds = [(s[0], s[1]) for s in seg["left"]]
    assert bounds[0][1] == 10 and bounds[1] == (10, 25)
    # right never moves → single nograsp segment
    assert len(seg["right"]) == 1 and "nograsp" in seg["right"][0][2]


def test_phases_debounce_removes_short_blips():
    # a 2-frame dip in an otherwise-open signal should be debounced away (min_segment=5)
    sig = np.ones(30) * 1.0
    sig[14:16] = 0.0
    seg = label_phases(_gt_with_gripper(sig, sig), PHASE_JOINTS, min_segment=5)
    assert len(seg["left"]) == 1  # blip merged → one phase


def test_segments_to_frame_map():
    left = np.concatenate([np.ones(8), np.zeros(8)])
    seg = label_phases(_gt_with_gripper(left, np.ones(16)), PHASE_JOINTS, min_segment=3)
    fmap = segments_to_frame_map(seg["left"])
    assert fmap[0].endswith("open#1") and fmap[15].endswith("closed#1")


def test_segments_to_boundaries():
    segs = [(0, 10, "left:open#1"), (10, 25, "left:closed#1"), (25, 40, "left:open#2")]
    assert segments_to_boundaries(segs) == [(10, "closed"), (25, "open")]  # frame 0 skipped


def test_plot_episode_joints_with_phase_lines(tmp_path):
    T, Dn = 40, 4
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(T, Dn)); gt = pred + 0.05
    m = compute_episode_metrics(pred, gt, PHASE_JOINTS, 0, "val")
    bounds = {"left": [(10, "closed"), (25, "open")], "right": [(15, "closed")]}
    out = tmp_path / "ep.png"
    plot_episode_joints(pred, gt, PHASE_JOINTS, m, out, phase_boundaries=bounds)
    assert out.exists() and out.stat().st_size > 0


def test_plot_phase_mae_timeline(tmp_path):
    T, Dn = 40, 4
    rng = np.random.default_rng(1)
    pred = rng.normal(size=(T, Dn)); gt = pred + 0.05
    bounds = {"left": [(10, "closed"), (25, "open")], "right": [(15, "closed")]}
    out = tmp_path / "ep_phase_mae.png"
    plot_phase_mae_timeline(pred, gt, PHASE_JOINTS, bounds, 0, "val", out)
    assert out.exists() and out.stat().st_size > 0
