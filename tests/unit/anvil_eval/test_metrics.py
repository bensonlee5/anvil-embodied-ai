"""Tests for metrics computation."""

import math

import numpy as np
import pytest

from anvil_eval.metrics import (
    EEMetrics,
    compute_ee_metrics,
    compute_episode_metrics,
    compute_summary_metrics,
)


JOINT_NAMES = ["j1", "j2", "j3"]


def test_perfect_prediction():
    """When predicted == ground_truth, all errors should be 0."""
    gt = np.random.randn(50, 3)
    m = compute_episode_metrics(gt.copy(), gt, JOINT_NAMES, 0, "val")

    assert m.mse == pytest.approx(0.0, abs=1e-10)
    assert m.mae == pytest.approx(0.0, abs=1e-10)
    assert m.rmse == pytest.approx(0.0, abs=1e-10)
    assert m.max_abs_error == pytest.approx(0.0, abs=1e-10)
    assert m.cosine_similarity == pytest.approx(1.0, abs=1e-6)
    for jn in JOINT_NAMES:
        assert m.per_joint_mse[jn] == pytest.approx(0.0, abs=1e-10)
        assert m.per_joint_mae[jn] == pytest.approx(0.0, abs=1e-10)


def test_known_constant_offset():
    """Test with a constant offset to verify MSE/MAE/RMSE formulas."""
    gt = np.ones((10, 3))
    pred = gt + 0.5  # constant offset of 0.5

    m = compute_episode_metrics(pred, gt, JOINT_NAMES, 0, "test")

    assert m.mse == pytest.approx(0.25, abs=1e-8)
    assert m.mae == pytest.approx(0.5, abs=1e-8)
    assert m.rmse == pytest.approx(0.5, abs=1e-8)
    assert m.max_abs_error == pytest.approx(0.5, abs=1e-8)
    assert m.num_frames == 10


def test_per_joint_metrics():
    """Verify per-joint metrics isolate correctly."""
    gt = np.zeros((20, 3))
    pred = np.zeros((20, 3))
    pred[:, 1] = 1.0  # only joint j2 has error

    m = compute_episode_metrics(pred, gt, JOINT_NAMES, 0, "val")

    assert m.per_joint_mae["j1"] == pytest.approx(0.0, abs=1e-10)
    assert m.per_joint_mae["j2"] == pytest.approx(1.0, abs=1e-10)
    assert m.per_joint_mae["j3"] == pytest.approx(0.0, abs=1e-10)
    assert m.max_abs_error_joint == "j2"


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should have cosine similarity ~0."""
    gt = np.zeros((10, 3))
    pred = np.zeros((10, 3))
    gt[:, 0] = 1.0
    pred[:, 1] = 1.0

    m = compute_episode_metrics(pred, gt, JOINT_NAMES, 0, "val")
    assert m.cosine_similarity == pytest.approx(0.0, abs=1e-6)


def test_smoothness_constant():
    """Constant actions should have zero smoothness (no deltas)."""
    actions = np.ones((10, 3)) * 2.0
    m = compute_episode_metrics(actions, actions, JOINT_NAMES, 0, "val")

    assert m.pred_smoothness_mean == pytest.approx(0.0, abs=1e-10)
    assert m.gt_smoothness_mean == pytest.approx(0.0, abs=1e-10)


def test_smoothness_linear():
    """Linearly increasing actions should have constant smoothness."""
    gt = np.zeros((10, 3))
    gt[:, 0] = np.arange(10) * 0.1  # linearly increasing

    m = compute_episode_metrics(gt, gt, JOINT_NAMES, 0, "val")

    # delta = 0.1 for each step, L2 norm = 0.1
    assert m.gt_smoothness_mean == pytest.approx(0.1, abs=1e-6)
    assert m.gt_smoothness_std == pytest.approx(0.0, abs=1e-6)


def test_single_frame():
    """Single frame should not crash on smoothness."""
    gt = np.ones((1, 3))
    m = compute_episode_metrics(gt, gt, JOINT_NAMES, 0, "val")
    assert m.num_frames == 1
    assert m.pred_smoothness_mean == 0.0


def test_summary_metrics():
    """Test summary aggregation across episodes."""
    gt1 = np.zeros((10, 3))
    pred1 = gt1 + 0.1
    gt2 = np.zeros((10, 3))
    pred2 = gt2 + 0.2

    m1 = compute_episode_metrics(pred1, gt1, JOINT_NAMES, 0, "val")
    m2 = compute_episode_metrics(pred2, gt2, JOINT_NAMES, 1, "val")

    summary = compute_summary_metrics([m1, m2])
    assert "val" in summary
    assert summary["val"]["num_episodes"] == 2
    assert summary["val"]["mae_mean"] == pytest.approx(0.15, abs=1e-6)


# =============================================================================
# EE Cartesian metrics
# =============================================================================

def _make_ee_names(arms):
    """Build action feature name list for n arms."""
    names = []
    for arm in arms:
        names += [f"{arm}_x", f"{arm}_y", f"{arm}_z",
                  f"{arm}_r0", f"{arm}_r1", f"{arm}_r2",
                  f"{arm}_r3", f"{arm}_r4", f"{arm}_r5",
                  f"{arm}_gripper"]
    return names


def _identity_rot6d():
    """rot6d for identity rotation: first two cols of I = [1,0,0, 0,1,0]."""
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _rot6d_90z():
    """rot6d for 90° rotation about Z: R = [[0,-1,0],[1,0,0],[0,0,1]]."""
    # First col = [0,1,0], second col = [-1,0,0]  → rot6d = [0,1,0,-1,0,0]
    return np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])


def _make_ee_frame(arms, pos_list, rot6d_list, gripper_list):
    """Build a single (20,) or (10,) action vector for the given arms."""
    parts = []
    for i, _arm in enumerate(arms):
        parts.extend(pos_list[i])
        parts.extend(rot6d_list[i])
        parts.append(gripper_list[i])
    return np.array(parts, dtype=np.float64)


class TestComputeEEMetrics:
    """Tests for compute_ee_metrics and the EE path in compute_episode_metrics."""

    def test_perfect_prediction_left_only(self):
        """When predicted == gt, all EE errors should be ~0."""
        T, arms = 10, ["left"]
        names = _make_ee_names(arms)
        frame = _make_ee_frame(arms,
                               pos_list=[[0.1, 0.2, 0.3]],
                               rot6d_list=[_identity_rot6d()],
                               gripper_list=[0.02])
        gt   = np.tile(frame, (T, 1))
        pred = gt.copy()

        ee = compute_ee_metrics(pred, gt, names)

        assert "left" in ee.position_error_m
        assert ee.position_error_m["left"] == pytest.approx(0.0, abs=1e-9)
        assert ee.orientation_error_rad["left"] == pytest.approx(0.0, abs=1e-9)
        assert ee.gripper_error_m["left"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_prediction_bimanual(self):
        """Bimanual: predicted == gt → all errors zero for both arms."""
        T, arms = 10, ["left", "right"]
        names = _make_ee_names(arms)
        frame = _make_ee_frame(arms,
                               pos_list=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                               rot6d_list=[_identity_rot6d(), _identity_rot6d()],
                               gripper_list=[0.02, 0.03])
        gt   = np.tile(frame, (T, 1))
        pred = gt.copy()

        ee = compute_ee_metrics(pred, gt, names)

        assert ee.position_error_m["left"]  == pytest.approx(0.0, abs=1e-9)
        assert ee.position_error_m["right"] == pytest.approx(0.0, abs=1e-9)
        assert ee.orientation_error_rad["left"]  == pytest.approx(0.0, abs=1e-9)
        assert ee.orientation_error_rad["right"] == pytest.approx(0.0, abs=1e-9)

    def test_known_position_offset(self):
        """1 m offset in x → position error = 1.0 m."""
        T, arms = 5, ["left"]
        names = _make_ee_names(arms)
        gt_frame   = _make_ee_frame(arms, [[0.0, 0.0, 0.0]], [_identity_rot6d()], [0.0])
        pred_frame = _make_ee_frame(arms, [[1.0, 0.0, 0.0]], [_identity_rot6d()], [0.0])
        gt   = np.tile(gt_frame,   (T, 1))
        pred = np.tile(pred_frame, (T, 1))

        ee = compute_ee_metrics(pred, gt, names)
        assert ee.position_error_m["left"] == pytest.approx(1.0, abs=1e-6)

    def test_known_rotation_90_degrees_z(self):
        """90° rotation about Z: geodesic error = π/2 radians."""
        T, arms = 5, ["left"]
        names = _make_ee_names(arms)
        gt_frame   = _make_ee_frame(arms, [[0.0]*3], [_identity_rot6d()], [0.0])
        pred_frame = _make_ee_frame(arms, [[0.0]*3], [_rot6d_90z()],      [0.0])
        gt   = np.tile(gt_frame,   (T, 1))
        pred = np.tile(pred_frame, (T, 1))

        ee = compute_ee_metrics(pred, gt, names)
        assert ee.orientation_error_rad["left"] == pytest.approx(math.pi / 2, abs=1e-5)

    def test_fallback_labels_when_names_empty(self):
        """Empty action_names → arm labels fall back to 'arm0', 'arm1'."""
        T = 5
        gt   = np.zeros((T, 20))
        pred = np.zeros((T, 20))

        ee = compute_ee_metrics(pred, gt, action_names=[])
        assert "arm0" in ee.position_error_m
        assert "arm1" in ee.position_error_m

    def test_ee_path_triggered_in_compute_episode_metrics(self):
        """compute_episode_metrics with action_type='ee_absolute' populates ee field."""
        T, arms = 10, ["left", "right"]
        names = _make_ee_names(arms)
        gt   = np.zeros((T, 20))
        pred = np.zeros((T, 20))

        m = compute_episode_metrics(pred, gt, names, 0, "val", action_type="ee_absolute")
        assert m.ee is not None
        assert "left"  in m.ee.position_error_m
        assert "right" in m.ee.position_error_m

    def test_ee_path_not_triggered_for_absolute(self):
        """compute_episode_metrics with default action_type='absolute' → ee is None."""
        gt   = np.zeros((10, 20))
        pred = np.zeros((10, 20))
        m = compute_episode_metrics(pred, gt, [], 0, "val")
        assert m.ee is None

    def test_ee_metrics_properties(self):
        """position_pass / orientation_pass work as properties (no threshold arg)."""
        ee = EEMetrics(
            position_error_m={"left": 0.01},
            orientation_error_rad={"left": 0.05},
            gripper_error_m={"left": 0.001},
        )
        assert ee.position_pass is True
        assert ee.orientation_pass is True

        ee_fail = EEMetrics(
            position_error_m={"left": 0.05},
            orientation_error_rad={"left": 0.2},
            gripper_error_m={"left": 0.01},
        )
        assert ee_fail.position_pass is False
        assert ee_fail.orientation_pass is False
