"""
Tests for DeltaActionTransform and TransformRunner._compute_delta_action_stats.

Covers:
  1. DeltaActionTransform.is_enabled() reflects config.use_delta_actions
  2. apply() is a no-op when action or observation.state is missing
  3. Basic delta: action - state for single-step state, single-joint action
  4. Multi-step state: state[..., -1, :] is used as reference
  5. Shape mismatch without info.json raises DataIntegrityError
  6. delta_exclude_joints: excluded joints keep absolute values
  7. TransformRunner._compute_delta_action_stats returns None when inactive
  8. TransformRunner._compute_delta_action_stats: delta_mean ≈ 0 for slow-moving
     synthetic data; excluded joints keep absolute stats
  9. Action joint missing from observation.state raises DataIntegrityError
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from anvil_trainer.transforms import DataIntegrityError, EEDeltaTransform
from anvil_trainer.train import (
    DeltaActionTransform,
    TrainingConfig,
    TransformRunner,
)


# =============================================================================
# 1. is_enabled reflects config
# =============================================================================

class TestIsEnabled:
    def test_disabled_by_default(self):
        cfg = TrainingConfig()
        assert DeltaActionTransform().is_enabled(cfg) is False

    def test_enabled_when_flag_set(self):
        cfg = TrainingConfig(action_type="delta_obs_t")
        assert DeltaActionTransform().is_enabled(cfg) is True


# =============================================================================
# 2. apply() is a no-op when keys are missing
# =============================================================================

class TestApplyNoop:
    def test_no_action_key(self):
        cfg = TrainingConfig(action_type="delta_obs_t")
        item = {"observation.state": torch.tensor([0.1, 0.2])}
        out = DeltaActionTransform().apply(item, cfg)
        assert out is item  # unchanged reference
        assert "action" not in out

    def test_no_observation_state_key(self):
        cfg = TrainingConfig(action_type="delta_obs_t")
        item = {"action": torch.tensor([0.1, 0.2])}
        out = DeltaActionTransform().apply(item, cfg)
        assert torch.equal(out["action"], torch.tensor([0.1, 0.2]))


# =============================================================================
# 3. Basic delta computation
# =============================================================================

class TestBasicDelta:
    def test_single_frame_same_shape(self):
        """action and state both [n_joints] → delta = action - state."""
        cfg = TrainingConfig(action_type="delta_obs_t")
        item = {
            "action": torch.tensor([1.0, 2.0, 3.0]),
            "observation.state": torch.tensor([0.5, 1.0, 2.5]),
        }
        out = DeltaActionTransform().apply(item, cfg)
        expected = torch.tensor([0.5, 1.0, 0.5])
        assert torch.allclose(out["action"], expected)

    def test_action_chunk_broadcasts(self):
        """action [horizon, n_joints], state [n_joints] → state broadcasts."""
        cfg = TrainingConfig(action_type="delta_obs_t")
        item = {
            "action": torch.tensor([[1.0, 2.0], [1.1, 2.2], [0.9, 1.8]]),
            "observation.state": torch.tensor([1.0, 2.0]),
        }
        out = DeltaActionTransform().apply(item, cfg)
        expected = torch.tensor([[0.0, 0.0], [0.1, 0.2], [-0.1, -0.2]])
        assert torch.allclose(out["action"], expected)


# =============================================================================
# 4. Multi-step state — state[-1] is the reference
# =============================================================================

class TestMultiStepState:
    def test_state_last_step_used(self):
        """state shape [n_obs_steps, n_joints] → use state[-1] as reference."""
        cfg = TrainingConfig(action_type="delta_obs_t")
        item = {
            "action": torch.tensor([5.0, 10.0]),
            # Two obs steps: [prev, current]
            "observation.state": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
        out = DeltaActionTransform().apply(item, cfg)
        # Uses current (last) state: 5-3=2, 10-4=6
        expected = torch.tensor([2.0, 6.0])
        assert torch.allclose(out["action"], expected)


# =============================================================================
# 5. Shape mismatch raises DataIntegrityError
# =============================================================================

class TestShapeMismatch:
    def test_mismatch_without_info_json_raises(self):
        """action [3] vs state [2] with no info.json → DataIntegrityError, not silent fallback."""
        cfg = TrainingConfig(action_type="delta_obs_t")  # no dataset_root
        item = {
            "action": torch.tensor([1.0, 2.0, 9.9]),
            "observation.state": torch.tensor([0.5, 1.0]),
        }
        with pytest.raises(DataIntegrityError, match="observation.state"):
            DeltaActionTransform().apply(item, cfg)

    def test_obs_wider_than_action_is_ok(self, tmp_path):
        """obs.state may have more joints than action (obs → action match not required)."""
        meta = tmp_path / "meta"
        meta.mkdir(parents=True)
        info = {
            "features": {
                "action": {"names": ["j0", "j1"]},
                "observation.state": {"names": ["j0", "j1", "velocity"]},
            }
        }
        (meta / "info.json").write_text(json.dumps(info))
        cfg = TrainingConfig(action_type="delta_obs_t", dataset_root=str(tmp_path))
        item = {
            "action": torch.tensor([1.0, 2.0]),
            "observation.state": torch.tensor([0.5, 1.0, 99.0]),  # extra velocity joint
        }
        out = DeltaActionTransform().apply(item, cfg)
        expected = torch.tensor([0.5, 1.0])
        assert torch.allclose(out["action"], expected)

    def test_action_joint_missing_from_state_raises(self, tmp_path):
        """action joint not in observation.state (and not excluded) → DataIntegrityError."""
        meta = tmp_path / "meta"
        meta.mkdir(parents=True)
        info = {
            "features": {
                "action": {"names": ["j0", "j1", "gripper"]},
                "observation.state": {"names": ["j0", "j1"]},  # gripper missing
            }
        }
        (meta / "info.json").write_text(json.dumps(info))
        cfg = TrainingConfig(action_type="delta_obs_t", dataset_root=str(tmp_path))
        item = {
            "action": torch.tensor([1.0, 2.0, 0.9]),
            "observation.state": torch.tensor([0.5, 1.0]),
        }
        with pytest.raises(DataIntegrityError, match="gripper"):
            DeltaActionTransform().apply(item, cfg)

    def test_action_joint_missing_but_excluded_is_ok(self, tmp_path):
        """action joint missing from state but listed in delta_exclude_joints → no error."""
        meta = tmp_path / "meta"
        meta.mkdir(parents=True)
        info = {
            "features": {
                "action": {"names": ["j0", "j1", "gripper"]},
                "observation.state": {"names": ["j0", "j1"]},
            }
        }
        (meta / "info.json").write_text(json.dumps(info))
        cfg = TrainingConfig(
            action_type="delta_obs_t",
            dataset_root=str(tmp_path),
            delta_exclude_joints=["gripper"],
        )
        item = {
            "action": torch.tensor([1.0, 2.0, 0.9]),
            "observation.state": torch.tensor([0.5, 1.0]),
        }
        out = DeltaActionTransform().apply(item, cfg)
        # j0: 1.0-0.5=0.5, j1: 2.0-1.0=1.0, gripper: stays 0.9
        expected = torch.tensor([0.5, 1.0, 0.9])
        assert torch.allclose(out["action"], expected)


# =============================================================================
# 6. delta_exclude_joints — excluded joints keep absolute values
# =============================================================================

class TestExcludeJoints:
    def _make_dataset_root(self, tmp_path: Path, joint_names: list[str]) -> Path:
        """Create a fake dataset root with meta/info.json for name→index lookup."""
        meta = tmp_path / "meta"
        meta.mkdir(parents=True)
        info = {"features": {"action": {"names": joint_names}}}
        (meta / "info.json").write_text(json.dumps(info))
        return tmp_path

    def test_single_excluded_joint(self, tmp_path):
        """delta_exclude_joints=['gripper'] → gripper keeps absolute value."""
        root = self._make_dataset_root(tmp_path, ["shoulder", "elbow", "gripper"])
        cfg = TrainingConfig(
            action_type="delta_obs_t",
            delta_exclude_joints=["gripper"],
            dataset_root=str(root),
        )
        item = {
            "action": torch.tensor([1.0, 2.0, 0.9]),
            "observation.state": torch.tensor([0.5, 1.0, 0.7]),
        }
        out = DeltaActionTransform().apply(item, cfg)
        # shoulder: 1.0 - 0.5 = 0.5
        # elbow:    2.0 - 1.0 = 1.0
        # gripper:  stays 0.9 (absolute)
        expected = torch.tensor([0.5, 1.0, 0.9])
        assert torch.allclose(out["action"], expected)

    def test_excluded_joint_in_action_chunk(self, tmp_path):
        """Excluded joint stays absolute across a multi-step action chunk."""
        root = self._make_dataset_root(tmp_path, ["j0", "j1"])
        cfg = TrainingConfig(
            action_type="delta_obs_t",
            delta_exclude_joints=["j1"],
            dataset_root=str(root),
        )
        item = {
            "action": torch.tensor([[1.0, 0.9], [1.5, 0.8], [2.0, 0.7]]),
            "observation.state": torch.tensor([0.5, 1.0]),
        }
        out = DeltaActionTransform().apply(item, cfg)
        # j0 deltas: [0.5, 1.0, 1.5]
        # j1 absolute: [0.9, 0.8, 0.7]
        expected = torch.tensor([[0.5, 0.9], [1.0, 0.8], [1.5, 0.7]])
        assert torch.allclose(out["action"], expected)

    def test_unknown_excluded_joint_silently_skipped(self, tmp_path):
        """Unknown joint name in delta_exclude_joints is logged but doesn't raise."""
        root = self._make_dataset_root(tmp_path, ["j0", "j1"])
        cfg = TrainingConfig(
            action_type="delta_obs_t",
            delta_exclude_joints=["nonexistent"],
            dataset_root=str(root),
        )
        item = {
            "action": torch.tensor([1.0, 2.0]),
            "observation.state": torch.tensor([0.5, 1.0]),
        }
        out = DeltaActionTransform().apply(item, cfg)
        # No exclusion took effect — everything becomes delta
        assert torch.allclose(out["action"], torch.tensor([0.5, 1.0]))


# =============================================================================
# 7 & 8. TransformRunner._compute_delta_action_stats
# =============================================================================

class TestComputeDeltaStats:
    def _make_fake_dataset(self, actions: np.ndarray, states: np.ndarray,
                          abs_stats: dict | None = None) -> MagicMock:
        """Build a mock LeRobotDataset with hf_dataset columns and meta.stats."""
        ds = MagicMock()
        n = len(actions)
        ds.hf_dataset = {
            "action": actions,
            "observation.state": states,
            "episode_index": np.zeros(n, dtype=np.int64),
        }
        ds.meta = MagicMock()
        ds.meta.stats = {"action": abs_stats or {}}
        return ds

    def test_returns_none_when_delta_inactive(self):
        cfg = TrainingConfig(action_type="absolute")
        runner = TransformRunner(cfg)
        ds = self._make_fake_dataset(np.zeros((10, 3)), np.zeros((10, 3)))
        assert runner._compute_delta_action_stats(ds) is None

    def test_delta_mean_near_zero_for_slow_motion(self):
        """When action closely tracks state (slow motion), delta_mean ≈ 0."""
        cfg = TrainingConfig(action_type="delta_obs_t")
        runner = TransformRunner(cfg)
        # Actions ≈ states with small offset (delta ≈ 0.01)
        rng = np.random.default_rng(0)
        states = rng.normal(loc=1.5, scale=0.5, size=(200, 3))
        actions = states + 0.01  # delta = 0.01 for all joints
        ds = self._make_fake_dataset(
            actions, states,
            abs_stats={"mean": actions.mean(axis=0).tolist(), "std": actions.std(axis=0).tolist(),
                       "min": actions.min(axis=0).tolist(), "max": actions.max(axis=0).tolist(),
                       "count": [200]},
        )
        stats = runner._compute_delta_action_stats(ds)
        assert stats is not None
        # delta_mean should be ≈ 0.01 (far from absolute mean 1.5)
        assert all(abs(m - 0.01) < 1e-6 for m in stats["mean"])
        # delta_std should be ≈ 0 (all deltas identical) → clamped to 1e-6
        assert all(s >= 1e-6 for s in stats["std"])
        # Patched in place on full_dataset
        assert ds.meta.stats["action"] is stats

    def test_excluded_joint_keeps_absolute_stats(self, tmp_path):
        """Excluded joint's mean/std in patched stats match original absolute stats."""
        # Create fake dataset_root with meta/info.json so _resolve_exclude_indices works
        meta = tmp_path / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps(
            {"features": {"action": {"names": ["j0", "j1", "gripper"]}}}
        ))
        cfg = TrainingConfig(
            action_type="delta_obs_t",
            delta_exclude_joints=["gripper"],
            dataset_root=str(tmp_path),
        )
        runner = TransformRunner(cfg)
        rng = np.random.default_rng(1)
        states = rng.normal(loc=1.0, scale=0.3, size=(150, 3))
        actions = states.copy()
        actions[:, :2] += 0.02       # j0, j1 delta = 0.02
        actions[:, 2] = 0.9          # gripper stays absolute
        abs_mean = actions.mean(axis=0).tolist()
        abs_std = actions.std(axis=0).tolist()
        ds = self._make_fake_dataset(
            actions, states,
            abs_stats={"mean": abs_mean, "std": abs_std,
                       "min": actions.min(axis=0).tolist(),
                       "max": actions.max(axis=0).tolist(),
                       "count": [150]},
        )
        stats = runner._compute_delta_action_stats(ds)
        assert stats is not None
        # j0, j1: delta mean ≈ 0.02
        assert abs(stats["mean"][0] - 0.02) < 1e-6
        assert abs(stats["mean"][1] - 0.02) < 1e-6
        # gripper (excluded): mean matches absolute mean
        assert abs(stats["mean"][2] - abs_mean[2]) < 1e-9
        assert abs(stats["std"][2] - abs_std[2]) < 1e-9

    def test_handles_stacked_observation_state(self):
        """observation.state shaped (N, n_obs_steps, D) uses the last step."""
        cfg = TrainingConfig(action_type="delta_obs_t")
        runner = TransformRunner(cfg)
        # Two obs steps; only the last one should be used
        states_2step = np.zeros((50, 2, 3))
        states_2step[:, 0, :] = 9.9        # ignored prev step
        states_2step[:, 1, :] = 1.0        # current step (reference)
        actions = np.full((50, 3), 1.05)  # delta = 0.05 relative to current
        ds = self._make_fake_dataset(actions, states_2step)
        stats = runner._compute_delta_action_stats(ds)
        assert stats is not None
        assert all(abs(m - 0.05) < 1e-9 for m in stats["mean"])

    def test_failure_returns_none(self):
        """Broken hf_dataset → warning + None, never raises."""
        cfg = TrainingConfig(action_type="delta_obs_t")
        runner = TransformRunner(cfg)
        ds = MagicMock()
        # Missing observation.state raises on access
        ds.hf_dataset = {"action": np.zeros((5, 2))}  # observation.state missing
        ds.meta = MagicMock()
        ds.meta.stats = {"action": {}}
        assert runner._compute_delta_action_stats(ds) is None

    def test_data_integrity_error_propagates(self, tmp_path):
        """DataIntegrityError from name mismatch is NOT swallowed by the try/except."""
        meta = tmp_path / "meta"
        meta.mkdir(parents=True)
        # action has 'missing_joint' which is absent from observation.state
        (meta / "info.json").write_text(json.dumps({
            "features": {
                "action": {"names": ["j0", "missing_joint"]},
                "observation.state": {"names": ["j0"]},
            }
        }))
        cfg = TrainingConfig(action_type="delta_obs_t", dataset_root=str(tmp_path))
        runner = TransformRunner(cfg)
        ds = self._make_fake_dataset(np.zeros((10, 2)), np.zeros((10, 1)))
        with pytest.raises(DataIntegrityError, match="missing_joint"):
            runner._compute_delta_action_stats(ds)


# =============================================================================
# EEDeltaTransform tests
# =============================================================================


def _identity_rot6d_t():
    """Torch tensor: rot6d for identity rotation [1,0,0, 0,1,0]."""
    return torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float32)


def _identity_quat_t():
    """Torch tensor: identity quaternion [qx,qy,qz,qw]."""
    return torch.tensor([0., 0., 0., 1.], dtype=torch.float32)


def _make_ee_item(n_arms=1, horizon=None):
    """Build a dataset item with EE state+action layout.

    state: (8*n_arms,)  [xyz, quat, gripper] per arm — identity pose
    action: (horizon, 10*n_arms) or (10*n_arms,) — identity rot6d pose
    """
    def arm_state():
        return torch.cat([torch.zeros(3), _identity_quat_t(), torch.tensor([0.05])])

    def arm_action():
        return torch.cat([torch.zeros(3), _identity_rot6d_t(), torch.tensor([0.05])])

    state = torch.cat([arm_state() for _ in range(n_arms)])
    if horizon is not None:
        action = torch.stack([torch.cat([arm_action() for _ in range(n_arms)])
                               for _ in range(horizon)])
    else:
        action = torch.cat([arm_action() for _ in range(n_arms)])
    return {"observation.state": state, "action": action}


class TestEEDeltaTransform:
    """Tests for EEDeltaTransform (SE(3) relative EE actions)."""

    def test_is_enabled_only_for_ee_delta(self):
        assert EEDeltaTransform().is_enabled(TrainingConfig(action_type="ee_delta"))
        assert not EEDeltaTransform().is_enabled(TrainingConfig(action_type="ee_absolute"))
        assert not EEDeltaTransform().is_enabled(TrainingConfig(action_type="absolute"))
        assert not EEDeltaTransform().is_enabled(TrainingConfig(action_type="delta_obs_t"))

    def test_noop_when_keys_missing(self):
        cfg = TrainingConfig(action_type="ee_delta")
        t = EEDeltaTransform()
        item_no_action = {"observation.state": torch.zeros(8)}
        item_no_state  = {"action": torch.zeros(10)}
        assert t.apply(item_no_action, cfg) == item_no_action
        assert t.apply(item_no_state,  cfg) == item_no_state

    def test_identity_pose_gives_zero_delta_xyz_and_identity_rot6d(self):
        """Identity pose → delta_xyz=0, delta_rot6d=identity [1,0,0,0,1,0]."""
        cfg = TrainingConfig(action_type="ee_delta")
        item = _make_ee_item(n_arms=1, horizon=None)
        result = EEDeltaTransform().apply(item, cfg)
        delta = result["action"].numpy()
        # xyz delta should be 0
        np.testing.assert_allclose(delta[:3], 0.0, atol=1e-6)
        # rot6d delta should be identity (R_state.T @ R_action = I → rot6d(I) = [1,0,0,0,1,0])
        np.testing.assert_allclose(delta[3:9], [1., 0., 0., 0., 1., 0.], atol=1e-6)
        # gripper kept absolute
        assert delta[9] == pytest.approx(0.05, abs=1e-6)

    def test_gripper_is_absolute_not_delta(self):
        """Gripper must pass through unchanged, not subtracted from state gripper."""
        cfg = TrainingConfig(action_type="ee_delta")
        item = _make_ee_item(n_arms=1)
        item["action"][9] = 0.03   # action gripper
        item["observation.state"][7] = 0.01  # state gripper (different)
        result = EEDeltaTransform().apply(item, cfg)
        # delta gripper == action gripper (absolute), NOT 0.03 - 0.01
        assert result["action"][9].item() == pytest.approx(0.03, abs=1e-6)

    def test_handles_chunk_2d_action(self):
        """2-D (horizon, 10*n_arms) action is correctly processed per step."""
        cfg = TrainingConfig(action_type="ee_delta")
        item = _make_ee_item(n_arms=1, horizon=4)
        result = EEDeltaTransform().apply(item, cfg)
        assert result["action"].shape == (4, 10)
        # All steps with identity pose → same delta
        for k in range(4):
            np.testing.assert_allclose(result["action"][k, :3].numpy(), 0.0, atol=1e-6)
            np.testing.assert_allclose(
                result["action"][k, 3:9].numpy(), [1., 0., 0., 0., 1., 0.], atol=1e-6
            )

    def test_bimanual_both_arms_processed(self):
        """Both arms in a bimanual item get their deltas computed."""
        cfg = TrainingConfig(action_type="ee_delta")
        item = _make_ee_item(n_arms=2)
        result = EEDeltaTransform().apply(item, cfg)
        delta = result["action"].numpy()
        assert delta.shape == (20,)
        # arm 0 (indices 0-9)
        np.testing.assert_allclose(delta[0:3],  0.0, atol=1e-6)
        np.testing.assert_allclose(delta[3:9],  [1., 0., 0., 0., 1., 0.], atol=1e-6)
        # arm 1 (indices 10-19)
        np.testing.assert_allclose(delta[10:13], 0.0, atol=1e-6)
        np.testing.assert_allclose(delta[13:19], [1., 0., 0., 0., 1., 0.], atol=1e-6)

    def test_uses_last_obs_step_for_multi_step_state(self):
        """When state is (n_obs_steps, 8), use state[-1] as reference."""
        cfg = TrainingConfig(action_type="ee_delta")
        item = _make_ee_item(n_arms=1)
        # Stack state as 2-step obs; last step is identity, first has non-zero xyz
        wrong_state = torch.cat([torch.tensor([9., 9., 9.]), _identity_quat_t(), torch.tensor([0.05])])
        right_state = item["observation.state"].clone()
        item["observation.state"] = torch.stack([wrong_state, right_state])  # (2, 8)
        result = EEDeltaTransform().apply(item, cfg)
        # Should use right_state (last), so delta_xyz = 0
        np.testing.assert_allclose(result["action"][:3].numpy(), 0.0, atol=1e-6)

    def test_shape_mismatch_raises_data_integrity_error(self):
        cfg = TrainingConfig(action_type="ee_delta")
        item = {"observation.state": torch.zeros(9), "action": torch.zeros(10)}
        with pytest.raises(DataIntegrityError, match="multiple of 8"):
            EEDeltaTransform().apply(item, cfg)
