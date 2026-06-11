"""Tests for anvil_shared.ee_transform.

Covers:
  1. n_arms_from_dims — valid single/bimanual, invalid state dim, invalid action dim
  2. ee_rel_forward / ee_rel_inverse round-trip (single arm, bimanual)
  3. ee_rel_forward single reference state vs per-sample state agreement
  4. Identity: zero rotation → forward returns zero rotation delta
  5. ee_action_to_poses layout: pos, quat_xyzw shape, gripper value
  6. Vectorised forward matches per-sample single calls
"""
from __future__ import annotations

import numpy as np
import pytest

from anvil_shared.ee_transform import (
    EE_ACTION_DIM_PER_ARM,
    EE_STATE_DIM_PER_ARM,
    ee_action_to_poses,
    ee_rel_forward,
    ee_rel_inverse,
    n_arms_from_dims,
)
from anvil_shared.rotation import matrix_to_quat, rot6d_to_matrix


# ── helpers ───────────────────────────────────────────────────────────────────

def _identity_state(n_arms: int = 1) -> np.ndarray:
    """EE state with identity rotation and zero position for each arm."""
    state = np.zeros(EE_STATE_DIM_PER_ARM * n_arms)
    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        # qx=0, qy=0, qz=0, qw=1  (identity quaternion)
        state[s0 + 6] = 1.0
    return state


def _identity_action(n_arms: int = 1) -> np.ndarray:
    """EE action with identity rotation and zero position for each arm."""
    action = np.zeros(EE_ACTION_DIM_PER_ARM * n_arms)
    for arm in range(n_arms):
        a0 = arm * EE_ACTION_DIM_PER_ARM
        # rot6d identity: R[:,0]=[1,0,0], R[:,1]=[0,1,0]  → [1,0,0,0,1,0]
        action[a0 + 3] = 1.0  # r0
        action[a0 + 7] = 1.0  # r4
    return action


def _random_action(rng: np.random.Generator, n_arms: int = 1) -> np.ndarray:
    """Random EE action with valid rot6d rotation per arm."""
    action = np.zeros(EE_ACTION_DIM_PER_ARM * n_arms)
    for arm in range(n_arms):
        a0 = arm * EE_ACTION_DIM_PER_ARM
        action[a0:a0 + 3] = rng.uniform(-0.5, 0.5, 3)  # xyz
        # Random rotation via QR decomposition → valid SO(3)
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        # rot6d = first two columns, column-major [Q[:,0], Q[:,1]]
        action[a0 + 3:a0 + 9] = np.concatenate([Q[:, 0], Q[:, 1]])
        action[a0 + 9] = rng.uniform(0.0, 0.05)  # gripper
    return action


def _random_state(rng: np.random.Generator, n_arms: int = 1) -> np.ndarray:
    """Random EE state with valid quaternion per arm."""
    state = np.zeros(EE_STATE_DIM_PER_ARM * n_arms)
    for arm in range(n_arms):
        s0 = arm * EE_STATE_DIM_PER_ARM
        state[s0:s0 + 3] = rng.uniform(-0.5, 0.5, 3)  # xyz
        q = rng.standard_normal(4)
        state[s0 + 3:s0 + 7] = q / np.linalg.norm(q)  # unit quaternion [qx,qy,qz,qw]
        state[s0 + 7] = rng.uniform(0.0, 0.05)  # gripper (state)
    return state


# ── 1. n_arms_from_dims ───────────────────────────────────────────────────────

class TestNArmsFromDims:
    def test_single_arm(self):
        assert n_arms_from_dims(8, 10) == 1

    def test_bimanual(self):
        assert n_arms_from_dims(16, 20) == 2

    def test_invalid_state_dim_zero(self):
        with pytest.raises(ValueError, match="positive multiple"):
            n_arms_from_dims(0, 10)

    def test_invalid_state_dim_not_multiple(self):
        with pytest.raises(ValueError, match="positive multiple"):
            n_arms_from_dims(9, 10)

    def test_invalid_action_dim(self):
        with pytest.raises(ValueError, match="action dim"):
            n_arms_from_dims(8, 12)  # should be 10

    def test_invalid_action_dim_bimanual(self):
        with pytest.raises(ValueError, match="action dim"):
            n_arms_from_dims(16, 10)  # should be 20


# ── 2. Round-trip: forward → inverse ─────────────────────────────────────────

class TestRoundTrip:
    def _check_round_trip(self, n_arms: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        action_abs = _random_action(rng, n_arms)
        state = _random_state(rng, n_arms)

        action_rel = ee_rel_forward(action_abs, state)
        recovered = ee_rel_inverse(action_rel, state)

        np.testing.assert_allclose(
            recovered, action_abs, atol=1e-10,
            err_msg=f"Round-trip failed for n_arms={n_arms}"
        )

    def test_single_arm(self):
        self._check_round_trip(n_arms=1)

    def test_bimanual(self):
        self._check_round_trip(n_arms=2)

    def test_batched_chunk(self):
        """Batch of T=8 steps with a single reference state."""
        rng = np.random.default_rng(99)
        T = 8
        n_arms = 1
        state = _random_state(rng, n_arms)  # (8,) single ref
        actions_abs = np.stack([_random_action(rng, n_arms) for _ in range(T)])  # (T, 10)

        actions_rel = ee_rel_forward(actions_abs, state)
        recovered = ee_rel_inverse(actions_rel, state)

        np.testing.assert_allclose(recovered, actions_abs, atol=1e-10)

    def test_gripper_preserved(self):
        """Gripper stays absolute through forward → inverse."""
        rng = np.random.default_rng(7)
        action_abs = _random_action(rng)
        state = _random_state(rng)
        # The gripper value in the action
        gripper_orig = float(action_abs[9])

        action_rel = ee_rel_forward(action_abs, state)
        # Forward: gripper in rel is still absolute (copied unchanged)
        assert abs(float(action_rel[9]) - gripper_orig) < 1e-12

        recovered = ee_rel_inverse(action_rel, state)
        assert abs(float(recovered[9]) - gripper_orig) < 1e-12


# ── 3. Identity state: zero delta for zero-offset action ─────────────────────

class TestIdentity:
    def test_identity_state_identity_action_zero_delta(self):
        """With identity rotation and zero position, forward → zero xyz delta."""
        state = _identity_state()
        action = _identity_action()
        action[:3] = 0.0  # zero position

        rel = ee_rel_forward(action, state)
        # xyz delta should be zero
        np.testing.assert_allclose(rel[:3], 0.0, atol=1e-12)
        # rot6d delta should be identity rot6d (relative rotation = I)
        expected_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(rel[3:9], expected_rot6d, atol=1e-12)


# ── 4. Single vs per-sample state agreement ───────────────────────────────────

class TestSingleVsPerSampleState:
    def test_forward_agrees(self):
        """Single ref state (8,) and per-sample state (T,8) give same result."""
        rng = np.random.default_rng(11)
        T = 5
        state = _random_state(rng)  # (8,) single ref
        actions = np.stack([_random_action(rng) for _ in range(T)])  # (T,10)

        # Single ref broadcast
        rel_single = ee_rel_forward(actions, state)

        # Per-sample: broadcast state to (T, 8)
        states_per_sample = np.tile(state, (T, 1))  # (T, 8)
        rel_per_sample = ee_rel_forward(actions, states_per_sample)

        np.testing.assert_allclose(rel_single, rel_per_sample, atol=1e-12)

    def test_inverse_agrees(self):
        """Same as forward but for inverse."""
        rng = np.random.default_rng(13)
        T = 5
        state = _random_state(rng)
        rels = np.stack([_random_action(rng) for _ in range(T)])

        abs_single = ee_rel_inverse(rels, state)
        states_per_sample = np.tile(state, (T, 1))
        abs_per_sample = ee_rel_inverse(rels, states_per_sample)

        np.testing.assert_allclose(abs_single, abs_per_sample, atol=1e-12)


# ── 5. ee_action_to_poses ─────────────────────────────────────────────────────

class TestEeActionToPoses:
    def test_single_step_single_arm(self):
        action = _identity_action(n_arms=1)
        action[:3] = [0.1, 0.2, 0.3]
        action[9] = 0.03

        poses = ee_action_to_poses(action, n_arms=1)
        assert len(poses) == 1
        step = poses[0]
        assert 0 in step
        assert step[0]["pos"].shape == (3,)
        np.testing.assert_allclose(step[0]["pos"], [0.1, 0.2, 0.3], atol=1e-12)
        assert step[0]["quat_xyzw"].shape == (4,)
        assert abs(step[0]["gripper"] - 0.03) < 1e-12

    def test_chunk_shape(self):
        rng = np.random.default_rng(21)
        T = 4
        n_arms = 2
        actions = np.stack([_random_action(rng, n_arms) for _ in range(T)])
        poses = ee_action_to_poses(actions, n_arms=n_arms)

        assert len(poses) == T
        for step in poses:
            assert set(step.keys()) == {0, 1}
            for arm in range(n_arms):
                assert step[arm]["pos"].shape == (3,)
                assert step[arm]["quat_xyzw"].shape == (4,)
                # Quaternion must be unit
                q = step[arm]["quat_xyzw"]
                np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)

    def test_n_arms_inferred(self):
        """n_arms=None → inferred from action dim."""
        action = _identity_action(n_arms=2)
        poses = ee_action_to_poses(action)  # n_arms defaults to None
        assert len(poses) == 1
        assert 0 in poses[0] and 1 in poses[0]

    def test_identity_action_gives_identity_quat(self):
        """Identity rot6d in action → quaternion should be [0,0,0,1]."""
        action = _identity_action(n_arms=1)
        poses = ee_action_to_poses(action, n_arms=1)
        q = poses[0][0]["quat_xyzw"]
        # Identity rotation: [qx,qy,qz,qw] = [0,0,0,±1]
        assert abs(abs(q[3]) - 1.0) < 1e-10, f"Expected |qw|=1, got {q}"
