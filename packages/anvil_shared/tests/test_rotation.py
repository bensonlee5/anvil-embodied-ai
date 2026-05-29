"""Tests for anvil_shared.rotation helpers."""
from __future__ import annotations

import numpy as np
import pytest

from anvil_shared.rotation import (
    matrix_to_quat,
    matrix_to_rot6d,
    quat_to_matrix,
    rot6d_to_matrix,
)


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(32):
        q = _random_quat(rng)
        R = quat_to_matrix(q)
        # Orthonormal: R R^T = I, det = +1
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
        q2 = matrix_to_quat(R)
        # Quaternions are double cover; align sign before comparing
        if np.dot(q, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q, q2, atol=1e-10)


def test_matrix_rot6d_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(32):
        q = _random_quat(rng)
        R = quat_to_matrix(q)
        r6 = matrix_to_rot6d(R)
        assert r6.shape == (6,)
        # The 6-vector is the first two columns of R flattened column-major.
        np.testing.assert_allclose(r6[:3], R[:, 0], atol=1e-12)
        np.testing.assert_allclose(r6[3:], R[:, 1], atol=1e-12)
        R2 = rot6d_to_matrix(r6)
        np.testing.assert_allclose(R, R2, atol=1e-10)


def test_rot6d_gram_schmidt_handles_unnormalized_input():
    # rot6d_to_matrix must orthonormalize, so non-orthonormal 6-vectors still
    # produce a valid rotation matrix.
    r6 = np.array([2.0, 0.0, 0.0, 0.1, 1.0, 0.0])  # a1 not unit, a2 not orthogonal
    R = rot6d_to_matrix(r6)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_quat_to_matrix_normalises_input():
    # Non-unit quaternion should still produce an orthonormal matrix.
    q = np.array([1.0, 2.0, 3.0, 4.0])
    R = quat_to_matrix(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_shape_validation():
    with pytest.raises(ValueError):
        quat_to_matrix([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        matrix_to_quat(np.eye(2))
    with pytest.raises(ValueError):
        matrix_to_rot6d(np.eye(2))
    with pytest.raises(ValueError):
        rot6d_to_matrix(np.zeros(5))


def test_rot6d_to_matrix_raises_on_zero_a1():
    """Near-zero first column must raise ValueError, not produce nan."""
    with pytest.raises(ValueError, match="near-zero"):
        rot6d_to_matrix(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_rot6d_to_matrix_raises_on_parallel_columns():
    """a1 and a2 parallel → b2 near-zero after projection."""
    with pytest.raises(ValueError, match="near-zero|degenerate"):
        rot6d_to_matrix(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))


def test_known_value_90_degrees_z():
    """90° rotation about Z: R = [[0,-1,0],[1,0,0],[0,0,1]]."""
    s = np.sin(np.pi / 4)
    quat_90z = np.array([0.0, 0.0, s, s])
    R = quat_to_matrix(quat_90z)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    np.testing.assert_allclose(R, expected, atol=1e-10)
    # Round-trip through rot6d
    r6d = matrix_to_rot6d(R)
    R2 = rot6d_to_matrix(r6d)
    np.testing.assert_allclose(R, R2, atol=1e-10)


def test_known_value_180_degrees_x():
    """180° about X: R = diag(1,-1,-1). Hits non-trace branch of Shepperd."""
    R_180x = np.diag([1.0, -1.0, -1.0])
    q = matrix_to_quat(R_180x)
    R_rt = quat_to_matrix(q)
    np.testing.assert_allclose(R_180x, R_rt, atol=1e-10)


def test_matrix_to_quat_all_shepperd_branches():
    """Exercise all four Shepperd branches with 90° rotations about each axis."""
    rng = np.random.default_rng(42)
    for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        # random angle in (30°, 150°) to avoid near-identity (trace>0 branch only)
        angle = rng.uniform(np.pi / 6, 5 * np.pi / 6)
        ax = np.array(axis, dtype=float)
        s, c = np.sin(angle / 2), np.cos(angle / 2)
        q = np.array([ax[0] * s, ax[1] * s, ax[2] * s, c])
        R = quat_to_matrix(q)
        q2 = matrix_to_quat(R)
        if np.dot(q, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q, q2, atol=1e-10)
    with pytest.raises(ValueError):
        matrix_to_rot6d(np.eye(2))
    with pytest.raises(ValueError):
        rot6d_to_matrix(np.zeros(5))
