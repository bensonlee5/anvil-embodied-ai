"""Rotation utilities for EE Cartesian-space action representations.

Quaternion convention is ``[x, y, z, w]`` (matches ROS / TF2 and the
existing :class:`lerobot.utils.rotation.Rotation` class). All functions
accept numpy arrays or any array-like; inputs are converted to
``float64`` internally.

The 6D rotation representation (Zhou et al. 2019) takes the first two
columns of the 3×3 rotation matrix and flattens them column-major into a
6-vector::

    rot6d = [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]

This is preferred over Euler / quaternion for regression targets in
diffusion policies because it has no discontinuities and is easy to
reconstruct (Gram-Schmidt of the first two columns + cross product for
the third).
"""
from __future__ import annotations

import numpy as np


def quat_to_matrix(quat_xyzw) -> np.ndarray:
    """Convert quaternion ``[x, y, z, w]`` to a 3×3 rotation matrix.

    Input is normalised internally; unit-norm is not required.
    """
    q = np.asarray(quat_xyzw, dtype=float).reshape(-1)
    if q.shape != (4,):
        raise ValueError(f"quat_to_matrix expects shape (4,), got {q.shape}")
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    qx, qy, qz, qw = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def matrix_to_quat(R) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a quaternion ``[x, y, z, w]``.

    Uses Shepperd's method for numerical stability.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"matrix_to_quat expects shape (3,3), got {R.shape}")
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=float)


def matrix_to_rot6d(R) -> np.ndarray:
    """Flatten the first two columns of a 3×3 rotation matrix into a 6-vector."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"matrix_to_rot6d expects shape (3,3), got {R.shape}")
    return R[:, :2].T.reshape(6).astype(float)


def rot6d_to_matrix(r6d) -> np.ndarray:
    """Reconstruct a 3×3 rotation matrix from a 6-vector via Gram-Schmidt.

    Inverse of :func:`matrix_to_rot6d` up to numerical precision.
    """
    v = np.asarray(r6d, dtype=float).reshape(-1)
    if v.shape != (6,):
        raise ValueError(f"rot6d_to_matrix expects shape (6,), got {v.shape}")
    a1, a2 = v[:3], v[3:6]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-10:
        raise ValueError(
            f"rot6d_to_matrix: first column (a1={a1}) has near-zero norm {n1:.2e}; "
            "input is degenerate and cannot be orthonormalized."
        )
    b1 = a1 / n1
    b2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(b2)
    if n2 < 1e-10:
        raise ValueError(
            f"rot6d_to_matrix: second column after projection has near-zero norm {n2:.2e}; "
            "a1 and a2 are nearly parallel — input is degenerate."
        )
    b2 = b2 / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)
