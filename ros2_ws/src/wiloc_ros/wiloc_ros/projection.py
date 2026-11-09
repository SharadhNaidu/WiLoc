"""World points -> pixels of the Studio camera (for drawing on top of the 3-D view).

The camera is a Gazebo camera sensor on a link posed at (x, y, z) with pitch
(positive: looking down) and yaw; Gazebo cameras look along the link's +x,
with +y to the left and +z up.  A pinhole model with the sensor's horizontal
field of view gives the pixel coordinates (u right, v down).
"""

from __future__ import annotations

import math

import numpy as np


def intrinsics(width: int, height: int, hfov: float) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) of a Gazebo camera with square pixels."""
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    return fx, fx, width / 2.0, height / 2.0


def rotation(pitch: float, yaw: float) -> np.ndarray:
    """Link orientation in the world: R = Rz(yaw) @ Ry(pitch)."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return rz @ ry


def project(points, cam_pose, K, near: float = 0.05):
    """Pixels (N, 2) of world points (N, 3) and a mask of the ones in front of the camera.

    ``cam_pose`` = (x, y, z, pitch, yaw); ``K`` = (fx, fy, cx, cy).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    x, y, z, pitch, yaw = cam_pose
    fx, fy, cx, cy = K
    local = (pts - np.array([x, y, z])) @ rotation(pitch, yaw)  # rows: R^T (p - c)
    depth = local[:, 0]
    ok = depth > near
    d = np.where(ok, depth, 1.0)
    u = cx - fx * local[:, 1] / d
    v = cy - fy * local[:, 2] / d
    return np.column_stack([u, v]), ok


def clip_segment(a, b, cam_pose, near: float = 0.1):
    """Part of the 3-D segment a-b in front of the camera (None if none of it is)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    x, y, z, pitch, yaw = cam_pose
    fwd = rotation(pitch, yaw)[:, 0]
    da, db = (a - (x, y, z)) @ fwd, (b - (x, y, z)) @ fwd
    if da < near and db < near:
        return None
    if da < near:
        a = a + (b - a) * (near - da) / (db - da)
    elif db < near:
        b = b + (a - b) * (near - db) / (da - db)
    return a, b
