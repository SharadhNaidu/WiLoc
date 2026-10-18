"""Vectorised 2-D geometry helpers (segment intersection, angle wrapping)."""

from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def segments_intersect(p, q, walls):
    """Proper-intersection test between many rays and many wall segments.

    Parameters
    ----------
    p, q : (N, 2) arrays
        Start and end points of N query segments.
    walls : (W, 4) array
        Wall segments as rows ``[x1, y1, x2, y2]``.

    Returns
    -------
    (N, W) boolean array, True where query segment i crosses wall j.
    """
    p = np.atleast_2d(np.asarray(p, dtype=float))
    q = np.atleast_2d(np.asarray(q, dtype=float))
    walls = np.atleast_2d(np.asarray(walls, dtype=float))
    if walls.size == 0:
        return np.zeros((p.shape[0], 0), dtype=bool)

    # r = q - p (query direction), s = b - a (wall direction)
    rx = (q[:, 0] - p[:, 0])[:, None]
    ry = (q[:, 1] - p[:, 1])[:, None]
    ax, ay = walls[None, :, 0], walls[None, :, 1]
    sx = (walls[:, 2] - walls[:, 0])[None, :]
    sy = (walls[:, 3] - walls[:, 1])[None, :]
    px, py = p[:, 0][:, None], p[:, 1][:, None]

    denom = _cross(rx, ry, sx, sy)
    qpx, qpy = ax - px, ay - py
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(qpx, qpy, sx, sy) / denom  # along query
        u = _cross(qpx, qpy, rx, ry) / denom  # along wall
    hit = (np.abs(denom) > 1e-12) & (t > 0.0) & (t < 1.0) & (u >= 0.0) & (u <= 1.0)
    return hit


def point_segment_distance(points, walls):
    """Euclidean distance from each point to each segment, shape (N, W)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    walls = np.atleast_2d(np.asarray(walls, dtype=float))
    a = walls[None, :, 0:2]
    b = walls[None, :, 2:4]
    p = points[:, None, :]
    ab = b - a
    denom = np.sum(ab * ab, axis=-1)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    t = np.clip(np.sum((p - a) * ab, axis=-1) / denom, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1)
