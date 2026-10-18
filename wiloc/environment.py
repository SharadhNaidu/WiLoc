"""Floor plan description: walls (with materials), access points and free space.

A floor plan is described once in YAML and shared by every part of the
framework: the radio simulator, the fingerprint survey, the particle filter's
map constraint, and the Gazebo world generator in ``wiloc_ros``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from .geometry import point_segment_distance, segments_intersect

# Per-wall penetration loss in dB at 2.4 GHz.  Typical values drawn from the
# COST 231 multi-wall model and indoor measurement campaigns (light walls
# ~3-4 dB, brick ~6-8 dB, concrete ~10-15 dB).
DEFAULT_MATERIALS = {
    "concrete": 12.0,
    "brick": 7.0,
    "drywall": 3.5,
    "glass": 2.0,
    "wood": 3.0,
}


@dataclass
class AccessPoint:
    id: str
    x: float
    y: float
    tx_power: float = 20.0  # dBm (EIRP)
    channel: int = 1

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class FloorPlan:
    name: str
    width: float
    height: float
    walls: np.ndarray  # (W, 4)
    wall_materials: list[str]
    materials: dict[str, float]
    access_points: list[AccessPoint]
    wall_height: float = 2.5
    wall_thickness: float = 0.15
    waypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    start_pose: tuple[float, float, float] = (1.0, 6.0, 0.0)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict) -> "FloorPlan":
        materials = dict(DEFAULT_MATERIALS)
        materials.update(d.get("materials", {}))
        walls, mats = [], []
        for w in d["walls"]:
            walls.append([float(v) for v in w["seg"]])
            mats.append(w.get("material", "drywall"))
        aps = [AccessPoint(**ap) for ap in d["access_points"]]
        return cls(
            name=d.get("name", "floorplan"),
            width=float(d["width"]),
            height=float(d["height"]),
            walls=np.array(walls, dtype=float).reshape(-1, 4),
            wall_materials=mats,
            materials=materials,
            access_points=aps,
            wall_height=float(d.get("wall_height", 2.5)),
            wall_thickness=float(d.get("wall_thickness", 0.15)),
            waypoints=np.array(d.get("waypoints", []), dtype=float).reshape(-1, 2),
            start_pose=tuple(d.get("start_pose", (1.0, 6.0, 0.0))),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FloorPlan":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def builtin(cls, name: str = "office") -> "FloorPlan":
        """Load one of the floor plans shipped in ``wiloc/data``."""
        ref = resources.files("wiloc") / "data" / f"{name}.yaml"
        with ref.open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    # ------------------------------------------------------------ queries
    @property
    def ap_positions(self) -> np.ndarray:
        return np.array([ap.pos for ap in self.access_points])

    @property
    def wall_losses(self) -> np.ndarray:
        return np.array([self.materials[m] for m in self.wall_materials])

    def wall_loss_between(self, points: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Total wall attenuation (dB) on the straight path points -> target."""
        points = np.atleast_2d(points)
        tgt = np.broadcast_to(np.asarray(target, dtype=float), points.shape)
        hits = segments_intersect(points, tgt, self.walls)
        return hits.astype(float) @ self.wall_losses

    def crosses_wall(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """True for each segment p_i -> q_i that crosses any wall."""
        return segments_intersect(p, q, self.walls).any(axis=1)

    def is_free(self, points: np.ndarray, clearance: float = 0.15) -> np.ndarray:
        """Inside the building and at least ``clearance`` metres from walls."""
        points = np.atleast_2d(points)
        inside = (
            (points[:, 0] > clearance)
            & (points[:, 0] < self.width - clearance)
            & (points[:, 1] > clearance)
            & (points[:, 1] < self.height - clearance)
        )
        if len(self.walls) == 0:
            return inside
        d = point_segment_distance(points, self.walls).min(axis=1)
        return inside & (d >= clearance)

    def grid(self, spacing: float, clearance: float = 0.15) -> np.ndarray:
        """Regular grid of free-space points (reference points for a survey)."""
        xs = np.arange(spacing / 2, self.width, spacing)
        ys = np.arange(spacing / 2, self.height, spacing)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        return pts[self.is_free(pts, clearance)]

    def sample_free(self, n: int, rng: np.random.Generator, clearance: float = 0.2) -> np.ndarray:
        """Uniform random points in free space (rejection sampling)."""
        out = np.empty((0, 2))
        while len(out) < n:
            cand = rng.uniform([0, 0], [self.width, self.height], size=(2 * n, 2))
            out = np.vstack([out, cand[self.is_free(cand, clearance)]])
        return out[:n]

    def subset_aps(self, ids: list[str]) -> "FloorPlan":
        """Copy of the plan keeping only the access points in ``ids``."""
        import copy

        fp = copy.copy(self)
        fp.access_points = [ap for ap in self.access_points if ap.id in ids]
        return fp
