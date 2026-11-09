"""Floor plan description: walls (with materials), access points and free space.

A floor plan is described once in YAML and shared by every part of the
framework: the radio simulator, the fingerprint survey, the particle filter's
map constraint, and the Gazebo world generator in ``wiloc_ros``.

Besides walls and access points a plan can describe what makes it look like a
real place: named ``rooms`` (for captions), ``furniture`` (typed boxes that the
robot must drive around: ``obstacles``, used by the planner) and curated
``kidnap_spots``, plus ``movers`` (people, forklifts, AGVs) and ``noise_zones``
(see ``wiloc.movers``).  Router placements come as named presets
(``ap_layouts:``; the top-level ``access_points`` are the "installed" one) and
``with_layout(name)`` switches between them.  Furniture is deliberately invisible to the radio model and
to the particle filter: the radio world and every result stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from .geometry import point_segment_distance, segments_intersect
from .movers import Movers

# Per-wall penetration loss in dB at 2.4 GHz.  Typical values drawn from the
# COST 231 multi-wall model and indoor measurement campaigns (light walls
# ~3-4 dB, brick ~6-8 dB, concrete ~10-15 dB).
DEFAULT_MATERIALS = {
    "concrete": 12.0,
    "brick": 7.0,
    "drywall": 3.5,
    "glass": 2.0,
    "wood": 3.0,
    "metal": 15.0,  # steel racks, machinery, lift shafts
}

# Furniture: default height (m) and footprint (m, for items placed with ``at:``).
FURNITURE = {
    "desk": (0.75, (1.6, 0.7)),
    "table": (0.75, (1.2, 0.8)),
    "meeting_table": (0.75, (1.8, 1.0)),
    "coffee_table": (0.42, (1.1, 0.55)),
    "chair": (0.9, (0.48, 0.48)),
    "sofa": (0.85, (2.2, 0.85)),
    "bookshelf": (1.8, (1.6, 0.35)),
    "cabinet": (1.1, (0.8, 0.45)),
    "counter": (0.92, (2.0, 0.65)),
    "fridge": (1.8, (0.7, 0.7)),
    "plant": (1.1, (0.5, 0.5)),
    "bed": (0.55, (2.0, 1.0)),
    "rack": (3.0, (6.0, 1.0)),
    "machine": (1.8, (2.0, 1.5)),
    "conveyor": (0.9, (6.0, 0.8)),
    "box": (1.0, (1.0, 1.0)),
}


@dataclass
class Furniture:
    type: str
    rect: tuple[float, float, float, float]  # x0, y0, x1, y1
    height: float
    back: str = "n"  # side of a backrest (chairs, sofas, beds' headboard): n | s | e | w

    @classmethod
    def from_dict(cls, d: dict) -> "Furniture":
        kind = str(d["type"])
        height, (w, depth) = FURNITURE.get(kind, (1.0, (1.0, 1.0)))
        if "rect" in d:
            x0, y0, x1, y1 = (float(v) for v in d["rect"])
        else:
            w, depth = (float(v) for v in d.get("size", (w, depth)))
            if d.get("rotate"):  # footprint turned by 90 degrees
                w, depth = depth, w
            x, y = (float(v) for v in d["at"])
            x0, y0, x1, y1 = x - w / 2, y - depth / 2, x + w / 2, y + depth / 2
        return cls(kind, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), float(d.get("height", height)), str(d.get("back", "n")))


@dataclass
class AccessPoint:
    id: str
    x: float
    y: float
    tx_power: float = 20.0  # dBm (EIRP)
    channel: int = 1

    def __post_init__(self):
        # YAML may give integers (x: 5); ROS message fields need floats.
        self.id = str(self.id)
        self.x, self.y, self.tx_power = float(self.x), float(self.y), float(self.tx_power)
        self.channel = int(self.channel)

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
    radio: dict = field(default_factory=dict)  # RadioConfig settings for this venue (``radio:``)
    furniture: list[Furniture] = field(default_factory=list)
    rooms: list[tuple[str, tuple[float, float, float, float]]] = field(default_factory=list)
    kidnap_spots: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    movers: Movers = field(default_factory=Movers)  # people / vehicles and noise zones (``movers:``, ``noise_zones:``)
    title: str = ""  # display name ("Hospital ward")
    caption: str = ""  # one line: what this venue shows
    camera: dict = field(default_factory=dict)  # named views: name -> [x, y, z, pitch, yaw]
    layouts: dict = field(default_factory=dict)  # router presets: name -> (label, [AccessPoint])
    layout: str = ""  # current preset

    def __post_init__(self):
        if not self.title:
            self.title = self.name.replace("_", " ").capitalize()
        if not self.layouts:
            self.layouts = {"installed": ("as installed", list(self.access_points))}
            self.layout = "installed"
        self.layout = self.layout or next(iter(self.layouts))

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict) -> "FloorPlan":
        materials = dict(DEFAULT_MATERIALS)
        materials.update(d.get("materials", {}))
        walls, mats = [], []
        for w in d["walls"]:
            walls.append([float(v) for v in w["seg"]])
            mats.append(w.get("material", "drywall"))
        layouts = {}
        if d.get("access_points"):
            layouts["installed"] = (str(d.get("layout_label", "as installed")), [AccessPoint(**ap) for ap in d["access_points"]])
        for key, lay in (d.get("ap_layouts") or {}).items():
            layouts[str(key)] = (str(lay.get("label", key)), [AccessPoint(**ap) for ap in lay["access_points"]])
        if not layouts:
            raise ValueError("a floor plan needs access_points or ap_layouts")
        first = next(iter(layouts))
        aps = list(layouts[first][1])
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
            start_pose=tuple(float(v) for v in d.get("start_pose", (1.0, 6.0, 0.0))),
            radio=dict(d.get("radio") or {}),
            furniture=[Furniture.from_dict(f) for f in d.get("furniture") or []],
            rooms=[(str(r["name"]), tuple(float(v) for v in r["rect"])) for r in d.get("rooms") or []],
            kidnap_spots=np.array(d.get("kidnap_spots") or [], dtype=float).reshape(-1, 2),
            movers=Movers.from_dicts(d.get("movers"), d.get("noise_zones")),
            title=str(d.get("title", "")),
            caption=str(d.get("caption", "")),
            camera={str(k): [float(v) for v in vals] for k, vals in (d.get("camera") or {}).items()},
            layouts=layouts,
            layout=first,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FloorPlan":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def builtin(cls, name: str = "office", layout: str | None = None) -> "FloorPlan":
        """Load one of the venues shipped in ``wiloc/data/venues`` (optionally with a router preset)."""
        ref = resources.files("wiloc") / "data" / "venues" / f"{name}.yaml"
        if not ref.is_file():
            raise FileNotFoundError(f"no built-in venue {name!r}; known: {cls.builtin_names()}")
        with ref.open("r", encoding="utf-8") as f:
            fp = cls.from_dict(yaml.safe_load(f))
        return fp.with_layout(layout) if layout else fp

    @staticmethod
    def builtin_names() -> list[str]:
        """Names of the venues shipped in ``wiloc/data/venues``, in story order."""
        names = [p.name[: -len(".yaml")] for p in (resources.files("wiloc") / "data" / "venues").iterdir() if p.name.endswith(".yaml")]
        story = ["office", "home", "hospital", "warehouse", "library", "factory"]  # the order the stand tells them
        return sorted(names, key=lambda n: (story.index(n) if n in story else len(story), n))

    def with_layout(self, name: str) -> "FloorPlan":
        """Copy of the plan with the access points of router preset ``name``."""
        import copy

        if name not in self.layouts:
            raise KeyError(f"{self.name} has no router layout {name!r}; known: {list(self.layouts)}")
        fp = copy.copy(self)
        fp.access_points = list(self.layouts[name][1])
        fp.layout = name
        return fp

    @property
    def layout_label(self) -> str:
        return self.layouts[self.layout][0]

    @property
    def max_access_points(self) -> int:
        """Most access points in any router preset (Gazebo pre-spawns this many AP discs)."""
        return max(len(aps) for _, aps in self.layouts.values())

    # ------------------------------------------------------------ queries
    @property
    def obstacles(self) -> np.ndarray:
        """Furniture footprints (K, 4) [x0, y0, x1, y1]: for path planning, not for radio."""
        return np.array([f.rect for f in self.furniture], dtype=float).reshape(-1, 4)

    def room_at(self, xy) -> str | None:
        """Name of the room containing point ``xy`` (first match), or None."""
        x, y = float(xy[0]), float(xy[1])
        for name, (x0, y0, x1, y1) in self.rooms:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return None

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
