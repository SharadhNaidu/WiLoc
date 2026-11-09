"""Path planning on the floor plan: occupancy grid, A*, path smoothing, following.

``OccupancyGrid``  free space at ``res`` metres, walls (and furniture, if the
                   plan has any) inflated by the robot radius.  Each cell keeps
                   its clearance (distance to the nearest obstacle), which the
                   planner turns into a cost so routes run through the middle
                   of doors and corridors instead of grazing door frames.
``plan``           A* on the 8-connected grid, then line-of-sight smoothing
                   into a few vertices that a turn-then-drive controller can
                   follow without stopping at every cell.
``PathFollower``   drives through those vertices with a ``WaypointController``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .environment import FloorPlan
from .motion import WaypointController

SQRT2 = math.sqrt(2.0)


def _rect_distance(points: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """Distance from points (N, 2) to axis-aligned rectangles (K, 4) [x0, y0, x1, y1]; 0 inside."""
    if len(rects) == 0:
        return np.full(len(points), np.inf)
    px, py = points[:, 0:1], points[:, 1:2]
    dx = np.maximum(np.maximum(rects[None, :, 0] - px, 0.0), px - rects[None, :, 2])
    dy = np.maximum(np.maximum(rects[None, :, 1] - py, 0.0), py - rects[None, :, 3])
    return np.hypot(dx, dy).min(axis=1)


def _wall_box_distance(points: np.ndarray, walls: np.ndarray, thickness: float) -> np.ndarray:
    """Distance from points (N, 2) to walls (K, 4) built as boxes, the way the Gazebo world builds them.

    Each wall is ``thickness`` thick and runs ``thickness / 2`` past both ends (they
    overlap at corners), so a 1 m door gap is really 1 m - thickness wide.
    """
    a, b = walls[:, 0:2], walls[:, 2:4]
    length = np.linalg.norm(b - a, axis=1)
    u = (b - a) / np.maximum(length, 1e-9)[:, None]
    mid = 0.5 * (a + b)
    rel = points[:, None, :] - mid[None, :, :]  # (N, K, 2)
    along = rel[..., 0] * u[None, :, 0] + rel[..., 1] * u[None, :, 1]
    across = rel[..., 0] * -u[None, :, 1] + rel[..., 1] * u[None, :, 0]
    dx = np.maximum(np.abs(along) - (0.5 * length[None, :] + 0.5 * thickness), 0.0)
    dy = np.maximum(np.abs(across) - 0.5 * thickness, 0.0)
    return np.hypot(dx, dy).min(axis=1)


@dataclass
class OccupancyGrid:
    res: float
    nx: int
    ny: int
    clearance: np.ndarray  # (ny, nx) metres to the nearest wall / obstacle / outer boundary
    inflate: float
    prefer: float = 0.8  # clearance beyond which a cell costs nothing extra
    weight: float = 4.0  # extra cost per metre travelled at zero spare clearance

    @classmethod
    def from_floorplan(cls, fp: FloorPlan, res: float = 0.1, inflate: float = 0.27, obstacles: bool = True, **kw):
        nx, ny = int(math.ceil(fp.width / res)), int(math.ceil(fp.height / res))
        gx, gy = np.meshgrid((np.arange(nx) + 0.5) * res, (np.arange(ny) + 0.5) * res)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        border = np.minimum.reduce([pts[:, 0], fp.width - pts[:, 0], pts[:, 1], fp.height - pts[:, 1]])
        clear = border
        if len(fp.walls):
            clear = np.minimum(clear, _wall_box_distance(pts, np.asarray(fp.walls, dtype=float), fp.wall_thickness))
        rects = getattr(fp, "obstacles", np.zeros((0, 4)))
        if obstacles and len(rects):
            clear = np.minimum(clear, _rect_distance(pts, np.asarray(rects, dtype=float)))
        return cls(res, nx, ny, clear.reshape(ny, nx), inflate, **kw)

    # ------------------------------------------------------------ queries
    @property
    def free(self) -> np.ndarray:
        return self.clearance >= self.inflate

    def cell(self, xy) -> tuple[int, int]:
        ix = int(np.clip(xy[0] / self.res, 0, self.nx - 1))
        iy = int(np.clip(xy[1] / self.res, 0, self.ny - 1))
        return iy, ix

    def center(self, iy: int, ix: int) -> np.ndarray:
        return np.array([(ix + 0.5) * self.res, (iy + 0.5) * self.res])

    def clearance_at(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        ix = np.clip((pts[:, 0] / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip((pts[:, 1] / self.res).astype(int), 0, self.ny - 1)
        return self.clearance[iy, ix]

    def is_free(self, pts: np.ndarray) -> np.ndarray:
        return self.clearance_at(pts) >= self.inflate

    def nearest_free(self, xy) -> np.ndarray:
        """The free cell centre closest to ``xy`` (``xy`` itself if already free)."""
        xy = np.asarray(xy, dtype=float)[:2]
        if self.is_free(xy[None])[0]:
            return xy.copy()
        iy, ix = np.nonzero(self.free)
        if len(ix) == 0:
            raise ValueError("occupancy grid has no free cell")
        c = np.column_stack([(ix + 0.5) * self.res, (iy + 0.5) * self.res])
        return c[np.argmin(np.sum((c - xy) ** 2, axis=1))]

    def line_free(self, a, b, margin: float = 0.0) -> bool:
        """Every point of segment a-b keeps the robot's clearance (plus ``margin``)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / (0.5 * self.res))) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        return bool((self.clearance_at(a + t * (b - a)) >= self.inflate + margin).all())

    # ------------------------------------------------------------ planning
    def astar(self, start, goal) -> list[tuple[int, int]] | None:
        """Cell path from start to goal (both must be free cells), or None."""
        free = self.free
        # cost multiplier per cell: 1 + weight * shortfall of clearance below `prefer`
        spare = np.clip((self.prefer - self.clearance) / max(self.prefer - self.inflate, 1e-6), 0.0, 1.0)
        mult = 1.0 + self.weight * spare
        s, g = self.cell(start), self.cell(goal)
        if not (free[s] and free[g]):
            return None
        gy, gx = g
        dist = {s: 0.0}
        parent = {s: None}
        heap = [(0.0, s)]
        steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]
        closed = np.zeros_like(free)
        while heap:
            _, cur = heapq.heappop(heap)
            if closed[cur]:
                continue
            if cur == g:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return path[::-1]
            closed[cur] = True
            cy, cx = cur
            d0 = dist[cur]
            for dy, dx, step in steps:
                ny_, nx_ = cy + dy, cx + dx
                if not (0 <= ny_ < self.ny and 0 <= nx_ < self.nx) or not free[ny_, nx_] or closed[ny_, nx_]:
                    continue
                if dy and dx and not (free[cy, nx_] and free[ny_, cx]):
                    continue  # no corner cutting past an obstacle
                nd = d0 + step * self.res * 0.5 * (mult[cy, cx] + mult[ny_, nx_])
                nb = (ny_, nx_)
                if nd < dist.get(nb, math.inf):
                    dist[nb] = nd
                    parent[nb] = cur
                    h = self.res * math.hypot(ny_ - gy, nx_ - gx)
                    heapq.heappush(heap, (nd + h, nb))
        return None

    def smooth(self, pts: np.ndarray, margin: float = 0.08) -> np.ndarray:
        """Greedy line-of-sight shortcutting: keep only the vertices that are needed.

        Shortcuts must keep ``margin`` metres of clearance beyond the robot
        radius, so a smoothed route does not graze door frames while the
        turn-then-drive controller settles its heading.
        """
        if len(pts) <= 2:
            return pts
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j], margin):
                j -= 1
            out.append(pts[j])
            i = j
        return np.array(out)

    def plan(self, start, goal) -> np.ndarray | None:
        """Smoothed vertex list (K, 2) from start to goal, or None if unreachable.

        Start and goal are snapped to the nearest free cell (a robot pressed
        against a wall, or a click inside furniture, still gets a route).  The
        returned path begins at the snapped start and ends at the snapped goal.
        """
        s = self.nearest_free(start)
        g = self.nearest_free(goal)
        cells = self.astar(s, g)
        if cells is None:
            return None
        pts = np.array([self.center(iy, ix) for iy, ix in cells])
        pts[0], pts[-1] = s, g
        return self.smooth(pts)


def path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0


@dataclass
class PathFollower:
    """Follow a vertex list with a turn-then-drive controller.

    Intermediate vertices use a looser tolerance so the robot flows through
    them; the final vertex uses the controller's own tolerance.
    """

    path: np.ndarray
    controller: WaypointController = field(default_factory=WaypointController)
    pass_tol: float = 0.25
    index: int = 1  # vertex 0 is where the robot started

    def __post_init__(self):
        self.path = np.atleast_2d(np.asarray(self.path, dtype=float))
        if len(self.path) < 2:
            self.index = len(self.path)

    @property
    def done(self) -> bool:
        return self.index >= len(self.path)

    @property
    def remaining(self) -> np.ndarray:
        return self.path[self.index :]

    def command(self, pose) -> tuple[float, float, bool]:
        """(v, w, finished) for the current pose."""
        while not self.done:
            last = self.index == len(self.path) - 1
            goal = self.path[self.index]
            if not last and np.hypot(goal[0] - pose[0], goal[1] - pose[1]) < self.pass_tol:
                self.index += 1
                continue
            v, w, reached = self.controller.command(pose, goal)
            if reached:
                self.index += 1
                continue
            return v, w, False
        return 0.0, 0.0, True


def validate_floorplan(fp: FloorPlan, grid: OccupancyGrid | None = None) -> list[str]:
    """Problems that would break a live run: things placed in walls or furniture, unreachable places.

    Checks the start pose, every patrol waypoint and kidnap spot against the
    inflated grid, that every kidnap spot can reach the patrol, and that every
    access point and room lies inside the building.
    """
    grid = grid or OccupancyGrid.from_floorplan(fp)
    problems = []
    start = np.asarray(fp.start_pose[:2], dtype=float)
    named = [("start pose", start)]
    named += [(f"waypoint {i + 1} {tuple(p)}", np.asarray(p)) for i, p in enumerate(fp.waypoints)]
    named += [(f"kidnap spot {tuple(p)}", np.asarray(p)) for p in fp.kidnap_spots]
    for name, p in named:
        if not grid.is_free(p[None])[0]:
            problems.append(f"{name} is not free (clearance {grid.clearance_at(p[None])[0]:.2f} m < {grid.inflate} m)")
    if not problems:
        for name, p in named[1:]:
            if grid.plan(start, p) is None:
                problems.append(f"{name} cannot be reached from the start pose")
    for ap in fp.access_points:
        if not (0 <= ap.x <= fp.width and 0 <= ap.y <= fp.height):
            problems.append(f"access point {ap.id} is outside the building")
    for name, (x0, y0, x1, y1) in fp.rooms:
        if x0 < 0 or y0 < 0 or x1 > fp.width or y1 > fp.height or x1 <= x0 or y1 <= y0:
            problems.append(f"room {name!r} has a bad rectangle")
    problems += _mover_problems(fp)
    return problems


def _mover_problems(fp: FloorPlan, step: float = 0.1, clearance: float = 0.2) -> list[str]:
    """Moving people and vehicles must not walk through walls or furniture (static ones may sit on chairs)."""
    problems = []
    obstacles = fp.obstacles
    for mv in fp.movers.movers:
        if mv.static:
            x, y = mv.route[0]
            if not (0 < x < fp.width and 0 < y < fp.height):
                problems.append(f"mover {mv.id} is outside the building")
            continue
        path = mv.path
        pts = np.vstack(
            [np.linspace(a, b, max(2, int(np.ceil(np.hypot(*(b - a)) / step)) + 1)) for a, b in zip(path[:-1], path[1:])]
        )
        bad = ~fp.is_free(pts, clearance)
        if len(obstacles):
            inside = (
                (pts[:, None, 0] > obstacles[None, :, 0])
                & (pts[:, None, 0] < obstacles[None, :, 2])
                & (pts[:, None, 1] > obstacles[None, :, 1])
                & (pts[:, None, 1] < obstacles[None, :, 3])
            ).any(axis=1)
            bad |= inside
        if bad.any():
            x, y = pts[np.flatnonzero(bad)[0]]
            problems.append(f"mover {mv.id} walks through a wall or furniture near ({x:.1f}, {y:.1f})")
    return problems


def kidnap_target(fp: FloorPlan, grid: OccupancyGrid, current_xy, rng: np.random.Generator, min_dist: float = 6.0):
    """Where to drop a kidnapped robot: a curated spot far away, else open free space far away."""
    current_xy = np.asarray(current_xy, dtype=float)[:2]
    spots = np.asarray(fp.kidnap_spots, dtype=float).reshape(-1, 2)
    if not len(spots):
        cells = np.column_stack(np.nonzero(grid.free & (grid.clearance >= 0.6)))
        spots = np.column_stack([(cells[:, 1] + 0.5) * grid.res, (cells[:, 0] + 0.5) * grid.res])
    far = spots[np.hypot(*(spots - current_xy).T) >= min_dist]
    pool = far if len(far) else spots
    return pool[rng.integers(len(pool))]
