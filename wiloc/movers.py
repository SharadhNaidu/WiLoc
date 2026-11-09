"""Things that move through a venue and get in the way of Wi-Fi: people, forklifts, AGVs.

A venue's ``movers:`` section lists them; each follows a fixed route at a fixed
speed, so where it is depends only on the time::

    movers:
      - {id: walker1, type: person, route: [[2, 6], [18, 6]], loop: false, pause: 2}
      - {id: desk_b, type: person, route: [[7.2, 2.4]], yaw: 1.57, anim: sitting}
      - {id: lift1, type: forklift, route: [[3, 2], [27, 2], [27, 18], [3, 18]], speed: 1.4}

* ``loop: true`` (default) goes round the route and back to its first point;
  ``loop: false`` walks it to the end and back again;
* ``pause`` seconds are spent standing at every route vertex;
* a route with one point is a static mover (``yaw`` sets which way it faces).

Positions are a pure function of time (``t mod period``), so every process
(the Wi-Fi sensor, the driver, the UI, a respawned node) and Gazebo's scripted
actors agree without sharing state.

**Body shadowing.**  A mover near the straight robot-AP path weakens that link by
``loss_db * exp(-(d / radius)^2)``, with ``d`` the distance from the mover to
the segment (a human body costs ~3-5 dB at 2.4 GHz; a steel forklift far
more).  The loss is continuous and symmetric in robot and AP.

**Noise zones** (``noise_zones:``) are circles of electrical noise near motors and
welding cells: inside, scans get ``sigma_db`` of extra temporal noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Defaults per mover type: attenuation of a link passing through it (dB),
# effective radius of its body (m), walking/driving speed (m/s).
MOVER_TYPES = {
    "person": {"loss_db": 4.0, "radius": 0.35, "speed": 1.0},
    "forklift": {"loss_db": 12.0, "radius": 0.9, "speed": 1.4},
    "agv": {"loss_db": 8.0, "radius": 0.6, "speed": 0.8},
    "gantry": {"loss_db": 10.0, "radius": 0.8, "speed": 0.5},
    "cart": {"loss_db": 3.0, "radius": 0.35, "speed": 0.6},
}
# Animations for people (Gazebo meshes whose facing matches the actor's yaw).
PERSON_ANIMS = ("walk", "stand", "sitting")


@dataclass
class Mover:
    id: str
    type: str
    route: np.ndarray  # (K, 2)
    speed: float
    loss_db: float
    radius: float
    loop: bool = True
    pause: float = 0.0
    yaw: float = 0.0  # facing of a static mover
    anim: str | None = None  # Gazebo animation (people): walk, stand or sitting
    # keyframes (built in __post_init__): times (F,), positions (F, 2), period
    times: np.ndarray = field(init=False, repr=False)
    points: np.ndarray = field(init=False, repr=False)
    period: float = field(init=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Mover":
        kind = str(d.get("type", "person"))
        if kind not in MOVER_TYPES:
            raise ValueError(f"unknown mover type {kind!r}; known: {sorted(MOVER_TYPES)}")
        base = MOVER_TYPES[kind]
        route = np.array(d["route"], dtype=float).reshape(-1, 2)
        anim = d.get("anim")
        if anim is not None and (kind != "person" or anim not in PERSON_ANIMS):
            raise ValueError(f"mover {d.get('id', kind)}: animation {anim!r} needs type person and one of {PERSON_ANIMS}")
        return cls(
            id=str(d.get("id", kind)),
            type=kind,
            route=route,
            speed=float(d.get("speed", base["speed"])),
            loss_db=float(d.get("loss_db", base["loss_db"])),
            radius=float(d.get("radius", base["radius"])),
            loop=bool(d.get("loop", True)),
            pause=float(d.get("pause", 0.0)),
            yaw=float(d.get("yaw", 0.0)),
            anim=anim,
        )

    def __post_init__(self):
        self.route = np.asarray(self.route, dtype=float).reshape(-1, 2)
        if self.static:
            self.times, self.points, self.period = np.zeros(1), self.route[:1].copy(), 1.0
            return
        if self.speed <= 0:
            raise ValueError(f"mover {self.id}: speed must be > 0")
        path = self.path
        times, points, t = [], [], 0.0
        for k, p in enumerate(path[:-1]):
            times.append(t)
            points.append(p)
            if self.pause > 0:
                t += self.pause
                times.append(t)
                points.append(p)
            t += float(np.hypot(*(path[k + 1] - p))) / self.speed
        times.append(t)
        points.append(path[-1])
        self.times, self.points, self.period = np.array(times), np.array(points), t

    # ---------------------------------------------------------------- shape
    @property
    def static(self) -> bool:
        return len(self.route) == 1

    @property
    def path(self) -> np.ndarray:
        """Closed path the mover follows in one period (first point repeated at the end)."""
        if self.static:
            return self.route[:1]
        if self.loop:
            return np.vstack([self.route, self.route[:1]])
        return np.vstack([self.route, self.route[-2::-1]])

    # ---------------------------------------------------------------- state
    def position(self, t: float) -> np.ndarray:
        if self.static:
            return self.route[0].copy()
        tau = float(t) % self.period
        return np.array([np.interp(tau, self.times, self.points[:, 0]), np.interp(tau, self.times, self.points[:, 1])])

    def heading(self, t: float) -> float:
        """Direction of travel (rad); while paused, the direction it arrived from."""
        if self.static:
            return self.yaw
        tau = float(t) % self.period
        k = int(np.clip(np.searchsorted(self.times, tau, side="right") - 1, 0, len(self.times) - 2))
        # step back over pause keyframes (no displacement) to the last real move
        for j in range(k, k - len(self.times), -1):
            d = self.points[(j + 1) % len(self.points)] - self.points[j % len(self.points)]
            if np.hypot(*d) > 1e-9:
                return float(np.arctan2(d[1], d[0]))
        return self.yaw

    def moving(self, t: float) -> bool:
        if self.static:
            return False
        tau = float(t) % self.period
        k = int(np.clip(np.searchsorted(self.times, tau, side="right") - 1, 0, len(self.times) - 2))
        return bool(np.hypot(*(self.points[k + 1] - self.points[k])) > 1e-9)

    def keyframes(self, step: float = 0.2) -> list[tuple[float, float, float, float]]:
        """Dense (time, x, y, yaw) samples of one period, for Gazebo's scripted actors.

        Gazebo interpolates actor poses with splines, which overshoot sparse
        corners; samples every ``step`` metres keep it on the straight route.
        """
        if self.static:
            x, y = self.route[0]
            return [(0.0, x, y, self.yaw), (1.0, x, y, self.yaw)]
        out = []
        for k in range(len(self.times) - 1):
            t0, t1 = self.times[k], self.times[k + 1]
            p0, p1 = self.points[k], self.points[k + 1]
            d = p1 - p0
            dist = float(np.hypot(*d))
            if dist < 1e-9:  # pause: stand, turning to face the next leg
                yaw_next = self.heading(t1 + 1e-6)
                out.append((t0, *p0, self.heading(t0)))
                out.append((t1, *p1, yaw_next))
                continue
            yaw = float(np.arctan2(d[1], d[0]))
            n = max(1, int(np.ceil(dist / step)))
            for i in range(n):
                f = i / n
                out.append((t0 + f * (t1 - t0), *(p0 + f * d), yaw))
        last = self.points[-1]
        out.append((self.period, *last, self.heading(self.period - 1e-6)))
        # merge exact-duplicate times (pause keyframes meet leg starts)
        merged = []
        for kf in out:
            if merged and abs(kf[0] - merged[-1][0]) < 1e-9:
                merged[-1] = kf
            else:
                merged.append(kf)
        return [tuple(float(v) for v in kf) for kf in merged]


@dataclass
class NoiseZone:
    center: np.ndarray
    radius: float
    sigma_db: float
    name: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseZone":
        return cls(np.array(d["at"], dtype=float), float(d.get("radius", 2.0)), float(d.get("sigma_db", 4.0)), str(d.get("name", "")))


@dataclass
class Movers:
    """All movers and noise zones of a venue."""

    movers: list[Mover] = field(default_factory=list)
    zones: list[NoiseZone] = field(default_factory=list)

    @classmethod
    def from_dicts(cls, movers: list[dict] | None, zones: list[dict] | None = None) -> "Movers":
        return cls([Mover.from_dict(m) for m in movers or []], [NoiseZone.from_dict(z) for z in zones or []])

    def __len__(self) -> int:
        return len(self.movers)

    def __bool__(self) -> bool:
        return bool(self.movers or self.zones)

    def positions(self, t: float) -> np.ndarray:
        """(M, 2) mover positions at time ``t`` (s)."""
        return np.array([m.position(t) for m in self.movers]).reshape(-1, 2)

    def headings(self, t: float) -> np.ndarray:
        return np.array([m.heading(t) for m in self.movers])

    def blocking_loss(self, robot_xy, ap_xy, t: float) -> np.ndarray:
        """Extra attenuation (dB) of each robot->AP link from bodies near it, shape (N, A)."""
        robot = np.atleast_2d(np.asarray(robot_xy, dtype=float))
        aps = np.atleast_2d(np.asarray(ap_xy, dtype=float))
        out = np.zeros((len(robot), len(aps)))
        if not self.movers:
            return out
        m = self.positions(t)  # (M, 2)
        L = np.array([mv.loss_db for mv in self.movers])
        r = np.array([mv.radius for mv in self.movers])
        p = robot[:, None, None, :]  # (N, 1, 1, 2)
        q = aps[None, :, None, :]  # (1, A, 1, 2)
        seg = q - p
        len2 = np.maximum((seg**2).sum(-1), 1e-12)
        s = np.clip(((m[None, None] - p) * seg).sum(-1) / len2, 0.0, 1.0)  # (N, A, M)
        closest = p + s[..., None] * seg
        d2 = ((m[None, None] - closest) ** 2).sum(-1)
        return (L * np.exp(-d2 / r**2)).sum(-1)

    def extra_sigma(self, xy) -> np.ndarray:
        """Extra temporal noise (dB, 1 sigma) at each point from noise zones, shape (N,)."""
        pts = np.atleast_2d(np.asarray(xy, dtype=float))
        out = np.zeros(len(pts))
        for z in self.zones:
            inside = np.hypot(*(pts - z.center).T) <= z.radius
            out = np.where(inside, np.hypot(out, z.sigma_db), out)
        return out

    def near(self, xy, t: float, dist: float, moving_only: bool = True) -> list[tuple[Mover, np.ndarray]]:
        """Movers within ``dist`` of ``xy`` at time ``t`` (with their positions)."""
        out = []
        for mv in self.movers:
            if moving_only and mv.static:
                continue
            p = mv.position(t)
            if np.hypot(*(p - np.asarray(xy, dtype=float)[:2])) <= dist:
                out.append((mv, p))
        return out
