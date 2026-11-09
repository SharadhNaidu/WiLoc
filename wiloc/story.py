"""The demo's story layer: how sure the robot is, how fast it re-finds itself, and the unattended tour.

Plain logic with explicit times (seconds, any clock), so it is the same in the
ROS nodes, the Studio window and the tests.
"""

from __future__ import annotations

LOCKED, UNCERTAIN, SEARCHING = "LOCKED", "UNCERTAIN", "SEARCHING"


class ConfidenceState:
    """LOCKED / UNCERTAIN / SEARCHING from the particle filter, with hysteresis.

    Raw state per update:
      SEARCHING  the filter's lost alarm (w_fast/w_slow) is up, or the spread is huge
      LOCKED     small spread and one dominant particle cluster
      UNCERTAIN  anything in between (e.g. two look-alike rooms)
    The shown state only changes after the raw one has held for ``hold`` s;
    SEARCHING shows after ``hold_search`` s, so a kidnap is visible at once.
    """

    def __init__(self, hold: float = 2.0, hold_search: float = 0.5, lock_spread: float = 1.0,
                 lost_spread: float = 4.0, lock_weight: float = 0.75):
        self.hold, self.hold_search = hold, hold_search
        self.lock_spread, self.lost_spread, self.lock_weight = lock_spread, lost_spread, lock_weight
        self.reset()

    def reset(self, state: str = SEARCHING):
        self.state = state
        self.candidate, self.since = None, None

    def raw(self, spread: float, lost: bool, top_weight: float = 1.0) -> str:
        if lost or spread > self.lost_spread:
            return SEARCHING
        if spread < self.lock_spread and top_weight >= self.lock_weight:
            return LOCKED
        return UNCERTAIN

    def update(self, t: float, spread: float, lost: bool, top_weight: float = 1.0) -> str:
        r = self.raw(spread, lost, top_weight)
        if r == self.state:
            self.candidate = self.since = None
        elif r != self.candidate:
            self.candidate, self.since = r, t
        if self.candidate is not None:
            if t - self.since >= (self.hold_search if self.candidate == SEARCHING else self.hold):
                self.state, self.candidate, self.since = self.candidate, None, None
        return self.state


def merge_modes(modes, radius: float = 1.5) -> list[tuple[float, float, float]]:
    """Particle clusters (x, y, weight), heaviest first, with clusters within ``radius`` m
    of a heavier one folded into it (two sides of a doorway are one place, not two)."""
    out: list[list[float]] = []
    for x, y, w in sorted(((float(m[0]), float(m[1]), float(m[2])) for m in modes), key=lambda m: -m[2]):
        for o in out:
            if (o[0] - x) ** 2 + (o[1] - y) ** 2 <= radius * radius:
                o[2] += w
                break
        else:
            out.append([x, y, w])
    out.sort(key=lambda m: -m[2])
    return [(x, y, w) for x, y, w in out]


class RecoveryClock:
    """Time from a kidnap until the estimate is back within ``threshold`` m (and stays there ``hold`` s).

    ``update`` returns the recovery time exactly once per kidnap, on the update
    that confirms it; the time counts to the start of the good stretch.
    """

    def __init__(self, threshold: float = 0.8, hold: float = 2.0, grace: float = 1.0):
        self.threshold, self.hold, self.grace = threshold, hold, grace
        self.t_kidnap = None
        self.good_since = None

    def kidnap(self, t: float):
        self.t_kidnap, self.good_since = t, None

    @property
    def waiting(self) -> bool:
        return self.t_kidnap is not None

    def update(self, t: float, error: float | None):
        if self.t_kidnap is None or error is None or t - self.t_kidnap < self.grace:
            return None  # the error is stale for a moment right after the teleport
        if error > self.threshold:
            self.good_since = None
            return None
        if self.good_since is None:
            self.good_since = t
        if t - self.good_since >= self.hold:
            took = self.good_since - self.t_kidnap
            self.t_kidnap = self.good_since = None
            return took
        return None


class AttractMode:
    """Unattended tour for the stand: after ``idle`` s without a visitor, patrol,
    kidnap every ``kidnap_every`` s and move to the next venue every ``rotate_every`` s.

    ``touch(t)`` on any visitor input; ``due(t)`` lists the actions to run now
    (``"start"``, ``"kidnap"``, ``"rotate"``).  A visitor ends the tour.
    """

    def __init__(self, idle: float = 60.0, kidnap_every: float = 150.0, rotate_every: float = 360.0,
                 first_kidnap: float = 45.0):
        self.idle, self.kidnap_every, self.rotate_every = idle, kidnap_every, rotate_every
        self.first_kidnap = first_kidnap
        self.enabled = True
        self.active = False
        self.last_touch = 0.0
        self.next_kidnap = self.next_rotate = None

    def touch(self, t: float):
        self.last_touch = t
        self.active = False

    def venue_loaded(self, t: float):
        """A new venue starts its own kidnap schedule (and the rotation clock keeps running)."""
        if self.active:
            self.next_kidnap = t + self.first_kidnap

    def due(self, t: float) -> list[str]:
        if not self.enabled:
            self.active = False
            return []
        out = []
        if not self.active:
            if t - self.last_touch < self.idle:
                return []
            self.active = True
            self.next_kidnap, self.next_rotate = t + self.first_kidnap, t + self.rotate_every
            out.append("start")
        if t >= self.next_rotate:
            self.next_rotate = t + self.rotate_every
            self.next_kidnap = t + self.first_kidnap
            out.append("rotate")
        elif t >= self.next_kidnap:
            self.next_kidnap = t + self.kidnap_every
            out.append("kidnap")
        return out


def tour_stops(venues: dict[str, list[str]]) -> list[tuple[str, str]]:
    """(venue, layout) pairs for the attract rotation: every venue as installed,
    then every venue with its first alternative, and so on."""
    stops, depth = [], max((len(v) for v in venues.values()), default=0)
    for k in range(depth):
        stops += [(name, layouts[k]) for name, layouts in venues.items() if k < len(layouts)]
    return stops
