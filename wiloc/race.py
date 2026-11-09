"""Live comparison of localisation methods against ground truth ("the race").

``RaceTracker`` receives, for each time step, the true position and every
method's estimate (with its 2x2 covariance when it has one) and keeps

* a bounded ring buffer of samples, so a demo can run for hours with flat memory;
* a rolling median error per method over ``window`` seconds: the fair
  leaderboard number.  Samples within ``settle`` seconds after a kidnapping
  are left out of it: right after a teleport every method is wrong, and that
  says nothing about how well it tracks;
* the recovery time after the latest kidnapping, per method: time until its
  error stays below ``recover_m`` for ``recover_hold`` seconds;
* the 95 % coverage per method: how often the truth lies inside the method's
  own 95 % uncertainty ellipse (a calibrated filter scores about 95 %).

Kidnappings are detected from the truth itself: a jump of more than
``jump_m`` between two consecutive samples.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

CHI2_95_2D = 5.991  # 95 % quantile of chi-square with 2 dof


@dataclass
class RaceTracker:
    methods: tuple[str, ...]
    window: float = 30.0
    settle: float = 20.0
    recover_m: float = 1.0
    recover_hold: float = 3.0
    coverage_window: float = 300.0
    jump_m: float = 1.0
    maxlen: int = 36000  # one hour at 10 Hz

    def __post_init__(self):
        self.samples: deque = deque(maxlen=self.maxlen)  # (t, err dict, inside dict)
        self.last_truth = None
        self.kidnap_t: float | None = None
        self.n_kidnaps = 0
        self.recovery: dict[str, float | None] = {m: None for m in self.methods}
        self._since_kidnap: dict[str, deque] = {m: deque() for m in self.methods}  # (t, err) in the hold window
        self.recoveries: dict[str, list[float]] = {m: [] for m in self.methods}

    # ------------------------------------------------------------ input
    def add(self, t: float, truth_xy, estimates: dict) -> dict[str, float]:
        """Record one step.  ``estimates``: method -> (xy, cov2x2 or None); missing methods are skipped."""
        truth_xy = np.asarray(truth_xy, dtype=float)[:2]
        if self.last_truth is not None and np.hypot(*(truth_xy - self.last_truth)) > self.jump_m:
            self.kidnap(t)
        self.last_truth = truth_xy
        err, inside = {}, {}
        for m, (xy, cov) in estimates.items():
            if m not in self.recovery or xy is None:
                continue
            d = np.asarray(xy, dtype=float)[:2] - truth_xy
            err[m] = float(np.hypot(*d))
            if cov is not None:
                c = np.asarray(cov, dtype=float)[:2, :2]
                try:
                    inside[m] = bool(d @ np.linalg.solve(c, d) <= CHI2_95_2D)
                except np.linalg.LinAlgError:
                    pass
        self.samples.append((t, err, inside))
        self._track_recovery(t, err)
        return err

    def kidnap(self, t: float) -> None:
        self.kidnap_t = t
        self.n_kidnaps += 1
        self.recovery = {m: None for m in self.methods}
        self._since_kidnap = {m: deque() for m in self.methods}

    def _track_recovery(self, t, err):
        """Recovered once the median error over ``recover_hold`` s is below ``recover_m``.

        A median rather than "every sample": a memoryless method such as WKNN is
        noisy around a correct mean and would otherwise never count as recovered.
        The median flips when half the window is past the switch, so the
        recovery time is taken at the window's midpoint.
        """
        if self.kidnap_t is None:
            return
        for m, e in err.items():
            if self.recovery[m] is not None:
                continue
            win = self._since_kidnap[m]
            win.append((t, e))
            while t - win[0][0] > self.recover_hold:
                win.popleft()
            if t - self.kidnap_t >= self.recover_hold and np.median([v for _, v in win]) < self.recover_m:
                self.recovery[m] = max(0.5 * (win[0][0] + t) - self.kidnap_t, 0.0)
                self.recoveries[m].append(self.recovery[m])

    # ------------------------------------------------------------ queries
    def settling(self, t: float) -> bool:
        return self.kidnap_t is not None and t - self.kidnap_t < self.settle

    def _recent(self, t, span, skip_settling):
        out = []
        for s in reversed(self.samples):
            if t - s[0] > span:
                break
            if skip_settling and self.kidnap_t is not None and 0 <= s[0] - self.kidnap_t < self.settle:
                continue
            out.append(s)
        return out

    def rolling_median(self, t: float) -> dict[str, float]:
        """Median error per method over the last ``window`` seconds, kidnap aftermath excluded."""
        recent = self._recent(t, self.window, skip_settling=True)
        out = {}
        for m in self.methods:
            e = [s[1][m] for s in recent if m in s[1]]
            if e:
                out[m] = float(np.median(e))
        return out

    def coverage(self, t: float) -> dict[str, float]:
        recent = self._recent(t, self.coverage_window, skip_settling=True)
        out = {}
        for m in self.methods:
            v = [s[2][m] for s in recent if m in s[2]]
            if len(v) >= 10:
                out[m] = float(np.mean(v))
        return out

    def overall(self, method: str) -> dict | None:
        """Error statistics (``metrics.summarize``) over the whole buffer, kidnaps included."""
        from .metrics import summarize

        e = [s[1][method] for s in self.samples if method in s[1]]
        return summarize(np.array(e)) if e else None

    def leader(self, t: float) -> str | None:
        med = self.rolling_median(t)
        return min(med, key=med.get) if med else None

    def summary(self, t: float) -> dict:
        return {
            "median": self.rolling_median(t),
            "leader": self.leader(t),
            "coverage95": self.coverage(t),
            "recovery_s": dict(self.recovery),
            "since_kidnap_s": None if self.kidnap_t is None else t - self.kidnap_t,
            "settling": self.settling(t),
            "kidnaps": self.n_kidnaps,
        }
