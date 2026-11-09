"""Recursive Bayes filters fusing wheel odometry with Wi-Fi RSSI.

``ParticleFilter``  Monte Carlo localisation (Thrun et al., *Probabilistic
                    Robotics*, ch. 8) with
                      - the odometry motion model for prediction,
                      - a map constraint (particles may not pass through walls),
                      - an RSSI measurement model read from a dense GP radio
                        map, including censored "AP not heard" readings,
                      - low-variance resampling triggered by the effective
                        sample size, and
                      - augmented-MCL random-particle injection
                        (w_slow / w_fast) for global localisation and
                        kidnapped-robot recovery,
                    and three options (all off by default):
                      - KLD sampling (Fox 2003): the particle count adapts,
                        few when the filter is sure, many when it is lost;
                      - a cluster-based estimate: the pose of the heaviest
                        particle cluster (``modes``), never a point between
                        two hypotheses or inside a wall;
                      - a lost-triggered re-seed of part of the particles;
                      - sensor-resetting injection (Lenser & Veloso 2000):
                        part of the injected particles are drawn around the
                        latest Wi-Fi fix (``hint``) with every heading, so
                        after a kidnapping the right position is covered
                        densely enough for motion to pick the right heading.
``WifiEKF``         extended Kalman filter: odometry prediction plus a Wi-Fi
                    position fix (e.g. WKNN) as a direct (x, y) measurement,
                    with Mahalanobis outlier gating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.special import log_ndtr

from .environment import FloorPlan
from .fingerprint import DenseRadioMap
from .geometry import wrap_angle
from .motion import OdometryMotionModel


@dataclass
class KLDConfig:
    """KLD-sampling bounds (Fox, "Adapting the sample size in particle filters", 2003).

    After resampling, particles are drawn until the count reaches the number
    needed so that, with probability ``1 - delta`` (``z`` is its normal
    quantile), the KL divergence between the sample and the true posterior
    stays below ``eps`` -- a function of ``k``, the number of histogram bins
    (``bin_xy`` x ``bin_xy`` x ``bin_yaw``) the particles occupy.
    """

    n_min: int = 500
    n_max: int = 5000
    bin_xy: float = 0.5
    bin_yaw: float = float(np.radians(20.0))
    eps: float = 0.05
    z: float = 2.326  # delta = 0.01

    def n_required(self, k) -> np.ndarray:
        """Particles needed when ``k`` bins are occupied (Wilson-Hilferty chi-square quantile)."""
        k = np.asarray(k, dtype=float)
        out = np.full(k.shape, float(self.n_min))
        m = k >= 2
        a = 2.0 / (9.0 * (k[m] - 1.0))
        out[m] = (k[m] - 1.0) / (2.0 * self.eps) * (1.0 - a + np.sqrt(a) * self.z) ** 3
        return np.clip(out, self.n_min, self.n_max)


class ParticleFilter:
    def __init__(
        self,
        floorplan: FloorPlan,
        radio_map: DenseRadioMap,
        n_particles: int = 1000,
        motion_model: OdometryMotionModel | None = None,
        sensitivity_dbm: float = -95.0,
        temper: float = 0.25,
        outlier_prob: float = 0.05,
        alpha_slow: float = 0.01,
        alpha_fast: float = 0.3,
        wall_penalty: float = 1e-3,
        max_inject: float = 0.05,
        inject_threshold: float = 0.3,
        rng: np.random.Generator | None = None,
        kld: KLDConfig | None = None,
        estimator: str = "meanshift",
        reseed_fraction: float = 0.0,
        reseed_after: int = 6,
        hint_fraction: float = 0.0,
    ):
        """``kld``: adapt the particle count (``n_particles`` is then ignored).
        ``estimator``: "meanshift" (default) or "cluster" (heaviest mode).
        ``reseed_fraction``: after ``reseed_after`` consecutive lost updates,
        replace this fraction of particles with random ones (0 = never).
        ``hint_fraction``: share of injected / re-seeded particles drawn
        around ``self.hint`` = (xy, 2x2 cov), e.g. the WKNN fix, set by the caller.
        """
        if estimator not in ("meanshift", "cluster"):
            raise ValueError(f"estimator must be 'meanshift' or 'cluster', not {estimator!r}")
        self.kld, self.estimator = kld, estimator
        self.reseed_fraction, self.reseed_after = reseed_fraction, reseed_after
        self.hint_fraction, self.hint = hint_fraction, None
        self.n_lost = self.n_reseeds = 0
        if kld is not None:
            n_particles = kld.n_max
        self.max_inject, self.inject_threshold = max_inject, inject_threshold
        self.n_resamples = self.n_injected = 0  # bookkeeping for monitoring
        self.fp, self.map = floorplan, radio_map
        self.n = n_particles
        self.motion = motion_model or OdometryMotionModel()
        self.sens = sensitivity_dbm
        self.temper, self.eps = temper, outlier_prob
        self.a_slow, self.a_fast = alpha_slow, alpha_fast
        self.wall_penalty = wall_penalty
        self.rng = rng or np.random.default_rng()
        self.w_slow = self.w_fast = None
        self.particles = np.zeros((n_particles, 3))
        self.logw = np.full(n_particles, -np.log(n_particles))

    # ----------------------------------------------------------- init
    def _set_count(self):
        """With KLD a (re)initialised filter starts with the full count: it knows little yet."""
        if self.kld is not None:
            self.n = self.kld.n_max
            self.logw = np.full(self.n, -np.log(self.n))
        self.n_lost = 0

    def init_pose(self, pose, std=(0.3, 0.3, 0.2)):
        self._set_count()
        self.particles = np.asarray(pose) + self.rng.standard_normal((self.n, 3)) * np.asarray(std)
        self.particles[:, 2] = wrap_angle(self.particles[:, 2])
        self.logw[:] = -np.log(self.n)

    def init_global(self):
        """Uniform over free space with random heading (global localisation)."""
        self._set_count()
        self.particles = self._random_particles(self.n)
        self.logw[:] = -np.log(self.n)

    def _random_particles(self, k):
        xy = self.fp.sample_free(k, self.rng)
        th = self.rng.uniform(-np.pi, np.pi, size=(k, 1))
        return np.hstack([xy, th])

    def _inject_particles(self, k):
        """Particles for injection / re-seeding: uniform, or partly around the hint (any heading)."""
        if self.hint_fraction <= 0 or self.hint is None:
            return self._random_particles(k)
        out = self._random_particles(k)
        k_h = int(self.rng.binomial(k, self.hint_fraction))
        if k_h:
            xy, cov = self.hint
            cov = np.asarray(cov, dtype=float)[:2, :2] + 0.25 * np.eye(2)  # at least 0.5 m spread
            pts = self.rng.multivariate_normal(np.asarray(xy, dtype=float)[:2], cov, size=k_h)
            pts = pts[self.fp.is_free(pts, 0.1)]
            out[: len(pts), :2] = pts
        return out

    # ----------------------------------------------------------- steps
    def predict(self, delta):
        if abs(delta[1]) < 1e-4 and abs(delta[0]) + abs(delta[2]) < 1e-4:
            return
        new = self.motion.sample(self.particles, delta, self.rng)
        blocked = self.fp.crosses_wall(self.particles[:, :2], new[:, :2]) | ~self.fp.is_free(new[:, :2], 0.05)
        self.logw[blocked] += np.log(self.wall_penalty)
        # Particles outside the building are impossible, not just unlikely.
        x, y = new[:, 0], new[:, 1]
        outside = (x < 0) | (x > self.fp.width) | (y < 0) | (y > self.fp.height)
        self.logw[outside] = -np.inf
        self.particles = new
        if outside.all():
            self.init_global()  # filter lost: restart global localisation
            return
        self._normalise()

    def measurement_loglik(self, z, xy):
        """log p(z | x) for scan z (A,) at positions xy (N, 2)."""
        mu, sd = self.map.lookup(xy)
        heard = ~np.isnan(z)
        ll = np.zeros(len(xy))
        if heard.any():
            r = (z[heard][None, :] - mu[:, heard]) / sd[:, heard]
            gauss = -0.5 * r**2 - np.log(sd[:, heard] * np.sqrt(2 * np.pi))
            # Robust mixture with a uniform over a 100 dB range.
            ll += np.logaddexp(np.log(1 - self.eps) + gauss, np.log(self.eps / 100.0)).sum(axis=1)
        if (~heard).any():
            # Censored reading: P(RSSI < sensitivity).
            ll += np.logaddexp(
                np.log(1 - self.eps) + log_ndtr((self.sens - mu[:, ~heard]) / sd[:, ~heard]), np.log(self.eps)
            ).sum(axis=1)
        return self.temper * ll

    def update(self, z):
        z = np.asarray(z, dtype=float)
        ll = self.measurement_loglik(z, self.particles[:, :2])
        # Augmented-MCL bookkeeping on the (tempered) total likelihood: per-AP
        # averaging flattens the drop after a kidnapping too much to detect it.
        w_avg = float(np.exp(np.logaddexp.reduce(self.logw + ll)))
        if self.w_slow is None:
            self.w_slow = self.w_fast = w_avg
        self.w_slow += self.a_slow * (w_avg - self.w_slow)
        self.w_fast += self.a_fast * (w_avg - self.w_fast)
        self.logw += ll
        self._normalise()
        # Resample when the weights degenerate, and also whenever the filter
        # looks lost: after a kidnapping all particles are *equally* wrong, so
        # the weights stay even and injection would otherwise never happen.
        if self.ess() < self.n / 2 or self.lost():
            self.resample()
        self._maybe_reseed()

    def _maybe_reseed(self):
        """Optional last resort: lost for several updates in a row -> spread part of the particles."""
        self.n_lost = self.n_lost + 1 if self.lost() else 0
        if self.reseed_fraction <= 0 or self.n_lost < self.reseed_after:
            return
        k = int(round(self.reseed_fraction * self.n))
        if k > 0:
            self.particles[self.rng.choice(self.n, k, replace=False)] = self._inject_particles(k)
            self.logw[:] = -np.log(self.n)
        self.n_lost = 0
        self.n_reseeds += 1

    def lost(self) -> bool:
        """Augmented-MCL alarm: short-term likelihood well below the long-term one."""
        return bool(self.w_slow) and self.w_fast / self.w_slow < self.inject_threshold

    def _p_inject(self) -> float:
        ratio = self.w_fast / self.w_slow if self.w_slow else 1.0
        return min(max(0.0, 1.0 - ratio / self.inject_threshold), self.max_inject)

    def resample(self):
        """Low-variance (systematic) resampling plus random injection."""
        if self.kld is not None:
            return self._resample_kld()
        w = np.exp(self.logw)
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(w), positions), self.n - 1)
        self.particles = self.particles[idx]
        # Inject only when the short-term likelihood has clearly fallen below the
        # long-term one (robot lost or kidnapped); ordinary scan-to-scan
        # fluctuations would otherwise keep sprinkling random particles.
        ratio = self.w_fast / self.w_slow if self.w_slow else 1.0
        p_inject = max(0.0, 1.0 - ratio / self.inject_threshold)
        k = int(self.rng.binomial(self.n, min(p_inject, self.max_inject)))
        self.n_resamples += 1
        self.n_injected += k
        if k > 0:
            self.particles[self.rng.choice(self.n, k, replace=False)] = self._inject_particles(k)
        self.logw[:] = -np.log(self.n)

    def _resample_kld(self):
        """KLD resampling in AMCL's order: draw (or inject), bin, stop when the bound is met.

        Injected random particles are drawn and binned like the others, so
        after a kidnapping they occupy new bins and the count grows.  The
        candidates are a systematic sample of ``n_max`` shuffled before the
        count is cut: systematic output is ordered by particle index, and
        cutting it unshuffled would keep only the first particles.
        """
        cfg = self.kld
        m = cfg.n_max
        w = np.exp(self.logw)
        positions = (self.rng.random() + np.arange(m)) / m
        idx = np.minimum(np.searchsorted(np.cumsum(w), positions), self.n - 1)
        cand = self.particles[idx[self.rng.permutation(m)]]
        inject = self.rng.random(m) < self._p_inject()
        if inject.any():
            cand[inject] = self._inject_particles(int(inject.sum()))
        n_yaw = int(np.ceil(2 * np.pi / cfg.bin_yaw))
        bx = np.floor(cand[:, 0] / cfg.bin_xy).astype(np.int64)
        by = np.floor(cand[:, 1] / cfg.bin_xy).astype(np.int64)
        bt = (np.floor((wrap_angle(cand[:, 2]) + np.pi) / cfg.bin_yaw).astype(np.int64)) % n_yaw
        key = ((bx + 4096) * 8192 + (by + 4096)) * n_yaw + bt
        new_bin = np.zeros(m, dtype=bool)
        new_bin[np.unique(key, return_index=True)[1]] = True
        need = cfg.n_required(np.cumsum(new_bin))
        met = np.nonzero(np.arange(1, m + 1) >= need)[0]
        n = max(int(met[0]) + 1 if len(met) else m, cfg.n_min)
        self.particles = cand[:n]
        self.n = n
        self.logw = np.full(n, -np.log(n))
        self.n_resamples += 1
        self.n_injected += int(inject[:n].sum())

    def modes(self, k: int = 3, bin_m: float = 0.5) -> list[dict]:
        """The ``k`` heaviest particle clusters, heaviest first.

        Particles are binned on a ``bin_m`` grid and touching occupied bins
        (8-connected) form a cluster.  Each mode: ``xy``, ``yaw``, ``weight``
        (its share of the total), ``cov`` (2x2) and ``n`` (particles).
        """
        w = np.exp(self.logw)
        nx, ny = int(np.ceil(self.fp.width / bin_m)), int(np.ceil(self.fp.height / bin_m))
        ix = np.clip((self.particles[:, 0] / bin_m).astype(int), 0, nx - 1)
        iy = np.clip((self.particles[:, 1] / bin_m).astype(int), 0, ny - 1)
        hist = np.zeros((ny, nx))
        np.add.at(hist, (iy, ix), w)
        labels, n_lab = ndimage.label(hist > 0, structure=np.ones((3, 3)))
        lab = labels[iy, ix]
        mass = np.bincount(lab, weights=w, minlength=n_lab + 1)[1:]
        out = []
        for j in np.argsort(mass)[::-1][:k]:
            sel = lab == j + 1
            ws = w[sel]
            tot = ws.sum()
            if tot <= 0:
                continue
            xy = ws @ self.particles[sel, :2] / tot
            d = self.particles[sel, :2] - xy
            th = np.arctan2(ws @ np.sin(self.particles[sel, 2]), ws @ np.cos(self.particles[sel, 2]))
            out.append({"xy": xy, "yaw": float(th), "weight": float(tot), "cov": (ws[:, None] * d).T @ d / tot, "n": int(sel.sum())})
        return out

    def _normalise(self):
        self.logw -= np.logaddexp.reduce(self.logw)

    def ess(self):
        return 1.0 / np.sum(np.exp(2 * self.logw))

    def estimate(self, radius: float = 1.5, iters: int = 3):
        """Pose of the dominant particle cluster and the overall xy covariance.

        A plain weighted mean is pulled towards stray (e.g. freshly injected)
        particles, so the mean is refined by a few mean-shift iterations with
        a flat kernel of ``radius`` metres -- similar in spirit to AMCL
        reporting its highest-weight cluster.
        """
        w = np.exp(self.logw)
        xy = w @ self.particles[:, :2]
        d = self.particles[:, :2] - xy
        cov = (w[:, None] * d).T @ d
        if self.estimator == "cluster":
            return self._cluster_estimate(w), cov
        sel = np.ones(self.n, dtype=bool)
        for _ in range(iters):
            near = np.hypot(*(self.particles[:, :2] - xy).T) < radius
            if w[near].sum() < 0.2:
                break
            sel = near
            xy = w[sel] @ self.particles[sel, :2] / w[sel].sum()
        th = np.arctan2(w[sel] @ np.sin(self.particles[sel, 2]), w[sel] @ np.cos(self.particles[sel, 2]))
        return np.array([xy[0], xy[1], th]), cov

    def _cluster_estimate(self, w):
        """Pose of the heaviest mode, moved to its nearest free-space particle if the mean is in a wall.

        The covariance stays the global one (see ``estimate``): with two
        hypotheses the filter *is* uncertain, and the spread must say so.
        """
        top = self.modes(k=1)[0]
        xy = top["xy"]
        if not self.fp.is_free(xy[None], 0.05)[0]:
            free = self.fp.is_free(self.particles[:, :2], 0.05) & (w > 0)
            if free.any():
                cand = np.nonzero(free)[0]
                xy = self.particles[cand[np.argmin(np.hypot(*(self.particles[cand, :2] - top["xy"]).T))], :2]
        return np.array([xy[0], xy[1], top["yaw"]])


class WifiEKF:
    def __init__(
        self,
        motion_model: OdometryMotionModel | None = None,
        gate_chi2: float = 13.8,
        inflate: float = 1.0,
        max_rejections: int = 6,
    ):
        self.motion = motion_model or OdometryMotionModel()
        self.gate, self.inflate = gate_chi2, inflate
        self.max_rejections, self.rejected = max_rejections, 0
        self.x = np.zeros(3)
        self.P = np.eye(3)

    def init_pose(self, pose, cov=None):
        self.x = np.array(pose, dtype=float)
        self.P = np.diag([0.3**2, 0.3**2, 0.2**2]) if cov is None else np.asarray(cov)

    def predict(self, delta):
        self.x, G, V, M = self.motion.predict_mean_jacobians(self.x, delta)
        self.P = G @ self.P @ G.T + V @ M @ V.T

    def update_position(self, z_xy, R):
        H = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        y = np.asarray(z_xy) - H @ self.x
        S = H @ self.P @ H.T + self.inflate * np.asarray(R)
        if y @ np.linalg.solve(S, y) > self.gate:
            self.rejected += 1
            if self.rejected < self.max_rejections:
                return False  # reject outlier fix
            # Many consecutive rejections: the filter has drifted, re-open it.
            self.P[:2, :2] += np.asarray(R) * 4.0
            S = H @ self.P @ H.T + self.inflate * np.asarray(R)
        self.rejected = 0
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        return True

    def estimate(self):
        return self.x.copy(), self.P[:2, :2].copy()
