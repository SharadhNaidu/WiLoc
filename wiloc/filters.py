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
                        kidnapped-robot recovery.
``WifiEKF``         extended Kalman filter: odometry prediction plus a Wi-Fi
                    position fix (e.g. WKNN) as a direct (x, y) measurement,
                    with Mahalanobis outlier gating.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr

from .environment import FloorPlan
from .fingerprint import DenseRadioMap
from .geometry import wrap_angle
from .motion import OdometryMotionModel


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
        rng: np.random.Generator | None = None,
    ):
        self.max_inject = max_inject
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
    def init_pose(self, pose, std=(0.3, 0.3, 0.2)):
        self.particles = np.asarray(pose) + self.rng.standard_normal((self.n, 3)) * np.asarray(std)
        self.particles[:, 2] = wrap_angle(self.particles[:, 2])
        self.logw[:] = -np.log(self.n)

    def init_global(self):
        """Uniform over free space with random heading (global localisation)."""
        self.particles = self._random_particles(self.n)
        self.logw[:] = -np.log(self.n)

    def _random_particles(self, k):
        xy = self.fp.sample_free(k, self.rng)
        th = self.rng.uniform(-np.pi, np.pi, size=(k, 1))
        return np.hstack([xy, th])

    # ----------------------------------------------------------- steps
    def predict(self, delta):
        if delta[1] < 1e-4 and abs(delta[0]) + abs(delta[2]) < 1e-4:
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
        # Augmented MCL bookkeeping on a per-AP-normalised likelihood scale.
        w_avg = float(np.exp(np.logaddexp.reduce(self.logw + ll / max(len(z), 1))))
        if self.w_slow is None:
            self.w_slow = self.w_fast = w_avg
        self.w_slow += self.a_slow * (w_avg - self.w_slow)
        self.w_fast += self.a_fast * (w_avg - self.w_fast)
        self.logw += ll
        self._normalise()
        if self.ess() < self.n / 2:
            self.resample()

    def resample(self):
        """Low-variance (systematic) resampling plus random injection."""
        w = np.exp(self.logw)
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(w), positions), self.n - 1)
        self.particles = self.particles[idx]
        p_inject = max(0.0, 1.0 - self.w_fast / self.w_slow) if self.w_slow else 0.0
        k = int(self.rng.binomial(self.n, min(p_inject, self.max_inject)))
        if k > 0:
            self.particles[self.rng.choice(self.n, k, replace=False)] = self._random_particles(k)
        self.logw[:] = -np.log(self.n)

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
        sel = np.ones(self.n, dtype=bool)
        for _ in range(iters):
            near = np.hypot(*(self.particles[:, :2] - xy).T) < radius
            if w[near].sum() < 0.2:
                break
            sel = near
            xy = w[sel] @ self.particles[sel, :2] / w[sel].sum()
        th = np.arctan2(w[sel] @ np.sin(self.particles[sel, 2]), w[sel] @ np.cos(self.particles[sel, 2]))
        return np.array([xy[0], xy[1], th]), cov


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
