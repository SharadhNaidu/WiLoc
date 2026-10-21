"""Online phase: single-scan position estimators.

Every estimator is fitted on a :class:`~wiloc.fingerprint.RadioMap` and maps
an RSSI scan ``z`` (N, A) -- NaN for APs that were not heard -- to positions
(N, 2).  ``predict_with_cov`` additionally returns a 2x2 covariance per scan,
which is what the EKF uses as the measurement noise of a Wi-Fi "position fix".

Implemented:
  * NN / KNN / WKNN  deterministic matching in signal space (RADAR,
    Bahl & Padmanabhan, INFOCOM 2000);
  * Probabilistic    per-RP Gaussian likelihood and posterior mean over RPs
    (Horus, Youssef & Agrawala, MobiSys 2005);
  * Trilateration    model-based baseline: invert a fitted log-distance model
    to ranges and solve non-linear least squares (no fingerprints).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .fingerprint import NOT_HEARD_DBM, RadioMap, fill_missing, fit_log_distance


class Estimator:
    name = "base"

    def fit(self, rmap: RadioMap) -> "Estimator":
        self.rmap = rmap
        return self

    def predict(self, z: np.ndarray) -> np.ndarray:
        return self.predict_with_cov(z)[0]

    def predict_with_cov(self, z: np.ndarray):
        """Positions (N, 2) and covariances (N, 2, 2) for scans z (N, A).

        A scan in which no AP was heard carries no position information, so
        it gets the centre of the surveyed area with a covariance spanning
        the whole area instead of a falsely confident match.
        """
        z = np.atleast_2d(np.asarray(z, dtype=float))
        xy, cov = self._predict_with_cov(z)
        empty = np.isnan(z).all(axis=1)
        if empty.any():
            pos = self.rmap.positions
            xy[empty] = pos.mean(axis=0)
            cov[empty] = np.cov(pos.T)
        return xy, cov

    def _predict_with_cov(self, z: np.ndarray):
        raise NotImplementedError


def _weighted_mean_cov(pos, w, min_var=0.25):
    """Weighted mean and covariance of candidate positions.

    pos: (N, K, 2), w: (N, K) normalised.  ``min_var`` keeps the covariance
    from collapsing when all neighbours coincide.
    """
    mu = np.einsum("nk,nkd->nd", w, pos)
    d = pos - mu[:, None, :]
    cov = np.einsum("nk,nki,nkj->nij", w, d, d) + min_var * np.eye(2)
    return mu, cov


class KNN(Estimator):
    """k-nearest neighbours in signal space; ``weighted`` gives WKNN."""

    def __init__(self, k: int = 3, weighted: bool = False, p: float = 2.0, eps: float = 1e-3):
        self.k, self.weighted, self.p, self.eps = k, weighted, p, eps
        self.name = ("WKNN" if weighted else ("NN" if k == 1 else "KNN")) + (f" (k={k})" if k > 1 else "")

    def signal_distance(self, z):
        z = fill_missing(z)
        diff = np.abs(z[:, None, :] - self.rmap.mean[None, :, :])  # (N, R, A)
        return np.sum(diff**self.p, axis=-1) ** (1.0 / self.p)

    def _predict_with_cov(self, z):
        z = np.atleast_2d(z)
        dist = self.signal_distance(z)
        k = min(self.k, dist.shape[1])
        idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(dist, idx, axis=1)
        w = 1.0 / (dk + self.eps) if self.weighted else np.ones_like(dk)
        w /= w.sum(axis=1, keepdims=True)
        return _weighted_mean_cov(self.rmap.positions[idx], w)


class Probabilistic(Estimator):
    """Horus-style Bayesian estimator with per-RP Gaussian RSSI models.

    log p(z | RP_r) = sum_a log N(z_a; mu_ra, sigma_ra^2); a missing AP is
    scored with the RP's probability of *not* hearing it.  The estimate is
    the posterior mean over the ``k`` most likely RPs (k=None: all RPs).
    """

    def __init__(self, k: int | None = 5, min_sigma: float = 2.0, temper: float = 1.0):
        self.k, self.min_sigma, self.temper = k, min_sigma, temper
        self.name = "Probabilistic"

    def log_likelihood(self, z):
        z = np.atleast_2d(z)
        mu = self.rmap.mean[None]
        sig = np.maximum(self.rmap.std, self.min_sigma)[None]
        heard = ~np.isnan(z)[:, None, :]
        zz = fill_missing(z)[:, None, :]
        ll_heard = -0.5 * ((zz - mu) / sig) ** 2 - np.log(sig)
        p_miss = np.clip(1.0 - self.rmap.detect_rate[None], 0.02, 1.0)
        p_hit = np.clip(self.rmap.detect_rate[None], 0.02, 1.0)
        ll = np.where(heard, ll_heard + np.log(p_hit), np.log(p_miss))
        return self.temper * ll.sum(axis=-1)  # (N, R)

    def _predict_with_cov(self, z):
        ll = self.log_likelihood(z)
        if self.k is not None and self.k < ll.shape[1]:
            idx = np.argpartition(-ll, self.k - 1, axis=1)[:, : self.k]
        else:
            idx = np.broadcast_to(np.arange(ll.shape[1]), ll.shape)
        llk = np.take_along_axis(ll, idx, axis=1)
        w = np.exp(llk - llk.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        return _weighted_mean_cov(self.rmap.positions[idx], w)


class Trilateration(Estimator):
    """Range-based baseline using a per-AP log-distance fit of the survey."""

    name = "Trilateration"

    def fit(self, rmap):
        super().fit(rmap)
        self.models = []
        for j in range(rmap.n_aps):
            ok = rmap.detect_rate[:, j] > 0.9
            if ok.sum() < 3:
                self.models.append(None)  # never reliably heard: no range model
                continue
            self.models.append(fit_log_distance(rmap.positions[ok], rmap.mean[ok, j], rmap.ap_positions[j]))
        self.usable = np.array([m is not None for m in self.models])
        lo, hi = rmap.positions.min(0), rmap.positions.max(0)
        self.bounds, self.x0 = (lo, hi), rmap.positions.mean(0)
        return self

    def _predict_with_cov(self, z):
        z = np.atleast_2d(z)
        aps = self.rmap.ap_positions
        out, covs = np.empty((len(z), 2)), np.empty((len(z), 2, 2))
        for i, zi in enumerate(z):
            ok = ~np.isnan(zi) & (zi > NOT_HEARD_DBM + 5) & self.usable
            if ok.sum() < 3:
                out[i], covs[i] = self.x0, np.eye(2) * 25.0
                continue
            a = np.array([self.models[j][0] for j in np.flatnonzero(ok)])
            n = np.array([max(self.models[j][1], 1.0) for j in np.flatnonzero(ok)])
            r = 10 ** ((a - zi[ok]) / (10 * n))
            w = 1.0 / np.maximum(r, 0.5)  # nearer APs are more reliable
            f = lambda x: w * (np.linalg.norm(aps[ok] - x, axis=1) - r)
            sol = least_squares(f, self.x0, bounds=self.bounds)
            out[i] = sol.x
            J = sol.jac
            resid_var = max(np.mean(sol.fun**2), 1e-2)
            covs[i] = np.linalg.pinv(J.T @ J) * resid_var + 0.25 * np.eye(2)
        return out, covs


def default_estimators():
    return [
        KNN(k=1),
        KNN(k=4),
        KNN(k=4, weighted=True),
        Probabilistic(k=5),
        Trilateration(),
    ]
