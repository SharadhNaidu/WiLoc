"""Offline phase: site survey -> radio map (fingerprint database).

``RadioMap``       sparse fingerprints at surveyed reference points (RPs):
                   per-AP mean, standard deviation and detection rate.
``DenseRadioMap``  a continuous map on a fine grid, obtained by Gaussian-
                   process regression over the RPs (Ferris, Haehnel & Fox,
                   RSS 2006).  The GP uses a log-distance path-loss model as
                   its prior mean and an RBF kernel on the residual; its
                   predictive variance tells the particle filter where the
                   map is uncertain.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .environment import FloorPlan
from .radio import RadioSimulator, bilinear

NOT_HEARD_DBM = -100.0  # value substituted for undetected APs


def fill_missing(z: np.ndarray, floor: float = NOT_HEARD_DBM) -> np.ndarray:
    z = np.array(z, dtype=float, copy=True)
    z[np.isnan(z)] = floor
    return z


@dataclass
class RadioMap:
    positions: np.ndarray  # (R, 2)
    mean: np.ndarray  # (R, A) dBm, NaN-free (floor substituted)
    std: np.ndarray  # (R, A) dB
    detect_rate: np.ndarray  # (R, A) in [0, 1]
    ap_ids: list[str]
    ap_positions: np.ndarray  # (A, 2)

    @classmethod
    def survey(
        cls,
        sim: RadioSimulator,
        points: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> "RadioMap":
        """Collect ``n_samples`` scans at each reference point (the site survey)."""
        z = sim.measure(points, rng, n_samples=max(n_samples, 2))  # (R, S, A)
        if n_samples == 1:
            z = z[:, :1, :]
        heard = ~np.isnan(z)
        rate = heard.mean(axis=1)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(np.where(heard, z, np.nan), axis=1)
            std = np.nanstd(np.where(heard, z, np.nan), axis=1)
        mean = np.where(rate > 0, mean, NOT_HEARD_DBM)
        # Mix in the "not heard" mass so that intermittently heard APs get a
        # realistic (lower) mean and wider spread.
        mean = rate * mean + (1 - rate) * NOT_HEARD_DBM
        std = np.where(rate > 0, np.nan_to_num(std), 0.0)
        return cls(
            positions=np.asarray(points, dtype=float),
            mean=mean,
            std=std,
            detect_rate=rate,
            ap_ids=list(sim.ap_ids),
            ap_positions=sim.fp.ap_positions,
        )

    @property
    def n_aps(self) -> int:
        return len(self.ap_ids)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            positions=self.positions,
            mean=self.mean,
            std=self.std,
            detect_rate=self.detect_rate,
            ap_ids=np.array(self.ap_ids),
            ap_positions=self.ap_positions,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RadioMap":
        d = np.load(path, allow_pickle=False)
        return cls(
            positions=d["positions"],
            mean=d["mean"],
            std=d["std"],
            detect_rate=d["detect_rate"],
            ap_ids=[str(s) for s in d["ap_ids"]],
            ap_positions=d["ap_positions"],
        )


# --------------------------------------------------------------------------
# Gaussian-process radio map
# --------------------------------------------------------------------------


def fit_log_distance(pos, rssi, ap_pos):
    """Least-squares fit of rssi = A - 10 n log10(d); returns (A, n)."""
    d = np.maximum(np.linalg.norm(pos - ap_pos, axis=1), 0.5)
    H = np.column_stack([np.ones_like(d), -10 * np.log10(d)])
    (a, n), *_ = np.linalg.lstsq(H, rssi, rcond=None)
    return float(a), float(n)


def _rbf(a, b, ell, sf):
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return sf**2 * np.exp(-0.5 * d2 / ell**2)


class GaussianProcess2D:
    """Minimal exact GP regressor (RBF kernel) with ML-II hyper-parameters."""

    def __init__(self, ell=2.0, sf=4.0, sn=1.0):
        self.theta = np.log([ell, sf, sn])

    def _nll(self, theta, X, y):
        ell, sf, sn = np.exp(theta)
        K = _rbf(X, X, ell, sf) + (sn**2 + 1e-6) * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y) * np.log(2 * np.pi)

    def fit(self, X, y, optimize=True):
        self.X, self.y = np.asarray(X), np.asarray(y)
        if optimize:
            bounds = [(np.log(0.3), np.log(15.0)), (np.log(0.1), np.log(30.0)), (np.log(0.05), np.log(10.0))]
            res = minimize(self._nll, self.theta, args=(self.X, self.y), method="L-BFGS-B", bounds=bounds)
            self.theta = res.x
        ell, sf, sn = np.exp(self.theta)
        K = _rbf(self.X, self.X, ell, sf) + (sn**2 + 1e-6) * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    @property
    def hyper(self):
        return dict(zip(("ell", "sf", "sn"), np.exp(self.theta)))

    def predict(self, Xs, chunk=4096):
        ell, sf, _ = np.exp(self.theta)
        mu, var = np.empty(len(Xs)), np.empty(len(Xs))
        for i in range(0, len(Xs), chunk):
            ks = _rbf(Xs[i : i + chunk], self.X, ell, sf)
            mu[i : i + chunk] = ks @ self.alpha
            v = np.linalg.solve(self.L, ks.T)
            var[i : i + chunk] = sf**2 - np.sum(v * v, axis=0)
        return mu, np.maximum(var, 0.0)


@dataclass
class DenseRadioMap:
    """Per-AP RSSI mean / std on a regular grid, for fast likelihood lookups."""

    res: float
    mean: np.ndarray  # (A, ny, nx)
    std: np.ndarray  # (A, ny, nx)
    ap_ids: list[str]
    width: float
    height: float

    @classmethod
    def from_radio_map(
        cls,
        rmap: RadioMap,
        floorplan: FloorPlan,
        res: float = 0.25,
        method: str = "gp",
        min_std: float = 1.5,
    ) -> "DenseRadioMap":
        """Interpolate the survey onto a dense grid.

        method = "gp"     GP regression on top of a fitted log-distance mean
                 "model"  fitted log-distance model only (no survey detail)
        """
        nx = int(np.ceil(floorplan.width / res)) + 1
        ny = int(np.ceil(floorplan.height / res)) + 1
        gx, gy = np.meshgrid(np.arange(nx) * res, np.arange(ny) * res)
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        # Measurement spread observed during the survey (temporal noise).
        heard = rmap.detect_rate > 0.5
        temporal = float(np.median(rmap.std[heard])) if heard.any() else 2.0

        means, stds = [], []
        for j in range(rmap.n_aps):
            y = rmap.mean[:, j]
            # Fit the path-loss prior on reliably heard RPs only; the floor
            # value substituted for missed readings would bias it.
            ok = rmap.detect_rate[:, j] > 0.9
            if ok.sum() < 3:
                ok = np.ones(len(y), dtype=bool)
            a, n = fit_log_distance(rmap.positions[ok], y[ok], rmap.ap_positions[j])
            prior = lambda p: a - 10 * n * np.log10(np.maximum(np.linalg.norm(p - rmap.ap_positions[j], axis=1), 0.5))
            if method == "model":
                resid = y - prior(rmap.positions)
                mu = prior(grid)
                sd = np.full(len(grid), np.std(resid))
            else:
                gp = GaussianProcess2D(ell=2.0, sf=max(np.std(y - prior(rmap.positions)), 1.0), sn=1.0)
                gp.fit(rmap.positions, y - prior(rmap.positions))
                r, var = gp.predict(grid)
                mu = prior(grid) + r
                sd = np.sqrt(var + temporal**2)
            means.append(np.maximum(mu, NOT_HEARD_DBM).reshape(ny, nx))
            stds.append(np.maximum(sd, min_std).reshape(ny, nx))
        return cls(res, np.array(means), np.array(stds), list(rmap.ap_ids), floorplan.width, floorplan.height)

    def lookup(self, pts: np.ndarray):
        """(mean, std) each (N, A) at metric points."""
        return bilinear(self.mean, self.res, pts), bilinear(self.std, self.res, pts)
