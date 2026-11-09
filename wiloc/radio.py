"""Wi-Fi RSSI propagation simulator.

The received signal strength from access point *a* at position *x* is

    RSSI_a(x, t) = P_tx,a - PL(d) - sum_k L_wall,k + S_a(x) + e_a(t)

* ``PL(d) = PL(d0) + 10 n log10(d / d0)``    log-distance path loss
  (Rappaport, *Wireless Communications*), with ``n`` close to 2 because the
  walls are modelled explicitly;
* ``sum_k L_wall,k``  multi-wall penetration loss of every wall crossed by the
  direct path (Motley-Keenan / COST 231 multi-wall model);
* ``S_a(x)``  *spatially correlated* log-normal shadowing: one fixed Gaussian
  random field per AP with the exponential autocorrelation of Gudmundson
  (1991), ``E[S(x)S(x')] = sigma_s^2 exp(-|x - x'| / d_corr)``;
* ``e_a(t)``  temporal fading / measurement noise, drawn fresh per scan.

The shadowing field is generated once per environment from ``seed`` and is
what makes fingerprinting worthwhile: it is deterministic in space, so a
survey can learn it, but it cannot be predicted from geometry alone.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, fields

import numpy as np

from .environment import FloorPlan
from .geometry import segments_intersect

SPEED_OF_LIGHT = 299_792_458.0


@dataclass
class RadioConfig:
    frequency_hz: float = 2.437e9
    path_loss_exponent: float = 2.0
    d0: float = 1.0
    shadowing_sigma_db: float = 5.0
    shadowing_corr_m: float = 3.0
    temporal_sigma_db: float = 3.0
    sensitivity_dbm: float = -95.0
    quantize: bool = True
    field_resolution_m: float = 0.25
    seed: int = 7

    @classmethod
    def from_dict(cls, d: dict | None, **overrides) -> "RadioConfig":
        """Config from a venue's ``radio:`` section; ``overrides`` that are not None win.

        Unknown keys raise, so a typo in a venue file cannot silently fall back
        to a default (and make the sensor and the localiser disagree).
        """
        merged = {**(d or {}), **{k: v for k, v in overrides.items() if v is not None}}
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(merged) - known)
        if unknown:
            raise ValueError(f"unknown radio settings {unknown}; known: {sorted(known)}")
        types = {f.name: type(f.default) for f in fields(cls)}
        return cls(**{k: types[k](v) for k, v in merged.items()})

    @property
    def pl_d0(self) -> float:
        """Free-space loss at the reference distance d0 (Friis)."""
        lam = SPEED_OF_LIGHT / self.frequency_hz
        return 20.0 * np.log10(4.0 * np.pi * self.d0 / lam)


class ShadowingField:
    """Stationary Gaussian random field with exponential covariance.

    Sampled exactly on a regular grid by circulant embedding (FFT), then
    bilinearly interpolated for arbitrary query points.
    """

    def __init__(self, width, height, sigma, corr, resolution, rng):
        self.res = resolution
        self.nx = int(np.ceil(width / resolution)) + 2
        self.ny = int(np.ceil(height / resolution)) + 2
        if sigma <= 0:
            self.values = np.zeros((self.ny, self.nx))
            return
        # Periodic embedding twice the size so the circulant covariance is valid.
        mx, my = 2 * self.nx, 2 * self.ny
        ix = np.minimum(np.arange(mx), mx - np.arange(mx)) * resolution
        iy = np.minimum(np.arange(my), my - np.arange(my)) * resolution
        d = np.hypot(ix[None, :], iy[:, None])
        cov = sigma**2 * np.exp(-d / corr)
        lam = np.clip(np.real(np.fft.fft2(cov)), 0.0, None)
        noise = rng.standard_normal((my, mx)) + 1j * rng.standard_normal((my, mx))
        # Re(FFT(sqrt(lambda / M) * eps)) has covariance exactly ``cov``.
        field = np.real(np.fft.fft2(np.sqrt(lam / (mx * my)) * noise))
        self.values = field[: self.ny, : self.nx]

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        return bilinear(self.values, self.res, pts)


def bilinear(grid: np.ndarray, res: float, pts: np.ndarray, origin=(0.0, 0.0)) -> np.ndarray:
    """Bilinear lookup on ``grid[iy, ix]`` (last axes) at metric points (N, 2).

    ``grid`` may carry leading channel axes: shape (..., ny, nx) -> (N, ...).
    """
    pts = np.atleast_2d(pts)
    ny, nx = grid.shape[-2:]
    fx = np.clip((pts[:, 0] - origin[0]) / res, 0, nx - 1.000001)
    fy = np.clip((pts[:, 1] - origin[1]) / res, 0, ny - 1.000001)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - x0, fy - y0
    g = np.moveaxis(grid, (-2, -1), (0, 1))  # (ny, nx, ...)
    shape = (-1,) + (1,) * (g.ndim - 2)
    tx, ty = tx.reshape(shape), ty.reshape(shape)
    return (
        g[y0, x0] * (1 - tx) * (1 - ty)
        + g[y0, x0 + 1] * tx * (1 - ty)
        + g[y0 + 1, x0] * (1 - tx) * ty
        + g[y0 + 1, x0 + 1] * tx * ty
    )


class RadioSimulator:
    """Ground-truth RSSI generator for a floor plan (the "real world")."""

    def __init__(self, floorplan: FloorPlan, config: RadioConfig | None = None):
        self.fp = floorplan
        self.cfg = config or RadioConfig()
        # One independent field per AP, seeded from the AP id (crc32 is stable
        # across processes, unlike hash()) so that every node that builds the
        # simulator sees the same world, and dropping APs leaves the others
        # unchanged.
        self.shadowing = {}
        for ap in floorplan.access_points:
            ap_rng = np.random.default_rng([self.cfg.seed, zlib.crc32(ap.id.encode())])
            self.shadowing[ap.id] = ShadowingField(
                floorplan.width,
                floorplan.height,
                self.cfg.shadowing_sigma_db,
                self.cfg.shadowing_corr_m,
                self.cfg.field_resolution_m,
                ap_rng,
            )

    @property
    def ap_ids(self) -> list[str]:
        return [ap.id for ap in self.fp.access_points]

    def path_loss(self, pts: np.ndarray) -> np.ndarray:
        """Deterministic large-scale loss (dB), shape (N, A): distance + walls."""
        pts = np.atleast_2d(pts)
        out = np.empty((len(pts), len(self.fp.access_points)))
        for j, ap in enumerate(self.fp.access_points):
            d = np.maximum(np.linalg.norm(pts - ap.pos, axis=1), 0.1)
            pl = self.cfg.pl_d0 + 10 * self.cfg.path_loss_exponent * np.log10(d / self.cfg.d0)
            out[:, j] = pl + self.fp.wall_loss_between(pts, ap.pos)
        return out

    def mean_rssi(self, pts: np.ndarray, include_shadowing: bool = True) -> np.ndarray:
        """Noise-free RSSI (dBm), shape (N, A)."""
        pts = np.atleast_2d(pts)
        tx = np.array([ap.tx_power for ap in self.fp.access_points])
        rssi = tx[None, :] - self.path_loss(pts)
        if include_shadowing:
            for j, ap in enumerate(self.fp.access_points):
                rssi[:, j] += self.shadowing[ap.id](pts)
        return rssi

    def breakdown(self, pt) -> list[dict]:
        """Per-AP terms of the propagation model at one point (for display).

        mean_rssi = tx_power - path_loss - wall_loss + shadowing
        """
        pt = np.asarray(pt, dtype=float)[None, :2]
        hits = segments_intersect(np.repeat(pt, len(self.fp.access_points), 0), self.fp.ap_positions, self.fp.walls)
        losses = self.fp.wall_losses
        rows = []
        for j, ap in enumerate(self.fp.access_points):
            d = max(float(np.linalg.norm(pt[0] - ap.pos)), 0.1)
            pl = self.cfg.pl_d0 + 10 * self.cfg.path_loss_exponent * np.log10(d / self.cfg.d0)
            crossed = [self.fp.wall_materials[k] for k in np.flatnonzero(hits[j])]
            wall = float(losses[hits[j]].sum())
            shadow = float(self.shadowing[ap.id](pt)[0])
            rows.append(
                {
                    "id": ap.id,
                    "distance_m": d,
                    "tx_power_dbm": ap.tx_power,
                    "path_loss_db": float(pl),
                    "walls": crossed,
                    "wall_loss_db": wall,
                    "shadowing_db": shadow,
                    "mean_rssi_dbm": float(ap.tx_power - pl - wall + shadow),
                }
            )
        return rows

    def measure(
        self,
        pts: np.ndarray,
        rng: np.random.Generator,
        n_samples: int = 1,
        extra_loss_db: np.ndarray | None = None,
        extra_sigma_db: np.ndarray | None = None,
    ) -> np.ndarray:
        """Noisy scans. Returns (N, A) for n_samples == 1 else (N, n_samples, A).

        APs below the receiver sensitivity are reported as NaN (not heard).
        ``extra_loss_db`` (N, A) is subtracted from the mean (bodies in the way,
        ``wiloc.movers``); ``extra_sigma_db`` (N,) adds temporal noise in
        quadrature (noise zones).  Without them the draw is unchanged.
        """
        mu = self.mean_rssi(pts)
        if extra_loss_db is not None:
            mu = mu - np.asarray(extra_loss_db, dtype=float).reshape(mu.shape)
        sigma = self.cfg.temporal_sigma_db
        if extra_sigma_db is not None:
            sigma = np.hypot(sigma, np.asarray(extra_sigma_db, dtype=float).reshape(-1, 1, 1))
        z = mu[:, None, :] + sigma * rng.standard_normal(
            (mu.shape[0], n_samples, mu.shape[1])
        )
        if self.cfg.quantize:
            z = np.round(z)
        z[z < self.cfg.sensitivity_dbm] = np.nan
        return z[:, 0, :] if n_samples == 1 else z
