"""Noticing that a router has been switched off.

A router that is off is never heard.  To a fingerprinting method that looks
exactly like "you are far from this router", so an unplugged router drags
every estimate away from where it used to be.  The detector compares each scan
with what the radio map predicts at the current estimate: a router that the
map says should be heard loudly (mean - ``k_sigma`` * std above the receiver
sensitivity) but has been silent for ``n_scans`` scans in a row is masked
(left out of the measurement model) until it is heard again.

It only adds masks while the estimate is trustworthy (``spread`` below
``max_spread``): a lost robot's estimate cannot tell "router off" from "I am
somewhere else", and masking then would throw away the evidence it needs.
"""

from __future__ import annotations

import numpy as np


class OutageDetector:
    def __init__(self, n_aps: int, n_scans: int = 4, k_sigma: float = 3.0, max_spread: float = 1.5):
        self.n_scans, self.k_sigma, self.max_spread = n_scans, k_sigma, max_spread
        self.reset(n_aps)

    def reset(self, n_aps: int):
        self.silent = np.zeros(n_aps, dtype=int)  # consecutive scans unheard while expected
        self.masked = np.zeros(n_aps, dtype=bool)

    @property
    def usable(self) -> np.ndarray:
        """bool (A,): APs the filters may use."""
        return ~self.masked

    def update(self, z, mu, sd, sensitivity: float, spread: float) -> np.ndarray:
        """Feed one scan ``z`` (A,) and the map's ``mu``/``sd`` (A,) at the current estimate; returns ``usable``."""
        z, mu, sd = (np.asarray(v, dtype=float).reshape(-1) for v in (z, mu, sd))
        heard = ~np.isnan(z)
        expected = mu - self.k_sigma * sd > sensitivity
        trusted = spread <= self.max_spread
        self.silent = np.where(heard, 0, np.where(expected & trusted, self.silent + 1, self.silent))
        self.masked = (self.masked & ~heard) | (self.silent >= self.n_scans)
        return self.usable
