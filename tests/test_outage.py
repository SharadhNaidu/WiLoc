"""Unplugged routers: masks in the filters and the outage detector."""

import numpy as np
import pytest

from wiloc.cache import cached_maps
from wiloc.environment import FloorPlan
from wiloc.estimators import KNN
from wiloc.filters import ParticleFilter
from wiloc.motion import OdometryMotionModel
from wiloc.outage import OutageDetector
from wiloc.radio import RadioConfig, RadioSimulator


def test_detector_masks_after_n_silent_scans_and_unmasks_when_heard():
    det = OutageDetector(3, n_scans=3)
    mu, sd = np.array([-50.0, -60.0, -94.0]), np.array([2.0, 2.0, 2.0])
    z_off = np.array([-52.0, np.nan, np.nan])  # AP2 off; AP3 is simply weak here
    for _ in range(2):
        assert det.update(z_off, mu, sd, -95.0, spread=0.5).all()
    usable = det.update(z_off, mu, sd, -95.0, spread=0.5)
    assert list(usable) == [True, False, True]  # the weak AP is never blamed
    assert list(det.update(np.array([-52.0, -61.0, np.nan]), mu, sd, -95.0, 0.5)) == [True, True, True]


def test_detector_does_not_mask_while_lost():
    det = OutageDetector(2, n_scans=2)
    for _ in range(10):
        usable = det.update(np.array([-50.0, np.nan]), np.array([-50.0, -40.0]), np.array([2.0, 2.0]), -95.0, spread=6.0)
    assert usable.all()


@pytest.fixture(scope="module")
def world():
    fp = FloorPlan.builtin("office")
    cfg = RadioConfig.from_dict(fp.radio)
    rmap, dense, _ = cached_maps(fp, cfg)
    return fp, RadioSimulator(fp, cfg), rmap, dense


def test_mask_removes_the_bias_of_an_unplugged_router(world):
    fp, sim, rmap, dense = world
    rng = np.random.default_rng(3)
    truth = np.array([[12.5, 9.0]])  # meeting room, right under AP2
    off = 1
    z = sim.measure(np.repeat(truth, 30, 0), rng)
    z[:, off] = np.nan
    knn = KNN(4, weighted=True).fit(rmap)
    err_plain = np.median(np.hypot(*(knn.predict(z) - truth).T))
    knn.ap_mask = np.ones(len(fp.access_points), bool)
    knn.ap_mask[off] = False
    err_masked = np.median(np.hypot(*(knn.predict(z) - truth).T))
    assert err_masked < err_plain

    pf = ParticleFilter(fp, dense, 800, OdometryMotionModel(), rng=np.random.default_rng(0))
    xy = np.array([[12.5, 9.0], [3.0, 3.0]])
    ll_plain = pf.measurement_loglik(z[0], xy)
    pf.ap_mask = knn.ap_mask
    ll_masked = pf.measurement_loglik(z[0], xy)
    # the PF's robust likelihood already floors a silent router's penalty; masking must never hurt
    assert ll_masked[0] - ll_masked[1] >= ll_plain[0] - ll_plain[1] - 1e-9
