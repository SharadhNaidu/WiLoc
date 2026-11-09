import numpy as np
import pytest

from wiloc.race import RaceTracker

M = ("pf", "ekf", "wknn")


def drive(rt, t0, t1, truth_fn, est_fn, dt=0.1):
    t = t0
    while t < t1 - 1e-9:
        tr = truth_fn(t)
        rt.add(t, tr, {m: est_fn(m, t, tr) for m in M})
        t += dt
    return t


def test_rolling_median_and_leader():
    rt = RaceTracker(M)
    off = {"pf": 0.2, "ekf": 0.5, "wknn": 1.5}
    drive(rt, 0, 40, lambda t: np.array([t * 0.1, 1.0]), lambda m, t, tr: (tr + [off[m], 0], None))
    med = rt.rolling_median(40)
    assert med == pytest.approx(off, abs=1e-9)
    assert rt.leader(40) == "pf"


def test_kidnap_excluded_from_leaderboard_and_recovery_timed():
    rt = RaceTracker(M, settle=20.0, recover_m=1.0, recover_hold=3.0)
    home, away = np.array([2.0, 2.0]), np.array([15.0, 9.0])

    def est(m, t, tr):
        if t < 30:
            return tr + [0.3, 0], None
        # after the kidnap at t=30 the estimates stay at home until each method recovers
        found = {"pf": 42.0, "ekf": 55.0, "wknn": 1e9}[m]
        return (home if t < found else tr + [0.3, 0]), None

    drive(rt, 0, 30, lambda t: home, est)
    drive(rt, 30, 70, lambda t: away, est)
    assert rt.n_kidnaps == 1
    assert rt.recovery["pf"] == pytest.approx(12.0, abs=0.15)
    assert rt.recovery["ekf"] == pytest.approx(25.0, abs=0.15)
    assert rt.recovery["wknn"] is None  # still lost
    # at t=45 the leaderboard ignores 30-50 s: only pre-kidnap samples (0.3 m) remain
    rt2 = RaceTracker(M)
    drive(rt2, 0, 30, lambda t: home, est)
    drive(rt2, 30, 45, lambda t: away, est)
    assert rt2.settling(45)
    assert rt2.rolling_median(45)["pf"] == pytest.approx(0.3, abs=1e-6)


def test_noisy_memoryless_method_counts_as_recovered():
    # WKNN-like: unbiased but noisy errors, often above 1 m; median ~0.8 m
    rng = np.random.default_rng(1)
    rt = RaceTracker(("wknn",))
    rt.add(0.0, [2, 2], {"wknn": ([2.5, 2], None)})
    rt.add(0.1, [15, 9], {"wknn": ([2.5, 2], None)})  # kidnap
    t = 0.2
    while t < 4.0:
        rt.add(t, [15, 9], {"wknn": (np.array([15, 9]) + [rng.exponential(0.8) + 0.25, 0], None)})
        t += 0.1
    # recovered at once; the 3 s median window reads that as half its length
    assert rt.recovery["wknn"] == pytest.approx(rt.recover_hold / 2, abs=0.15)


def test_coverage_counts_truth_inside_95_ellipse():
    rng = np.random.default_rng(0)
    rt = RaceTracker(("good", "over"))
    cov = np.diag([0.25, 0.25])
    for i in range(3000):
        t = i * 0.1
        noise = rng.multivariate_normal([0, 0], cov)
        rt.add(t, [5, 5], {"good": (np.array([5, 5]) + noise, cov), "over": (np.array([5, 5]) + noise, cov / 9)})
    cov95 = rt.coverage(300)
    assert cov95["good"] == pytest.approx(0.95, abs=0.03)
    assert cov95["over"] < 0.5  # an overconfident filter is caught


def test_memory_is_bounded():
    rt = RaceTracker(M, maxlen=500)
    drive(rt, 0, 200, lambda t: np.array([1.0, 1.0]), lambda m, t, tr: (tr, None))
    assert len(rt.samples) == 500
