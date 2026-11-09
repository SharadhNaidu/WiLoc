"""KLD sampling, mode clustering and re-seeding in the particle filter."""

import numpy as np
import pytest

from wiloc.filters import KLDConfig, ParticleFilter
from wiloc.motion import OdometryMotionModel
from wiloc.simulation import World

MM = OdometryMotionModel(0.05, 0.02, 0.05, 0.01)


@pytest.fixture(scope="module")
def world():
    return World.build(map_method="model", survey_samples=10)


def pf(world, seed=0, **kw):
    return ParticleFilter(world.floorplan, world.dense_map, 1500, MM, rng=np.random.default_rng(seed), **kw)


def walk(f, world, start, steps, rng, trace=None):
    """Drive 0.1 m per step (turning on the spot at walls), one scan per step at the true pose.

    Odometry increments are (rot1, trans, rot2).  ``trace`` collects the particle count.
    """
    pose = np.array(start, dtype=float)
    for _ in range(steps):
        nxt = pose + [0.1 * np.cos(pose[2]), 0.1 * np.sin(pose[2]), 0.0]
        if world.floorplan.is_free(nxt[None, :2], 0.4)[0]:
            f.predict(np.array([0.0, 0.1, 0.0]))
            pose = nxt
        else:
            f.predict(np.array([0.0, 0.0, 0.4]))
            pose[2] += 0.4
        f.update(world.radio.measure(pose[None, :2], rng)[0])
        if trace is not None:
            trace.append(f.n)
    return pose


def test_walk_tracks(world):
    # guards the helper itself: a filter fed this walk must track it
    f = pf(world)
    f.init_pose([2.5, 6.0, 0.0])
    pose = walk(f, world, [2.5, 6.0, 0.0], 60, np.random.default_rng(3))
    assert np.hypot(*(f.estimate()[0][:2] - pose[:2])) < 1.0


def test_n_required_bound():
    cfg = KLDConfig()
    k = np.array([0, 1, 2, 10, 100, 1000, 10**6])
    n = cfg.n_required(k)
    assert n[0] == n[1] == cfg.n_min
    assert np.all(np.diff(n) >= 0) and n[-1] == cfg.n_max
    # textbook value: k=100 bins, eps 0.05, delta 0.01 -> about 1,350 (unclipped)
    assert KLDConfig(n_min=1, n_max=10**6).n_required(np.array([100]))[0] == pytest.approx(1350, rel=0.05)


def test_kld_shrinks_when_converged_and_grows_after_kidnap(world):
    rng = np.random.default_rng(3)
    f = pf(world, kld=KLDConfig())
    f.init_pose([2.5, 6.0, 0.0])
    assert f.n == 5000
    pose = walk(f, world, [2.5, 6.0, 0.0], 60, rng)
    converged = f.n
    assert converged <= 1200, converged
    # kidnap: scans suddenly come from Room D; injected particles open new bins
    trace = []
    walk(f, world, [17.5, 9.5, np.pi / 2], 30, rng, trace)
    assert max(trace) > 2 * converged, (converged, trace)
    assert f.n_injected > 0
    # and it has every particle array consistent with the count
    assert f.particles.shape == (f.n, 3) and f.logw.shape == (f.n,)
    assert np.isclose(np.exp(f.logw).sum(), 1.0)


def test_kld_truncation_is_unbiased(world):
    f = pf(world, kld=KLDConfig(n_min=500, n_max=5000))
    f.init_pose([2.5, 6.0, 0.0])
    # two equally weighted hypotheses, stored one after the other
    half = f.n // 2
    f.particles[:half, :2] = [2.5, 2.5]
    f.particles[half:, :2] = [17.5, 9.5]
    f.w_slow = f.w_fast = 1.0  # not lost: no injection
    f._resample_kld()
    share = np.mean(f.particles[:, 0] < 10)
    assert f.n < 5000 and share == pytest.approx(0.5, abs=0.06)


def test_init_restores_full_count(world):
    f = pf(world, kld=KLDConfig())
    f.init_pose([2.5, 6.0, 0.0])
    walk(f, world, [2.5, 6.0, 0.0], 40, np.random.default_rng(1))
    assert f.n < 5000
    f.init_global()
    assert f.n == 5000 and f.particles.shape == (5000, 3)


def test_modes_and_cluster_estimate_with_two_hypotheses(world):
    f = pf(world, estimator="cluster")
    rng = np.random.default_rng(0)
    a, b = np.array([2.5, 2.5]), np.array([12.5, 9.5])  # Room E and Room C
    n_a = int(0.55 * f.n)
    f.particles[:, :2] = np.vstack([a + 0.2 * rng.standard_normal((n_a, 2)), b + 0.2 * rng.standard_normal((f.n - n_a, 2))])
    f.particles[:, 2] = 0.3
    f.logw[:] = -np.log(f.n)
    modes = f.modes()
    assert len(modes) == 2
    assert modes[0]["weight"] == pytest.approx(0.55, abs=0.01) and modes[1]["weight"] == pytest.approx(0.45, abs=0.01)
    assert np.hypot(*(modes[0]["xy"] - a)) < 0.1 and np.hypot(*(modes[1]["xy"] - b)) < 0.1
    est, cov = f.estimate()
    assert np.hypot(*(est[:2] - a)) < 0.1 and est[2] == pytest.approx(0.3, abs=1e-6)
    # the spread still reports the ambiguity (global covariance, metres apart)
    assert np.sqrt(np.trace(cov)) > 3.0


def test_cluster_estimate_never_inside_a_wall(world):
    f = pf(world, estimator="cluster")
    rng = np.random.default_rng(0)
    # one blob straddling the wall at y = 5 (x = 4, between the doors): its mean lies in the wall
    f.particles[:, :2] = np.column_stack([4.0 + 0.3 * rng.standard_normal(f.n), np.where(rng.random(f.n) < 0.5, 4.7, 5.3)])
    f.logw[:] = -np.log(f.n)
    est, _ = f.estimate()
    assert world.floorplan.is_free(est[None, :2], 0.05)[0]
    assert abs(est[1] - 5.0) > 0.2


def test_sensor_reset_injects_around_the_hint(world):
    f = pf(world, hint_fraction=0.5, max_inject=1.0)
    f.init_pose([2.5, 6.0, 0.0])
    f.hint = (np.array([17.5, 9.5]), np.eye(2) * 0.04)
    f.w_slow, f.w_fast = 1.0, 0.0  # maximally lost: every particle is injected
    f.resample()
    near = np.hypot(*(f.particles[:, :2] - [17.5, 9.5]).T) < 2.0
    assert near.mean() == pytest.approx(0.5, abs=0.06)
    # every heading is represented around the hint (motion has to pick it)
    assert np.std(np.cos(f.particles[near, 2])) > 0.6
    assert world.floorplan.is_free(f.particles[:, :2], 0.05).all()
    # without a hint injection stays uniform
    g = pf(world, hint_fraction=0.5, max_inject=1.0)
    g.init_pose([2.5, 6.0, 0.0])
    g.w_slow, g.w_fast = 1.0, 0.0
    g.resample()
    assert np.mean(np.hypot(*(g.particles[:, :2] - [17.5, 9.5]).T) < 2.0) < 0.15


def test_reseed_only_when_lost_for_a_while(world):
    # injection off, so the filter cannot recover by itself and stays lost
    rng = np.random.default_rng(5)
    f = pf(world, reseed_fraction=0.3, reseed_after=6, max_inject=0.0)
    f.init_pose([2.5, 6.0, 0.0])
    walk(f, world, [2.5, 6.0, 0.0], 60, rng)
    assert f.n_reseeds == 0  # tracking normally: never fires
    pose = walk(f, world, [17.5, 9.5, np.pi / 2], 40, rng)
    assert f.n_reseeds >= 1
    assert np.hypot(*(f.estimate()[0][:2] - pose[:2])) < 1.5  # and the re-seed found it
