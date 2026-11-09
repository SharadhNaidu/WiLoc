"""People and vehicles that move through a venue and attenuate Wi-Fi (wiloc.movers)."""

import numpy as np
import pytest

from wiloc.environment import FloorPlan
from wiloc.movers import Mover, Movers
from wiloc.radio import RadioSimulator


def walker(**kw):
    d = {"id": "w", "type": "person", "route": [[0, 0], [4, 0]], "speed": 1.0, "loop": False}
    d.update(kw)
    return Mover.from_dict(d)


def test_route_timing_and_period():
    m = walker()
    assert m.period == pytest.approx(8.0)  # there and back at 1 m/s
    np.testing.assert_allclose(m.position(1.0), [1, 0])
    np.testing.assert_allclose(m.position(6.0), [2, 0])
    np.testing.assert_allclose(m.position(9.0), m.position(1.0))  # periodic
    assert m.heading(1.0) == pytest.approx(0.0)
    assert abs(m.heading(5.0)) == pytest.approx(np.pi)


def test_pause_holds_position():
    m = walker(pause=2.0)
    assert m.period == pytest.approx(12.0)
    np.testing.assert_allclose(m.position(0.5), [0, 0])  # pausing at the start
    np.testing.assert_allclose(m.position(3.0), [1, 0])
    np.testing.assert_allclose(m.position(6.5), [4, 0])  # pausing at the far end
    assert not m.moving(0.5) and m.moving(3.0)
    assert m.heading(6.5) == pytest.approx(0.0)  # still facing the way it came


def test_loop_route_closes():
    m = Mover.from_dict({"type": "agv", "route": [[0, 0], [2, 0], [2, 2], [0, 2]], "speed": 1.0})
    assert m.period == pytest.approx(8.0)
    np.testing.assert_allclose(m.position(7.0), [0, 1])
    assert m.heading(7.0) == pytest.approx(-np.pi / 2)


def test_static_mover():
    m = Mover.from_dict({"type": "person", "route": [[3, 4]], "yaw": 1.0, "anim": "sitting"})
    assert m.static and not m.moving(5.0)
    np.testing.assert_allclose(m.position(123.0), [3, 4])
    assert m.heading(0.0) == 1.0


def test_keyframes_follow_route():
    m = walker(pause=1.0)
    kf = np.array(m.keyframes(step=0.2))
    assert kf[0, 0] == 0.0 and kf[-1, 0] == pytest.approx(m.period)
    assert np.all(np.diff(kf[:, 0]) > 0)
    for t, x, y, _ in kf[::7]:
        np.testing.assert_allclose([x, y], m.position(t), atol=1e-9)
    assert np.max(np.hypot(*np.diff(kf[:, 1:3], axis=0).T)) <= 0.2 + 1e-9


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        Mover.from_dict({"type": "dragon", "route": [[0, 0]]})


def test_blocking_loss_shape_and_peak():
    ms = Movers.from_dicts([{"type": "person", "route": [[5, 0]]}])
    robot = np.array([[0.0, 0.0], [0.0, 3.0]])
    aps = np.array([[10.0, 0.0], [0.0, 10.0]])
    L = ms.blocking_loss(robot, aps, 0.0)
    assert L.shape == (2, 2)
    assert L[0, 0] == pytest.approx(4.0)  # body right on the link
    assert L[1, 1] < 1e-6  # far from the other link


def test_blocking_loss_symmetric_and_continuous():
    ms = Movers.from_dicts([{"type": "forklift", "route": [[0, 0], [10, 0]], "speed": 1.0, "loop": False}])
    a, b = np.array([[2.0, -3.0]]), np.array([[3.0, 4.0]])
    for t in np.linspace(0, 20, 41):
        assert ms.blocking_loss(a, b, t)[0, 0] == pytest.approx(ms.blocking_loss(b, a, t)[0, 0])
    # sweeping the robot past the end of the link: no jumps
    xs = np.linspace(-3, 3, 601)
    robots = np.column_stack([xs, np.zeros_like(xs)])
    loss = Movers.from_dicts([{"type": "person", "route": [[0, 0]]}]).blocking_loss(robots, np.array([[10.0, 0.0]]), 0)[:, 0]
    assert np.max(np.abs(np.diff(loss))) < 0.1


def test_noise_zones():
    ms = Movers.from_dicts([], [{"at": [5, 5], "radius": 2, "sigma_db": 4}, {"at": [6, 5], "radius": 2, "sigma_db": 3}])
    s = ms.extra_sigma([[5, 5], [0, 0], [3.5, 5]])
    assert s[0] == pytest.approx(5.0) and s[1] == 0.0 and s[2] == pytest.approx(4.0)
    assert bool(ms) and len(ms) == 0


def test_measure_with_movers_and_noise():
    fp = FloorPlan.builtin("office")
    sim = RadioSimulator(fp)
    pts = np.array([[3.0, 6.0], [12.0, 6.0]])
    base = sim.measure(pts, np.random.default_rng(1))
    same = sim.measure(pts, np.random.default_rng(1), extra_loss_db=np.zeros((2, 6)), extra_sigma_db=np.zeros(2))
    np.testing.assert_array_equal(np.nan_to_num(base, nan=-999), np.nan_to_num(same, nan=-999))
    weaker = sim.measure(pts, np.random.default_rng(1), extra_loss_db=np.full((2, 6), 10.0))
    ok = np.isfinite(weaker)
    assert np.all(weaker[ok] <= base[ok] - 9)


def test_near():
    ms = Movers.from_dicts([{"type": "person", "route": [[1, 0]]}, {"type": "person", "route": [[0, 0], [4, 0]], "loop": False}])
    assert [m.id for m, _ in ms.near([0.5, 0], 0.0, 0.8)] == ["person"] and len(ms.near([0.5, 0], 0.0, 0.8)) == 1
    assert len(ms.near([1, 0], 0.0, 1.5, moving_only=False)) == 2
