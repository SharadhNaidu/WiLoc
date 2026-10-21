import numpy as np
import pytest

from wiloc import (
    KNN,
    DiffDrive,
    FloorPlan,
    OdometryMotionModel,
    ParticleFilter,
    Probabilistic,
    RadioConfig,
    RadioMap,
    RadioSimulator,
    WifiEKF,
    World,
    drive_patrol,
    run_trackers,
)
from wiloc.fingerprint import GaussianProcess2D
from wiloc.geometry import segments_intersect, wrap_angle
from wiloc.metrics import errors
from wiloc.motion import odom_delta
from wiloc.radio import ShadowingField


@pytest.fixture(scope="module")
def fp():
    return FloorPlan.builtin("office")


@pytest.fixture(scope="module")
def world():
    return World.build(seed=0)


# --------------------------------------------------------------- geometry
def test_segment_intersection():
    walls = np.array([[0, 0, 0, 2.0]])
    assert segments_intersect([[-1, 1]], [[1, 1]], walls)[0, 0]
    assert not segments_intersect([[-1, 3]], [[1, 3]], walls)[0, 0]
    assert not segments_intersect([[1, 0]], [[1, 2]], walls)[0, 0]  # parallel


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)


def test_floorplan_loads(fp):
    assert len(fp.access_points) == 6
    assert fp.walls.shape[1] == 4
    # door gap at x=2.5 lets the robot pass between corridor and room A
    assert not fp.crosses_wall(np.array([[2.5, 6.0]]), np.array([[2.5, 9.0]]))[0]
    assert fp.crosses_wall(np.array([[1.0, 6.0]]), np.array([[1.0, 9.0]]))[0]


def test_waypoint_route_is_collision_free(fp):
    pts = np.vstack([fp.start_pose[:2], fp.waypoints])
    assert not fp.crosses_wall(pts[:-1], pts[1:]).any()


# ------------------------------------------------------------------ radio
def test_wall_adds_attenuation(fp):
    sim = RadioSimulator(fp, RadioConfig(shadowing_sigma_db=0.0))
    ap = fp.access_points[0]
    # same distance from AP1, one point through a brick wall
    same_room = np.array([[ap.x, ap.y - 2.0]])
    other_room = np.array([[ap.x + 3.0, ap.y]])  # beyond the x=5 wall
    d1 = np.linalg.norm(same_room - ap.pos)
    d2 = np.linalg.norm(other_room - ap.pos)
    pl = sim.path_loss(np.vstack([same_room, other_room]))[:, 0]
    free_space_diff = 10 * 2.0 * np.log10(d2 / d1)
    assert pl[1] - pl[0] == pytest.approx(free_space_diff + fp.materials["drywall"], abs=1e-6)


def test_shadowing_statistics():
    rng = np.random.default_rng(0)
    f = ShadowingField(60, 60, sigma=5.0, corr=3.0, resolution=0.5, rng=rng)
    assert 3.5 < f.values.std() < 6.5
    v = f.values
    # correlation at lag 3 m (6 cells) should be near exp(-1)
    c = np.corrcoef(v[:, :-6].ravel(), v[:, 6:].ravel())[0, 1]
    assert 0.2 < c < 0.55


def test_simulator_is_deterministic_across_instances(fp):
    a = RadioSimulator(fp).mean_rssi(np.array([[4.0, 4.0]]))
    b = RadioSimulator(fp).mean_rssi(np.array([[4.0, 4.0]]))
    np.testing.assert_allclose(a, b)


def test_sensitivity_gives_nan(fp):
    sim = RadioSimulator(fp, RadioConfig(sensitivity_dbm=-40.0))
    z = sim.measure(np.array([[19.5, 0.5]]), np.random.default_rng(0))
    assert np.isnan(z).any()


# ------------------------------------------------------------ fingerprint
def test_radio_map_roundtrip(tmp_path, world):
    p = tmp_path / "map.npz"
    world.radio_map.save(p)
    m = RadioMap.load(p)
    np.testing.assert_allclose(m.mean, world.radio_map.mean)
    assert m.ap_ids == world.radio_map.ap_ids


def test_gp_interpolates_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, (80, 2))
    f = lambda x: np.sin(x[:, 0] / 2) * 5 + x[:, 1]
    gp = GaussianProcess2D().fit(X, f(X) + 0.1 * rng.standard_normal(80))
    Xs = rng.uniform(1, 9, (50, 2))
    mu, var = gp.predict(Xs)
    assert np.sqrt(np.mean((mu - f(Xs)) ** 2)) < 0.5
    assert np.all(var >= 0)


def test_gp_map_beats_pathloss_only(world):
    from wiloc.fingerprint import DenseRadioMap

    model = DenseRadioMap.from_radio_map(world.radio_map, world.floorplan, method="model")
    pts = world.floorplan.sample_free(400, np.random.default_rng(3))
    truth = world.radio.mean_rssi(pts)
    e_gp = np.sqrt(np.mean((world.dense_map.lookup(pts)[0] - truth) ** 2))
    e_model = np.sqrt(np.mean((model.lookup(pts)[0] - truth) ** 2))
    assert e_gp < e_model


# ------------------------------------------------------------- estimators
def test_knn_exact_on_noise_free_fingerprints(world):
    rp = world.radio_map.positions[::7]
    z = world.radio_map.mean[::7]
    est = KNN(k=1).fit(world.radio_map).predict(z)
    np.testing.assert_allclose(est, rp)


def test_estimators_reach_metre_level(world):
    rng = np.random.default_rng(5)
    pts = world.floorplan.sample_free(200, rng)
    z = world.radio.measure(pts, rng)
    for est in (KNN(4, weighted=True), Probabilistic()):
        med = np.median(errors(est.fit(world.radio_map).predict(z), pts))
        assert med < 2.0, est.name


# ----------------------------------------------------------------- motion
def test_diff_drive_kinematics():
    r = DiffDrive()
    wl, wr = r.wheel_speeds(0.1, 0.5)  # within wheel-speed limits
    v, w = r.body_twist(wl, wr)
    assert v == pytest.approx(0.1) and w == pytest.approx(0.5)
    # full circle returns to start
    p = np.zeros(3)
    for _ in range(1000):
        p = DiffDrive.integrate(p, 0.2, 2 * np.pi / 10.0, 0.01)
    np.testing.assert_allclose(p, [0, 0, 0], atol=1e-6)


def test_odometry_motion_model_noise_free():
    mm = OdometryMotionModel(0, 0, 0, 0)
    a, b = np.array([1.0, 1.0, 0.3]), np.array([2.0, 1.5, 1.0])
    out = mm.sample(np.array([a]), odom_delta(a, b), np.random.default_rng(0))
    np.testing.assert_allclose(out[0], b, atol=1e-9)


# ---------------------------------------------------------------- filters
def test_filters_beat_raw_wifi_and_odometry(world):
    rng = np.random.default_rng(9)
    log = drive_patrol(world, rng)
    res = run_trackers(world, log, np.random.default_rng(10), n_particles=500)
    truth = log.truth[log.scan_idx]
    rmse = {k: np.sqrt(np.mean(errors(v, truth) ** 2)) for k, v in res.items() if not k.startswith("_")}
    wifi = [v for k, v in rmse.items() if "per scan" in k][0]
    assert rmse["Particle filter"] < wifi
    assert rmse["EKF (odom + WKNN)"] < wifi
    assert rmse["Particle filter"] < 1.0


def test_particle_filter_global_localization(world):
    rng = np.random.default_rng(3)
    log = drive_patrol(world, rng)
    res = run_trackers(world, log, np.random.default_rng(4), n_particles=2000, global_init=True)
    e = errors(res["Particle filter"], log.truth[log.scan_idx])
    assert np.median(e[40:]) < 1.0


def test_pf_recovers_when_all_particles_leave_map(world):
    pf = ParticleFilter(world.floorplan, world.dense_map, 200, rng=np.random.default_rng(0))
    pf.init_pose([19.9, 6.0, 0.0], std=(0.01, 0.01, 0.0))
    pf.predict((0.0, 1.0, 0.0))  # drive 1 m east, out of the building
    assert world.floorplan.is_free(pf.particles[:, :2], 0.0).all()


def test_ekf_gates_outliers():
    ekf = WifiEKF()
    ekf.init_pose([0, 0, 0])
    assert not ekf.update_position([50.0, 50.0], np.eye(2) * 0.5)
    assert ekf.update_position([0.2, -0.1], np.eye(2) * 0.5)


def test_odom_delta_handles_reversing():
    a = np.array([1.0, 1.0, 0.0])
    b = np.array([0.9, 1.0, 0.05])  # 10 cm backwards while turning slightly
    rot1, trans, rot2 = odom_delta(a, b)
    assert trans == pytest.approx(-0.1) and abs(rot1) < 1e-9 and rot2 == pytest.approx(0.05)
    out = OdometryMotionModel(0, 0, 0, 0).sample(np.array([a]), (rot1, trans, rot2), np.random.default_rng(0))
    np.testing.assert_allclose(out[0], b, atol=1e-9)


def test_estimators_report_no_information_for_empty_scan(world):
    z = np.full((1, world.radio_map.n_aps), np.nan)
    for est in (KNN(4, weighted=True), Probabilistic()):
        xy, cov = est.fit(world.radio_map).predict_with_cov(z)
        np.testing.assert_allclose(xy[0], world.radio_map.positions.mean(axis=0))
        assert np.sqrt(np.trace(cov[0])) > 4.0  # spans the building, not ~1 m


def test_breakdown_matches_mean_rssi(fp):
    sim = RadioSimulator(fp)
    pt = np.array([7.5, 2.5])
    rows = sim.breakdown(pt)
    np.testing.assert_allclose([r["mean_rssi_dbm"] for r in rows], sim.mean_rssi(pt[None])[0], atol=1e-9)
    assert any(r["walls"] for r in rows)  # some AP is behind a wall from room F


def test_integer_coordinates_in_yaml_are_floats():
    d = {"width": 10, "height": 8, "walls": [{"seg": [0, 0, 10, 0]}], "access_points": [{"id": "A", "x": 5, "y": 4}], "start_pose": [1, 1, 0]}
    fp = FloorPlan.from_dict(d)
    ap = fp.access_points[0]
    assert isinstance(ap.x, float) and isinstance(ap.tx_power, float)
    assert all(isinstance(v, float) for v in fp.start_pose)


def test_particle_filter_recovers_from_kidnapping(world):
    """Mid-run kidnapping: a confident cloud moved far away must re-localise
    through augmented-MCL injection, without false injections beforehand."""
    rng = np.random.default_rng(21)
    log = drive_patrol(world, rng)
    pf = ParticleFilter(world.floorplan, world.dense_map, 1500, rng=np.random.default_rng(21))
    pf.init_pose(world.floorplan.start_pose)
    prev, scan_k, err = log.odom[0], dict(zip(log.scan_idx.tolist(), range(len(log.scan_idx)))), []
    for k in range(len(log.t)):
        pf.predict(odom_delta(prev, log.odom[k]))
        prev = log.odom[k]
        if k in scan_k:
            if scan_k[k] == 200:
                assert pf.n_injected == 0  # no false alarms while tracking
                pf.particles[:, :2] = np.array([17.0, 9.5]) + 0.1 * rng.standard_normal((pf.n, 2))
            pf.update(log.scans[scan_k[k]])
            err.append(np.linalg.norm(pf.estimate()[0][:2] - log.truth[k, :2]))
    err = np.array(err)
    assert err[200] > 5.0 and pf.n_injected > 0
    assert (err[200:260] < 1.0).any()  # recovered within 30 s
    assert np.median(err[-200:]) < 1.0


def test_particle_filter_recovers_from_kidnapping_while_parked(world):
    """No motion after the kidnapping: recovery must come from the scans alone.

    (Spots where two distant places have near-identical fingerprints -- e.g.
    right at a wall corner -- can need motion to disambiguate; random spots
    recover from scans alone, 12/12 in a sweep, median 7 scans.)
    """
    log = drive_patrol(world, np.random.default_rng(31))
    for i, spot in enumerate(world.floorplan.sample_free(3, np.random.default_rng(5), clearance=0.4)):
        rng = np.random.default_rng(100 + i)
        pf = ParticleFilter(world.floorplan, world.dense_map, 1500, rng=np.random.default_rng(i))
        pf.init_pose(world.floorplan.start_pose)
        prev, scan_k = log.odom[0], dict(zip(log.scan_idx.tolist(), range(len(log.scan_idx))))
        for k in range(len(log.t)):  # normal tracking first, so w_slow knows "good"
            pf.predict(odom_delta(prev, log.odom[k]))
            prev = log.odom[k]
            if k in scan_k:
                pf.update(log.scans[scan_k[k]])
            if scan_k.get(k) == 150:
                break
        assert np.linalg.norm(pf.estimate()[0][:2] - spot) > 5.0  # kidnapped far away
        err = []
        for _ in range(60):  # 30 s of scans at 2 Hz, robot standing still
            pf.update(world.radio.measure(spot[None], rng)[0])
            err.append(np.linalg.norm(pf.estimate()[0][:2] - spot))
        assert min(err) < 1.5 and err[-1] < 2.0
