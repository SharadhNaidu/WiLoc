"""Lightweight 2-D simulator (no ROS needed) for repeatable experiments.

``World`` bundles a floor plan with its radio "ground truth" and builds the
radio map from a simulated site survey.  ``drive_patrol`` drives a
differential-drive robot through the plan's waypoints, recording true poses,
encoder odometry and periodic Wi-Fi scans.  ``run_trackers`` replays that log
through every tracker so they all see identical data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .environment import FloorPlan
from .estimators import KNN, Estimator
from .filters import ParticleFilter, WifiEKF
from .fingerprint import DenseRadioMap, RadioMap
from .motion import DiffDrive, OdometryMotionModel, WaypointController, WheelOdometry, odom_delta
from .radio import RadioConfig, RadioSimulator


@dataclass
class World:
    floorplan: FloorPlan
    radio: RadioSimulator
    radio_map: RadioMap
    dense_map: DenseRadioMap

    @classmethod
    def build(
        cls,
        floorplan: FloorPlan | None = None,
        radio_config: RadioConfig | None = None,
        survey_spacing: float = 1.0,
        survey_samples: int = 20,
        map_method: str = "gp",
        seed: int = 0,
    ) -> "World":
        fp = floorplan or FloorPlan.builtin("office")
        radio = RadioSimulator(fp, radio_config)
        rng = np.random.default_rng(seed)
        rmap = RadioMap.survey(radio, fp.grid(survey_spacing), survey_samples, rng)
        dense = DenseRadioMap.from_radio_map(rmap, fp, method=map_method)
        return cls(fp, radio, rmap, dense)


@dataclass
class PatrolLog:
    t: np.ndarray
    truth: np.ndarray  # (T, 3)
    odom: np.ndarray  # (T, 3)
    scan_idx: np.ndarray  # indices into t where a scan was taken
    scans: np.ndarray  # (S, A)
    meta: dict = field(default_factory=dict)


def drive_patrol(
    world: World,
    rng: np.random.Generator,
    dt: float = 0.05,
    scan_period: float = 0.5,
    robot: DiffDrive | None = None,
    odometry: WheelOdometry | None = None,
    laps: int = 1,
    max_time: float = 2000.0,
) -> PatrolLog:
    robot = robot or DiffDrive()
    odo = odometry or WheelOdometry(robot)
    ctrl = WaypointController()
    fp = world.floorplan
    pose = np.array(fp.start_pose, dtype=float)
    odo.reset(pose)
    goals = list(fp.waypoints) * laps
    gi = 0
    every = max(1, int(round(scan_period / dt)))
    T, X, O, S_idx, S = [], [], [], [], []
    k, t = 0, 0.0
    while gi < len(goals) and t < max_time:
        v, w, reached = ctrl.command(pose, goals[gi])
        if reached:
            gi += 1
            continue
        wl, wr = robot.wheel_speeds(v, w)
        v_true, w_true = robot.body_twist(wl, wr)
        pose = DiffDrive.integrate(pose, v_true, w_true, dt)
        o = odo.step(wl, wr, dt, rng)
        t += dt
        T.append(t)
        X.append(pose.copy())
        O.append(o)
        if k % every == 0:
            S_idx.append(k)
            S.append(world.radio.measure(pose[None, :2], rng)[0])
        k += 1
    return PatrolLog(np.array(T), np.array(X), np.array(O), np.array(S_idx), np.array(S), {"dt": dt})


def run_trackers(
    world: World,
    log: PatrolLog,
    rng: np.random.Generator,
    n_particles: int = 1000,
    fix_estimator: Estimator | None = None,
    global_init: bool = False,
    motion_model: OdometryMotionModel | None = None,
) -> dict[str, np.ndarray]:
    """Return estimated (x, y) at every scan instant for each tracker."""
    mm = motion_model or OdometryMotionModel(alpha1=0.05, alpha2=0.02, alpha3=0.05, alpha4=0.01)
    fix = (fix_estimator or KNN(k=4, weighted=True)).fit(world.radio_map)
    pf = ParticleFilter(world.floorplan, world.dense_map, n_particles, mm, world.radio.cfg.sensitivity_dbm, rng=rng)
    ekf = WifiEKF(mm)
    start = np.array(world.floorplan.start_pose, dtype=float)
    pf.init_global() if global_init else pf.init_pose(start)
    ekf.init_pose(start, np.diag([25.0, 25.0, 3.0]) if global_init else None)

    out = {"Odometry only": [], f"{fix.name} (per scan)": [], "EKF (odom + WKNN)": [], "Particle filter": []}
    pf_cov = []
    prev = log.odom[0]
    scan_set = {int(i): s for i, s in zip(log.scan_idx, range(len(log.scan_idx)))}
    odo_offset = start - log.odom[0]  # odometry starts at the true start pose
    for k in range(len(log.t)):
        d = odom_delta(prev, log.odom[k])
        prev = log.odom[k]
        pf.predict(d)
        ekf.predict(d)
        if k in scan_set:
            z = log.scans[scan_set[k]]
            fx, fcov = fix.predict_with_cov(z[None])
            pf.update(z)
            ekf.update_position(fx[0], fcov[0])
            out["Odometry only"].append(log.odom[k, :2] + odo_offset[:2])
            out[f"{fix.name} (per scan)"].append(fx[0])
            out["EKF (odom + WKNN)"].append(ekf.estimate()[0][:2])
            est, cov = pf.estimate()
            out["Particle filter"].append(est[:2])
            pf_cov.append(cov)
    res = {k: np.array(v) for k, v in out.items()}
    res["_pf_cov"] = np.array(pf_cov)
    return res
