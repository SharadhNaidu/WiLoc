"""Offline experiment: does each particle-filter option (KLD, cluster estimate, re-seed) help?

Each run patrols the office on planned paths for ``duration`` seconds and is
kidnapped (teleported >= 6 m, random heading) every ``kidnap_every``
seconds; the odometry never sees the teleport, exactly as in Gazebo.  The
filter runs at the ROS node's cadence (prediction at 10 Hz, a scan every
0.5 s).  Scoring uses ``RaceTracker``, the same definitions as the live
leaderboard:

  recovery     seconds until the 3 s median error is below 1 m (failed if not within the next kidnap)
  tracking     median error outside the 20 s after each kidnap
  coverage     share of samples with the truth inside the filter's 95 % ellipse
               (overall, and in the 20 s right after each recovery)
  false alarm  re-seeds on a kidnap-free patrol (must be 0)
  cost         mean particle count and milliseconds per predict + update

    python -m wiloc filters [--runs 8] [--out results]
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import cached_maps
from .environment import FloorPlan
from .estimators import KNN
from .filters import KLDConfig, ParticleFilter
from .motion import DiffDrive, OdometryMotionModel, WheelOdometry, odom_delta
from .planning import OccupancyGrid, PathFollower
from .race import RaceTracker
from .radio import RadioConfig, RadioSimulator

VARIANTS = {
    "baseline": {},
    "KLD": {"kld": KLDConfig()},
    "cluster estimate": {"estimator": "cluster"},
    "re-seed": {"reseed_fraction": 0.3},
    "all three": {"kld": KLDConfig(), "estimator": "cluster", "reseed_fraction": 0.3},
    "all + sensor reset": {"kld": KLDConfig(), "estimator": "cluster", "reseed_fraction": 0.3, "hint_fraction": 0.5},
}
MM = OdometryMotionModel(alpha1=0.05, alpha2=0.02, alpha3=0.05, alpha4=0.01)


@dataclass
class Drive:
    t: np.ndarray
    truth: np.ndarray
    odom: np.ndarray
    kidnaps: list


def _world(name="office"):
    fp = FloorPlan.builtin(name)
    cfg = RadioConfig.from_dict(fp.radio)
    rmap, dense, _ = cached_maps(fp, cfg)
    return fp, RadioSimulator(fp, cfg), dense, KNN(4, weighted=True).fit(rmap)


def drive(fp, rng, duration, kidnap_every, dt=0.05):
    """Patrol on planned paths, teleported every ``kidnap_every`` s (0: never)."""
    grid = OccupancyGrid.from_floorplan(fp)
    safe = grid.free & (grid.clearance >= 0.6)
    cells = np.column_stack(np.nonzero(safe))
    spots = np.column_stack([(cells[:, 1] + 0.5) * grid.res, (cells[:, 0] + 0.5) * grid.res])
    robot = DiffDrive()
    odo = WheelOdometry(robot)
    pose = np.array(fp.start_pose, dtype=float)
    odo.reset(pose)
    patrol = [np.asarray(p, dtype=float) for p in fp.waypoints]
    pi = 0
    follower = None
    T, X, O, kid = [], [], [], []
    t, next_k = 0.0, kidnap_every or np.inf
    while t < duration:
        if t >= next_k:
            far = spots[np.hypot(*(spots - pose[:2]).T) >= 6.0]
            xy = far[rng.integers(len(far))]
            pose = np.array([xy[0], xy[1], rng.uniform(-np.pi, np.pi)])
            kid.append(t)
            follower = None
            next_k += kidnap_every
        if follower is None or follower.done:
            if follower is not None:
                pi = (pi + 1) % len(patrol)
            path = grid.plan(pose[:2], patrol[pi])
            if path is None:
                pi = (pi + 1) % len(patrol)
                continue
            follower = PathFollower(path)
        v, w, finished = follower.command(pose)
        wl, wr = robot.wheel_speeds(v, w)
        vt, wt = robot.body_twist(wl, wr)
        pose = DiffDrive.integrate(pose, vt, wt, dt)
        o = odo.step(wl, wr, dt, rng)
        t += dt
        T.append(t)
        X.append(pose.copy())
        O.append(o)
    return Drive(np.array(T), np.array(X), np.array(O), kid)


def run(args):
    """One (variant, seed) run; returns its records."""
    variant, seed, duration, kidnap_every = args
    fp, radio, dense, fix = _world()
    rng = np.random.default_rng(seed)
    d = drive(fp, rng, duration, kidnap_every)
    f = ParticleFilter(fp, dense, 1500, MM, radio.cfg.sensitivity_dbm, rng=np.random.default_rng(seed + 1000), **VARIANTS[variant])
    f.init_pose(np.array(fp.start_pose, dtype=float))
    race = RaceTracker(("pf",))
    srng = np.random.default_rng(seed + 2000)
    used = d.odom[0]
    step_ms, ns = [], []
    for k in range(len(d.t)):
        if k % 2:  # predict at 10 Hz
            continue
        t0 = time.perf_counter()
        f.predict(odom_delta(used, d.odom[k]))
        used = d.odom[k]
        if k % 10 == 0:  # a scan every 0.5 s
            z = radio.measure(d.truth[k][None, :2], srng)[0]
            xy, cov = fix.predict_with_cov(z[None])
            f.hint = (xy[0], cov[0])  # the WKNN fix, as the ROS node provides it
            f.update(z)
            est, cov = f.estimate()
            step_ms.append((time.perf_counter() - t0) * 1e3)
            ns.append(f.n)
            race.add(d.t[k], d.truth[k, :2], {"pf": (est[:2], cov)})
    samples = list(race.samples)
    t = np.array([s[0] for s in samples])
    err = np.array([s[1]["pf"] for s in samples])
    inside = np.array([s[2].get("pf", False) for s in samples])
    kid = np.array(d.kidnaps)
    settle = np.zeros(len(t), dtype=bool)
    for tk in kid:
        settle |= (t >= tk) & (t < tk + 20.0)
    rec = []
    post_cov = []
    for i, tk in enumerate(kid):
        end = kid[i + 1] if i + 1 < len(kid) else t[-1]
        r = _recovery(t, err, tk, end)
        rec.append(r)
        if r is not None:
            w = (t >= tk + r) & (t < min(tk + r + 20.0, end))
            if w.any():
                post_cov.append(float(inside[w].mean()))
    return {
        "variant": variant,
        "seed": seed,
        "recovery": rec,
        "tracking_median": float(np.median(err[~settle])) if (~settle).any() else None,
        "coverage": float(inside[~settle].mean()) if (~settle).any() else None,
        "post_recovery_coverage": post_cov,
        "mean_n": float(np.mean(ns)),
        "ms_per_update": float(np.mean(step_ms)),
        "reseeds": f.n_reseeds,
        "kidnaps": len(kid),
        "kidnap_t": [float(x) for x in kid],
        "trace": {"t": t[::2].round(2).tolist(), "err": err[::2].round(3).tolist(), "n": ns[::2]},
    }


def _recovery(t, err, tk, end, hold=3.0, thresh=1.0):
    """Same rule as RaceTracker: 3 s median below 1 m; time taken at the window midpoint."""
    sel = (t >= tk) & (t < end)
    ts, es = t[sel], err[sel]
    for i in range(len(ts)):
        if ts[i] - tk < hold:
            continue
        w = (ts > ts[i] - hold) & (ts <= ts[i])
        if np.median(es[w]) < thresh:
            return float(0.5 * (ts[w][0] + ts[i]) - tk)
    return None


def summarise(results):
    out = {}
    for v in VARIANTS:
        rs = [r for r in results if r["variant"] == v]
        if not rs:
            continue
        rec = [x for r in rs if r["kidnaps"] for x in r["recovery"]]
        ok = np.array([x for x in rec if x is not None])
        clean = [r for r in rs if not r["kidnaps"]]
        out[v] = {
            "kidnaps": len(rec),
            "recovered": int(len(ok)),
            "recovery_median_s": float(np.median(ok)) if len(ok) else None,
            "recovery_p90_s": float(np.percentile(ok, 90)) if len(ok) else None,
            "tracking_median_m": float(np.median([r["tracking_median"] for r in rs if r["tracking_median"] is not None])),
            "coverage": float(np.mean([r["coverage"] for r in rs if r["coverage"] is not None])),
            "post_recovery_coverage": float(np.mean([c for r in rs for c in r["post_recovery_coverage"]] or [np.nan])),
            "mean_n": float(np.mean([r["mean_n"] for r in rs])),
            "ms_per_update": float(np.mean([r["ms_per_update"] for r in rs])),
            "false_alarm_reseeds": int(sum(r["reseeds"] for r in clean)),
            "clean_runs": len(clean),
        }
    return out


def exp_filters(out="results", runs=8, duration=480.0, kidnap_every=60.0, workers=None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _world()  # build the cached radio map once before the workers start
    jobs = [(v, s, duration, kidnap_every) for v in VARIANTS for s in range(runs)]
    jobs += [(v, 100 + s, duration, 0.0) for v in VARIANTS for s in range(max(2, runs // 2))]  # kidnap-free
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run, jobs))
    summary = summarise(results)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    for v in VARIANTS:
        rec = sorted(x if x is not None else np.inf for r in results if r["variant"] == v and r["kidnaps"] for x in r["recovery"])
        if rec:
            axes[0].step(rec, np.arange(1, len(rec) + 1) / len(rec), where="post", lw=2, label=v)
    axes[0].set_xlim(0, 40)
    axes[0].set_xlabel("time to re-localise after a kidnapping [s]")
    axes[0].set_ylabel("share of kidnappings")
    axes[0].legend(fontsize=8, loc="lower right")
    tr = next(r for r in results if r["variant"] == "KLD" and r["kidnaps"])["trace"]
    ax = axes[1]
    ax.plot(tr["t"], tr["n"], color="#4B4FD6", lw=1.6)
    ax.set_ylabel("particles (KLD)", color="#4B4FD6")
    ax.set_xlabel("time [s]")
    ax2 = ax.twinx()
    ax2.plot(tr["t"], tr["err"], color="#E8743B", lw=1.0)
    ax2.set_ylabel("error [m]", color="#E8743B")
    ax2.set_ylim(0, 8)
    ax.set_title("KLD: particles spent only while lost")
    names = list(summary)
    axes[2].axis("off")
    cols = ["recovered", "median s", "track m", "cover", "n", "ms"]
    cells = [[f"{s['recovered']}/{s['kidnaps']}", f"{s['recovery_median_s']:.1f}" if s["recovery_median_s"] else "-",
              f"{s['tracking_median_m']:.2f}", f"{s['coverage'] * 100:.0f}%", f"{s['mean_n']:.0f}", f"{s['ms_per_update']:.1f}"]
             for s in summary.values()]
    tbl = axes[2].table(cellText=cells, rowLabels=names, colLabels=cols, loc="center")
    tbl.scale(1, 1.6)
    for a in axes[:1]:
        a.grid(alpha=0.3)
    fig.savefig(out / "filters.png", dpi=130)
    plt.close(fig)
    with open(out / "filters.json", "w") as f:
        json.dump({"summary": summary, "runs": runs, "duration": duration, "kidnap_every": kidnap_every}, f, indent=2)
    return summary
