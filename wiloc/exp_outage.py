"""Offline experiment: what an unplugged router does to each method, with and without the outage detector.

Each run patrols the office for ``duration`` seconds; at ``t_off`` one router
is switched off (every router in turn, several seeds).  The particle filter
(as configured live), the Kalman filter fed by WKNN fixes, and WKNN itself run
twice on the same scans: plain, and with ``OutageDetector`` masking routers
that fall silent where the map says they should be loud.

    python -m wiloc outage [--runs 4] [--out results]
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .exp_filters import MM, VARIANTS, _world, drive
from .filters import ParticleFilter, WifiEKF
from .motion import odom_delta
from .outage import OutageDetector

METHODS = ("pf", "ekf", "wknn")


def run(args):
    seed, off, duration, t_off = args
    fp, radio, dense, fix = _world()
    d = drive(fp, np.random.default_rng(seed), duration, 0)
    srng = np.random.default_rng(seed + 2000)
    out = {}
    for variant in ("plain", "masked"):
        pf = ParticleFilter(fp, dense, 1500, MM, radio.cfg.sensitivity_dbm, rng=np.random.default_rng(seed + 1000),
                            **VARIANTS["all + sensor reset"])
        pf.init_pose(np.array(fp.start_pose, dtype=float))
        ekf = WifiEKF(MM)
        ekf.init_pose(np.array(fp.start_pose, dtype=float))
        det = OutageDetector(len(fp.access_points))
        fix.ap_mask = None
        srng = np.random.default_rng(seed + 2000)  # identical scans for both variants
        used = d.odom[0]
        errs = {m: [] for m in METHODS}
        masked_at = None
        for k in range(0, len(d.t), 2):
            delta = odom_delta(used, d.odom[k])
            used = d.odom[k]
            pf.predict(delta)
            ekf.predict(delta)
            if k % 10:
                continue
            z = radio.measure(d.truth[k][None, :2], srng)[0]
            if d.t[k] >= t_off:
                z[off] = np.nan
            if variant == "masked":
                est, cov = pf.estimate()
                mu, sd = pf.map.lookup(est[None, :2])
                usable = det.update(z, mu[0], sd[0], radio.cfg.sensitivity_dbm, float(np.sqrt(max(np.trace(cov[:2, :2]), 0))))
                pf.ap_mask = fix.ap_mask = usable if not usable.all() else None
                if masked_at is None and not usable[off]:
                    masked_at = float(d.t[k] - t_off)
            xy, cov = fix.predict_with_cov(z[None])
            pf.hint = (xy[0], cov[0])
            pf.update(z)
            if not np.isnan(z).all():
                ekf.update_position(xy[0], cov[0])
            if d.t[k] >= t_off + 10:
                truth = d.truth[k, :2]
                errs["pf"].append(np.hypot(*(pf.estimate()[0][:2] - truth)))
                errs["ekf"].append(np.hypot(*(ekf.estimate()[0][:2] - truth)))
                errs["wknn"].append(np.hypot(*(xy[0] - truth)))
        out[variant] = {m: float(np.median(v)) for m, v in errs.items()}
        out[variant]["p90"] = {m: float(np.percentile(v, 90)) for m, v in errs.items()}
        out[variant]["reseeds"] = pf.n_reseeds
        if variant == "masked":
            out[variant]["detected_after_s"] = masked_at
    return {"seed": seed, "off": fp.access_points[off].id, **out}


def exp_outage(out="results", runs=4, duration=180.0, t_off=40.0, workers=None) -> dict:
    from .environment import FloorPlan

    n_aps = len(FloorPlan.builtin("office").access_points)
    jobs = [(seed, off, duration, t_off) for seed in range(runs) for off in range(n_aps)]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(run, jobs))
    summary = {}
    for variant in ("plain", "masked"):
        summary[variant] = {m: float(np.median([r[variant][m] for r in results])) for m in METHODS}
        summary[variant]["p90"] = {m: float(np.median([r[variant]["p90"][m] for r in results])) for m in METHODS}
        summary[variant]["reseeds"] = int(sum(r[variant]["reseeds"] for r in results))
    det = [r["masked"]["detected_after_s"] for r in results]
    summary["detected"] = f"{sum(d is not None for d in det)}/{len(det)}"
    summary["detect_median_s"] = float(np.median([d for d in det if d is not None])) if any(d is not None for d in det) else None
    summary["runs"] = len(results)
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "outage.json", "w") as f:
        json.dump({"summary": summary, "runs": results}, f, indent=2)
    return summary
