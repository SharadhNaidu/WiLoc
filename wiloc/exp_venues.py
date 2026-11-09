"""Offline experiment: every venue x router layout x method ("where Wi-Fi localisation works, and why").

Each run patrols a venue for ``duration`` s with its people / forklifts / AGVs
blocking Wi-Fi (as in the live demo) and is kidnapped once at ``t_kidnap``.
The particle filter (as configured live), the Kalman filter fed by WKNN fixes
and WKNN itself see the same scans.  Reported per venue and layout: median and
90th-percentile tracking error before the kidnap, and the time each method
needs to get back within 1 m after it.

    python -m wiloc venues [--runs 3] [--out results]
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .cache import cached_maps
from .environment import FloorPlan
from .estimators import KNN
from .exp_filters import MM, VARIANTS, drive
from .filters import ParticleFilter, WifiEKF
from .motion import odom_delta
from .radio import RadioConfig, RadioSimulator
from .story import RecoveryClock

METHODS = ("pf", "ekf", "wknn")
NAMES = {"pf": "Particle filter", "ekf": "Kalman filter", "wknn": "Wi-Fi only (WKNN)"}


def _world(venue: str, layout: str):
    fp = FloorPlan.builtin(venue, layout)
    cfg = RadioConfig.from_dict(fp.radio)
    rmap, dense, _ = cached_maps(fp, cfg)
    return fp, RadioSimulator(fp, cfg), dense, KNN(4, weighted=True).fit(rmap)


def run(args):
    venue, layout, seed, duration, t_kidnap = args
    fp, radio, dense, fix = _world(venue, layout)
    d = drive(fp, np.random.default_rng(seed), duration, t_kidnap)
    srng = np.random.default_rng(seed + 2000)
    pf = ParticleFilter(fp, dense, 1500, MM, radio.cfg.sensitivity_dbm, rng=np.random.default_rng(seed + 1000),
                        **VARIANTS["all + sensor reset"])
    pf.init_pose(np.array(fp.start_pose, dtype=float))
    ekf = WifiEKF(MM)
    ekf.init_pose(np.array(fp.start_pose, dtype=float))
    aps = np.array([[a.x, a.y] for a in fp.access_points])
    errs = {m: [] for m in METHODS}
    clocks = {m: RecoveryClock(threshold=1.0, hold=3.0, grace=0.0) for m in METHODS}
    recovered = {m: None for m in METHODS}
    kidnapped = False
    used = d.odom[0]
    for k in range(0, len(d.t), 2):  # predict at 10 Hz
        delta = odom_delta(used, d.odom[k])
        used = d.odom[k]
        pf.predict(delta)
        ekf.predict(delta)
        if k % 10:  # a scan every 0.5 s
            continue
        t, truth = float(d.t[k]), d.truth[k]
        if d.kidnaps and not kidnapped and t >= d.kidnaps[0]:
            kidnapped = True
            for c in clocks.values():
                c.kidnap(t)
        loss = sig = None
        if fp.movers:
            loss = fp.movers.blocking_loss(truth[None, :2], aps, t)
            sig = fp.movers.extra_sigma(truth[None, :2])
        z = radio.measure(truth[None, :2], srng, extra_loss_db=loss, extra_sigma_db=sig)[0]
        xy, cov = fix.predict_with_cov(z[None])
        pf.hint = (xy[0], cov[0])
        pf.update(z)
        if not np.isnan(z).all():
            ekf.update_position(xy[0], cov[0])
        e = {
            "pf": float(np.hypot(*(pf.estimate()[0][:2] - truth[:2]))),
            "ekf": float(np.hypot(*(ekf.estimate()[0][:2] - truth[:2]))),
            "wknn": float(np.hypot(*(xy[0] - truth[:2]))),
        }
        if not kidnapped and t >= 20.0:  # tracking, after the start-up transient
            for m in METHODS:
                errs[m].append(e[m])
        if kidnapped:
            for m in METHODS:
                took = clocks[m].update(t, e[m])
                if took is not None and recovered[m] is None:
                    recovered[m] = took
    return {
        "venue": venue,
        "layout": layout,
        "seed": seed,
        "n_aps": len(fp.access_points),
        "median": {m: float(np.median(v)) for m, v in errs.items()},
        "p90": {m: float(np.percentile(v, 90)) for m, v in errs.items()},
        "recovery_s": recovered,
    }


def summarise(results):
    rows = []
    for venue in FloorPlan.builtin_names():
        fp = FloorPlan.builtin(venue)
        for layout, (label, aps) in fp.layouts.items():
            rs = [r for r in results if r["venue"] == venue and r["layout"] == layout]
            if not rs:
                continue
            row = {"venue": venue, "title": fp.title, "layout": layout, "label": label, "n_aps": len(aps), "runs": len(rs)}
            for m in METHODS:
                row[f"{m}_median"] = float(np.median([r["median"][m] for r in rs]))
                row[f"{m}_p90"] = float(np.median([r["p90"][m] for r in rs]))
                rec = [r["recovery_s"][m] for r in rs]
                row[f"{m}_recovered"] = f"{sum(x is not None for x in rec)}/{len(rec)}"
                row[f"{m}_recovery_s"] = round(float(np.median([x for x in rec if x is not None])), 1) if any(x is not None for x in rec) else None
            rows.append(row)
    return rows


def figure(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['title']} · {r['n_aps']} router{'s' if r['n_aps'] != 1 else ''}\n{r['label']}" for r in rows]
    y = np.arange(len(rows))
    colors = {"pf": "#4B4FD6", "ekf": "#D9822B", "wknn": "#159A8C"}
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.6))
    h = 0.26
    for i, m in enumerate(("wknn", "ekf", "pf")):
        med = np.array([r[f"{m}_median"] for r in rows])
        p90 = np.array([r[f"{m}_p90"] for r in rows])
        yy = y + (i - 1) * h
        ax.barh(yy, med, height=h, color=colors[m], label=NAMES[m])
        ax.hlines(yy, med, p90, color=colors[m], alpha=0.45, lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.1, 25)
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("tracking error, median (bar) to 90th percentile (line) [m]")
    ax.axvline(1.0, color="#A39E94", lw=0.8, ls="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Where Wi-Fi localisation works: six venues, their router layouts, three methods", fontsize=11, loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def exp_venues(out="results", runs=3, duration=240.0, t_kidnap=150.0, workers=None) -> list[dict]:
    jobs = [
        (venue, layout, seed, duration, t_kidnap)
        for venue in FloorPlan.builtin_names()
        for layout in FloorPlan.builtin(venue).layouts
        for seed in range(runs)
    ]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(run, jobs))
    rows = summarise(results)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "venues.json", "w") as f:
        json.dump({"rows": rows, "runs": results}, f, indent=2)
    figure(rows, out / "venues.png")
    md = ["| Venue | Routers | PF median / p90 | EKF median / p90 | WKNN median / p90 | PF re-found after kidnap |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        rec = r["pf_recovery_s"]
        md.append(
            f"| {r['title']} | {r['label']} ({r['n_aps']}) | {r['pf_median']:.2f} / {r['pf_p90']:.2f} m | "
            f"{r['ekf_median']:.2f} / {r['ekf_p90']:.2f} m | {r['wknn_median']:.2f} / {r['wknn_p90']:.2f} m | "
            f"{r['pf_recovered']}{f', median {rec:.1f} s' if rec is not None else ''} |"
        )
    (out / "venues.md").write_text("\n".join(md) + "\n")
    return rows
