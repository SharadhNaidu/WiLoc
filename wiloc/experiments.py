"""Reproducible experiments and figures.

Run all of them with ``python -m wiloc all --out results`` (see ``__main__``).
Every experiment is seeded; figures are written as PNG and the numbers as
``summary.json`` and ``summary.md``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .environment import FloorPlan  # noqa: E402
from .estimators import KNN, Probabilistic, default_estimators  # noqa: E402
from .fingerprint import DenseRadioMap, RadioMap  # noqa: E402
from .metrics import ecdf, errors, format_table, summarize  # noqa: E402
from .radio import RadioConfig, RadioSimulator  # noqa: E402
from .simulation import World, drive_patrol, run_trackers  # noqa: E402


def draw_floorplan(ax, fp: FloorPlan, aps=True, labels=True):
    for (x1, y1, x2, y2), m in zip(fp.walls, fp.wall_materials):
        lw = {"concrete": 3.0, "brick": 2.2, "drywall": 1.4, "glass": 1.4}.get(m, 1.4)
        ls = "--" if m == "glass" else "-"
        ax.plot([x1, x2], [y1, y2], color="0.15", lw=lw, ls=ls, solid_capstyle="butt")
    if aps:
        p = fp.ap_positions
        ax.scatter(p[:, 0], p[:, 1], marker="^", s=90, c="#d62728", edgecolor="k", zorder=5, label="Access point")
        if labels:
            for ap in fp.access_points:
                ax.annotate(ap.id, (ap.x, ap.y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, fp.width + 0.3)
    ax.set_ylim(-0.3, fp.height + 0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _test_scans(world: World, n: int, seed: int):
    rng = np.random.default_rng(seed)
    pts = world.floorplan.sample_free(n, rng)
    return pts, world.radio.measure(pts, rng)


# ------------------------------------------------------------------ figures
def exp_radio_maps(world: World, out: Path, ap_index: int = 1) -> dict:
    """Ground-truth field vs. path-loss-only model vs. GP map for one AP."""
    fp, j = world.floorplan, ap_index
    res = world.dense_map.res
    ny, nx = world.dense_map.mean.shape[1:]
    gx, gy = np.meshgrid(np.arange(nx) * res, np.arange(ny) * res)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    truth = world.radio.mean_rssi(grid)[:, j].reshape(ny, nx)
    no_shadow = world.radio.mean_rssi(grid, include_shadowing=False)[:, j].reshape(ny, nx)
    gp = world.dense_map.mean[j]
    sd = world.dense_map.std[j]
    free = fp.is_free(grid, 0.05).reshape(ny, nx)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    panels = [
        (truth, "Ground truth: path loss + walls + shadowing", "viridis"),
        (no_shadow, "Deterministic part only (no shadowing)", "viridis"),
        (gp, "GP radio map from site survey", "viridis"),
        (sd, "GP predictive std [dB]", "magma"),
    ]
    vmin, vmax = np.percentile(truth[free], [1, 99])
    for ax, (img, title, cmap) in zip(axes.ravel(), panels):
        kw = dict(vmin=vmin, vmax=vmax) if cmap == "viridis" else {}
        im = ax.imshow(np.where(free, img, np.nan), origin="lower", extent=(0, fp.width, 0, fp.height), cmap=cmap, **kw)
        draw_floorplan(ax, fp)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8, label="dB" if cmap == "magma" else "RSSI [dBm]")
    rp = world.radio_map.positions
    axes[1, 0].scatter(rp[:, 0], rp[:, 1], s=4, c="w", alpha=0.7, label="Survey RPs")
    fig.suptitle(f"Radio map for {world.radio_map.ap_ids[j]}")
    fig.savefig(out / "radio_maps.png", dpi=130)
    plt.close(fig)
    m = free
    return {
        "gp_map_rmse_db": float(np.sqrt(np.mean((gp - truth)[m] ** 2))),
        "pathloss_model_rmse_db": float(np.sqrt(np.mean((no_shadow - truth)[m] ** 2))),
    }


def exp_static_benchmark(world: World, out: Path, n_test: int = 1000, seed: int = 11) -> dict:
    pts, z = _test_scans(world, n_test, seed)
    results, errs = {}, {}
    for est in default_estimators():
        est.fit(world.radio_map)
        e = errors(est.predict(z), pts)
        errs[est.name], results[est.name] = e, summarize(e)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name, e in errs.items():
        x, y = ecdf(e)
        ax.plot(x, y, lw=2, label=f"{name} (median {np.median(e):.2f} m)")
    ax.set_xlabel("Localisation error [m]")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 6)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Single-scan estimators, random test points")
    fig.savefig(out / "static_cdf.png", dpi=130)
    plt.close(fig)
    return results


def exp_sweeps(world: World, out: Path, n_test: int = 500, seed: int = 21) -> dict:
    fp, cfg = world.floorplan, world.radio.cfg
    sweeps = {}

    # (a) number of access points
    ids = [ap.id for ap in fp.access_points]
    rows = []
    for k in range(3, len(ids) + 1):
        sub = fp.subset_aps(ids[:k])
        w = World.build(sub, cfg, seed=1, map_method="model")
        pts, z = _test_scans(w, n_test, seed)
        row = {"n_aps": k}
        for est in (KNN(4, weighted=True), Probabilistic()):
            row[est.name] = float(np.median(errors(est.fit(w.radio_map).predict(z), pts)))
        rows.append(row)
    sweeps["n_aps"] = rows

    # (b) survey grid spacing (survey cost vs accuracy)
    rows = []
    for s in (0.5, 1.0, 1.5, 2.0, 3.0):
        rmap = RadioMap.survey(world.radio, fp.grid(s), 20, np.random.default_rng(2))
        pts, z = _test_scans(world, n_test, seed)
        row = {"spacing": s, "n_rps": len(rmap.positions)}
        for est in (KNN(4, weighted=True), Probabilistic()):
            row[est.name] = float(np.median(errors(est.fit(rmap).predict(z), pts)))
        rows.append(row)
    sweeps["spacing"] = rows

    # (c) k in (W)KNN
    rows = []
    pts, z = _test_scans(world, n_test, seed)
    for k in (1, 2, 3, 4, 5, 6, 8, 10, 15):
        rows.append(
            {
                "k": k,
                "KNN": float(np.median(errors(KNN(k).fit(world.radio_map).predict(z), pts))),
                "WKNN": float(np.median(errors(KNN(k, True).fit(world.radio_map).predict(z), pts))),
            }
        )
    sweeps["k"] = rows

    # (d) temporal noise level
    rows = []
    for s in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
        sim = RadioSimulator(fp, replace(cfg, temporal_sigma_db=s))
        rmap = RadioMap.survey(sim, fp.grid(1.0), 20, np.random.default_rng(3))
        rng = np.random.default_rng(seed)
        pts = fp.sample_free(n_test, rng)
        z = sim.measure(pts, rng)
        rows.append(
            {
                "sigma_t": s,
                "WKNN (k=4)": float(np.median(errors(KNN(4, True).fit(rmap).predict(z), pts))),
                "Probabilistic": float(np.median(errors(Probabilistic().fit(rmap).predict(z), pts))),
            }
        )
    sweeps["temporal_noise"] = rows

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
    specs = [
        ("n_aps", "n_aps", "Number of APs"),
        ("spacing", "spacing", "Survey grid spacing [m]"),
        ("k", "k", "k (neighbours)"),
        ("temporal_noise", "sigma_t", "Temporal noise sigma [dB]"),
    ]
    for ax, (key, xk, xl) in zip(axes, specs):
        rows = sweeps[key]
        for col in [c for c in rows[0] if c not in (xk, "n_rps")]:
            ax.plot([r[xk] for r in rows], [r[col] for r in rows], "o-", label=col)
        ax.set_xlabel(xl)
        ax.set_ylabel("Median error [m]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out / "sweeps.png", dpi=130)
    plt.close(fig)
    return sweeps


def exp_tracking(world: World, out: Path, seed: int = 5, laps: int = 2, n_particles: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    log = drive_patrol(world, rng, laps=laps)
    res = run_trackers(world, log, np.random.default_rng(seed + 1), n_particles=n_particles)
    truth = log.truth[log.scan_idx]
    t = log.t[log.scan_idx]
    tracks = {k: v for k, v in res.items() if not k.startswith("_")}
    stats = {k: summarize(errors(v, truth)) for k, v in tracks.items()}

    colors = {"Odometry only": "#7f7f7f", "EKF (odom + WKNN)": "#2ca02c", "Particle filter": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(12, 7.5), constrained_layout=True)
    draw_floorplan(ax, world.floorplan)
    ax.plot(log.truth[:, 0], log.truth[:, 1], "k-", lw=2.5, label="Ground truth")
    for name, v in tracks.items():
        if "per scan" in name:
            ax.scatter(v[:, 0], v[:, 1], s=5, c="#ff7f0e", alpha=0.35, label=name)
        else:
            ax.plot(v[:, 0], v[:, 1], lw=1.4, color=colors.get(name), label=name)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9)
    ax.set_title(f"Patrol, {laps} laps, {log.t[-1]:.0f} s, {len(log.scans)} Wi-Fi scans")
    fig.savefig(out / "tracking_trajectory.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for name, v in tracks.items():
        e = errors(v, truth)
        axes[0].plot(t, e, lw=1.0 if "per scan" not in name else 0.6, alpha=1 if "per scan" not in name else 0.5, label=name)
        x, y = ecdf(e)
        axes[1].plot(x, y, lw=2, label=f"{name} (RMSE {stats[name]['rmse']:.2f} m)")
    sd = np.sqrt(np.trace(res["_pf_cov"], axis1=1, axis2=2))
    axes[0].fill_between(t, 0, sd, color="#1f77b4", alpha=0.15, label="PF 1-sigma radius")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("error [m]")
    axes[0].set_ylim(0, 5)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("error [m]")
    axes[1].set_ylabel("CDF")
    axes[1].set_xlim(0, 4)
    axes[1].legend(fontsize=8, loc="lower right")
    for a in axes:
        a.grid(alpha=0.3)
    fig.savefig(out / "tracking_error.png", dpi=130)
    plt.close(fig)
    return stats


def exp_global_localization(world: World, out: Path, runs: int = 10, seed: int = 31) -> dict:
    """Particle filter started with no prior (uniform particles)."""
    curves, conv_times = [], []
    for r in range(runs):
        rng = np.random.default_rng(seed + r)
        log = drive_patrol(world, rng, laps=1)
        res = run_trackers(world, log, np.random.default_rng(seed + 100 + r), n_particles=3000, global_init=True)
        e = errors(res["Particle filter"], log.truth[log.scan_idx])
        e = e[:240]
        curves.append(e)
        # Settling time: from here on the error never exceeds 1.5 m again.
        above = np.flatnonzero(e > 1.5)
        settle = 0 if len(above) == 0 else above[-1] + 1
        conv_times.append(float(settle * 0.5) if settle < len(e) else float("nan"))
    curves = np.array([c[: min(map(len, curves))] for c in curves])
    t = np.arange(curves.shape[1]) * 0.5
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for c in curves:
        ax.plot(t, c, color="0.6", lw=0.8)
    ax.plot(t, np.median(curves, axis=0), color="#1f77b4", lw=2.5, label="median of runs")
    ax.set_xlabel("time since start [s]")
    ax.set_ylabel("PF error [m]")
    ax.set_title(f"Global localisation from a uniform prior ({runs} runs)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out / "global_localization.png", dpi=130)
    plt.close(fig)
    return {
        "median_settling_time_s": float(np.nanmedian(conv_times)),
        "settling_threshold_m": 1.5,
        "converged_runs": int(np.sum(np.isfinite(conv_times))),
        "runs": runs,
        "final_median_error_m": float(np.median(curves[:, -20:])),
    }


def exp_survey_scaling(world: World, out: Path) -> dict:
    """GP map vs path-loss-only map as the particle filter's sensor model."""
    out_rows = {}
    rng = np.random.default_rng(41)
    log = drive_patrol(world, rng, laps=1)
    truth = log.truth[log.scan_idx]
    for method in ("model", "gp"):
        dense = DenseRadioMap.from_radio_map(world.radio_map, world.floorplan, method=method)
        w = World(world.floorplan, world.radio, world.radio_map, dense)
        res = run_trackers(w, log, np.random.default_rng(42))
        out_rows[{"model": "log-distance fit", "gp": "GP radio map"}[method]] = summarize(errors(res["Particle filter"], truth))
    return out_rows


def plot_eval_csv(csv_path: str | Path, fp: FloorPlan, out: str | Path) -> dict:
    """Figure + statistics for a Gazebo run logged by ``wiloc_ros``'s evaluator."""
    a = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = a["t"] - a["t"][0]
    truth = np.column_stack([a["true_x"], a["true_y"]])
    series = {
        "Odometry only": np.column_stack([a["odom_x"], a["odom_y"]]),
        "Wi-Fi fix (WKNN)": np.column_stack([a["fix_x"], a["fix_y"]]),
        "WiLoc filter": np.column_stack([a["est_x"], a["est_y"]]),
    }
    colors = {"Odometry only": "#7f7f7f", "Wi-Fi fix (WKNN)": "#ff7f0e", "WiLoc filter": "#1f77b4"}
    fig = plt.figure(figsize=(15, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.3, 1])
    ax = fig.add_subplot(gs[0])
    draw_floorplan(ax, fp)
    ax.plot(truth[:, 0], truth[:, 1], "k-", lw=2.5, label="Ground truth (Gazebo)")
    stats = {}
    ax_t, ax_c = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for name, xy in series.items():
        ok = np.all(np.isfinite(xy), axis=1)
        if not ok.any():
            continue
        e = np.linalg.norm(xy[ok] - truth[ok], axis=1)
        stats[name] = summarize(e)
        if name.startswith("Wi-Fi"):
            ax.scatter(xy[ok, 0], xy[ok, 1], s=4, alpha=0.3, c=colors[name], label=name)
        else:
            ax.plot(xy[ok, 0], xy[ok, 1], lw=1.4, c=colors[name], label=name)
        ax_t.plot(t[ok], e, lw=0.8 if name.startswith("Wi-Fi") else 1.5, alpha=0.5 if name.startswith("Wi-Fi") else 1, c=colors[name], label=name)
        x, y = ecdf(e)
        ax_c.plot(x, y, lw=2, c=colors[name], label=f"{name}\nRMSE {stats[name]['rmse']:.2f} m")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    ax.set_title("ROS 2 / Gazebo run")
    ax_t.set_xlabel("time [s]")
    ax_t.set_ylabel("error [m]")
    ax_t.set_ylim(0, max(4.0, min(10.0, ax_t.get_ylim()[1])))
    ax_c.set_xlabel("error [m]")
    ax_c.set_ylabel("CDF")
    ax_c.set_xlim(0, 4)
    ax_c.legend(fontsize=7, loc="lower right")
    for a_ in (ax_t, ax_c):
        a_.grid(alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return stats


def run_all(out: str | Path = "results", quick: bool = False) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    world = World.build(radio_config=RadioConfig(), seed=0)
    summary = {
        "radio_config": world.radio.cfg.__dict__,
        "n_reference_points": int(len(world.radio_map.positions)),
    }
    print("[1/6] radio maps")
    summary["radio_maps"] = exp_radio_maps(world, out)
    print("[2/6] static benchmark")
    summary["static"] = exp_static_benchmark(world, out, n_test=300 if quick else 1000)
    print("[3/6] parameter sweeps")
    summary["sweeps"] = exp_sweeps(world, out, n_test=200 if quick else 500)
    print("[4/6] tracking")
    summary["tracking"] = exp_tracking(world, out, laps=1 if quick else 2)
    print("[5/6] sensor model ablation")
    summary["sensor_model"] = exp_survey_scaling(world, out)
    print("[6/6] global localisation")
    summary["global"] = exp_global_localization(world, out, runs=3 if quick else 10)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    md = [
        "# WiLoc results",
        "",
        f"Reference points: {summary['n_reference_points']}, radio map RMSE: "
        f"GP {summary['radio_maps']['gp_map_rmse_db']:.2f} dB vs path-loss model "
        f"{summary['radio_maps']['pathloss_model_rmse_db']:.2f} dB",
        "",
        "## Single-scan estimators",
        format_table(summary["static"]),
        "",
        "## Tracking (odometry + Wi-Fi)",
        format_table(summary["tracking"]),
        "",
        "## Particle-filter sensor model",
        format_table(summary["sensor_model"]),
        "",
        "## Global localisation",
        "```",
        json.dumps(summary["global"], indent=2),
        "```",
    ]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return summary
