"""Command line interface.

    python -m wiloc all     [--out results] [--quick]   run every experiment
    python -m wiloc survey  [--floorplan F] [--spacing S] [--samples N] --map out.npz
    python -m wiloc locate  --map out.npz --rssi "-45,-70,nan,..."
    python -m wiloc plot-eval ~/.ros/wiloc_eval.csv --out results/gazebo_run.png
    python -m wiloc warm-cache                          build the localiser's radio maps for every venue
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .environment import FloorPlan
from .estimators import KNN
from .fingerprint import RadioMap
from .radio import RadioConfig, RadioSimulator


def _floorplan(arg):
    return FloorPlan.load(arg) if arg else FloorPlan.builtin("office")


def main(argv=None):
    p = argparse.ArgumentParser(prog="wiloc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("all", help="run all experiments and write figures")
    a.add_argument("--out", default="results")
    a.add_argument("--quick", action="store_true")

    s = sub.add_parser("survey", help="simulate a site survey and save the radio map")
    s.add_argument("--floorplan")
    s.add_argument("--spacing", type=float, default=1.0)
    s.add_argument("--samples", type=int, default=20)
    s.add_argument("--seed", type=int, default=7, help="radio environment seed")
    s.add_argument("--map", required=True)

    loc = sub.add_parser("locate", help="WKNN position from one RSSI scan")
    loc.add_argument("--map", required=True)
    loc.add_argument("--rssi", required=True, help="comma separated dBm values, 'nan' if unheard")
    loc.add_argument("-k", type=int, default=4)

    pe = sub.add_parser("plot-eval", help="plot the CSV written by the ROS evaluator node")
    pe.add_argument("csv")
    pe.add_argument("--floorplan")
    pe.add_argument("--out", default="results/gazebo_run.png")

    sub.add_parser("warm-cache", help="pre-build the cached radio maps the ROS localiser uses")

    fl = sub.add_parser("filters", help="particle-filter options (KLD, cluster estimate, re-seed) under kidnappings")
    fl.add_argument("--out", default="results")
    fl.add_argument("--runs", type=int, default=8)
    fl.add_argument("--duration", type=float, default=480.0)

    ou = sub.add_parser("outage", help="an unplugged router: each method with and without the outage detector")
    ou.add_argument("--out", default="results")
    ou.add_argument("--runs", type=int, default=4)

    ve = sub.add_parser("venues", help="every venue x router layout x method: tracking error and kidnap recovery")
    ve.add_argument("--out", default="results")
    ve.add_argument("--runs", type=int, default=3)

    args = p.parse_args(argv)
    if args.cmd == "venues":
        from .exp_venues import exp_venues

        for r in exp_venues(args.out, runs=args.runs):
            print(f"{r['title']:14s} {r['layout']:14s} {r['n_aps']} APs  PF {r['pf_median']:.2f}/{r['pf_p90']:.2f}  "
                  f"EKF {r['ekf_median']:.2f}/{r['ekf_p90']:.2f}  WKNN {r['wknn_median']:.2f}/{r['wknn_p90']:.2f}  "
                  f"PF re-found {r['pf_recovered']} {r['pf_recovery_s']}")
        return
    if args.cmd == "outage":
        from .exp_outage import exp_outage

        print(json.dumps(exp_outage(args.out, runs=args.runs), indent=2))
        return
    if args.cmd == "filters":
        from .exp_filters import exp_filters

        print(json.dumps(exp_filters(args.out, runs=args.runs, duration=args.duration), indent=2))
        return
    if args.cmd == "warm-cache":
        import time

        from .cache import cache_dir, cached_maps

        for name in FloorPlan.builtin_names():
            venue = FloorPlan.builtin(name)
            for layout in venue.layouts:
                fp = venue.with_layout(layout)
                t0 = time.monotonic()
                _, _, hit = cached_maps(fp, RadioConfig.from_dict(fp.radio))
                print(f"{name:10s} {layout:14s} {'cached' if hit else 'built'} in {time.monotonic() - t0:.1f} s -> {cache_dir()}")
        return
    if args.cmd == "plot-eval":
        from .experiments import plot_eval_csv

        print(plot_eval_csv(args.csv, _floorplan(args.floorplan), args.out))
        return
    if args.cmd == "all":
        from .experiments import run_all

        run_all(args.out, quick=args.quick)
    elif args.cmd == "survey":
        fp = _floorplan(args.floorplan)
        sim = RadioSimulator(fp, RadioConfig(seed=args.seed))
        rmap = RadioMap.survey(sim, fp.grid(args.spacing), args.samples, np.random.default_rng(0))
        rmap.save(args.map)
        print(f"saved {len(rmap.positions)} fingerprints x {rmap.n_aps} APs -> {args.map}")
    elif args.cmd == "locate":
        rmap = RadioMap.load(args.map)
        z = np.array([float(v) for v in args.rssi.split(",")])
        xy, cov = KNN(args.k, weighted=True).fit(rmap).predict_with_cov(z[None])
        print(f"x={xy[0, 0]:.2f} y={xy[0, 1]:.2f}  sigma={np.sqrt(np.trace(cov[0])):.2f} m")


if __name__ == "__main__":
    main()
