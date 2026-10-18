"""Command line interface.

    python -m wiloc all     [--out results] [--quick]   run every experiment
    python -m wiloc survey  [--floorplan F] [--spacing S] [--samples N] --map out.npz
    python -m wiloc locate  --map out.npz --rssi "-45,-70,nan,..."
    python -m wiloc plot-eval ~/.ros/wiloc_eval.csv --out results/gazebo_run.png
"""

from __future__ import annotations

import argparse

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

    args = p.parse_args(argv)
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
