"""Pre-flight check for the stand (``./wiloc.sh check``): a headless run of the real demo, about 2 minutes.

Starts the simulation the way the Studio does, then checks, with numbers:
  * it comes up (robot pose and localiser status) within ``--startup`` s
  * the 3-D view's camera delivers frames, and the simulation runs near real time
  * the particle filter tracks the patrolling robot (median error)
  * after a kidnap the particle filter finds the robot again
Exit code 0 when every check passes.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from wiloc.environment import FloorPlan
from wiloc.story import RecoveryClock

from .ui.backend import Backend
from .ui.bridge import Bridge


def _pf_error(snap):
    if snap.truth is None or "pf" not in snap.poses:
        return None
    return math.hypot(snap.poses["pf"][0] - snap.truth[0], snap.poses["pf"][1] - snap.truth[1])


def main(argv=None):
    is_wsl = False
    try:
        with open("/proc/version", "r") as f:
            is_wsl = "microsoft" in f.read().lower()
    except OSError:
        pass

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--venue", default="office")
    p.add_argument("--layout", default="")
    p.add_argument("--startup", type=float, default=90.0, help="max seconds until the simulation is up")
    p.add_argument("--settle", type=float, default=25.0)
    p.add_argument("--track", type=float, default=40.0, help="seconds of tracking to measure")
    p.add_argument("--max-error", type=float, default=0.6, help="max PF median error (m)")
    p.add_argument("--max-recovery", type=float, default=45.0, help="max seconds to re-find after a kidnap")
    p.add_argument("--min-fps", type=float, default=4.0 if is_wsl else 10.0, help="min camera frames/s")
    p.add_argument("--min-rtf", type=float, default=0.5 if is_wsl else 0.8, help="min simulation speed (x real time)")
    p.add_argument("--attach", action="store_true", help="check an already running simulation")
    a, _ = p.parse_known_args(argv if argv is not None else sys.argv[1:])

    if "ROS_LOCALHOST_ONLY" not in os.environ:
        os.environ["ROS_LOCALHOST_ONLY"] = "0" if is_wsl else "1"

    rclpy.init(args=None)
    venue = FloorPlan.builtin(a.venue, a.layout or None)
    bridge = Bridge(venue)
    ex = SingleThreadedExecutor()
    ex.add_node(bridge)
    spin_thread = threading.Thread(target=ex.spin, daemon=True)
    spin_thread.start()
    backend = Backend(attach=a.attach, launch_args=[])
    results = []

    def report(name, ok, detail):
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}", flush=True)

    print(f"WiLoc check: {venue.title}, {venue.layout_label}", flush=True)
    try:
        t0 = time.monotonic()
        backend.start(venue.name, venue.layout)
        if not a.attach:
            bridge.set_layout(venue.layout)
        up = problem = None
        while time.monotonic() - t0 < a.startup:
            s = bridge.snapshot()
            if s.truth is not None and s.status and "pf" in s.poses:
                up = time.monotonic() - t0
                break
            problem = backend.problem()
            if problem is not None or not backend.running:
                break
            time.sleep(0.5)
        if up is not None:
            detail = f"{up:.0f} s"
        elif problem is not None:
            detail = problem.reason + (" Try: ./wiloc.sh check --software-gl" if problem.graphics and not backend.software_gl else "")
        elif not backend.running:
            detail = "the simulation stopped while starting"
        else:
            detail = f"nothing after {a.startup:.0f} s"
        report("simulation starts", up is not None, detail)
        if up is None:
            return 1
        time.sleep(a.settle)

        s0 = bridge.snapshot()
        w0, n0, sim0 = time.monotonic(), s0.image_count, s0.sim_t
        errs = []
        while time.monotonic() - w0 < a.track:
            e = _pf_error(bridge.snapshot())
            if e is not None:
                errs.append(e)
            time.sleep(0.2)
        s1 = bridge.snapshot()
        wall = time.monotonic() - w0
        fps = (s1.image_count - n0) / wall
        rtf = (s1.sim_t - sim0) / wall if s1.sim_t is not None and sim0 is not None else 0.0
        med = float(np.median(errs)) if errs else float("inf")
        report("3-D view camera", fps >= a.min_fps, f"{fps:.1f} frames/s")
        report("simulation speed", rtf >= a.min_rtf, f"{rtf:.2f} x real time")
        report("particle filter tracks the robot", med <= a.max_error,
               f"median error {med:.2f} m, 90 % {np.percentile(errs, 90):.2f} m" if errs else "no estimates")

        clock = RecoveryClock(threshold=1.0, hold=3.0)
        where = bridge.kidnap()
        clock.kidnap(time.monotonic())
        took = None
        while where and time.monotonic() - clock.t_kidnap < a.max_recovery + 5.0:
            took = clock.update(time.monotonic(), _pf_error(bridge.snapshot()))
            if took is not None:
                break
            time.sleep(0.2)
        report("re-found after a kidnap", took is not None and took <= a.max_recovery,
               f"{took:.1f} s" if took is not None else ("teleport failed" if not where else f"not within {a.max_recovery:.0f} s"))
    finally:
        backend.stop()
        try:
            ex.cancel()
        except Exception:
            pass
        try:
            ex.shutdown(timeout_sec=0.5)
        except Exception:
            pass
        try:
            spin_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            bridge.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()
    ok = all(results)
    print("WiLoc check:", "all good" if ok else "PROBLEMS (see above)", flush=True)
    return 0 if ok else 1


def run():
    sys.exit(main())
