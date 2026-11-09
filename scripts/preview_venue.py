#!/usr/bin/env python3
"""Render still images of a venue's Gazebo world (headless), for reviewing how it looks.

    python3 scripts/preview_venue.py office --out ~/wiloc_runs/preview

Builds the venue's world, adds a few camera models (an overview and views
from inside the rooms), runs gzserver without a window, saves one PNG per
camera and stops.  Needs a display for the GL context (WSLg: DISPLAY=:0) and a
sourced ROS 2 + workspace.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np


def _camera(name, x, y, z, pitch, yaw, w=1280, h=720, fov=1.25):
    return f"""
    <model name="preview_{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 {pitch:.3f} {yaw:.3f}</pose>
      <link name="link">
        <sensor name="cam" type="camera">
          <update_rate>5</update_rate>
          <camera>
            <horizontal_fov>{fov}</horizontal_fov>
            <image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
            <clip><near>0.05</near><far>120</far></clip>
          </camera>
          <plugin name="cam_{name}" filename="libgazebo_ros_camera.so">
            <ros><namespace>/preview</namespace></ros>
            <camera_name>{name}</camera_name>
            <frame_name>preview_{name}</frame_name>
          </plugin>
        </sensor>
      </link>
    </model>"""


def default_views(fp):
    """An overview plus up to three views from inside rooms, looking across them."""
    cx, cy = fp.width / 2, fp.height / 2
    span = max(fp.width, fp.height * 1.78)
    views = {"overview": (cx, cy - 0.62 * span, 0.62 * span, 0.85, math.pi / 2)}
    rooms = [r for r in fp.rooms if "corridor" not in r[0].lower()][:3]
    for i, (name, (x0, y0, x1, y1)) in enumerate(rooms):
        # stand in a corner-ish spot and look at the room centre, slightly down
        px, py = x0 + 0.25 * (x1 - x0), y0 + 0.2 * (y1 - y0)
        yaw = math.atan2((y0 + y1) / 2 + 0.2 * (y1 - y0) - py, (x0 + x1) / 2 + 0.25 * (x1 - x0) - px)
        views[f"room{i}"] = (px, py, 1.7, 0.32, yaw)
    return views


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("venue", nargs="?", default="office")
    p.add_argument("--out", default=os.path.expanduser("~/wiloc_runs/preview"))
    p.add_argument("--timeout", type=float, default=60.0)
    a = p.parse_args(argv)

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    from wiloc_ros.common import load_floorplan
    from wiloc_ros.world_gen import world_sdf

    fp = load_floorplan(a.venue)
    views = default_views(fp)
    sdf = world_sdf(fp)
    cams = "".join(_camera(n, *v) for n, v in views.items())
    sdf = sdf.replace("\n  </world>", cams + "\n  </world>")
    world = os.path.join(tempfile.gettempdir(), f"wiloc_preview_{fp.name}.world")
    with open(world, "w") as f:
        f.write(sdf)
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, GAZEBO_MODEL_DATABASE_URI="")
    log = open(os.path.join(a.out, f"{fp.name}_gzserver.log"), "w")
    gz = subprocess.Popen(["gzserver", "-s", "libgazebo_ros_init.so", world], stdout=log, stderr=log, env=env,
                          start_new_session=True)
    saved = {}
    try:
        rclpy.init()
        node = rclpy.create_node("venue_preview")

        ready_at = {}

        def on_image(name, msg):
            # the first frames can show a half-loaded scene: keep one from 3 s after the first
            ready_at.setdefault(name, time.monotonic() + 3.0)
            if name in saved or time.monotonic() < ready_at[name]:
                return
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            path = os.path.join(a.out, f"{fp.name}_{name}.png")
            plt.imsave(path, img)
            saved[name] = path
            print(f"saved {path}", flush=True)

        subs = [node.create_subscription(Image, f"/preview/{n}/image_raw", lambda m, n=n: on_image(n, m), qos_profile_sensor_data)
                for n in views]
        t0 = time.monotonic()
        while len(saved) < len(views) and time.monotonic() - t0 < a.timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
        del subs
        rclpy.try_shutdown()
    finally:
        os.killpg(gz.pid, signal.SIGINT)
        try:
            gz.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(gz.pid, signal.SIGKILL)
    missing = set(views) - set(saved)
    if missing:
        print(f"no frames from {sorted(missing)} (see {log.name})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
