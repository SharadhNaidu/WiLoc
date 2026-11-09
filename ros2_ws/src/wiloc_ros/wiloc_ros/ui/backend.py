"""The simulation behind the Studio window: started, stopped and restarted from it.

The Studio is the parent process.  ``Backend.start(venue, layout)`` runs
``ros2 launch wiloc_ros sim.launch.py`` (headless, no other windows) in its own
process group; switching venue stops that group and starts a new one, and a
backend that dies is restarted the same way.  ``attach=True`` (``--attach``)
starts nothing and uses whatever simulation is already running (development).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

LOG_DIR = os.path.expanduser("~/wiloc_runs")


class Backend:
    def __init__(self, attach: bool = False, launch_args: list[str] | None = None):
        self.attach = attach
        self.launch_args = list(launch_args or [])
        self.proc: subprocess.Popen | None = None
        self.venue = self.layout = None
        self.started_at = None
        self.restarts = 0

    @property
    def running(self) -> bool:
        return self.attach or (self.proc is not None and self.proc.poll() is None)

    def start(self, venue: str, layout: str = ""):
        self.venue, self.layout = venue, layout
        self.started_at = time.monotonic()
        if self.attach:
            return
        self.stop()
        os.makedirs(LOG_DIR, exist_ok=True)
        cmd = [
            "ros2", "launch", "wiloc_ros", "sim.launch.py",
            "ui:=none", "gui:=false", "rviz:=false", "dashboard:=false", "tile_windows:=false",
            f"floorplan:={venue}", f"ap_layout:={layout}",
            *self.launch_args,
        ]
        log = open(os.path.join(LOG_DIR, "studio_backend.log"), "a")
        log.write(f"\n===== {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n")
        log.flush()
        env = dict(os.environ, GAZEBO_MODEL_DATABASE_URI="")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    def stop(self, timeout: float = 10.0):
        if self.attach:
            return
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc = None
        # Gazebo sometimes outlives its launch file; a leftover server blocks the next start.
        for _ in range(20):
            if subprocess.run(["pgrep", "-x", "gzserver"], capture_output=True).returncode != 0:
                break
            subprocess.run(["pkill", "-9", "-x", "gzserver"], capture_output=True)
            time.sleep(0.5)

    def restart(self):
        self.restarts += 1
        self.start(self.venue, self.layout)
