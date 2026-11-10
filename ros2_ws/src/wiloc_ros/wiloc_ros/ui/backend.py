"""The simulation behind the Studio window: started, stopped and restarted from it.

The Studio is the parent process.  ``Backend.start(venue, layout)`` runs
``ros2 launch wiloc_ros sim.launch.py`` (headless, no other windows) in its own
process group; switching venue stops that group and starts a new one, and a
backend that dies is restarted the same way.  ``attach=True`` (``--attach``)
starts nothing and uses whatever simulation is already running (development).

``problem()`` reads this run's part of the launch log and says, in plain
words, why the simulation did not come up (``diagnose``), so the window can
show a reason instead of waiting for ever.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

LOG_DIR = os.path.expanduser("~/wiloc_runs")
LOG_PATH = os.path.join(LOG_DIR, "studio_backend.log")


@dataclass
class Problem:
    reason: str  # one or two plain sentences for the window
    graphics: bool = False  # Gazebo's renderer failed: software OpenGL may help


_LAUNCH_EXC = re.compile(r"Caught exception in launch[^:]*:\s*(.+)")
_MISSING_PKG = re.compile(r"package '([^']+)' not found")
# A process launch gave up on; exit code -2 is our own SIGINT (venue switch, quit).
_DIED = re.compile(r"\[ERROR\] \[(gzserver|spawn_entity\.py)-\d+\]: process has died \[pid \d+, exit code (-?\d+)")


def diagnose(log: str) -> Problem | None:
    """Why a simulation run failed to start, from its launch log; None when nothing fatal shows."""
    for line in log.splitlines():
        m = _LAUNCH_EXC.search(line)
        if m:
            pkg = _MISSING_PKG.search(line)
            if pkg:
                return Problem(f"The ROS package '{pkg.group(1)}' is not installed. Quit, run ./wiloc.sh setup, then try again.")
            return Problem(f"The simulation could not be launched: {m.group(1).strip()}")
        m = _DIED.search(line)
        if m and m.group(2) != "-2":
            if m.group(1) == "gzserver":
                return Problem("Gazebo, the 3-D simulator, crashed while starting. "
                               "This usually means the graphics driver cannot run it.", graphics=True)
            return Problem("The robot could not be placed in the 3-D world (Gazebo did not answer).")
    return None


class Backend:
    def __init__(self, attach: bool = False, launch_args: list[str] | None = None, software_gl: bool = False):
        self.attach = attach
        self.launch_args = list(launch_args or [])
        self.software_gl = software_gl or os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1"
        self.proc: subprocess.Popen | None = None
        self.venue = self.layout = None
        self.started_at = None
        self.restarts = 0
        self.starting = False  # inside start(): stopping the old run, not yet launched
        self.closed = False  # shutdown(): the window is closing, start nothing more
        self._lock = threading.Lock()  # one start/stop at a time (they run on worker threads)
        self._log_pos = 0  # how far problem() has read this run's part of the log
        self._problem: Problem | None = None  # the first fatal line of this run (it cannot un-happen)

    @property
    def running(self) -> bool:
        """Launched (or about to be) and the launch process is alive."""
        return self.attach or self.starting or (self.proc is not None and self.proc.poll() is None)

    def problem(self) -> Problem | None:
        """Why this run is failing, read from its launch log (None: nothing fatal so far)."""
        if self.attach or self.starting or self.proc is None or self._problem is not None:
            return self._problem
        try:
            with open(LOG_PATH, "rb") as f:  # bytes: seek/tell offsets stay exact
                f.seek(self._log_pos)
                chunk = f.read()
        except OSError:
            return None
        end = chunk.rfind(b"\n") + 1  # whole lines only; a partial one is read next time
        self._log_pos += end
        self._problem = diagnose(chunk[:end].decode(errors="replace"))
        return self._problem

    def start(self, venue: str, layout: str = ""):
        self.venue, self.layout = venue, layout
        self.started_at = time.monotonic()
        if self.attach:
            return
        self.starting = True
        try:
            with self._lock:
                if self.closed:
                    return
                self._stop()
                os.makedirs(LOG_DIR, exist_ok=True)
                cmd = [
                    "ros2", "launch", "wiloc_ros", "sim.launch.py",
                    "ui:=none", "gui:=false", "rviz:=false",
                    f"floorplan:={venue}", f"ap_layout:={layout}",
                    *self.launch_args,
                ]
                # unbuffered: problem() reads the log while the launch runs
                env = dict(os.environ, GAZEBO_MODEL_DATABASE_URI="", PYTHONUNBUFFERED="1")
                if self.software_gl:
                    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
                log = open(LOG_PATH, "a")
                self._log_pos, self._problem = log.tell(), None
                log.write(f"\n===== {time.strftime('%H:%M:%S')}{' [software GL]' if self.software_gl else ''} {' '.join(cmd)}\n")
                log.flush()
                self.started_at = time.monotonic()
                self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                log.close()  # the child keeps its own copy
        finally:
            self.starting = False

    def stop(self, timeout: float = 10.0):
        if self.attach:
            return
        with self._lock:
            self._stop(timeout)

    def shutdown(self):
        """Stop for good: a start still queued on a worker thread will not launch anything."""
        self.closed = True
        self.stop()

    def _stop(self, timeout: float = 10.0):
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

    def start_async(self, venue: str, layout: str = "", restart: bool = False):
        """start() on a worker thread; ``running`` is true from this call on (no gap a watchdog could see)."""
        if self.closed:
            return
        if not self.attach:
            self.starting = True
            self.started_at = time.monotonic()
        if restart:
            self.restarts += 1
        threading.Thread(target=self.start, args=(venue, layout), daemon=True).start()
