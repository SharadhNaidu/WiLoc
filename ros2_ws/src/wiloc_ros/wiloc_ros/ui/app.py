"""WiLoc Studio: the whole demo in one window.

    ros2 run wiloc_ros studio                      # starts the simulation itself (office)
    ros2 run wiloc_ros studio --venue hospital     # another venue
    ros2 run wiloc_ros studio --attach             # use a simulation that is already running

Header: what this is, where we are, whether the robot knows where it is, and
how far off it is.  Left: the live 3-D world.  Right: the map of what the
robot believes.  Below: the method race and the things a visitor can try.

Visitor controls are the buttons and a click on the map.  Operator keys:
Ctrl+Shift+R soft reset, Ctrl+Shift+F forget (global re-localisation),
Ctrl+Shift+Q quit, F11 full screen; arrow keys / WASD drive by hand (the
Patrol button hands back to the autopilot).  Esc never quits.

Automation (checks, recordings): ``WILOC_SNAPSHOT_DIR`` saves a PNG of the
window every ``WILOC_SNAPSHOT_EVERY`` seconds; ``WILOC_STUDIO_SCRIPT`` runs
timed actions, e.g. ``"20:kidnap;45:layout=corridor;70:venue=home;100:quit"``.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
import time

import numpy as np
import rclpy
from PyQt5 import QtCore, QtGui, QtWidgets
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from wiloc.environment import FloorPlan
from wiloc.story import AttractMode, RecoveryClock, merge_modes, tour_stops

from . import style as S
from .backend import LOG_PATH, Backend
from .bridge import Bridge
from .map_view import MapView
from .recorder import Recorder
from .panels import Card, CameraView, ChipRow, Header, Leaderboard, Overlay, Pill, RaceChart, Toast

START_ATTEMPTS = 3  # tries to bring a venue's simulation up before showing why it failed
START_TIMEOUT = 90.0  # s without the robot and localiser showing up: count the try as failed (normally < 10 s)
TELEOP_KEYS = {
    QtCore.Qt.Key_Up: "fwd", QtCore.Qt.Key_W: "fwd", QtCore.Qt.Key_Down: "back", QtCore.Qt.Key_S: "back",
    QtCore.Qt.Key_Left: "left", QtCore.Qt.Key_A: "left", QtCore.Qt.Key_Right: "right", QtCore.Qt.Key_D: "right",
}


def localisation_state(status: dict | None) -> str:
    """LOCKED / UNCERTAIN / SEARCHING from the localiser's status (its own ``state`` when it sends one)."""
    if not status:
        return "WAITING"
    if status.get("state"):
        return status["state"]
    if status.get("lost"):
        return "SEARCHING"
    return "LOCKED" if status.get("spread_m", 99) < 1.0 else "UNCERTAIN"


def caption(fp: FloorPlan, status: dict | None, state: str) -> str:
    """One plain-English line about what the robot believes."""
    if not status:
        return "Waiting for the first Wi-Fi scan"
    x, y = status["pose"][:2]
    room = fp.room_at((x, y))
    if state == "LOCKED":
        return f"Sure it is in the {room}" if room else "Sure where it is"
    if state == "SEARCHING":
        return "Lost: comparing the Wi-Fi it hears with every place in the building"
    modes = [m for m in merge_modes(status.get("modes") or []) if m[2] >= 0.1]
    rooms = []
    for mx, my, w in modes:
        r = fp.room_at((mx, my)) or "somewhere"
        if r not in [a for a, _ in rooms]:
            rooms.append((r, w))
    if len(rooms) >= 2:
        return f"Not sure: {rooms[0][0]} ({rooms[0][1] * 100:.0f}%) or {rooms[1][0]} ({rooms[1][1] * 100:.0f}%)"
    return f"Narrowing it down{' in the ' + room if room else ''}"


class Studio(QtWidgets.QWidget):
    def __init__(self, bridge: Bridge, backend: Backend, venue: FloorPlan):
        super().__init__()
        self.bridge, self.backend = bridge, backend
        self.venue = venue
        self.setWindowTitle("WiLoc Studio")
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.held = set()
        self.v = self.w = 0.0
        self.err_ema = None
        self.last_kidnaps = 0
        self.switching = None  # venue being loaded
        self.switch_t = 0.0
        self.attempt = 1  # of START_ATTEMPTS, for the venue being loaded
        self.gave_up = False  # every attempt failed: the overlay says why
        self.ready = False
        self.body_on = True
        self.manual = False
        self.recorder = None  # --record

        # ---- widgets
        self.header = Header()
        self.venue_chips = ChipRow(size=13)
        self.layout_chips = ChipRow(size=13)
        self.venue_label = QtWidgets.QLabel("Venue")
        self.layout_label = QtWidgets.QLabel("Routers")
        self.camera_card = Card("The real world", "live 3-D simulation")
        self.camera = CameraView()
        self.camera_card.body.addWidget(self.camera)
        self.map_card = Card("What the robot believes", "")
        self.map = MapView()
        self.map_card.body.addWidget(self.map)
        self.race_card = Card("The race", "who finds the robot best")
        self.leaderboard = Leaderboard()
        self.chart = RaceChart()
        self.race_card.body.addWidget(self.leaderboard)
        self.race_card.body.addWidget(self.chart, 1)
        self.actions_card = Card("Try it", "")
        self.btn_kidnap = Pill("Kidnap the robot", "primary", size=16)
        self.btn_unplug = Pill("Unplug a router", size=15)
        self.btn_people = Pill("People block Wi-Fi: on", size=15)
        self.btn_patrol = Pill("Patrol", size=15)
        self.btn_view = Pill("View: overview", size=15)
        self.hint = QtWidgets.QLabel("Or click anywhere on the map to send the robot there.")
        self.hint.setWordWrap(True)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.btn_kidnap, 0, 0, 1, 2)
        grid.addWidget(self.btn_unplug, 1, 0)
        grid.addWidget(self.btn_people, 1, 1)
        grid.addWidget(self.btn_patrol, 2, 0)
        grid.addWidget(self.btn_view, 2, 1)
        grid.addWidget(self.hint, 3, 0, 1, 2)
        grid.setRowStretch(4, 1)
        self.actions_grid = grid
        self.actions_card.body.addLayout(grid)
        self.overlay = Overlay(self)
        self.toast = Toast(self)
        self.hint_text = "Or click anywhere on the map to send the robot there."

        # ---- story: re-found timer and the unattended tour
        self.recovery = RecoveryClock()
        self.attract = AttractMode(idle=90.0)
        self.attract.enabled = False
        self.attract.touch(time.monotonic())
        self.tour = tour_stops({n: list(FloorPlan.builtin(n).layouts) for n in FloorPlan.builtin_names()})

        # ---- layout
        chips = QtWidgets.QHBoxLayout()
        chips.addWidget(self.venue_label)
        chips.addWidget(self.venue_chips, 0)
        chips.addStretch(1)
        chips.addWidget(self.layout_label)
        chips.addWidget(self.layout_chips, 0)
        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.camera_card, 11)
        body.addWidget(self.map_card, 10)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.race_card, 14)
        bottom.addWidget(self.actions_card, 7)
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self.header)
        root.addLayout(chips)
        root.addLayout(body, 62)
        root.addLayout(bottom, 30)
        self.rows = (chips, body, bottom, root)

        # ---- signals
        self.venue_chips.chosen.connect(self.switch_venue)
        self.layout_chips.chosen.connect(self.switch_layout)
        self.map.goal_clicked.connect(self.on_goal)
        self.map.ap_clicked.connect(self.on_ap_clicked)
        self.btn_kidnap.clicked.connect(self.kidnap)
        self.btn_people.clicked.connect(self.toggle_people)
        self.btn_patrol.clicked.connect(self.patrol)
        self.btn_view.clicked.connect(self.toggle_view)
        self.btn_unplug.clicked.connect(self.unplug)
        self.btn_unplug.setToolTip("Switch off the router the robot hears loudest (or click a router on the map)")
        self.view_mode = "overview"

        self._load_venue(venue)
        QtCore.QTimer(self, timeout=self.frame, interval=50).start()
        QtCore.QTimer(self, timeout=self.sample, interval=200).start()
        QtCore.QTimer(self, timeout=self.watchdog, interval=1000).start()
        QtCore.QTimer(self, timeout=self.teleop, interval=50).start()
        self.rescale()

    # ------------------------------------------------------------ sizing
    def resizeEvent(self, e):
        self.rescale()
        self.overlay.setGeometry(self.rect())
        super().resizeEvent(e)

    def eventFilter(self, obj, e):
        if e.type() in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.KeyPress, QtCore.QEvent.Wheel):
            if self.attract.active:
                self.hint.setText(self.hint_text)
            self.attract.touch(time.monotonic())  # a visitor is here: the tour steps aside
        return False

    def rescale(self):
        S.set_scale(min(self.width() / 1600.0, self.height() / 940.0) if self.width() > 0 else 1.0)
        m = S.px(20)
        chips, body, bottom, root = self.rows
        root.setContentsMargins(m, S.px(14), m, m)
        root.setSpacing(S.px(12))
        for lay in (body, bottom):
            lay.setSpacing(S.px(14))
        chips.setSpacing(S.px(8))
        self.header.setFixedHeight(S.px(84))
        for lab in (self.venue_label, self.layout_label):
            lab.setFont(S.font(13, S.SEMIBOLD))
            lab.setStyleSheet(f"color: {S.MUTED};")
        self.hint.setFont(S.font(14))
        self.hint.setStyleSheet(f"color: {S.MUTED}; background: transparent;")
        self.leaderboard.setFixedHeight(S.px(40))
        for c in (self.camera_card, self.map_card, self.race_card, self.actions_card):
            c.rescale()
        self.venue_chips.rescale()
        self.layout_chips.rescale()
        for b in (self.btn_kidnap, self.btn_unplug, self.btn_people, self.btn_patrol, self.btn_view):
            b.setMinimumHeight(b.sizeHint().height())
            b.setMinimumWidth(S.px(120))
        self.actions_grid.setSpacing(S.px(10))
        self.update()

    # ------------------------------------------------------------ venue / layout
    def _load_venue(self, venue: FloorPlan):
        self.venue = venue
        self.bridge.set_venue(venue)
        self.map.set_plan(venue)
        self.map.disabled_aps = set()
        self.venue_chips.set_items(
            [(n, FloorPlan.builtin(n).title, FloorPlan.builtin(n).caption) for n in FloorPlan.builtin_names()], venue.name
        )
        self.layout_chips.set_items([(k, lab, lab) for k, (lab, _) in venue.layouts.items()], venue.layout)
        self.chart.clear()
        self.err_ema = None
        self.last_kidnaps = None  # first evaluator message of this venue sets the baseline
        self.recovery = RecoveryClock()
        self.rescale()

    def switch_venue(self, name: str, layout: str | None = None):
        if (self.switching and not self.gave_up) or name == self.venue.name:
            self.venue_chips.select(self.venue.name)
            if not self.switching and layout and layout in self.venue.layouts:
                self.layout_chips.select(layout)
                self.switch_layout(layout)
            return
        if self.backend.attach:
            self.venue_chips.select(self.venue.name)
            self.overlay.show_message("Venue switching needs the Studio to run the simulation",
                                      "Started with --attach: restart without it to switch venues.")
            QtCore.QTimer.singleShot(3000, self.overlay.hide)
            return
        venue = FloorPlan.builtin(name, layout if layout in FloorPlan.builtin(name).layouts else None)
        self.begin_loading(name)
        self.overlay.show_message(f"Loading {venue.title}…", venue.caption)
        self._load_venue(venue)
        # the old venue's latched choices would otherwise reach the new simulation
        self.bridge.set_layout(venue.layout)
        self._set_disabled(set())
        self.backend.start_async(name, venue.layout)

    def begin_loading(self, name: str):
        """A venue's simulation is (re)starting: the overlay stays until it is up (``frame``) or has failed (``watchdog``)."""
        self.switching, self.switch_t = name, time.monotonic()
        self.attempt, self.gave_up = 1, False

    def switch_layout(self, name: str):
        if name == self.bridge.fp.layout:
            return
        self.bridge.set_layout(name)
        self.map.set_plan(self.bridge.fp)
        self._set_disabled(set())

    # ------------------------------------------------------------ actions
    def on_goal(self, x, y):
        self.manual = False
        self.bridge.send_goal(x, y)

    def on_ap_clicked(self, i):
        ap = self.bridge.fp.access_points[i].id
        self._set_disabled(self.map.disabled_aps ^ {ap})

    def _set_disabled(self, ids):
        self.map.disabled_aps = set(ids)
        self.bridge.set_disabled(ids)
        self.btn_unplug.setText("Plug the router back in" if ids else "Unplug a router")
        self.btn_unplug.setMinimumWidth(S.px(120))

    def unplug(self):
        """Switch off the loudest router (the one the robot leans on most); pressed again: plug everything back in."""
        if self.map.disabled_aps:
            self._set_disabled(set())
            return
        scan = self.bridge.snapshot().scan or {}
        heard = [(a.get("measured_dbm"), a["id"]) for a in scan.get("aps", []) if a.get("measured_dbm") is not None]
        if heard:
            self._set_disabled({max(heard)[1]})

    def kidnap(self):
        self.bridge.kidnap()
        self.recovery.kidnap(time.monotonic())

    def toggle_people(self):
        self.body_on = not self.body_on
        self.bridge.set_body_shadowing(self.body_on)
        self.btn_people.setText(f"People block Wi-Fi: {'on' if self.body_on else 'off'}")
        self.btn_people.setFixedSize(self.btn_people.sizeHint())

    def patrol(self):
        self.manual = False
        self.bridge.set_mode("auto")

    def toggle_view(self):
        self.view_mode = "follow" if self.view_mode == "overview" else "overview"
        self.bridge.set_camera_mode(self.view_mode)
        self.btn_view.setText(f"View: {self.view_mode}")

    def soft_reset(self):
        self.bridge.soft_reset()
        self.body_on = False
        self.toggle_people()
        self.layout_chips.select(next(iter(self.venue.layouts)))
        self.map.set_plan(self.bridge.fp)
        self._set_disabled(set())
        self.manual = False
        self.chart.clear()

    def set_attract(self, on: bool):
        """Unattended tour (Ctrl+Shift+A): patrol, kidnap now and then, move through the venues."""
        self.attract.enabled = on
        self.attract.touch(time.monotonic())
        self.toast.show_message(f"Auto tour {'on' if on else 'off'}",
                                "Starts after 90 s without a visitor" if on else "", S.INDIGO, secs=3.0)

    def run_attract(self):
        for action in self.attract.due(time.monotonic()):
            if action == "start":
                self.soft_reset()
                self.hint.setText("Auto tour running: touch anything to take over.")
            elif action == "kidnap" and not self.switching and self.ready:
                self.kidnap()
            elif action == "rotate" and not self.backend.attach:
                here = (self.venue.name, self.bridge.fp.layout)
                i = self.tour.index(here) + 1 if here in self.tour else 0
                name, layout = self.tour[i % len(self.tour)]
                if name != self.venue.name:
                    self.switch_venue(name, layout)
                else:
                    self.layout_chips.select(layout)
                    self.switch_layout(layout)

    # ------------------------------------------------------------ keyboard
    def keyPressEvent(self, e):
        mods = e.modifiers()
        if mods & QtCore.Qt.ControlModifier and mods & QtCore.Qt.ShiftModifier:
            if e.key() == QtCore.Qt.Key_R:
                self.soft_reset()
            elif e.key() == QtCore.Qt.Key_A:
                self.set_attract(not self.attract.enabled)
            elif e.key() == QtCore.Qt.Key_F:
                self.bridge.forget()
            elif e.key() == QtCore.Qt.Key_Q:
                self.close()
            return
        if e.key() == QtCore.Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif e.key() in TELEOP_KEYS and not e.isAutoRepeat():
            self.held.add(TELEOP_KEYS[e.key()])
            if not self.manual:
                self.manual = True
                self.bridge.set_mode("manual")

    def keyReleaseEvent(self, e):
        if e.key() in TELEOP_KEYS and not e.isAutoRepeat():
            self.held.discard(TELEOP_KEYS[e.key()])

    def teleop(self):
        if not self.manual:
            self.v = self.w = 0.0
            return
        tv = 0.3 * (("fwd" in self.held) - ("back" in self.held))
        tw = 1.2 * (("left" in self.held) - ("right" in self.held))
        self.v += float(np.clip(tv - self.v, -0.06, 0.06))
        self.w += float(np.clip(tw - self.w, -0.4, 0.4))
        self.bridge.drive(self.v, self.w)

    # ------------------------------------------------------------ updates
    def frame(self):
        snap = self.bridge.snapshot()
        self.camera.set_scene(snap, self.bridge.fp)
        self.camera.set_frame(snap.image, snap.image_count)
        self.map.set_snapshot(snap)
        if self.backend.attach and snap.status and not self.switching:
            self._follow_simulation(snap.status)
        st = snap.status if snap.status and snap.status.get("venue", self.venue.name) == self.venue.name else None
        state = localisation_state(st)
        self.header.state = state
        if snap.truth is not None and "pf" in snap.poses:
            e = math.hypot(snap.poses["pf"][0] - snap.truth[0], snap.poses["pf"][1] - snap.truth[1])
            self.err_ema = e if self.err_ema is None else self.err_ema + 0.1 * (e - self.err_ema)
        self.header.error = self.err_ema
        self.header.caption = caption(self.venue, st, state)
        fp = self.bridge.fp
        self.map_card.subtitle = f"{fp.title} · {fp.layout_label}"
        self.map_card.right_text = "click to send the robot"
        if snap.errors and st is not None:
            self.leaderboard.medians = snap.errors.get("median") or {}
            racers = {m: v for m, v in self.leaderboard.medians.items() if m in ("pf", "ekf", "wknn") and v is not None}
            self.leaderboard.leader = min(racers, key=racers.get) if racers else None  # wheels-only is the baseline
            k = snap.errors.get("kidnaps", 0)
            if self.last_kidnaps is not None and k > self.last_kidnaps:
                self.chart.mark_kidnap(time.monotonic())
                if not self.recovery.waiting:  # kidnapped from elsewhere (keyboard, another terminal)
                    self.recovery.kidnap(time.monotonic())
            self.last_kidnaps = k
        if self.switching and st and snap.truth is not None and time.monotonic() - self.switch_t > 3:
            self.switching = None
            self.overlay.hide()
            self.attract.venue_loaded(time.monotonic())
        cam = self.camera_card.geometry()
        self.toast.place(cam.center().x(), cam.top() + S.px(56))
        self.toast.tick()
        if not self.ready and st and snap.truth is not None:
            self.ready = True
            if not self.switching:
                self.overlay.hide()
        self.header.update()
        self.map_card.update()
        self.leaderboard.update()
        self.chart.update()
        if self.overlay.isVisible():
            self.overlay.update()

    def _follow_simulation(self, status):
        """Attached to a simulation someone else controls: show its venue and router preset."""
        venue, layout = status.get("venue"), status.get("layout")
        if venue and venue != self.venue.name and venue in FloorPlan.builtin_names():
            self._load_venue(FloorPlan.builtin(venue, layout or None))
        elif layout and layout != self.bridge.fp.layout and time.monotonic() - self.bridge.layout_requested > 3.0:
            self.bridge.adopt_layout(layout)
            self.map.set_plan(self.bridge.fp)
            self.layout_chips.select(layout)

    def sample(self):
        snap = self.bridge.snapshot()
        if snap.truth is None or self.switching or not snap.status or snap.status.get("venue") != self.venue.name:
            return  # nothing yet, or still the previous venue's simulation
        tx, ty = snap.truth[:2]
        errs = {m: math.hypot(p[0] - tx, p[1] - ty) for m, p in snap.poses.items()}
        took = self.recovery.update(time.monotonic(), errs.get("pf"))
        if took is not None:
            self.toast.show_message(f"Re-found in {took:.1f} s", "kidnapped, then found again using only Wi-Fi",
                                    S.STATE_COLORS["LOCKED"])
        if snap.errors:
            errs["odometry"] = (snap.errors.get("current") or {}).get("odometry")
        self.chart.add(time.monotonic(), errs)

    def watchdog(self):
        if self.backend.closed:
            return
        self.run_attract()
        snap = self.bridge.snapshot()
        silent = snap.clock_wall is None or time.monotonic() - snap.clock_wall > 8.0
        if self.backend.attach:
            if silent and self.ready:
                self.overlay.show_message("Waiting for the simulation…", "Nothing heard from Gazebo for a while.")
            elif not silent and self.overlay.isVisible() and self.ready:
                self.overlay.hide()
            return
        if self.switching:
            self._watch_startup()
            return
        # running: restart a simulation that died or went quiet
        problem = self.backend.problem()
        if not self.backend.running or problem is not None or silent:
            self.begin_loading(self.venue.name)
            self.overlay.show_message("Restarting the simulation…", problem.reason if problem else "Back in a few seconds.")
            self._load_venue(self.venue)
            self.backend.start_async(self.venue.name, self.venue.layout, restart=True)

    def _watch_startup(self):
        """While a venue loads: retry a run that failed (software graphics if Gazebo crashed), then say why."""
        if self.gave_up or self.backend.starting:
            return
        since = time.monotonic() - (self.backend.started_at or self.switch_t)
        problem = self.backend.problem()
        if problem is None and self.backend.running and since <= START_TIMEOUT:
            if since > 45:
                self.overlay.show_message("Still loading…", f"The simulation is taking longer than usual ({since:.0f} s).")
            return
        if problem is not None:
            why = problem.reason
        elif not self.backend.running:
            why = "The simulation stopped while starting."
        else:
            why = f"Nothing from the simulation after {since:.0f} s."
        if self.attempt < START_ATTEMPTS:
            self.attempt += 1
            # Gazebo that crashed or hangs (stuck creating its 3-D renderer, typical in VMs): software graphics
            graphics = problem.graphics if problem is not None else self.backend.running
            if graphics and not self.backend.software_gl:
                self.backend.software_gl = True
                why += " Trying again with software graphics (slower, but works without a 3-D graphics driver)."
            self.overlay.show_message(f"Trying again… (attempt {self.attempt} of {START_ATTEMPTS})", why)
            self.backend.start_async(self.venue.name, self.venue.layout, restart=True)
            return
        self.gave_up = True
        log = LOG_PATH.replace(os.path.expanduser("~"), "~", 1)
        self.overlay.show_message(
            "WiLoc could not start the simulation",
            f"{why}\n\nPress Ctrl+Shift+Q to quit, then send the file {log} to the team.",
            busy=False,
        )

    def closeEvent(self, e):
        if self.recorder is not None:
            rec, self.recorder = self.recorder, None
            path = rec.stop()
            print(f"[studio] video: {path} ({rec.written} frames, {rec.dropped} dropped)", flush=True)
        self.backend.closed = True  # before processEvents: the watchdog must not restart what we stop
        self.overlay.show_message("Stopping…")
        QtWidgets.QApplication.processEvents()
        self.backend.shutdown()
        super().closeEvent(e)


# ====================================================================== main
def _run_script(app, win: Studio, script: str):
    """Timed actions for checks and recordings: "t:action;t:action" (seconds after start)."""
    for item in filter(None, (s.strip() for s in script.split(";"))):
        t, action = item.split(":", 1)
        name, _, arg = action.partition("=")

        def run(name=name, arg=arg):
            if name == "kidnap":
                win.kidnap()
            elif name == "layout":
                win.layout_chips.select(arg)
                win.switch_layout(arg)
            elif name == "venue":
                win.switch_venue(arg)
            elif name == "people":
                if (arg == "on") != win.body_on:
                    win.toggle_people()
            elif name == "goal":
                x, y = (float(v) for v in arg.split(","))
                win.on_goal(x, y)
            elif name == "unplug":
                win.unplug()
            elif name == "forget":
                win.bridge.forget()
            elif name == "reset":
                win.soft_reset()
            elif name == "view":
                if arg != win.view_mode:
                    win.toggle_view()
            elif name == "attract":
                win.set_attract(arg != "off")
            elif name == "quit":
                win.close()
                app.quit()

        QtCore.QTimer.singleShot(int(float(t) * 1000), run)


def main(argv=None):
    p = argparse.ArgumentParser(description="WiLoc Studio")
    p.add_argument("--venue", default=os.environ.get("WILOC_VENUE", "office"))
    p.add_argument("--layout", default="")
    p.add_argument("--attach", action="store_true", help="use a running simulation instead of starting one")
    p.add_argument("--size", default="", help="window size WxH (default: full screen)")
    p.add_argument("--windowed", action="store_true", help="a normal window instead of full screen")
    p.add_argument("--launch-arg", action="append", default=[], help="extra sim.launch.py argument (repeatable)")
    p.add_argument("--record", default="", help="record the window to this video file (.mp4)")
    p.add_argument("--record-fps", type=float, default=15.0)
    p.add_argument("--attract", action="store_true", help="unattended tour after 90 s without a visitor (Ctrl+Shift+A)")
    p.add_argument("--no-overlay", action="store_true", help=argparse.SUPPRESS)  # layout reviews without a simulation
    a, _ = p.parse_known_args(argv if argv is not None else sys.argv[1:])

    if "ROS_LOCALHOST_ONLY" not in os.environ:
        is_wsl = False
        try:
            with open("/proc/version", "r") as f:
                is_wsl = "microsoft" in f.read().lower()
        except OSError:
            pass
        os.environ["ROS_LOCALHOST_ONLY"] = "0" if is_wsl else "1"

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    venue = FloorPlan.builtin(a.venue, a.layout or None)
    bridge = Bridge(venue)
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    threading.Thread(target=executor.spin, daemon=True).start()

    app = QtWidgets.QApplication(sys.argv[:1])
    S.load_fonts()
    app.setStyleSheet(S.qss())
    backend = Backend(attach=a.attach, launch_args=a.launch_arg)
    win = Studio(bridge, backend, venue)
    app.installEventFilter(win)
    win.attract.enabled = a.attract
    if os.environ.get("WILOC_ATTRACT_TIMES"):  # checks: "idle,kidnap_every,rotate_every,first_kidnap" in seconds
        idle, kev, rev, first = (float(v) for v in os.environ["WILOC_ATTRACT_TIMES"].split(","))
        win.attract.idle, win.attract.kidnap_every, win.attract.rotate_every, win.attract.first_kidnap = idle, kev, rev, first
    if a.size:
        w, h = (int(v) for v in a.size.lower().split("x"))
        win.resize(w, h)
        win.show()
    elif a.windowed:
        win.resize(1600, 940)
        win.show()
    else:
        win.showFullScreen()
    if not a.no_overlay:
        win.overlay.show_message("Starting WiLoc…", venue.caption)
    if not a.attach:  # a fresh simulation: make its router preset the latched one
        win.begin_loading(venue.name)
        backend.start_async(venue.name, venue.layout)
        bridge.set_layout(venue.layout)

    signal.signal(signal.SIGINT, lambda *_: (win.close(), app.quit()))
    signal.signal(signal.SIGTERM, lambda *_: (win.close(), app.quit()))
    QtCore.QTimer(timeout=lambda: None, interval=200).start()  # let Python handle signals

    snap_dir = os.environ.get("WILOC_SNAPSHOT_DIR")
    if snap_dir:
        os.makedirs(snap_dir, exist_ok=True)
        counter = {"i": 0}

        def snapshot():
            counter["i"] += 1
            win.grab().save(os.path.join(snap_dir, f"studio_{counter['i']:03d}.png"))

        QtCore.QTimer(win, timeout=snapshot, interval=int(1000 * float(os.environ.get("WILOC_SNAPSHOT_EVERY", 10)))).start()
    if a.record:
        win.recorder = Recorder(win, os.path.expanduser(a.record), fps=a.record_fps)
        win.recorder.start()
    if os.environ.get("WILOC_STUDIO_SCRIPT"):
        _run_script(app, win, os.environ["WILOC_STUDIO_SCRIPT"])

    app.exec_()
    backend.stop()
    try:
        executor.cancel()
    except Exception:
        pass
    try:
        executor.shutdown(timeout_sec=1.0)
    except Exception:
        pass
    try:
        bridge.destroy_node()
    except Exception:
        pass
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
