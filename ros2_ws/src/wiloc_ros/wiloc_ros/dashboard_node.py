"""WiLoc dashboard: one window with tabs for everything, plus robot control.

A PyQt5 window that is also a ROS 2 node.  A status bar and the controls stay
on top; the tabs below show

  Live map      floor plan with the RSSI heat map (H cycles APs), walls by
                material, APs, particle cloud, true pose vs. estimate with its
                2-sigma ellipse, raw Wi-Fi fix, robot->AP links coloured by
                RSSI, and the true / estimated / dead-reckoning paths;
  Wi-Fi         latest scan vs. the radio map's prediction at the estimate and
                a table of how each simulated reading was formed;
  Localization  filter / Wi-Fi-only / odometry-only error over time and the
                particle filter's internals;
  ROS graph     the WiLoc nodes and topics with live message rates.

Driving works from this window or from the Gazebo window (key presses there
arrive on /wiloc/gazebo_keys via the wiloc_gazebo plugin):

  W / Up      forward          M   toggle auto patrol / manual driving
  S / Down    backward         G   global re-localisation (forget the pose)
  A / Left    turn left        K   kidnap: teleport the robot somewhere random
  D / Right   turn right       R   reset the filter at the true pose
  Shift       2x speed         H   cycle the RSSI heat map (strongest / per AP)
  Space       stop

Pressing a drive key while the patrol is running switches to manual mode.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import deque

import numpy as np
import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int32, String
from std_srvs.srv import Empty
from tf2_msgs.msg import TFMessage

from wiloc.radio import RadioConfig, RadioSimulator
from wiloc_msgs.msg import RssiScan

from . import theme
from .common import load_floorplan, pose2d_from_odom, quat_to_yaw, yaw_to_quat

import matplotlib  # noqa: E402

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Patch  # noqa: E402
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

BG, PANEL, FG, MUTED, GRID = theme.BG, theme.PANEL, theme.TEXT, theme.MUTED, theme.BORDER
C_FILTER, C_WIFI, C_ODOM, C_MAP = theme.C_FILTER, theme.C_WIFI, theme.C_ODOM, theme.C_MAP
OK, BAD = theme.TEAL, theme.CORAL
HEAT_CMAP = LinearSegmentedColormap.from_list("wiloc_heat", [(s, c) for s, c in theme.HEAT_STOPS])
WALL_WIDTH = {"concrete": 4.5, "brick": 3.5, "drywall": 2.5, "glass": 2.5, "wood": 2.5}

# Qt key codes as reported by Gazebo's KeyboardGUIPlugin (forwarded on /wiloc/gazebo_keys)
GZ_DRIVE_KEYS = {
    0x57: "fwd", 0x01000013: "fwd",  # W, Up
    0x53: "back", 0x01000015: "back",  # S, Down
    0x41: "left", 0x01000012: "left",  # A, Left
    0x44: "right", 0x01000014: "right",  # D, Right
}
GZ_HOLD_S = 0.45  # Gazebo reports presses only: a key counts as held this long per press

# Live node graph: node name -> (label, x, y); edges: (src, dst, topic)
GRAPH_NODES = {
    "wiloc_dashboard": ("dashboard\n(keyboard)", 1.3, 5.4),
    "gazebo": ("Gazebo\n(physics)", 1.3, 3.0),
    "waypoint_driver": ("waypoint\ndriver", 1.3, 0.6),
    "odom_noise": ("odom_noise", 5.2, 5.0),
    "wifi_sensor": ("wifi_sensor\n(RSSI sim)", 5.2, 1.0),
    "wiloc_localization": ("localization\n(PF / EKF)", 8.6, 3.0),
    "wiloc_evaluator": ("evaluator", 11.2, 5.0),
}
GRAPH_EDGES = [
    ("gazebo", "odom_noise", "/odom"),
    ("odom_noise", "wiloc_localization", "/wheel_odom"),
    ("gazebo", "wifi_sensor", "/ground_truth"),
    ("wifi_sensor", "wiloc_localization", "/wifi/scan"),
    ("wiloc_localization", "wiloc_evaluator", "/wiloc/pose"),
    ("waypoint_driver", "gazebo", "/cmd_vel"),
    ("wiloc_dashboard", "gazebo", "/cmd_vel"),
]
BOX_W, BOX_H = 2.3, 0.95


def _box_exit(x1, y1, x2, y2):
    """Point where the segment from box centre (x1, y1) towards (x2, y2) leaves the box."""
    dx, dy = x2 - x1, y2 - y1
    t = min((BOX_W / 2 + 0.08) / abs(dx) if dx else 1e9, (BOX_H / 2 + 0.08) / abs(dy) if dy else 1e9)
    return x1 + dx * t, y1 + dy * t


def rssi_rgb(rssi, lo=-85.0, hi=-30.0):
    """Darker 40-100 % part of the heat ramp: readable as text / lines on white."""
    return tuple(theme.heat_rgb(0.4 + 0.6 * np.clip((rssi - lo) / (hi - lo), 0, 1)))


class RateMeter:
    """Message rate over a sliding window."""

    def __init__(self, window=3.0):
        self.window, self.t = window, deque()

    def tick(self):
        self.t.append(time.monotonic())

    def hz(self):
        now = time.monotonic()
        while self.t and now - self.t[0] > self.window:
            self.t.popleft()
        return len(self.t) / self.window


class Track:
    """Bounded 2-D path; a jump (e.g. a kidnapping) breaks the line."""

    def __init__(self, maxlen=4000, min_step=0.05, jump=1.0):
        self.pts, self.min_step, self.jump = deque(maxlen=maxlen), min_step, jump
        self.last = None

    def add(self, x, y):
        if self.last is not None:
            d = math.hypot(x - self.last[0], y - self.last[1])
            if d < self.min_step:
                return
            if d > self.jump:
                self.pts.append((np.nan, np.nan))
        self.pts.append((x, y))
        self.last = (x, y)

    def xy(self):
        a = np.array(list(self.pts))
        return (a[:, 0], a[:, 1]) if len(a) else ([], [])


class DashboardNode(Node):
    def __init__(self):
        super().__init__("wiloc_dashboard")
        self.declare_parameter("floorplan", "office")
        self.declare_parameter("radio_seed", 7)
        self.declare_parameter("v_max", 0.22)
        self.declare_parameter("w_max", 1.2)
        self.declare_parameter("mode", "auto")
        self.fp = load_floorplan(self.get_parameter("floorplan").value)
        self.radio_seed = int(self.get_parameter("radio_seed").value)
        self.v_max = float(self.get_parameter("v_max").value)
        self.w_max = float(self.get_parameter("w_max").value)
        self.start = np.array(self.fp.start_pose, dtype=float)

        # latest data (written by the ROS thread, read by the GUI thread)
        self.truth = self.est = self.est_cov = self.fix = None
        self.particles = np.zeros((0, 2))
        self.scan_debug = None
        self.status = None
        self.driver_status = "waiting..."
        self.errors = deque(maxlen=4000)  # (t, filter, wifi, odom)
        self.internals = deque(maxlen=4000)  # (t, ess_frac, spread)
        self.rmse = {}
        self.tracks = {"truth": Track(), "est": Track(), "odom": Track()}
        self.rates = {topic: RateMeter() for _, _, topic in GRAPH_EDGES}
        self.rates["/tf"] = RateMeter()
        self.rates["/wiloc/particles"] = RateMeter()

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.mode_pub = self.create_publisher(String, "/wiloc/drive_mode", 10)
        self.heat_pub = self.create_publisher(Int32, "/wiloc/heatmap_select", 10)
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.global_cli = self.create_client(Empty, "/wiloc/global_localization")
        self.teleport_cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")

        sensor = qos_profile_sensor_data
        self.create_subscription(Odometry, "/ground_truth", self.on_truth, sensor)
        self.create_subscription(Odometry, "/odom", lambda m: self.rates["/odom"].tick(), sensor)
        self.create_subscription(Odometry, "/wheel_odom", self.on_wheel_odom, 20)
        self.create_subscription(RssiScan, "/wifi/scan", lambda m: self.rates["/wifi/scan"].tick(), 10)
        self.create_subscription(String, "/wifi/scan_debug", self.on_scan_debug, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/wiloc/pose", self.on_pose, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/wiloc/wifi_fix", self.on_fix, 10)
        self.create_subscription(PoseArray, "/wiloc/particles", self.on_particles, 5)
        self.create_subscription(Twist, "/cmd_vel", lambda m: self.rates["/cmd_vel"].tick(), 10)
        self.create_subscription(TFMessage, "/tf", lambda m: self.rates["/tf"].tick(), 50)
        self.create_subscription(String, "/wiloc/status", self.on_status, 10)
        self.create_subscription(String, "/wiloc/errors", self.on_errors, 10)
        self.create_subscription(String, "/wiloc/driver_status", self.on_driver_status, 10)
        self.create_subscription(Int32, "/wiloc/gazebo_keys", self.on_gazebo_key, 50)
        self.gz_until = {}  # drive direction -> time until which a Gazebo key counts as held
        self._gz_last_cmd = {}

        self.mode = str(self.get_parameter("mode").value)
        self.held = set()  # drive directions held in the dashboard window (GUI thread)
        self.v = self.w = 0.0
        self.create_timer(0.05, self.teleop_tick)
        self.heat_choice = -1
        self.log = deque(maxlen=8)

    # ------------------------------------------------------------ callbacks
    def on_truth(self, msg):
        self.rates["/ground_truth"].tick()
        self.truth = np.array(pose2d_from_odom(msg))
        self.tracks["truth"].add(self.truth[0], self.truth[1])

    def on_wheel_odom(self, msg):
        # Dead reckoning in the map frame, anchored at the start pose (as the evaluator does).
        self.rates["/wheel_odom"].tick()
        x, y, _ = pose2d_from_odom(msg)
        c, s = math.cos(self.start[2]), math.sin(self.start[2])
        self.tracks["odom"].add(self.start[0] + c * x - s * y, self.start[1] + s * x + c * y)

    def on_pose(self, msg):
        self.rates["/wiloc/pose"].tick()
        p = msg.pose.pose
        self.est = np.array([p.position.x, p.position.y, quat_to_yaw(p.orientation)])
        c = msg.pose.covariance
        self.est_cov = np.array([[c[0], c[1]], [c[6], c[7]]])
        self.tracks["est"].add(self.est[0], self.est[1])

    def on_fix(self, msg):
        self.fix = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    def on_particles(self, msg):
        self.rates["/wiloc/particles"].tick()
        self.particles = np.array([(p.position.x, p.position.y) for p in msg.poses]).reshape(-1, 2)

    def on_scan_debug(self, msg):
        self.scan_debug = json.loads(msg.data)

    def on_status(self, msg):
        self.status = json.loads(msg.data)
        if "ess" in self.status:
            self.internals.append((self.sim_time(), self.status["ess"] / self.status["n_particles"], self.status["spread_m"]))

    def on_errors(self, msg):
        d = json.loads(msg.data)
        c = d["current"]
        self.errors.append((d["t"], c.get("filter", np.nan), c.get("wifi_fix", np.nan), c.get("odometry", np.nan)))
        if d.get("rmse"):
            self.rmse = d["rmse"]

    def on_driver_status(self, msg):
        self.driver_status = msg.data

    def on_gazebo_key(self, msg):
        """Keys pressed in the Gazebo window (via the wiloc_gazebo key forwarder)."""
        code, now = int(msg.data), time.monotonic()
        if code in GZ_DRIVE_KEYS:
            if self.mode != "manual":
                self.set_mode("manual")
            self.gz_until[GZ_DRIVE_KEYS[code]] = now + GZ_HOLD_S
            return
        actions = {
            0x20: self.stop_all,  # Space
            0x4D: lambda: self.set_mode("auto" if self.mode == "manual" else "manual"),  # M
            0x4B: self.kidnap,  # K
            0x47: self.global_localization,  # G
            0x48: self.cycle_heatmap,  # H
        }
        if code in actions and now - self._gz_last_cmd.get(code, 0.0) > 0.4:  # ignore auto-repeat
            self._gz_last_cmd[code] = now
            actions[code]()

    def active_keys(self):
        now = time.monotonic()
        return set(self.held) | {d for d, t in list(self.gz_until.items()) if t > now}

    def sim_time(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------ actions
    def note(self, text):
        self.log.appendleft(f"[{self.sim_time():7.1f}s] {text}")
        self.get_logger().info(text)

    def set_mode(self, mode):
        if mode != self.mode:
            self.mode = mode
            self.mode_pub.publish(String(data=mode))
            self.note(f"drive mode: {mode}")
            if mode == "manual":
                self.cmd_pub.publish(Twist())

    def drive(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.cmd_pub.publish(t)

    def stop_all(self):
        self.held.clear()
        self.gz_until.clear()
        self.v = self.w = 0.0
        if self.mode == "manual":
            self.drive(0, 0)

    def teleop_tick(self):
        """20 Hz /cmd_vel from the held keys (runs in the ROS executor thread,
        so driving stays smooth even while the GUI is busy redrawing)."""
        if self.mode != "manual":
            self.v = self.w = 0.0
            return
        held = self.active_keys()
        boost = 2.0 if "boost" in held else 1.0
        tv = boost * self.v_max * (("fwd" in held) - ("back" in held))
        tw = boost * self.w_max * (("left" in held) - ("right" in held))
        # Ramp up gently (no wheel slip) but stop quickly once keys are released.
        dv = 0.12 if abs(tv) < abs(self.v) else 0.05
        dw = 0.6 if abs(tw) < abs(self.w) else 0.35
        self.v += float(np.clip(tv - self.v, -dv, dv))
        self.w += float(np.clip(tw - self.w, -dw, dw))
        self.drive(self.v, self.w)

    def global_localization(self):
        if not self.global_cli.service_is_ready():
            self.note("global localisation service not available")
            return
        self.global_cli.call_async(Empty.Request())
        self.note("global localisation: particles spread over the building")

    def kidnap(self):
        if not self.teleport_cli.service_is_ready():
            self.note("teleport service /gazebo/set_entity_state not available")
            return
        rng = np.random.default_rng()
        x, y = self.fp.sample_free(1, rng, clearance=0.6)[0]
        yaw = float(rng.uniform(-math.pi, math.pi))
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = "wiloc_bot"
        req.state.reference_frame = "world"
        req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = float(x), float(y), 0.02
        req.state.pose.orientation = yaw_to_quat(yaw)
        self.set_mode("manual")
        self.teleport_cli.call_async(req)
        self.note(f"KIDNAP: robot teleported to ({x:.1f}, {y:.1f}); odometry did not see it")

    def reset_at_truth(self):
        if self.truth is None:
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(self.truth[0]), float(self.truth[1])
        msg.pose.pose.orientation = yaw_to_quat(float(self.truth[2]))
        self.init_pub.publish(msg)
        self.note("filter reset at the true pose")

    def cycle_heatmap(self):
        n = len(self.fp.access_points)
        self.heat_choice = -1 if self.heat_choice >= n - 1 else self.heat_choice + 1
        self.heat_pub.publish(Int32(data=self.heat_choice))
        self.note(f"heat map: {self.heat_name()}")

    def heat_name(self):
        return "strongest AP" if self.heat_choice < 0 else self.fp.access_points[self.heat_choice].id

    def alive_nodes(self):
        return set(self.get_node_names())


# =============================================================================
class Dashboard(QtWidgets.QMainWindow):
    DRIVE_KEYS = {
        QtCore.Qt.Key_W: "fwd",
        QtCore.Qt.Key_Up: "fwd",
        QtCore.Qt.Key_S: "back",
        QtCore.Qt.Key_Down: "back",
        QtCore.Qt.Key_A: "left",
        QtCore.Qt.Key_Left: "left",
        QtCore.Qt.Key_D: "right",
        QtCore.Qt.Key_Right: "right",
        QtCore.Qt.Key_Shift: "boost",
    }

    def __init__(self, node: DashboardNode, font_scale: float):
        super().__init__()
        self.node, self.s = node, font_scale
        self._release_token, self._last_cmd_key = {}, {}
        self.n_ticks = 0
        self.alive = set()
        self.setWindowTitle("WiLoc - Wi-Fi RSSI localisation")
        self._style_matplotlib()
        self._prepare_heat()
        self._build()
        self.draw_timer = QtCore.QTimer(self, timeout=self.redraw, interval=250)
        self.draw_timer.start()
        self.input_timer = QtCore.QTimer(self, timeout=self.refresh_input, interval=100)
        self.input_timer.start()
        self.node_check = QtCore.QTimer(self, timeout=self.refresh_nodes, interval=2000)
        self.node_check.start()

    # ------------------------------------------------------------ setup
    def _style_matplotlib(self):
        fs = 8.5 * self.s
        matplotlib.rcParams.update(
            {
                "font.size": fs,
                "axes.titlesize": fs * 1.15,
                "axes.titleweight": "bold",
                "axes.labelsize": fs,
                "legend.fontsize": fs * 0.85,
                "xtick.labelsize": fs * 0.85,
                "ytick.labelsize": fs * 0.85,
                "axes.facecolor": BG,
                "figure.facecolor": BG,
                "axes.edgecolor": GRID,
                "axes.labelcolor": FG,
                "text.color": FG,
                "xtick.color": MUTED,
                "ytick.color": MUTED,
                "grid.color": GRID,
                "legend.facecolor": BG,
                "legend.edgecolor": GRID,
            }
        )

    def _prepare_heat(self, res=0.2):
        fp = self.node.fp
        sim = RadioSimulator(fp, RadioConfig(seed=self.node.radio_seed))
        nx, ny = int(round(fp.width / res)), int(round(fp.height / res))
        gx, gy = np.meshgrid((np.arange(nx) + 0.5) * res, (np.arange(ny) + 0.5) * res)
        self.heat_rssi = sim.mean_rssi(np.column_stack([gx.ravel(), gy.ravel()])).reshape(ny, nx, -1)
        self.heat_shown = None

    def _button(self, text, fn, tip=""):
        b = QtWidgets.QPushButton(text)
        b.setFocusPolicy(QtCore.Qt.NoFocus)
        b.clicked.connect(fn)
        if tip:
            b.setToolTip(tip)
        return b

    def _canvas(self, fig):
        c = FigureCanvasQTAgg(fig)
        c.setFocusPolicy(QtCore.Qt.NoFocus)
        return c

    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        # --- top bar: status (left) and controls (right), always visible
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)
        status_col = QtWidgets.QVBoxLayout()
        self.banner = QtWidgets.QLabel()
        self.banner.setTextFormat(QtCore.Qt.RichText)
        self.banner.setStyleSheet(f"background:{PANEL}; padding:6px 10px; border-radius:8px;")
        status_col.addWidget(self.banner)
        self.cmd_label = QtWidgets.QLabel()
        self.cmd_label.setTextFormat(QtCore.Qt.RichText)
        self.cmd_label.setWordWrap(True)
        status_col.addWidget(self.cmd_label)
        self.log_label = QtWidgets.QLabel()
        self.log_label.setStyleSheet(f"color:{MUTED}; font-family: monospace;")
        status_col.addWidget(self.log_label)
        top.addLayout(status_col, 5)

        pad = QtWidgets.QGridLayout()
        pad.setSpacing(4)
        self.pad_buttons = {}
        for key, text, fn, r, c in (
            ("fwd", "W  forward", lambda: self.tap("fwd"), 0, 1),
            ("left", "A  left", lambda: self.tap("left"), 1, 0),
            ("stop", "Space  stop", self.stop, 1, 1),
            ("right", "D  right", lambda: self.tap("right"), 1, 2),
            ("back", "S  back", lambda: self.tap("back"), 2, 1),
            ("boost", "Shift  2x", lambda: None, 2, 2),
        ):
            b = self._button(text, fn)
            self.pad_buttons[key] = b
            pad.addWidget(b, r, c)
        top.addLayout(pad, 3)
        acts = QtWidgets.QGridLayout()
        acts.setSpacing(4)
        for i, (text, fn, tip) in enumerate(
            (
                ("M  auto / manual", self.toggle_mode, "Switch between the waypoint patrol and keyboard driving"),
                ("K  kidnap robot", self.node.kidnap, "Teleport the robot in Gazebo; odometry does not notice"),
                ("G  global re-localise", self.node.global_localization, "Spread particles over the whole building"),
                ("R  reset at truth", self.node.reset_at_truth, "Re-initialise the filter at the true pose"),
                ("H  cycle heat map", self.node.cycle_heatmap, "Heat map of the strongest AP, or of each AP in turn"),
            )
        ):
            acts.addWidget(self._button(text, fn, tip), i // 2, i % 2)
        top.addLayout(acts, 3)

        # --- tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tabs.setStyleSheet(
            f"QTabBar::tab {{ padding: 7px 18px; background:{PANEL}; border:1px solid {GRID}; border-bottom:none;"
            f" border-top-left-radius:6px; border-top-right-radius:6px; margin-right:3px; }}"
            f"QTabBar::tab:selected {{ background:{BG}; color:{theme.BLUE}; font-weight:700; border-top:3px solid {theme.BLUE}; }}"
            f"QTabWidget::pane {{ border:1px solid {GRID}; }}"
        )
        root.addWidget(self.tabs, 1)

        # Overview: live map + every key analytic in one view (no tab switching while driving)
        self.fig_map = Figure()
        self.ax_map = self.fig_map.add_axes([0.035, 0.2, 0.6, 0.75])
        self.ax_cbar = self.fig_map.add_axes([0.06, 0.155, 0.25, 0.012])
        self.ax_legend = self.fig_map.add_axes([0.0, 0.0, 0.66, 0.11])
        self.ax_legend.axis("off")
        self.ov_rssi = self.fig_map.add_axes([0.71, 0.72, 0.27, 0.23])
        self.ov_err = self.fig_map.add_axes([0.71, 0.42, 0.27, 0.23])
        self.ov_pf = self.fig_map.add_axes([0.71, 0.13, 0.25, 0.22])
        self.ov_pf2 = self.ov_pf.twinx()
        self.ov_health = self.fig_map.add_axes([0.69, 0.0, 0.31, 0.07])
        self.ov_health.axis("off")
        self.canvas_map = self._canvas(self.fig_map)
        self.tabs.addTab(self.canvas_map, "Overview")
        self._setup_map()

        # Wi-Fi
        wifi = QtWidgets.QWidget()
        wl = QtWidgets.QHBoxLayout(wifi)
        self.fig_rssi = Figure(constrained_layout=True)
        self.ax_rssi = self.fig_rssi.add_subplot(111)
        self.canvas_rssi = self._canvas(self.fig_rssi)
        wl.addWidget(self.canvas_rssi, 5)
        tbl_box = QtWidgets.QGroupBox("How the latest scan was formed (dBm / dB)")
        tl = QtWidgets.QVBoxLayout(tbl_box)
        formula = QtWidgets.QLabel(
            "<b>RSSI = P<sub>tx</sub> &minus; PL(d) &minus; &Sigma; walls + shadowing + noise</b><br>"
            f"<span style='color:{MUTED}'>walls crossed: C concrete &middot; B brick &middot; D drywall &middot; G glass"
            " &nbsp;|&nbsp; map = radio-map prediction at the estimated pose</span>"
        )
        formula.setTextFormat(QtCore.Qt.RichText)
        formula.setWordWrap(True)
        tl.addWidget(formula)
        cols = ["AP", "d [m]", "Ptx", "PL(d)", "walls", "wall", "shadow", "noise", "RSSI", "map"]
        self.table = QtWidgets.QTableWidget(len(self.node.fp.access_points), len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setFocusPolicy(QtCore.Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        tl.addWidget(self.table, 1)
        wl.addWidget(tbl_box, 5)
        self.tabs.addTab(wifi, "Wi-Fi details")

        # Localization
        self.fig_loc = Figure(constrained_layout=True)
        self.ax_err = self.fig_loc.add_subplot(121)
        self.ax_pf = self.fig_loc.add_subplot(122)
        self.ax_pf2 = self.ax_pf.twinx()
        self.canvas_loc = self._canvas(self.fig_loc)
        self.tabs.addTab(self.canvas_loc, "Localization details")

        # ROS graph
        self.fig_graph = Figure(constrained_layout=True)
        self.ax_graph = self.fig_graph.add_subplot(111)
        self.canvas_graph = self._canvas(self.fig_graph)
        self.tabs.addTab(self.canvas_graph, "ROS graph")

        self.tabs.currentChanged.connect(lambda _: self.redraw(force=True))
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def _setup_map(self):
        fp, ax = self.node.fp, self.ax_map
        ax.set_aspect("equal")
        ax.set_xlim(-0.4, fp.width + 0.4)
        ax.set_ylim(-0.4, fp.height + 0.4)
        ax.set_xticks(range(0, int(fp.width) + 1, 2))
        ax.set_yticks(range(0, int(fp.height) + 1, 2))
        ax.tick_params(length=0)
        self.heat_img = ax.imshow(
            np.zeros(self.heat_rssi.shape[:2]),
            extent=(0, fp.width, 0, fp.height),
            origin="lower",
            cmap=HEAT_CMAP,
            interpolation="bilinear",
            alpha=0.85,
            zorder=0,
        )
        self.fig_map.colorbar(self.heat_img, cax=self.ax_cbar, orientation="horizontal")
        self.ax_cbar.tick_params(labelsize=matplotlib.rcParams["font.size"] * 0.8)
        segs = [((w[0], w[1]), (w[2], w[3])) for w in fp.walls]
        ax.add_collection(
            LineCollection(
                segs,
                colors=[theme.MATERIALS.get(m, FG) for m in fp.wall_materials],
                linewidths=[WALL_WIDTH.get(m, 2.5) for m in fp.wall_materials],
                capstyle="butt",
                zorder=2,
            )
        )
        self.link_lines = LineCollection([], zorder=3, linewidths=2.0)
        ax.add_collection(self.link_lines)
        fs = matplotlib.rcParams["font.size"]
        self.link_labels = [
            ax.text(0, 0, "", fontsize=fs * 0.72, ha="center", va="center", zorder=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
            for _ in fp.access_points
        ]
        for i, ap in enumerate(fp.access_points):
            ax.scatter([ap.x], [ap.y], s=170, marker="o", color=theme.ap_color(i), edgecolors="white", linewidths=2, zorder=6)
            ax.text(ap.x, ap.y + 0.45, ap.id, color=theme.ap_color(i), fontsize=fs * 0.9, fontweight="bold", ha="center", zorder=7)
        (self.line_odom,) = ax.plot([], [], color=C_ODOM, lw=1.4, ls="--", zorder=4)
        (self.line_truth,) = ax.plot([], [], color=FG, lw=1.6, zorder=4)
        (self.line_est,) = ax.plot([], [], color=theme.BLUE, lw=2.0, zorder=4)
        self.particles_sc = ax.scatter([], [], s=5, color=theme.AMBER, alpha=0.55, zorder=5, linewidths=0)
        self.fix_sc = ax.scatter([], [], s=90, marker="X", color=theme.CORAL, edgecolors="white", linewidths=1, zorder=8)
        self.ellipse = Ellipse((0, 0), 0, 0, fc=theme.BLUE, ec=theme.BLUE, alpha=0.18, lw=1.5, zorder=6)
        ax.add_patch(self.ellipse)
        self.robot_true = Circle((0, 0), 0.25, fc="white", ec=FG, lw=2, zorder=9)
        ax.add_patch(self.robot_true)
        (self.head_true,) = ax.plot([], [], color=FG, lw=2.5, zorder=10)
        (self.est_dot,) = ax.plot([], [], "o", color=theme.BLUE, ms=9, mec="white", mew=1.5, zorder=10)
        (self.head_est,) = ax.plot([], [], color=theme.BLUE, lw=2.5, zorder=10)

        handles = [
            Line2D([], [], color=FG, lw=1.6, marker="o", mfc="white", ms=9, label="true robot & path"),
            Line2D([], [], color=theme.BLUE, lw=2.0, marker="o", ms=8, label="WiLoc estimate & path"),
            Patch(fc=theme.BLUE, alpha=0.2, label="estimate 2-sigma"),
            Line2D([], [], color=theme.AMBER, marker="o", ls="none", ms=4, label="particles"),
            Line2D([], [], color=theme.CORAL, marker="X", ls="none", ms=8, label="raw Wi-Fi fix (WKNN)"),
            Line2D([], [], color=C_ODOM, lw=1.4, ls="--", label="odometry only"),
            Line2D([], [], color=theme.heat_rgb(0.9), lw=2.0, label="robot-AP link (RSSI colour)"),
        ]
        handles += [
            Line2D([], [], color=theme.MATERIALS[m], lw=WALL_WIDTH[m], label=f"{m} wall")
            for m in ("concrete", "brick", "drywall", "glass")
        ]
        self.ax_legend.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=fs * 0.75,
                              handlelength=1.8, columnspacing=1.2, labelspacing=0.5)
        self.map_title = ax.set_title("")

    # ------------------------------------------------------------ input
    def refresh_input(self):
        """Light up held keys; show the live command and keyboard focus."""
        held, n = self.node.active_keys(), self.node
        for key, b in self.pad_buttons.items():
            on = key in held
            b.setStyleSheet(f"background:{theme.BLUE}; color:white; border-color:{theme.BLUE};" if on else "")
        focus = (
            f"<span style='color:{theme.GREEN}; font-weight:700'>keys active here</span>"
            if self.isActiveWindow()
            else f"<span style='color:{MUTED}; font-weight:700'>keys: click this window, or the Gazebo 3D view</span>"
        )
        self.cmd_label.setText(
            f"{focus} &nbsp;|&nbsp; command <b>v = {n.v:+.2f} m/s</b>, <b>&omega; = {n.w:+.2f} rad/s</b>"
            f"<br><span style='color:{MUTED}'>Drive here or in Gazebo (click its 3D view first; arrow keys work best "
            f"there because Gazebo uses S/R/T as tool shortcuts). Hold W/S with A/D to curve.</span>"
        )

    def refresh_nodes(self):
        self.alive = self.node.alive_nodes()

    def keyPressEvent(self, e):
        k = e.key()
        if k in self.DRIVE_KEYS:
            d = self.DRIVE_KEYS[k]
            self._release_token[d] = self._release_token.get(d, 0) + 1  # cancel a pending release
            if not e.isAutoRepeat() and d != "boost":
                self.node.set_mode("manual")
            self.node.held.add(d)
            return
        if e.isAutoRepeat():
            return
        # X11 under WSLg may deliver auto-repeat as plain press/release pairs:
        # ignore repeats of the same command key within 0.4 s.
        now = time.monotonic()
        if now - self._last_cmd_key.get(k, 0.0) < 0.4:
            self._last_cmd_key[k] = now
            return
        self._last_cmd_key[k] = now
        actions = {
            QtCore.Qt.Key_Space: self.stop,
            QtCore.Qt.Key_M: self.toggle_mode,
            QtCore.Qt.Key_G: self.node.global_localization,
            QtCore.Qt.Key_K: self.node.kidnap,
            QtCore.Qt.Key_R: self.node.reset_at_truth,
            QtCore.Qt.Key_H: self.node.cycle_heatmap,
        }
        if k in actions:
            actions[k]()

    def keyReleaseEvent(self, e):
        k = e.key()
        if k not in self.DRIVE_KEYS or e.isAutoRepeat():
            return
        d = self.DRIVE_KEYS[k]
        token = self._release_token.get(d, 0)
        # Only a real release if no new press follows within 50 ms (some X
        # servers send auto-repeat as release+press pairs).
        QtCore.QTimer.singleShot(50, lambda: self._finish_release(d, token))

    def _finish_release(self, d, token):
        if self._release_token.get(d, 0) == token:
            self.node.held.discard(d)

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        self.node.held.clear()  # never keep driving after the window loses focus

    def tap(self, direction):
        """Button press: drive in that direction for 0.6 s."""
        self.node.set_mode("manual")
        self.node.held.add(direction)
        QtCore.QTimer.singleShot(600, lambda: self.node.held.discard(direction))

    def stop(self):
        self.node.stop_all()

    def toggle_mode(self):
        self.node.held.clear()
        self.node.set_mode("auto" if self.node.mode == "manual" else "manual")

    # ------------------------------------------------------------ drawing
    def redraw(self, force=False):
        """Banner every tick; only the visible tab is redrawn (map at 4 Hz, others at 1 Hz)."""
        self.n_ticks += 1
        self._update_banner()
        tab = self.tabs.currentIndex()
        if tab == 0:
            self._draw_map()
            if force or self.n_ticks % 4 == 1:
                self._draw_rssi(self.ov_rssi, compact=True)
                self._draw_errors(self.ov_err, compact=True)
                self._draw_internals(self.ov_pf, self.ov_pf2, compact=True)
                self._draw_health()
            self.canvas_map.draw_idle()
        elif force or self.n_ticks % 4 == 0:
            if tab == 1:
                self._update_table()
                self._draw_rssi()
                self.canvas_rssi.draw_idle()
            elif tab == 2:
                self._draw_errors()
                self._draw_internals()
                self.canvas_loc.draw_idle()
            elif tab == 3:
                self._draw_graph()
                self.canvas_graph.draw_idle()

    def _update_banner(self):
        n = self.node
        st = n.status or {}
        err = n.errors[-1] if n.errors else (0, np.nan, np.nan, np.nan)
        mode_color = theme.TEAL if n.mode == "auto" else theme.AMBER
        method = {"pf": "Particle filter", "ekf": "EKF", "wknn": "WKNN (no filter)"}.get(st.get("method"), "-")
        rm = n.rmse.get("filter")
        self.banner.setText(
            f"<span style='font-size:{int(15 * self.s)}pt; font-weight:800; color:{FG}'>WiLoc</span>"
            f"&nbsp;&nbsp;<span style='color:{mode_color}; font-weight:700'>{n.mode.upper()}</span>"
            f"&nbsp;&nbsp;<span style='color:{MUTED}'>{n.driver_status}</span>"
            f"&nbsp;&nbsp;&nbsp;<span style='color:{FG}'>{method}</span> &nbsp; t = {n.sim_time():.0f} s<br>"
            f"error now <b style='color:{C_FILTER}'>{err[1]:.2f} m</b> &nbsp; RMSE <b>{'-' if rm is None else f'{rm:.2f} m'}</b>"
            f" &nbsp;&nbsp; Wi-Fi fix <b style='color:{theme.CORAL}'>{err[2]:.2f} m</b>"
            f" &nbsp; odometry <b style='color:{C_ODOM}'>{err[3]:.2f} m</b>"
        )
        self.log_label.setText(" | ".join(list(n.log)[:2]) if n.log else "Press K to kidnap the robot, G to forget its pose, H to change the heat map.")

    def _draw_map(self):
        n = self.node
        # heat map (only when the selection changes)
        if self.heat_shown != n.heat_choice:
            vals = self.heat_rssi.max(axis=2) if n.heat_choice < 0 else self.heat_rssi[:, :, n.heat_choice]
            lo, hi = np.percentile(vals, [3, 99])
            self.heat_img.set_data(vals)
            self.heat_img.set_clim(5 * np.floor(lo / 5), 5 * np.ceil(hi / 5))
            self.ax_cbar.set_xlabel(f"RSSI [dBm], heat map: {n.heat_name()}", fontsize=matplotlib.rcParams["font.size"] * 0.85)
            self.heat_shown = n.heat_choice
        for key, line in (("truth", self.line_truth), ("est", self.line_est), ("odom", self.line_odom)):
            line.set_data(*n.tracks[key].xy())
        p = n.particles
        self.particles_sc.set_offsets(p if len(p) else np.zeros((0, 2)))
        self.fix_sc.set_offsets([n.fix] if n.fix is not None else np.zeros((0, 2)))

        if n.truth is not None:
            x, y, th = n.truth
            self.robot_true.center = (x, y)
            self.head_true.set_data([x, x + 0.45 * math.cos(th)], [y, y + 0.45 * math.sin(th)])
        if n.est is not None:
            x, y, th = n.est
            self.est_dot.set_data([x], [y])
            self.head_est.set_data([x, x + 0.6 * math.cos(th)], [y, y + 0.6 * math.sin(th)])
            if n.est_cov is not None:
                vals, vecs = np.linalg.eigh(n.est_cov + 1e-6 * np.eye(2))
                vals = np.maximum(vals, 1e-6)
                self.ellipse.set_center((x, y))
                self.ellipse.width, self.ellipse.height = 4 * math.sqrt(vals[1]), 4 * math.sqrt(vals[0])
                self.ellipse.angle = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))

        d = n.scan_debug
        if d:
            rx, ry = d["pose"]
            segs, cols = [], []
            for i, (ap, row) in enumerate(zip(n.fp.access_points, d["aps"])):
                m = row["measured_dbm"]
                segs.append(((rx, ry), (ap.x, ap.y)))
                cols.append((*rssi_rgb(m), 0.9) if m is not None else (0.6, 0.6, 0.6, 0.4))
                f = 0.6
                lbl = self.link_labels[i]
                lbl.set_position((rx + f * (ap.x - rx), ry + f * (ap.y - ry)))
                walls = len(row["walls"])
                lbl.set_text(f"{m:.0f} dBm" + (f" | {walls}w" if walls else "") if m is not None else "not heard")
                lbl.set_color(rssi_rgb(m) if m is not None else MUTED)
            self.link_lines.set_segments(segs)
            self.link_lines.set_color(cols)
        st = n.status or {}
        extra = f"  |  N_eff {st['ess']:.0f}/{st['n_particles']}" if "ess" in st else ""
        self.map_title.set_text(f"Live map  |  heat map: {n.heat_name()}  |  spread {st.get('spread_m', float('nan')):.2f} m{extra}")

    def _draw_health(self):
        """One-line ROS health summary for the Overview tab (full graph in its own tab)."""
        ax, r = self.ov_health, self.node.rates
        ax.clear()
        ax.axis("off")
        alive = len(self.alive & set(GRAPH_NODES))
        parts = [f"nodes {alive}/{len(GRAPH_NODES)} up"] + [
            f"{t} {r[t].hz():.0f} Hz" for t in ("/wifi/scan", "/wheel_odom", "/wiloc/pose", "/cmd_vel")
        ]
        ok = alive == len(GRAPH_NODES) and r["/wifi/scan"].hz() > 0.3
        text = "ROS  " + "   ".join(parts[:3]) + "\n        " + "   ".join(parts[3:])
        ax.text(0.0, 0.5, text, va="center", fontsize=matplotlib.rcParams["font.size"] * 0.75,
                color=theme.GREEN if ok else theme.CORAL, fontweight="bold")

    def _update_table(self):
        d, st = self.node.scan_debug, self.node.status or {}
        if not d:
            return
        pred = dict(zip(st.get("ap_ids", []), st.get("map_rssi_at_estimate", [])))
        for i, ap in enumerate(d["aps"]):
            walls = " ".join(m[0].upper() for m in ap["walls"]) if ap["walls"] else "LOS"
            meas = ap["measured_dbm"]
            vals = [
                ap["id"],
                f"{ap['distance_m']:.1f}",
                f"{ap['tx_power_dbm']:.0f}",
                f"-{ap['path_loss_db']:.0f}",
                walls,
                f"-{ap['wall_loss_db']:.1f}" if ap["wall_loss_db"] else "0",
                f"{ap['shadowing_db']:+.1f}",
                f"{ap['noise_db']:+.1f}" if ap["noise_db"] is not None else "-",
                f"{meas:.0f}" if meas is not None else "not heard",
                f"{pred[ap['id']]:.0f}" if ap["id"] in pred else "-",
            ]
            for j, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if j == 0 or (j == 8 and meas is not None):
                    color = QtGui.QColor(theme.ap_color(i)) if j == 0 else QtGui.QColor.fromRgbF(*rssi_rgb(meas))
                    item.setForeground(color)
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self.table.setItem(i, j, item)

    def _draw_rssi(self, ax=None, compact=False):
        ax = ax or self.ax_rssi
        d, st = self.node.scan_debug, self.node.status or {}
        ax.clear()
        ax.set_title("Latest Wi-Fi scan vs. radio map")
        if not d:
            ax.text(0.5, 0.5, "waiting for /wifi/scan ...", ha="center", transform=ax.transAxes, color=MUTED)
            return
        ids = [a["id"] for a in d["aps"]]
        x = np.arange(len(ids))
        meas = np.array([a["measured_dbm"] if a["measured_dbm"] is not None else np.nan for a in d["aps"]])
        floor = -100.0
        ap_cols = [theme.ap_color(i) for i in range(len(ids))]
        ax.bar(x - 0.2, np.nan_to_num(meas, nan=floor) - floor, 0.38, bottom=floor, color=ap_cols, label="measured (this scan)")
        ax.scatter(x - 0.2, [a["mean_rssi_dbm"] for a in d["aps"]], marker="_", s=400, color=FG, lw=2, label="true mean (no noise)", zorder=3)
        pred = dict(zip(st.get("ap_ids", []), zip(st.get("map_rssi_at_estimate", []), st.get("map_std_at_estimate", []))))
        if pred:
            mu = np.array([pred[i][0] for i in ids])
            sd = np.array([pred[i][1] for i in ids])
            ax.bar(x + 0.2, mu - floor, 0.38, bottom=floor, color=ap_cols, alpha=0.35, hatch="///", edgecolor=ap_cols, label="radio map @ estimate")
            ax.errorbar(x + 0.2, mu, yerr=sd, fmt="none", ecolor=FG, capsize=3, lw=1)
        ax.axhline(d["sensitivity_dbm"], color=MUTED, ls=":", lw=1, label="receiver sensitivity")
        ax.set_xticks(x, ids)
        for lbl, c in zip(ax.get_xticklabels(), ap_cols):
            lbl.set_color(c)
            lbl.set_fontweight("bold")
        ax.set_ylim(floor, 5 if not compact else 0)
        ax.set_ylabel("RSSI [dBm]")
        ax.grid(axis="y", alpha=0.5)
        if compact:
            ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=matplotlib.rcParams["font.size"] * 0.65)
        else:
            ax.legend(loc="upper center", ncol=2, frameon=False)

    def _draw_errors(self, ax=None, compact=False):
        ax, e = ax or self.ax_err, self.node.errors
        ax.clear()
        ax.set_title("Position error vs. ground truth")
        if len(e) < 2:
            ax.text(0.5, 0.5, "waiting for /wiloc/errors ...", ha="center", transform=ax.transAxes, color=MUTED)
            return
        a = np.array(list(e))
        a = a[a[:, 0] > a[-1, 0] - 180]
        t = a[:, 0] - a[-1, 0]
        ax.plot(t, a[:, 3], color=C_ODOM, lw=1.4, ls="--", label="odometry only")
        ax.plot(t, a[:, 2], color=theme.CORAL, lw=0.8, alpha=0.55, label="Wi-Fi fix only (WKNN)")
        ax.plot(t, a[:, 1], color=C_FILTER, lw=2.4, label="WiLoc filter")
        ax.set_xlabel("time [s] (last 3 min)")
        ax.set_ylabel("error [m]")
        top = np.nanpercentile(a[:, 1:], 98) if np.isfinite(a[:, 1:]).any() else 3
        ax.set_ylim(0, max(2.0, min(12.0, top * 1.15)))
        ax.grid(alpha=0.5)
        ax.legend(loc="upper left", fontsize=matplotlib.rcParams["font.size"] * (0.65 if compact else 0.85))

    def _draw_internals(self, ax=None, ax2=None, compact=False):
        ax, ax2, st = ax or self.ax_pf, ax2 or self.ax_pf2, self.node.status or {}
        ax.clear()
        ax2.clear()
        ax.set_title("Particle filter internals")
        if st.get("method") != "pf" or len(self.node.internals) < 2:
            c = st.get("counts", {})
            ax2.set_visible(False)
            ax.text(0.5, 0.55, f"method: {st.get('method', '-')}\nWi-Fi updates: {c.get('updates', 0)}\nspread: {st.get('spread_m', float('nan')):.2f} m",
                    ha="center", va="center", transform=ax.transAxes, color=FG)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        ax2.set_visible(True)
        a = np.array(list(self.node.internals))
        a = a[a[:, 0] > a[-1, 0] - 180]
        t = a[:, 0] - a[-1, 0]
        ax.plot(t, a[:, 1], color=theme.PURPLE, lw=1.6, label="N_eff / N (particle diversity)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("N_eff / N", color=theme.PURPLE)
        ax.set_xlabel("time [s]")
        ax2.plot(t, a[:, 2], color=C_FILTER, lw=1.8, label="spread sigma [m]")
        ax2.set_ylabel("spread sigma [m]", color=C_FILTER)
        ax2.set_ylim(0, max(1.0, np.nanpercentile(a[:, 2], 99) * 1.15))
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        c = st["counts"]
        if compact:
            ax.legend(*[a + b for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())],
                      loc="lower left", fontsize=matplotlib.rcParams["font.size"] * 0.65)
            ax.grid(alpha=0.5)
            return
        ax.text(0.02, 0.03,
                f"N = {st['n_particles']}   Wi-Fi updates {c['updates']}   skipped while standing {c['skipped_stationary']}\n"
                f"resamples {st['resamples']}   injected particles {st['injected']}   w_fast/w_slow {st['w_ratio']:.2f}",
                transform=ax.transAxes, color=MUTED, fontsize=matplotlib.rcParams["font.size"] * 0.8)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")
        ax.grid(alpha=0.5)

    def _draw_graph(self):
        ax = self.ax_graph
        ax.clear()
        ax.set_title("ROS 2 node graph (live message rates)")
        ax.set_xlim(-0.1, 12.6)
        ax.set_ylim(-0.2, 6.2)
        ax.axis("off")
        fs = matplotlib.rcParams["font.size"]
        alive = self.alive
        for src, dst, topic in GRAPH_EDGES:
            hz = self.node.rates[topic].hz()
            if topic == "/cmd_vel":  # one topic, two possible publishers
                active_pub = "wiloc_dashboard" if self.node.mode == "manual" else "waypoint_driver"
                hz = hz if src == active_pub else 0.0
                col = (theme.AMBER if src == "wiloc_dashboard" else OK) if hz > 0.3 else GRID
            else:
                col = OK if hz > 0.3 else BAD
            (x1, y1), (x2, y2) = GRAPH_NODES[src][1:], GRAPH_NODES[dst][1:]
            a, b = _box_exit(x1, y1, x2, y2), _box_exit(x2, y2, x1, y1)
            ax.annotate("", xy=b, xytext=a,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4 + min(hz, 30) / 12, shrinkA=0, shrinkB=0, mutation_scale=16))
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            dx, dy = b[0] - a[0], b[1] - a[1]
            if abs(dx) < 1e-6:  # vertical edge: label to the right
                lx, ly, ha = mx + 0.15, my, "left"
            else:
                nrm = np.array([-dy, dx]) / np.hypot(dx, dy)
                nrm = nrm if nrm[1] >= 0 else -nrm
                lx, ly, ha = mx + 0.42 * nrm[0], my + 0.42 * nrm[1], "center"
            ax.text(lx, ly, f"{topic}  {hz:.1f} Hz", ha=ha, va="center", color=col if hz > 0.3 else MUTED, fontsize=fs * 0.85, fontweight="bold")
        for name, (label, x, y) in GRAPH_NODES.items():
            up = name in alive
            ax.add_patch(FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H, boxstyle="round,pad=0.04,rounding_size=0.15",
                                        fc=PANEL if up else BG, ec=theme.BLUE if up else BAD, lw=1.8))
            ax.text(x, y, label, ha="center", va="center", color=FG if up else MUTED, fontsize=fs * 0.9, linespacing=1.1)
        n_alive = len(alive & set(GRAPH_NODES))
        ax.text(12.5, 0.4,
                f"nodes alive  {n_alive}/{len(GRAPH_NODES)}\n/tf  {self.node.rates['/tf'].hz():.0f} Hz\nparticles  {self.node.rates['/wiloc/particles'].hz():.0f} Hz",
                ha="right", va="bottom", color=MUTED, fontsize=fs * 0.85)


def _light_palette(app):
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    for role, color in (
        (QtGui.QPalette.Window, BG),
        (QtGui.QPalette.WindowText, FG),
        (QtGui.QPalette.Base, BG),
        (QtGui.QPalette.AlternateBase, PANEL),
        (QtGui.QPalette.Text, FG),
        (QtGui.QPalette.Button, PANEL),
        (QtGui.QPalette.ButtonText, FG),
        (QtGui.QPalette.Highlight, theme.BLUE),
    ):
        p.setColor(role, QtGui.QColor(color))
    app.setPalette(p)
    app.setStyleSheet(theme.QSS)


def main():
    import os
    import signal
    import threading

    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=sys.argv, signal_handler_options=SignalHandlerOptions.NO)
    node = DashboardNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    # ROS callbacks run in a background thread so message rates stay accurate
    # while matplotlib redraws; the Qt thread only reads the collected data.
    threading.Thread(target=executor.spin, daemon=True).start()

    app = QtWidgets.QApplication(sys.argv)
    _light_palette(app)
    font_scale = max(1.0, float(os.environ.get("QT_FONT_DPI", 96)) / 96.0)
    win = Dashboard(node, font_scale)
    geo = os.environ.get("WILOC_DASHBOARD_GEOMETRY")
    if geo:
        x, y, w, h = (int(v) for v in geo.split(","))
        win.setGeometry(x, y, w, h)
    else:
        win.resize(1500, 1000)
    win.show()
    win.activateWindow()

    # Ctrl-C / SIGINT from ros2 launch: quit the Qt loop cleanly.  Qt blocks
    # Python signal handling, so a no-op timer lets the handler run.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    tick = QtCore.QTimer(timeout=lambda: None, interval=200)
    tick.start()

    snap_dir = os.environ.get("WILOC_SNAPSHOT_DIR")
    if snap_dir:
        # Periodically save a PNG of every tab (documentation / automated checks).
        os.makedirs(snap_dir, exist_ok=True)
        counter = {"i": 0}

        def snapshot():
            counter["i"] += 1
            for t in range(win.tabs.count()):
                win.tabs.setCurrentIndex(t)
                win.redraw(force=True)
                for c in (win.canvas_map, win.canvas_rssi, win.canvas_loc, win.canvas_graph):
                    c.draw()
                app.processEvents()
                win.grab().save(os.path.join(snap_dir, f"tab{t}_{counter['i']:03d}.png"))
            win.tabs.setCurrentIndex(0)

        snap_timer = QtCore.QTimer(win, timeout=snapshot, interval=15000)
        snap_timer.start()

    keys = os.environ.get("WILOC_DASHBOARD_KEYS")
    if keys:
        # Scripted key presses for demos and automated checks, e.g.
        # "20:K,40:W+,43:W-,50:M" (seconds after start : key, + press / - release).
        names = {"SHIFT": QtCore.Qt.Key_Shift, "SPACE": QtCore.Qt.Key_Space, "UP": QtCore.Qt.Key_Up,
                 "DOWN": QtCore.Qt.Key_Down, "LEFT": QtCore.Qt.Key_Left, "RIGHT": QtCore.Qt.Key_Right}
        for item in keys.split(","):
            t, k = item.split(":")
            kind = QtCore.QEvent.KeyRelease if k.endswith("-") else QtCore.QEvent.KeyPress
            k = k.rstrip("+-").upper()
            code = names.get(k, getattr(QtCore.Qt, f"Key_{k}", None))
            QtCore.QTimer.singleShot(
                int(float(t) * 1000),
                lambda kind=kind, code=code: app.postEvent(win, QtGui.QKeyEvent(kind, code, QtCore.Qt.NoModifier)),
            )

    app.exec_()
    if node.mode == "manual":
        node.drive(0, 0)
    executor.shutdown(timeout_sec=1.0)
    node.destroy_node()
    rclpy.try_shutdown()
