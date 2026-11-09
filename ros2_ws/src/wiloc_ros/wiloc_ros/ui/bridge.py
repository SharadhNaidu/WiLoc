"""ROS side of WiLoc Studio: one node whose callbacks fill a lock-guarded store.

The node spins in a background thread; the Qt thread never touches ROS
objects except to publish (thread-safe in rclpy) and reads the store through
``snapshot()``, which copies.  The Studio node does *not* use sim time: its
timers and charts run on the wall clock, so a venue relaunch (sim time back to
0) cannot confuse them.  Sim time is still tracked (``/clock``) for drawing
people where Gazebo has them.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import time

import numpy as np
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty

from wiloc.environment import FloorPlan
from wiloc.planning import OccupancyGrid, kidnap_target

from ..common import AP_DISABLED_TOPIC, AP_LAYOUT_TOPIC, latched_qos, quat_to_yaw, yaw_to_quat

METHODS = ("pf", "ekf", "wknn")


class Store:
    """Latest value of everything the window shows (written by ROS callbacks)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clear()

    def clear(self):
        self.truth = None  # (x, y, yaw)
        self.sim_t = None  # latest /clock (s)
        self.clock_wall = None  # wall time of the latest /clock
        self.status = None  # /wiloc/status JSON
        self.errors = None  # /wiloc/errors JSON (evaluator summary)
        self.particles = np.zeros((0, 3))
        self.poses = {}  # method -> (x, y, yaw)
        self.plan = np.zeros((0, 2))
        self.scan = None  # /wifi/scan_debug JSON
        self.driver = ""
        self.image = None  # (h, w, bytes)
        self.image_count = 0
        self.ap_enabled = None
        self.camera = None  # /wiloc/camera_pose JSON: where the 3-D view's camera is


class Bridge(Node):
    def __init__(self, venue: FloorPlan):
        super().__init__("wiloc_studio")
        self.store = Store()
        self.set_venue(venue)
        s = self.store

        def put(attr, value):
            with s.lock:
                setattr(s, attr, value)

        self.create_subscription(Odometry, "/ground_truth", self._on_truth, 10)
        self.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data)
        self.create_subscription(String, "/wiloc/status", lambda m: put("status", json.loads(m.data)), 10)
        self.create_subscription(String, "/wiloc/errors", self._on_errors, 10)
        self.create_subscription(String, "/wifi/scan_debug", lambda m: put("scan", json.loads(m.data)), 10)
        self.create_subscription(String, "/wiloc/driver_status", lambda m: put("driver", m.data), 10)
        self.create_subscription(PoseArray, "/wiloc/particles", self._on_particles, 2)
        self.create_subscription(Path, "/wiloc/plan", self._on_plan, 5)
        for m in METHODS:
            self.create_subscription(PoseWithCovarianceStamped, f"/wiloc/pose/{m}", lambda msg, m=m: self._on_pose(m, msg), 10)
        self.create_subscription(Image, "/wiloc/camera/image_raw", self._on_image, qos_profile_sensor_data)
        self.create_subscription(String, "/wiloc/camera_pose", lambda m: put("camera", json.loads(m.data)), 5)

        self.goal_pub = self.create_publisher(PoseStamped, "/wiloc/goal", 10)
        self.mode_pub = self.create_publisher(String, "/wiloc/drive_mode", 10)
        self.layout_pub = self.create_publisher(String, AP_LAYOUT_TOPIC, latched_qos())
        self.body_pub = self.create_publisher(Bool, "/wiloc/body_shadowing", latched_qos())
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.camera_mode_pub = self.create_publisher(String, "/wiloc/camera_mode", latched_qos())
        self.disabled_pub = self.create_publisher(String, AP_DISABLED_TOPIC, latched_qos())
        self.teleport = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.global_cli = self.create_client(Empty, "/wiloc/global_localization")
        self.rng = np.random.default_rng()
        self.layout_requested = -1e9

    # ------------------------------------------------------------ venue
    def set_venue(self, venue: FloorPlan):
        self.venue = venue
        self.fp = venue
        self.grid = OccupancyGrid.from_floorplan(venue)
        with self.store.lock:
            self.store.clear()

    # ------------------------------------------------------------ inputs
    def _on_truth(self, msg: Odometry):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        with self.store.lock:
            self.store.truth = (p.x, p.y, quat_to_yaw(q))

    def _on_clock(self, msg: Clock):
        with self.store.lock:
            self.store.sim_t = msg.clock.sec + 1e-9 * msg.clock.nanosec
            self.store.clock_wall = time.monotonic()

    def _on_errors(self, msg: String):
        d = json.loads(msg.data)
        if "median" in d:
            with self.store.lock:
                self.store.errors = d

    def _on_particles(self, msg: PoseArray):
        arr = np.array([(p.position.x, p.position.y, quat_to_yaw(p.orientation)) for p in msg.poses]).reshape(-1, 3)
        with self.store.lock:
            self.store.particles = arr

    def _on_plan(self, msg: Path):
        arr = np.array([(p.pose.position.x, p.pose.position.y) for p in msg.poses]).reshape(-1, 2)
        with self.store.lock:
            self.store.plan = arr

    def _on_pose(self, method, msg):
        p = msg.pose.pose
        with self.store.lock:
            self.store.poses[method] = (p.position.x, p.position.y, quat_to_yaw(p.orientation))

    def _on_image(self, msg: Image):
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        data = bytes(msg.data)
        if msg.encoding == "bgr8":
            data = np.frombuffer(data, np.uint8).reshape(msg.height, msg.width, 3)[:, :, ::-1].tobytes()
        with self.store.lock:
            self.store.image = (msg.height, msg.width, data)
            self.store.image_count += 1

    def snapshot(self) -> Store:
        """A copy of the store for one UI frame (big arrays are replaced, never mutated, so a shallow copy is safe)."""
        with self.store.lock:
            snap = copy.copy(self.store)
            snap.poses = dict(self.store.poses)
        snap.lock = None
        return snap

    # ------------------------------------------------------------ actions
    def send_goal(self, x: float, y: float):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = float(x), float(y)
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def set_mode(self, mode: str):
        self.mode_pub.publish(String(data=mode))

    def set_layout(self, name: str):
        self.fp = self.venue.with_layout(name)
        self.layout_requested = time.monotonic()
        self.layout_pub.publish(String(data=name))

    def adopt_layout(self, name: str):
        """The simulation is on another preset (switched by someone else): follow it without publishing."""
        self.fp = self.venue.with_layout(name)

    def set_disabled(self, ids):
        """Unplug these routers (and plug every other one back in)."""
        self.disabled_pub.publish(String(data=json.dumps(sorted(ids))))

    def set_body_shadowing(self, on: bool):
        self.body_pub.publish(Bool(data=bool(on)))

    def set_camera_mode(self, mode: str):
        self.camera_mode_pub.publish(String(data=mode))

    def drive(self, v: float, w: float):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.cmd_pub.publish(t)

    def _teleport(self, x, y, yaw) -> bool:
        if not self.teleport.service_is_ready():
            return False
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name, req.state.reference_frame = "wiloc_bot", "world"
        req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = float(x), float(y), 0.02
        req.state.pose.orientation = yaw_to_quat(float(yaw))
        self.teleport.call_async(req)
        return True

    def kidnap(self):
        """Teleport the robot somewhere far away; odometry does not see it."""
        with self.store.lock:
            truth = self.store.truth
        here = np.asarray(truth[:2]) if truth else np.asarray(self.venue.start_pose[:2])
        x, y = kidnap_target(self.venue, self.grid, here, self.rng)
        yaw = float(self.rng.uniform(-math.pi, math.pi))
        return (float(x), float(y)) if self._teleport(x, y, yaw) else None

    def forget(self):
        """Throw away what the filters know: global localisation."""
        if self.global_cli.service_is_ready():
            self.global_cli.call_async(Empty.Request())
            return True
        return False

    def soft_reset(self):
        """Back to a clean demo state without restarting anything (people keep walking)."""
        x, y, yaw = self.venue.start_pose
        self._teleport(x, y, yaw)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(x), float(y)
        msg.pose.pose.orientation = yaw_to_quat(float(yaw))
        self.init_pub.publish(msg)
        self.set_mode("auto")
        self.set_body_shadowing(True)
        self.set_disabled([])
        self.set_layout(next(iter(self.venue.layouts)))
