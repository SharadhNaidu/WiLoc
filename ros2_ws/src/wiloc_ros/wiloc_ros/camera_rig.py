"""Camera rig for the Studio's 3-D view: an overview of the venue, or following the robot.

Moves the ``wiloc_camera`` model (``world_gen._camera_model``) with
``/gazebo/set_entity_state`` at ``rate`` Hz; position and heading are
low-pass filtered, so switching mode is a smooth fly between the two views.
The commanded pose goes out on ``/wiloc/camera_pose`` for drawing on top of
the image (``projection.py``).

Topics
  sub  /ground_truth        nav_msgs/Odometry
  sub  /wiloc/camera_mode   std_msgs/String  "overview" | "follow" (latched)
  pub  /wiloc/camera_pose   std_msgs/String  JSON {"pose": [x, y, z, pitch, yaw], "hfov", "width", "height", "mode"}
"""

from __future__ import annotations

import json
import math

import numpy as np
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from .common import latched_qos, pose2d_from_odom, run_node, venue_from_params
from .world_gen import CAMERA_HFOV, CAMERA_SIZE, overview_pose


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def follow_pose(robot, back: float = 3.8, height: float = 6.2, pitch: float = 1.05):
    """Chase view from above and behind the robot (high enough to see over 2.5 m walls)."""
    x, y, yaw = robot
    return np.array([x - back * math.cos(yaw), y - back * math.sin(yaw), height, pitch, yaw])


class CameraRig(Node):
    def __init__(self):
        super().__init__("camera_rig")
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("smoothing", 0.07)  # per tick: ~0.5 s to settle at 30 Hz
        self.declare_parameter("mode", "overview")
        _, self.fp = venue_from_params(self)
        self.mode = str(self.get_parameter("mode").value)
        self.alpha = float(self.get_parameter("smoothing").value)
        self.robot = None
        self.heading = None  # smoothed robot heading for the chase view
        self.pose = np.array(overview_pose(self.fp), dtype=float)
        self.cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.pub = self.create_publisher(String, "/wiloc/camera_pose", 10)
        self.create_subscription(Odometry, "/ground_truth", self.on_truth, 10)
        self.create_subscription(String, "/wiloc/camera_mode", self.on_mode, latched_qos())
        self.create_timer(1.0 / float(self.get_parameter("rate").value), self.tick)
        self.pending = None

    def on_truth(self, msg):
        self.robot = np.array(pose2d_from_odom(msg))

    def on_mode(self, msg):
        mode = msg.data.strip().lower()
        if mode in ("overview", "follow") and mode != self.mode:
            self.mode = mode
            self.get_logger().info(f"camera: {mode}")

    def target(self):
        if self.mode == "follow" and self.robot is not None:
            if self.heading is None:
                self.heading = float(self.robot[2])
            self.heading += 0.05 * _wrap(self.robot[2] - self.heading)  # slow: turning in place should not spin the view
            return follow_pose((self.robot[0], self.robot[1], self.heading))
        return np.array(overview_pose(self.fp), dtype=float)

    def tick(self):
        goal = self.target()
        d = goal - self.pose
        d[4] = _wrap(d[4])
        self.pose = self.pose + self.alpha * d
        self.pose[4] = _wrap(self.pose[4])
        moving = np.abs(d[:3]).max() > 1e-3 or abs(d[3]) + abs(d[4]) > 1e-4
        if moving and self.cli.service_is_ready() and (self.pending is None or self.pending.done()):
            x, y, z, pitch, yaw = (float(v) for v in self.pose)
            req = SetEntityState.Request()
            req.state = EntityState()
            req.state.name, req.state.reference_frame = "wiloc_camera", "world"
            req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = x, y, z
            # q = Rz(yaw) * Ry(pitch)
            cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
            q = req.state.pose.orientation
            q.w, q.x, q.y, q.z = cy * cp, -sy * sp, cy * sp, sy * cp
            self.pending = self.cli.call_async(req)
        w, h = CAMERA_SIZE
        self.pub.publish(String(data=json.dumps(
            {"pose": [float(v) for v in self.pose], "hfov": CAMERA_HFOV, "width": w, "height": h, "mode": self.mode}
        )))


def main():
    run_node(CameraRig)
