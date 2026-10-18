"""Drive the robot around the floor plan's patrol route.

The driver steers on the ground-truth pose: it plays the role of a teleoperator
generating motion so that the localiser can be evaluated on realistic data.
(Closing the loop on ``/wiloc/pose`` instead is possible with
``use_estimate:=true`` -- a good exercise once the filter is tuned.)
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from wiloc.motion import WaypointController

from .common import load_floorplan, pose2d_from_odom, quat_to_yaw, run_node


class WaypointDriver(Node):
    def __init__(self):
        super().__init__("waypoint_driver")
        self.declare_parameter("floorplan", "office")
        self.declare_parameter("laps", 1)
        self.declare_parameter("use_estimate", False)
        self.declare_parameter("v_max", 0.22)
        self.declare_parameter("start_delay", 3.0)
        fp = load_floorplan(self.get_parameter("floorplan").value)
        laps = int(self.get_parameter("laps").value)
        self.goals = list(fp.waypoints) * max(laps, 1)
        self.ctrl = WaypointController(v_max=float(self.get_parameter("v_max").value))
        self.gi, self.pose = 0, None
        self.t0 = self.get_clock().now()
        self.delay = float(self.get_parameter("start_delay").value)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        if self.get_parameter("use_estimate").value:
            self.create_subscription(PoseWithCovarianceStamped, "/wiloc/pose", self.on_estimate, 10)
        else:
            self.create_subscription(Odometry, "/ground_truth", self.on_truth, 10)
        self.create_timer(0.05, self.on_timer)

    def on_truth(self, msg):
        self.pose = np.array(pose2d_from_odom(msg))

    def on_estimate(self, msg):
        p = msg.pose.pose
        self.pose = np.array([p.position.x, p.position.y, quat_to_yaw(p.orientation)])

    def on_timer(self):
        if self.pose is None or (self.get_clock().now() - self.t0).nanoseconds * 1e-9 < self.delay:
            return
        cmd = Twist()
        while self.gi < len(self.goals):
            v, w, reached = self.ctrl.command(self.pose, self.goals[self.gi])
            if not reached:
                cmd.linear.x, cmd.angular.z = v, w
                break
            self.gi += 1
            self.get_logger().info(f"waypoint {self.gi}/{len(self.goals)} reached")
        else:
            if not getattr(self, "_done", False):
                self.get_logger().info("patrol complete")
                self._done = True
        self.pub.publish(cmd)


def main():
    run_node(WaypointDriver, on_shutdown=lambda n: n.pub.publish(Twist()))
