"""Make Gazebo's wheel odometry behave like a real robot's.

The Gazebo diff-drive plugin integrates joint encoders on a perfect floor, so
its odometry barely drifts.  This node re-integrates each odometry increment
with calibration errors and random-walk noise (rate independent, i.e. noise
grows with sqrt of distance / rotation travelled):

    trans' = trans (1 + trans_bias) + k_trans * sqrt|trans| * N(0,1)
    rot'   = rot   (1 + rot_bias)   + k_rot * sqrt|rot| * N(0,1)
                                    + k_rot_per_m * sqrt|trans| * N(0,1)

It publishes ``/wheel_odom`` and the ``odom -> base_footprint`` transform, so
the TF tree is consistent with the odometry the localiser actually uses.
Set every parameter to 0 to pass odometry through unchanged.
"""

from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from wiloc.geometry import wrap_angle
from wiloc.motion import odom_delta

from .common import pose2d_from_odom, run_node, yaw_to_quat


class OdomNoiseNode(Node):
    def __init__(self):
        super().__init__("odom_noise")
        for name, default in (
            ("trans_bias", 0.015),
            ("rot_bias", 0.04),
            ("k_trans", 0.02),
            ("k_rot", 0.05),
            ("k_rot_per_m", 0.02),
            ("seed", 1),
        ):
            self.declare_parameter(name, default)
        self.p = {n: self.get_parameter(n).value for n in ("trans_bias", "rot_bias", "k_trans", "k_rot", "k_rot_per_m")}
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.prev = None
        self.pose = None
        self.pub = self.create_publisher(Odometry, "/wheel_odom", 20)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)

    def on_odom(self, msg: Odometry):
        cur = pose2d_from_odom(msg)
        if self.prev is None:
            self.prev, self.pose = cur, np.array(cur)
        rot1, trans, rot2 = odom_delta(self.prev, cur)
        self.prev = cur
        p, n = self.p, self.rng.standard_normal(5)
        trans_n = trans * (1 + p["trans_bias"]) + p["k_trans"] * math.sqrt(trans) * n[0]
        rot1_n = rot1 * (1 + p["rot_bias"]) + p["k_rot"] * math.sqrt(abs(rot1)) * n[1] + p["k_rot_per_m"] * math.sqrt(trans) * n[2]
        rot2_n = rot2 * (1 + p["rot_bias"]) + p["k_rot"] * math.sqrt(abs(rot2)) * n[3]
        x, y, th = self.pose
        x += trans_n * math.cos(th + rot1_n)
        y += trans_n * math.sin(th + rot1_n)
        th = float(wrap_angle(th + rot1_n + rot2_n))
        self.pose = np.array([x, y, th])

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "odom"
        out.child_frame_id = "base_footprint"
        out.pose.pose.position.x, out.pose.pose.position.y = float(x), float(y)
        out.pose.pose.orientation = yaw_to_quat(th)
        out.twist = msg.twist
        self.pub.publish(out)

        t = TransformStamped()
        t.header = out.header
        t.child_frame_id = "base_footprint"
        t.transform.translation.x, t.transform.translation.y = float(x), float(y)
        t.transform.rotation = out.pose.pose.orientation
        self.tf.sendTransform(t)


def main():
    run_node(OdomNoiseNode)
