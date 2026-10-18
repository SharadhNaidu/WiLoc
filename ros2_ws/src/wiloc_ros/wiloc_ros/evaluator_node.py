"""Online evaluation against Gazebo ground truth.

Publishes the instantaneous position error of the filter, of the raw Wi-Fi
fix and of dead reckoning, logs a summary every ``report_period`` seconds and
writes a CSV (``output_csv``) on shutdown for offline plotting.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Float64

from wiloc.metrics import summarize

from .common import load_floorplan, pose2d_from_odom, run_node


class Evaluator(Node):
    def __init__(self):
        super().__init__("wiloc_evaluator")
        self.declare_parameter("floorplan", "office")
        self.declare_parameter("output_csv", os.path.expanduser("~/.ros/wiloc_eval.csv"))
        self.declare_parameter("report_period", 15.0)
        start = load_floorplan(self.get_parameter("floorplan").value).start_pose
        self.start = np.array(start, dtype=float)
        self.truth = None
        self.rows = []  # t, true x, y, est x, y, fix x, y, odom x, y
        self.est = self.fix = self.odom = None
        self.err_pub = self.create_publisher(Float64, "/wiloc/error", 10)
        self.fix_err_pub = self.create_publisher(Float64, "/wiloc/wifi_fix_error", 10)
        self.truth_path_pub = self.create_publisher(Path, "/wiloc/ground_truth_path", 5)
        self.truth_path = Path()
        self.truth_path.header.frame_id = "map"
        self.create_subscription(Odometry, "/ground_truth", self.on_truth, 20)
        self.create_subscription(PoseWithCovarianceStamped, "/wiloc/pose", self.on_est, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/wiloc/wifi_fix", self.on_fix, 10)
        self.create_subscription(Odometry, "/wheel_odom", self.on_odom, 20)
        self.create_timer(float(self.get_parameter("report_period").value), self.report)

    def on_truth(self, msg):
        self.truth = np.array(pose2d_from_odom(msg))
        if not self.truth_path.poses or len(self.rows) % 5 == 0:
            ps = PoseStamped()
            ps.header = msg.header
            ps.header.frame_id = "map"
            ps.pose = msg.pose.pose
            self.truth_path.poses.append(ps)
            self.truth_path.poses = self.truth_path.poses[-3000:]
            self.truth_path.header.stamp = msg.header.stamp
            self.truth_path_pub.publish(self.truth_path)

    def on_odom(self, msg):
        # Express dead reckoning in the map frame by anchoring it at the start pose.
        x, y, th = pose2d_from_odom(msg)
        c, s = np.cos(self.start[2]), np.sin(self.start[2])
        self.odom = self.start[:2] + np.array([c * x - s * y, s * x + c * y])

    def on_fix(self, msg):
        self.fix = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        if self.truth is not None:
            self.fix_err_pub.publish(Float64(data=float(np.linalg.norm(self.fix - self.truth[:2]))))

    def on_est(self, msg):
        if self.truth is None:
            return
        self.est = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        e = float(np.linalg.norm(self.est - self.truth[:2]))
        self.err_pub.publish(Float64(data=e))
        t = self.get_clock().now().nanoseconds * 1e-9
        nan2 = [np.nan, np.nan]
        self.rows.append(
            [t, *self.truth[:2], *self.est, *(self.fix if self.fix is not None else nan2), *(self.odom if self.odom is not None else nan2)]
        )

    def _errors(self):
        a = np.array(self.rows)
        tru = a[:, 1:3]
        out = {}
        for name, cols in (("filter", slice(3, 5)), ("wifi_fix", slice(5, 7)), ("odometry", slice(7, 9))):
            e = np.linalg.norm(a[:, cols] - tru, axis=1)
            e = e[np.isfinite(e)]
            if len(e):
                out[name] = summarize(e)
        return out

    def report(self):
        if len(self.rows) < 5:
            return
        msg = " | ".join(f"{k}: mean {v['mean']:.2f} m, rmse {v['rmse']:.2f} m, p90 {v['p90']:.2f} m" for k, v in self._errors().items())
        self.get_logger().info(f"[{len(self.rows)} samples] {msg}")
        self.save(quiet=True)

    def save(self, quiet=False):
        path = self.get_parameter("output_csv").value
        if not self.rows:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "true_x", "true_y", "est_x", "est_y", "fix_x", "fix_y", "odom_x", "odom_y"])
            w.writerows(self.rows)
        if not quiet:
            self.get_logger().info(f"wrote {len(self.rows)} rows to {path}")


def main():
    run_node(Evaluator, on_shutdown=lambda n: (n.report(), n.save()))
