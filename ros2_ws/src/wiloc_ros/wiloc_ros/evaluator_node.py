"""Online evaluation against Gazebo ground truth.

Publishes the instantaneous position error of the filter, of the raw Wi-Fi
fix and of dead reckoning, logs a summary every ``report_period`` seconds and
writes a CSV (``output_csv``) periodically and on shutdown for offline plotting.

Topics
  pub  /wiloc/error                std_msgs/Float64  filter position error [m]
  pub  /wiloc/wifi_fix_error       std_msgs/Float64
  pub  /wiloc/errors               std_msgs/String   JSON: current errors + running RMSE
  pub  /wiloc/ground_truth_path    nav_msgs/Path
  pub  /wiloc/odom_path            nav_msgs/Path     dead reckoning in the map frame
  pub  /wiloc/truth_marker         visualization_msgs/Marker  true robot pose
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Float64, String
from visualization_msgs.msg import Marker

from wiloc.metrics import summarize

from . import theme
from .common import load_floorplan, pose2d_from_odom, run_node, yaw_to_quat


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
        self.errors_pub = self.create_publisher(String, "/wiloc/errors", 10)
        self.truth_path_pub = self.create_publisher(Path, "/wiloc/ground_truth_path", 5)
        self.odom_path_pub = self.create_publisher(Path, "/wiloc/odom_path", 5)
        self.truth_marker_pub = self.create_publisher(Marker, "/wiloc/truth_marker", 5)
        self.truth_path, self.odom_path = Path(), Path()
        self.truth_path.header.frame_id = self.odom_path.header.frame_id = "map"
        self.create_subscription(Odometry, "/ground_truth", self.on_truth, 20)
        self.create_subscription(PoseWithCovarianceStamped, "/wiloc/pose", self.on_est, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/wiloc/wifi_fix", self.on_fix, 10)
        self.create_subscription(Odometry, "/wheel_odom", self.on_odom, 20)
        self.create_timer(float(self.get_parameter("report_period").value), self.report)
        self.create_timer(0.5, self.publish_paths)

    def on_truth(self, msg):
        self.truth = np.array(pose2d_from_odom(msg))
        self.truth_stamp = msg.header.stamp

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
        current = {"filter": e}
        if self.fix is not None:
            current["wifi_fix"] = float(np.linalg.norm(self.fix - self.truth[:2]))
        if self.odom is not None:
            current["odometry"] = float(np.linalg.norm(self.odom - self.truth[:2]))
        rmse = {k: v["rmse"] for k, v in self._errors().items()} if len(self.rows) % 10 == 0 else None
        self.errors_pub.publish(String(data=json.dumps({"t": t, "current": current, "rmse": rmse, "samples": len(self.rows)})))

    def publish_paths(self):
        if self.truth is None:
            return
        stamp = self.get_clock().now().to_msg()
        for path, xy, pub in ((self.truth_path, self.truth[:2], self.truth_path_pub), (self.odom_path, self.odom, self.odom_path_pub)):
            if xy is None:
                continue
            ps = PoseStamped()
            ps.header.stamp, ps.header.frame_id = stamp, "map"
            ps.pose.position.x, ps.pose.position.y = float(xy[0]), float(xy[1])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
            path.poses = path.poses[-2000:]
            path.header.stamp = stamp
            pub.publish(path)

        m = Marker()
        m.header.frame_id, m.header.stamp = "map", stamp
        m.ns, m.id, m.type, m.action = "ground_truth", 0, Marker.ARROW, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(self.truth[0]), float(self.truth[1]), 0.05
        m.pose.orientation = yaw_to_quat(float(self.truth[2]))
        m.scale.x, m.scale.y, m.scale.z = 0.55, 0.12, 0.12
        r, g, b = theme.rgb(theme.TEXT)
        m.color = ColorRGBA(r=r, g=g, b=b, a=0.95)
        self.truth_marker_pub.publish(m)

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
