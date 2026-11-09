"""Online evaluation against Gazebo ground truth: the live method race.

Every method the localiser runs (``/wiloc/pose/{pf,ekf,wknn}``) and plain
wheel odometry are scored against the true pose with ``wiloc.race.RaceTracker``:
rolling 30 s median error (the leaderboard, excluding the first 20 s after a
kidnapping), recovery time after a kidnapping, and how often the truth lies
inside each method's own 95 % ellipse.  Memory stays flat however long it runs.

Every sample is appended to a CSV (``output_csv``) for offline plotting
(``python -m wiloc plot-eval``).  The first columns are the classic ones
(``est_*`` = the primary method); per-method columns follow.  A respawned
evaluator keeps appending to the same file.

Topics
  pub  /wiloc/error                std_msgs/Float64  primary method's position error [m]
  pub  /wiloc/wifi_fix_error       std_msgs/Float64
  pub  /wiloc/errors               std_msgs/String   JSON: current errors, leaderboard, recovery, coverage
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

from wiloc.race import RaceTracker

from . import theme
from .common import load_floorplan, pose2d_from_odom, run_node, yaw_to_quat

FILTERS = ("pf", "ekf", "wknn")
RACERS = (*FILTERS, "odometry")
HEADER = ["t", "true_x", "true_y", "est_x", "est_y", "fix_x", "fix_y", "odom_x", "odom_y",
          *(f"{m}_{c}" for m in FILTERS for c in "xy"), "settling"]


def _xy_cov(msg):
    p = msg.pose.pose.position
    c = np.array(msg.pose.covariance).reshape(6, 6)[:2, :2]
    return np.array([p.x, p.y]), c


class Evaluator(Node):
    def __init__(self):
        super().__init__("wiloc_evaluator")
        self.declare_parameter("floorplan", "office")
        self.declare_parameter("method", "pf")
        self.declare_parameter("output_csv", os.path.expanduser("~/.ros/wiloc_eval.csv"))
        self.declare_parameter("report_period", 15.0)
        self.declare_parameter("respawn_append_after", 30.0)
        start = load_floorplan(self.get_parameter("floorplan").value).start_pose
        self.start = np.array(start, dtype=float)
        self.primary = str(self.get_parameter("method").value)
        self.csv_path = self.get_parameter("output_csv").value
        self.race = RaceTracker(RACERS)
        self.truth = None
        self.est = {m: None for m in FILTERS}  # method -> (xy, cov)
        self.fix = self.odom = None
        self.csv_file = self.csv_writer = None
        self.n_rows = 0

        self.err_pub = self.create_publisher(Float64, "/wiloc/error", 10)
        self.fix_err_pub = self.create_publisher(Float64, "/wiloc/wifi_fix_error", 10)
        self.errors_pub = self.create_publisher(String, "/wiloc/errors", 10)
        self.truth_path_pub = self.create_publisher(Path, "/wiloc/ground_truth_path", 5)
        self.odom_path_pub = self.create_publisher(Path, "/wiloc/odom_path", 5)
        self.truth_marker_pub = self.create_publisher(Marker, "/wiloc/truth_marker", 5)
        self.truth_path, self.odom_path = Path(), Path()
        self.truth_path.header.frame_id = self.odom_path.header.frame_id = "map"
        self.create_subscription(Odometry, "/ground_truth", self.on_truth, 20)
        self.create_subscription(PoseWithCovarianceStamped, "/wiloc/pose", self.on_primary, 10)
        for m in FILTERS:
            self.create_subscription(PoseWithCovarianceStamped, f"/wiloc/pose/{m}", lambda msg, m=m: self.on_method(m, msg), 10)
        self.create_subscription(PoseWithCovarianceStamped, "/wiloc/wifi_fix", self.on_fix, 10)
        self.create_subscription(Odometry, "/wheel_odom", self.on_odom, 20)
        self.create_timer(float(self.get_parameter("report_period").value), self.report)
        self.create_timer(0.5, self.publish_paths)

    # ------------------------------------------------------------ inputs
    def on_truth(self, msg):
        self.truth = np.array(pose2d_from_odom(msg))

    def on_odom(self, msg):
        # Express dead reckoning in the map frame by anchoring it at the start pose.
        x, y, th = pose2d_from_odom(msg)
        c, s = np.cos(self.start[2]), np.sin(self.start[2])
        self.odom = self.start[:2] + np.array([c * x - s * y, s * x + c * y])

    def on_fix(self, msg):
        self.fix = _xy_cov(msg)[0]
        if self.truth is not None:
            self.fix_err_pub.publish(Float64(data=float(np.linalg.norm(self.fix - self.truth[:2]))))

    def on_method(self, method, msg):
        self.est[method] = _xy_cov(msg)

    def on_primary(self, msg):
        """One evaluation step per primary pose (10 Hz)."""
        if self.truth is None:
            return
        self.est[self.primary] = _xy_cov(msg)
        t = self.get_clock().now().nanoseconds * 1e-9
        estimates = {m: v for m, v in self.est.items() if v is not None}
        if self.odom is not None:
            estimates["odometry"] = (self.odom, None)
        err = self.race.add(t, self.truth[:2], estimates)
        self.err_pub.publish(Float64(data=err[self.primary]))
        self._write_row(t)

        current = {"filter": err[self.primary], **err}
        if self.fix is not None:
            current["wifi_fix"] = float(np.linalg.norm(self.fix - self.truth[:2]))
        out = {"t": t, "primary": self.primary, "current": current, "samples": len(self.race.samples)}
        if self.n_rows % 10 == 0:  # the statistics cost more than one sample: 1 Hz is plenty
            out.update(self.race.summary(t))
            legacy = {"filter": self.primary, "wifi_fix": "wknn", "odometry": "odometry"}
            stats = {k: self.race.overall(m) for k, m in legacy.items()}
            out["rmse"] = {k: v["rmse"] for k, v in stats.items() if v}
            self.csv_file.flush()  # a crash loses at most a second of the log
        self.errors_pub.publish(String(data=json.dumps(out)))

    # ------------------------------------------------------------ csv
    def _open_csv(self, t):
        """Fresh file at start-up; a respawned evaluator appends to the existing one."""
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        append = False
        if t > float(self.get_parameter("respawn_append_after").value) and os.path.exists(self.csv_path):
            with open(self.csv_path, newline="") as f:
                append = next(csv.reader(f), None) == HEADER
        self.csv_file = open(self.csv_path, "a" if append else "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not append:
            self.csv_writer.writerow(HEADER)
        self.get_logger().info(f"{'appending to' if append else 'writing'} {self.csv_path}")

    def _write_row(self, t):
        if self.csv_writer is None:
            self._open_csv(t)
        nan2 = [np.nan, np.nan]
        xy = lambda m: list(self.est[m][0]) if self.est[m] is not None else nan2
        row = [t, *self.truth[:2], *xy(self.primary), *(self.fix if self.fix is not None else nan2),
               *(self.odom if self.odom is not None else nan2), *(v for m in FILTERS for v in xy(m)),
               int(self.race.settling(t))]
        self.csv_writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in map(_num, row)])
        self.n_rows += 1

    # ------------------------------------------------------------ outputs
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

    def report(self):
        if len(self.race.samples) < 5:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        s = self.race.summary(t)
        med = " | ".join(f"{m} {e:.2f} m" for m, e in sorted(s["median"].items(), key=lambda kv: kv[1]))
        rec = ", ".join(f"{m} {v:.1f} s" for m, v in s["recovery_s"].items() if v is not None)
        cov = ", ".join(f"{m} {v * 100:.0f}%" for m, v in s["coverage95"].items())
        self.get_logger().info(
            f"[{self.n_rows} samples] median 30 s: {med or '-'}"
            + (f" | recovered after kidnap {s['kidnaps']}: {rec}" if s["kidnaps"] else "")
            + (f" | inside 95% ellipse: {cov}" if cov else "")
        )
        if self.csv_file:
            self.csv_file.flush()

    def close(self):
        self.report()
        if self.csv_file:
            self.csv_file.close()
            self.get_logger().info(f"wrote {self.n_rows} rows to {self.csv_path}")


def _num(v):
    return float(v) if isinstance(v, (np.floating, np.integer)) and not isinstance(v, np.bool_) else v


def main():
    run_node(Evaluator, on_shutdown=lambda n: n.close())
