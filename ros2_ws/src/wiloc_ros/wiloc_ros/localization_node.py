"""Wi-Fi + odometry localisation node (the AMCL of WiLoc).

On start-up the node loads a radio map (``radio_map`` .npz from
``python -m wiloc survey``) or, if none is given, performs the simulated site
survey itself.  It then fuses ``/wheel_odom`` with ``/wifi/scan`` using

    method = "pf"    particle filter (default)
             "ekf"   EKF with WKNN position fixes
             "wknn"  scan-by-scan WKNN, no filtering (baseline)

Publishes
  /wiloc/pose        geometry_msgs/PoseWithCovarianceStamped   (map frame)
  /wiloc/particles   geometry_msgs/PoseArray                   (pf only)
  /wiloc/wifi_fix    geometry_msgs/PoseWithCovarianceStamped   raw WKNN fix
  /wiloc/path        nav_msgs/Path
  TF map -> odom
Subscribes
  /initialpose       re-initialise around a pose set in RViz ("2D Pose Estimate")
"""

from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster

from wiloc.estimators import KNN
from wiloc.filters import ParticleFilter, WifiEKF
from wiloc.fingerprint import DenseRadioMap, RadioMap
from wiloc.motion import OdometryMotionModel, odom_delta
from wiloc.radio import RadioConfig, RadioSimulator
from wiloc_msgs.msg import RssiScan

from .common import load_floorplan, pose2d_from_odom, quat_to_yaw, run_node, yaw_to_quat


class LocalizationNode(Node):
    def __init__(self):
        super().__init__("wiloc_localization")
        params = {
            "floorplan": "office",
            "method": "pf",
            "n_particles": 1500,
            "global_init": False,
            "radio_map": "",
            "radio_seed": 7,
            "survey_spacing": 1.0,
            "survey_samples": 20,
            "map_method": "gp",
            "likelihood_temper": 0.25,
            "alpha1": 0.05,
            "alpha2": 0.02,
            "alpha3": 0.05,
            "alpha4": 0.01,
            "knn_k": 4,
            "publish_rate": 10.0,
        }
        for k, v in params.items():
            self.declare_parameter(k, v)
        gp = lambda n: self.get_parameter(n).value

        self.fp = load_floorplan(gp("floorplan"))
        self.method = gp("method")
        rmap = self._load_or_survey(gp("radio_map"), int(gp("radio_seed")), gp("survey_spacing"), gp("survey_samples"))
        dense = DenseRadioMap.from_radio_map(rmap, self.fp, method=gp("map_method"))
        self.fix_est = KNN(int(gp("knn_k")), weighted=True).fit(rmap)
        mm = OdometryMotionModel(gp("alpha1"), gp("alpha2"), gp("alpha3"), gp("alpha4"))
        self.pf = ParticleFilter(
            self.fp, dense, int(gp("n_particles")), mm, temper=float(gp("likelihood_temper")), rng=np.random.default_rng(0)
        )
        self.ekf = WifiEKF(mm)
        self.global_init = bool(gp("global_init"))
        self.reset(np.array(self.fp.start_pose, dtype=float), self.global_init)

        self.odom_pose = None  # latest odometry pose (odom frame)
        self.used_odom = None  # odometry pose up to which the filter has been propagated
        self.last_fix = None

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/wiloc/pose", 10)
        self.fix_pub = self.create_publisher(PoseWithCovarianceStamped, "/wiloc/wifi_fix", 10)
        self.cloud_pub = self.create_publisher(PoseArray, "/wiloc/particles", 5)
        self.path_pub = self.create_publisher(Path, "/wiloc/path", 5)
        self.tf = TransformBroadcaster(self)
        self.path = Path()
        self.path.header.frame_id = "map"

        self.create_subscription(Odometry, "/wheel_odom", self.on_odom, 50)
        self.create_subscription(RssiScan, "/wifi/scan", self.on_scan, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initialpose, 1)
        self.create_service(Empty, "/wiloc/global_localization", self.on_global)
        self.create_timer(1.0 / float(gp("publish_rate")), self.on_timer)
        self.get_logger().info(
            f"WiLoc localisation: method={self.method}, {len(rmap.positions)} fingerprints, "
            f"{rmap.n_aps} APs, {'global' if self.global_init else 'tracking'} initialisation"
        )

    # ---------------------------------------------------------- setup
    def _load_or_survey(self, path, seed, spacing, samples) -> RadioMap:
        if path:
            self.get_logger().info(f"loading radio map {path}")
            return RadioMap.load(path)
        self.get_logger().info(f"no radio_map given: running simulated site survey ({spacing} m grid, {samples} scans/RP)")
        sim = RadioSimulator(self.fp, RadioConfig(seed=seed))
        return RadioMap.survey(sim, self.fp.grid(float(spacing)), int(samples), np.random.default_rng(123))

    def reset(self, pose, global_init=False):
        if global_init:
            self.pf.init_global()
            self.ekf.init_pose(pose, np.diag([25.0, 25.0, 3.0]))
        else:
            self.pf.init_pose(pose)
            self.ekf.init_pose(pose)
        self.wknn_pose = np.array(pose, dtype=float)

    # ---------------------------------------------------------- callbacks
    def on_odom(self, msg):
        self.odom_pose = np.array(pose2d_from_odom(msg))
        if self.used_odom is None:
            self.used_odom = self.odom_pose.copy()

    def propagate(self):
        if self.odom_pose is None:
            return
        d = odom_delta(self.used_odom, self.odom_pose)
        if abs(d[1]) < 0.005 and abs(d[0]) + abs(d[2]) < 0.005:
            return
        self.used_odom = self.odom_pose.copy()
        if self.method == "pf":
            self.pf.predict(d)
        elif self.method == "ekf":
            self.ekf.predict(d)

    def on_scan(self, msg: RssiScan):
        z = np.array(msg.rssi_dbm, dtype=float)
        if len(z) != self.pf.map.mean.shape[0]:
            self.get_logger().warn("scan AP count does not match radio map; ignoring")
            return
        self.propagate()
        xy, cov = self.fix_est.predict_with_cov(z[None])
        self.last_fix = (xy[0], cov[0])
        if self.method == "pf":
            self.pf.update(z)
        elif self.method == "ekf":
            self.ekf.update_position(xy[0], cov[0])
        else:
            self.wknn_pose[:2] = xy[0]
        self.publish_fix(msg.header.stamp)

    def on_initialpose(self, msg):
        p = msg.pose.pose
        pose = np.array([p.position.x, p.position.y, quat_to_yaw(p.orientation)])
        self.get_logger().info(f"re-initialising at {pose.round(2)}")
        self.reset(pose)

    def on_global(self, req, resp):
        self.get_logger().info("global localisation requested: spreading particles")
        self.reset(np.array(self.fp.start_pose), global_init=True)
        return resp

    # ---------------------------------------------------------- output
    def current_estimate(self):
        if self.method == "pf":
            pose, cov = self.pf.estimate()
        elif self.method == "ekf":
            pose, cov = self.ekf.estimate()
        else:
            pose = self.wknn_pose.copy()
            if self.odom_pose is not None:
                pose[2] = self.odom_pose[2]
            cov = self.last_fix[1] if self.last_fix else np.eye(2)
        return pose, cov

    def on_timer(self):
        self.propagate()
        pose, cov = self.current_estimate()
        now = self.get_clock().now().to_msg()

        msg = PoseWithCovarianceStamped()
        msg.header.stamp, msg.header.frame_id = now, "map"
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(pose[0]), float(pose[1])
        msg.pose.pose.orientation = yaw_to_quat(float(pose[2]))
        c = np.zeros(36)
        c[0], c[1], c[6], c[7] = cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]
        c[35] = 0.1
        msg.pose.covariance = c.tolist()
        self.pose_pub.publish(msg)

        ps = PoseStamped()
        ps.header, ps.pose = msg.header, msg.pose.pose
        self.path.header.stamp = now
        self.path.poses.append(ps)
        self.path.poses = self.path.poses[-3000:]
        self.path_pub.publish(self.path)

        if self.method == "pf":
            arr = PoseArray()
            arr.header = msg.header
            step = max(1, self.pf.n // 400)
            for x, y, th in self.pf.particles[::step]:
                p = Pose()
                p.position.x, p.position.y = float(x), float(y)
                p.orientation = yaw_to_quat(float(th))
                arr.poses.append(p)
            self.cloud_pub.publish(arr)

        if self.odom_pose is not None:
            # map->odom = T(map->base) * T(odom->base)^-1
            ox, oy, oth = self.odom_pose
            yaw = float(pose[2]) - oth
            c_, s_ = math.cos(yaw), math.sin(yaw)
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = now, "map", "odom"
            t.transform.translation.x = float(pose[0] - (c_ * ox - s_ * oy))
            t.transform.translation.y = float(pose[1] - (s_ * ox + c_ * oy))
            t.transform.rotation = yaw_to_quat(yaw)
            self.tf.sendTransform(t)

    def publish_fix(self, stamp):
        xy, cov = self.last_fix
        msg = PoseWithCovarianceStamped()
        msg.header.stamp, msg.header.frame_id = stamp, "map"
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(xy[0]), float(xy[1])
        msg.pose.pose.orientation.w = 1.0
        c = np.zeros(36)
        c[0], c[1], c[6], c[7] = cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]
        msg.pose.covariance = c.tolist()
        self.fix_pub.publish(msg)


def main():
    run_node(LocalizationNode)
