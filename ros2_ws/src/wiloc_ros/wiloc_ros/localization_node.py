"""Wi-Fi + odometry localisation node (the AMCL of WiLoc).

On start-up the node loads a radio map (``radio_map`` .npz from
``python -m wiloc survey``) or, if none is given, performs the simulated site
survey of the venue's radio world itself (cached on disk by ``wiloc.cache``,
so later starts take well under a second).  It then fuses ``/wheel_odom``
with ``/wifi/scan`` with three methods side by side, all fed every scan, so
they can be compared live on the same data ("the race"):

    "pf"    particle filter
    "ekf"   EKF with WKNN position fixes
    "wknn"  scan-by-scan WKNN, no filtering (baseline)

``method`` picks the primary one, which drives ``/wiloc/pose`` and the
map -> odom transform.

Publishes
  /wiloc/pose        geometry_msgs/PoseWithCovarianceStamped   primary method (map frame)
  /wiloc/pose/pf, /wiloc/pose/ekf, /wiloc/pose/wknn            every method
  /wiloc/particles   geometry_msgs/PoseArray
  /wiloc/wifi_fix    geometry_msgs/PoseWithCovarianceStamped   raw WKNN fix
  /wiloc/path        nav_msgs/Path   primary method, 1 Hz, last 10 min
  /wiloc/status      std_msgs/String  JSON: filter internals + radio-map
                     prediction at the estimate (for the dashboard)
  /wiloc/hud         visualization_msgs/MarkerArray  status text for RViz
  TF map -> odom
Subscribes
  /initialpose       re-initialise around a pose set in RViz ("2D Pose Estimate")
Services
  /wiloc/global_localization   std_srvs/Empty: forget the pose, spread particles
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from wiloc.estimators import KNN
from wiloc.filters import ParticleFilter, WifiEKF
from wiloc.fingerprint import DenseRadioMap, RadioMap
from wiloc.geometry import wrap_angle
from wiloc.motion import OdometryMotionModel, odom_delta
from wiloc.cache import cached_maps
from wiloc_msgs.msg import RssiScan

from . import theme
from .common import load_floorplan, pose2d_from_odom, quat_to_yaw, radio_config, run_node, yaw_to_quat


METHODS = ("pf", "ekf", "wknn")


class LocalizationNode(Node):
    def __init__(self):
        super().__init__("wiloc_localization")
        params = {
            "floorplan": "office",
            "method": "pf",
            "n_particles": 1500,
            "global_init": False,
            "radio_map": "",
            "map_cache": True,
            "respawn_global_after": 30.0,
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
            "update_min_d": 0.1,
            "update_min_a": 0.2,
            "update_max_interval": 2.0,
        }
        for k, v in params.items():
            self.declare_parameter(k, v)
        gp = lambda n: self.get_parameter(n).value
        self.update_min_d, self.update_min_a = float(gp("update_min_d")), float(gp("update_min_a"))
        self.update_max_interval = float(gp("update_max_interval"))

        self.fp = load_floorplan(gp("floorplan"))
        self.method = gp("method")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, not {self.method!r}")
        self.radio_cfg = radio_config(self, self.fp)
        self.rmap, self.dense = self._radio_maps(gp)
        self.fix_est = KNN(int(gp("knn_k")), weighted=True).fit(self.rmap)
        mm = OdometryMotionModel(gp("alpha1"), gp("alpha2"), gp("alpha3"), gp("alpha4"))
        self.pf = ParticleFilter(
            self.fp, self.dense, int(gp("n_particles")), mm, temper=float(gp("likelihood_temper")), rng=np.random.default_rng(0)
        )
        self.ekf = WifiEKF(mm)
        self.global_init = bool(gp("global_init"))
        # A localiser that starts long after the simulation did was respawned
        # after a crash: the robot has moved since, so it must not trust the
        # start pose (checked on the first scan, once sim time is known).
        self.respawn_global_after = float(gp("respawn_global_after"))
        self.start_checked = False

        # The diff-drive plugin's odometry starts at the odom-frame origin when
        # the robot is spawned at ``start_pose``; using that origin (instead of
        # the first message received) stays correct even if the robot has
        # already moved by the time this node finishes its survey.
        self.odom_pose = None  # latest odometry pose (odom frame)
        self.used_odom = np.zeros(3)  # odometry pose up to which the filter has been propagated
        self.last_fix = None
        self.counts = {"scans": 0, "updates": 0, "skipped_stationary": 0, "ignored": 0}
        self.reset(np.array(self.fp.start_pose, dtype=float), self.global_init, odom_yaw=0.0)

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/wiloc/pose", 10)
        self.method_pubs = {m: self.create_publisher(PoseWithCovarianceStamped, f"/wiloc/pose/{m}", 10) for m in METHODS}
        self.fix_pub = self.create_publisher(PoseWithCovarianceStamped, "/wiloc/wifi_fix", 10)
        self.cloud_pub = self.create_publisher(PoseArray, "/wiloc/particles", 5)
        self.path_pub = self.create_publisher(Path, "/wiloc/path", 5)
        self.status_pub = self.create_publisher(String, "/wiloc/status", 10)
        self.hud_pub = self.create_publisher(MarkerArray, "/wiloc/hud", 5)
        self.tf = TransformBroadcaster(self)
        self.path = Path()
        self.path.header.frame_id = "map"
        self.last_path_t = -1e9

        self.create_subscription(Odometry, "/wheel_odom", self.on_odom, 50)
        self.create_subscription(RssiScan, "/wifi/scan", self.on_scan, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initialpose, 1)
        self.create_service(Empty, "/wiloc/global_localization", self.on_global)
        self.create_timer(1.0 / float(gp("publish_rate")), self.on_timer)
        self.get_logger().info(
            f"WiLoc localisation: method={self.method}, {len(self.rmap.positions)} fingerprints, "
            f"{self.rmap.n_aps} APs, {'global' if self.global_init else 'tracking'} initialisation"
        )

    # ---------------------------------------------------------- setup
    def _radio_maps(self, gp):
        """Fingerprints and the dense GP map: from a file, the cache, or a fresh survey."""
        t0 = time.monotonic()
        path, method = gp("radio_map"), gp("map_method")
        if path:
            self.get_logger().info(f"loading radio map {path}")
            rmap = RadioMap.load(path)
            return rmap, DenseRadioMap.from_radio_map(rmap, self.fp, method=method)
        spacing, samples = float(gp("survey_spacing")), int(gp("survey_samples"))
        rmap, dense, hit = cached_maps(
            self.fp, self.radio_cfg, spacing=spacing, samples=samples, map_method=method, use_cache=bool(gp("map_cache"))
        )
        how = "loaded from cache" if hit else f"simulated site survey ({spacing} m grid, {samples} scans/RP)"
        self.get_logger().info(f"radio map {how} in {time.monotonic() - t0:.1f} s")
        return rmap, dense

    def reset(self, pose, global_init=False, odom_yaw=None):
        if global_init:
            self.pf.init_global()
            self.ekf.init_pose(pose, np.diag([25.0, 25.0, 3.0]))
        else:
            self.pf.init_pose(pose)
            self.ekf.init_pose(pose)
        self.wknn_pose = np.array(pose, dtype=float)
        # WKNN gives no heading: carry it from odometry, offset to the map frame.
        if odom_yaw is None:
            odom_yaw = self.odom_pose[2] if self.odom_pose is not None else 0.0
        self.wknn_yaw_offset = float(pose[2] - odom_yaw)
        self.update_odom = None  # odometry pose at the last PF measurement update
        self.last_update_t = -1e9

    # ---------------------------------------------------------- callbacks
    def on_odom(self, msg):
        self.odom_pose = np.array(pose2d_from_odom(msg))

    def propagate(self):
        if self.odom_pose is None:
            return
        d = odom_delta(self.used_odom, self.odom_pose)
        if abs(d[1]) < 0.005 and abs(d[0]) + abs(d[2]) < 0.005:
            return
        self.used_odom = self.odom_pose.copy()
        self.pf.predict(d)
        self.ekf.predict(d)

    def _should_update(self) -> bool:
        """AMCL-style gating: re-weight particles after the robot has moved.

        Repeated scans from a standing robot share the same map error, so
        applying them all makes the filter collapse onto one (possibly wrong)
        mode before motion can disambiguate it.  A standing robot still gets an
        update every ``update_max_interval`` seconds, otherwise a parked robot
        could never re-localise after global initialisation or a kidnapping.
        """
        if self.update_odom is None or self.odom_pose is None:
            return True
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_update_t >= self.update_max_interval:
            return True
        d = self.odom_pose - self.update_odom
        turned = abs((d[2] + math.pi) % (2 * math.pi) - math.pi)
        return math.hypot(d[0], d[1]) >= self.update_min_d or turned >= self.update_min_a

    def _scan_to_map_order(self, msg: RssiScan):
        """RSSI vector ordered like the radio map's APs (matched by AP id)."""
        index = {ap: i for i, ap in enumerate(msg.ap_ids)}
        missing = [ap for ap in self.rmap.ap_ids if ap not in index]
        if len(missing) == len(self.rmap.ap_ids):
            return None
        if missing and self.counts["scans"] == 0:
            self.get_logger().warn(f"APs in the radio map but not in scans (treated as not heard): {missing}")
        return np.array([msg.rssi_dbm[index[ap]] if ap in index else np.nan for ap in self.rmap.ap_ids], dtype=float)

    def on_scan(self, msg: RssiScan):
        if not self.start_checked:
            self.start_checked = True
            t = self.get_clock().now().nanoseconds * 1e-9
            if not self.global_init and t > self.respawn_global_after:
                self.get_logger().warn(f"started at sim time {t:.0f} s (respawned?): localising globally")
                self.reset(np.array(self.fp.start_pose, dtype=float), global_init=True)
        z = self._scan_to_map_order(msg)
        if z is None:
            self.counts["ignored"] += 1
            self.get_logger().warn("scan shares no AP ids with the radio map; ignoring", throttle_duration_sec=5.0)
            return
        self.pf.sens = float(msg.sensitivity_dbm)
        self.counts["scans"] += 1
        self.propagate()
        xy, cov = self.fix_est.predict_with_cov(z[None])
        self.last_fix = (xy[0], cov[0])
        self.last_scan = z
        # The motion gate is the particle filter's (see _should_update); the
        # EKF gates its own updates (chi-square) and WKNN has no memory.
        if self._should_update():
            self.pf.update(z)
            self.counts["updates"] += 1
            self.last_update_t = self.get_clock().now().nanoseconds * 1e-9
            self.update_odom = None if self.odom_pose is None else self.odom_pose.copy()
        else:
            self.counts["skipped_stationary"] += 1
        if not np.isnan(z).all():
            self.ekf.update_position(xy[0], cov[0])
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
    def current_estimate(self, method=None):
        method = method or self.method
        if method == "pf":
            pose, cov = self.pf.estimate()
        elif method == "ekf":
            pose, cov = self.ekf.estimate()
        else:
            pose = self.wknn_pose.copy()
            if self.odom_pose is not None:
                pose[2] = float(wrap_angle(self.odom_pose[2] + self.wknn_yaw_offset))
            cov = self.last_fix[1] if self.last_fix else np.eye(2)
        return pose, cov

    @staticmethod
    def _pose_msg(pose, cov, stamp):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp, msg.header.frame_id = stamp, "map"
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(pose[0]), float(pose[1])
        msg.pose.pose.orientation = yaw_to_quat(float(pose[2]))
        c = np.zeros(36)
        c[0], c[1], c[6], c[7] = cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]
        c[35] = 0.1
        msg.pose.covariance = c.tolist()
        return msg

    def on_timer(self):
        self.propagate()
        now = self.get_clock().now().to_msg()
        estimates = {m: self.current_estimate(m) for m in METHODS}
        for m, (p, c) in estimates.items():
            self.method_pubs[m].publish(self._pose_msg(p, c, now))
        pose, cov = estimates[self.method]
        msg = self._pose_msg(pose, cov, now)
        self.pose_pub.publish(msg)

        t = self.get_clock().now().nanoseconds * 1e-9
        if t - self.last_path_t >= 1.0:
            self.last_path_t = t
            ps = PoseStamped()
            ps.header, ps.pose = msg.header, msg.pose.pose
            self.path.header.stamp = now
            self.path.poses.append(ps)
            self.path.poses = self.path.poses[-600:]
            self.path_pub.publish(self.path)

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

        self.publish_status(pose, cov, now, estimates)

    def publish_status(self, pose, cov, stamp, estimates):
        mu, sd = self.dense.lookup(pose[None, :2])
        spread = float(math.sqrt(max(np.trace(cov), 0.0)))
        status = {
            "method": self.method,
            "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
            "spread_m": spread,
            "counts": self.counts,
            "ap_ids": list(self.rmap.ap_ids),
            "map_rssi_at_estimate": [float(v) for v in mu[0]],
            "map_std_at_estimate": [float(v) for v in sd[0]],
            "fix": None if self.last_fix is None else [float(self.last_fix[0][0]), float(self.last_fix[0][1])],
            # every method: x, y, yaw and 1-sigma position spread
            "estimates": {
                m: [float(p[0]), float(p[1]), float(p[2]), float(math.sqrt(max(np.trace(c[:2, :2]), 0.0)))]
                for m, (p, c) in estimates.items()
            },
            # particle filter internals (it always runs)
            "n_particles": self.pf.n,
            "ess": float(self.pf.ess()),
            "resamples": self.pf.n_resamples,
            "injected": self.pf.n_injected,
            "w_ratio": float(self.pf.w_fast / self.pf.w_slow) if self.pf.w_slow else 1.0,
        }
        self.status_pub.publish(String(data=json.dumps(status)))

        hud = Marker()
        hud.header.frame_id, hud.header.stamp = "map", stamp
        hud.ns, hud.id, hud.type, hud.action = "hud", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        hud.pose.position.x, hud.pose.position.y, hud.pose.position.z = self.fp.width / 2, -1.0, 0.5
        hud.pose.orientation.w = 1.0
        hud.scale.z = 0.5
        r, g, b = theme.rgb(theme.AMBER)
        hud.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
        name = {"pf": "Particle filter", "ekf": "EKF", "wknn": "WKNN (no filter)"}.get(self.method, self.method)
        text = f"{name}  |  pose ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg)  |  sigma {spread:.2f} m"
        text += f"  |  N_eff {status['ess']:.0f}/{self.pf.n}"
        text += f"  |  Wi-Fi updates {self.counts['updates']}"
        hud.text = text
        arr = MarkerArray()
        arr.markers.append(hud)
        self.hud_pub.publish(arr)

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
