"""Simulated Wi-Fi receiver on the robot.

Gazebo has no realistic indoor RSSI model, so this node plays the part of the
robot's Wi-Fi card: it reads the robot's true pose from the Gazebo p3d plugin
(``/ground_truth``) and publishes ``wiloc_msgs/RssiScan`` at ``scan_rate`` Hz
using the WiLoc propagation model (log-distance + multi-wall + correlated
shadowing + temporal noise).  It also publishes the floor plan and access
points as RViz markers and a heat map of the strongest-AP RSSI.

Note: ``radio_seed`` and the noise parameters define the "real" radio world.
If you change them, the localiser's radio map must come from a survey of the
same world (same ``radio_seed``; a ``radio_map`` file surveyed with matching
settings), otherwise the fingerprints describe a different building.

Topics
  sub  /ground_truth          nav_msgs/Odometry
  pub  /wifi/scan             wiloc_msgs/RssiScan
  pub  /wiloc/environment     visualization_msgs/MarkerArray (latched)
  pub  /wiloc/rssi_heatmap    nav_msgs/OccupancyGrid (latched)
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from wiloc.radio import RadioConfig, RadioSimulator
from wiloc_msgs.msg import RssiScan

from .common import load_floorplan, run_node

MATERIAL_COLORS = {
    "concrete": (0.35, 0.35, 0.35),
    "brick": (0.65, 0.3, 0.2),
    "drywall": (0.85, 0.85, 0.8),
    "glass": (0.4, 0.7, 0.95),
    "wood": (0.55, 0.4, 0.25),
}


class WifiSensorNode(Node):
    def __init__(self):
        super().__init__("wifi_sensor")
        self.declare_parameter("floorplan", "office")
        self.declare_parameter("radio_seed", 7)
        self.declare_parameter("scan_rate", 2.0)
        self.declare_parameter("temporal_sigma_db", 3.0)
        self.declare_parameter("shadowing_sigma_db", 5.0)
        self.declare_parameter("noise_seed", 0)
        gp = lambda n: self.get_parameter(n).value
        self.fp = load_floorplan(gp("floorplan"))
        cfg = RadioConfig(
            seed=int(gp("radio_seed")),
            temporal_sigma_db=float(gp("temporal_sigma_db")),
            shadowing_sigma_db=float(gp("shadowing_sigma_db")),
        )
        self.sim = RadioSimulator(self.fp, cfg)
        self.rng = np.random.default_rng(int(gp("noise_seed")))
        self.pose = None

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(RssiScan, "/wifi/scan", 10)
        self.env_pub = self.create_publisher(MarkerArray, "/wiloc/environment", latched)
        self.heat_pub = self.create_publisher(OccupancyGrid, "/wiloc/rssi_heatmap", latched)
        self.create_subscription(Odometry, "/ground_truth", self.on_truth, 10)
        self.create_timer(1.0 / float(gp("scan_rate")), self.on_timer)
        self.publish_environment()
        self.publish_heatmap()
        self.get_logger().info(
            f"Wi-Fi sensor ready: {len(self.fp.access_points)} APs, {gp('scan_rate')} Hz, "
            f"shadowing {cfg.shadowing_sigma_db} dB / {cfg.shadowing_corr_m} m, noise {cfg.temporal_sigma_db} dB"
        )

    def on_truth(self, msg: Odometry):
        p = msg.pose.pose.position
        self.pose = np.array([p.x, p.y])

    def on_timer(self):
        if self.pose is None:
            return
        z = self.sim.measure(self.pose[None, :], self.rng)[0]
        msg = RssiScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_footprint"
        msg.ap_ids = list(self.sim.ap_ids)
        msg.rssi_dbm = [float(v) for v in z]
        msg.sensitivity_dbm = float(self.sim.cfg.sensitivity_dbm)
        self.pub.publish(msg)

    # ------------------------------------------------------------ visuals
    def publish_environment(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, (seg, mat) in enumerate(zip(self.fp.walls, self.fp.wall_materials)):
            m = Marker()
            m.header.frame_id, m.header.stamp = "map", stamp
            m.ns, m.id, m.type, m.action = "walls", i, Marker.CUBE, Marker.ADD
            x1, y1, x2, y2 = seg
            yaw = np.arctan2(y2 - y1, x2 - x1)
            m.pose.position.x, m.pose.position.y = (x1 + x2) / 2, (y1 + y2) / 2
            m.pose.position.z = self.fp.wall_height / 2
            m.pose.orientation.z, m.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
            m.scale.x = float(np.hypot(x2 - x1, y2 - y1) + self.fp.wall_thickness)
            m.scale.y, m.scale.z = self.fp.wall_thickness, self.fp.wall_height
            r, g, b = MATERIAL_COLORS.get(mat, (0.8, 0.8, 0.8))
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.45 if mat == "glass" else 0.8
            arr.markers.append(m)
        for i, ap in enumerate(self.fp.access_points):
            m = Marker()
            m.header.frame_id, m.header.stamp = "map", stamp
            m.ns, m.id, m.type, m.action = "access_points", i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = ap.x, ap.y, self.fp.wall_height
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.35
            m.color.r, m.color.a = 1.0, 1.0
            arr.markers.append(m)
            t = Marker()
            t.header = m.header
            t.ns, t.id, t.type, t.action = "ap_labels", i, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = ap.x, ap.y, self.fp.wall_height + 0.5
            t.pose.orientation.w = 1.0
            t.scale.z = 0.45
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = ap.id
            arr.markers.append(t)
        route = Marker()
        route.header.frame_id, route.header.stamp = "map", stamp
        route.ns, route.id, route.type, route.action = "route", 0, Marker.LINE_STRIP, Marker.ADD
        route.pose.orientation.w = 1.0
        route.scale.x = 0.03
        route.color.g, route.color.b, route.color.a = 0.8, 0.3, 0.6
        for x, y in np.vstack([self.fp.start_pose[:2], self.fp.waypoints]):
            route.points.append(Point(x=float(x), y=float(y), z=0.02))
        arr.markers.append(route)
        self.env_pub.publish(arr)

    def publish_heatmap(self, res: float = 0.2):
        nx, ny = int(self.fp.width / res), int(self.fp.height / res)
        gx, gy = np.meshgrid((np.arange(nx) + 0.5) * res, (np.arange(ny) + 0.5) * res)
        best = self.sim.mean_rssi(np.column_stack([gx.ravel(), gy.ravel()])).max(axis=1)
        lo, hi = -80.0, -25.0
        cells = np.clip((best - lo) / (hi - lo) * 100.0, 0, 100).astype(np.int8)
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = res
        grid.info.width, grid.info.height = nx, ny
        grid.info.origin.orientation.w = 1.0
        grid.data = cells.tolist()
        self.heat_pub.publish(grid)


def main():
    run_node(WifiSensorNode)
