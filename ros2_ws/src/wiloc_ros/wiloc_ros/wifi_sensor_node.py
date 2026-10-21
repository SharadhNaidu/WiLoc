"""Simulated Wi-Fi receiver on the robot.

Gazebo has no realistic indoor RSSI model, so this node plays the part of the
robot's Wi-Fi card: it reads the robot's true pose from the Gazebo p3d plugin
(``/ground_truth``) and publishes ``wiloc_msgs/RssiScan`` at ``scan_rate`` Hz
using the WiLoc propagation model (log-distance + multi-wall + correlated
shadowing + temporal noise).

For visualisation and as evidence of what the simulator is doing it also
publishes, for every scan, the robot->AP radio links (coloured by RSSI and
labelled with dBm and walls crossed), a scan "pulse", and a JSON breakdown of
how each reading was formed (path loss, wall loss, shadowing, noise).

Note: ``radio_seed`` and the noise parameters define the "real" radio world.
If you change them, the localiser's radio map must come from a survey of the
same world (same ``radio_seed``; a ``radio_map`` file surveyed with matching
settings), otherwise the fingerprints describe a different building.

Topics
  sub  /ground_truth            nav_msgs/Odometry
  sub  /wiloc/heatmap_select    std_msgs/Int32   (-1 = strongest AP, i = AP index)
  pub  /wifi/scan               wiloc_msgs/RssiScan
  pub  /wifi/scan_debug         std_msgs/String  (JSON RSSI breakdown per AP)
  pub  /wiloc/wifi_links        visualization_msgs/MarkerArray
  pub  /wiloc/environment       visualization_msgs/MarkerArray (latched)
  pub  /wiloc/rssi_heatmap      nav_msgs/OccupancyGrid (latched)
  pub  /wiloc/rssi_heatmap_markers  visualization_msgs/MarkerArray: coloured heat map + legend (latched)
"""

from __future__ import annotations

import json

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import ColorRGBA, Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from wiloc.radio import RadioConfig, RadioSimulator
from wiloc_msgs.msg import RssiScan

from . import theme
from .common import load_floorplan, run_node

HEAT_LO, HEAT_HI = -85.0, -25.0  # dBm range of the heat-map / link colour ramp


def _rgba(hex_color: str, a: float = 1.0) -> ColorRGBA:
    r, g, b = theme.rgb(hex_color)
    return ColorRGBA(r=r, g=g, b=b, a=a)


def rssi_color(rssi: float, alpha: float = 0.95) -> ColorRGBA:
    """Colour of a reading on the shared heat ramp (weak -> strong).

    Links use the darker 40-100 % part of the ramp so weak ones stay visible on white.
    """
    if not np.isfinite(rssi):
        return _rgba(theme.MUTED, 0.35)
    t = 0.4 + 0.6 * float(np.clip((rssi - HEAT_LO) / (HEAT_HI - HEAT_LO), 0, 1))
    r, g, b = theme.heat_rgb(t)
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=alpha)


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
        self.n_scans = 0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(RssiScan, "/wifi/scan", 10)
        self.debug_pub = self.create_publisher(String, "/wifi/scan_debug", 10)
        self.links_pub = self.create_publisher(MarkerArray, "/wiloc/wifi_links", 10)
        self.env_pub = self.create_publisher(MarkerArray, "/wiloc/environment", latched)
        self.heat_pub = self.create_publisher(OccupancyGrid, "/wiloc/rssi_heatmap", latched)
        self.heat_marker_pub = self.create_publisher(MarkerArray, "/wiloc/rssi_heatmap_markers", latched)
        self.create_subscription(Odometry, "/ground_truth", self.on_truth, 10)
        self.create_subscription(Int32, "/wiloc/heatmap_select", self.on_heatmap_select, 10)
        self.create_timer(1.0 / float(gp("scan_rate")), self.on_timer)
        self._prepare_heatmaps()
        self.heat_choice = -1
        self.publish_environment()
        self.publish_heatmap()
        self.get_logger().info(
            f"Wi-Fi sensor ready: {len(self.fp.access_points)} APs, {gp('scan_rate')} Hz, "
            f"shadowing {cfg.shadowing_sigma_db} dB / {cfg.shadowing_corr_m} m, noise {cfg.temporal_sigma_db} dB"
        )

    def on_truth(self, msg: Odometry):
        p = msg.pose.pose.position
        self.pose = np.array([p.x, p.y])

    def on_heatmap_select(self, msg: Int32):
        self.heat_choice = int(msg.data) if -1 <= msg.data < len(self.fp.access_points) else -1
        self.publish_heatmap()
        self.publish_environment()

    def on_timer(self):
        if self.pose is None:
            return
        pose = self.pose.copy()
        z = self.sim.measure(pose[None, :], self.rng)[0]
        stamp = self.get_clock().now().to_msg()
        self.n_scans += 1

        msg = RssiScan()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_footprint"
        msg.ap_ids = list(self.sim.ap_ids)
        msg.rssi_dbm = [float(v) for v in z]
        msg.sensitivity_dbm = float(self.sim.cfg.sensitivity_dbm)
        self.pub.publish(msg)

        rows = self.sim.breakdown(pose)
        for r, zj in zip(rows, z):
            r["measured_dbm"] = None if np.isnan(zj) else float(zj)
            r["noise_db"] = None if np.isnan(zj) else float(zj - r["mean_rssi_dbm"])
        self.debug_pub.publish(
            String(
                data=json.dumps(
                    {
                        "scan": self.n_scans,
                        "pose": [float(pose[0]), float(pose[1])],
                        "sensitivity_dbm": self.sim.cfg.sensitivity_dbm,
                        "temporal_sigma_db": self.sim.cfg.temporal_sigma_db,
                        "aps": rows,
                    }
                )
            )
        )
        self.publish_links(pose, z, rows, stamp)

    # ------------------------------------------------------------ visuals
    def _marker(self, ns, mid, mtype, stamp, lifetime=0.0):
        m = Marker()
        m.header.frame_id, m.header.stamp = "map", stamp
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        if lifetime:
            m.lifetime = Duration(sec=int(lifetime), nanosec=int((lifetime % 1) * 1e9))
        return m

    def publish_links(self, pose, z, rows, stamp):
        arr = MarkerArray()
        antenna = Point(x=float(pose[0]), y=float(pose[1]), z=0.25)
        h = self.fp.wall_height
        for j, (ap, zj, row) in enumerate(zip(self.fp.access_points, z, rows)):
            line = self._marker("wifi_links", j, Marker.LINE_LIST, stamp, lifetime=1.5)
            line.scale.x = 0.06 if np.isfinite(zj) else 0.02
            line.color = rssi_color(zj, 0.85)
            line.points = [antenna, Point(x=ap.x, y=ap.y, z=h)]
            arr.markers.append(line)

            label = self._marker("wifi_labels", j, Marker.TEXT_VIEW_FACING, stamp, lifetime=1.5)
            f = 0.62  # place labels closer to the AP so they do not pile up on the robot
            label.pose.position.x = float(pose[0] + f * (ap.x - pose[0]))
            label.pose.position.y = float(pose[1] + f * (ap.y - pose[1]))
            label.pose.position.z = 0.25 + f * (h - 0.25) + 0.3
            label.scale.z = 0.34
            label.color = rssi_color(zj, 1.0)
            walls = len(row["walls"])
            wall_txt = f"{walls} wall{'s' if walls != 1 else ''} -{row['wall_loss_db']:.0f} dB" if walls else "line of sight"
            label.text = f"{ap.id} {zj:.0f} dBm | {wall_txt}" if np.isfinite(zj) else f"{ap.id} not heard"
            arr.markers.append(label)

        pulse = self._marker("scan_pulse", 0, Marker.CYLINDER, stamp, lifetime=0.35)
        pulse.pose.position.x, pulse.pose.position.y, pulse.pose.position.z = float(pose[0]), float(pose[1]), 0.05
        pulse.scale.x = pulse.scale.y = 0.9
        pulse.scale.z = 0.02
        pulse.color = _rgba(theme.BLUE, 0.45)
        arr.markers.append(pulse)
        self.links_pub.publish(arr)

    def publish_environment(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, (seg, mat) in enumerate(zip(self.fp.walls, self.fp.wall_materials)):
            m = self._marker("walls", i, Marker.CUBE, stamp)
            x1, y1, x2, y2 = seg
            yaw = np.arctan2(y2 - y1, x2 - x1)
            m.pose.position.x, m.pose.position.y = (x1 + x2) / 2, (y1 + y2) / 2
            m.pose.position.z = 0.3
            m.pose.orientation.z, m.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
            m.scale.x = float(np.hypot(x2 - x1, y2 - y1) + self.fp.wall_thickness)
            m.scale.y, m.scale.z = self.fp.wall_thickness, 0.6  # low walls keep the top view readable
            m.color = _rgba(theme.MATERIALS.get(mat, theme.TEXT), 0.65 if mat == "glass" else 1.0)
            arr.markers.append(m)
        for i, ap in enumerate(self.fp.access_points):
            selected = self.heat_choice in (-1, i)
            m = self._marker("access_points", i, Marker.CYLINDER, stamp)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = ap.x, ap.y, self.fp.wall_height
            m.scale.x = m.scale.y = 0.45
            m.scale.z = 0.12
            m.color = _rgba(theme.ap_color(i), 1.0 if selected else 0.35)
            arr.markers.append(m)
            t = self._marker("ap_labels", i, Marker.TEXT_VIEW_FACING, stamp)
            t.pose.position.x, t.pose.position.y, t.pose.position.z = ap.x, ap.y + 0.45, self.fp.wall_height + 0.4
            t.scale.z = 0.42
            t.color = _rgba(theme.ap_color(i))
            t.text = f"{ap.id}  ch {ap.channel}"
            arr.markers.append(t)
        route = self._marker("route", 0, Marker.LINE_STRIP, stamp)
        route.scale.x = 0.03
        route.color = _rgba(theme.GREEN, 0.35)
        for x, y in np.vstack([self.fp.start_pose[:2], self.fp.waypoints]):
            route.points.append(Point(x=float(x), y=float(y), z=0.02))
        arr.markers.append(route)
        title = self._marker("hud", 100, Marker.TEXT_VIEW_FACING, stamp)
        title.pose.position.x, title.pose.position.y, title.pose.position.z = self.fp.width / 2, self.fp.height + 1.2, 0.5
        title.scale.z = 0.6
        title.color = _rgba(theme.TEXT)
        which = "strongest AP" if self.heat_choice < 0 else self.fp.access_points[self.heat_choice].id
        title.text = f"WiLoc  |  {self.fp.name} floor  |  heat map: {which}"
        arr.markers.append(title)
        floor = self._marker("floor", 0, Marker.CUBE, stamp)
        floor.pose.position.x, floor.pose.position.y, floor.pose.position.z = self.fp.width / 2, self.fp.height / 2, -0.03
        floor.scale.x, floor.scale.y, floor.scale.z = self.fp.width, self.fp.height, 0.02
        floor.color = _rgba(theme.PANEL)
        arr.markers.append(floor)
        self.env_pub.publish(arr)

    def _prepare_heatmaps(self, res: float = 0.2):
        self.heat_res = res
        self.heat_nx, self.heat_ny = int(self.fp.width / res), int(self.fp.height / res)
        gx, gy = np.meshgrid((np.arange(self.heat_nx) + 0.5) * res, (np.arange(self.heat_ny) + 0.5) * res)
        self.heat_rssi = self.sim.mean_rssi(np.column_stack([gx.ravel(), gy.ravel()]))  # (cells, A)

    def publish_heatmap(self):
        vals = self.heat_rssi.max(axis=1) if self.heat_choice < 0 else self.heat_rssi[:, self.heat_choice]
        lo, hi = -85.0, -25.0
        # RViz's "costmap" scheme draws 1..98 as a blue->red gradient but 99
        # and 100 in cyan/magenta, so keep the heat map inside 1..98.
        cells = np.clip(1 + (vals - lo) / (hi - lo) * 97.0, 1, 98).astype(np.int8)
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self.heat_res
        grid.info.width, grid.info.height = self.heat_nx, self.heat_ny
        grid.info.origin.orientation.w = 1.0
        grid.data = cells.tolist()
        self.heat_pub.publish(grid)
        # Colour range adapted to what is shown, so the map is not one saturated colour.
        lo, hi = np.percentile(vals, [3, 99])
        self.publish_heat_markers(vals, 5 * np.floor(lo / 5), 5 * np.ceil(hi / 5))

    def publish_heat_markers(self, vals, lo, hi):
        """Heat map as one CUBE_LIST in the theme's colour ramp, plus a dBm legend."""
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        tiles = self._marker("rssi_heatmap", 0, Marker.CUBE_LIST, stamp)
        res = self.heat_res
        tiles.scale.x = tiles.scale.y = res * 1.04  # slight overlap: no seams between tiles
        tiles.scale.z = 0.01
        t = (vals - lo) / (hi - lo)
        cols = theme.heat_rgb(t)
        gx, gy = np.meshgrid((np.arange(self.heat_nx) + 0.5) * res, (np.arange(self.heat_ny) + 0.5) * res)
        for x, y, (r, g, b) in zip(gx.ravel(), gy.ravel(), cols):
            tiles.points.append(Point(x=float(x), y=float(y), z=-0.01))
            tiles.colors.append(ColorRGBA(r=float(r), g=float(g), b=float(b), a=0.9))
        arr.markers.append(tiles)

        # legend under the floor plan
        n, x0, width, y0 = 24, 1.0, 7.0, -2.1
        bar = self._marker("rssi_legend", 0, Marker.CUBE_LIST, stamp)
        bar.scale.x, bar.scale.y, bar.scale.z = width / n, 0.35, 0.01
        for i, (r, g, b) in enumerate(theme.heat_rgb(np.linspace(0, 1, n))):
            bar.points.append(Point(x=x0 + (i + 0.5) * width / n, y=y0, z=0.0))
            bar.colors.append(ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0))
        arr.markers.append(bar)
        for j, (xx, txt) in enumerate(((x0, f"{lo:.0f} dBm"), (x0 + width, f"{hi:.0f} dBm"), (x0 + width / 2, "RSSI"))):
            lbl = self._marker("rssi_legend_text", j, Marker.TEXT_VIEW_FACING, stamp)
            lbl.pose.position.x, lbl.pose.position.y, lbl.pose.position.z = xx, y0 - 0.5, 0.3
            lbl.scale.z = 0.32
            lbl.color = _rgba(theme.MUTED)
            lbl.text = txt
            arr.markers.append(lbl)
        self.heat_marker_pub.publish(arr)


def main():
    run_node(WifiSensorNode)
