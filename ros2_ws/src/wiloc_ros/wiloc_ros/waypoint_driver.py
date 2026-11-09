"""Drive the robot: planned patrol, click-to-go, and recovery when stuck.

The driver steers on the ground-truth pose: it plays the role of a teleoperator
generating motion so that the localiser can be evaluated on realistic data.
(``use_estimate:=true`` closes the loop on ``/wiloc/pose`` instead.)

Every leg is planned on the floor plan (``wiloc.planning``): A* on an
occupancy grid with the walls inflated by the robot radius, smoothed into a few
vertices.  So the patrol, a goal clicked on the map, and resuming after a
kidnapping all go through doors instead of into walls.

Modes (``/wiloc/drive_mode``, std_msgs/String):
  "auto"    follow the patrol route (endless with ``laps:=0``) or drive to a goal
  "manual"  publish nothing on /cmd_vel so the keyboard can drive
Goals (``/wiloc/goal``, geometry_msgs/PoseStamped) switch to "auto", drive
there, wait ``goal_hold`` seconds and then resume the patrol.

Safety nets: a pose jump of more than 1 m (kidnapping) triggers a replan; if
the robot is commanded forward but does not move for ``stuck_time`` seconds it
backs up and replans; if it stays stuck for ``watchdog_time`` seconds it is
teleported to the nearest patrol waypoint so a live demo can never stay wedged.

People and vehicles (the venue's ``movers:``) have no collision in Gazebo, so the
driver gives way instead: it slows down when one is ahead and stops while one
is within ``yield_dist`` in front, for at most ``yield_max`` seconds (then it
carries on, so a person who stops in a doorway cannot stall the demo).

Topics
  pub  /cmd_vel               geometry_msgs/Twist
  pub  /wiloc/plan            nav_msgs/Path       current planned route (map frame)
  pub  /wiloc/driver_status   std_msgs/String
  pub  /wiloc/drive_mode      std_msgs/String     ("auto" when a goal arrives)
"""

from __future__ import annotations

from collections import deque

import numpy as np
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String

from wiloc.motion import WaypointController
from wiloc.planning import OccupancyGrid, PathFollower

from .common import load_floorplan, pose2d_from_odom, quat_to_yaw, run_node, yaw_to_quat


class WaypointDriver(Node):
    def __init__(self):
        super().__init__("waypoint_driver")
        for name, default in (
            ("floorplan", "office"),
            ("laps", 0),
            ("use_estimate", False),
            ("v_max", 0.22),
            ("start_delay", 3.0),
            ("mode", "auto"),
            ("goal_hold", 8.0),
            ("stuck_time", 2.0),
            ("watchdog_time", 20.0),
            ("entity", "wiloc_bot"),
            ("yield_dist", 0.9),
            ("yield_max", 6.0),
        ):
            self.declare_parameter(name, default)
        gp = lambda n: self.get_parameter(n).value
        self.fp = load_floorplan(gp("floorplan"))
        self.grid = OccupancyGrid.from_floorplan(self.fp)
        self.patrol = [np.asarray(p, dtype=float) for p in self.fp.waypoints]
        self.laps = int(gp("laps"))
        self.ctrl = WaypointController(v_max=float(gp("v_max")))
        self.mode = str(gp("mode"))
        self.delay = float(gp("start_delay"))
        self.goal_hold = float(gp("goal_hold"))
        self.stuck_time = float(gp("stuck_time"))
        self.watchdog_time = float(gp("watchdog_time"))
        self.entity = str(gp("entity"))
        self.yield_dist = float(gp("yield_dist"))
        self.yield_max = float(gp("yield_max"))
        self.yield_since = None  # giving way to a mover since (sim time)
        self.ignore_movers_until = 0.0

        self.pose = None
        self.t0 = None  # first tick once sim time (/clock) is running
        self.pi, self.laps_done, self.done = 0, 0, False
        self.goal = None  # clicked goal (x, y), or None when patrolling
        self.hold_until = None  # after reaching a goal: wait here until this time
        self.follower: PathFollower | None = None
        self.replan = True
        self.back_until = None  # backing up after getting stuck
        self.motion = deque()  # (t, x, y, v_cmd) for stuck detection
        self.stuck_since = None
        self.n_stuck = self.n_rescues = 0

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.plan_pub = self.create_publisher(Path, "/wiloc/plan", 5)
        self.status_pub = self.create_publisher(String, "/wiloc/driver_status", 10)
        self.mode_pub = self.create_publisher(String, "/wiloc/drive_mode", 10)
        if gp("use_estimate"):
            self.create_subscription(PoseWithCovarianceStamped, "/wiloc/pose", self.on_estimate, 10)
        else:
            self.create_subscription(Odometry, "/ground_truth", self.on_truth, 10)
        self.create_subscription(String, "/wiloc/drive_mode", self.on_mode, 10)
        self.create_subscription(PoseStamped, "/wiloc/goal", self.on_goal, 10)
        self.teleport = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.create_timer(0.05, self.on_timer)
        self.create_timer(0.5, self.publish_status)
        self.create_timer(1.0, self.publish_plan)

    # ------------------------------------------------------------ inputs
    def _set_pose(self, pose):
        if self.pose is not None and np.hypot(*(pose[:2] - self.pose[:2])) > 1.0:
            self.get_logger().info("pose jumped (kidnapped?): replanning")
            self.replan = True
            self.motion.clear()
            self.stuck_since = None
        self.pose = pose

    def on_truth(self, msg):
        self._set_pose(np.array(pose2d_from_odom(msg)))

    def on_estimate(self, msg):
        p = msg.pose.pose
        self._set_pose(np.array([p.position.x, p.position.y, quat_to_yaw(p.orientation)]))

    def on_mode(self, msg):
        mode = msg.data.strip().lower()
        if mode not in ("auto", "manual") or mode == self.mode:
            return
        self.mode = mode
        self.get_logger().info(f"drive mode -> {mode}")
        if mode == "manual":
            self.goal = self.hold_until = self.back_until = None
            self.pub.publish(Twist())  # hand over from a standstill
        else:
            self.done = False
        self.replan = True
        self.motion.clear()
        self.stuck_since = None

    def on_goal(self, msg):
        xy = np.array([msg.pose.position.x, msg.pose.position.y])
        self.goal = self.grid.nearest_free(xy)
        self.hold_until = self.back_until = None
        self.replan = True
        self.get_logger().info(f"goal ({self.goal[0]:.1f}, {self.goal[1]:.1f})")
        if self.mode != "auto":
            self.mode = "auto"
            self.mode_pub.publish(String(data="auto"))

    # ------------------------------------------------------------ planning
    def _target(self):
        return self.goal if self.goal is not None else self.patrol[self.pi]

    def _make_plan(self):
        self.replan = False
        for _ in range(len(self.patrol) + 1):
            path = self.grid.plan(self.pose[:2], self._target())
            if path is not None:
                self.follower = PathFollower(path, self.ctrl)
                self.publish_plan()
                return
            if self.goal is not None:
                self.get_logger().warn("clicked goal is unreachable; resuming the patrol")
                self.goal = None
            else:
                self.get_logger().warn(f"patrol waypoint {self.pi + 1} unreachable; skipping it")
                self._advance_patrol()
        self.follower = None

    def _advance_patrol(self):
        self.pi += 1
        if self.pi >= len(self.patrol):
            self.pi = 0
            self.laps_done += 1
            if self.laps and self.laps_done >= self.laps:
                self.done = True
                self.get_logger().info("patrol complete")

    # ------------------------------------------------------------ control
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_timer(self):
        t = self.now()
        if self.t0 is None:
            if t == 0.0:
                return  # /clock not received yet
            self.t0 = t
        if self.mode != "auto" or self.pose is None or t - self.t0 < self.delay or not self.patrol:
            return
        cmd = Twist()
        if self.back_until is not None:
            if t < self.back_until:
                cmd.linear.x = -0.12
                self.pub.publish(cmd)
                return
            self.back_until = None
            self.replan = True
        if self.done and self.goal is None:
            self.pub.publish(cmd)
            return
        if self.hold_until is not None:
            if t < self.hold_until:
                self.pub.publish(cmd)
                return
            self.hold_until = None
            self.replan = True  # resume the patrol from here
        if self.replan or self.follower is None:
            self._make_plan()
            if self.follower is None:
                self.pub.publish(cmd)
                return
        v, w, finished = self.follower.command(self.pose)
        if v > 0:
            v, w = self._give_way(t, v, w)
        if finished:
            if self.goal is not None:
                self.get_logger().info("goal reached")
                self.goal = None
                self.hold_until = t + self.goal_hold
            else:
                self._advance_patrol()
                self.replan = True
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.pub.publish(cmd)
        self._check_stuck(t, v)

    def _give_way(self, t, v, w):
        """Slow down for movers ahead; stop while one is close in front (bounded by ``yield_max``)."""
        if not len(self.fp.movers) or t < self.ignore_movers_until:
            self.yield_since = None
            return v, w
        x, y, th = self.pose
        ahead = None
        for mv, p in self.fp.movers.near((x, y), t, 2.0 * self.yield_dist):
            dx, dy = p[0] - x, p[1] - y
            bearing = np.arctan2(dy, dx) - th
            if abs((bearing + np.pi) % (2 * np.pi) - np.pi) < np.radians(55):
                d = float(np.hypot(dx, dy))
                ahead = d if ahead is None else min(ahead, d)
        if ahead is None:
            self.yield_since = None
            return v, w
        if ahead > self.yield_dist:
            return 0.5 * v, w  # someone ahead: slow down
        self.yield_since = self.yield_since or t
        if t - self.yield_since > self.yield_max:  # they are not moving out of the way: carry on
            self.ignore_movers_until = t + 4.0
            self.yield_since = None
            return v, w
        return 0.0, 0.0

    def _check_stuck(self, t, v):
        self.motion.append((t, self.pose[0], self.pose[1], v))
        while self.motion and t - self.motion[0][0] > self.stuck_time:
            self.motion.popleft()
        if len(self.motion) < 10 or t - self.motion[0][0] < 0.9 * self.stuck_time:
            return
        m = np.array(self.motion)
        moved = np.hypot(m[-1, 1] - m[0, 1], m[-1, 2] - m[0, 2])
        if np.all(m[:, 3] > 0.05) and moved < 0.05:
            self.n_stuck += 1
            self.stuck_since = self.stuck_since or t
            x, y, th = self.pose
            self.get_logger().warn(
                f"stuck at ({x:.2f}, {y:.2f}, {np.degrees(th):.0f} deg), commanded forward but moved "
                f"{moved * 100:.0f} cm: backing up")
            self.back_until = t + 1.5
            self.motion.clear()
            if t - self.stuck_since > self.watchdog_time:
                self.rescue()
        elif moved > 0.2:
            self.stuck_since = None

    def rescue(self):
        """Last resort for a live demo: teleport to the nearest free patrol waypoint."""
        if not self.teleport.service_is_ready():
            return
        pts = np.array(self.patrol)
        x, y = self.grid.nearest_free(pts[np.argmin(np.linalg.norm(pts - self.pose[:2], axis=1))])
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name, req.state.reference_frame = self.entity, "world"
        req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = float(x), float(y), 0.02
        req.state.pose.orientation = yaw_to_quat(float(self.pose[2]))
        self.teleport.call_async(req)
        self.n_rescues += 1
        self.stuck_since = None
        self.get_logger().warn(f"watchdog: robot wedged for {self.watchdog_time:.0f} s, moved to ({x:.1f}, {y:.1f})")

    # ------------------------------------------------------------ outputs
    def publish_status(self):
        if self.mode == "manual":
            text = "manual (keyboard)"
        elif self.back_until is not None:
            text = "auto: stuck, backing up"
        elif self.yield_since is not None:
            text = "auto: giving way"
        elif self.goal is not None:
            text = f"auto: going to ({self.goal[0]:.1f}, {self.goal[1]:.1f})"
        elif self.hold_until is not None:
            text = "auto: arrived"
        elif self.done:
            text = "auto: patrol complete"
        else:
            text = f"auto: waypoint {self.pi + 1}/{len(self.patrol)}"
        self.status_pub.publish(String(data=text))

    def publish_plan(self):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        if self.follower is not None and self.mode == "auto" and not self.follower.done:
            pts = [self.pose[:2]] if self.pose is not None else []
            for x, y in [*pts, *self.follower.remaining]:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x, ps.pose.position.y = float(x), float(y)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
        self.plan_pub.publish(path)


def main():
    run_node(WaypointDriver, on_shutdown=lambda n: n.pub.publish(Twist()))
