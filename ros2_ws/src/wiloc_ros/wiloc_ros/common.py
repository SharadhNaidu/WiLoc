"""Small helpers shared by the WiLoc ROS 2 nodes."""

from __future__ import annotations

import math

from geometry_msgs.msg import Quaternion

from wiloc.environment import FloorPlan


def load_floorplan(name_or_path: str) -> FloorPlan:
    if name_or_path.endswith((".yaml", ".yml")):
        return FloorPlan.load(name_or_path)
    return FloorPlan.builtin(name_or_path)


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quat_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def pose2d_from_odom(msg):
    p = msg.pose.pose
    return (p.position.x, p.position.y, quat_to_yaw(p.orientation))


def run_node(node_cls, on_shutdown=None):
    """Spin a node until Ctrl-C, then clean up even if more SIGINTs arrive.

    ``ros2 launch`` may deliver several interrupts on shutdown; without this,
    a second KeyboardInterrupt aborts cleanup (and e.g. loses the evaluation
    log).
    """
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = node_cls()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for step in (lambda: on_shutdown and on_shutdown(node), node.destroy_node, rclpy.try_shutdown):
            try:
                step()
            except KeyboardInterrupt:
                pass
