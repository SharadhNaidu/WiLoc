"""Small helpers shared by the WiLoc ROS 2 nodes."""

from __future__ import annotations

import math

from geometry_msgs.msg import Quaternion

from wiloc.environment import FloorPlan
from wiloc.radio import RadioConfig

# Radio overrides every node that simulates or models Wi-Fi accepts; a negative
# value means "use the venue's ``radio:`` setting".
RADIO_PARAMS = {"radio_seed": -1, "shadowing_sigma_db": -1.0, "temporal_sigma_db": -1.0}


def load_floorplan(name_or_path: str) -> FloorPlan:
    if name_or_path.endswith((".yaml", ".yml")):
        return FloorPlan.load(name_or_path)
    return FloorPlan.builtin(name_or_path)


def radio_config(node, fp: FloorPlan) -> RadioConfig:
    """The venue's radio settings with the node's overrides applied.

    The Wi-Fi sensor, the localiser's survey and the dashboard's heat map all
    go through here, so they always simulate the same radio world.
    """
    for name, default in RADIO_PARAMS.items():
        if not node.has_parameter(name):
            node.declare_parameter(name, default)
    val = {name: node.get_parameter(name).value for name in RADIO_PARAMS}
    pick = lambda name, cast: cast(val[name]) if val[name] is not None and val[name] >= 0 else None
    return RadioConfig.from_dict(
        fp.radio,
        seed=pick("radio_seed", int),
        shadowing_sigma_db=pick("shadowing_sigma_db", float),
        temporal_sigma_db=pick("temporal_sigma_db", float),
    )


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
            except (KeyboardInterrupt, Exception):  # e.g. context already invalid
                pass
