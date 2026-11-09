"""Small helpers shared by the WiLoc ROS 2 nodes."""

from __future__ import annotations

import math

from geometry_msgs.msg import Quaternion

from wiloc.environment import FloorPlan
from wiloc.radio import RadioConfig

# Radio overrides every node that simulates or models Wi-Fi accepts; a negative
# value means "use the venue's ``radio:`` setting".
RADIO_PARAMS = {"radio_seed": -1, "shadowing_sigma_db": -1.0, "temporal_sigma_db": -1.0}


def load_floorplan(name_or_path: str, layout: str = "") -> FloorPlan:
    """A built-in venue by name, or a YAML file; ``layout`` picks a router preset ("" = the first)."""
    fp = FloorPlan.load(name_or_path) if name_or_path.endswith((".yaml", ".yml")) else FloorPlan.builtin(name_or_path)
    return fp.with_layout(layout) if layout else fp


# Router preset switching: whoever changes the layout publishes its name here,
# latched, so a node that starts (or respawns) later still gets the current one.
AP_LAYOUT_TOPIC = "/wiloc/ap_layout"
# Routers switched off ("unplugged"): latched JSON list of AP ids.
AP_DISABLED_TOPIC = "/wiloc/ap_disabled"


def latched_qos(depth: int = 1):
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(depth=depth, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


def venue_from_params(node) -> tuple[FloorPlan, FloorPlan]:
    """(venue with all presets, plan with the ``ap_layout`` preset) from the node's parameters."""
    for name, default in (("floorplan", "office"), ("ap_layout", "")):
        if not node.has_parameter(name):
            node.declare_parameter(name, default)
    venue = load_floorplan(node.get_parameter("floorplan").value)
    layout = node.get_parameter("ap_layout").value
    return venue, venue.with_layout(layout) if layout else venue


def subscribe_ap_layout(node, venue: FloorPlan, callback):
    """Call ``callback(name)`` when another router preset of ``venue`` is published (unknown names are logged)."""
    from std_msgs.msg import String

    def on_msg(msg):
        name = msg.data.strip()
        if name not in venue.layouts:
            node.get_logger().warn(f"unknown router layout {name!r}; known: {list(venue.layouts)}")
            return
        callback(name)

    return node.create_subscription(String, AP_LAYOUT_TOPIC, on_msg, latched_qos())


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
