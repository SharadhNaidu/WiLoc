"""WiLoc simulation: Gazebo world + diff-drive robot + Wi-Fi + localisation.

Normally the WiLoc Studio window starts this itself (``./wiloc.sh run``, or
``ros2 run wiloc_ros studio``) and restarts it to switch venues.  Directly:

    ros2 launch wiloc_ros sim.launch.py                       # simulation + Studio window (this venue only)
    ros2 launch wiloc_ros sim.launch.py floorplan:=hospital ap_layout:=staggered
    ros2 launch wiloc_ros sim.launch.py ui:=none              # headless (benchmarks, checks)
    ros2 launch wiloc_ros sim.launch.py gui:=true rviz:=true  # also Gazebo's own window and RViz
    ros2 launch wiloc_ros sim.launch.py method:=ekf global_init:=true
"""

import configparser
import os
import re
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _write_gzclient_geometry(rect=None, min_w=400, min_h=300):
    """Give gzclient a sensible window size in ~/.gazebo/gui.ini.

    A size is only written when none is stored yet (or ``rect`` forces one):
    on a first run under WSLg the file has no width/height and the window
    opens as a thin strip.
    """
    path = os.path.expanduser("~/.gazebo/gui.ini")
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if not cfg.has_section("geometry"):
        cfg.add_section("geometry")
    geo = cfg["geometry"]
    if rect is None:
        if int(geo.get("width", 0) or 0) >= min_w and int(geo.get("height", 0) or 0) >= min_h:
            return
        rect = (int(geo.get("x", 60) or 60), int(geo.get("y", 60) or 60), 1600, 1000)
    geo["x"], geo["y"], geo["width"], geo["height"] = (str(v) for v in rect)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        cfg.write(f, space_around_delimiters=False)


def _rviz_config(template: str, fp, rect, panel: bool) -> str:
    """Copy of the RViz config with window geometry and a view fitted to the floor plan."""
    x, y, w, h = rect
    with open(template) as f:
        text = f.read()
    # Top-down view: fit the building (plus margins for the title/status text).
    scale = min(w / (fp.width + 2.0), (h - 60) / (fp.height + 4.0))
    text = re.sub(r"(\n      X: )[-\d.]+", rf"\g<1>{fp.width / 2:.2f}", text, count=1)
    text = re.sub(r"(\n      Y: )[-\d.]+", rf"\g<1>{fp.height / 2 - 0.3:.2f}", text, count=1)
    text = re.sub(r"(\n      Scale: )[-\d.]+", rf"\g<1>{scale:.1f}", text, count=1)
    geometry = (
        "Window Geometry:\n"
        "  Displays:\n"
        f"    collapsed: {'false' if panel else 'true'}\n"
        f"  Hide Left Dock: {'false' if panel else 'true'}\n"
        "  Hide Right Dock: true\n"
        f"  X: {x}\n  Y: {y}\n  Width: {w}\n  Height: {h}\n"
    )
    text = re.sub(r"Window Geometry:\n(?:  .*\n?)*", geometry, text)
    out = os.path.join(tempfile.gettempdir(), f"wiloc_{fp.name}.rviz")
    with open(out, "w") as f:
        f.write(text)
    return out


def _setup(context):
    from wiloc_ros.common import load_floorplan
    from wiloc_ros.theme import QSS
    from wiloc_ros.world_gen import world_sdf

    lc = lambda n: LaunchConfiguration(n).perform(context)
    is_true = lambda n: lc(n).lower() == "true"
    floorplan = lc("floorplan")
    fp = load_floorplan(floorplan, lc("ap_layout"))
    world_path = os.path.join(tempfile.gettempdir(), f"wiloc_{fp.name}.world")
    with open(world_path, "w") as f:
        f.write(world_sdf(fp))

    if is_true("gui"):
        _write_gzclient_geometry()

    share = get_package_share_directory("wiloc_ros")
    xacro_file = os.path.join(share, "urdf", "wiloc_bot.urdf.xacro")
    rviz_cfg = _rviz_config(os.path.join(share, "rviz", "wiloc.rviz"), fp, (60, 60, 1500, 900), panel=is_true("rviz_panel"))
    gazebo_launch = os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
    sim_time = {"use_sim_time": True}
    common = {"floorplan": floorplan, "ap_layout": fp.layout, **sim_time}
    x, y, yaw = fp.start_pose
    qss_path = os.path.join(tempfile.gettempdir(), "wiloc_theme.qss")
    with open(qss_path, "w") as f:
        f.write(QSS)
    # Radio overrides for every node that simulates or models Wi-Fi (empty
    # launch argument: the venue's ``radio:`` value, see common.radio_config).
    radio = {
        "radio_seed": int(lc("radio_seed") or -1),
        "shadowing_sigma_db": float(lc("shadowing_sigma_db") or -1.0),
        "temporal_sigma_db": float(lc("temporal_sigma_db") or -1.0),
    }
    # A crashed node comes back by itself (a respawned localiser re-localises
    # globally, see localization_node); a live demo must not need a restart.
    respawn = {"respawn": True, "respawn_delay": 2.0}
    env = [
        # Never reach out to the online model database: venues may be offline,
        # and with no network Gazebo would stall for a long timeout first.
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
    ]
    if is_true("localhost_only"):
        # Ignore other ROS 2 users on the same network (another stand, a lab).
        env.append(SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"))
    else:
        env.append(SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "0"))

    return env + [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={"world": world_path, "gui": lc("gui"), "verbose": "false"}.items(),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": ParameterValue(Command(["xacro ", xacro_file]), value_type=str), **sim_time}],
        ),
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=["-timeout", "120", "-topic", "robot_description", "-entity", "wiloc_bot", "-x", str(x), "-y", str(y), "-z", "0.02", "-Y", str(yaw)],
            output="screen",
        ),
        Node(
            package="wiloc_ros",
            executable="wifi_sensor",
            parameters=[{**common, **radio, "scan_rate": float(lc("scan_rate"))}],
            output="screen",
            **respawn,
        ),
        Node(
            package="wiloc_ros",
            executable="camera_rig",
            parameters=[common],
            **respawn,
        ),
        Node(
            package="wiloc_ros",
            executable="odom_noise",
            parameters=[{**sim_time, "rot_bias": float(lc("odom_rot_bias")), "trans_bias": float(lc("odom_trans_bias"))}],
            **respawn,
        ),
        Node(
            package="wiloc_ros",
            executable="localization",
            parameters=[
                {
                    **common,
                    "method": lc("method"),
                    "n_particles": int(lc("n_particles")),
                    "kld": is_true("kld"),
                    "estimator": lc("estimator"),
                    "reseed_fraction": float(lc("reseed")),
                    "sensor_reset": float(lc("sensor_reset")),
                    "global_init": is_true("global_init"),
                    "radio_map": lc("radio_map"),
                    **radio,
                }
            ],
            output="screen",
            **respawn,
        ),
        Node(
            package="wiloc_ros",
            executable="evaluator",
            parameters=[{**common, "method": lc("method"), "output_csv": lc("output_csv")}],
            output="screen",
            **respawn,
        ),
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package="wiloc_ros",
                    executable="waypoint_driver",
                    parameters=[
                        {
                            **common,
                            "laps": int(lc("laps")),
                            "use_estimate": is_true("use_estimate"),
                            "mode": "auto" if is_true("drive") else "manual",
                        }
                    ],
                    output="screen",
                    **respawn,
                )
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_cfg, "-stylesheet", qss_path],
            parameters=[sim_time],
            condition=IfCondition(lc("rviz")),
        ),
        # The Studio normally starts the simulation itself (ros2 run wiloc_ros studio);
        # ui:=studio runs it inside this launch instead, attached to this simulation.
        Node(
            package="wiloc_ros",
            executable="studio",
            arguments=["--attach", "--venue", fp.name, "--layout", fp.layout, "--windowed"],
            output="screen",
            condition=IfCondition(str(lc("ui") == "studio")),
        ),
        Node(package="rqt_graph", executable="rqt_graph", condition=IfCondition(lc("rqt_graph"))),
    ]


def generate_launch_description():
    default_localhost = os.environ.get("ROS_LOCALHOST_ONLY")
    if default_localhost is None:
        is_wsl = False
        try:
            with open("/proc/version", "r") as f:
                is_wsl = "microsoft" in f.read().lower()
        except OSError:
            pass
        default_localhost = "false" if is_wsl else "true"
    else:
        default_localhost = "true" if default_localhost == "1" else "false"

    args = [
        ("floorplan", "office", "venue: office | home | hospital | warehouse | library | factory, or a YAML file"),
        ("ap_layout", "", "router preset of the venue (empty: as installed); switch live on /wiloc/ap_layout"),
        ("ui", "studio", "window: studio (attached to this simulation) | none (headless)"),
        ("gui", "false", "also open Gazebo's own window (the Studio already shows the 3-D view)"),
        ("rviz", "false", "also start RViz"),
        ("rviz_panel", "false", "show RViz's Displays panel"),
        ("rqt_graph", "false", "also open rqt_graph (ROS's own node graph)"),
        ("method", "pf", "primary method: pf | ekf | wknn (all three always run, for the live race)"),
        ("n_particles", "1500", "particle count when kld:=false"),
        ("kld", "true", "adapt the particle count (KLD sampling, 500-5000)"),
        ("estimator", "cluster", "particle-filter estimate: cluster (heaviest mode) | meanshift"),
        ("reseed", "0.3", "share of particles re-spread after the filter is lost for 3 s (0: never)"),
        ("sensor_reset", "0.5", "share of injected particles drawn around the Wi-Fi fix (0: uniform)"),
        ("global_init", "false", "start with no knowledge of the initial pose"),
        ("radio_map", "", "radio map .npz (empty: survey at start-up)"),
        ("radio_seed", "", "seed of the radio environment (shadowing field); empty: the venue's"),
        ("shadowing_sigma_db", "", "shadowing std-dev in dB; empty: the venue's"),
        ("temporal_sigma_db", "", "scan-to-scan RSSI noise std-dev in dB; empty: the venue's"),
        ("localhost_only", default_localhost, "keep ROS 2 traffic on this machine (ROS_LOCALHOST_ONLY=1)"),
        ("scan_rate", "2.0", "Wi-Fi scans per second"),
        ("odom_rot_bias", "0.04", "relative heading scale error of odometry"),
        ("odom_trans_bias", "0.015", "relative distance scale error of odometry"),
        ("drive", "true", "start the waypoint patrol (false: start in manual/keyboard mode)"),
        ("laps", "0", "patrol laps (0: patrol forever)"),
        ("use_estimate", "false", "patrol steers on the WiLoc estimate instead of ground truth"),
        ("output_csv", os.path.expanduser("~/.ros/wiloc_eval.csv"), "evaluation log"),
    ]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args] + [OpaqueFunction(function=_setup)])
