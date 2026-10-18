"""WiLoc full simulation: Gazebo world + diff-drive robot + Wi-Fi + localisation.

    ros2 launch wiloc_ros sim.launch.py                    # GUI + RViz, particle filter
    ros2 launch wiloc_ros sim.launch.py method:=ekf
    ros2 launch wiloc_ros sim.launch.py global_init:=true  # kidnapped / unknown start
    ros2 launch wiloc_ros sim.launch.py gui:=false rviz:=false   # headless benchmark
"""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _setup(context):
    from wiloc_ros.common import load_floorplan
    from wiloc_ros.world_gen import world_sdf

    lc = lambda n: LaunchConfiguration(n).perform(context)
    floorplan = lc("floorplan")
    fp = load_floorplan(floorplan)
    world_path = os.path.join(tempfile.gettempdir(), f"wiloc_{fp.name}.world")
    with open(world_path, "w") as f:
        f.write(world_sdf(fp))

    share = get_package_share_directory("wiloc_ros")
    xacro_file = os.path.join(share, "urdf", "wiloc_bot.urdf.xacro")
    gazebo_launch = os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
    sim_time = {"use_sim_time": True}
    common = {"floorplan": floorplan, **sim_time}
    x, y, yaw = fp.start_pose

    return [
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
            arguments=["-topic", "robot_description", "-entity", "wiloc_bot", "-x", str(x), "-y", str(y), "-z", "0.02", "-Y", str(yaw)],
            output="screen",
        ),
        Node(
            package="wiloc_ros",
            executable="wifi_sensor",
            parameters=[{**common, "radio_seed": int(lc("radio_seed")), "scan_rate": float(lc("scan_rate"))}],
            output="screen",
        ),
        Node(
            package="wiloc_ros",
            executable="odom_noise",
            parameters=[{**sim_time, "rot_bias": float(lc("odom_rot_bias")), "trans_bias": float(lc("odom_trans_bias"))}],
        ),
        Node(
            package="wiloc_ros",
            executable="localization",
            parameters=[
                {
                    **common,
                    "method": lc("method"),
                    "n_particles": int(lc("n_particles")),
                    "global_init": lc("global_init").lower() == "true",
                    "radio_map": lc("radio_map"),
                    "radio_seed": int(lc("radio_seed")),
                }
            ],
            output="screen",
        ),
        Node(package="wiloc_ros", executable="evaluator", parameters=[{**common, "output_csv": lc("output_csv")}], output="screen"),
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package="wiloc_ros",
                    executable="waypoint_driver",
                    parameters=[{**common, "laps": int(lc("laps"))}],
                    output="screen",
                    condition=IfCondition(lc("drive")),
                )
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(share, "rviz", "wiloc.rviz")],
            parameters=[sim_time],
            condition=IfCondition(lc("rviz")),
        ),
    ]


def generate_launch_description():
    args = [
        ("floorplan", "office", "built-in floor plan name or path to a YAML file"),
        ("gui", "true", "start the Gazebo client"),
        ("rviz", "true", "start RViz"),
        ("method", "pf", "pf | ekf | wknn"),
        ("n_particles", "1500", "particle count"),
        ("global_init", "false", "start with no knowledge of the initial pose"),
        ("radio_map", "", "radio map .npz (empty: survey at start-up)"),
        ("radio_seed", "7", "seed of the radio environment (shadowing field)"),
        ("scan_rate", "2.0", "Wi-Fi scans per second"),
        ("odom_rot_bias", "0.04", "relative heading scale error of odometry"),
        ("odom_trans_bias", "0.015", "relative distance scale error of odometry"),
        ("drive", "true", "run the waypoint patrol"),
        ("laps", "1", "patrol laps"),
        ("output_csv", os.path.expanduser("~/.ros/wiloc_eval.csv"), "evaluation log"),
    ]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args] + [OpaqueFunction(function=_setup)])
