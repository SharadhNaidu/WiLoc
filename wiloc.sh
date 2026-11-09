#!/usr/bin/env bash
# WiLoc: one entry point for setting up, building and running the simulation.
#
#   ./wiloc.sh setup          install missing packages (sudo apt) and build
#   ./wiloc.sh build          build the ROS 2 workspace (~/wiloc_ws)
#   ./wiloc.sh run [args]     start Gazebo + the WiLoc window (args go to ros2 launch)
#   ./wiloc.sh run --software-gl ...   same, with software OpenGL (VMs without 3D acceleration)
#   ./wiloc.sh stop           stop a running simulation
#   ./wiloc.sh test           run the unit tests
#   ./wiloc.sh experiments    run the offline experiments (figures in ./results)
#
# Requires Ubuntu 22.04 with ROS 2 Humble (Gazebo 11 / gazebo_ros).
#
# Examples of launch arguments:
#   ./wiloc.sh run method:=ekf             EKF instead of the particle filter
#   ./wiloc.sh run drive:=true             start with the automatic patrol
#   ./wiloc.sh run global_init:=true       unknown start pose
#   ./wiloc.sh run rviz:=true              also open RViz
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${WILOC_WS:-$HOME/wiloc_ws}"
ROS_SETUP=/opt/ros/humble/setup.bash
# The simulation keeps ROS 2 traffic on this machine (launch arg localhost_only);
# so must every ros2 command run from here, or it would not see the topics.
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

say() { printf '\033[1;34m[wiloc]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[wiloc]\033[0m %s\n' "$*" >&2; exit 1; }

source_ros() {
    [ -f "$ROS_SETUP" ] || die "ROS 2 Humble not found at /opt/ros/humble. WiLoc needs Ubuntu 22.04 + ROS 2 Humble (Gazebo classic)."
    # shellcheck disable=SC1090
    source "$ROS_SETUP"
}

cmd_setup() {
    source_ros
    say "installing system packages (needs sudo)"
    sudo apt-get update
    sudo apt-get install -y \
        ros-humble-gazebo-ros-pkgs ros-humble-xacro ros-humble-robot-state-publisher \
        ros-humble-rviz2 ros-humble-rqt-graph python3-colcon-common-extensions \
        python3-numpy python3-scipy python3-matplotlib python3-yaml python3-pyqt5 python3-pytest x11-utils \
        python3-pyqtgraph python3-opengl python3-pyqt5.qtopengl fonts-inter mesa-utils
    cmd_build
    say "setup done. Start the simulation with:  ./wiloc.sh run"
}

cmd_build() {
    source_ros
    say "building workspace in $WS"
    WILOC_WS="$WS" bash "$REPO/scripts/build_ws.sh"
    say "preparing radio maps (cached, so the simulation starts fast)"
    (cd "$REPO" && PYTHONPATH="$REPO" python3 -m wiloc warm-cache)
}

cmd_stop() {
    pkill -f "ros2 launch wiloc_ros" 2>/dev/null || true
    pkill -f "lib/wiloc_ros/" 2>/dev/null || true
    pkill -x rviz2 2>/dev/null || true
    for _ in $(seq 20); do
        pgrep -x gzserver >/dev/null || pgrep -x gzclient >/dev/null || return 0
        pkill -9 -x gzserver 2>/dev/null || true
        pkill -9 -x gzclient 2>/dev/null || true
        sleep 1
    done
}

cmd_run() {
    if [ "$1" = "--software-gl" ]; then
        export LIBGL_ALWAYS_SOFTWARE=1
        shift
    fi
    [ -f "$WS/install/setup.bash" ] || cmd_build
    cmd_stop  # a leftover Gazebo would make the new one fail to start
    source_ros
    # shellcheck disable=SC1091
    source "$WS/install/setup.bash"
    say "starting Gazebo + WiLoc (Ctrl-C to stop)"
    say "drive: click Gazebo's 3D view and hold the arrow keys, or use W/A/S/D in the WiLoc window"
    exec ros2 launch wiloc_ros sim.launch.py drive:=false "$@"
}

cmd_test() {
    cd "$REPO"
    PYTHONPATH="$REPO" python3 -m pytest -q
}

cmd_experiments() {
    cd "$REPO"
    PYTHONPATH="$REPO" python3 -m wiloc all --out results "$@"
}

case "${1:-help}" in
    setup) cmd_setup ;;
    build) cmd_build ;;
    run) shift; cmd_run "$@" ;;
    stop) cmd_stop; say "stopped" ;;
    test) cmd_test ;;
    experiments) shift; cmd_experiments "$@" ;;
    *) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
