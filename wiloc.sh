#!/usr/bin/env bash
# WiLoc: one entry point for setting up, building and running the simulation.
#
#   ./wiloc.sh setup          install missing packages (sudo apt) and build
#   ./wiloc.sh build          build the ROS 2 workspace (~/wiloc_ws)
#   ./wiloc.sh run [options]  open WiLoc Studio, the one-window demo (it starts the simulation)
#   ./wiloc.sh demo           the same for the stand: auto tour when nobody touches it
#   ./wiloc.sh check          pre-flight check for the stand: headless run, tracking and a kidnap (~2 min)
#   ./wiloc.sh record [file]  record the backup video off screen (default results/wiloc_demo.mp4)
#   ./wiloc.sh launch [args]  the bare simulation (ros2 launch; ui:=none for headless)
#   ./wiloc.sh stop           stop everything (simulation and window)
#   ./wiloc.sh test           run the unit tests
#   ./wiloc.sh experiments    run the offline experiments (figures in ./results)
#
# Requires Ubuntu 22.04 with ROS 2 Humble (Gazebo 11 / gazebo_ros).
#
# Studio options:
#   --venue hospital --layout staggered    start in another venue / router preset
#   --windowed                             a window instead of full screen
#   --software-gl                          software OpenGL (VMs without 3D acceleration)
#   --launch-arg method:=ekf               any simulation argument (repeatable), e.g. global_init:=true
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${WILOC_WS:-$HOME/wiloc_ws}"
ROS_SETUP=/opt/ros/humble/setup.bash
# The simulation keeps ROS 2 traffic on this machine (launch arg localhost_only).
# In WSL2 or environments where loopback multicast is not routed, ROS_LOCALHOST_ONLY=1
# breaks discovery between the Studio window and the simulation. Default to 0 in WSL.
if [ -z "$ROS_LOCALHOST_ONLY" ]; then
    if grep -qi microsoft /proc/version 2>/dev/null; then
        export ROS_LOCALHOST_ONLY=0
    else
        export ROS_LOCALHOST_ONLY=1
    fi
fi

say() { printf '\033[1;34m[wiloc]\033[0m %b\n' "$*"; }
die() { printf '\033[1;31m[wiloc]\033[0m %b\n' "$*" >&2; exit 1; }

source_ros() {
    if [ ! -f "$ROS_SETUP" ]; then
        if command -v wsl.exe >/dev/null 2>&1 && wsl.exe -l -v 2>/dev/null | tr -d '\0' | grep -qi "Ubuntu-22.04"; then
            die "ROS 2 Humble not found in this environment ($(lsb_release -ds 2>/dev/null || echo Linux)).\nYou have 'Ubuntu-22.04' installed in WSL! Please switch to it by running:\n  wsl -d Ubuntu-22.04\nand run wiloc from inside that distribution."
        fi
        die "ROS 2 Humble not found at /opt/ros/humble. WiLoc needs Ubuntu 22.04 + ROS 2 Humble (Gazebo classic)."
    fi
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
        python3-numpy python3-scipy python3-matplotlib python3-yaml python3-pyqt5 python3-pytest python3-opencv \
        mesa-utils
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
    # anchored at the interpreter, so a shell whose command line merely mentions these paths survives
    pkill -f "^[^ ]*python3 [^ ]*/ros2 launch wiloc_ros" 2>/dev/null || true
    pkill -f "^[^ ]*python3 [^ ]*/lib/wiloc_ros/" 2>/dev/null || true  # the nodes and the Studio window
    pkill -x rviz2 2>/dev/null || true
    for _ in $(seq 20); do
        pgrep -x gzserver >/dev/null || pgrep -x gzclient >/dev/null || return 0
        pkill -9 -x gzserver 2>/dev/null || true
        pkill -9 -x gzclient 2>/dev/null || true
        sleep 1
    done
}

prepare() {
    # usage: prepare "$@"; then use "${ARGS[@]}" (--software-gl removed)
    ARGS=()
    for a in "$@"; do
        if [ "$a" = "--software-gl" ]; then export LIBGL_ALWAYS_SOFTWARE=1; else ARGS+=("$a"); fi
    done
    [ -f "$WS/install/setup.bash" ] || cmd_build
    cmd_stop  # a leftover Gazebo would make the new one fail to start
    source_ros
    # shellcheck disable=SC1091
    source "$WS/install/setup.bash"
}

cmd_run() {
    prepare "$@"
    say "starting WiLoc Studio: F11 full screen / window, Ctrl+Shift+Q quits"
    exec ros2 run wiloc_ros studio "${ARGS[@]}"
}

cmd_demo() {
    prepare "$@"
    say "starting WiLoc Studio for the stand: auto tour after 90 s without a visitor (Ctrl+Shift+A toggles)"
    exec ros2 run wiloc_ros studio --attract "${ARGS[@]}"
}

# The backup video: tour, kidnap and recovery, click-to-go, unplug a router, another venue.
RECORD_SCRIPT="30:view=follow;55:kidnap;95:view=overview;105:goal=17,9.5;140:unplug;170:unplug;180:venue=hospital;245:kidnap;290:layout=staggered;335:quit"

cmd_record() {
    local out="${1:-$REPO/results/wiloc_demo.mp4}"
    [ $# -gt 0 ] && shift
    prepare "$@"
    say "recording the demo off screen to $out (about 6 minutes)"
    QT_QPA_PLATFORM=offscreen WILOC_STUDIO_SCRIPT="$RECORD_SCRIPT" \
        ros2 run wiloc_ros studio --size 1920x1080 --record "$out" "${ARGS[@]}"
}

cmd_check() {
    prepare "$@"
    say "pre-flight check (headless, about 2 minutes)"
    ros2 run wiloc_ros check "${ARGS[@]}"
}

cmd_launch() {
    prepare "$@"
    exec ros2 launch wiloc_ros sim.launch.py "${ARGS[@]}"
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
    demo) shift; cmd_demo "$@" ;;
    record) shift; cmd_record "$@" ;;
    check) shift; cmd_check "$@" ;;
    launch) shift; cmd_launch "$@" ;;
    stop) cmd_stop; say "stopped" ;;
    test) cmd_test ;;
    experiments) shift; cmd_experiments "$@" ;;
    *) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
