#!/usr/bin/env bash
# Build the WiLoc ROS 2 workspace in WSL / Ubuntu 22.04 (ROS 2 Humble).
# Sources live in this repo; build artefacts go to ~/wiloc_ws so they never
# land on a Windows (NTFS) drive.
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WILOC_WS:-$HOME/wiloc_ws}"
mkdir -p "$WS/src"
ln -sfn "$REPO"                         "$WS/src/wiloc"
ln -sfn "$REPO/ros2_ws/src/wiloc_msgs"  "$WS/src/wiloc_msgs"
ln -sfn "$REPO/ros2_ws/src/wiloc_ros"   "$WS/src/wiloc_ros"
ln -sfn "$REPO/ros2_ws/src/wiloc_gazebo" "$WS/src/wiloc_gazebo"
source /opt/ros/humble/setup.bash
cd "$WS"
colcon build --symlink-install --base-paths src/wiloc src/wiloc_msgs src/wiloc_ros src/wiloc_gazebo "$@"
echo
echo "Built. Now run:  source $WS/install/setup.bash && ros2 launch wiloc_ros sim.launch.py"
