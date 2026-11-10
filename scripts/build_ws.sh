#!/usr/bin/env bash
# Build the WiLoc ROS 2 workspace in WSL / Ubuntu 22.04 (ROS 2 Humble).
# Sources live in this repo; build artefacts go to ~/wiloc_ws so they never
# land on a Windows (NTFS) drive.
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WILOC_WS:-$HOME/wiloc_ws}"

if [[ "$REPO" =~ [[:space:]] ]]; then
    echo -e "\033[1;33m[wiloc warning]\033[0m Path '$REPO' contains spaces."
    echo -e "ROS 2 colcon/rosidl generators can fail on paths with spaces. If you encounter build issues, clone or copy WiLoc to a path without spaces (such as ~/WiLoc)."
fi

mkdir -p "$WS/src"
ln -sfn "$REPO"                         "$WS/src/wiloc"
ln -sfn "$REPO/ros2_ws/src/wiloc_msgs"  "$WS/src/wiloc_msgs"
ln -sfn "$REPO/ros2_ws/src/wiloc_ros"   "$WS/src/wiloc_ros"
rm -f "$WS/src/wiloc_gazebo"  # old key-forwarding plugin, no longer used
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
cd "$WS"
colcon build --symlink-install --base-paths src/wiloc src/wiloc_msgs src/wiloc_ros "$@"
echo
echo "Built. Now run:  ./wiloc.sh run"
