import os
from glob import glob

from setuptools import find_packages, setup

package_name = "wiloc_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.xacro")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        # Inter (SIL Open Font License 1.1) for the Studio window
        (os.path.join("share", package_name, "fonts"), glob("fonts/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SharadhNaidu",
    maintainer_email="sharadhnaiduwork@gmail.com",
    description="ROS 2 / Gazebo integration of the WiLoc Wi-Fi localisation framework",
    license="MIT",
    entry_points={
        "console_scripts": [
            "wifi_sensor = wiloc_ros.wifi_sensor_node:main",
            "odom_noise = wiloc_ros.odom_noise_node:main",
            "localization = wiloc_ros.localization_node:main",
            "waypoint_driver = wiloc_ros.waypoint_driver:main",
            "evaluator = wiloc_ros.evaluator_node:main",
            "world_gen = wiloc_ros.world_gen:main",
            "studio = wiloc_ros.ui.app:main",
            "camera_rig = wiloc_ros.camera_rig:main",
            "check = wiloc_ros.check:run",
        ],
    },
)
