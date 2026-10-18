"""WiLoc core library.

Installable with pip (``pip install -e .``) and also usable as an ament_python
package, so ``colcon build`` in the ROS 2 workspace makes ``import wiloc``
available to the ``wiloc_ros`` nodes.
"""

from setuptools import find_packages, setup

package_name = "wiloc"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=["wiloc", "wiloc.*"]),
    package_data={"wiloc": ["data/*.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["numpy", "scipy", "matplotlib", "pyyaml"],
    python_requires=">=3.10",
    zip_safe=False,
    description="Indoor localisation of a mobile robot by Wi-Fi RSSI fingerprinting (simulation framework)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["wiloc = wiloc.__main__:main"]},
)
