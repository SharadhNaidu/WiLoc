"""Generate a Gazebo (classic) world from a WiLoc floor plan.

The same YAML file drives the radio simulation and the physical world, so the
walls the robot bumps into are exactly the walls that attenuate the signal.
"""

from __future__ import annotations

import argparse
import math

from wiloc.environment import FloorPlan

MATERIAL_SCRIPTS = {
    "concrete": "Gazebo/Grey",
    "brick": "Gazebo/Bricks",
    "drywall": "Gazebo/White",
    "glass": "Gazebo/BlueTransparent",
    "wood": "Gazebo/Wood",
}


def _wall_model(i, seg, material, height, thickness):
    x1, y1, x2, y2 = seg
    length = math.hypot(x2 - x1, y2 - y1) + thickness  # overlap at corners
    cx, cy, yaw = (x1 + x2) / 2, (y1 + y2) / 2, math.atan2(y2 - y1, x2 - x1)
    mat = MATERIAL_SCRIPTS.get(material, "Gazebo/White")
    size = f"{length:.3f} {thickness:.3f} {height:.3f}"
    return f"""
    <model name="wall_{i}_{material}">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} {height / 2:.3f} 0 0 {yaw:.4f}</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>{size}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{mat}</name></script></material>
        </visual>
      </link>
    </model>"""


def _ap_model(ap, height):
    return f"""
    <model name="ap_{ap.id}">
      <static>true</static>
      <pose>{ap.x:.3f} {ap.y:.3f} {height - 0.1:.3f} 0 0 0</pose>
      <link name="link">
        <visual name="body">
          <geometry><cylinder><radius>0.12</radius><length>0.05</length></cylinder></geometry>
          <material><ambient>0.85 0.1 0.1 1</ambient><diffuse>0.85 0.1 0.1 1</diffuse><emissive>0.4 0 0 1</emissive></material>
        </visual>
      </link>
    </model>"""


def world_sdf(fp: FloorPlan) -> str:
    walls = "".join(
        _wall_model(i, seg, m, fp.wall_height, fp.wall_thickness) for i, (seg, m) in enumerate(zip(fp.walls, fp.wall_materials))
    )
    aps = "".join(_ap_model(ap, fp.wall_height) for ap in fp.access_points)
    cx, cy = fp.width / 2, fp.height / 2
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="wiloc_{fp.name}">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gui>
      <camera name="overview">
        <pose>{cx:.2f} {cy - 9:.2f} 18 0 1.05 1.5708</pose>
      </camera>
    </gui>{walls}{aps}
  </world>
</sdf>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--floorplan", default="office", help="built-in name or path to YAML")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    fp = FloorPlan.builtin(a.floorplan) if not a.floorplan.endswith((".yaml", ".yml")) else FloorPlan.load(a.floorplan)
    with open(a.out, "w") as f:
        f.write(world_sdf(fp))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
