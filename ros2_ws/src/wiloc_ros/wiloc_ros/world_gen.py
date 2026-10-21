"""Generate a Gazebo (classic) world from a WiLoc floor plan.

The same YAML file drives the radio simulation and the physical world, so the
walls the robot bumps into are exactly the walls that attenuate the signal.
"""

from __future__ import annotations

import argparse
import math

from wiloc.environment import FloorPlan

from . import theme

def _material(hex_color, alpha=1.0, emissive=0.0):
    r, g, b = theme.rgb(hex_color)
    e = f"{r * emissive:.3f} {g * emissive:.3f} {b * emissive:.3f} 1"
    return (
        f"<material><ambient>{r:.3f} {g:.3f} {b:.3f} {alpha}</ambient>"
        f"<diffuse>{r:.3f} {g:.3f} {b:.3f} {alpha}</diffuse>"
        f"<specular>0.1 0.1 0.1 1</specular><emissive>{e}</emissive></material>"
    )


def _wall_model(i, seg, material, height, thickness):
    x1, y1, x2, y2 = seg
    length = math.hypot(x2 - x1, y2 - y1) + thickness  # overlap at corners
    cx, cy, yaw = (x1 + x2) / 2, (y1 + y2) / 2, math.atan2(y2 - y1, x2 - x1)
    glass = material == "glass"
    mat = _material(theme.MATERIALS.get(material, theme.TEXT), alpha=0.45 if glass else 1.0)
    transparency = "<transparency>0.55</transparency>" if glass else ""
    size = f"{length:.3f} {thickness:.3f} {height:.3f}"
    return f"""
    <model name="wall_{i}_{material}">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} {height / 2:.3f} 0 0 {yaw:.4f}</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>{size}</size></box></geometry>
          {mat}{transparency}
        </visual>
      </link>
    </model>"""


def _ap_model(i, ap, height):
    color = theme.ap_color(i)
    return f"""
    <model name="ap_{ap.id}">
      <static>true</static>
      <pose>{ap.x:.3f} {ap.y:.3f} {height - 0.1:.3f} 0 0 0</pose>
      <link name="link">
        <visual name="body">
          <geometry><cylinder><radius>0.16</radius><length>0.06</length></cylinder></geometry>
          {_material(color, emissive=0.6)}
        </visual>
        <visual name="halo">
          <pose>0 0 -0.02 0 0 0</pose>
          <geometry><cylinder><radius>0.45</radius><length>0.01</length></cylinder></geometry>
          {_material(color, alpha=0.35, emissive=0.3)}
          <transparency>0.6</transparency>
        </visual>
      </link>
    </model>"""


def _floor_model(fp):
    return f"""
    <model name="floor_slab">
      <static>true</static>
      <pose>{fp.width / 2:.3f} {fp.height / 2:.3f} 0.001 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{fp.width:.3f} {fp.height:.3f} 0.002</size></box></geometry>
          {_material("#cfd6df")}
        </visual>
      </link>
    </model>"""


def world_sdf(fp: FloorPlan) -> str:
    walls = "".join(
        _wall_model(i, seg, m, fp.wall_height, fp.wall_thickness) for i, (seg, m) in enumerate(zip(fp.walls, fp.wall_materials))
    )
    aps = "".join(_ap_model(i, ap, fp.wall_height) for i, ap in enumerate(fp.access_points)) + _floor_model(fp)
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
    <!-- keys pressed in the Gazebo window -> /wiloc/gazebo_keys (driving from Gazebo) -->
    <plugin name="wiloc_key_forwarder" filename="libwiloc_key_forwarder.so"/>
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>1.0</update_rate>
    </plugin>
    <scene>
      <ambient>0.6 0.62 0.66 1</ambient>
      <background>0.93 0.95 0.97 1</background>
      <shadows>false</shadows>
    </scene>
    <gui>
      <plugin name="keyboard" filename="libKeyboardGUIPlugin.so"/>
      <camera name="user_camera">
        <pose>{cx:.2f} {cy - 10.5:.2f} 17.5 0 0.98 1.5708</pose>
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
