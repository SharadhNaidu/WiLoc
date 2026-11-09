"""Generate a Gazebo (classic) world from a WiLoc floor plan.

The same YAML file drives the radio simulation and the physical world, so the
walls the robot bumps into are exactly the walls that attenuate the signal.

The world is meant to look like a real place: walls in plain painted colours
(the radio material shows only in the 2-D map), a light floor in large tiles,
soft even lighting, and the plan's furniture built from primitives -- every piece a static model with one
collision box, the footprint the path planner avoids.
"""

from __future__ import annotations

import argparse
import math

from wiloc.environment import FloorPlan, Furniture

from . import theme

# What walls look like in 3-D (the 2-D map colours them by radio material instead).
WALL_LOOK = {
    "drywall": "#F1EDE6",
    "brick": "#DDD3C6",  # painted brick
    "concrete": "#BDB8B0",
    "glass": "#A9C7D8",
    "wood": "#A57B52",
    "metal": "#7D8790",
}
WOOD, WOOD_DARK, LAMINATE, FRAME = "#BC8E62", "#8A6240", "#EEEAE3", "#3A3D42"


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
    mat = _material(WALL_LOOK.get(material, "#E8E4DD"), alpha=0.35 if glass else 1.0)
    transparency = "<transparency>0.6</transparency>" if glass else ""
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


FLOOR, FLOOR_ALT, CORRIDOR = "#D9C6A8", "#D2BE9F", "#E4DED4"


def _floor_model(fp, tile=1.0):
    """Flat light-oak floor in ``tile``-metre tiles (a faint checker), lighter in corridors.

    Plain colours instead of Gazebo's photo textures, which render far too
    saturated: the scene should look calm next to the UI.
    """
    corridors = [r for name, r in fp.rooms if "corridor" in name.lower() or "aisle" in name.lower()]
    visuals = []
    nx, ny = math.ceil(fp.width / tile), math.ceil(fp.height / tile)
    for i in range(nx):
        for j in range(ny):
            w = min(tile, fp.width - i * tile)
            h = min(tile, fp.height - j * tile)
            cx, cy = i * tile + w / 2, j * tile + h / 2
            in_corridor = any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in corridors)
            color = CORRIDOR if in_corridor else (FLOOR if (i + j) % 2 else FLOOR_ALT)
            visuals.append(
                f"""
        <visual name="tile_{i}_{j}">
          <pose>{cx:.3f} {cy:.3f} 0.001 0 0 0</pose>
          <geometry><box><size>{w - 0.01:.3f} {h - 0.01:.3f} 0.002</size></box></geometry>
          {_material(color)}
        </visual>"""
            )
    return f"""
    <model name="floor_tiles">
      <static>true</static>
      <link name="link">{"".join(visuals)}
      </link>
    </model>"""


def _ground_model(fp):
    """Warm, light ground around the building (Gazebo's default plane is dark grey)."""
    size = 4 * max(fp.width, fp.height) + 40
    return f"""
    <model name="ground">
      <static>true</static>
      <pose>{fp.width / 2:.3f} {fp.height / 2:.3f} 0 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>{size} {size}</size></plane></geometry>
          <surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>{size} {size}</size></plane></geometry>
          {_material("#ECE7DF")}
        </visual>
      </link>
    </model>"""


# ---------------------------------------------------------------- furniture
class _Parts:
    """Visual primitives of one furniture model, in its local frame (origin: footprint centre, floor)."""

    def __init__(self):
        self.items = []

    def box(self, x, y, z, sx, sy, sz, color, alpha=1.0):
        self._add(x, y, z, f"<box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box>", color, alpha)

    def cyl(self, x, y, z, r, length, color):
        self._add(x, y, z, f"<cylinder><radius>{r:.3f}</radius><length>{length:.3f}</length></cylinder>", color)

    def sphere(self, x, y, z, r, color):
        self._add(x, y, z, f"<sphere><radius>{r:.3f}</radius></sphere>", color)

    def _add(self, x, y, z, geometry, color, alpha=1.0):
        n = len(self.items)
        self.items.append(
            f"""
        <visual name="v{n}">
          <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
          <geometry>{geometry}</geometry>
          {_material(color, alpha)}
        </visual>"""
        )


def _back_offset(back, w, d, inset):
    """Local (x, y) of a backrest centre and its (sx, sy) footprint."""
    return {
        "n": ((0.0, d / 2 - inset / 2), (w, inset)),
        "s": ((0.0, -d / 2 + inset / 2), (w, inset)),
        "e": ((w / 2 - inset / 2, 0.0), (inset, d)),
        "w": ((-w / 2 + inset / 2, 0.0), (inset, d)),
    }.get(back, ((0.0, d / 2 - inset / 2), (w, inset)))


def _furniture_parts(f: Furniture) -> _Parts:
    w, d, h = f.rect[2] - f.rect[0], f.rect[3] - f.rect[1], f.height
    p = _Parts()
    along_x = w >= d
    if f.type == "desk":
        p.box(0, 0, h - 0.02, w, d, 0.04, WOOD)
        for s in (-1, 1):
            if along_x:
                p.box(s * (w / 2 - 0.03), 0, (h - 0.04) / 2, 0.04, d - 0.06, h - 0.04, LAMINATE)
            else:
                p.box(0, s * (d / 2 - 0.03), (h - 0.04) / 2, w - 0.06, 0.04, h - 0.04, LAMINATE)
    elif f.type in ("table", "coffee_table"):
        p.box(0, 0, h - 0.02, w, d, 0.04, WOOD)
        for sx in (-1, 1):
            for sy in (-1, 1):
                p.box(sx * (w / 2 - 0.06), sy * (d / 2 - 0.06), (h - 0.04) / 2, 0.05, 0.05, h - 0.04, FRAME)
    elif f.type == "meeting_table":
        p.box(0, 0, h - 0.025, w, d, 0.05, WOOD_DARK)
        p.box(0, 0, (h - 0.05) / 2, w * 0.55 if along_x else 0.16, 0.16 if along_x else d * 0.55, h - 0.05, FRAME)
    elif f.type == "chair":
        p.cyl(0, 0, 0.015, min(w, d) * 0.5, 0.03, FRAME)
        p.cyl(0, 0, 0.23, 0.03, 0.42, FRAME)
        p.box(0, 0, 0.47, w * 0.95, d * 0.95, 0.06, "#4A4E55")
        (bx, by), (sx, sy) = _back_offset(f.back, w * 0.95, d * 0.95, 0.06)
        p.box(bx, by, 0.72, sx, sy, 0.44, "#4A4E55")
    elif f.type == "sofa":
        fabric = "#C9784F"
        p.box(0, 0, 0.21, w, d, 0.42, fabric)
        (bx, by), (sx, sy) = _back_offset(f.back, w, d, 0.22)
        p.box(bx, by, 0.42 + (h - 0.42) / 2, sx, sy, h - 0.42, fabric)
        for s in (-1, 1):
            if f.back in ("n", "s"):
                p.box(s * (w / 2 - 0.09), 0, 0.31, 0.18, d, 0.62, fabric)
            else:
                p.box(0, s * (d / 2 - 0.09), 0.31, w, 0.18, 0.62, fabric)
    elif f.type == "bookshelf":
        p.box(0, 0, h / 2, w, d, h, WOOD)
        for k in range(1, 5):
            z = h * k / 5
            p.box(0, 0, z, w - 0.06 if along_x else w + 0.01, d + 0.01 if along_x else d - 0.06, 0.025, WOOD_DARK)
    elif f.type == "cabinet":
        p.box(0, 0, (h - 0.03) / 2, w, d, h - 0.03, "#D9D5CE")
        p.box(0, 0, h - 0.015, w + 0.01, d + 0.01, 0.03, "#B5B0A8")
    elif f.type == "counter":
        p.box(0, 0, (h - 0.04) / 2, w, d, h - 0.04, "#F0EDE8")
        p.box(0, 0, h - 0.02, w + 0.02, d + 0.02, 0.04, "#4A4A4A")
    elif f.type == "fridge":
        p.box(0, 0, h / 2, w, d, h, "#E4E6E8")
        p.box(0, 0, h * 0.62, w + 0.005, d + 0.005, 0.012, "#9AA0A6")
    elif f.type == "plant":
        r = min(w, d) / 2
        p.cyl(0, 0, 0.175, r * 0.7, 0.35, "#B4673F")
        p.sphere(0, 0, 0.35 + r * 0.95, r * 1.05, "#4F8A3C")
        p.sphere(r * 0.35, -r * 0.25, 0.35 + r * 1.6, r * 0.6, "#5E9C48")
    elif f.type == "bed":
        p.box(0, 0, 0.18, w, d, 0.36, "#E9E6E1")
        p.box(0, 0, 0.42, w - 0.06, d - 0.06, 0.12, "#FFFFFF")
        (bx, by), (sx, sy) = _back_offset(f.back, w, d, 0.06)
        p.box(bx, by, 0.5, sx, sy, 1.0, "#9FB4C3")
    elif f.type == "rack":
        n_up = max(2, int(round((w if along_x else d) / 2.7)) + 1)
        for k in range(n_up):
            t = -0.5 + k / (n_up - 1)
            for s in (-1, 1):
                x, y = (t * (w - 0.08), s * (d / 2 - 0.04)) if along_x else (s * (w / 2 - 0.04), t * (d - 0.08))
                p.box(x, y, h / 2, 0.08, 0.08, h, "#3F5F86")
        for z in (0.1, h / 3, 2 * h / 3, h - 0.05):
            for s in (-1, 1):
                if along_x:
                    p.box(0, s * (d / 2 - 0.04), z, w, 0.06, 0.1, "#E07B2E")
                else:
                    p.box(s * (w / 2 - 0.04), 0, z, 0.06, d, 0.1, "#E07B2E")
        for level, z in enumerate((0.15, h / 3 + 0.05, 2 * h / 3 + 0.05)):
            n_box = max(1, int((w if along_x else d) / 1.3))
            for k in range(n_box):
                if (k + level) % 3 == 2:
                    continue  # some empty slots
                t = -0.5 + (k + 0.5) / n_box
                bx, by = (t * (w - 0.3), 0) if along_x else (0, t * (d - 0.3))
                p.box(bx, by, z + 0.3, 0.9 if along_x else d * 0.8, d * 0.8 if along_x else 0.9, 0.6, "#C8A06A")
    elif f.type == "machine":
        p.box(0, 0, h * 0.45, w, d, h * 0.9, "#5A6B7B")
        p.box(0, 0, h * 0.9 + 0.05, w * 0.8, d * 0.8, 0.1, "#3E4A56")
        p.box(0, 0, h * 0.3, w + 0.01, d + 0.01, 0.08, "#E7B416")
        p.box(w / 2 * 0.6 if along_x else 0, d / 2 + 0.03 if along_x else d / 2 * 0.6, h * 0.6, 0.3, 0.06, 0.4, "#20262D")
    elif f.type == "conveyor":
        p.box(0, 0, h - 0.04, w, d, 0.08, "#2B2F33")
        p.box(0, 0, h - 0.09, w, d + 0.04, 0.04, "#8D99A6")
        n_leg = max(2, int((w if along_x else d) / 1.5) + 1)
        for k in range(n_leg):
            t = -0.5 + k / (n_leg - 1)
            for s in (-1, 1):
                x, y = (t * (w - 0.1), s * (d / 2 - 0.05)) if along_x else (s * (w / 2 - 0.05), t * (d - 0.1))
                p.box(x, y, (h - 0.1) / 2, 0.06, 0.06, h - 0.1, "#8D99A6")
    elif f.type == "box":
        p.box(0, 0, h / 2, w, d, h, "#C8A06A")
    else:
        p.box(0, 0, h / 2, w, d, h, "#B8B4AD")
    return p


def _furniture_model(i, f: Furniture) -> str:
    cx, cy = (f.rect[0] + f.rect[2]) / 2, (f.rect[1] + f.rect[3]) / 2
    w, d, h = f.rect[2] - f.rect[0], f.rect[3] - f.rect[1], f.height
    return f"""
    <model name="furniture_{i}_{f.type}">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <pose>0 0 {h / 2:.3f} 0 0 0</pose>
          <geometry><box><size>{w:.3f} {d:.3f} {h:.3f}</size></box></geometry>
        </collision>{"".join(_furniture_parts(f).items)}
      </link>
    </model>"""


def world_sdf(fp: FloorPlan) -> str:
    walls = "".join(
        _wall_model(i, seg, m, fp.wall_height, fp.wall_thickness) for i, (seg, m) in enumerate(zip(fp.walls, fp.wall_materials))
    )
    aps = "".join(_ap_model(i, ap, fp.wall_height) for i, ap in enumerate(fp.access_points))
    furniture = "".join(_furniture_model(i, f) for i, f in enumerate(fp.furniture))
    cx, cy = fp.width / 2, fp.height / 2
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="wiloc_{fp.name}">
    <light name="key" type="directional">
      <cast_shadows>false</cast_shadows>
      <diffuse>0.52 0.51 0.49 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.35 0.45 -0.82</direction>
    </light>
    <light name="fill" type="directional">
      <cast_shadows>false</cast_shadows>
      <diffuse>0.22 0.23 0.24 1</diffuse>
      <specular>0 0 0 1</specular>
      <direction>0.5 -0.4 -0.77</direction>
    </light>
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
      <ambient>0.6 0.59 0.57 1</ambient>
      <background>0.98 0.97 0.95 1</background>
      <shadows>false</shadows>
    </scene>
    <gui>
      <plugin name="keyboard" filename="libKeyboardGUIPlugin.so"/>
      <camera name="user_camera">
        <pose>{cx:.2f} {cy - 10.5:.2f} 17.5 0 0.98 1.5708</pose>
      </camera>
    </gui>{_ground_model(fp)}{_floor_model(fp)}{walls}{furniture}{aps}
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
