"""Studio camera projection (ros2_ws/src/wiloc_ros/wiloc_ros/projection.py; plain numpy)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "projection", Path(__file__).resolve().parents[1] / "ros2_ws/src/wiloc_ros/wiloc_ros/projection.py"
)
proj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proj)

K = proj.intrinsics(960, 540, 1.15)


def test_point_ahead_is_image_centre():
    cam = (0.0, 0.0, 1.0, 0.0, 0.0)
    uv, ok = proj.project([[5.0, 0.0, 1.0]], cam, K)
    assert ok[0] and uv[0] == pytest.approx([480, 270])


def test_left_up_and_behind():
    cam = (0.0, 0.0, 1.0, 0.0, 0.0)
    uv, ok = proj.project([[5.0, 1.0, 1.0], [5.0, 0.0, 2.0], [-5.0, 0.0, 1.0]], cam, K)
    assert uv[0, 0] < 480  # +y (left) -> smaller u
    assert uv[1, 1] < 270  # +z (up) -> smaller v
    assert not ok[2]


def test_edge_of_field_of_view_and_yaw_pitch():
    cam = (0.0, 0.0, 0.0, 0.0, 0.0)
    a = 1.15 / 2
    uv, _ = proj.project([[math.cos(a), -math.sin(a), 0.0]], cam, K)
    assert uv[0, 0] == pytest.approx(960, abs=1e-6)  # right image edge
    # looking north and 45 deg down at a point on the floor straight ahead
    cam = (2.0, 3.0, 4.0, math.pi / 4, math.pi / 2)
    uv, ok = proj.project([[2.0, 7.0, 0.0]], cam, K)
    assert ok[0] and uv[0] == pytest.approx([480, 270])


def test_clip_segment():
    cam = (0.0, 0.0, 0.0, 0.0, 0.0)
    assert proj.clip_segment([-2, 0, 0], [-1, 0, 0], cam) is None
    a, b = proj.clip_segment([-2, 0, 0], [3, 0, 0], cam)
    assert a[0] == pytest.approx(0.1) and b[0] == 3
