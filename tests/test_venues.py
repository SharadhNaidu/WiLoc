"""Every shipped venue must be safe to run live (parametrised over all of them)."""

import numpy as np
import pytest

from wiloc.environment import FloorPlan, Furniture
from wiloc.planning import OccupancyGrid, kidnap_target, validate_floorplan

VENUES = FloorPlan.builtin_names()


@pytest.fixture(scope="module", params=VENUES)
def venue(request):
    fp = FloorPlan.builtin(request.param)
    return fp, OccupancyGrid.from_floorplan(fp)


def test_venue_is_valid(venue):
    fp, grid = venue
    assert validate_floorplan(fp, grid) == []


def test_furniture_blocks_the_planner_but_not_the_radio_world(venue):
    fp, grid = venue
    for f in fp.furniture:
        cx, cy = (f.rect[0] + f.rect[2]) / 2, (f.rect[1] + f.rect[3]) / 2
        assert not grid.is_free(np.array([[cx, cy]]))[0], f"{f.type} at {f.rect} not blocked"
    # furniture-blind: free space for radio / particle filter ignores it
    blind = OccupancyGrid.from_floorplan(fp, obstacles=False)
    assert blind.free.sum() >= grid.free.sum()


def test_rooms_cover_the_kidnap_spots(venue):
    fp, _ = venue
    if fp.rooms:
        for p in fp.kidnap_spots:
            assert fp.room_at(p) is not None


def test_kidnap_target_is_far_and_free(venue):
    fp, grid = venue
    rng = np.random.default_rng(0)
    here = np.asarray(fp.start_pose[:2])
    for _ in range(20):
        p = kidnap_target(fp, grid, here, rng)
        assert grid.is_free(p[None])[0]
        assert np.hypot(*(p - here)) >= 6.0


def test_venue_presentation(venue):
    fp, _ = venue
    assert fp.title and fp.caption, "every venue needs a title and a one-line caption"
    assert len(fp.camera.get("overview", [])) == 5
    assert fp.layout == next(iter(fp.layouts)) and len(fp.layouts) >= 2


def test_layouts_switch(venue):
    fp, _ = venue
    for name, (label, aps) in fp.layouts.items():
        g = fp.with_layout(name)
        assert g.layout == name and g.layout_label == label and [a.id for a in g.access_points] == [a.id for a in aps]
        assert fp.max_access_points >= len(aps) >= 1
    assert fp.access_points == list(fp.layouts[fp.layout][1])  # the original is untouched
    with pytest.raises(KeyError):
        fp.with_layout("no-such-layout")


def test_furniture_parsing():
    f = Furniture.from_dict({"type": "chair", "at": [2.0, 3.0]})
    assert f.rect == pytest.approx((1.76, 2.76, 2.24, 3.24)) and f.height == 0.9
    g = Furniture.from_dict({"type": "desk", "at": [5, 5], "rotate": True, "height": 0.7})
    assert g.rect == pytest.approx((4.65, 4.2, 5.35, 5.8)) and g.height == 0.7
    h = Furniture.from_dict({"type": "spaceship", "rect": [3, 3, 1, 1]})
    assert h.rect == (1.0, 1.0, 3.0, 3.0) and h.height == 1.0


def test_office_rooms():
    fp = FloorPlan.builtin("office")
    assert fp.room_at([12.5, 9.5]) == "Meeting room"
    assert fp.room_at([10.0, 6.0]) == "Corridor"
    assert fp.room_at([-1, -1]) is None
