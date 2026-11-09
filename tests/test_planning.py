import itertools

import numpy as np
import pytest

from wiloc import FloorPlan
from wiloc.motion import DiffDrive, WaypointController
from wiloc.planning import OccupancyGrid, PathFollower, path_length

ROOM_CENTRES = [(2.5, 9.5), (7.5, 9.5), (12.5, 9.5), (17.5, 9.5), (2.5, 2.5), (7.5, 2.5), (12.5, 2.5), (17.5, 2.5), (10.0, 6.0)]


@pytest.fixture(scope="module")
def fp():
    return FloorPlan.builtin("office")


@pytest.fixture(scope="module")
def grid(fp):
    return OccupancyGrid.from_floorplan(fp)


def _assert_valid(fp, grid, path):
    assert path is not None and len(path) >= 2
    legs = fp.crosses_wall(path[:-1], path[1:])
    assert not legs.any(), "a planned leg crosses a wall"
    for a, b in zip(path[:-1], path[1:]):
        assert grid.line_free(a, b), "a planned leg violates the robot's clearance"


def test_grid_clearance_matches_geometry(grid):
    # corridor centre line is 1 m from both wall centre lines; the walls are
    # 0.15 m thick boxes (as in Gazebo), so 0.925 m to the wall face
    assert grid.clearance_at(np.array([[10.0, 6.0]]))[0] == pytest.approx(0.925, abs=0.06)
    # a 1 m door gap: walls overrun their ends by half the thickness, so the
    # door centre is 0.5 - 0.075 m from the jamb
    assert grid.clearance_at(np.array([[2.5, 5.0]]))[0] == pytest.approx(0.425, abs=0.06)
    # a point on a wall has (almost) no clearance and is not free
    assert not grid.is_free(np.array([[4.0, 5.0]]))[0]


def test_paths_connect_every_pair_of_rooms(fp, grid):
    for a, b in itertools.combinations(ROOM_CENTRES, 2):
        path = grid.plan(a, b)
        _assert_valid(fp, grid, path)
        assert np.allclose(path[0], a) and np.allclose(path[-1], b)
        # through a door, never absurdly long: at most 3x the straight line + corridor detour
        assert path_length(path) < 3 * np.hypot(b[0] - a[0], b[1] - a[1]) + 12


def test_paths_pass_through_the_middle_of_doors(grid):
    # Room E (bottom-left) to room A (top-left): both doors span x = 2..3
    path = grid.plan((1.0, 2.0), (1.0, 10.0))
    dense = np.vstack([np.linspace(a, b, 50) for a, b in zip(path[:-1], path[1:])])
    at_wall = dense[np.abs(dense[:, 1] - 5.0) < 0.05]
    assert len(at_wall) and np.all(np.abs(at_wall[:, 0] - 2.5) < 0.15)


def test_patrol_waypoints_are_all_reachable(fp, grid):
    pts = np.vstack([fp.start_pose[:2], fp.waypoints])
    for a, b in zip(pts[:-1], pts[1:]):
        _assert_valid(fp, grid, grid.plan(a, b))


def test_goal_inside_a_wall_snaps_to_free_space(grid):
    path = grid.plan((2.5, 6.0), (7.0, 5.0))  # (7, 5) is on the corridor wall
    assert path is not None
    assert grid.is_free(path[-1:])[0]
    assert np.hypot(*(path[-1] - [7.0, 5.0])) < 0.4


def test_robot_kidnapped_into_a_room_drives_out_without_touching_walls(fp, grid):
    """The failure seen live: after a kidnap the patrol drove straight at a wall."""
    robot = DiffDrive()
    pose = np.array([1.7, 4.0, -2.5])  # room E, facing away from the door
    follower = PathFollower(grid.plan(pose, (7.5, 6.0)), WaypointController(v_max=0.3))
    body_r = 0.22
    for _ in range(4000):
        v, w, done = follower.command(pose)
        if done:
            break
        wl, wr = robot.wheel_speeds(v, w)
        new = DiffDrive.integrate(pose, *robot.body_twist(wl, wr), 0.05)
        assert not fp.crosses_wall(pose[None, :2], new[None, :2])[0]
        pose = new
        assert grid.clearance_at(pose[None, :2])[0] > body_r
    assert done and np.hypot(pose[0] - 7.5, pose[1] - 6.0) < 0.2


def test_follower_finishes_immediately_on_a_single_point_path():
    f = PathFollower(np.array([[1.0, 1.0]]))
    assert f.command(np.array([1.0, 1.0, 0.0])) == (0.0, 0.0, True)
