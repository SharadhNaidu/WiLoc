"""Studio backend: reading why the simulation did not start (ros2_ws/src/wiloc_ros/wiloc_ros/ui/backend.py; stdlib only)."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "backend", Path(__file__).resolve().parents[1] / "ros2_ws/src/wiloc_ros/wiloc_ros/ui/backend.py"
)
backend = importlib.util.module_from_spec(_spec)
sys.modules["backend"] = backend  # dataclasses look their module up
_spec.loader.exec_module(backend)
diagnose = backend.diagnose

HEALTHY = """
===== 17:31:43 ros2 launch wiloc_ros sim.launch.py ui:=none gui:=false rviz:=false floorplan:=office ap_layout:=installed
[INFO] [launch]: All log files can be found below /home/u/.ros/log/2026-11-09-17-31-43
[INFO] [gzserver-1]: process started with pid [1416]
[gzserver-1] 2026-11-09 17:31:44.498 [RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_port7411: open_and_lock_file failed -> Function open_port_internal
[localization-7] [INFO] [1794225705.326720920] [wiloc_localization]: radio map loaded from cache in 0.0 s
[spawn_entity.py-3] [INFO] [1794225706.650369013] [spawn_entity]: Spawn status: SpawnEntity: Successfully spawned entity [wiloc_bot]
[INFO] [spawn_entity.py-3]: process has finished cleanly [pid 1420]
"""


def test_healthy_log_has_no_problem():
    assert diagnose(HEALTHY) is None
    assert diagnose("") is None


def test_launch_exception_is_reported_with_its_message():
    log = "[ERROR] [launch]: Caught exception in launch (see debug for traceback): no built-in venue 'x'; known: ['office']\n"
    p = diagnose(log)
    assert p is not None and not p.graphics
    assert "no built-in venue 'x'" in p.reason


def test_missing_package_points_to_setup():
    log = "[ERROR] [launch]: Caught exception in launch (see debug for traceback): \"package 'gazebo_ros' not found, searching: ['/opt/ros/humble']\"\n"
    p = diagnose(log)
    assert "gazebo_ros" in p.reason and "./wiloc.sh setup" in p.reason


def test_gazebo_crash_is_a_graphics_problem():
    log = HEALTHY + "[ERROR] [gzserver-1]: process has died [pid 1416, exit code -11, cmd 'gzserver /tmp/wiloc_office.world'].\n"
    p = diagnose(log)
    assert p is not None and p.graphics
    assert "Gazebo" in p.reason


def test_gazebo_stopped_by_us_is_not_a_crash():
    # SIGINT from the Studio (exit code -2) at a venue switch or quit
    log = HEALTHY + "[WARNING] [launch]: user interrupted with ctrl-c (SIGINT)\n[ERROR] [gzserver-1]: process has died [pid 1416, exit code -2, cmd 'gzserver'].\n"
    assert diagnose(log) is None


def test_robot_spawn_failure():
    log = "[ERROR] [spawn_entity.py-3]: process has died [pid 9, exit code 1, cmd 'spawn_entity.py -timeout 120'].\n"
    p = diagnose(log)
    assert p is not None and "robot" in p.reason


def test_problem_reads_only_this_run_and_new_lines(tmp_path, monkeypatch):
    log = tmp_path / "studio_backend.log"
    died = "[ERROR] [gzserver-1]: process has died [pid 1, exit code -11, cmd 'gzserver'].\n"
    log.write_text(died)  # an earlier run's crash
    monkeypatch.setattr(backend, "LOG_PATH", str(log))
    b = backend.Backend()
    b.proc = object()  # launched (problem() does not touch the process)
    b._log_pos = log.stat().st_size  # what start() records
    assert b.problem() is None
    with open(log, "a") as f:
        f.write(HEALTHY + died[:30])  # the death line only half written so far
    assert b.problem() is None
    with open(log, "a") as f:
        f.write(died[30:])
    p = b.problem()
    assert p is not None and p.graphics
    assert b.problem() is p  # cached: the log is not read again


def test_respawning_nodes_are_not_fatal():
    log = HEALTHY + "[ERROR] [wifi_sensor-4]: process has died [pid 451, exit code 1, cmd 'wifi_sensor'].\n"
    assert diagnose(log) is None
