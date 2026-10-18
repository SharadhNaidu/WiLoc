"""Differential-drive robot: kinematics, wheel odometry and motion models.

* ``DiffDrive``              exact unicycle kinematics from wheel speeds.
* ``WheelOdometry``          dead reckoning from noisy encoder ticks with
                             systematic errors (wheel radius / track width
                             miscalibration), so odometry drifts like a real
                             robot's.
* ``OdometryMotionModel``    the odometry motion model of Thrun, Burgard &
                             Fox, *Probabilistic Robotics* (2005), sec. 5.4:
                             a relative motion (rot1, trans, rot2) with noise
                             parameters alpha1..alpha4.  Used for sampling in
                             the particle filter and linearised for the EKF.
* ``WaypointController``     simple go-to-goal controller to drive patrols.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import wrap_angle


@dataclass
class DiffDrive:
    wheel_radius: float = 0.033  # TurtleBot3 Burger-like geometry
    wheel_separation: float = 0.160
    max_wheel_speed: float = 6.6  # rad/s

    def wheel_speeds(self, v: float, w: float):
        """Inverse kinematics: body twist -> (left, right) wheel rad/s."""
        wl = (v - 0.5 * w * self.wheel_separation) / self.wheel_radius
        wr = (v + 0.5 * w * self.wheel_separation) / self.wheel_radius
        s = max(1.0, abs(wl) / self.max_wheel_speed, abs(wr) / self.max_wheel_speed)
        return wl / s, wr / s

    def body_twist(self, wl: float, wr: float, radius=None, separation=None):
        r = self.wheel_radius if radius is None else radius
        b = self.wheel_separation if separation is None else separation
        return r * (wr + wl) / 2.0, r * (wr - wl) / b

    @staticmethod
    def integrate(pose, v, w, dt):
        """Exact arc integration of a constant twist."""
        x, y, th = pose
        if abs(w) < 1e-9:
            return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), th])
        r = v / w
        th2 = th + w * dt
        return np.array([x + r * (np.sin(th2) - np.sin(th)), y - r * (np.cos(th2) - np.cos(th)), wrap_angle(th2)])


@dataclass
class WheelOdometry:
    """Encoder dead reckoning with random slip and calibration errors."""

    robot: DiffDrive
    slip_sigma: float = 0.05  # relative per-step noise on each wheel's rotation
    radius_bias: float = 0.015  # +1.5 % wheel radius error
    separation_bias: float = 0.04  # +4 % track width error

    def __post_init__(self):
        self.pose = np.zeros(3)

    def reset(self, pose):
        self.pose = np.array(pose, dtype=float)

    def step(self, wl, wr, dt, rng):
        dl = wl * dt * (1 + self.slip_sigma * rng.standard_normal())
        dr = wr * dt * (1 + self.slip_sigma * rng.standard_normal())
        v, w = self.robot.body_twist(
            dl / dt,
            dr / dt,
            radius=self.robot.wheel_radius * (1 + self.radius_bias),
            separation=self.robot.wheel_separation * (1 + self.separation_bias),
        )
        self.pose = DiffDrive.integrate(self.pose, v, w, dt)
        return self.pose.copy()


def odom_delta(p_prev, p_now):
    """Decompose an odometry increment into (rot1, trans, rot2)."""
    dx, dy = p_now[0] - p_prev[0], p_now[1] - p_prev[1]
    trans = np.hypot(dx, dy)
    rot1 = wrap_angle(np.arctan2(dy, dx) - p_prev[2]) if trans > 1e-6 else 0.0
    rot2 = wrap_angle(p_now[2] - p_prev[2] - rot1)
    return float(rot1), float(trans), float(rot2)


@dataclass
class OdometryMotionModel:
    alpha1: float = 0.05  # rot noise from rot
    alpha2: float = 0.01  # rot noise from trans
    alpha3: float = 0.05  # trans noise from trans
    alpha4: float = 0.01  # trans noise from rot

    def sample(self, poses, delta, rng):
        """Propagate particles (N, 3) through a noisy odometry increment."""
        rot1, trans, rot2 = delta
        n = len(poses)
        s_r1 = np.sqrt(self.alpha1 * rot1**2 + self.alpha2 * trans**2)
        s_t = np.sqrt(self.alpha3 * trans**2 + self.alpha4 * (rot1**2 + rot2**2))
        s_r2 = np.sqrt(self.alpha1 * rot2**2 + self.alpha2 * trans**2)
        r1 = rot1 + s_r1 * rng.standard_normal(n)
        t = trans + s_t * rng.standard_normal(n)
        r2 = rot2 + s_r2 * rng.standard_normal(n)
        out = np.empty_like(poses)
        out[:, 0] = poses[:, 0] + t * np.cos(poses[:, 2] + r1)
        out[:, 1] = poses[:, 1] + t * np.sin(poses[:, 2] + r1)
        out[:, 2] = wrap_angle(poses[:, 2] + r1 + r2)
        return out

    def predict_mean_jacobians(self, pose, delta):
        """g(x, u), G = dg/dx, V = dg/du and M = cov(u) for the EKF."""
        rot1, trans, rot2 = delta
        th = pose[2]
        c, s = np.cos(th + rot1), np.sin(th + rot1)
        mean = np.array([pose[0] + trans * c, pose[1] + trans * s, wrap_angle(th + rot1 + rot2)])
        G = np.array([[1, 0, -trans * s], [0, 1, trans * c], [0, 0, 1]])
        V = np.array([[-trans * s, c, 0], [trans * c, s, 0], [1, 0, 1]])
        M = np.diag(
            [
                self.alpha1 * rot1**2 + self.alpha2 * trans**2 + 1e-8,
                self.alpha3 * trans**2 + self.alpha4 * (rot1**2 + rot2**2) + 1e-8,
                self.alpha1 * rot2**2 + self.alpha2 * trans**2 + 1e-8,
            ]
        )
        return mean, G, V, M


@dataclass
class WaypointController:
    """Turn-then-drive go-to-goal controller producing (v, w)."""

    v_max: float = 0.22
    w_max: float = 1.5
    k_heading: float = 2.5
    goal_tol: float = 0.12

    def command(self, pose, goal):
        dx, dy = goal[0] - pose[0], goal[1] - pose[1]
        dist = np.hypot(dx, dy)
        if dist < self.goal_tol:
            return 0.0, 0.0, True
        err = wrap_angle(np.arctan2(dy, dx) - pose[2])
        w = float(np.clip(self.k_heading * err, -self.w_max, self.w_max))
        # Slow down while badly misaligned so the robot stays on the line.
        v = self.v_max * max(0.0, np.cos(err)) ** 3 * min(1.0, dist / 0.5 + 0.3)
        return float(v), w, False
