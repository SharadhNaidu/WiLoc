"""WiLoc: indoor localisation of a mobile robot by Wi-Fi RSSI fingerprinting.

Pipeline
--------
environment  -> floor plan, walls, access points (YAML)
radio        -> RSSI simulator: log-distance + multi-wall + correlated shadowing
fingerprint  -> offline site survey, radio map, GP-interpolated dense map
estimators   -> NN / KNN / WKNN / probabilistic / trilateration (single scan)
motion       -> differential-drive kinematics, odometry, motion model
filters      -> particle filter (MCL) and EKF fusing odometry + Wi-Fi
metrics      -> error statistics and CDFs
simulation   -> ROS-free simulator used by the experiments and tests
"""

from .environment import AccessPoint, FloorPlan
from .estimators import KNN, Probabilistic, Trilateration, default_estimators
from .filters import ParticleFilter, WifiEKF
from .fingerprint import DenseRadioMap, RadioMap
from .motion import DiffDrive, OdometryMotionModel, WaypointController, WheelOdometry
from .radio import RadioConfig, RadioSimulator
from .simulation import World, drive_patrol, run_trackers

__version__ = "0.1.0"

__all__ = [
    "AccessPoint",
    "FloorPlan",
    "KNN",
    "Probabilistic",
    "Trilateration",
    "default_estimators",
    "ParticleFilter",
    "WifiEKF",
    "DenseRadioMap",
    "RadioMap",
    "DiffDrive",
    "OdometryMotionModel",
    "WaypointController",
    "WheelOdometry",
    "RadioConfig",
    "RadioSimulator",
    "World",
    "drive_patrol",
    "run_trackers",
]
