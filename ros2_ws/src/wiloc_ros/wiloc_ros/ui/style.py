"""Design tokens for WiLoc Studio: warm ivory, ink text, one restrained accent gradient.

Readable from 1.5 m: body text >= 16 px and the headline number >= 48 px at
1080p.  Everything is sized through ``px()`` so the window scales with the
screen (``set_scale``).
"""

from __future__ import annotations

import os

from PyQt5 import QtGui

# ------------------------------------------------------------------ colours
IVORY = "#FAF7F2"  # window
CARD = "#FFFFFF"  # cards
INK = "#141413"  # primary text
MUTED = "#6B675F"  # secondary text
FAINT = "#A39E94"  # tertiary text, axis labels
HAIRLINE = "#E6E1D8"  # card borders, grid
WASH = "#F3EEE6"  # pressed / inactive pill fill
ORANGE = "#E8743B"  # accent gradient start
INDIGO = "#4B4FD6"  # accent gradient end, and the particle filter

# One colour per method (distinct for the common colour-vision deficiencies).
METHOD_COLORS = {"pf": INDIGO, "ekf": "#D9822B", "wknn": "#159A8C", "odometry": "#9A958C"}
METHOD_NAMES = {"pf": "Particle filter", "ekf": "Kalman filter", "wknn": "Wi-Fi only", "odometry": "Wheels only"}
METHOD_SHORT = {"pf": "PF", "ekf": "EKF", "wknn": "WKNN", "odometry": "Wheels"}

# Localisation state (status pill)
STATE_COLORS = {"LOCKED": "#1F9D55", "UNCERTAIN": "#D9822B", "SEARCHING": INDIGO, "WAITING": FAINT}
STATE_TEXT = {"LOCKED": "Located", "UNCERTAIN": "Not sure", "SEARCHING": "Searching", "WAITING": "Starting"}

# Walls on the map, by radio material (darker = stops more signal)
WALL_COLORS = {"concrete": "#2E2B27", "brick": "#6E4A3A", "drywall": "#8F8577", "glass": "#6FA6CC", "wood": "#7A5535", "metal": "#4A5562"}

# Heat map: weak -> strong signal, a warm sequential ramp on ivory
HEAT_STOPS = [(0.0, "#FAF7F2"), (0.3, "#F6E6D6"), (0.55, "#F2CBAA"), (0.8, "#EBA478"), (1.0, "#E07E4C")]
HEAT_DBM = (-90.0, -30.0)  # one fixed, honest scale for every venue and layout

# ------------------------------------------------------------------ sizing
_scale = 1.0


def set_scale(s: float):
    global _scale
    _scale = max(0.6, min(2.5, float(s)))


def scale() -> float:
    return _scale


def px(v: float) -> int:
    return max(1, int(round(v * _scale)))


# ------------------------------------------------------------------ fonts
FAMILY = "Inter"


def load_fonts() -> str:
    """Register the bundled Inter weights; fall back to Ubuntu / DejaVu Sans."""
    global FAMILY
    dirs = []
    try:
        from ament_index_python.packages import get_package_share_directory

        dirs.append(os.path.join(get_package_share_directory("wiloc_ros"), "fonts"))
    except Exception:
        pass
    dirs.append(os.path.join(os.path.dirname(__file__), "..", "..", "fonts"))
    loaded = False
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith((".otf", ".ttf")) and QtGui.QFontDatabase.addApplicationFont(os.path.join(d, name)) >= 0:
                loaded = True
        if loaded:
            break
    families = set(QtGui.QFontDatabase().families())
    FAMILY = next((f for f in ("Inter", "Ubuntu", "DejaVu Sans") if f in families), QtGui.QFont().family())
    return FAMILY


def font(size: float, weight: int = QtGui.QFont.Normal, tabular: bool = False) -> QtGui.QFont:
    f = QtGui.QFont(FAMILY)
    f.setPixelSize(px(size))
    f.setWeight(weight)
    f.setHintingPreference(QtGui.QFont.PreferNoHinting)
    if tabular:
        f.setStyleHint(QtGui.QFont.Monospace, QtGui.QFont.PreferQuality)
    return f


MEDIUM, SEMIBOLD, BOLD = QtGui.QFont.Medium, QtGui.QFont.DemiBold, QtGui.QFont.Bold


def qcolor(hex_color: str, alpha: float = 1.0) -> QtGui.QColor:
    c = QtGui.QColor(hex_color)
    c.setAlphaF(max(0.0, min(1.0, alpha)))
    return c


def accent_gradient(x0, y0, x1, y1) -> QtGui.QLinearGradient:
    g = QtGui.QLinearGradient(x0, y0, x1, y1)
    g.setColorAt(0.0, QtGui.QColor(ORANGE))
    g.setColorAt(1.0, QtGui.QColor(INDIGO))
    return g


def qss() -> str:
    return f"""
    QWidget {{ background: {IVORY}; color: {INK}; font-family: "{FAMILY}"; }}
    QToolTip {{ background: {INK}; color: {IVORY}; border: none; padding: {px(6)}px {px(8)}px; border-radius: {px(6)}px; }}
    """
