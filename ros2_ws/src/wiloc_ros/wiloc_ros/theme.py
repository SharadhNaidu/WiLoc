"""WiLoc visual theme: white background, near-black text, saturated accents.

One palette is shared by the RViz markers, the Gazebo world and the
dashboard, so an access point or a wall material has the same colour
everywhere.
"""

from __future__ import annotations

import numpy as np

# Base
BG = "#ffffff"  # window / plot background
PANEL = "#f4f6f9"  # raised panels, table rows, floor slab
BORDER = "#d5dbe3"  # borders, grid lines
TEXT = "#1b1f24"  # primary text
MUTED = "#5f6b7a"  # secondary text

# Accents (dark enough to read on white)
CORAL = "#e4572e"
AMBER = "#e09f1f"
TEAL = "#17a398"
BLUE = "#2f80ed"
PURPLE = "#8e5bd8"
GREEN = "#5aa832"

AP_COLORS = [CORAL, AMBER, TEAL, BLUE, PURPLE, GREEN]

# Estimator series
C_FILTER = BLUE
C_WIFI = AMBER
C_ODOM = "#8a94a3"
C_MAP = TEAL
C_TRUTH = TEXT

MATERIALS = {  # wall colour by material
    "concrete": "#5b6778",
    "brick": "#c0563b",
    "drywall": "#c9ab78",
    "glass": "#36a9d6",
    "wood": "#9a6b43",
}

# Heat map ramp: weak -> strong signal (pale blue on white -> teal -> amber -> coral)
HEAT_STOPS = [(0.0, "#e8f1fb"), (0.3, "#9fd3e6"), (0.55, "#17a398"), (0.78, "#e09f1f"), (1.0, "#e4572e")]


def rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def ap_color(i: int) -> str:
    return AP_COLORS[i % len(AP_COLORS)]


def heat_rgb(t: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB (N, 3) along HEAT_STOPS."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    xs = [s for s, _ in HEAT_STOPS]
    cols = np.array([rgb(c) for _, c in HEAT_STOPS])
    return np.stack([np.interp(t, xs, cols[:, k]) for k in range(3)], axis=-1)


QSS = f"""
* {{ color: {TEXT}; }}
QMainWindow, QWidget {{ background: {BG}; }}
QDockWidget::title {{ background: {PANEL}; padding: 4px; }}
QTreeView, QListView, QTableView, QTableWidget, QTextEdit, QLineEdit, QComboBox, QSpinBox {{
    background: {BG}; alternate-background-color: {PANEL}; border: 1px solid {BORDER};
    selection-background-color: {BLUE}; selection-color: white; gridline-color: {BORDER};
}}
QHeaderView::section {{ background: {PANEL}; color: {TEXT}; padding: 4px; border: none;
    border-bottom: 2px solid {BORDER}; font-weight: 600; }}
QPushButton, QToolButton {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 10px;
}}
QPushButton:hover, QToolButton:hover {{ border-color: {BLUE}; color: {BLUE}; }}
QPushButton:pressed, QToolButton:checked {{ background: {BLUE}; color: white; border-color: {BLUE}; }}
QMenuBar, QMenu, QToolBar, QStatusBar {{ background: {PANEL}; }}
QMenu::item:selected, QMenuBar::item:selected {{ background: {BLUE}; color: white; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 8px; margin-top: 1.3em; padding-top: 0.5em; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {CORAL}; font-weight: 700; }}
QSplitter::handle {{ background: {BORDER}; }}
QScrollBar {{ background: {PANEL}; }}
QScrollBar::handle {{ background: {BORDER}; border-radius: 4px; }}
QToolTip {{ background: {TEXT}; color: white; border: none; }}
"""
