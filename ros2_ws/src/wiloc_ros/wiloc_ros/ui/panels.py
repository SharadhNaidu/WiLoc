"""Studio building blocks: cards, pill buttons, header, venue chips, camera view, race chart, actions, overlay."""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import theme
from ..projection import clip_segment, intrinsics, project
from . import style as S


# ====================================================================== basics
class Card(QtWidgets.QFrame):
    """White rounded card with a title row; children go in ``body``."""

    def __init__(self, title: str = "", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.title, self.subtitle = title, subtitle
        self.right_text = ""
        self.body = QtWidgets.QVBoxLayout()
        lay = QtWidgets.QVBoxLayout(self)
        lay.addSpacing(S.px(34) if title else 0)
        lay.addLayout(self.body, 1)
        self._lay = lay
        self.setAttribute(QtCore.Qt.WA_StyledBackground, False)

    def rescale(self):
        m = S.px(14)
        self._lay.setContentsMargins(m, m if not self.title else S.px(10), m, m)
        self._lay.itemAt(0).changeSize(0, S.px(30) if self.title else 0)

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QtGui.QPen(QtGui.QColor(S.HAIRLINE), 1))
        p.setBrush(QtGui.QColor(S.CARD))
        p.drawRoundedRect(r, S.px(16), S.px(16))
        if self.title:
            x, y = S.px(16), S.px(12)
            p.setFont(S.font(15, S.SEMIBOLD))
            p.setPen(QtGui.QColor(S.INK))
            fm = QtGui.QFontMetrics(p.font())
            p.drawText(QtCore.QPointF(x, y + fm.ascent()), self.title)
            if self.subtitle:
                p.setFont(S.font(13))
                p.setPen(QtGui.QColor(S.MUTED))
                p.drawText(QtCore.QPointF(x + fm.horizontalAdvance(self.title) + S.px(10), y + fm.ascent()), self.subtitle)
            if self.right_text:
                p.setFont(S.font(13))
                p.setPen(QtGui.QColor(S.MUTED))
                p.drawText(QtCore.QRectF(0, y, self.width() - S.px(16), fm.height()), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, self.right_text)


class Pill(QtWidgets.QAbstractButton):
    """Rounded pill button: ``primary`` (accent gradient), ``secondary`` (outline) or ``chip`` (selectable)."""

    def __init__(self, text: str, kind: str = "secondary", parent=None, size: float = 14):
        super().__init__(parent)
        self.setText(text)
        self.kind, self.size = kind, size
        self.dot = None  # optional colour dot before the text
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self._hover = False

    def sizeHint(self):
        fm = QtGui.QFontMetrics(S.font(self.size, S.MEDIUM))
        extra = S.px(16) if self.dot else 0
        return QtCore.QSize(fm.horizontalAdvance(self.text()) + S.px(2 * 16) + extra, S.px(self.size * 2.5))

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)
        rad = r.height() / 2
        on = self.isChecked() or self.isDown()
        if self.kind == "primary":
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(S.accent_gradient(r.left(), 0, r.right(), 0))
            if self._hover and not self.isDown():
                p.setOpacity(0.92)
            p.drawRoundedRect(r, rad, rad)
            p.setOpacity(1.0)
            fg = QtGui.QColor("#FFFFFF")
        elif self.kind == "chip" and on:
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(S.INK))
            p.drawRoundedRect(r, rad, rad)
            fg = QtGui.QColor(S.IVORY)
        else:
            p.setPen(QtGui.QPen(QtGui.QColor(S.INK if (self._hover or on) else S.HAIRLINE), 1.2))
            p.setBrush(QtGui.QColor(S.WASH if on else S.CARD))
            p.drawRoundedRect(r, rad, rad)
            fg = QtGui.QColor(S.INK if self.isEnabled() else S.FAINT)
        x = r.left() + S.px(16)
        if self.dot:
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(self.dot))
            p.drawEllipse(QtCore.QPointF(x + S.px(4), r.center().y()), S.px(4), S.px(4))
            x += S.px(14)
        p.setFont(S.font(self.size, S.MEDIUM))
        p.setPen(fg)
        p.drawText(QtCore.QRectF(x, r.top(), r.right() - x - S.px(14), r.height()), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignHCenter if not self.dot else QtCore.Qt.AlignVCenter, self.text())


class ChipRow(QtWidgets.QWidget):
    """A row of exclusive chips; emits the chosen key."""

    chosen = QtCore.pyqtSignal(str)

    def __init__(self, size: float = 13, parent=None):
        super().__init__(parent)
        self.size = size
        self.lay = QtWidgets.QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}

    def set_items(self, items: list[tuple[str, str, str]], current: str):
        """items: (key, label, tooltip)."""
        for b in self.buttons.values():
            self.group.removeButton(b)
            b.deleteLater()
        self.buttons = {}
        while self.lay.count():
            self.lay.takeAt(0)
        for key, label, tip in items:
            b = Pill(label, "chip", size=self.size)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.chosen.emit(k))
            self.group.addButton(b)
            self.lay.addWidget(b)
            self.buttons[key] = b
        self.lay.addStretch(1)
        self.select(current)

    def select(self, key: str):
        if key in self.buttons:
            self.buttons[key].setChecked(True)

    def rescale(self):
        self.lay.setSpacing(S.px(6))
        for b in self.buttons.values():
            b.setFixedSize(b.sizeHint())


# ====================================================================== header
class Header(QtWidgets.QWidget):
    """Wordmark and tagline | where we are | status pill and the big error number."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.venue_line = ""
        self.state = "WAITING"
        self.error = None
        self.caption = ""
        self.t0 = time.monotonic()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        h = self.height()
        # wordmark: gradient dot + WiLoc
        x = S.px(4)
        d = S.px(22)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(S.accent_gradient(x, 0, x + d, d))
        p.drawEllipse(QtCore.QRectF(x, S.px(10), d, d))
        p.setFont(S.font(28, S.BOLD))
        p.setPen(QtGui.QColor(S.INK))
        fm = QtGui.QFontMetrics(p.font())
        p.drawText(QtCore.QPointF(x + d + S.px(10), S.px(10) + d / 2 + fm.capHeight() / 2), "WiLoc")
        p.setFont(S.font(15))
        p.setPen(QtGui.QColor(S.MUTED))
        p.drawText(QtCore.QPointF(x, S.px(58)), "Finding a robot with nothing but Wi-Fi")

        # right: big number + label, status pill left of it
        right = self.width() - S.px(4)
        num = "–" if self.error is None else f"{self.error:.2f}"
        p.setFont(S.font(52, S.SEMIBOLD))
        fm = QtGui.QFontMetrics(p.font())
        unit_font = S.font(22, S.MEDIUM)
        ufm = QtGui.QFontMetrics(unit_font)
        uw = ufm.horizontalAdvance(" m")
        nw = fm.horizontalAdvance(num)
        base = S.px(52)
        p.setPen(QtGui.QColor(S.INK))
        p.drawText(QtCore.QPointF(right - uw - nw, base), num)
        p.setFont(unit_font)
        p.setPen(QtGui.QColor(S.MUTED))
        p.drawText(QtCore.QPointF(right - uw, base), " m")
        p.setFont(S.font(13))
        lab = "how far the particle filter is off"
        lw = QtGui.QFontMetrics(p.font()).horizontalAdvance(lab)
        p.drawText(QtCore.QPointF(right - lw, S.px(72)), lab)

        # status pill
        col = QtGui.QColor(S.STATE_COLORS.get(self.state, S.FAINT))
        text = S.STATE_TEXT.get(self.state, self.state.title())
        p.setFont(S.font(16, S.SEMIBOLD))
        tw = QtGui.QFontMetrics(p.font()).horizontalAdvance(text)
        pw, ph = tw + S.px(46), S.px(38)
        px_ = right - max(uw + nw, lw) - S.px(28) - pw
        py_ = S.px(16)
        pill = QtCore.QRectF(px_, py_, pw, ph)
        fill = QtGui.QColor(col)
        fill.setAlphaF(0.12)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(pill, ph / 2, ph / 2)
        dot = QtCore.QPointF(px_ + S.px(19), py_ + ph / 2)
        if self.state == "SEARCHING":  # breathing ring while searching
            phase = (time.monotonic() - self.t0) % 1.4 / 1.4
            ring = QtGui.QColor(col)
            ring.setAlphaF(0.5 * (1 - phase))
            p.setBrush(ring)
            p.drawEllipse(dot, S.px(5) + S.px(7) * phase, S.px(5) + S.px(7) * phase)
        p.setBrush(col)
        p.drawEllipse(dot, S.px(5), S.px(5))
        p.setPen(col.darker(115))
        p.drawText(QtCore.QRectF(px_ + S.px(32), py_, pw - S.px(36), ph), QtCore.Qt.AlignVCenter, text)

        # where: venue + router layout, and the plain-English caption under it
        p.setFont(S.font(13))
        p.setPen(QtGui.QColor(S.MUTED))
        cap_rect = QtCore.QRectF(px_ - S.px(2) - S.px(520), py_ + ph + S.px(8), S.px(520) + pw, S.px(20))
        p.drawText(cap_rect, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, self.caption)


# ====================================================================== camera
class CameraView(QtWidgets.QWidget):
    """Live Gazebo camera, letterboxed into a rounded frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img = None
        self._buf = None
        self.count = 0
        self.last_frame = 0.0
        self.label = "Live 3-D view"
        self.snap = self.fp = None
        self.show_rays = True
        self.setMinimumSize(320, 180)

    def set_scene(self, snap, fp):
        """What to draw on top of the image (rays, estimates, route)."""
        self.snap, self.fp = snap, fp

    def set_frame(self, frame, count):
        if frame is None or count == self.count:
            return
        h, w, data = frame
        self._buf = data  # QImage below views this buffer: keep it alive
        self.img = QtGui.QImage(self._buf, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.count = count
        self.last_frame = time.monotonic()
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        r = QtCore.QRectF(self.rect())
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, S.px(12), S.px(12))
        p.setClipPath(path)
        p.fillRect(r, QtGui.QColor("#EEE9E1"))
        fresh = self.img is not None and time.monotonic() - self.last_frame < 3.0
        if self.img is not None:
            iw, ih = self.img.width(), self.img.height()
            k = max(r.width() / iw, r.height() / ih)  # fill, crop the overflow
            tw, th = iw * k, ih * k
            dest = QtCore.QRectF(r.center().x() - tw / 2, r.center().y() - th / 2, tw, th)
            p.drawImage(dest, self.img)
            if fresh:
                self._overlay(p, dest, k)
            if not fresh:
                p.fillRect(r, S.qcolor(S.IVORY, 0.6))
        if not fresh:
            p.setFont(S.font(16, S.MEDIUM))
            p.setPen(QtGui.QColor(S.MUTED))
            p.drawText(r, QtCore.Qt.AlignCenter, "Starting the 3-D view…" if self.img is None else "3-D view paused…")
        if fresh and self.show_rays and self.snap is not None and self.snap.scan:
            self._ray_legend(p, r)
        # label chip
        p.setFont(S.font(12, S.SEMIBOLD))
        fm = QtGui.QFontMetrics(p.font())
        cw = fm.horizontalAdvance(self.label) + S.px(30)
        chip = QtCore.QRectF(S.px(12), S.px(12), cw, S.px(26))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(S.qcolor(S.INK, 0.62))
        p.drawRoundedRect(chip, chip.height() / 2, chip.height() / 2)
        p.setBrush(QtGui.QColor("#FF5A4E" if fresh else S.FAINT))
        p.drawEllipse(QtCore.QPointF(chip.left() + S.px(12), chip.center().y()), S.px(3.5), S.px(3.5))
        p.setPen(QtGui.QColor("#FFFFFF"))
        p.drawText(chip.adjusted(S.px(20), 0, 0, 0), QtCore.Qt.AlignVCenter, self.label)


    # ------------------------------------------------------------ augmented reality
    def _overlay(self, p, dest, k):
        snap, fp = self.snap, self.fp
        cam = (snap.camera or {}) if snap is not None else {}
        if fp is None or "pose" not in cam:
            return
        pose = cam["pose"]
        K = intrinsics(cam["width"], cam["height"], cam["hfov"])

        def to_w(uv):
            return QtCore.QPointF(dest.left() + uv[0] * k, dest.top() + uv[1] * k)

        def seg(a, b):
            c = clip_segment(a, b, pose)
            if c is None:
                return None
            uv, ok = project(np.array(c), pose, K)
            return (to_w(uv[0]), to_w(uv[1])) if ok.all() else None

        # planned route on the floor
        if len(snap.plan) >= 2:
            pen = QtGui.QPen(S.qcolor(S.INK, 0.7), S.px(2.5), QtCore.Qt.DashLine)
            pen.setDashPattern([2, 2])
            p.setPen(pen)
            for a, b in zip(snap.plan[:-1], snap.plan[1:]):
                s = seg([a[0], a[1], 0.03], [b[0], b[1], 0.03])
                if s:
                    p.drawLine(*s)

        # Wi-Fi rays: one per router heard; thicker = louder, dashed = through walls, faded = a person in the way
        if self.show_rays and snap.scan and snap.truth is not None:
            rx, ry = snap.truth[:2]
            aps = {ap.id: (i, ap) for i, ap in enumerate(fp.access_points)}
            for row in snap.scan.get("aps", []):
                if row["id"] not in aps or row.get("measured_dbm") is None:
                    continue
                i, ap = aps[row["id"]]
                s = seg([rx, ry, 0.35], [ap.x, ap.y, fp.wall_height - 0.12])
                if s is None:
                    continue
                strength = float(np.clip((row["measured_dbm"] + 90) / 55, 0, 1))
                blocked = row.get("blocking_db", 0) > 1.0
                col = S.qcolor(theme.ap_color(i), (0.35 if blocked else 0.9) * (0.45 + 0.55 * strength))
                pen = QtGui.QPen(col, S.px(1.2 + 3.6 * strength))
                pen.setCapStyle(QtCore.Qt.RoundCap)
                if row.get("walls"):
                    pen.setStyle(QtCore.Qt.DashLine)
                p.setPen(pen)
                p.drawLine(*s)

        # each method's guess, as a ring on the floor
        for m, radius in (("wknn", 0.32), ("ekf", 0.42), ("pf", 0.55)):
            if m not in snap.poses:
                continue
            x, y, _ = snap.poses[m]
            ang = np.linspace(0, 2 * np.pi, 33)
            ring = np.column_stack([x + radius * np.cos(ang), y + radius * np.sin(ang), np.full(33, 0.03)])
            uv, ok = project(ring, pose, K)
            if not ok.all():
                continue
            pen = QtGui.QPen(QtGui.QColor(S.METHOD_COLORS[m]), S.px(3.2 if m == "pf" else 2.4))
            p.setPen(pen)
            p.setBrush(S.qcolor(S.METHOD_COLORS[m], 0.10 if m == "pf" else 0.0))
            p.drawPolygon(QtGui.QPolygonF([to_w(q) for q in uv]))
            if m == "pf":
                top = min(uv, key=lambda q: q[1])
                tag = "particle filter's guess"
                p.setFont(S.font(12, S.SEMIBOLD))
                fm = QtGui.QFontMetrics(p.font())
                tw = fm.horizontalAdvance(tag) + S.px(16)
                c = to_w(top)
                box = QtCore.QRectF(c.x() - tw / 2, c.y() - S.px(30), tw, S.px(22))
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(S.qcolor(S.INDIGO, 0.92))
                p.drawRoundedRect(box, box.height() / 2, box.height() / 2)
                p.setPen(QtGui.QColor("#FFFFFF"))
                p.drawText(box, QtCore.Qt.AlignCenter, tag)

    def _ray_legend(self, p, r):
        p.setFont(S.font(12))
        txt = "Lines: Wi-Fi from each router · thicker = louder · dashed = through walls · faded = a person in the way"
        fm = QtGui.QFontMetrics(p.font())
        w = min(r.width() - S.px(24), fm.horizontalAdvance(txt) + S.px(24))
        box = QtCore.QRectF(S.px(12), r.bottom() - S.px(38), w, S.px(26))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(S.qcolor(S.CARD, 0.88))
        p.drawRoundedRect(box, box.height() / 2, box.height() / 2)
        p.setPen(QtGui.QColor(S.MUTED))
        p.drawText(box.adjusted(S.px(12), 0, -S.px(12), 0), QtCore.Qt.AlignVCenter, fm.elidedText(txt, QtCore.Qt.ElideRight, int(w - S.px(24))))


# ====================================================================== race chart
class RaceChart(QtWidgets.QWidget):
    """Rolling position error of each method (wall clock, last ``window`` s)."""

    def __init__(self, window: float = 120.0, parent=None):
        super().__init__(parent)
        self.window = window
        self.series = {m: deque() for m in S.METHOD_COLORS}
        self.kidnaps = deque()
        self.setMinimumHeight(120)

    def clear(self):
        for d in self.series.values():
            d.clear()
        self.kidnaps.clear()
        self.update()

    def add(self, t: float, errs: dict):
        for m, e in errs.items():
            if m in self.series and e is not None and np.isfinite(e):
                self.series[m].append((t, float(e)))
        for d in list(self.series.values()) + [self.kidnaps]:
            while d and t - (d[0][0] if isinstance(d[0], tuple) else d[0]) > self.window + 5:
                d.popleft()

    def mark_kidnap(self, t: float):
        self.kidnaps.append(t)

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        now = time.monotonic()
        lm, rm, tm, bm = S.px(34), S.px(96), S.px(6), S.px(22)
        plot = QtCore.QRectF(lm, tm, self.width() - lm - rm, self.height() - tm - bm)
        vals = [e for d in self.series.values() for t, e in d if now - t <= self.window]
        top = max(2.0, min(6.0, float(np.percentile(vals, 95)) * 1.1 if vals else 2.0))
        step = 0.5 if top <= 2.5 else 1.0 if top <= 6 else 2.0
        top = math.ceil(top / step) * step

        def X(t):
            return plot.right() - (now - t) / self.window * plot.width()

        def Y(e):
            return plot.bottom() - min(e, top) / top * plot.height()

        p.setFont(S.font(11))
        for v in np.arange(0, top + 1e-9, step):
            y = Y(v)
            p.setPen(QtGui.QPen(QtGui.QColor(S.HAIRLINE), 1))
            p.drawLine(QtCore.QPointF(plot.left(), y), QtCore.QPointF(plot.right(), y))
            p.setPen(QtGui.QColor(S.FAINT))
            p.drawText(QtCore.QRectF(0, y - S.px(8), lm - S.px(6), S.px(16)), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"{v:g} m")
        for s in range(0, int(self.window) + 1, 30):
            x = X(now - s)
            p.setPen(QtGui.QColor(S.FAINT))
            p.drawText(QtCore.QRectF(x - S.px(30), plot.bottom() + S.px(4), S.px(60), S.px(16)), QtCore.Qt.AlignHCenter, "now" if s == 0 else f"{s} s ago")
        p.setClipRect(plot.adjusted(-2, -4, 2, 2))
        for t in self.kidnaps:
            if now - t <= self.window:
                x = X(t)
                p.setPen(QtGui.QPen(S.qcolor(S.INK, 0.5), S.px(1.2), QtCore.Qt.DashLine))
                p.drawLine(QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom()))
                p.setFont(S.font(11, S.SEMIBOLD))
                p.setPen(QtGui.QColor(S.INK))
                p.drawText(QtCore.QPointF(x + S.px(4), plot.top() + S.px(12)), "kidnap")
        ends = []
        for m in ("odometry", "wknn", "ekf", "pf"):
            d = [(t, e) for t, e in self.series[m] if now - t <= self.window + 2]
            if len(d) < 2:
                continue
            pen = QtGui.QPen(S.qcolor(S.METHOD_COLORS[m], 1.0 if m in ("pf", "ekf") else 0.75), S.px({"pf": 3.2, "ekf": 2.4}.get(m, 1.6)))
            pen.setJoinStyle(QtCore.Qt.RoundJoin)
            if m == "odometry":
                pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(X(t), Y(e)) for t, e in d]))
            ends.append((Y(d[-1][1]), m, d[-1][1]))
        p.setClipping(False)
        # end labels: de-overlapped and kept inside the plot's height
        ends.sort()
        gap = S.px(16)
        ys = []
        for y, _, _ in ends:
            ys.append(max(y, ys[-1] + gap if ys else plot.top() + S.px(8)))
        over = (ys[-1] - plot.bottom()) if ys else 0
        if over > 0:
            ys = [y - over for y in ys]
            for i in range(len(ys) - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - gap)
        p.setFont(S.font(12, S.SEMIBOLD))
        for y, (_, m, e) in zip(ys, ends):
            p.setPen(QtGui.QColor(S.METHOD_COLORS[m]))
            p.drawText(QtCore.QPointF(plot.right() + S.px(8), y + S.px(4)), f"{S.METHOD_SHORT[m]} {e:.1f} m")


class Leaderboard(QtWidgets.QWidget):
    """Rolling 30 s median per method; the leader gets the accent."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.medians = {}
        self.leader = None
        self.recovery = None

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        x = 0.0
        h = self.height()
        for m in ("pf", "ekf", "wknn", "odometry"):
            v = self.medians.get(m)
            txt = f"{S.METHOD_NAMES[m]}  " + ("–" if v is None else f"{v:.2f} m")
            p.setFont(S.font(13, S.MEDIUM))
            w = QtGui.QFontMetrics(p.font()).horizontalAdvance(txt) + S.px(36)
            r = QtCore.QRectF(x, (h - S.px(30)) / 2, w, S.px(30))
            lead = m == self.leader
            p.setPen(QtGui.QPen(QtGui.QColor(S.METHOD_COLORS[m] if lead else S.HAIRLINE), 1.5 if lead else 1))
            p.setBrush(S.qcolor(S.METHOD_COLORS[m], 0.08) if lead else QtGui.QColor(S.CARD))
            p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(S.METHOD_COLORS[m]))
            p.drawEllipse(QtCore.QPointF(r.left() + S.px(14), r.center().y()), S.px(4.5), S.px(4.5))
            p.setPen(QtGui.QColor(S.INK))
            p.drawText(r.adjusted(S.px(26), 0, 0, 0), QtCore.Qt.AlignVCenter, txt)
            x += w + S.px(8)
            if x > self.width():
                break


# ====================================================================== overlay
class Overlay(QtWidgets.QWidget):
    """Full-window message while a venue loads or the simulation restarts."""

    def __init__(self, parent):
        super().__init__(parent)
        self.text, self.sub = "", ""
        self.t0 = time.monotonic()
        self.hide()

    def show_message(self, text, sub=""):
        self.text, self.sub = text, sub
        self.t0 = time.monotonic()
        self.setGeometry(self.parent().rect())
        self.show()
        self.raise_()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), S.qcolor(S.IVORY, 0.88))
        w, h = S.px(520), S.px(170)
        r = QtCore.QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        p.setPen(QtGui.QPen(QtGui.QColor(S.HAIRLINE), 1))
        p.setBrush(QtGui.QColor(S.CARD))
        p.drawRoundedRect(r, S.px(18), S.px(18))
        p.setFont(S.font(22, S.SEMIBOLD))
        p.setPen(QtGui.QColor(S.INK))
        p.drawText(r.adjusted(0, S.px(30), 0, -S.px(80)), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, self.text)
        p.setFont(S.font(14))
        p.setPen(QtGui.QColor(S.MUTED))
        p.drawText(r.adjusted(S.px(20), S.px(72), -S.px(20), -S.px(40)), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap, self.sub)
        # three dots
        ph = (time.monotonic() - self.t0) * 2.2
        for i in range(3):
            a = 0.25 + 0.75 * max(0.0, math.sin(ph - i * 0.7))
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(S.qcolor(S.INDIGO, a))
            p.drawEllipse(QtCore.QPointF(r.center().x() + (i - 1) * S.px(18), r.bottom() - S.px(32)), S.px(5), S.px(5))
