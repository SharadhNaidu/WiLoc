"""Top-down map: what the robot believes, on top of what the Wi-Fi is really like.

Layers, bottom to top: Wi-Fi heat map (strongest router, one fixed dBm scale),
room names, furniture, walls coloured by radio material, routers, people and
vehicles, the planned route, particles, each method's estimate, possible
places while lost, and the robot where it really is.  Clicking the floor sends
the robot there; clicking a router asks to unplug or plug it in.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from wiloc.environment import FloorPlan
from wiloc.radio import RadioConfig, RadioSimulator

from .. import theme
from . import style as S


def heat_range(rssi: np.ndarray) -> tuple[float, float]:
    """dBm range the colours span: this venue's own weak-to-strong spread (5 dB steps), as the
    original dashboard did, so a well-covered office is not one flat red."""
    lo = 5.0 * math.floor(float(np.percentile(rssi, 2)) / 5.0)
    hi = 5.0 * math.ceil(float(np.percentile(rssi, 98)) / 5.0)
    return lo, max(hi, lo + 10.0)


def heat_image(fp: FloorPlan, res: float = 0.2) -> tuple[QtGui.QImage, np.ndarray, tuple[float, float]]:
    """RGBA image of the strongest router's mean RSSI (row 0 = north), the array it views, and its dBm range."""
    sim = RadioSimulator(fp, RadioConfig.from_dict(fp.radio))
    nx, ny = max(1, int(math.ceil(fp.width / res))), max(1, int(math.ceil(fp.height / res)))
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * res, (np.arange(ny) + 0.5) * res)
    rssi = sim.mean_rssi(np.column_stack([gx.ravel(), gy.ravel()])).max(axis=1).reshape(ny, nx)
    lo, hi = heat_range(rssi)
    t = np.clip((rssi - lo) / (hi - lo), 0, 1)
    xs = [s for s, _ in S.HEAT_STOPS]
    cols = np.array([theme.rgb(c) for _, c in S.HEAT_STOPS])
    rgb = np.stack([np.interp(t, xs, cols[:, k]) for k in range(3)], axis=-1)
    rgba = np.empty((ny, nx, 4), np.uint8)
    rgba[..., :3] = (rgb * 255).astype(np.uint8)
    rgba[..., 3] = 255
    rgba = np.ascontiguousarray(rgba[::-1])  # image rows go north -> south
    img = QtGui.QImage(rgba.data, nx, ny, 4 * nx, QtGui.QImage.Format_RGBA8888).copy()
    return img, rgba, (lo, hi)


def draw_router(p, c, r, col, off=False):
    """A router: white rounded badge with the Wi-Fi symbol in the router's colour (struck through when off)."""
    p.save()
    box = QtCore.QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
    p.setPen(QtGui.QPen(S.qcolor(S.INK, 0.8), S.px(1.6)))
    p.setBrush(QtGui.QColor(S.CARD))
    p.drawRoundedRect(box, r * 0.35, r * 0.35)
    base = QtCore.QPointF(c.x(), c.y() + r * 0.5)
    pen = QtGui.QPen(col, max(1.5, r * 0.17), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.NoBrush)
    for rad in (r * 0.42, r * 0.78, r * 1.12):  # three arcs, 45 deg either side of up
        arc = QtCore.QRectF(base.x() - rad, base.y() - rad, 2 * rad, 2 * rad)
        p.drawArc(arc, 45 * 16, 90 * 16)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(col)
    p.drawEllipse(base, r * 0.14, r * 0.14)
    if off:
        p.setPen(QtGui.QPen(S.qcolor(S.INK, 0.8), S.px(2), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawLine(QtCore.QPointF(box.left() + r * 0.3, box.bottom() - r * 0.3), QtCore.QPointF(box.right() - r * 0.3, box.top() + r * 0.3))
    p.restore()


def draw_person(p, c, heading, size):
    """A person seen from above: shoulders across the walking direction, head in the middle."""
    p.save()
    p.translate(c)
    p.rotate(-math.degrees(heading))
    w, d = size, size * 0.5
    p.setPen(QtGui.QPen(QtGui.QColor(S.CARD), S.px(1.4)))
    p.setBrush(QtGui.QColor("#3F3A34"))
    p.drawRoundedRect(QtCore.QRectF(-d / 2, -w / 2, d, w), d / 2, d / 2)  # shoulders (x = forward)
    p.setBrush(QtGui.QColor("#C9A27E"))
    p.drawEllipse(QtCore.QPointF(d * 0.08, 0), size * 0.24, size * 0.24)  # head
    p.restore()


class MapView(QtWidgets.QWidget):
    goal_clicked = QtCore.pyqtSignal(float, float)
    ap_clicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fp: FloorPlan | None = None
        self.snap = None
        self.heat = None
        self.heat_dbm = S.HEAT_DBM
        self.disabled_aps: set[str] = set()
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setMinimumSize(320, 200)

    # ------------------------------------------------------------ data
    def set_plan(self, fp: FloorPlan):
        """New venue or router preset: recompute the heat map."""
        self.fp = fp
        self.heat, _, self.heat_dbm = heat_image(fp)
        self.update()

    def set_snapshot(self, snap):
        self.snap = snap
        self.update()

    # ------------------------------------------------------------ geometry
    def _frame(self):
        """World -> widget transform: (scale px/m, origin x px, origin y px) with y up."""
        fp = self.fp
        m = S.px(14)
        w, h = self.width() - 2 * m, self.height() - 2 * m - S.px(40)  # router labels + legend strip below
        k = min(w / fp.width, h / fp.height)
        ox = m + (w - k * fp.width) / 2
        oy = m + (h - k * fp.height) / 2 + k * fp.height
        return k, ox, oy

    def to_px(self, x, y):
        k, ox, oy = self._frame()
        return QtCore.QPointF(ox + k * x, oy - k * y)

    def to_world(self, p: QtCore.QPointF):
        k, ox, oy = self._frame()
        return (p.x() - ox) / k, (oy - p.y()) / k

    # ------------------------------------------------------------ input
    def mousePressEvent(self, e):
        if self.fp is None or e.button() != QtCore.Qt.LeftButton:
            return
        x, y = self.to_world(QtCore.QPointF(e.pos()))
        k, _, _ = self._frame()
        for i, ap in enumerate(self.fp.access_points):
            if math.hypot(ap.x - x, ap.y - y) * k < S.px(16):
                self.ap_clicked.emit(i)
                return
        if 0 <= x <= self.fp.width and 0 <= y <= self.fp.height:
            self.goal_clicked.emit(float(x), float(y))

    # ------------------------------------------------------------ paint
    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        if self.fp is None:
            return
        fp, snap = self.fp, self.snap
        k, ox, oy = self._frame()
        plan_rect = QtCore.QRectF(ox, oy - k * fp.height, k * fp.width, k * fp.height)

        # heat map, clipped to a rounded floor
        path = QtGui.QPainterPath()
        path.addRoundedRect(plan_rect, S.px(6), S.px(6))
        p.save()
        p.setClipPath(path)
        p.fillRect(plan_rect, QtGui.QColor(S.IVORY))
        if self.heat is not None:
            p.setOpacity(0.85)
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)  # bilinear, as the original map
            p.drawImage(plan_rect, self.heat)
            p.setOpacity(1.0)
        p.restore()

        # room names
        p.setFont(S.font(11, S.MEDIUM))
        for name, (x0, y0, x1, y1) in fp.rooms:
            if (x1 - x0) * k < S.px(60):
                continue
            r = QtCore.QRectF(self.to_px(x0, y1), self.to_px(x1, y0))
            p.setPen(S.qcolor(S.MUTED, 0.75))
            p.drawText(r.adjusted(S.px(6), S.px(4), -S.px(6), -S.px(4)), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, name.upper())

        # furniture
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(S.qcolor("#D8D0C3", 0.55))
        for x0, y0, x1, y1 in fp.obstacles:
            p.drawRoundedRect(QtCore.QRectF(self.to_px(x0, y1), self.to_px(x1, y0)), S.px(2), S.px(2))

        # walls
        t_px = max(2.0, k * fp.wall_thickness)
        for (x1, y1, x2, y2), mat in zip(fp.walls, fp.wall_materials):
            pen = QtGui.QPen(S.qcolor(S.WALL_COLORS.get(mat, S.INK), 0.9 if mat != "glass" else 0.8), t_px)
            pen.setCapStyle(QtCore.Qt.SquareCap)
            p.setPen(pen)
            p.drawLine(self.to_px(x1, y1), self.to_px(x2, y2))

        if snap is not None:
            self._paint_dynamic(p, snap, k)
        self._paint_routers(p, k, snap)
        self._paint_legend(p, plan_rect)

    def _paint_routers(self, p, k, snap):
        heard = {}
        if snap is not None and snap.scan:
            heard = {a["id"]: a.get("measured_dbm") for a in snap.scan.get("aps", [])}
        noticed = set((snap.status or {}).get("masked_aps", [])) if snap is not None else set()
        for i, ap in enumerate(self.fp.access_points):
            c = self.to_px(ap.x, ap.y)
            off = ap.id in self.disabled_aps
            col = S.qcolor(S.FAINT if off else theme.ap_color(i))
            r = S.px(11)
            draw_router(p, c, r, col, off)
            p.setFont(S.font(11, S.SEMIBOLD))
            p.setPen(QtGui.QColor(S.MUTED if off else S.INK))
            label = ap.id + ((" off · noticed" if ap.id in noticed else " off") if off else "")
            if off and ap.id in noticed:
                p.setPen(QtGui.QPen(S.qcolor(S.INDIGO, 0.9), S.px(1.6), QtCore.Qt.DashLine))
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawEllipse(c, r + S.px(5), r + S.px(5))
                p.setPen(QtGui.QColor(S.INDIGO))
            p.drawText(QtCore.QRectF(c.x() - S.px(40), c.y() + r + S.px(1), S.px(80), S.px(16)), QtCore.Qt.AlignHCenter, label)

    def _paint_dynamic(self, p, snap, k):
        fp = self.fp
        # people and vehicles (positions follow sim time, like Gazebo's actors)
        if snap.sim_t is not None and len(fp.movers):
            t = snap.sim_t
            for mv in fp.movers.movers:
                x, y = mv.position(t)
                c = self.to_px(x, y)
                if mv.type == "person":
                    draw_person(p, c, mv.heading(t), max(S.px(19), 0.55 * k))
                else:
                    p.save()
                    p.translate(c)
                    p.rotate(-math.degrees(mv.heading(t)))
                    L, W = {"forklift": (2.2, 1.1), "agv": (1.2, 0.8), "gantry": (0.6, 3.6), "cart": (0.7, 0.5)}.get(mv.type, (1, 1))
                    p.setPen(QtGui.QPen(QtGui.QColor(S.CARD), S.px(1.5)))
                    p.setBrush(S.qcolor("#E7B416" if mv.type in ("forklift", "gantry") else "#5D6B78", 0.9))
                    p.drawRoundedRect(QtCore.QRectF(-L * k / 2, -W * k / 2, L * k, W * k), S.px(3), S.px(3))
                    p.restore()

        # planned route and goal
        if len(snap.plan) >= 2:
            pen = QtGui.QPen(S.qcolor(S.INK, 0.55), S.px(2), QtCore.Qt.DashLine)
            pen.setDashPattern([3, 3])
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawPolyline(QtGui.QPolygonF([self.to_px(x, y) for x, y in snap.plan]))
            g = self.to_px(*snap.plan[-1])
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(S.qcolor(S.INK, 0.75))
            p.drawEllipse(g, S.px(3.5), S.px(3.5))

        # particles
        parts = snap.particles
        if len(parts):
            kk, ox, oy = self._frame()
            pts = [QtCore.QPointF(ox + kk * x, oy - kk * y) for x, y in parts[:, :2]]
            pen = QtGui.QPen(S.qcolor(S.INDIGO, 0.28 if len(parts) > 1500 else 0.4), S.px(2.6))
            pen.setCapStyle(QtCore.Qt.RoundCap)
            p.setPen(pen)
            p.drawPoints(QtGui.QPolygonF(pts))

        # possible places while unsure (the filter's top hypotheses)
        st = snap.status or {}
        modes = st.get("modes") or []
        if len(modes) > 1 and st.get("spread_m", 0) > 1.0:
            p.setFont(S.font(12, S.SEMIBOLD))
            for x, y, w in modes[:3]:
                if w < 0.08:
                    continue
                c = self.to_px(x, y)
                r = S.px(16)
                p.setPen(QtGui.QPen(S.qcolor(S.INDIGO, 0.9), S.px(2), QtCore.Qt.DashLine))
                p.setBrush(S.qcolor(S.CARD, 0.35))
                p.drawEllipse(c, r, r)
                p.setPen(QtGui.QColor(S.INDIGO))
                p.drawText(QtCore.QRectF(c.x() - S.px(30), c.y() - r - S.px(18), S.px(60), S.px(16)), QtCore.Qt.AlignHCenter,
                           f"{w * 100:.0f}%")

        # each method's estimate
        for m in ("wknn", "ekf", "pf"):
            if m not in snap.poses:
                continue
            x, y, yaw = snap.poses[m]
            c = self.to_px(x, y)
            col = QtGui.QColor(S.METHOD_COLORS[m])
            if m == "pf":
                r = S.px(9)
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(S.qcolor(S.INK, 0.75))
                p.drawEllipse(c, r + S.px(2.6), r + S.px(2.6))
                p.setPen(QtGui.QPen(QtGui.QColor(S.CARD), S.px(2.5)))
                p.setBrush(col)
                p.drawEllipse(c, r, r)
                p.setPen(QtGui.QPen(QtGui.QColor(S.CARD), S.px(2.2)))
                p.drawLine(c, c + QtCore.QPointF(math.cos(yaw) * r * 0.95, -math.sin(yaw) * r * 0.95))
            elif m == "ekf":
                r = S.px(8)
                p.setBrush(QtCore.Qt.NoBrush)
                p.setPen(QtGui.QPen(S.qcolor(S.INK, 0.75), S.px(6)))
                p.drawEllipse(c, r, r)
                p.setPen(QtGui.QPen(QtGui.QColor(S.CARD), S.px(4.4)))
                p.drawEllipse(c, r, r)
                p.setPen(QtGui.QPen(col, S.px(3)))
                p.drawEllipse(c, r, r)
            else:
                r = S.px(7)
                p.setPen(QtGui.QPen(S.qcolor(S.INK, 0.75), S.px(4.5)))
                p.setBrush(col)
                diamond = QtGui.QPolygonF([c + QtCore.QPointF(0, -r), c + QtCore.QPointF(r, 0), c + QtCore.QPointF(0, r), c + QtCore.QPointF(-r, 0)])
                p.drawPolygon(diamond)
                p.setPen(QtGui.QPen(QtGui.QColor(S.CARD), S.px(1.8)))
                p.drawPolygon(diamond)

        # the robot where it really is
        if snap.truth is not None:
            x, y, yaw = snap.truth
            c = self.to_px(x, y)
            r = max(S.px(9), 0.2 * k)
            p.save()
            p.translate(c)
            p.rotate(-math.degrees(yaw))
            p.setPen(QtGui.QPen(QtGui.QColor(S.CARD), S.px(2)))
            p.setBrush(QtGui.QColor(S.INK))
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(r * 1.25, 0), QtCore.QPointF(-r * 0.8, r * 0.8), QtCore.QPointF(-r * 0.35, 0), QtCore.QPointF(-r * 0.8, -r * 0.8)]))
            p.restore()

    def _paint_legend(self, p, plan_rect):
        y = plan_rect.bottom() + S.px(24)
        x = plan_rect.left()
        p.setFont(S.font(11))
        p.setPen(QtGui.QColor(S.MUTED))
        txt = "Wi-Fi signal"
        fm = QtGui.QFontMetrics(p.font())
        p.drawText(QtCore.QPointF(x, y + S.px(11)), txt)
        bx = x + fm.horizontalAdvance(txt) + S.px(8)
        bw, bh = S.px(72), S.px(8)
        g = QtGui.QLinearGradient(bx, 0, bx + bw, 0)
        for s, c in S.HEAT_STOPS:
            g.setColorAt(s, QtGui.QColor(c))
        p.setPen(QtGui.QPen(QtGui.QColor(S.HAIRLINE), 1))
        p.setBrush(g)
        p.drawRoundedRect(QtCore.QRectF(bx, y + S.px(3), bw, bh), bh / 2, bh / 2)
        p.setPen(QtGui.QColor(S.FAINT))
        lo, hi = self.heat_dbm
        p.drawText(QtCore.QPointF(bx + bw + S.px(6), y + S.px(11)), f"{lo:.0f} to {hi:.0f} dBm")
        # marker key
        kx = bx + bw + S.px(6) + fm.horizontalAdvance(f"{lo:.0f} to {hi:.0f} dBm") + S.px(18)
        items = [("router", "Router"), ("person", "Person"), ("truth", "Real robot"), ("pf", "Particle filter"),
                 ("ekf", "Kalman"), ("wknn", "Wi-Fi only")]
        for key, label in items:
            c = QtCore.QPointF(kx + S.px(6), y + S.px(7))
            if key == "router":
                draw_router(p, c, S.px(7), QtGui.QColor(S.INK))
            elif key == "person":
                draw_person(p, c, math.pi / 2, S.px(14))
            elif key == "truth":
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(S.INK))
                r = S.px(6)
                p.drawPolygon(QtGui.QPolygonF([c + QtCore.QPointF(r, 0), c + QtCore.QPointF(-r * 0.8, r * 0.8), c + QtCore.QPointF(-r * 0.8, -r * 0.8)]))
            elif key == "ekf":
                p.setPen(QtGui.QPen(QtGui.QColor(S.METHOD_COLORS[key]), S.px(2)))
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawEllipse(c, S.px(5), S.px(5))
            elif key == "wknn":
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(S.METHOD_COLORS[key]))
                r = S.px(5.5)
                p.drawPolygon(QtGui.QPolygonF([c + QtCore.QPointF(0, -r), c + QtCore.QPointF(r, 0), c + QtCore.QPointF(0, r), c + QtCore.QPointF(-r, 0)]))
            else:
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(S.METHOD_COLORS[key]))
                p.drawEllipse(c, S.px(5.5), S.px(5.5))
            p.setPen(QtGui.QColor(S.MUTED))
            p.drawText(QtCore.QPointF(kx + S.px(16), y + S.px(11)), label)
            kx += S.px(16) + fm.horizontalAdvance(label) + S.px(10)
            if kx > self.width() - S.px(60):
                break
