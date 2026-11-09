"""Record the Studio window to a video file (the backup video for the stand).

Frames are grabbed on the GUI thread (``QWidget.grab``) at ``fps`` and
encoded on a worker thread with OpenCV: MPEG-4 (.mp4), or Motion-JPEG (.avi)
if this OpenCV build cannot write MPEG-4.
"""

from __future__ import annotations

import os
import queue
import threading

import numpy as np
from PyQt5 import QtCore, QtGui


class Recorder:
    def __init__(self, widget, path: str, fps: float = 15.0, width: int = 0):
        self.widget, self.path, self.fps = widget, path, fps
        self.width = width  # 0: the window's own size
        self.frames: queue.Queue = queue.Queue(maxsize=int(fps * 4))
        self.dropped = self.written = 0
        self.size = None
        self.worker = threading.Thread(target=self._encode, daemon=True)
        self.timer = QtCore.QTimer(widget, timeout=self._grab, interval=int(1000 / fps))

    def start(self):
        self.worker.start()
        self.timer.start()

    def _grab(self):
        img = self.widget.grab().toImage()
        if self.width and img.width() != self.width:
            img = img.scaledToWidth(self.width, QtCore.Qt.SmoothTransformation)
        img = img.convertToFormat(QtGui.QImage.Format_RGB888)
        if self.size is None:
            self.size = (img.width() // 2 * 2, img.height() // 2 * 2)  # codecs want even sizes
        w, h = img.width(), img.height()
        ptr = img.constBits()
        ptr.setsize(img.bytesPerLine() * h)
        rgb = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3).copy()
        try:
            self.frames.put_nowait(rgb)
        except queue.Full:
            self.dropped += 1

    def _open(self, cv2):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        if not writer.isOpened():
            self.path = os.path.splitext(self.path)[0] + ".avi"
            writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, self.size)
        return writer

    def _encode(self):
        import cv2

        writer = None
        while True:
            rgb = self.frames.get()
            if rgb is None:
                break
            if writer is None:
                writer = self._open(cv2)
            if (rgb.shape[1], rgb.shape[0]) != self.size:
                rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.written += 1
        if writer is not None:
            writer.release()

    def stop(self) -> str:
        """Finish the file; returns its path (the extension may have changed to .avi)."""
        self.timer.stop()
        self.frames.put(None)
        self.worker.join(timeout=30)
        return self.path
