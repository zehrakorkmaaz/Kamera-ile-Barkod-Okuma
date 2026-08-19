"""Lightweight camera discovery — separate from opening a scanner session.

Discovery only checks whether an index can deliver at least one frame and
reports basic technical details.  It does not run profile negotiation, autofocus
setup, or the barcode pipeline.  Use ``CameraDevice.open()`` when you actually
want to scan.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
import time

import cv2

from services.camera_device import CameraDevice, open_failed_message


# OpenCV / AVFoundation do not expose stable, user-friendly device names on every
# platform.  When no name is available we show index + backend + resolution.
NAMES_UNAVAILABLE_NOTE = (
    "OpenCV bu platformda güvenilir kamera adı sağlamıyor; "
    "index ve teknik bilgiler gösteriliyor."
)


def _default_backend():
    return cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY


@dataclass(frozen=True)
class DiscoveredCamera:
    index: int
    available: bool
    name: str | None = None
    backend: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fourcc: str = ""
    error: str | None = None

    def label(self) -> str:
        if self.name:
            return self.name
        return f"Camera {self.index}"

    def as_dict(self) -> dict:
        return {"index": self.index, "available": self.available, "name": self.name,
                "label": self.label(), "backend": self.backend,
                "resolution": f"{self.width}x{self.height}" if self.width else "",
                "width": self.width, "height": self.height,
                "fps": round(self.fps, 1) if self.fps else 0.0,
                "fourcc": self.fourcc, "error": self.error}


def _suppress_opencv_logs():
    """Return a restore callback; no-op when OpenCV logging API is unavailable."""
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        previous = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        return lambda: cv2.utils.logging.setLogLevel(previous)
    return lambda: None


@contextmanager
def _silence_stderr():
    """Hide OpenCV C++ stderr spam when probing a missing camera index."""
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)


def probe_index(index: int, backend=None) -> DiscoveredCamera:
    """Try to open one index and read a single frame; always releases the device."""
    if index < 0:
        return DiscoveredCamera(index=index, available=False,
                                error=f"Invalid camera index {index}.")

    restore_logs = _suppress_opencv_logs()
    try:
        backend = _default_backend() if backend is None else backend
        with _silence_stderr():
            capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            return DiscoveredCamera(index=index, available=False,
                                    error=open_failed_message(index, permission_hint=False))

        backend_name = _backend_name(capture)
        ok, frame = False, None
        for _ in range(4):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                break
            time.sleep(0.05)

        fps_value = capture.get(cv2.CAP_PROP_FPS)
        fps = float(fps_value) if fps_value and fps_value > 0 else 0.0
        fourcc = CameraDevice._fourcc(capture)
        capture.release()

        if not ok or frame is None:
            return DiscoveredCamera(index=index, available=False, backend=backend_name,
                                    error=f"Camera {index} opened but no frame could be read.")

        height, width = frame.shape[:2]
        return DiscoveredCamera(index=index, available=True, backend=backend_name,
                                width=width, height=height, fps=fps, fourcc=fourcc)
    finally:
        restore_logs()


def discover_cameras(max_index: int = 9, stop_after_misses: int = 3) -> list[DiscoveredCamera]:
    """Return indices in ``0..max_index`` that can deliver at least one frame.

    Scanning stops after ``stop_after_misses`` consecutive unavailable indices so
    single-camera systems do not trigger pointless OpenCV warnings.
    """
    if max_index < 0:
        return []
    available = []
    misses = 0
    for index in range(max_index + 1):
        camera = probe_index(index)
        if camera.available:
            available.append(camera)
            misses = 0
            continue
        misses += 1
        if misses >= stop_after_misses:
            break
    return available


def _backend_name(capture) -> str:
    try:
        return str(capture.getBackendName())
    except (AttributeError, cv2.error):
        return ""


def render_discovery_list(cameras: list[DiscoveredCamera]) -> str:
    lines = ["Available Cameras", ""]
    if not cameras:
        lines.append("  (hiç kamera bulunamadı)")
        lines.append("")
        lines.append(f"  Not: {NAMES_UNAVAILABLE_NOTE}")
        return "\n".join(lines)

    lines.append(f"  Not: {NAMES_UNAVAILABLE_NOTE}")
    lines.append("")
    for camera in cameras:
        lines.append(f"[{camera.index}] {camera.label()}")
        lines.append(f"    Backend: {camera.backend or 'bilinmiyor'}")
        lines.append(f"    Resolution: {camera.width}x{camera.height}")
        lines.append(f"    FPS: {camera.fps or '?'}")
        if camera.fourcc:
            lines.append(f"    Pixel format: {camera.fourcc}")
        else:
            lines.append("    Pixel format: bilinmiyor (henüz müzakere edilmedi)")
        lines.append("")
    return "\n".join(lines).rstrip()
