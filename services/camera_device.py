"""USB/built-in camera device handling: profile negotiation and recovery.

Two rules drive this module.

*Verify, never assume*: asking OpenCV for 1920x1080 is a request, not a result.
The driver may quietly hand back 640x480, or grant the resolution at 5 fps.  So
every profile is applied, then confirmed by actually reading a frame, and the
first profile that works is kept.

*Probe, do not meddle*: brightness, contrast, saturation, sharpness and white
balance are read and reported but never overwritten.  Cameras ship with sane
auto settings, and a blind write is far more likely to make barcode reading
worse than better.  Only the settings that demonstrably matter -- pixel format,
resolution, frame rate, autofocus and a short buffer -- are set.
"""
from dataclasses import dataclass, field
import logging
import time

import cv2

from services.config import DEFAULT_CONFIG, ScannerConfig

logger = logging.getLogger("smartcart.camera")

# Backward-compatible aliases; prefer the index-aware helpers below.
PERMISSION_HINT = "macOS: System Settings > Privacy & Security > Camera."
PERMISSION_ERROR = f"Camera could not be opened. {PERMISSION_HINT}"
READ_ERROR = "Could not read frames from the camera. Check the connection."


def open_failed_message(index: int, *, permission_hint: bool = True) -> str:
    """Human-readable message when ``VideoCapture`` fails to open an index."""
    message = f"Camera {index} could not be opened."
    if permission_hint:
        message = f"{message} {PERMISSION_HINT}"
    return message


def read_failed_message(index: int) -> str:
    """Human-readable message when consecutive frame reads fail."""
    return f"Camera {index}: could not read frames. Check the connection."


def invalid_index_message(index: int) -> str:
    return f"Invalid camera index {index}. Use 0 or a positive integer."

#  Read-only diagnostics: reported in the capability report, never written.
PROBE_PROPERTIES = {
    "brightness": "CAP_PROP_BRIGHTNESS", "contrast": "CAP_PROP_CONTRAST",
    "saturation": "CAP_PROP_SATURATION", "sharpness": "CAP_PROP_SHARPNESS",
    "gain": "CAP_PROP_GAIN", "exposure": "CAP_PROP_EXPOSURE",
    "auto_exposure": "CAP_PROP_AUTO_EXPOSURE", "autofocus": "CAP_PROP_AUTOFOCUS",
    "focus": "CAP_PROP_FOCUS", "auto_white_balance": "CAP_PROP_AUTO_WB",
    "white_balance": "CAP_PROP_WB_TEMPERATURE", "zoom": "CAP_PROP_ZOOM",
}


def _backend_name(capture) -> str:
    """Backend label, tolerating captures that do not implement the call."""
    try:
        return str(capture.getBackendName())
    except (AttributeError, cv2.error):
        return ""


@dataclass
class CameraProfile:
    """What the camera actually granted, as opposed to what was requested."""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fourcc: str = ""
    backend: str = ""
    requested: tuple[int, int] = (0, 0)
    autofocus: bool | None = None

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fps": round(self.fps, 1),
                "fourcc": self.fourcc, "backend": self.backend,
                "requested": list(self.requested), "autofocus": self.autofocus,
                "resolution": f"{self.width}x{self.height}"}


@dataclass
class CameraDevice:
    """A `cv2.VideoCapture` that negotiates a usable profile and can recover."""
    config: ScannerConfig = DEFAULT_CONFIG
    index: int = 0
    capture: cv2.VideoCapture | None = field(default=None, repr=False)
    profile: CameraProfile = field(default_factory=CameraProfile)
    capabilities: dict = field(default_factory=dict)
    error: str | None = None
    consecutive_failures: int = 0
    reconnects: int = 0
    _backoff: float = 0.0

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> bool:
        """Open the device and negotiate the best profile it will actually give."""
        self.close()
        if self.index < 0:
            self.error = invalid_index_message(self.index)
            logger.warning("CAMERA_INVALID_INDEX index=%s", self.index)
            return False

        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.index, backend)
        if not capture.isOpened():
            capture.release()
            self.error = open_failed_message(self.index)
            logger.warning("CAMERA_OPEN_FAILED index=%s", self.index)
            return False

        self.capture = capture
        profile = self._negotiate(capture)
        if profile is None:
            self.close()
            self.error = read_failed_message(self.index)
            return False

        self.profile = profile
        self.capabilities = self._probe(capture)
        self.error = None
        self.consecutive_failures = 0
        self._backoff = 0.0
        logger.info("CAMERA_READY %s", profile.as_dict())
        return True

    def close(self) -> None:
        if self.capture is not None:
            try:
                self.capture.release()
            except cv2.error:
                pass
        self.capture = None

    def is_open(self) -> bool:
        return self.capture is not None

    # -- frames ------------------------------------------------------------

    def read(self):
        """Read one frame, returning (ok, frame); tracks failures for recovery."""
        if self.capture is None:
            return False, None
        try:
            ok, frame = self.capture.read()
        except cv2.error as exc:
            logger.warning("CAMERA_READ_ERROR %s", exc)
            ok, frame = False, None
        if ok and frame is not None and frame.size:
            self.consecutive_failures = 0
            return True, frame
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.config.reconnect_after_failures:
            self.error = read_failed_message(self.index)
        return False, None

    def needs_reconnect(self) -> bool:
        return self.consecutive_failures >= self.config.reconnect_after_failures

    def reconnect(self) -> bool:
        """Reopen after the camera was unplugged, slept, or taken by another app."""
        self._backoff = min(self.config.reconnect_max_backoff_seconds,
                            max(self.config.reconnect_backoff_seconds, self._backoff * 2))
        logger.info("CAMERA_RECONNECT index=%s backoff=%.1fs", self.index, self._backoff)
        time.sleep(self._backoff)
        self.consecutive_failures = 0
        if self.open():
            self.reconnects += 1
            return True
        return False

    # -- negotiation and probing -------------------------------------------

    def _candidate_resolutions(self) -> list[tuple[int, int]]:
        wanted = (self.config.camera_width, self.config.camera_height)
        ladder = [wanted, *self.config.camera_fallback_resolutions]
        return list(dict.fromkeys(ladder))

    def _negotiate(self, capture) -> CameraProfile | None:
        """Try resolutions in order and keep the first that really delivers frames."""
        wanted = (self.config.camera_width, self.config.camera_height)
        fallback = None
        for width, height in self._candidate_resolutions():
            self._apply(capture, width, height)
            ok, frame = self._warm_up(capture)
            if not ok:
                continue
            actual_h, actual_w = frame.shape[:2]
            profile = CameraProfile(
                width=actual_w, height=actual_h, fps=self._reported_fps(capture),
                fourcc=self._fourcc(capture), backend=_backend_name(capture),
                requested=(width, height), autofocus=self._enable_autofocus(capture))
            # A driver that answers 1080p with 5 fps is worse than clean 720p.
            if profile.fps and profile.fps < 15 and (width, height) != self.config.camera_fallback_resolutions[-1]:
                logger.info("CAMERA_PROFILE_REJECTED %s (%.1f fps)", profile.as_dict(), profile.fps)
                fallback = fallback or profile
                continue
            if (actual_w, actual_h) != (width, height):
                logger.info("CAMERA_PROFILE_SUBSTITUTED requested=%dx%d granted=%dx%d",
                            width, height, actual_w, actual_h)
            return profile
        if fallback is not None:
            return fallback
        logger.warning("CAMERA_NO_USABLE_PROFILE requested=%dx%d", *wanted)
        return None

    def _apply(self, capture, width: int, height: int) -> None:
        if self.config.camera_fourcc:
            capture.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*self.config.camera_fourcc[:4]))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        # A one-frame buffer keeps `read()` on the newest image instead of
        # replaying a queue the driver built up while we were decoding.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    @staticmethod
    def _warm_up(capture, attempts: int = 6):
        """Some drivers return empty frames right after a format change."""
        for _ in range(attempts):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                return True, frame
            time.sleep(0.05)
        return False, None

    def _enable_autofocus(self, capture) -> bool | None:
        """Autofocus is the one setting worth asking for: it decides sharpness."""
        if not self.config.camera_autofocus:
            return None
        current = capture.get(cv2.CAP_PROP_AUTOFOCUS)
        if current == -1:
            return None          # property not supported by this camera
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        return bool(capture.get(cv2.CAP_PROP_AUTOFOCUS))

    @staticmethod
    def _reported_fps(capture) -> float:
        fps = capture.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 0.0

    @staticmethod
    def _fourcc(capture) -> str:
        value = int(capture.get(cv2.CAP_PROP_FOURCC))
        if not value:
            return ""
        return "".join(chr((value >> shift) & 0xFF) for shift in (0, 8, 16, 24)).strip()

    @staticmethod
    def _probe(capture) -> dict:
        """Report which controls this camera exposes, without changing any."""
        found = {}
        for name, constant in PROBE_PROPERTIES.items():
            prop = getattr(cv2, constant, None)
            if prop is None:
                continue
            value = capture.get(prop)
            found[name] = {"supported": value != -1, "value": round(float(value), 3)}
        return found

    def probe_resolutions(self, resolutions=None) -> list[dict]:
        """Which resolution/FPS combinations this camera really supports.

        Used by the diagnostics script.  Every entry is verified by reading a
        frame, because `set()` succeeding means nothing on most drivers.
        """
        if self.capture is None:
            return []
        resolutions = resolutions or [(3840, 2160), (1920, 1080), (1600, 896), (1280, 720),
                                      (1024, 576), (800, 600), (640, 480), (320, 240)]
        original = (self.profile.width, self.profile.height)
        supported = []
        for width, height in resolutions:
            self._apply(self.capture, width, height)
            ok, frame = self._warm_up(self.capture, attempts=3)
            if not ok:
                continue
            actual_h, actual_w = frame.shape[:2]
            entry = {"width": actual_w, "height": actual_h,
                     "requested": f"{width}x{height}", "fps": self._reported_fps(self.capture),
                     "fourcc": self._fourcc(self.capture),
                     "exact": (actual_w, actual_h) == (width, height)}
            if entry not in supported:
                supported.append(entry)
        if original != (0, 0):
            self._apply(self.capture, *original)
            self._warm_up(self.capture, attempts=3)
        return supported

    def measure_fps(self, seconds: float = 2.0) -> float:
        """Actually measured frame rate -- drivers frequently misreport it."""
        if self.capture is None:
            return 0.0
        frames, started = 0, time.monotonic()
        while time.monotonic() - started < seconds:
            ok, _ = self.read()
            if ok:
                frames += 1
        elapsed = time.monotonic() - started
        return round(frames / elapsed, 1) if elapsed else 0.0
