"""Run the scanner from a folder of images instead of a camera.

`FolderFrameSource` implements the same small surface as `CameraDevice`, so the
whole application -- capture loop, pipeline, tracker, preview stream and web UI
-- runs unchanged against saved pictures:

    SMARTCART_TEST_IMAGE_DIR=test_images python app.py

That makes the image pipeline testable with no hardware attached, and makes a
bad read reproducible: drop the picture that failed into the folder and watch
the same frame go through the real code path.
"""
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

import cv2
import numpy as np

from services.camera_device import CameraProfile
from services.config import DEFAULT_CONFIG, ScannerConfig

logger = logging.getLogger("smartcart.camera")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_images(folder) -> list[tuple[str, np.ndarray]]:
    """Read every image in a folder, sorted by name."""
    directory = Path(folder)
    if not directory.is_dir():
        return []
    images = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None and image.size:
            images.append((path.name, image))
        else:
            logger.warning("TEST_IMAGE_UNREADABLE %s", path.name)
    return images


def letterbox(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fit an image into a camera-sized frame without distorting it.

    The scan guide is defined as a fraction of the frame, so test images have to
    sit in a frame shaped like the camera's for the ROI to mean the same thing.
    """
    target_w, target_h = size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    resized = cv2.resize(image, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((target_h, target_w, 3), np.uint8)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


@dataclass
class FolderFrameSource:
    """A `CameraDevice` stand-in that replays images from a folder."""
    folder: str
    config: ScannerConfig = DEFAULT_CONFIG
    index: int = 0
    fps: float = 10.0
    #  Seconds each image is shown, so temporal confirmation behaves as it would
    #  with a product held in front of a real camera.
    seconds_per_image: float = 1.5
    loop: bool = True
    profile: CameraProfile = field(default_factory=CameraProfile)
    capabilities: dict = field(default_factory=dict)
    error: str | None = None
    consecutive_failures: int = 0
    reconnects: int = 0
    _images: list = field(default_factory=list, repr=False)
    _position: int = 0
    _shown_since: float = 0.0
    _last_frame_at: float = 0.0

    def open(self) -> bool:
        self._images = load_images(self.folder)
        if not self._images:
            self.error = f"Test görüntüsü bulunamadı: {self.folder}"
            logger.warning("TEST_MODE_EMPTY folder=%s", self.folder)
            return False
        size = (self.config.camera_width, self.config.camera_height)
        self._images = [(name, letterbox(image, size)) for name, image in self._images]
        self.profile = CameraProfile(width=size[0], height=size[1], fps=self.fps,
                                     fourcc="TEST", backend=f"test-mode:{self.folder}",
                                     requested=size, autofocus=None)
        self.capabilities = {"test_mode": {"supported": True, "value": len(self._images)}}
        self.error = None
        self._position, self._shown_since = 0, time.monotonic()
        logger.info("TEST_MODE_READY folder=%s images=%d", self.folder, len(self._images))
        return True

    def close(self) -> None:
        self._images = []

    def is_open(self) -> bool:
        return bool(self._images)

    @property
    def current_name(self) -> str:
        return self._images[self._position][0] if self._images else ""

    def read(self):
        """Return the current image, advancing on a fixed schedule."""
        if not self._images:
            self.consecutive_failures += 1
            return False, None
        now = time.monotonic()
        # Pace the replay so the scan loop is not spinning on identical frames.
        wait = (1.0 / self.fps) - (now - self._last_frame_at)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        self._last_frame_at = now
        if now - self._shown_since >= self.seconds_per_image:
            self._shown_since = now
            if self._position + 1 >= len(self._images) and not self.loop:
                return False, None
            self._position = (self._position + 1) % len(self._images)
            logger.info("TEST_IMAGE %s", self.current_name)
        self.consecutive_failures = 0
        return True, self._images[self._position][1].copy()

    def needs_reconnect(self) -> bool:
        return False

    def reconnect(self) -> bool:
        return self.open()

    def probe_resolutions(self, resolutions=None) -> list[dict]:
        return [{"width": self.profile.width, "height": self.profile.height,
                 "requested": "test-mode", "fps": self.fps, "fourcc": "TEST", "exact": True}]

    def measure_fps(self, seconds: float = 2.0) -> float:
        return self.fps
