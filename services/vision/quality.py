"""Frame quality analysis.

The point of this module is *not* to reject frames.  The cheap decode pass runs
on every frame regardless, because a decoder sometimes succeeds on an image a
metric calls poor.  What quality analysis buys us is:

1. honest, image-derived user feedback ("hold still", "move closer") instead of
   hard-coded guesses, and
2. a reason to skip the expensive upscale/threshold passes on a frame that is
   physically hopeless (heavy motion blur), which is where the CPU budget goes.
"""
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from services.config import DEFAULT_CONFIG, ScannerConfig


class QualityHint(str, Enum):
    """Why a frame is hard to decode, in priority order of usefulness."""
    OK = "OK"
    TOO_DARK = "TOO_DARK"
    TOO_BRIGHT = "TOO_BRIGHT"
    GLARE = "GLARE"
    BLURRY = "BLURRY"
    LOW_CONTRAST = "LOW_CONTRAST"


# User-facing Turkish guidance, one message per measurable cause.
HINT_MESSAGES = {
    QualityHint.OK: "Barkodu kameraya gösterin",
    QualityHint.TOO_DARK: "Barkodu daha aydınlık bir alana çevirin",
    QualityHint.TOO_BRIGHT: "Ortam çok parlak, kamerayı ışıktan uzaklaştırın",
    QualityHint.GLARE: "Parlama var, ürünü hafifçe eğin",
    QualityHint.BLURRY: "Ürünü sabit tutun",
    QualityHint.LOW_CONTRAST: "Barkodun daha görünür olduğundan emin olun",
}


@dataclass(frozen=True)
class FrameQuality:
    sharpness: float
    brightness: float
    contrast: float
    glare_ratio: float
    hint: QualityHint
    #  Sharp enough that heavy enhancement passes have a chance of paying off.
    worth_heavy_passes: bool

    @property
    def message(self) -> str:
        return HINT_MESSAGES[self.hint]

    @property
    def is_good(self) -> bool:
        return self.hint is QualityHint.OK

    def as_dict(self) -> dict:
        return {"sharpness": round(self.sharpness, 1), "brightness": round(self.brightness, 1),
                "contrast": round(self.contrast, 1), "glare_ratio": round(self.glare_ratio, 3),
                "hint": self.hint.value, "message": self.message}


def to_gray(image: np.ndarray) -> np.ndarray:
    """Grayscale view of a BGR or already-gray image."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def analyse_quality(image: np.ndarray, config: ScannerConfig = DEFAULT_CONFIG) -> FrameQuality:
    """Measure sharpness/brightness/contrast/glare on a downscaled copy.

    Metrics are computed at a fixed working width so the thresholds stay
    meaningful when the camera resolution changes -- variance of the Laplacian
    is strongly resolution dependent, so measuring at native 1080p and at 720p
    would otherwise need two different "blurry" thresholds.
    """
    if image is None or image.size == 0:
        return FrameQuality(0.0, 0.0, 0.0, 0.0, QualityHint.BLURRY, False)

    gray = to_gray(image)
    height, width = gray.shape[:2]
    scale = min(1.0, config.quality_work_width / max(1, width))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))),
                          interpolation=cv2.INTER_AREA)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    glare_ratio = float(np.count_nonzero(gray >= 250) / gray.size)

    hint = _classify(sharpness, brightness, contrast, glare_ratio, config)
    worth_heavy = sharpness >= config.heavy_pass_min_sharpness
    return FrameQuality(sharpness, brightness, contrast, glare_ratio, hint, worth_heavy)


def _classify(sharpness, brightness, contrast, glare_ratio, config) -> QualityHint:
    """Pick the single most actionable problem for the user.

    Order matters: lighting problems are reported before blur because a dark
    frame is *also* measured as soft, and telling the user to "hold still" when
    the real problem is the light would send them down the wrong path.
    """
    if brightness < config.brightness_min:
        return QualityHint.TOO_DARK
    if brightness > config.brightness_max:
        return QualityHint.TOO_BRIGHT
    if glare_ratio > config.glare_ratio_max:
        return QualityHint.GLARE
    if sharpness < config.blur_threshold:
        return QualityHint.BLURRY
    if contrast < config.contrast_min:
        return QualityHint.LOW_CONTRAST
    return QualityHint.OK
