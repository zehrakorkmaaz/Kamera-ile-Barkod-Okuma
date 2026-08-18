"""Confidence fusion, with a seat reserved for load-cell verification.

The ESP32/HX711 hardware is not part of this build.  What *is* built is the
seam it will plug into: the scanner already produces a `VisionSignal`, the
catalogue already carries an expected weight, and `combine_confidence` already
knows how to merge a `WeightSignal` when one exists.  Until a real
`WeightSource` is wired in, `NullWeightSource` returns nothing and the final
confidence is exactly the vision confidence -- today's behaviour, unchanged.

Adding the load cell later means implementing one method:

    class Hx711WeightSource:
        def read_delta_grams(self) -> float | None: ...
"""
from dataclasses import dataclass
from typing import Protocol
import re

from services.config import DEFAULT_CONFIG, ScannerConfig
from services.vision.confidence import ConfidenceLevel, level_for


#  Vision keeps the larger share: a barcode identifies the exact item, whereas
#  weight only tells you the item is plausible.
VISION_SHARE = 0.65
WEIGHT_SHARE = 0.35
#  Fraction of the expected weight a reading may deviate by and still agree.
DEFAULT_TOLERANCE = 0.08


class WeightSource(Protocol):
    """Anything that can report the cart's weight change for the last scan."""

    def read_delta_grams(self) -> float | None:
        ...


class NullWeightSource:
    """The only implementation today: no scale attached, so no weight evidence."""

    def read_delta_grams(self) -> float | None:
        return None


@dataclass(frozen=True)
class VisionSignal:
    barcode: str
    confidence: float
    level: ConfidenceLevel

    def as_dict(self) -> dict:
        return {"barcode": self.barcode, "confidence": round(self.confidence, 1),
                "level": self.level.value}


@dataclass(frozen=True)
class WeightSignal:
    """Measured cart weight change versus the catalogue weight of the product."""
    measured_grams: float
    expected_grams: float | None
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def agrees(self) -> bool:
        if not self.expected_grams:
            return False
        allowed = max(self.expected_grams * self.tolerance, 5.0)
        return abs(self.measured_grams - self.expected_grams) <= allowed

    @property
    def confidence(self) -> float:
        """100 at the expected weight, decaying to 0 at three times tolerance."""
        if not self.expected_grams:
            return 0.0
        allowed = max(self.expected_grams * self.tolerance, 5.0)
        error = abs(self.measured_grams - self.expected_grams)
        return float(max(0.0, min(100.0, 100.0 * (1.0 - error / (allowed * 3)))))

    def as_dict(self) -> dict:
        return {"measured_grams": round(self.measured_grams, 1),
                "expected_grams": self.expected_grams, "agrees": self.agrees,
                "confidence": round(self.confidence, 1)}


@dataclass(frozen=True)
class FinalConfidence:
    score: float
    level: ConfidenceLevel
    vision_confidence: float
    weight_confidence: float | None
    weight_verified: bool | None

    def as_dict(self) -> dict:
        return {"final_confidence": round(self.score, 1), "level": self.level.value,
                "vision_confidence": round(self.vision_confidence, 1),
                "weight_confidence": (None if self.weight_confidence is None
                                      else round(self.weight_confidence, 1)),
                "weight_verified": self.weight_verified}


def combine_confidence(vision: VisionSignal, weight: WeightSignal | None = None,
                       config: ScannerConfig = DEFAULT_CONFIG) -> FinalConfidence:
    """Merge vision and (optional) weight evidence into one score."""
    if weight is None or weight.expected_grams is None:
        return FinalConfidence(vision.confidence, vision.level, vision.confidence, None, None)
    score = vision.confidence * VISION_SHARE + weight.confidence * WEIGHT_SHARE
    return FinalConfidence(score, level_for(score, config), vision.confidence,
                           weight.confidence, weight.agrees)


_WEIGHT_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|g|gr|ml|lt|l)?", re.IGNORECASE)
_TO_GRAMS = {"kg": 1000.0, "lt": 1000.0, "l": 1000.0, "g": 1.0, "gr": 1.0, "ml": 1.0}


def expected_weight_grams(weight_text: str | None) -> float | None:
    """Read grams out of the catalogue's free-text weight field.

    The existing `products.weight` column holds strings like "1050 g", "1000g"
    or "1,5 kg", entered by hand.  Parsing it here means weight verification can
    be switched on later without a schema change or re-entering the catalogue.
    """
    if not weight_text:
        return None
    match = _WEIGHT_PATTERN.search(str(weight_text))
    if not match:
        return None
    try:
        amount = float(match.group(1).replace(",", "."))
    except ValueError:
        return None
    unit = (match.group(2) or "g").lower()
    grams = amount * _TO_GRAMS.get(unit, 1.0)
    return round(grams, 2) if grams > 0 else None
