"""Frame-only barcode decoding API, independent of HTTP, database and camera.

This module is now a thin front door onto `services.vision`, which holds the
real implementation (quality analysis, detection, rectification, multi-pass
decoding, confidence and tracking).  The functions here keep their original
signatures so existing callers and tests are unaffected.
"""
from dataclasses import dataclass
import time

from services.config import DEFAULT_CONFIG, ScannerConfig
from services.vision.decoding import (  # re-exported for backwards compatibility
    SUPPORTED_FORMATS, expand_upce, has_valid_mod10_checksum, is_valid_barcode,
    lookup_aliases, normalize_barcode,
)
from services.vision.pipeline import BarcodePipeline

__all__ = ["SUPPORTED_FORMATS", "BarcodeDebouncer", "BarcodePresenceTracker",
           "BarcodeStabilizer", "expand_upce", "has_valid_mod10_checksum",
           "is_valid_barcode", "lookup_aliases", "normalize_barcode",
           "scan_barcode", "scan_barcode_details"]

_default_pipeline = BarcodePipeline(DEFAULT_CONFIG)


def scan_barcode_details(frame, roi=None, config: ScannerConfig | None = None):
    """Decode a frame and return (value, diagnostics), without altering `frame`.

    `roi` is the visible scanning guide in camera coordinates and is used as the
    primary search image; the decoder never receives a mirrored display frame.
    """
    pipeline = _default_pipeline if config is None else BarcodePipeline(config)
    source = roi if roi is not None and roi.size else frame
    result = pipeline.process(source)
    sizes = [[candidate.width, candidate.height] for candidate in result.candidates]
    diagnostics = {
        "source_size": [int(source.shape[1]), int(source.shape[0])],
        "candidate_count": len(result.candidates), "candidate_sizes": sizes,
        # A candidate below this size is the distant-barcode case, which is what
        # the heavier upscaling passes exist for.
        "small_barcode_mode": any(max(size) < 380 for size in sizes),
        "attempts": result.attempts, "method": result.method or None,
        "confidence": result.confidence, "level": result.level.value,
        "format": result.format, "conflict": result.conflict,
        "too_far": result.too_far, "duration_ms": result.duration_ms,
        "quality": result.quality.as_dict() if result.quality else None,
    }
    return result.value, diagnostics


def scan_barcode(frame, roi=None):
    """Return a verified barcode from a BGR frame, or None."""
    return scan_barcode_details(frame, roi)[0]


# ---------------------------------------------------------------------------
# Standalone debounce/confirmation primitives.
#
# `services.vision.tracking.ScanTracker` is what the live CameraService uses --
# it combines all three behaviours below with confidence scoring and a proper
# scan state machine.  These small classes are kept because they are useful on
# their own (and directly testable) for callers that only need one of the
# behaviours, such as an external scanner feed or a stricter hardware mode.
# ---------------------------------------------------------------------------

@dataclass
class BarcodeDebouncer:
    """Suppress the same value for a fixed period after it was accepted."""
    cooldown_seconds: float = 1.5
    last_value: str | None = None
    last_time: float = 0.0

    def accept(self, value: str | None, now: float | None = None) -> bool:
        if not value:
            return False
        now = time.monotonic() if now is None else now
        if value == self.last_value and now - self.last_time < self.cooldown_seconds:
            return False
        self.last_value, self.last_time = value, now
        return True


@dataclass
class BarcodeStabilizer:
    """Require repeated valid detections before a value is published."""
    required_hits: int = 2
    value: str | None = None
    hits: int = 0
    last_seen: float = 0.0
    window_seconds: float = 1.25

    def observe(self, value: str | None, now: float | None = None) -> str | None:
        now = time.monotonic() if now is None else now
        if not value:
            return None
        if value == self.value and now - self.last_seen <= self.window_seconds:
            self.hits += 1
        else:
            self.value, self.hits = value, 1
        self.last_seen = now
        return value if self.hits >= self.required_hits else None


@dataclass
class BarcodePresenceTracker:
    """Emit one event per barcode for as long as it stays continuously in view.

    Debounce is presence-based rather than time-based: repeat events are
    suppressed only while the code keeps being read, and a short run of misses
    (the product turning away, motion blur, or being lifted) clears the active
    code so the same product can be scanned again on its next appearance.
    """
    miss_tolerance: int = 3
    active_value: str | None = None
    misses: int = 0

    def observe(self, value: str | None) -> str | None:
        if not value:
            self.misses += 1
            if self.misses >= self.miss_tolerance:
                self.active_value = None
            return None
        self.misses = 0
        if value == self.active_value:
            return None
        self.active_value = value
        return value
