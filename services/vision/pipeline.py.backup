"""Per-frame orchestration: quality -> detection -> ROI -> decode -> confidence.

The escalation order is the whole point.  A clean, well-presented barcode is
decoded by the first, cheapest pass in a few milliseconds; the expensive
rectify/upscale/threshold work only runs for the frames that actually need it,
and stops when the frame's time budget is spent.  That keeps a well-lit
close-up at full frame rate while still giving a small, tilted, badly-lit code
every chance we can afford.
"""
from dataclasses import dataclass, field
import logging
import time

import numpy as np

from services.config import DEFAULT_CONFIG, ScannerConfig
from services.vision import preprocess
from services.vision.confidence import ConfidenceBreakdown, ConfidenceLevel, score_confidence
from services.vision.decoding import DecodeResult, decode_image, decode_with_opencv
from services.vision.detection import Candidate, detect_candidates
from services.vision.quality import FrameQuality, analyse_quality, to_gray

logger = logging.getLogger("smartcart.scan")


@dataclass
class FrameResult:
    """Everything one frame produced, for the tracker and for diagnostics."""
    value: str | None = None
    format: str = ""
    confidence: float = 0.0
    level: ConfidenceLevel = ConfidenceLevel.LOW
    quality: FrameQuality | None = None
    candidates: list[Candidate] = field(default_factory=list)
    decoders: tuple[str, ...] = ()
    conflict: bool = False
    conflicting_values: tuple[str, ...] = ()
    attempts: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    too_far: bool = False
    quad: np.ndarray | None = field(default=None, repr=False)
    breakdown: ConfidenceBreakdown | None = None
    checksum_failures: int = 0
    #  Name of the pass that produced the winning read, for diagnostics.
    method: str = ""

    @property
    def decoded(self) -> bool:
        return bool(self.value)

    def as_dict(self) -> dict:
        return {"value": self.value, "format": self.format, "method": self.method,
                "confidence": round(self.confidence, 1), "level": self.level.value,
                "conflict": self.conflict, "conflicting_values": list(self.conflicting_values),
                "too_far": self.too_far, "duration_ms": round(self.duration_ms, 1),
                "candidate_count": len(self.candidates),
                "candidates": [c.as_dict() for c in self.candidates],
                "decoders": list(self.decoders), "attempts": self.attempts,
                "quality": self.quality.as_dict() if self.quality else None,
                "confidence_breakdown": self.breakdown.as_dict() if self.breakdown else None}


class BarcodePipeline:
    """Stateless per-frame processing; safe to own one per scanning thread."""

    def __init__(self, config: ScannerConfig = DEFAULT_CONFIG, metrics=None):
        self.config = config
        self.metrics = metrics

    def process(self, image: np.ndarray, hits_for=None) -> FrameResult:
        """Decode one scan area.  `hits_for(value)` supplies temporal agreement."""
        started = time.perf_counter()
        result = FrameResult()
        if image is None or image.size == 0:
            result.quality = analyse_quality(image, self.config)
            return result

        config = self.config
        result.quality = analyse_quality(image, config)
        deadline = started + config.frame_budget_ms / 1000.0
        # region key -> decode results, so decoder disagreement is judged per
        # physical code rather than across two different products in view.
        by_region: dict[str, list[DecodeResult]] = {}
        gray = to_gray(image)

        # --- pass 1: the whole scan area, no detection, no enhancement -------
        self._attempt(result, "roi-gray")
        hits = decode_image(gray, config, variant="roi-gray")
        if hits:
            by_region.setdefault("roi", []).extend(hits)
            # Fast path: a valid ZXing/EAN read is already strong evidence.
            # Do not pay for a second decoder on every successful frame; the
            # OpenCV decoder remains available on candidate/rectified fallback
            # paths for difficult or rotated barcodes.

        # --- pass 2: find candidate regions and work on them -----------------
        if not _has_valid(by_region):
            result.candidates = detect_candidates(image, config)
            if result.candidates:
                self._count("barcode_candidates", len(result.candidates))
                logger.debug("BARCODE_CANDIDATE_FOUND count=%d", len(result.candidates))
            for index, candidate in enumerate(result.candidates):
                key = f"region-{index}"
                self._decode_candidate(image, candidate, key, by_region, result, deadline)
                if _has_valid(by_region):
                    break

        self._finalise(result, by_region, image, hits_for)
        result.duration_ms = (time.perf_counter() - started) * 1000
        self._observe_latency(result)
        return result

    # -- decoding helpers ---------------------------------------------------

    def _decode_candidate(self, image, candidate: Candidate, key: str,
                          by_region: dict, result: FrameResult, deadline: float) -> None:
        """Escalate through rectification and enhancement for one region."""
        config = self.config

        # OpenCV reads straight off the quad and costs well under a millisecond;
        # it is the single best-value attempt for a rotated code.
        if config.use_opencv_decoder:
            self._attempt(result, f"{key}:opencv-quad")
            found = decode_with_opencv(image, candidate.quad)
            if found:
                by_region.setdefault(key, []).extend(found)

        regions = []
        rectified = preprocess.rectify(image, candidate.quad, config)
        if rectified is not None:
            logger.debug("ROI_CREATED %s %s", key, rectified.as_dict())
            regions.append(rectified)
        # Rectification can distort a loosely-bounded region, so the plain crop
        # stays in the ladder as a second opinion.
        cropped = preprocess.crop(image, candidate.box, config)
        if cropped is not None:
            regions.append(cropped)

        for level in range(preprocess.MAX_LEVEL + 1):
            if level == preprocess.MAX_LEVEL and not self._heavy_allowed(result, deadline):
                return
            if level and time.perf_counter() > deadline:
                logger.debug("BUDGET_EXHAUSTED %s level=%d", key, level)
                return
            for region in regions:
                for name, prepared in preprocess.variants(region.image, level, config):
                    label = f"{key}:{region.method}-{name}"
                    self._attempt(result, label)
                    found = decode_image(prepared, config, variant=label)
                    if found:
                        for item in found:
                            by_region.setdefault(key, []).append(item)
                        if _valid_values(by_region.get(key, [])):
                            logger.debug("DECODER_SUCCESS %s level=%d variant=%s", key, level, name)
                            return

    def _confirm_with_opencv(self, image, hits: list[DecodeResult], by_region: dict,
                             key: str, result: FrameResult) -> None:
        """Ask the second decoder about a region zxing-cpp already read.

        Agreement is the strongest signal we have, and it is nearly free here
        because the barcode's corners are already known.
        """
        if not self.config.use_opencv_decoder:
            return
        quad = next((hit.quad for hit in hits if hit.quad is not None), None)
        if quad is None:
            return
        self._attempt(result, f"{key}:opencv-confirm")
        found = decode_with_opencv(image, quad, variant="opencv-confirm")
        if found:
            by_region.setdefault(key, []).extend(found)

    def _heavy_allowed(self, result: FrameResult, deadline: float) -> bool:
        """Heavy passes need both a sharp enough frame and time left."""
        if result.quality is not None and not result.quality.worth_heavy_passes:
            return False
        return time.perf_counter() <= deadline

    # -- result assembly -----------------------------------------------------

    def _finalise(self, result: FrameResult, by_region: dict, image, hits_for) -> None:
        """Pick a winner, detect decoder conflicts and score confidence."""
        result.checksum_failures = sum(1 for items in by_region.values()
                                       for item in items if not item.valid)
        if result.checksum_failures:
            self._count("checksum_failure", result.checksum_failures)

        winner_key, winner = None, None
        for key, items in by_region.items():
            values = _valid_values(items)
            if not values:
                continue
            if len(values) > 1:
                # Same physical region, two different valid readings: never guess.
                result.conflict = True
                result.conflicting_values = tuple(sorted(values))
                self._count("decoder_conflicts")
                logger.warning("RESULT_CONFLICT region=%s values=%s", key, result.conflicting_values)
                return
            candidate_value = next(iter(values))
            if winner is None or _region_rank(key) > _region_rank(winner_key):
                winner_key, winner = key, [item for item in items if item.value == candidate_value]

        self._count("decode_attempts")
        if not winner:
            self._count("decode_failure")
            result.too_far = self._is_too_far(result, image)
            return

        self._count("decode_success")
        best = winner[0]
        decoders = tuple(sorted({item.decoder for item in winner}))
        result.value, result.format, result.decoders = best.value, best.format, decoders
        result.method = best.variant or winner_key or ""
        result.quad = _pick_quad(winner, result, winner_key)
        logger.debug("CHECKSUM_VALID value=%s format=%s decoders=%s",
                     best.value, best.format, decoders)

        width = max(1, image.shape[1])
        size_ratio = _quad_long_edge(result.quad) / width if result.quad is not None else 0.4
        perspective = (preprocess.perspective_score(result.quad)
                       if result.quad is not None else 1.0)
        agreements = hits_for(best.value) if hits_for else 1
        result.breakdown = score_confidence(
            decoded=True, checksum_valid=best.valid, decoder_agreement=len(decoders) > 1,
            size_ratio=size_ratio, sharpness=result.quality.sharpness if result.quality else 0.0,
            perspective=perspective, agreements=agreements, config=self.config)
        result.confidence, result.level = result.breakdown.score, result.breakdown.level

    def _is_too_far(self, result: FrameResult, image) -> bool:
        """A candidate was found but is too few pixels wide to carry the modules."""
        if not result.candidates:
            return False
        width = max(1, image.shape[1])
        biggest = max(candidate.long_edge for candidate in result.candidates)
        return biggest / width < self.config.min_barcode_width_ratio

    # -- bookkeeping ---------------------------------------------------------

    def _attempt(self, result: FrameResult, name: str) -> None:
        if len(result.attempts) < self.config.debug_attempt_history:
            result.attempts.append(name)

    def _count(self, name: str, amount: int = 1) -> None:
        if self.metrics is not None:
            self.metrics.increment(name, amount)

    def _observe_latency(self, result: FrameResult) -> None:
        if self.metrics is not None:
            self.metrics.observe_decode_latency(result.duration_ms)


def _valid_values(items: list[DecodeResult]) -> set[str]:
    return {item.value for item in items if item.valid}


def _has_valid(by_region: dict) -> bool:
    return any(_valid_values(items) for items in by_region.values())


def _region_rank(key: str | None) -> int:
    """Prefer the whole-area read, then earlier (larger) candidate regions."""
    if key == "roi":
        return 100
    try:
        return 50 - int(str(key).split("-")[-1])
    except ValueError:
        return 0


def _pick_quad(items: list[DecodeResult], result: FrameResult, key: str | None):
    for item in items:
        if item.quad is not None:
            return item.quad
    index = _candidate_index(key)
    if index is not None and index < len(result.candidates):
        return result.candidates[index].quad
    return None


def _candidate_index(key: str | None) -> int | None:
    if not key or not str(key).startswith("region-"):
        return None
    try:
        return int(str(key).split("-")[1])
    except ValueError:
        return None


def _quad_long_edge(quad) -> float:
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    edges = [float(np.linalg.norm(quad[i] - quad[(i + 1) % 4])) for i in range(4)]
    return max(edges)
