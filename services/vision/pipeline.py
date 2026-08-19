"""Per-frame orchestration: locate -> decode -> OCR fallback -> confidence.

The pipeline always tries to *find* barcode-shaped regions before spending
decode time.  Full-frame blind decoding is only a last resort when nothing
barcode-like is visible, which avoids reading random text or logo blobs.
"""
from dataclasses import dataclass, field
import logging
import time

import numpy as np
import cv2

from services.config import DEFAULT_CONFIG, ScannerConfig
from services.vision import preprocess
from services.vision.confidence import ConfidenceBreakdown, ConfidenceLevel, level_for, score_confidence
from services.vision.decoding import DecodeResult, decode_image, decode_with_opencv
from services.vision.detection import detect_candidates, looks_like_barcode
from services.vision.ocr_fallback import decode_digits_below_barcode
from services.vision.quality import FrameQuality, QualityHint, analyse_quality, to_gray

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
        """Locate barcodes first, then decode only where they were found."""
        started = time.perf_counter()
        result = FrameResult()
        if image is None or image.size == 0:
            result.quality = analyse_quality(image, self.config)
            return result

        config = self.config
        result.quality = analyse_quality(image, config)
        deadline = started + config.frame_budget_ms / 1000.0
        by_region: dict[str, list[DecodeResult]] = {}
        gray = to_gray(image)

        # --- pass 0: one-shot OpenCV locate+decode on the scan area ----------
        if config.use_opencv_decoder:
            self._attempt(result, "full-opencv")
            found = decode_with_opencv(image, quad=None, variant="full-opencv")
            if not found:
                # OpenCV misses some codes at 1080p; a downscaled copy often succeeds
                h, w = image.shape[:2]
                if w > 960:
                    small = cv2.resize(image, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA)
                    self._attempt(result, "full-opencv-small")
                    found = decode_with_opencv(small, quad=None, variant="full-opencv-small")
            if found:
                by_region.setdefault("full", []).extend(found)
                if _has_valid(by_region):
                    self._finalise(result, by_region, image, hits_for)
                    result.duration_ms = (time.perf_counter() - started) * 1000
                    self._observe_latency(result)
                    return result

        # --- pass 1: find barcode-shaped regions -----------------------------
        result.candidates = detect_candidates(image, config)
        if result.candidates:
            self._count("barcode_candidates", len(result.candidates))
            logger.debug("BARCODE_CANDIDATE_FOUND count=%d", len(result.candidates))

        glare = _has_glare(result.quality, config)
        targets = [c for c in result.candidates if looks_like_barcode(c, config)] or result.candidates[:1]

        # --- pass 2: decode each located region, best first ------------------
        for index, candidate in enumerate(targets):
            key = f"region-{index}"
            self._decode_candidate(image, candidate, key, by_region, result, deadline, glare=glare)
            if _has_valid(by_region):
                break

        # --- pass 3: OCR -- on glossy labels digits often decode first -------
        if not _has_valid(by_region) and targets:
            ocr_limit = 2 if glare else 1
            for index, candidate in enumerate(targets[:ocr_limit]):
                if time.perf_counter() > deadline:
                    break
                key = f"region-{index}"
                self._attempt(result, f"{key}:ocr")
                found = decode_digits_below_barcode(image, candidate, config,
                                                    variant=f"{key}:ocr")
                if found:
                    by_region.setdefault(key, []).extend(found)
                    if _has_valid(by_region):
                        break

        # --- pass 4: escalate the best located region ------------------------
        if not _has_valid(by_region) and targets:
            self._decode_candidate_escalation(image, targets[0],
                                                "region-0", by_region, result, deadline)

        # --- pass 5: last resort -- full scan area when nothing decoded yet ----
        if not _has_valid(by_region):
            self._attempt(result, "roi-gray")
            hits = decode_image(gray, config, variant="roi-gray")
            if hits:
                by_region.setdefault("roi", []).extend(hits)
            elif config.use_opencv_decoder:
                self._attempt(result, "roi-opencv")
                found = decode_with_opencv(image, quad=None, variant="roi-opencv")
                if found:
                    by_region.setdefault("roi", []).extend(found)

        self._finalise(result, by_region, image, hits_for)
        result.duration_ms = (time.perf_counter() - started) * 1000
        self._observe_latency(result)
        return result

    def _decode_candidate(self, image, candidate: Candidate, key: str,
                          by_region: dict, result: FrameResult, deadline: float,
                          *, glare: bool = False) -> None:
        """Fast decode attempts on one located barcode region."""
        config = self.config

        if config.use_opencv_decoder:
            self._attempt(result, f"{key}:opencv-quad")
            found = decode_with_opencv(image, candidate.quad, variant=f"{key}:opencv-quad")
            if found:
                by_region.setdefault(key, []).extend(found)
                if _valid_values(by_region.get(key, [])):
                    return

        cropped = preprocess.crop(image, candidate.box, config)
        if cropped is not None and config.use_opencv_decoder:
            self._attempt(result, f"{key}:opencv-crop")
            found = decode_with_opencv(cropped.image, quad=None, variant=f"{key}:opencv-crop")
            if found:
                by_region.setdefault(key, []).extend(found)
                if _valid_values(by_region.get(key, [])):
                    return

        rectified = preprocess.rectify(image, candidate.quad, config)
        regions = []
        if rectified is not None:
            logger.debug("ROI_CREATED %s %s", key, rectified.as_dict())
            regions.append(rectified)
        if cropped is not None:
            regions.append(cropped)

        levels = (0, 1) if glare else (0,)
        for region in regions:
            for level in levels:
                for name, prepared in preprocess.variants(region.image, level, config, glare=glare):
                    label = f"{key}:{region.method}-{name}"
                    self._attempt(result, label)
                    found = decode_image(prepared, config, variant=label)
                    if found:
                        for item in found:
                            by_region.setdefault(key, []).append(item)
                        if _valid_values(by_region.get(key, [])):
                            logger.debug("DECODER_SUCCESS %s variant=%s", key, name)
                            return
                    if glare and config.use_opencv_decoder:
                        self._attempt(result, f"{label}-opencv")
                        found = decode_with_opencv(prepared, quad=None, variant=f"{label}-opencv")
                        if found:
                            by_region.setdefault(key, []).extend(found)
                            if _valid_values(by_region.get(key, [])):
                                return
                if time.perf_counter() > deadline:
                    return
            if time.perf_counter() > deadline:
                return

    def _decode_candidate_escalation(self, image, candidate: Candidate, key: str,
                                     by_region: dict, result: FrameResult,
                                     deadline: float) -> None:
        """Heavy enhancement on the best located region only."""
        config = self.config
        regions = []
        rectified = preprocess.rectify(image, candidate.quad, config)
        if rectified is not None:
            regions.append(rectified)
        cropped = preprocess.crop(image, candidate.box, config)
        if cropped is not None:
            regions.append(cropped)
        if not regions:
            return

        for level in range(1, preprocess.MAX_LEVEL + 1):
            if level == preprocess.MAX_LEVEL and not self._heavy_allowed(
                    result, deadline, candidate=candidate):
                return
            if time.perf_counter() > deadline:
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
                            return
                    if config.use_opencv_decoder:
                        self._attempt(result, f"{label}-opencv")
                        found = decode_with_opencv(prepared, quad=None, variant=f"{label}-opencv")
                        if found:
                            by_region.setdefault(key, []).extend(found)
                            if _valid_values(by_region.get(key, [])):
                                return

    def _heavy_allowed(self, result: FrameResult, deadline: float,
                       candidate=None) -> bool:
        glare = _has_glare(result.quality, self.config)
        if result.quality is not None and not result.quality.worth_heavy_passes:
            small = candidate is not None and candidate.long_edge < 140
            if not small and not glare:
                return False
        return time.perf_counter() <= deadline

    def _finalise(self, result: FrameResult, by_region: dict, image, hits_for) -> None:
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
        if best.decoder.startswith("ocr") and result.confidence < self.config.confidence_threshold:
            result.confidence = max(result.confidence, self.config.confidence_threshold)
            result.level = level_for(result.confidence, self.config)

    def _is_too_far(self, result: FrameResult, image) -> bool:
        if not result.candidates:
            return False
        width = max(1, image.shape[1])
        biggest = max(candidate.long_edge for candidate in result.candidates)
        return biggest / width < self.config.min_barcode_width_ratio

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
    if key == "full":
        return 110
    if key == "roi":
        return 20
    try:
        return 80 - int(str(key).split("-")[-1])
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


def _has_glare(quality: FrameQuality | None, config: ScannerConfig) -> bool:
    if quality is None:
        return False
    if quality.hint is QualityHint.GLARE:
        return True
    return quality.glare_ratio >= config.glare_ratio_max * 0.65
