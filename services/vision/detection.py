"""Locate probable 1D-barcode regions before spending decode time on them.

Two independent detectors run, because measurement showed they fail on
different things:

* OpenCV's `cv2.barcode.BarcodeDetector` returns rotated quadrangles and copes
  well with codes held at 30-60 degrees, where a plain full-frame decode gives
  up entirely -- but it misses strongly perspective-skewed labels.
* The gradient/morphology detector (vertical-edge density) finds those skewed
  and low-contrast labels, but only as coarse blobs.

Union of the two, deduplicated, is meaningfully better than either alone and
costs a few milliseconds on a downscaled working copy.
"""
from dataclasses import dataclass
import threading

import cv2
import numpy as np

from services.config import DEFAULT_CONFIG, ScannerConfig
from services.vision.preprocess import order_quad
from services.vision.quality import to_gray


_local = threading.local()


def _opencv_detector():
    """One detector per thread; the OpenCV object is not documented as thread-safe."""
    detector = getattr(_local, "barcode_detector", None)
    if detector is None:
        detector = cv2.barcode.BarcodeDetector()
        _local.barcode_detector = detector
    return detector


@dataclass
class Candidate:
    """A possible barcode location in *source image* pixel coordinates."""
    quad: np.ndarray           # (4, 2) float32, corners in detector order
    box: tuple[int, int, int, int]   # axis-aligned bounds (x1, y1, x2, y2)
    long_edge: float
    short_edge: float
    source: str                # which detector produced it
    weight: float              # per-detector trust, used to order candidates

    @property
    def score(self) -> float:
        """Bigger, more trusted candidates are decoded first."""
        return self.weight * self.long_edge

    @property
    def aspect(self) -> float:
        return self.long_edge / max(1.0, self.short_edge)

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    def as_dict(self) -> dict:
        return {"box": list(self.box), "long_edge": round(self.long_edge),
                "short_edge": round(self.short_edge), "source": self.source,
                "score": round(self.score, 2)}


def detect_candidates(image: np.ndarray, config: ScannerConfig = DEFAULT_CONFIG) -> list[Candidate]:
    """Return likely barcode regions, best first, in `image` coordinates."""
    if image is None or image.size == 0:
        return []
    height, width = image.shape[:2]
    scale = min(1.0, config.detection_width / max(1, width))
    work = image
    if scale < 1.0:
        work = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                          interpolation=cv2.INTER_AREA)
    gray = to_gray(work)

    found: list[Candidate] = []
    if config.use_opencv_detector:
        found.extend(_detect_with_opencv(work, gray))
    if config.use_gradient_detector:
        found.extend(_detect_with_gradients(gray))

    rescaled = [_rescale(candidate, 1.0 / scale) for candidate in found] if scale < 1.0 else found
    return _deduplicate([c for c in rescaled if _is_plausible(c, config)], config.max_candidates)


def _detect_with_opencv(work: np.ndarray, gray: np.ndarray) -> list[Candidate]:
    """Rotation-aware quadrangles from OpenCV's dedicated barcode detector."""
    try:
        ok, points = _opencv_detector().detectMulti(work)
    except cv2.error:
        return []
    if not ok or points is None:
        return []
    candidates = []
    for quad in np.asarray(points, dtype=np.float32).reshape(-1, 4, 2):
        candidates.append(_from_quad(quad, source="opencv", weight=2.0))
    return candidates


def _detect_with_gradients(gray: np.ndarray) -> list[Candidate]:
    """Vertical-edge density: the classic barcode signature (bars, no verticals).

    This is the detector the previous implementation used; it is kept because it
    still catches skewed and low-contrast labels the OpenCV detector drops.  The
    difference is that regions are now fitted with `minAreaRect`, so a tilted
    label produces a tilted quad instead of a loose axis-aligned box.
    """
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.subtract(grad_x, grad_y))
    edges = cv2.GaussianBlur(edges, (7, 7), 0)
    _, mask = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        candidates.append(_from_quad(quad, source="gradient", weight=1.0))
    return candidates


def _from_quad(quad: np.ndarray, source: str, weight: float) -> Candidate:
    # Corners are canonicalised here so downstream code never has to care which
    # detector produced them (OpenCV and minAreaRect use different conventions).
    quad = order_quad(quad)
    edges = [float(np.linalg.norm(quad[i] - quad[(i + 1) % 4])) for i in range(4)]
    long_edge = max((edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2)
    short_edge = min((edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2)
    x1, y1 = quad.min(axis=0)
    x2, y2 = quad.max(axis=0)
    box = (int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2)))
    return Candidate(quad, box, long_edge, short_edge, source, weight)


def _rescale(candidate: Candidate, factor: float) -> Candidate:
    return _from_quad(candidate.quad * factor, candidate.source, candidate.weight)


def _is_plausible(candidate: Candidate, config: ScannerConfig) -> bool:
    """Drop logo/text blobs and anything too small or too large to be a barcode."""
    if candidate.long_edge < config.min_barcode_size or candidate.long_edge > config.max_barcode_size:
        return False
    if candidate.short_edge < 8:
        return False
    return candidate.aspect >= config.candidate_min_aspect


def _deduplicate(candidates: list[Candidate], maximum: int) -> list[Candidate]:
    """Keep the strongest candidates, merging boxes the two detectors share."""
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if any(_overlap(candidate.box, existing.box) > 0.55 for existing in kept):
            continue
        kept.append(candidate)
        if len(kept) >= maximum:
            break
    return kept


def _overlap(a: tuple, b: tuple) -> float:
    """Intersection over the smaller box; robust when one detector is coarser."""
    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = inter_w * inter_h
    if not intersection:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / max(1, smaller)
