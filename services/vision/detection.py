"""Locate probable 1D-barcode regions before spending decode time on them.

Two independent detectors run, because they fail on different things:

* OpenCV's `cv2.barcode.BarcodeDetector` returns rotated quadrangles.
* The gradient detector finds low-contrast / skewed labels as coarse bands.

Candidates are scored by how *barcode-like* they are (stripe density, aspect,
typical size, centre of the scan window) — not by how large the blob is.
Packaging edges, glasses and whole-product silhouettes used to outrank the
real code because size was the primary score.
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
    quad: np.ndarray
    box: tuple[int, int, int, int]
    long_edge: float
    short_edge: float
    source: str
    weight: float
    stripe_score: float = 0.0
    centre_score: float = 1.0

    @property
    def score(self) -> float:
        """Prefer barcode-shaped, stripe-like, mid-sized, centred regions."""
        aspect_fit = _aspect_fit(self.aspect)
        size_fit = _size_fit(self.long_edge)
        stripe = 0.2 + 0.8 * max(0.0, min(1.0, self.stripe_score))
        centre = 0.45 + 0.55 * max(0.0, min(1.0, self.centre_score))
        return self.weight * stripe * aspect_fit * size_fit * centre

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
                "stripe_score": round(self.stripe_score, 2),
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
    full_gray = to_gray(image)
    scored = [_enrich(candidate, full_gray, width, height) for candidate in rescaled]
    plausible = [c for c in scored if _is_plausible(c, config)]
    return _deduplicate(plausible, config.max_candidates)


def looks_like_barcode(candidate: Candidate, config: ScannerConfig = DEFAULT_CONFIG) -> bool:
    """True when the overlay / decode queue should treat this as a real code."""
    if not _is_plausible(candidate, config):
        return False
    if candidate.source == "opencv":
        return candidate.stripe_score >= 0.22 or candidate.aspect >= 1.8
    return candidate.stripe_score >= 0.45 and 1.8 <= candidate.aspect <= 8.0


def _detect_with_opencv(work: np.ndarray, gray: np.ndarray) -> list[Candidate]:
    """Rotation-aware quadrangles; also try a contrast-boosted copy."""
    found = _opencv_points(work, weight=2.4)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    boosted = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    found.extend(_opencv_points(boosted, weight=2.0))
    return found


def _opencv_points(image: np.ndarray, weight: float) -> list[Candidate]:
    try:
        ok, points = _opencv_detector().detectMulti(image)
    except cv2.error:
        return []
    if not ok or points is None:
        return []
    return [_from_quad(quad, source="opencv", weight=weight)
            for quad in np.asarray(points, dtype=np.float32).reshape(-1, 4, 2)]


def _detect_with_gradients(gray: np.ndarray) -> list[Candidate]:
    """Find horizontal bands of dense vertical bars — not whole packages."""
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # Keep pixels that look like bars (strong X edge, weak Y edge).
    bars = cv2.subtract(cv2.convertScaleAbs(grad_x), cv2.convertScaleAbs(grad_y))
    bars = cv2.GaussianBlur(bars, (5, 5), 0)
    _, mask = cv2.threshold(bars, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Join neighbouring bars into a wide, short band.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (41, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape[:2]
    image_area = float(max(1, width * height))
    candidates = []
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in ranked[:16]:
        area = cv2.contourArea(contour)
        if area < 80 or area / image_area > 0.28:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _angle = rect
        long_e, short_e = max(rw, rh), min(rw, rh)
        if short_e < 8 or long_e < 24:
            continue
        if long_e / max(1.0, short_e) < 1.4:
            continue
        quad = cv2.boxPoints(rect).astype(np.float32)
        candidates.append(_from_quad(quad, source="gradient", weight=1.0))
    return candidates


def _from_quad(quad: np.ndarray, source: str, weight: float) -> Candidate:
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


def _enrich(candidate: Candidate, gray: np.ndarray, width: int, height: int) -> Candidate:
    stripe = _stripe_score(gray, candidate.box)
    cx = (candidate.box[0] + candidate.box[2]) / 2.0 / max(1, width)
    cy = (candidate.box[1] + candidate.box[3]) / 2.0 / max(1, height)
    dist = float(np.hypot(cx - 0.5, cy - 0.5))
    centre = max(0.0, 1.0 - dist * 1.6)
    return Candidate(candidate.quad, candidate.box, candidate.long_edge, candidate.short_edge,
                     candidate.source, candidate.weight, stripe_score=stripe, centre_score=centre)


def _stripe_score(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Alternating light/dark transitions that are consistent across scanlines."""
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 12 or y2 - y1 < 6:
        return 0.0
    roi = gray[y1:y2, x1:x2]
    band = roi[roi.shape[0] // 5:4 * roi.shape[0] // 5, :]
    if band.size == 0:
        return 0.0
    transitions = []
    step = max(1, band.shape[0] // 7)
    for row in band[::step]:
        binary = row >= np.median(row)
        transitions.append(float(np.sum(np.abs(np.diff(binary.astype(np.int8))))))
    if not transitions:
        return 0.0
    mean_t = float(np.mean(transitions))
    # A real EAN/UPC has dozens of bar/space edges; glasses/text have far fewer.
    if mean_t < 14:
        return max(0.0, mean_t / 14.0 * 0.25)
    consistency = 1.0 - min(1.0, float(np.std(transitions)) / (mean_t + 1e-6))
    density = mean_t / max(1, band.shape[1])
    return max(0.0, min(1.0, (density / 0.12) * (0.45 + 0.55 * consistency)))


def _aspect_fit(aspect: float) -> float:
    """1D retail codes are wide and short; ~3:1 is typical, >10:1 is a line."""
    if 2.0 <= aspect <= 7.0:
        return 1.0
    if 1.5 <= aspect < 2.0 or 7.0 < aspect <= 10.0:
        return 0.45
    return 0.12


def _size_fit(long_edge: float) -> float:
    """Typical 20–35 cm EAN in a 1080p scan window is tens-to-few-hundred px."""
    if 55 <= long_edge <= 480:
        return 1.0
    if 28 <= long_edge < 55 or 480 < long_edge <= 720:
        return 0.5
    return 0.18


def _is_plausible(candidate: Candidate, config: ScannerConfig) -> bool:
    """Drop logo/text blobs and anything too small or too large to be a barcode."""
    if candidate.long_edge < config.min_barcode_size or candidate.long_edge > config.max_barcode_size:
        return False
    if candidate.short_edge < config.min_barcode_short_edge:
        return False
    if candidate.short_edge > config.max_barcode_short_edge:
        return False
    if candidate.aspect < config.candidate_min_aspect:
        return False
    if candidate.aspect > config.candidate_max_aspect:
        return False
    if candidate.source == "gradient" and candidate.stripe_score < config.min_stripe_score:
        return False
    return True


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
