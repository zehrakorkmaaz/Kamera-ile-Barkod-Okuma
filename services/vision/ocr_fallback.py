"""Read human-readable digits below a barcode when bar decoding fails.

Retail labels print the EAN/UPC number under the bars. When glare, curvature
or distance break the stripe decoder, those digits are often still readable.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from services.config import DEFAULT_CONFIG, ScannerConfig
from services.vision.decoding import DecodeResult, is_valid_barcode, normalize_barcode
from services.vision.detection import Candidate
from services.vision.quality import to_gray

try:
    import pytesseract
except ImportError:
    pytesseract = None

_DIGIT_RE = re.compile(r"\d{8,13}")
_TEMPLATES: dict[str, np.ndarray] | None = None


def decode_digits_below_barcode(image: np.ndarray, candidate: Candidate,
                                config: ScannerConfig = DEFAULT_CONFIG,
                                variant: str = "ocr") -> list[DecodeResult]:
    """Try to read printed digits under a detected barcode region."""
    if image is None or image.size == 0 or not config.use_ocr_fallback:
        return []
    region = _digits_region(image, candidate)
    if region is None or region.size == 0:
        return []

    results: list[DecodeResult] = []
    for name, prepared in _ocr_variants(region, config):
        label = f"{variant}-{name}" if variant else name
        results.extend(_read_digits(prepared, label))
        if results:
            return results
    return results


def _digits_region(image: np.ndarray, candidate: Candidate) -> np.ndarray | None:
    """Crop the label strip: barcode band plus the number line beneath it."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = candidate.box
    band_h = max(12, y2 - y1)
    band_w = max(20, x2 - x1)
    pad_x = int(band_w * 0.08) + 4
    pad_top = int(band_h * 0.15)
    pad_bottom = int(band_h * 1.6)
    x1 = max(0, x1 - pad_x)
    x2 = min(width, x2 + pad_x)
    y1 = max(0, y1 - pad_top)
    y2 = min(height, y2 + pad_bottom)
    if x2 - x1 < 24 or y2 - y1 < 20:
        return None
    return image[y1:y2, x1:x2].copy()


def _ocr_variants(region: np.ndarray, config: ScannerConfig):
    """Yield preprocessed views, cheapest first."""
    gray = to_gray(region)
    scale = min(config.upscale_factor, max(2.0, 480 / max(1, gray.shape[1])))
    if scale > 1.05:
        gray = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)),
                                 max(1, int(gray.shape[0] * scale))),
                          interpolation=cv2.INTER_CUBIC)
    yield "gray", gray
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    yield "clahe", clahe
    yield "otsu", cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    yield "adaptive", cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 31, 5)


def _read_digits(image: np.ndarray, variant: str) -> list[DecodeResult]:
    for value in _tesseract_values(image):
        result = _as_result(value, "EAN-13" if len(value) == 13 else "EAN-8", variant, "tesseract")
        if result:
            return [result]
    for value in _template_values(image):
        result = _as_result(value, "EAN-13" if len(value) == 13 else "EAN-8", variant, "template")
        if result:
            return [result]
    return []


def _tesseract_values(image: np.ndarray) -> list[str]:
    if pytesseract is None:
        return []
    try:
        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(image, config=config)
    except Exception:
        return []
    return _valid_numeric_candidates(text)


def _template_values(image: np.ndarray) -> list[str]:
    gray = image if len(image.shape) == 2 else to_gray(image)
    values = []
    for inverted in (False, True):
        binary = _binarize_for_digits(gray, inverted)
        digits = _segment_and_recognize(binary)
        if digits:
            values.append(digits)
    return _valid_numeric_candidates(*values)


def _valid_numeric_candidates(*texts: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        for match in _DIGIT_RE.findall(str(text)):
            value = normalize_barcode(match)
            if value in seen:
                continue
            if len(value) in (8, 12, 13) and is_valid_barcode(value):
                seen.add(value)
                out.append(value)
    return out


def _binarize_for_digits(gray: np.ndarray, invert: bool) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        binary = cv2.bitwise_not(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return binary


def _segment_and_recognize(binary: np.ndarray) -> str:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    height = binary.shape[0]
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < height * 0.25 or h > height * 0.95:
            continue
        if w < 3 or w > height:
            continue
        if w / max(1, h) > 1.2:
            continue
        boxes.append((x, y, w, h))
    if len(boxes) < 8:
        return ""
    boxes.sort(key=lambda item: item[0])
    chars = []
    for x, y, w, h in boxes:
        patch = binary[y:y + h, x:x + w]
        digit = _match_digit(patch)
        if digit is None:
            return ""
        chars.append(digit)
    return "".join(chars)


def _match_digit(patch: np.ndarray) -> str | None:
    if patch.size == 0:
        return None
    normalized = cv2.resize(patch, (20, 32), interpolation=cv2.INTER_AREA)
    best_digit, best_score = None, -1.0
    for digit, template in _digit_templates().items():
        score = float(cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)[0][0])
        if score > best_score:
            best_digit, best_score = digit, score
    return best_digit if best_score >= 0.35 else None


def _digit_templates() -> dict[str, np.ndarray]:
    global _TEMPLATES
    if _TEMPLATES is not None:
        return _TEMPLATES
    templates: dict[str, np.ndarray] = {}
    for digit in range(10):
        canvas = np.zeros((40, 24), dtype=np.uint8)
        cv2.putText(canvas, str(digit), (1, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.05, 255, 2, cv2.LINE_AA)
        _, binary = cv2.threshold(canvas, 160, 255, cv2.THRESH_BINARY)
        templates[str(digit)] = cv2.resize(binary, (20, 32), interpolation=cv2.INTER_AREA)
    _TEMPLATES = templates
    return templates


def _as_result(value: str, format_name: str, variant: str, engine: str) -> DecodeResult | None:
    value = normalize_barcode(value)
    if not value or not is_valid_barcode(value, format_name):
        return None
    return DecodeResult(value, format_name, f"ocr-{engine}", True, variant)
