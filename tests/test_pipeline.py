"""Tests for detection-first pipeline and OCR fallback."""
import cv2
import numpy as np
import pytest
import zxingcpp

from services.config import ScannerConfig
from services.vision.detection import Candidate, _is_plausible, _stripe_score, detect_candidates
from services.vision.pipeline import BarcodePipeline
from services.vision.decoding import DecodeResult
from services.vision.quality import to_gray


def _ean_image(value="8690530046269", scale=2, canvas_size=(576, 1101)):
    encoded = np.asarray(zxingcpp.write_barcode_to_image(
        zxingcpp.create_barcode(value, zxingcpp.BarcodeFormat.EAN13), scale=scale))
    canvas = np.full(canvas_size, 255, dtype=np.uint8)
    h, w = encoded.shape
    y = (canvas_size[0] - h) // 2
    x = (canvas_size[1] - w) // 2
    canvas[y:y + h, x:x + w] = encoded
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_pipeline_locates_before_blind_roi_decode():
    """Located regions are tried before the full-area roi-gray fallback."""
    image = _ean_image(scale=2)
    pipeline = BarcodePipeline(ScannerConfig(use_ocr_fallback=False))
    result = pipeline.process(image)
    assert result.value == "8690530046269"
    assert result.attempts[0] in {"full-opencv", "region-0:opencv-quad", "region-0:opencv-crop"}
    if "roi-gray" in result.attempts:
        assert result.attempts.index("roi-gray") > 0


def test_pipeline_reads_small_barcode_at_simulated_30cm():
    """A ~120 px wide EAN-13 should decode at typical 30 cm framing."""
    image = _ean_image(scale=1, canvas_size=(576, 1101))
    pipeline = BarcodePipeline(ScannerConfig(use_ocr_fallback=False))
    result = pipeline.process(image)
    assert result.value == "8690530046269"


def test_stripe_score_prefers_real_barcodes_over_blank_areas():
    barcode = _ean_image(scale=2)
    gray = to_gray(barcode)
    h, w = gray.shape
    real = _stripe_score(gray, (w // 4, h // 3, 3 * w // 4, 2 * h // 3))
    blank = _stripe_score(gray, (20, 20, 180, 120))
    assert real > blank
    assert real >= 0.35


def test_gradient_candidate_without_stripes_is_rejected():
    quad = np.array([[10, 10], [220, 12], [218, 90], [8, 88]], dtype=np.float32)
    candidate = Candidate(quad, (8, 10, 220, 90), 210, 78, "gradient", 1.0, stripe_score=0.05)
    config = ScannerConfig(min_stripe_score=0.28)
    assert not _is_plausible(candidate, config)


def test_ocr_fallback_reads_printed_digits(monkeypatch):
    """When bars fail, digits below the detected region can still confirm the code."""
    image = _ean_image(scale=2)
    quad = np.array([[300, 180], [800, 182], [798, 260], [298, 258]], dtype=np.float32)
    candidate = Candidate(quad, (298, 180, 800, 320), 500, 80, "opencv", 2.0, stripe_score=0.8)
    fake = DecodeResult("8690530046269", "EAN-13", "ocr-template", True, "region-0:ocr")

    monkeypatch.setattr("services.vision.pipeline.detect_candidates", lambda *_a, **_k: [candidate])
    monkeypatch.setattr("services.vision.pipeline.decode_digits_below_barcode",
                        lambda *_a, **_k: [fake])
    monkeypatch.setattr(BarcodePipeline, "_decode_candidate", lambda *a, **k: None)
    monkeypatch.setattr(BarcodePipeline, "_decode_candidate_escalation", lambda *a, **k: None)
    monkeypatch.setattr("services.vision.pipeline.decode_with_opencv", lambda *_a, **_k: [])

    pipeline = BarcodePipeline(ScannerConfig(use_ocr_fallback=True))
    result = pipeline.process(image)
    assert result.value == "8690530046269"
    assert any("ocr" in attempt for attempt in result.attempts)


def test_reduce_glare_keeps_barcode_readable():
    from services.vision.preprocess import reduce_glare
    encoded = np.asarray(zxingcpp.write_barcode_to_image(
        zxingcpp.create_barcode("8690530046269", zxingcpp.BarcodeFormat.EAN13), scale=2))
    gray = encoded.copy()
    gray[20:35, 40:180] = 255  # simulate specular strip across bars
    recovered = reduce_glare(gray)
    assert float(recovered.mean()) <= float(gray.mean())
    assert recovered.shape == gray.shape


def test_detect_candidates_finds_ean_and_ignores_packaging_blob():
    image = _ean_image(scale=2)
    candidates = detect_candidates(image, ScannerConfig())
    assert candidates
    assert candidates[0].long_edge < 700
    assert candidates[0].short_edge < 300


def test_huge_package_blob_is_not_plausible():
    quad = np.array([[10, 10], [820, 12], [818, 540], [8, 538]], dtype=np.float32)
    candidate = Candidate(quad, (8, 10, 820, 540), 806, 529, "gradient", 1.0, stripe_score=0.63)
    assert not _is_plausible(candidate, ScannerConfig())


def test_thin_edge_line_is_not_plausible():
    quad = np.array([[0, 0], [428, 0], [428, 10], [0, 10]], dtype=np.float32)
    candidate = Candidate(quad, (0, 0, 428, 10), 428, 10, "gradient", 1.0, stripe_score=0.62)
    assert not _is_plausible(candidate, ScannerConfig())
