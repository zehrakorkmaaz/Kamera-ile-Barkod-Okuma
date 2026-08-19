"""Decoding, normalisation and checksum validation.

Two independent decoders are used, both already present in the project's
dependencies:

* `zxing-cpp` -- the existing decoder, strongest on head-on and small codes.
* OpenCV's `cv2.barcode` decoder -- reads a code straight off a detected
  quadrangle and, in measurements on this project, succeeded on 30-60 degree
  rotations where zxing-cpp returned nothing.

Agreement between them is the strongest confidence signal available; a
disagreement is reported as a conflict rather than silently picking a winner.
"""
from dataclasses import dataclass, field
import re
import threading

import cv2
import numpy as np

from services.config import DEFAULT_CONFIG, ScannerConfig
from services.vision.preprocess import expand_quad

try:
    import zxingcpp
except ImportError:  # Gives a clear runtime error if dependencies were not installed.
    zxingcpp = None


SUPPORTED_FORMATS = {"EAN-13", "EAN-8", "UPC-A", "UPC-E", "Code 128", "QR Code"}
#  Formats whose value we re-verify ourselves instead of trusting the decoder.
CHECKSUM_LENGTHS = {"EAN-13": 13, "EAN-8": 8, "UPC-A": 12}
MAX_BARCODE_LENGTH = 128

_CONTROL_CHARS = re.compile("[\x00-\x1f\x7f​-‏﻿]")
_NUMERIC_WITH_SEPARATORS = re.compile(r"^[\d\s\-]+$")
#  OpenCV reports formats with underscores; normalise to the zxing spelling.
_OPENCV_FORMATS = {"EAN_13": "EAN-13", "EAN_8": "EAN-8", "UPC_A": "UPC-A", "UPC_E": "UPC-E"}
#  Quiet zone added around a detected quad before handing it to OpenCV; without
#  it the decoder rejects an otherwise perfectly readable code.
OPENCV_QUAD_MARGIN = (0.06, 0.16)

_local = threading.local()
_FORMATS = None


@dataclass(frozen=True)
class DecodeResult:
    value: str
    format: str
    decoder: str
    valid: bool
    variant: str = ""
    quad: np.ndarray | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict:
        return {"value": self.value, "format": self.format, "decoder": self.decoder,
                "valid": self.valid, "variant": self.variant}


# --------------------------------------------------------------------------
# Normalisation and validation
# --------------------------------------------------------------------------

def normalize_barcode(value: str | None) -> str:
    """Canonical form of a decoded string, so equal codes compare equal.

    Decoders occasionally return the same retail code with padding or grouping
    ("  869 1234 567890 "), which must not look like a different product.
    Separators are only stripped from purely numeric values, because a Code 128
    payload may legitimately contain spaces and hyphens.
    """
    if not value:
        return ""
    cleaned = _CONTROL_CHARS.sub("", str(value)).strip()
    if _NUMERIC_WITH_SEPARATORS.match(cleaned):
        cleaned = re.sub(r"[\s\-]", "", cleaned)
    return cleaned[:MAX_BARCODE_LENGTH]


def has_valid_mod10_checksum(value: str) -> bool:
    """Validate the GS1 modulo-10 check digit used by EAN/UPC codes."""
    if not value.isdigit() or len(value) < 2:
        return False
    total = sum(int(digit) * (3 if index % 2 == 0 else 1)
                for index, digit in enumerate(reversed(value[:-1])))
    return (10 - total % 10) % 10 == int(value[-1])


def expand_upce(value: str) -> str | None:
    """Expand an 8-digit UPC-E code to its 12-digit UPC-A equivalent."""
    if len(value) != 8 or not value.isdigit():
        return None
    system, body, check = value[0], value[1:7], value[7]
    if system not in "01":
        return None
    last = body[5]
    if last in "012":
        digits = f"{system}{body[:2]}{last}0000{body[2:5]}"
    elif last == "3":
        digits = f"{system}{body[:3]}00000{body[3:5]}"
    elif last == "4":
        digits = f"{system}{body[:4]}00000{body[4]}"
    else:
        digits = f"{system}{body[:5]}0000{last}"
    return f"{digits}{check}"


def is_valid_barcode(value: str, barcode_format=None) -> bool:
    """Reject invalid codes before they can reach the UI or the database."""
    if not value:
        return False
    value = normalize_barcode(value)
    if not value or len(value) > MAX_BARCODE_LENGTH:
        return False
    format_name = str(barcode_format or "")

    if not format_name and len(value) == 13 and value.isdigit():
        format_name = "EAN-13"
    expected_length = CHECKSUM_LENGTHS.get(format_name)
    if expected_length is not None:
        return len(value) == expected_length and has_valid_mod10_checksum(value)
    if format_name == "UPC-E":
        # The UPC-E check digit is defined over the expanded UPC-A form.
        expanded = expand_upce(value)
        return bool(expanded) and has_valid_mod10_checksum(expanded)
    # zxing-cpp has already validated the symbol's own integrity for these.
    return format_name in {"Code 128", "QR Code"}


def lookup_aliases(value: str, barcode_format: str = "") -> list[str]:
    """Equivalent representations of the same retail item, for product lookup.

    A catalogue may store a product as UPC-A (12 digits) while the camera
    reports the EAN-13 form with a leading zero (or the other way round); both
    identify the same item, so both are worth looking up.
    """
    value = normalize_barcode(value)
    aliases = [value]
    if not value.isdigit():
        return aliases
    if len(value) == 12:
        aliases.append(f"0{value}")
    elif len(value) == 13 and value.startswith("0"):
        aliases.append(value[1:])
    elif len(value) == 8 and str(barcode_format) == "UPC-E":
        expanded = expand_upce(value)
        if expanded:
            aliases.extend([expanded, f"0{expanded}"])
    return list(dict.fromkeys(aliases))


# --------------------------------------------------------------------------
# Decoders
# --------------------------------------------------------------------------

def _zxing_formats():
    """Restrict the decoder to the retail formats SmartCart supports.

    Cached: this runs on every decode attempt, several times per frame.
    """
    global _FORMATS
    if _FORMATS is None:
        _FORMATS = (zxingcpp.BarcodeFormat.EAN13, zxingcpp.BarcodeFormat.EAN8,
                    zxingcpp.BarcodeFormat.UPCA, zxingcpp.BarcodeFormat.UPCE,
                    zxingcpp.BarcodeFormat.Code128, zxingcpp.BarcodeFormat.QRCode)
    return _FORMATS


def decode_image(image: np.ndarray, config: ScannerConfig = DEFAULT_CONFIG,
                 variant: str = "") -> list[DecodeResult]:
    """Run zxing-cpp over one prepared image and return validated reads."""
    if zxingcpp is None:
        raise RuntimeError("zxing-cpp yüklü değil. `pip install -r requirements.txt` çalıştırın.")
    if image is None or image.size == 0:
        return []
    # Downscaling helps large codes but destroys small/distant ones in a crop.
    use_downscale = config.try_downscale and max(image.shape[:2]) >= 720
    try:
        symbols = zxingcpp.read_barcodes(
            image, formats=_zxing_formats(), try_rotate=config.try_rotate,
            try_downscale=use_downscale, try_invert=config.try_invert)
    except Exception:
        # A decoder crash on one odd frame must never take the scan loop down.
        return []

    results = []
    for symbol in symbols:
        value = normalize_barcode(symbol.text)
        format_name = str(symbol.format)
        if not value:
            continue
        valid = bool(symbol.valid) and is_valid_barcode(value, format_name)
        results.append(DecodeResult(value, format_name, "zxing-cpp", valid, variant,
                                    _quad_from_position(symbol.position)))
    return results


def decode_with_opencv(image: np.ndarray, quad: np.ndarray | None = None,
                       variant: str = "opencv-quad") -> list[DecodeResult]:
    """Decode using OpenCV's barcode reader, optionally on a known quadrangle."""
    if image is None or image.size == 0:
        return []
    detector = getattr(_local, "barcode_detector", None)
    if detector is None:
        detector = cv2.barcode.BarcodeDetector()
        _local.barcode_detector = detector
    try:
        if quad is None:
            ok, values, formats, _points = detector.detectAndDecodeWithType(image)
        else:
            padded = expand_quad(quad, *OPENCV_QUAD_MARGIN)
            ok, values, formats = detector.decodeWithType(image, padded.reshape(1, 4, 2))
    except cv2.error:
        return []
    if not ok or not values:
        return []

    results = []
    for value, format_name in zip(values, list(formats) + [""] * len(values)):
        value = normalize_barcode(value)
        if not value:
            continue
        canonical = _OPENCV_FORMATS.get(str(format_name), str(format_name))
        results.append(DecodeResult(value, canonical, "opencv", is_valid_barcode(value, canonical),
                                    variant, np.asarray(quad) if quad is not None else None))
    return results


def _quad_from_position(position) -> np.ndarray | None:
    """Convert a zxing-cpp `Position` into a (4, 2) corner array."""
    try:
        corners = (position.top_left, position.top_right,
                   position.bottom_right, position.bottom_left)
        return np.array([[float(p.x), float(p.y)] for p in corners], dtype=np.float32)
    except (AttributeError, TypeError):
        return None
