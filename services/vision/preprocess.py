"""Turn a detected region into images a decoder has a real chance with.

Two things happen here.  First, perspective correction: a barcode photographed
at an angle has bars that converge, and rectifying the detected quadrangle turns

    /||||||||/        into        ||||||||||

which is the difference between "no read" and "instant read" for a code held at
30 degrees or more.  Second, a small ladder of enhancement variants, ordered
cheapest first, so the caller can stop as soon as something decodes.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from services.config import DEFAULT_CONFIG, ScannerConfig
from services.vision.quality import to_gray


@dataclass
class RectifiedRegion:
    image: np.ndarray
    #  0..1, how rectangular the source quad was (1.0 = head-on, no skew).
    perspective_score: float
    scale: float
    method: str

    def as_dict(self) -> dict:
        height, width = self.image.shape[:2]
        return {"size": [int(width), int(height)], "method": self.method,
                "perspective_score": round(self.perspective_score, 3),
                "scale": round(self.scale, 2)}


def order_quad(quad: np.ndarray) -> np.ndarray:
    """Order corners so the barcode's long axis becomes horizontal.

    Detectors return corners in their own conventions (OpenCV starts at the
    bottom-left, `minAreaRect` is unordered).  Normalising here means the
    rectified output always has upright bars, whatever found the region.
    """
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    centre = quad.mean(axis=0)
    # Sort corners counter-clockwise around the centre, then rotate the sequence
    # so index 0 is the top-left-most point.
    angles = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])
    ordered = quad[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)

    edge_top = np.linalg.norm(ordered[0] - ordered[1])
    edge_side = np.linalg.norm(ordered[1] - ordered[2])
    if edge_side > edge_top:
        # The long edge is currently vertical: rotate so bars stand upright.
        ordered = np.roll(ordered, -1, axis=0)
    return ordered.astype(np.float32)


def expand_quad(quad: np.ndarray, fraction_x: float, fraction_y: float) -> np.ndarray:
    """Grow a quad along its own axes to include the barcode's quiet zone.

    Detectors report the bars themselves, but EAN/UPC decoders need the blank
    margin either side to find the symbol's start and stop patterns -- measured
    here, OpenCV's decoder fails on a tight quad and succeeds on a padded one.
    """
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    axis_x = quad[1] - quad[0]
    axis_y = quad[3] - quad[0]
    width, height = float(np.linalg.norm(axis_x)), float(np.linalg.norm(axis_y))
    if width < 1e-3 or height < 1e-3:
        return quad
    step_x = axis_x / width * width * fraction_x
    step_y = axis_y / height * height * fraction_y
    offsets = np.stack([-step_x - step_y, step_x - step_y, step_x + step_y, -step_x + step_y])
    return (quad + offsets).astype(np.float32)


def perspective_score(quad: np.ndarray) -> float:
    """How close the quad is to a rectangle: 1.0 head-on, ~0 heavily skewed."""
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    edges = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    if min(edges) < 1e-3:
        return 0.0
    # Opposite edges of a rectangle are equal; the ratio degrades with skew.
    ratio_a = min(edges[0], edges[2]) / max(edges[0], edges[2])
    ratio_b = min(edges[1], edges[3]) / max(edges[1], edges[3])
    angles = []
    for i in range(4):
        v1 = quad[(i - 1) % 4] - quad[i]
        v2 = quad[(i + 1) % 4] - quad[i]
        cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        angles.append(abs(cosine))          # 0 when the corner is a right angle
    squareness = 1.0 - min(1.0, float(np.mean(angles)))
    return float(max(0.0, min(1.0, (ratio_a + ratio_b) / 2 * 0.6 + squareness * 0.4)))


def rectify(image: np.ndarray, quad: np.ndarray,
            config: ScannerConfig = DEFAULT_CONFIG) -> RectifiedRegion | None:
    """Warp a detected quadrangle into an upright, padded, well-sized barcode.

    The warp deliberately samples *outside* the quad to build the padding, so
    the quiet zone contains the label's real surroundings rather than replicated
    bar pixels, which would confuse the decoders.
    """
    if image is None or image.size == 0:
        return None
    ordered = order_quad(quad)
    width = float(max(np.linalg.norm(ordered[0] - ordered[1]),
                      np.linalg.norm(ordered[3] - ordered[2])))
    height = float(max(np.linalg.norm(ordered[1] - ordered[2]),
                       np.linalg.norm(ordered[0] - ordered[3])))
    if width < 12 or height < 6:
        return None

    scale = _target_scale(width, config)
    out_w, out_h = int(round(width * scale)), int(round(height * scale))
    pad_x, pad_y = int(out_w * config.roi_padding_x), int(out_h * config.roi_padding_y)
    pad_x, pad_y = max(pad_x, 8), max(pad_y, 4)
    canvas_w, canvas_h = out_w + 2 * pad_x, out_h + 2 * pad_y
    if max(canvas_w, canvas_h) > config.max_processing_edge:
        return None

    destination = np.float32([[pad_x, pad_y], [pad_x + out_w, pad_y],
                              [pad_x + out_w, pad_y + out_h], [pad_x, pad_y + out_h]])
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    warped = cv2.warpPerspective(image, matrix, (canvas_w, canvas_h), flags=interpolation,
                                 borderMode=cv2.BORDER_REPLICATE)
    return RectifiedRegion(warped, perspective_score(quad), scale, "rectified")


def crop(image: np.ndarray, box: tuple[int, int, int, int],
         config: ScannerConfig = DEFAULT_CONFIG) -> RectifiedRegion | None:
    """Padded axis-aligned crop -- the fallback when rectification is refused.

    Keeping this path matters: perspective correction can distort a region the
    detector bounded loosely, and the untouched pixels sometimes decode when the
    warped ones do not.
    """
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * config.roi_padding_x) + 6
    pad_y = int((y2 - y1) * config.roi_padding_y) + 4
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    if x2 - x1 < 12 or y2 - y1 < 6:
        return None
    region = image[y1:y2, x1:x2]
    scale = _target_scale(x2 - x1, config)
    if scale != 1.0:
        region = _resize(region, scale, config)
    return RectifiedRegion(region, 1.0, scale, "crop")


def _target_scale(width: float, config: ScannerConfig) -> float:
    """Scale that brings a region towards the decoder's comfortable width."""
    if width <= 0:
        return 1.0
    scale = config.roi_target_width / width
    return float(min(config.upscale_factor, max(0.5, scale)))


def _resize(image: np.ndarray, scale: float, config: ScannerConfig) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(scale, config.max_processing_edge / max(1, max(height, width)))
    if abs(scale - 1.0) < 0.05:
        return image
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=interpolation)


def reduce_glare(gray: np.ndarray, threshold: int = 230) -> np.ndarray:
    """Fill specular highlights so bar modules become readable again.

    Uses a gentler inpaint radius so thin bars adjacent to the highlight
    are not smeared away.
    """
    bright = (gray >= threshold).astype(np.uint8) * 255
    density = bright.sum() / (gray.size * 255)
    if density < 0.003:
        return gray
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(bright, kernel, iterations=1)
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


def unsharp(gray: np.ndarray, strength: float = 1.6, sigma: float = 1.0) -> np.ndarray:
    """Unsharp mask to recover bar contrast lost on shiny plastics."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)


def glare_variants(region: np.ndarray, config: ScannerConfig = DEFAULT_CONFIG):
    """Fast glare recovery variants, run before the heavy escalation ladder."""
    gray = to_gray(region)

    # path 1: unsharp mask — often enough when contrast is the only problem
    sharp = unsharp(gray)
    yield "unsharp", sharp
    clahe_sharp = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(sharp)
    yield "unsharp-clahe", clahe_sharp
    yield "unsharp-adaptive", cv2.adaptiveThreshold(
        clahe_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)

    # path 2: inpaint highlights then enhance
    deglared = reduce_glare(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(deglared)
    yield "deglare-clahe", clahe
    yield "deglare-adaptive", cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)
    if max(region.shape[:2]) < 400:
        scale = min(config.upscale_factor, 3.0)
        upscaled = _resize(clahe, scale, config)
        yield "deglare-upscale", upscaled
        yield "deglare-upscale-adaptive", cv2.adaptiveThreshold(
            upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 6)


def variants(region: np.ndarray, level: int, config: ScannerConfig = DEFAULT_CONFIG,
             *, glare: bool = False):
    """Yield (name, image) decoding inputs for one escalation level.

    Level 0 is nearly free and handles clean, well-lit labels.  Level 1 fixes
    uneven lighting and softness.  Level 2 is the expensive last resort for
    small, shiny or low-contrast codes and only runs when the frame is sharp
    enough for it to matter.
    """
    gray = to_gray(region)
    if level == 0:
        yield "gray", gray
        if glare:
            yield from glare_variants(region, config)
        return
    if level == 1:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        yield "clahe", clahe
        yield "sharpen", cv2.addWeighted(clahe, 1.45, cv2.GaussianBlur(clahe, (0, 0), 1.4), -0.45, 0)
        return
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    yield "otsu", cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    yield "adaptive", cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 31, 4)
    upscale = 3.0 if max(region.shape[:2]) < 280 else 2.0
    upscaled = _resize(clahe, upscale, config)
    yield "upscale-clahe", upscaled
    yield "upscale-adaptive", cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 41, 6)


MAX_LEVEL = 2
