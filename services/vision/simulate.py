"""Synthetic barcode scenes and image degradations for tests and benchmarks.

This is the hardware-free half of the test strategy: it renders a real EAN/UPC
symbol into a camera-sized frame and then applies the things that actually go
wrong in a shop -- the product is tilted, held too far away, moving, under a
dim shelf, or the label is glossy.  Unit tests and `benchmark_barcode.py` both
build their inputs from here, so the pipeline can be measured without a camera.
"""
import cv2
import numpy as np

try:
    import zxingcpp
except ImportError:
    zxingcpp = None


def render_barcode(value: str = "8690530046269", barcode_format: str = "EAN-13",
                   scale: int = 3) -> np.ndarray:
    """Render a real symbol as a grayscale image."""
    if zxingcpp is None:
        raise RuntimeError("zxing-cpp yüklü değil.")
    formats = {"EAN-13": zxingcpp.BarcodeFormat.EAN13, "EAN-8": zxingcpp.BarcodeFormat.EAN8,
               "UPC-A": zxingcpp.BarcodeFormat.UPCA, "Code 128": zxingcpp.BarcodeFormat.Code128}
    symbol = zxingcpp.create_barcode(value, formats[barcode_format])
    return np.asarray(zxingcpp.write_barcode_to_image(symbol, scale=scale))


def scene(value: str = "8690530046269", barcode_format: str = "EAN-13", scale: int = 3,
          size: tuple[int, int] = (1280, 720), background: int = 235,
          position: tuple[float, float] = (0.5, 0.5),
          code_width_px: int | None = None) -> np.ndarray:
    """Place a barcode on a plain background, as a BGR camera-sized frame.

    `code_width_px` renders the symbol at high resolution and scales it down to
    an exact width, which is how distance is simulated: what limits a far-away
    read is pixels-per-module, and that only behaves realistically if the
    symbol is resampled rather than drawn with fewer, crisper modules.
    """
    width, height = size
    canvas = np.full((height, width), background, np.uint8)
    code = render_barcode(value, barcode_format, 6 if code_width_px else scale)
    if code_width_px:
        source_h, source_w = code.shape[:2]
        target_w = max(8, int(code_width_px))
        target_h = max(6, int(source_h * target_w / source_w))
        code = cv2.resize(code, (target_w, target_h), interpolation=cv2.INTER_AREA)
    code_h, code_w = code.shape[:2]
    if code_h >= height or code_w >= width:
        code = cv2.resize(code, (min(code_w, width - 4), min(code_h, height - 4)),
                          interpolation=cv2.INTER_AREA)
        code_h, code_w = code.shape[:2]
    x = int(np.clip(width * position[0] - code_w / 2, 0, width - code_w))
    y = int(np.clip(height * position[1] - code_h / 2, 0, height - code_h))
    #  A quiet margin around the label, as on real packaging.
    canvas[max(0, y - 12):y + code_h + 12, max(0, x - 16):x + code_w + 16] = 255
    canvas[y:y + code_h, x:x + code_w] = code
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def blank(size: tuple[int, int] = (1280, 720), value: int = 235) -> np.ndarray:
    return np.full((size[1], size[0], 3), value, np.uint8)


# --- degradations ---------------------------------------------------------

def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def perspective(image: np.ndarray, amount: float = 0.2) -> np.ndarray:
    """Tilt the frame as if the product were held at an angle to the lens."""
    height, width = image.shape[:2]
    shift = width * amount * 0.5
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32([[shift, height * amount * 0.25], [width, 0],
                         [width - shift, height - height * amount * 0.25], [0, height]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target),
                               (width, height), borderMode=cv2.BORDER_REPLICATE)


def blur(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma) if sigma > 0 else image


def motion_blur(image: np.ndarray, length: int = 9) -> np.ndarray:
    """Horizontal smear, i.e. a product moving past the camera."""
    if length < 2:
        return image
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel)


def brightness(image: np.ndarray, factor: float = 1.0, offset: float = 0.0) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=factor, beta=offset)


def noise(image: np.ndarray, sigma: float = 8.0, seed: int | None = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    noisy = image.astype(np.int16) + generator.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def glare(image: np.ndarray, strength: float = 0.75, radius: float = 0.22) -> np.ndarray:
    """Specular highlight, as from a shop light on glossy packaging."""
    height, width = image.shape[:2]
    y, x = np.ogrid[:height, :width]
    centre_y, centre_x = height * 0.45, width * 0.5
    distance = np.sqrt(((x - centre_x) / (width * radius)) ** 2 +
                       ((y - centre_y) / (height * radius)) ** 2)
    mask = np.clip(1.0 - distance, 0, 1) * strength
    if image.ndim == 3:
        mask = mask[:, :, None]
    return np.clip(image + mask * 255, 0, 255).astype(np.uint8)


def resize(image: np.ndarray, factor: float) -> np.ndarray:
    """Simulate distance: the same scene captured with fewer pixels on target."""
    height, width = image.shape[:2]
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (max(1, int(width * factor)), max(1, int(height * factor))),
                      interpolation=interpolation)


def barcode_pixel_width(centimetres: float, frame_width: int = 1280,
                        symbol_mm: float = 38.0, fov_degrees: float = 60.0) -> int:
    """How many pixels wide an EAN-13 is at a given distance.

    A typical webcam has a ~60 degree horizontal field of view, so at distance d
    it sees a strip 2*d*tan(fov/2) wide; a 38 mm symbol occupies that fraction
    of the frame.  This is the number that decides whether a read is physically
    possible at all -- an EAN-13 is 95 modules, and the decoders need roughly
    1.5-2 px per module.
    """
    field_width_mm = 2 * centimetres * 10 * np.tan(np.radians(fov_degrees / 2))
    return max(8, int(round(frame_width * symbol_mm / field_width_mm)))


def at_distance(value: str = "8690530046269", centimetres: float = 30,
                size: tuple[int, int] = (1280, 720), **kwargs) -> np.ndarray:
    """A frame whose barcode is sized as if photographed from `centimetres`."""
    return scene(value=value, size=size,
                 code_width_px=barcode_pixel_width(centimetres, size[0]), **kwargs)
