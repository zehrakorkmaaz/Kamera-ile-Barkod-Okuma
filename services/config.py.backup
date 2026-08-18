"""Central tuning for the camera and barcode pipeline.

Every threshold the scanner depends on lives here instead of being scattered
through the vision modules, so behaviour can be tuned per machine/camera from
the environment without editing code.  Defaults are chosen for a typical
720p/1080p USB webcam at 20-75 cm and are deliberately conservative: nothing
here disables the cheap decode path, the values only decide how much *extra*
work a frame is worth.

Override any field with a `SMARTCART_` prefixed environment variable, e.g.

    SMARTCART_CAMERA_INDEX=1 SMARTCART_CAMERA_WIDTH=1920 python app.py
"""
from dataclasses import dataclass, fields
import os


ENV_PREFIX = "SMARTCART_"


@dataclass(frozen=True)
class ScannerConfig:
    # --- camera capture -------------------------------------------------
    camera_index: int = 0
    # 1080p is the default because reading distance is decided by pixels per
    # barcode module, and nothing else: measured on this pipeline, 720p reads an
    # EAN-13 to roughly 30 cm and 1080p to roughly 50 cm.  The extra cost is
    # about 10 ms, and only on frames that fail to decode.
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 30
    # MJPG lets most USB cameras deliver 720p/1080p at full frame rate; the raw
    # YUYV fallback often caps 1080p at ~5 fps, which ruins scan latency.
    camera_fourcc: str = "MJPG"
    # Resolutions tried, in order, when the requested one is not granted.
    camera_fallback_resolutions: tuple[tuple[int, int], ...] = (
        (1280, 720), (1024, 576), (640, 480))
    camera_autofocus: bool = True
    # Consecutive failed reads before the device is reopened (USB unplug/replug).
    reconnect_after_failures: int = 25
    reconnect_backoff_seconds: float = 1.5
    reconnect_max_backoff_seconds: float = 10.0

    # --- preview --------------------------------------------------------
    preview_fps: int = 20
    preview_quality: int = 75
    # Draw the detected barcode quad on the preview so the user sees the lock-on.
    preview_overlay: bool = True

    # --- frame quality ---------------------------------------------------
    quality_work_width: int = 480
    # Variance-of-Laplacian below this reads as motion blur / out of focus.
    blur_threshold: float = 18.0
    brightness_min: float = 55.0
    # Deliberately high: a white label filling the scan area is normal, so real
    # overexposure is better caught by the glare ratio below.
    brightness_max: float = 235.0
    contrast_min: float = 18.0
    # Fraction of near-white pixels that indicates specular glare on the label.
    glare_ratio_max: float = 0.18

    # --- barcode detection ------------------------------------------------
    detection_width: int = 640
    # Long edge of a candidate, in full-resolution pixels.
    min_barcode_size: int = 40
    max_barcode_size: int = 2400
    max_candidates: int = 3
    candidate_min_aspect: float = 1.25
    use_opencv_detector: bool = True
    use_gradient_detector: bool = True
    # Candidate long edge / ROI width below this means "too far away".
    min_barcode_width_ratio: float = 0.09

    # --- ROI / preprocessing ----------------------------------------------
    # Quiet zone kept around the code; EAN/UPC need clear margins to decode.
    roi_padding_x: float = 0.12
    roi_padding_y: float = 0.30
    # Rectified codes are scaled towards this width instead of a blind fixed
    # multiplier: an EAN-13 is 95 modules wide, so ~480 px gives ~5 px/module,
    # comfortably above the ~2 px/module both decoders need.
    roi_target_width: int = 480
    # Hard cap on enlargement.  A region too small to carry the modules cannot be
    # rescued by interpolation, and pretending otherwise only burns CPU.
    upscale_factor: float = 3.0
    max_processing_edge: int = 1800

    # --- decoding ----------------------------------------------------------
    use_opencv_decoder: bool = True
    try_rotate: bool = True
    try_downscale: bool = True
    try_invert: bool = True
    # Time budget for one frame's escalation.  The cheap passes always run; the
    # heavy passes stop once the budget is gone, which bounds latency and CPU.
    frame_budget_ms: float = 55.0
    # Heavy (upscale/threshold) passes are pointless on a badly blurred frame.
    heavy_pass_min_sharpness: float = 12.0

    # --- confidence ---------------------------------------------------------
    weight_decode: float = 45.0
    weight_checksum: float = 15.0
    weight_decoder_agreement: float = 12.0
    weight_size: float = 10.0
    weight_sharpness: float = 8.0
    weight_perspective: float = 5.0
    weight_temporal: float = 5.0
    confidence_high: float = 90.0
    confidence_medium: float = 70.0
    confidence_threshold: float = 70.0
    # A single read this strong is trusted immediately (keeps the "supermarket
    # scanner" feel); anything weaker must repeat across frames.
    instant_confirm_confidence: float = 70.0

    # --- temporal confirmation / state machine ------------------------------
    confirmation_frames: int = 2
    confirmation_window_seconds: float = 1.5
    # Consecutive empty frames before an in-view barcode is considered gone.
    miss_tolerance: int = 3
    scan_cooldown_seconds: float = 1.5
    # A hint must hold this long before it is shown, so the UI does not flicker.
    # At 20-30 scans/second this is several agreeing frames, which is enough to
    # stop flapping while still reacting before the user has moved on.
    hint_hold_seconds: float = 0.25
    # How long a finished scan keeps its result on screen before the UI goes
    # back to guiding the user towards the next product.
    result_hold_seconds: float = 1.2

    # --- diagnostics / security ----------------------------------------------
    # Off by default: frames are never written to disk unless explicitly asked.
    debug_image_dir: str = ""
    # Set to a folder of images to run the whole app without a camera.
    test_image_dir: str = ""
    debug_attempt_history: int = 14
    latency_window: int = 120

    @classmethod
    def from_env(cls, env=None, **overrides) -> "ScannerConfig":
        """Build a config from SMARTCART_* environment variables."""
        env = os.environ if env is None else env
        values = {}
        for spec in fields(cls):
            raw = env.get(f"{ENV_PREFIX}{spec.name.upper()}")
            if raw is None or spec.name in overrides:
                continue
            try:
                values[spec.name] = _coerce(raw, spec.type, getattr(cls, spec.name))
            except (TypeError, ValueError):
                # A malformed override must never stop the scanner from starting.
                continue
        return cls(**{**values, **overrides})

    def replace(self, **changes) -> "ScannerConfig":
        from dataclasses import replace as _replace
        return _replace(self, **changes)


def _coerce(raw: str, type_hint, default):
    """Convert an environment string to the field's declared type."""
    raw = raw.strip()
    if type_hint in (bool, "bool"):
        return raw.lower() in {"1", "true", "yes", "on"}
    if type_hint in (int, "int"):
        return int(float(raw))
    if type_hint in (float, "float"):
        return float(raw)
    if isinstance(default, tuple):
        # "1280x720,640x480" -> ((1280, 720), (640, 480))
        pairs = []
        for item in raw.split(","):
            width, _, height = item.strip().lower().partition("x")
            pairs.append((int(width), int(height)))
        return tuple(pairs)
    return raw


DEFAULT_CONFIG = ScannerConfig()
