"""Owns one local camera and turns its frames into confirmed scan events.

Threading is deliberately minimal -- two threads, one shared "newest frame"
slot, no queues:

- `_capture_loop` only reads frames and refreshes the preview.  It never
  decodes, so a slow decode can neither stall the live image nor delay the next
  grab, and it never lets frames pile up.
- `_scan_loop` always works on the *latest* frame.  Frames that arrive while it
  is busy are skipped, not queued, so scan latency is bounded by one decode
  rather than by how much backlog accumulated.  Skipped frames are counted as
  `dropped_frames` instead of being silently lost.

The pipeline, tracker and metrics do the actual thinking; this class is the
plumbing that connects them to a device, an MJPEG preview and the product
catalogue.
"""
import logging
import threading
import time

import cv2
import numpy as np

from services.camera_device import CameraDevice, open_failed_message
from services.config import DEFAULT_CONFIG, ScannerConfig
from services.vision.metrics import ScanMetrics
from services.vision.pipeline import BarcodePipeline
from services.vision.tracking import ScanState, ScanTracker
from services.vision.verification import (
    NullWeightSource, VisionSignal, WeightSignal, combine_confidence, expected_weight_grams,
)

logger = logging.getLogger("smartcart.camera")

OVERLAY_SECONDS = 0.35
OVERLAY_COLOURS = {"locked": (90, 220, 120), "tracking": (250, 200, 60)}


class CameraService:
    """Camera capture, real-time barcode scanning and scan-event publication."""

    def __init__(self, index=0, scan_every_n_frames=5, config: ScannerConfig | None = None,
                 lookup=None, weight_source=None, device=None):
        self.config = config or ScannerConfig.from_env()
        self.index = index
        # Retained for API/diagnostic compatibility: the scan loop is driven by
        # frame availability, not by a fixed frame count.
        self.scan_every_n_frames = scan_every_n_frames
        self.metrics = ScanMetrics(window=self.config.latency_window)
        self.pipeline = BarcodePipeline(self.config, self.metrics)
        self.tracker = ScanTracker(self.config, self.metrics)
        #  Injected by the app so this module never imports the database layer.
        self._lookup = lookup
        self._weight_source = weight_source or NullWeightSource()

        # `device` is any object with the CameraDevice surface, which is how
        # test mode replays a folder of images through the real pipeline.
        self._device = device or CameraDevice(config=self.config, index=index)
        self._capture_thread = None
        self._scan_thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._jpeg = None
        self._error = None
        self._frame_count = 0
        self._event_id = 0
        self._last_barcode = None
        self._last_event = None
        self._last_product = None
        self._last_found = None
        self._latest_frame = None
        self._latest_frame_id = 0
        self._overlay = None
        self._fps = 0.0
        self._fps_started = time.monotonic()
        self._fps_frames = 0
        self._scan_fps = 0.0
        self._scan_fps_started = time.monotonic()
        self._scan_fps_frames = 0
        self._last_decode_ms = None
        self._debug = {"last_scan_at": None, "candidate_count": 0, "decoder_result": None,
                       "decoder_method": None, "roi": None, "attempts": []}

    # -- lifecycle ---------------------------------------------------------

    @property
    def error(self):
        with self._lock:
            return self._error

    def _set_error(self, message):
        with self._lock:
            self._error = message

    def start(self) -> bool:
        if self._capture_thread and self._capture_thread.is_alive():
            return True
        if not self._device.open():
            self._set_error(self._device.error or open_failed_message(self.index))
            return False
        self._set_error(None)
        self._stop.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._capture_thread.start()
        self._scan_thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        for thread in (self._capture_thread, self._scan_thread):
            if thread:
                thread.join(timeout=2)
        self._device.close()
        self._capture_thread = self._scan_thread = None
        with self._lock:
            self._latest_frame, self._latest_frame_id = None, 0
            self._jpeg, self._overlay = None, None

    @property
    def running(self) -> bool:
        return bool(self._capture_thread and self._capture_thread.is_alive())

    # -- scan area ---------------------------------------------------------

    @staticmethod
    def _visible_guide_roi(frame):
        """Map the CSS 16:9/object-contain guide (7-93%, 10-90%) to frame pixels."""
        height, width = frame.shape[:2]
        container_aspect = 16 / 9
        frame_aspect = width / height
        if frame_aspect >= container_aspect:
            rendered_width, rendered_height = 1.0, container_aspect / frame_aspect
            offset_x, offset_y = 0.0, (1.0 - rendered_height) / 2
        else:
            rendered_width, rendered_height = frame_aspect / container_aspect, 1.0
            offset_x, offset_y = (1.0 - rendered_width) / 2, 0.0

        # Intersect the visual guide with the actual rendered video (important
        # when a fallback camera returns 4:3 and object-fit introduces bars).
        guide_left, guide_top, guide_right, guide_bottom = .07, .10, .93, .90
        left = max(guide_left, offset_x)
        right = min(guide_right, offset_x + rendered_width)
        top = max(guide_top, offset_y)
        bottom = min(guide_bottom, offset_y + rendered_height)
        x1 = int(max(0, min(width, (left - offset_x) / rendered_width * width)))
        x2 = int(max(0, min(width, (right - offset_x) / rendered_width * width)))
        y1 = int(max(0, min(height, (top - offset_y) / rendered_height * height)))
        y2 = int(max(0, min(height, (bottom - offset_y) / rendered_height * height)))
        return x1, y1, x2, y2

    # -- capture -----------------------------------------------------------

    def _capture_loop(self):
        """Grabs frames and refreshes the preview; never decodes."""
        preview_interval = 1.0 / max(1, self.config.preview_fps)
        next_preview = 0.0
        while not self._stop.is_set():
            ok, frame = self._device.read()
            if not ok:
                if self._device.needs_reconnect():
                    self._set_error(self._device.error)
                    if not self._device.reconnect():
                        continue
                    self.metrics.increment("camera_reconnects")
                    self._set_error(None)
                else:
                    time.sleep(0.02)
                continue

            self._frame_count += 1
            self.metrics.increment("total_frames")
            self._track_capture_fps()
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_id += 1
                if self._error:
                    self._error = None

            now = time.monotonic()
            if now >= next_preview:
                next_preview = now + preview_interval
                self._encode_preview(frame)

    def _track_capture_fps(self):
        self._fps_frames += 1
        elapsed = time.monotonic() - self._fps_started
        if elapsed >= 1:
            self._fps = self._fps_frames / elapsed
            self._fps_frames, self._fps_started = 0, time.monotonic()

    def _encode_preview(self, frame):
        """JPEG for the MJPEG stream, with the current lock-on drawn on top.

        Encoding is throttled to the preview rate rather than run per captured
        frame: at 30 fps capture that alone was a constant, pointless CPU cost.
        """
        if self.config.preview_overlay:
            overlay = self._current_overlay()
            if overlay is not None:
                frame = frame.copy()
                quad, colour = overlay
                cv2.polylines(frame, [quad.astype(np.int32)], True,
                              OVERLAY_COLOURS.get(colour, OVERLAY_COLOURS["tracking"]), 3,
                              lineType=cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.config.preview_quality])
        if ok:
            with self._lock:
                self._jpeg = encoded.tobytes()

    def _current_overlay(self):
        with self._lock:
            overlay = self._overlay
        if overlay is None or time.monotonic() - overlay[2] > OVERLAY_SECONDS:
            return None
        return overlay[0], overlay[1]

    # -- scanning ----------------------------------------------------------

    def _scan_loop(self):
        """Always decodes the newest frame available; never queues stale ones."""
        last_processed_id = 0
        while not self._stop.is_set():
            with self._lock:
                frame, frame_id = self._latest_frame, self._latest_frame_id
            if frame is None or frame_id == last_processed_id:
                time.sleep(0.004)
                continue
            skipped = frame_id - last_processed_id - 1
            if skipped > 0:
                self.metrics.increment("dropped_frames", skipped)
            last_processed_id = frame_id
            try:
                self._scan_frame(frame)
            except Exception as exc:                      # never kill the loop
                self.metrics.increment("pipeline_errors")
                logger.exception("SCAN_FAILED %s", exc)
                self._set_error(f"Barkod taraması başarısız: {exc}")

    def _scan_frame(self, frame):
        """One frame: crop the scan area, run the pipeline, update the tracker."""
        roi_bounds = list(self._visible_guide_roi(frame))
        scan_area = frame[roi_bounds[1]:roi_bounds[3], roi_bounds[0]:roi_bounds[2]]
        result = self.pipeline.process(scan_area, hits_for=self._hits_for)
        self.metrics.increment("processed_frames")
        event = self.tracker.update(result)

        self._track_scan_fps()
        self._publish_overlay(result, roi_bounds)
        self._publish_debug(result, roi_bounds)
        if event is not None:
            self._handle_event(event)

    def _hits_for(self, value: str) -> int:
        """Frames that recently agreed on this value, including the current one."""
        track = self.tracker.tracks.get(value)
        window = self.config.confirmation_window_seconds
        return 1 + (track.recent_hits(time.monotonic(), window) if track else 0)

    def _handle_event(self, event):
        """Publish the barcode immediately, then enrich it with product data."""
        logger.info("SCAN_CONFIRMED barcode=%s confidence=%.1f frames=%d in %.0fms",
                    event.value, event.confidence, event.frames, event.confirmation_ms)

        # Critical latency path: the barcode is already valid. Publish the event
        # before SQLite/product lookup so the UI can react immediately. The
        # product fields are filled in just below and remain available to the
        # next status poll.
        initial = event.as_dict()
        with self._lock:
            self._last_barcode = event.value
            self._event_id += 1
            self._last_event = initial
            self._last_product = None
            self._last_found = None

        product, found = None, None
        if self._lookup is not None:
            try:
                product = self._lookup(event.value)
                found = product is not None
                logger.info("PRODUCT_LOOKUP barcode=%s found=%s", event.value, found)
            except Exception as exc:                     # a dead database must not
                logger.warning("PRODUCT_LOOKUP_FAILED %s", exc)   # break scanning
                found = None
        if found is not None:
            self.tracker.note_product(found)

        final = self._verify(event, product)
        with self._lock:
            self._last_event = {**initial, **final.as_dict()}
            self._last_product = product
            self._last_found = found

    def _verify(self, event, product):
        """Fuse vision with weight evidence when a scale is eventually attached.

        With no load cell present `read_delta_grams()` returns None and the
        final confidence is exactly the vision confidence.
        """
        vision = VisionSignal(event.value, event.confidence, event.level)
        weight = None
        try:
            measured = self._weight_source.read_delta_grams()
        except Exception as exc:
            logger.warning("WEIGHT_SOURCE_FAILED %s", exc)
            measured = None
        if measured is not None:
            expected = expected_weight_grams((product or {}).get("weight"))
            weight = WeightSignal(measured_grams=measured, expected_grams=expected)
        return combine_confidence(vision, weight, self.config)

    def _track_scan_fps(self):
        self._scan_fps_frames += 1
        elapsed = time.monotonic() - self._scan_fps_started
        if elapsed >= 1:
            self._scan_fps = self._scan_fps_frames / elapsed
            self._scan_fps_frames, self._scan_fps_started = 0, time.monotonic()

    def _publish_overlay(self, result, roi_bounds):
        """Store the detected barcode outline in full-frame coordinates."""
        if not self.config.preview_overlay:
            return
        quad = result.quad
        if quad is None and result.candidates:
            quad = result.candidates[0].quad
        if quad is None:
            return
        offset = np.array([roi_bounds[0], roi_bounds[1]], dtype=np.float32)
        colour = "locked" if result.value else "tracking"
        with self._lock:
            self._overlay = (np.asarray(quad, dtype=np.float32) + offset, colour,
                             time.monotonic())

    def _publish_debug(self, result, roi_bounds):
        with self._lock:
            self._last_decode_ms = round(result.duration_ms, 1)
            self._debug = {"last_scan_at": time.time(), "candidate_count": len(result.candidates),
                           "decoder_result": result.value, "decoder_method": result.attempts[-1]
                           if result.attempts else None, "roi": roi_bounds,
                           "attempts": result.attempts, "decode_ms": self._last_decode_ms,
                           "frame": result.as_dict()}

    # -- reporting ---------------------------------------------------------

    def status(self):
        """Compact status for the UI poll loop."""
        tracker = self.tracker
        with self._lock:
            return {"running": self.running, "error": self._error,
                    "camera_index": self.index, "barcode": self._last_barcode,
                    "event_id": self._event_id, "state": tracker.state.value,
                    "message": tracker.message, "product": self._last_product,
                    "found": self._last_found, "scan": self._last_event}

    def debug_status(self):
        """Everything a person debugging a bad read could want."""
        with self._lock:
            debug = dict(self._debug)
            error = self._error
        return {"running": self.running, "error": error, "camera_index": self.index,
                "camera": self._device.profile.as_dict(),
                "capabilities": self._device.capabilities,
                "resolution": [self._device.profile.width, self._device.profile.height],
                "fps": round(self._fps, 1), "scan_fps": round(self._scan_fps, 1),
                "last_decode_ms": self._last_decode_ms,
                "scan_every_n_frames": self.scan_every_n_frames,
                "reconnects": self._device.reconnects,
                "metrics": self.metrics.snapshot(), "tracker": self.tracker.status(), **debug}

    def mjpeg_frames(self):
        """Multipart MJPEG generator for the live preview."""
        interval = 1.0 / max(1, self.config.preview_fps)
        idle_deadline = time.monotonic() + 5
        while not self._stop.is_set():
            if not self.running and time.monotonic() > idle_deadline:
                # The camera was stopped or never started: end the response
                # instead of holding the connection open forever.
                return
            with self._lock:
                jpeg = self._jpeg
            if jpeg:
                idle_deadline = time.monotonic() + 5
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(interval)
