import time

import pytest
import cv2
import numpy as np
import zxingcpp
from app import create_app
from services.barcode_scanner import (
    BarcodeDebouncer, BarcodePresenceTracker, BarcodeStabilizer,
    is_valid_barcode, scan_barcode, scan_barcode_details,
)
from services import camera as camera_module
from services.camera import CameraService


@pytest.fixture()
def client(tmp_path):
    app = create_app({"TESTING": True, "DATABASE": str(tmp_path / "test.db"), "UPLOAD_FOLDER": str(tmp_path / "uploads")})
    return app.test_client()


def product(barcode="8691234567890"):
    return {"barcode": barcode, "name": "Coca Cola 1 L", "brand": "Coca Cola", "category": "İçecek", "price": 49.90, "weight": "1050 g"}


def test_product_lifecycle_and_scan_lookup(client):
    response = client.post("/api/products", json=product())
    assert response.status_code == 201
    assert response.json["barcode"] == "8691234567890"
    assert client.post("/api/products", json=product()).status_code == 409
    found = client.post("/api/scan", json={"barcode": "8691234567890"})
    assert found.json["found"] is True
    assert found.json["product"]["name"] == "Coca Cola 1 L"
    missing = client.post("/api/scan", json={"barcode": "999"})
    assert missing.json == {"found": False, "barcode": "999", "product": None}
    updated = client.put("/api/products/8691234567890", json={**product(), "name": "Coca Cola 1.5 L"})
    assert updated.status_code == 200
    assert updated.json["name"] == "Coca Cola 1.5 L"


def test_product_listing_search(client):
    client.post("/api/products", json=product())
    assert len(client.get("/api/products?q=Coca").json) == 1
    assert client.get("/api/products?q=ayran").json == []


def test_debouncer_allows_other_codes_immediately():
    d = BarcodeDebouncer(cooldown_seconds=1.5)
    assert d.accept("8691234567890", now=10)
    assert not d.accept("8691234567890", now=10.2)
    assert d.accept("12345678", now=10.3)
    assert d.accept("8691234567890", now=12)


def test_ean13_checksum_validation():
    assert is_valid_barcode("8691234567890", "EAN-13")
    assert is_valid_barcode("8690530046269", "EAN-13")
    assert not is_valid_barcode("8691234567891", "EAN-13")
    assert not is_valid_barcode("869123456789", "EAN-13")


def test_distant_ean13_in_a_720p_frame_is_decoded():
    encoded = np.asarray(zxingcpp.write_barcode_to_image(
        zxingcpp.create_barcode("8690530046269", zxingcpp.BarcodeFormat.EAN13), scale=2))
    canvas = np.full((720, 1280), 255, dtype=np.uint8)
    height, width = encoded.shape
    canvas[260:260 + height, 400:400 + width] = encoded
    frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    roi = frame[72:648, 89:1190]
    value, diagnostics = scan_barcode_details(frame, roi)
    assert value == "8690530046269"
    assert diagnostics["source_size"] == [1101, 576]
    assert diagnostics["method"] is not None


def test_stabilizer_requires_three_matching_detections():
    stabilizer = BarcodeStabilizer(required_hits=3)
    assert stabilizer.observe("8690530046269") is None
    assert stabilizer.observe("8690530046269") is None
    assert stabilizer.observe("8690530046269") == "8690530046269"
    assert stabilizer.observe("8695188000014") is None


def test_visible_guide_roi_matches_16_by_9_and_4_by_3_preview_frames():
    assert CameraService._visible_guide_roi(np.zeros((720, 1280, 3), dtype=np.uint8)) == (89, 72, 1190, 648)
    # With object-fit: contain, the 4:3 image has horizontal bars in the 16:9 UI.
    assert CameraService._visible_guide_roi(np.zeros((480, 640, 3), dtype=np.uint8)) == (0, 48, 640, 432)


# --- BarcodePresenceTracker (replaces the stabilizer+debouncer combo in the
# live CameraService; see services/barcode_scanner.py for the rationale) ---

def test_presence_tracker_emits_on_the_very_first_read():
    tracker = BarcodePresenceTracker()
    assert tracker.observe("8690530046269") == "8690530046269"


def test_presence_tracker_suppresses_repeats_while_continuously_visible():
    tracker = BarcodePresenceTracker()
    assert tracker.observe("8690530046269") == "8690530046269"
    for _ in range(10):
        assert tracker.observe("8690530046269") is None


def test_presence_tracker_switches_between_codes_immediately():
    tracker = BarcodePresenceTracker()
    assert tracker.observe("8690530046269") == "8690530046269"
    assert tracker.observe("8690530046269") is None
    assert tracker.observe("8695188000014") == "8695188000014"  # different product -> instant event


def test_presence_tracker_re_emits_after_leaving_and_returning_to_frame():
    tracker = BarcodePresenceTracker(miss_tolerance=3)
    assert tracker.observe("8690530046269") == "8690530046269"
    assert tracker.observe(None) is None  # product briefly lifted / motion blur
    assert tracker.observe(None) is None
    assert tracker.observe(None) is None  # 3rd consecutive miss clears "active"
    assert tracker.observe("8690530046269") == "8690530046269"  # same code, shown again -> new event


# --- End-to-end camera pipeline, using a fake VideoCapture so this runs
# without real hardware. Verifies the capture/scan threads are actually
# decoupled and that a checksum-verified barcode produces exactly one event
# while it stays in view (no repeat events, no camera hardware required). ---

class _FakeCapture:
    """Stands in for cv2.VideoCapture, replaying a fixed list of frames."""
    def __init__(self, frames, frame_delay=0.01):
        self._frames = frames
        self._i = 0
        self._delay = frame_delay

    def isOpened(self):
        return True

    def set(self, *_args, **_kwargs):
        return True

    def get(self, prop_id):
        return 1280 if prop_id == cv2.CAP_PROP_FRAME_WIDTH else 720

    def read(self):
        time.sleep(self._delay)
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        return True, frame.copy()

    def release(self):
        pass


def _frame_with_barcode(value="8690530046269", scale=3):
    encoded = np.asarray(zxingcpp.write_barcode_to_image(
        zxingcpp.create_barcode(value, zxingcpp.BarcodeFormat.EAN13), scale=scale))
    canvas = np.full((720, 1280), 255, dtype=np.uint8)
    h, w = encoded.shape
    canvas[260:260 + h, 300:300 + w] = encoded
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _blank_frame():
    return np.full((720, 1280, 3), 255, dtype=np.uint8)


def test_camera_service_capture_and_scan_run_on_separate_threads(monkeypatch):
    monkeypatch.setattr(camera_module.cv2, "VideoCapture",
                         lambda index, backend=None: _FakeCapture([_frame_with_barcode()]))
    service = CameraService(index=0)
    try:
        assert service.start()
        assert service._capture_thread is not service._scan_thread
        assert service._capture_thread.is_alive()
        assert service._scan_thread.is_alive()
    finally:
        service.stop()


def test_camera_service_emits_one_event_while_barcode_stays_in_view(monkeypatch):
    monkeypatch.setattr(camera_module.cv2, "VideoCapture",
                         lambda index, backend=None: _FakeCapture([_frame_with_barcode()], frame_delay=0.005))
    service = CameraService(index=0)
    try:
        assert service.start()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and service.status()["event_id"] == 0:
            time.sleep(0.02)
        status = service.status()
        assert status["barcode"] == "8690530046269"
        assert status["event_id"] == 1
        # The same barcode keeps being decoded on every subsequent frame; it
        # must NOT keep re-emitting while it's continuously in view.
        time.sleep(0.3)
        assert service.status()["event_id"] == 1
    finally:
        service.stop()


def test_camera_service_re_emits_after_barcode_leaves_and_returns(monkeypatch):
    frames = [_frame_with_barcode()] * 6 + [_blank_frame()] * 6 + [_frame_with_barcode()] * 20
    monkeypatch.setattr(camera_module.cv2, "VideoCapture",
                         lambda index, backend=None: _FakeCapture(frames, frame_delay=0.005))
    service = CameraService(index=0)
    try:
        assert service.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and service.status()["event_id"] < 2:
            time.sleep(0.02)
        status = service.status()
        assert status["event_id"] == 2  # left frame, came back -> a 2nd event
        assert status["barcode"] == "8690530046269"
    finally:
        service.stop()


# --- Camera index management (Stage 1) ------------------------------------

def test_open_failed_message_includes_index():
    from services.camera_device import open_failed_message, read_failed_message, invalid_index_message
    assert open_failed_message(2).startswith("Camera 2 could not be opened.")
    assert read_failed_message(1).startswith("Camera 1:")
    assert "Invalid camera index -1" in invalid_index_message(-1)


def test_camera_device_rejects_negative_index():
    from services.camera_device import CameraDevice, invalid_index_message
    device = CameraDevice(index=-1)
    assert not device.open()
    assert device.error == invalid_index_message(-1)


def test_camera_service_reports_index_in_status():
    service = CameraService(index=2)
    try:
        status = service.status()
        assert status["camera_index"] == 2
        debug = service.debug_status()
        assert debug["camera_index"] == 2
    finally:
        service.stop()


def test_camera_service_open_failure_includes_index(monkeypatch):
    from services.camera_device import open_failed_message

    class _ClosedCapture:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr("services.camera_device.cv2.VideoCapture",
                        lambda index, backend=None: _ClosedCapture())
    service = CameraService(index=2)
    try:
        assert not service.start()
        assert service.error == open_failed_message(2)
    finally:
        service.stop()


def test_scanner_config_rejects_negative_camera_index():
    from services.config import ScannerConfig
    import pytest
    with pytest.raises(ValueError, match="camera_index must be >= 0"):
        ScannerConfig.from_env({"SMARTCART_CAMERA_INDEX": "-1"})
