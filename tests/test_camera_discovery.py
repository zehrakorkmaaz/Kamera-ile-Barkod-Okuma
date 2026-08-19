import cv2
import numpy as np

from services.camera_device import CameraDevice
from services.camera_discovery import discover_cameras, probe_index, render_discovery_list


class _FakeCapture:
    def __init__(self, index, opens=True, frame=None):
        self.index = index
        self._opens = opens
        self._frame = frame

    def isOpened(self):
        return self._opens

    def read(self):
        if self._frame is None:
            return False, None
        return True, self._frame.copy()

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        if prop_id == cv2.CAP_PROP_FOURCC:
            return int.from_bytes(b"MJPG", "little")
        return -1

    def getBackendName(self):
        return "MOCK"

    def release(self):
        pass


def test_probe_index_reports_available_camera(monkeypatch):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr("services.camera_discovery.cv2.VideoCapture",
                        lambda index, backend=None: _FakeCapture(index, frame=frame))
    camera = probe_index(1)
    assert camera.available is True
    assert camera.index == 1
    assert camera.width == 1280
    assert camera.height == 720
    assert camera.backend == "MOCK"
    assert camera.name is None


def test_probe_index_reports_unavailable_camera(monkeypatch):
    monkeypatch.setattr("services.camera_discovery.cv2.VideoCapture",
                        lambda index, backend=None: _FakeCapture(index, opens=False))
    camera = probe_index(2)
    assert camera.available is False
    assert camera.index == 2
    assert "Camera 2 could not be opened" in (camera.error or "")


def test_discover_cameras_lists_only_available_indices(monkeypatch):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def factory(index, backend=None):
        if index == 1:
            return _FakeCapture(index, frame=frame)
        return _FakeCapture(index, opens=False)

    monkeypatch.setattr("services.camera_discovery.cv2.VideoCapture", factory)
    cameras = discover_cameras(max_index=5, stop_after_misses=2)
    assert [camera.index for camera in cameras] == [1]


def test_discover_cameras_stops_after_consecutive_misses(monkeypatch):
    calls = {"count": 0}

    def factory(index, backend=None):
        calls["count"] += 1
        return _FakeCapture(index, opens=False)

    monkeypatch.setattr("services.camera_discovery.cv2.VideoCapture", factory)
    discover_cameras(max_index=9, stop_after_misses=3)
    assert calls["count"] == 3


def test_fourcc_ignores_unset_avfoundation_value():
    class _Capture:
        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FOURCC:
                return -1
            return 0

    assert CameraDevice._fourcc(_Capture()) == ""


def test_render_discovery_list_includes_index_and_technical_details():
    from services.camera_discovery import DiscoveredCamera
    text = render_discovery_list([
        DiscoveredCamera(index=0, available=True, backend="AVFOUNDATION",
                         width=1920, height=1080, fps=30.0, fourcc="MJPG"),
    ])
    assert "Available Cameras" in text
    assert "[0]" in text
    assert "AVFOUNDATION" in text
    assert "1920x1080" in text
    assert "OpenCV bu platformda" in text
