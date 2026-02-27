import numpy as np

import cv2
from PyQt5 import QtWidgets

from ecl_analysis.video_analyzer import VideoAnalyzer


class DummyCapture:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_FPS:
            return 24.0
        return 0.0

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame.copy()
        return False, None

    def release(self):
        pass


def test_load_video_success(tmp_path, qt_application: QtWidgets.QApplication, monkeypatch):
    frames = [np.full((16, 16, 3), fill_value=idx * 10, dtype=np.uint8) for idx in range(5)]
    dummy_video = tmp_path / "dummy.avi"
    dummy_video.write_bytes(b"stub")

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: DummyCapture(frames))

    window = VideoAnalyzer()
    window.video_path = str(dummy_video)

    # Prevent modal dialogs from blocking the test
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    window.load_video()

    assert window.total_frames == len(frames)
    assert window.playback_fps == 24.0
    assert window.frame is not None
    cached_frame = window.frame_cache.get(0)
    assert cached_frame is not None
    assert cached_frame.shape == frames[0].shape

    window.close()
