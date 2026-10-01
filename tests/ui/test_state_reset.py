from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt5 import QtWidgets

from ecl_analysis.video_analyzer import VideoAnalyzer


class _DummyCapture:
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

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.index = int(value)
        return True

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame.copy()
        return False, None

    def release(self):
        pass


@pytest.fixture
def no_modals(monkeypatch):
    """Fail the test if any modal message box would be shown."""

    def _fail(*args, **kwargs):
        pytest.fail(f"unexpected modal dialog: {args[1:3]}")

    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, _fail)


def _load_dummy_video(window: VideoAnalyzer, tmp_path, monkeypatch, frame_count: int = 5) -> None:
    frames = [np.full((16, 16, 3), fill_value=idx * 10, dtype=np.uint8) for idx in range(frame_count)]
    video = tmp_path / "dummy.avi"
    video.write_bytes(b"stub")
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: _DummyCapture(frames))
    monkeypatch.setattr(window, "_add_recent_file", lambda path: None)
    window.video_path = str(video)
    window.load_video()


def test_reset_state_clears_all_roi_dependent_state(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    window.rects = [((0, 0), (10, 10)), ((20, 20), (30, 30))]
    window.selected_rect_idx = 1
    window.background_roi_idx = 1
    window.fixed_roi_masks = [np.ones((10, 10), dtype=bool), None]
    window.mask_source_frames = [4, None]
    window._set_use_fixed_mask_silently(True)

    window._reset_state()

    assert window.rects == []
    assert window.selected_rect_idx is None
    assert window.background_roi_idx is None
    assert window.fixed_roi_masks == []
    assert window.mask_source_frames == []
    assert window.use_fixed_mask is False
    assert not window.use_fixed_mask_checkbox.isChecked()
    assert window.mask_status_label.text() == "Mask: none"
    assert window._undo_history == []

    window.close()


def test_load_video_leaves_no_stray_history_entries(
    qt_application: QtWidgets.QApplication,
    tmp_path,
    monkeypatch,
    no_modals,
) -> None:
    window = VideoAnalyzer()
    window._set_use_fixed_mask_silently(True)

    _load_dummy_video(window, tmp_path, monkeypatch)

    assert window.total_frames == 5
    assert window.use_fixed_mask is False
    assert not window.use_fixed_mask_checkbox.isChecked()
    assert window._undo_history == []
    assert not window.undo_action.isEnabled()

    window.close()
