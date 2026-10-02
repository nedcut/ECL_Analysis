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


def test_load_with_existing_rois_skips_audio_detection_without_duration(
    qt_application: QtWidgets.QApplication,
    tmp_path,
    monkeypatch,
    no_modals,
) -> None:
    window = VideoAnalyzer()
    window.rects = [((1, 1), (8, 8))]
    monkeypatch.setattr(window.audio_analyzer, "is_available", lambda: True)
    window.run_duration_spin.setValue(0.0)

    _load_dummy_video(window, tmp_path, monkeypatch)

    assert window._audio_thread is None
    assert not window._analysis_in_progress
    assert "expected run duration" in window.statusBar().currentMessage()

    window.close()


def test_load_with_existing_rois_skips_audio_detection_without_librosa(
    qt_application: QtWidgets.QApplication,
    tmp_path,
    monkeypatch,
    no_modals,
) -> None:
    window = VideoAnalyzer()
    window.rects = [((1, 1), (8, 8))]
    monkeypatch.setattr(window.audio_analyzer, "is_available", lambda: False)
    window.run_duration_spin.setValue(10.0)

    _load_dummy_video(window, tmp_path, monkeypatch)

    assert window._audio_thread is None
    assert "librosa" in window.statusBar().currentMessage()

    window.close()


def test_user_initiated_audio_detection_still_warns_without_duration(
    qt_application: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    window = VideoAnalyzer()
    window.video_path = "video.mp4"
    monkeypatch.setattr(window.audio_analyzer, "is_available", lambda: True)
    window.run_duration_spin.setValue(0.0)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args[1:3]))

    window.auto_detect_range()

    assert len(warnings) == 1
    assert window._audio_thread is None

    window.close()


def _gradient_frames(offset: int, frame_count: int = 3):
    ramp = np.tile(np.arange(0, 240, 15, dtype=np.uint8), (16, 1))
    return [np.dstack([np.clip(ramp.astype(int) + offset, 0, 255).astype(np.uint8)] * 3) for _ in range(frame_count)]


def _load_frames(window: VideoAnalyzer, tmp_path, monkeypatch, frames, name: str) -> None:
    video = tmp_path / name
    video.write_bytes(b"stub")
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: _DummyCapture(frames))
    monkeypatch.setattr(window, "_add_recent_file", lambda path: None)
    monkeypatch.setattr(window, "_auto_detect_range_after_load", lambda: None)
    window.video_path = str(video)
    window.load_video()


def _assert_readout_is_current(window: VideoAnalyzer) -> None:
    shown = window.brightness_display_label.text()
    window._update_current_brightness_display()
    assert shown == window.brightness_display_label.text()


def test_load_video_with_rois_drops_old_masks_before_drawing(
    qt_application: QtWidgets.QApplication,
    tmp_path,
    monkeypatch,
) -> None:
    window = VideoAnalyzer()
    _load_frames(window, tmp_path, monkeypatch, _gradient_frames(0), "first.avi")
    window.rects = [((1, 1), (15, 15))]
    window.update_rect_list()
    window.manual_threshold = 50.0
    window._capture_fixed_masks(0)
    window.use_fixed_mask_checkbox.setChecked(True)
    window.show_pixel_mask = True
    window._update_current_brightness_display()

    overlay_mask_states = []
    original_overlay = window._apply_pixel_mask_overlay

    def _spy(frame):
        overlay_mask_states.append((window.use_fixed_mask, [m is not None for m in window.fixed_roi_masks]))
        return original_overlay(frame)

    monkeypatch.setattr(window, "_apply_pixel_mask_overlay", _spy)

    _load_frames(window, tmp_path, monkeypatch, _gradient_frames(40), "second.avi")

    # Every draw of the new video already sees the masks reset ...
    assert overlay_mask_states and all(state == (False, [False]) for state in overlay_mask_states)
    # ... and the readout shows the new video's values, not the old video's.
    _assert_readout_is_current(window)
    assert "ROI 1" in window.brightness_display_label.text()

    window.close()


def test_clear_all_rectangles_refreshes_readout(
    qt_application: QtWidgets.QApplication,
    tmp_path,
    monkeypatch,
) -> None:
    for use_fixed_mask in (False, True):
        window = VideoAnalyzer()
        _load_frames(window, tmp_path, monkeypatch, _gradient_frames(0), "clip.avi")
        window.rects = [((1, 1), (15, 15))]
        window.update_rect_list()
        if use_fixed_mask:
            window._capture_fixed_masks(0)
            window.use_fixed_mask_checkbox.setChecked(True)
        window._update_current_brightness_display()
        assert "ROI 1" in window.brightness_display_label.text()
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.Yes
        )

        window.clear_all_rectangles()

        assert window.brightness_display_label.text() == "N/A"
        window.close()
