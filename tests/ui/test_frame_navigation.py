from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

from ecl_analysis.video_analyzer import VideoAnalyzer


class _UnreadableCapture:
    """Capture that is open but fails every read (e.g. a corrupt frame)."""

    def isOpened(self) -> bool:
        return True

    def set(self, prop, value) -> bool:
        return True

    def read(self):
        return False, None

    def release(self) -> None:
        return None


def _prepare_window(total_frames: int = 100) -> VideoAnalyzer:
    window = VideoAnalyzer()
    window.cap = _UnreadableCapture()
    window.frame = np.zeros((120, 200, 3), dtype=np.uint8)
    window.total_frames = total_frames
    window.playback_fps = 25.0
    window.start_frame = 0
    window.end_frame = total_frames - 1
    window.current_frame_index = 0
    window.frame_slider.setRange(0, total_frames - 1)
    window.frame_spinbox.setRange(1, total_frames)
    window._sync_analysis_range_widgets()
    return window


def test_go_to_box_uses_one_based_frame_numbers(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _prepare_window()
    window._seek_to_frame = lambda frame_index: setattr(window, "current_frame_index", frame_index)

    window.frame_spinbox.setValue(10)
    assert window.current_frame_index == 9
    assert window.frame_slider.value() == 9

    window.frame_slider.setValue(41)
    assert window.current_frame_index == 41
    assert window.frame_spinbox.value() == 42

    window.close()


def test_failed_seek_resyncs_navigation_widgets(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _prepare_window()

    window.frame_slider.setValue(50)

    assert window.current_frame_index == 0
    assert window.frame_slider.value() == 0
    assert window.frame_spinbox.value() == 1
    assert "Could not read frame 51" in window.statusBar().currentMessage()

    window.close()


def test_failed_seek_from_go_to_box_resyncs_widgets(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _prepare_window()

    window.frame_spinbox.setValue(30)

    assert window.current_frame_index == 0
    assert window.frame_slider.value() == 0
    assert window.frame_spinbox.value() == 1

    window.close()


def test_mask_status_reports_one_based_source_frame(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _prepare_window()
    window.rects = [((10, 10), (60, 60))]

    window._capture_fixed_masks(source_frame_idx=4)

    assert window.mask_status_label.text() == "Mask: captured from frame 5"

    window.close()
