from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

from ecl_analysis.video_analyzer import VideoAnalyzer


class _StubCapture:
    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        return None


def _prepare_loaded_window(window: VideoAnalyzer, total_frames: int = 100) -> None:
    window.cap = _StubCapture()
    window.frame = np.zeros((120, 200, 3), dtype=np.uint8)
    window.total_frames = total_frames
    window.playback_fps = 25.0
    window.start_frame = 0
    window.end_frame = total_frames - 1
    window.current_frame_index = 0
    window.frame_slider.setRange(0, total_frames - 1)
    window.frame_spinbox.setRange(0, total_frames - 1)
    window._seek_to_frame = lambda frame_index: setattr(window, "current_frame_index", frame_index)
    window._sync_analysis_range_widgets()


def test_analysis_range_changes_support_undo_and_redo(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    _prepare_loaded_window(window, total_frames=120)

    window.current_frame_index = 24
    window.set_start_frame()

    assert window.start_frame == 24
    assert window.range_start_spinbox.value() == 25
    assert "Frames 25-120" in window.analysis_range_summary_label.text()

    window.undo_last_action()
    assert window.start_frame == 0
    assert window.range_start_spinbox.value() == 1

    window.redo_last_action()
    assert window.start_frame == 24
    assert window.range_start_spinbox.value() == 25

    window.close()


def test_roi_addition_supports_undo_and_redo(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    _prepare_loaded_window(window)

    initial_rects = list(window.rects)
    window.roi_width_spin.setValue(60)
    window.roi_height_spin.setValue(40)
    window.add_roi_by_size()

    assert len(window.rects) == len(initial_rects) + 1

    window.undo_last_action()
    assert window.rects == initial_rects

    window.redo_last_action()
    assert len(window.rects) == len(initial_rects) + 1
    assert window.selected_rect_idx == 0

    window.close()
