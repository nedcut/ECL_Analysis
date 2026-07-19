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
    window.frame[0:60, 0:60, :] = np.linspace(0, 255, 60, dtype=np.uint8).reshape(1, 60, 1)
    window.total_frames = total_frames
    window.playback_fps = 25.0
    window.start_frame = 0
    window.end_frame = total_frames - 1
    window.current_frame_index = 0
    window.frame_slider.setRange(0, total_frames - 1)
    window.frame_spinbox.setRange(0, total_frames - 1)
    window._seek_to_frame = lambda frame_index: setattr(window, "current_frame_index", frame_index)
    window._sync_analysis_range_widgets()


def test_display_threshold_matches_analysis_threshold(
    qt_application: QtWidgets.QApplication,
) -> None:
    """The 'Active Threshold' display must equal the percentile-based value
    used by the analysis worker for the same inputs, not a mean-based value."""
    window = VideoAnalyzer()
    _prepare_loaded_window(window)

    window.rects = [((0, 0), (60, 60))]
    window.background_roi_idx = 0
    window.background_percentile = 75.0

    displayed = window._calculate_background_threshold()
    analysis_value = window._compute_background_brightness(window.frame)

    assert displayed is not None
    assert displayed == analysis_value

    window.close()


def test_changing_background_percentile_invalidates_fixed_masks(
    qt_application: QtWidgets.QApplication,
) -> None:
    """Changing the background percentile invalidates previously captured
    fixed masks so stale masks aren't used with a new threshold."""
    window = VideoAnalyzer()
    _prepare_loaded_window(window)

    window.rects = [((0, 0), (60, 60)), ((60, 60), (120, 120))]
    window.background_roi_idx = 0
    window.background_percentile = 75.0

    window._capture_fixed_masks(source_frame_idx=0)
    assert any(mask is not None for mask in window.fixed_roi_masks)

    new_value = int(window.background_percentile) + 5
    window._on_bg_percentile_changed(new_value)

    assert all(mask is None for mask in window.fixed_roi_masks)
    assert "cleared" in window.mask_status_label.text().lower()

    window.close()
