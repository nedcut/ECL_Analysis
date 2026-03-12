from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

from ecl_analysis.analysis.models import MaskCaptureMetadata
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


def test_background_percentile_change_invalidates_fixed_masks(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    _prepare_loaded_window(window)

    window.rects = [((0, 0), (20, 20))]
    window.fixed_roi_masks = [np.ones((20, 20), dtype=bool)]
    window.mask_source_frames = [5]
    window.fixed_mask_metadata = [
        MaskCaptureMetadata(
            capture_mode="manual",
            source_frames=[5],
            primary_source_frame=5,
            pixel_count=400,
            confidence_label="high",
            noise_floor_threshold=5.0,
            morphological_kernel_size=3,
        )
    ]

    window._on_bg_percentile_changed(95)

    assert window.fixed_roi_masks == [None]
    assert window.mask_source_frames == [None]
    assert window.fixed_mask_metadata == [None]
    assert "cleared" in window.mask_status_label.text().lower()

    window.close()
