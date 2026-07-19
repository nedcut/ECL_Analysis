from __future__ import annotations

import numpy as np
import pytest
from PyQt5 import QtWidgets

from ecl_analysis.analysis.background import BackgroundComputationError
from ecl_analysis.video_analyzer import VideoAnalyzer


def _half_bright_frame() -> np.ndarray:
    """Left half dark (L* ~ 0), right half bright (L* ~ 86)."""
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, 20:, :] = 220
    return frame


def _prepare_window(window: VideoAnalyzer) -> VideoAnalyzer:
    window.frame = _half_bright_frame()
    window.rects = [((0, 0), (40, 40))]
    window.background_roi_idx = None
    window.use_fixed_mask = False
    window.fixed_roi_masks = []
    return window


def test_effective_threshold_uses_manual_when_no_background_roi(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _prepare_window(VideoAnalyzer())

    window.manual_threshold = 50.0
    assert window._effective_analysis_threshold(window.frame) == pytest.approx(50.0)

    window.manual_threshold = 0.0
    assert window._effective_analysis_threshold(window.frame) is None


def test_effective_threshold_prefers_background_roi(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _prepare_window(VideoAnalyzer())
    window.rects = [((0, 0), (40, 40)), ((0, 0), (20, 40))]
    window.background_roi_idx = 1
    window.manual_threshold = 99.0

    expected = window._compute_background_brightness(window.frame)
    assert expected is not None
    assert window._effective_analysis_threshold(window.frame) == pytest.approx(expected)


def test_pixel_mask_overlay_honors_manual_threshold(
    qt_application: QtWidgets.QApplication,
) -> None:
    """With a manual threshold and no background ROI, only pixels above the
    threshold may be tinted - the dark half must remain untouched."""
    window = _prepare_window(VideoAnalyzer())
    window.manual_threshold = 50.0

    overlay = window._apply_pixel_mask_overlay(window.frame)

    assert np.array_equal(overlay[:, :20], window.frame[:, :20])
    assert not np.array_equal(overlay[:, 20:], window.frame[:, 20:])


def test_pixel_mask_overlay_tints_everything_when_threshold_disabled(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _prepare_window(VideoAnalyzer())
    window.manual_threshold = 0.0

    overlay = window._apply_pixel_mask_overlay(window.frame)

    assert not np.array_equal(overlay[:, 20:], window.frame[:, 20:])
    # Dark pixels gain a red tint too (mask covers the full ROI).
    assert overlay[:, :20, 2].max() > 0


def test_pixel_mask_overlay_survives_background_failure(
    qt_application: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _prepare_window(VideoAnalyzer())
    window.rects = [((0, 0), (40, 40)), ((0, 0), (20, 40))]
    window.background_roi_idx = 1

    def boom(*args, **kwargs):
        raise BackgroundComputationError("synthetic failure")

    monkeypatch.setattr(window, "_compute_background_brightness", boom)

    overlay = window._apply_pixel_mask_overlay(window.frame)

    assert np.array_equal(overlay, window.frame)


def test_brightness_display_reports_error_instead_of_crashing(
    qt_application: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _prepare_window(VideoAnalyzer())
    window.rects = [((0, 0), (40, 40)), ((0, 0), (20, 40))]
    window.background_roi_idx = 1

    def boom(*args, **kwargs):
        raise BackgroundComputationError("synthetic failure")

    monkeypatch.setattr(window, "_compute_background_brightness", boom)

    window._update_current_brightness_display()

    assert "error" in window.brightness_display_label.text().lower()


def test_threshold_display_survives_background_failure(
    qt_application: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _prepare_window(VideoAnalyzer())
    window.rects = [((0, 0), (40, 40)), ((0, 0), (20, 40))]
    window.background_roi_idx = 1

    def boom(*args, **kwargs):
        raise BackgroundComputationError("synthetic failure")

    monkeypatch.setattr(window, "_compute_background_brightness", boom)

    assert window._calculate_background_threshold() is None
    window._update_threshold_display()
    assert "Background ROI 2" in window.threshold_display_label.text()


def test_capture_fixed_masks_honors_manual_threshold(
    qt_application: QtWidgets.QApplication,
) -> None:
    """Captured masks must gate by the manual threshold in manual mode, matching
    what the analysis worker applies, instead of covering the whole ROI."""
    window = _prepare_window(VideoAnalyzer())
    window.manual_threshold = 50.0
    window.frame_slider.setRange(0, 10)

    window._capture_fixed_masks(source_frame_idx=0)

    mask = window.fixed_roi_masks[0]
    assert mask is not None
    assert not mask[:, :20].any()
    assert mask[:, 20:].all()
