"""The live preview and the analysis worker must agree on what they measure."""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

from ecl_analysis.analysis.models import MASK_STATUS_APPLIED, AnalysisRequest, AnalysisResult
from ecl_analysis.video_analyzer import VideoAnalyzer
from ecl_analysis.workers import AnalysisWorker


class _FrameCapture:
    """Minimal cv2.VideoCapture stand-in that replays in-memory frames."""

    def __init__(self, frames: List[np.ndarray]):
        self._frames = frames
        self._index = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._index = int(value)

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._index)
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def read(self):
        if 0 <= self._index < len(self._frames):
            frame = self._frames[self._index].copy()
            self._index += 1
            return True, frame
        return False, None

    def release(self):
        pass


def _graded_frame() -> np.ndarray:
    """40x40 frame: dark left half, bright gradient on the right reaching the edge."""
    frame = np.full((40, 40, 3), 15, dtype=np.uint8)
    for col in range(20, 40):
        frame[:, col, :] = 120 + 6 * (col - 20)
    frame[30:, :10, :] = 90  # mid-grey block for a background ROI
    return frame


def _request_from_ui(window: VideoAnalyzer, monkeypatch: pytest.MonkeyPatch, tmp_path) -> AnalysisRequest:
    """Build the AnalysisRequest exactly as the UI does, without starting the thread."""
    window.video_path = "dummy.mp4"
    window.start_frame = 0
    window.end_frame = 0
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *args, **kwargs: str(tmp_path)),
    )
    monkeypatch.setattr(QtCore.QThread, "start", lambda self, *args, **kwargs: None)
    window.analyze_video()
    assert window._analysis_worker is not None
    request = window._analysis_worker._request
    if window._analysis_progress is not None:
        window._analysis_progress.close()
    return request


def _run_worker(request: AnalysisRequest, frame: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> AnalysisResult:
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: _FrameCapture([frame]))
    worker = AnalysisWorker(request)
    captured: Dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.error.connect(lambda message: captured.setdefault("error", message))
    worker.run()
    assert "error" not in captured, captured.get("error")
    return captured["result"]


def test_edge_touching_roi_mask_captured_from_current_is_applied_by_analysis(
    qt_application: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """An ROI dragged against the right/bottom frame edge has x2 == frame width.

    "Capture From Current" used to clamp to width - 1 and so produced a mask one
    column narrower than the worker's ROI; the worker then dropped it on a shape
    mismatch and silently ran unmasked while the UI said "Mask: active".
    """
    window = VideoAnalyzer()
    try:
        frame = _graded_frame()
        window.frame = frame
        window.rects = [((20, 0), (40, 40))]  # touches right and bottom edges
        window.background_roi_idx = None
        window.manual_threshold = 0.0
        window.frame_slider.setRange(0, 10)

        window._capture_fixed_masks(source_frame_idx=0)

        mask = window.fixed_roi_masks[0]
        assert mask is not None
        assert mask.shape == (40, 20)
        assert window.use_fixed_mask

        request = _request_from_ui(window, monkeypatch, tmp_path)
        result = _run_worker(request, frame, monkeypatch)

        assert result.mask_status == {0: MASK_STATUS_APPLIED}
        assert result.pixel_count_data[0][0] == 40 * 20
    finally:
        window.close()


def test_capture_from_current_and_per_roi_auto_capture_use_the_same_rule(
    qt_application: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both capture paths must build identical masks from the same frame and
    settings in manual-threshold mode (no background ROI)."""
    window = VideoAnalyzer()
    try:
        frame = _graded_frame()
        window.frame = frame
        window.rects = [((0, 0), (40, 40)), ((20, 0), (40, 40))]
        window.background_roi_idx = None
        window.manual_threshold = 70.0
        window.morphological_kernel_size = 3
        window.frame_slider.setRange(0, 10)

        window._capture_fixed_masks(source_frame_idx=0)
        from_current = list(window.fixed_roi_masks)

        # Build the per-ROI scan request exactly as the UI does.
        window.cap = object()
        window.video_path = "dummy.mp4"
        window.total_frames = 2
        window.start_frame = 0
        window.end_frame = 1
        started = {}
        monkeypatch.setattr(window, "_start_mask_worker", lambda worker, **kwargs: started.setdefault("worker", worker))
        window._auto_capture_per_roi_brightest_masks()
        worker = started["worker"]
        assert worker._request.manual_threshold == 70.0

        monkeypatch.setattr(cv2, "VideoCapture", lambda _path: _FrameCapture([frame, frame]))
        captured: Dict[str, object] = {}
        worker.finished.connect(lambda payload: captured.setdefault("result", payload))
        worker.run()
        auto = captured["result"].masks

        for current_mask, auto_mask in zip(from_current, auto):
            assert current_mask is not None and auto_mask is not None
            assert np.array_equal(current_mask, auto_mask)
        # The threshold actually gated pixels (dark half excluded).
        assert not from_current[0][:, :20].any()
    finally:
        window.cap = None
        window.close()


def _assert_readout_matches_worker(window: VideoAnalyzer, frame: np.ndarray, monkeypatch, tmp_path) -> None:
    window.frame = frame
    window._update_current_brightness_display()
    readout_text = window.brightness_display_label.text()
    analysis, _ = window._preview_frame_analysis()

    request = _request_from_ui(window, monkeypatch, tmp_path)
    result = _run_worker(request, frame, monkeypatch)

    assert [roi.roi_idx for roi in analysis.rois] == result.non_background_rois
    expected_threshold = analysis.threshold if analysis.threshold is not None else 0.0
    assert result.background_values_per_frame == [expected_threshold]
    for data_idx, roi in enumerate(analysis.rois):
        # Bit-identical: the readout and the worker run the same function.
        assert roi.mean == result.brightness_mean_data[data_idx][0]
        assert roi.median == result.brightness_median_data[data_idx][0]
        assert roi.blue_mean == result.blue_mean_data[data_idx][0]
        assert roi.blue_median == result.blue_median_data[data_idx][0]
        assert roi.pixel_count == result.pixel_count_data[data_idx][0]
        if roi.thresholded:
            assert f"{result.brightness_mean_data[data_idx][0]:.1f}, " in readout_text
        assert f"Blue: {result.blue_mean_data[data_idx][0]:.0f}" in readout_text


def test_readout_matches_analysis_in_manual_threshold_mode(
    qt_application: QtWidgets.QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window = VideoAnalyzer()
    try:
        window.rects = [((0, 0), (40, 40)), ((20, 0), (40, 40)), ((5, 5), (15, 15))]
        window.background_roi_idx = None
        window.manual_threshold = 60.0
        window.morphological_kernel_size = 3
        window.noise_floor_threshold = 70.0

        _assert_readout_matches_worker(window, _graded_frame(), monkeypatch, tmp_path)

        analysis, _ = window._preview_frame_analysis()
        assert analysis.threshold == 60.0
        # The manual threshold changes what is measured; the readout must show it
        # (it used to show raw L* only whenever no background ROI was set).
        window._update_current_brightness_display()
        assert "Thr-Sub" in window.brightness_display_label.text()
        assert analysis.rois[0].mean != analysis.rois[0].stats.l_raw_mean
    finally:
        window.close()


def test_readout_matches_analysis_in_background_roi_mode(
    qt_application: QtWidgets.QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window = VideoAnalyzer()
    try:
        window.rects = [((0, 0), (40, 40)), ((20, 0), (40, 40)), ((0, 30), (10, 40))]
        window.background_roi_idx = 2
        window.manual_threshold = 99.0  # ignored while a background ROI is set
        window.background_percentile = 80.0
        window.morphological_kernel_size = 3

        _assert_readout_matches_worker(window, _graded_frame(), monkeypatch, tmp_path)

        analysis, _ = window._preview_frame_analysis()
        assert analysis.threshold == pytest.approx(window._compute_background_brightness(window.frame))
    finally:
        window.close()


def test_readout_matches_analysis_in_fixed_mask_mode(
    qt_application: QtWidgets.QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    window = VideoAnalyzer()
    try:
        window.rects = [((0, 0), (40, 40)), ((20, 0), (40, 40))]
        window.background_roi_idx = None
        window.manual_threshold = 75.0
        window.morphological_kernel_size = 3
        window.frame = _graded_frame()
        window.frame_slider.setRange(0, 10)
        window._capture_fixed_masks(source_frame_idx=0)
        assert window.use_fixed_mask

        # Measure a different (dimmer) frame through the captured masks.
        dimmer = (_graded_frame().astype(np.int16) - 40).clip(0, 255).astype(np.uint8)
        _assert_readout_matches_worker(window, dimmer, monkeypatch, tmp_path)

        analysis, _ = window._preview_frame_analysis()
        assert [roi.mask_status for roi in analysis.rois] == [MASK_STATUS_APPLIED, MASK_STATUS_APPLIED]
    finally:
        window.close()


def test_background_percentile_computed_once_per_frame_change(
    qt_application: QtWidgets.QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ecl_analysis.analysis.frame as frame_module

    window = VideoAnalyzer()
    try:
        window.rects = [((0, 0), (40, 40)), ((0, 30), (10, 40))]
        window.background_roi_idx = 1
        window.show_pixel_mask = True

        calls = []
        real = frame_module.compute_background_brightness

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(frame_module, "compute_background_brightness", counting)

        for expected_calls, frame in enumerate([_graded_frame(), _graded_frame()], start=1):
            # What a frame change refreshes: overlay, readout, threshold display.
            window.frame = frame
            window.show_frame()
            window._update_current_brightness_display()
            window._update_threshold_display()
            assert len(calls) == expected_calls

        # A settings change recomputes.
        window.background_percentile = 50.0
        window._update_threshold_display()
        assert len(calls) == 3
    finally:
        window.close()


def _window_with_captured_masks() -> VideoAnalyzer:
    window = VideoAnalyzer()
    window.frame = _graded_frame()
    window.rects = [((0, 0), (40, 40)), ((20, 0), (40, 40)), ((0, 30), (10, 40))]
    window.background_roi_idx = None
    window.threshold_spin.blockSignals(True)
    window.threshold_spin.setValue(60.0)
    window.threshold_spin.blockSignals(False)
    window.manual_threshold = 60.0
    window.frame_slider.setRange(0, 10)
    window._capture_fixed_masks(source_frame_idx=0)
    assert any(isinstance(mask, np.ndarray) for mask in window.fixed_roi_masks)
    return window


def test_manual_threshold_change_invalidates_masks_and_refreshes_preview(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _window_with_captured_masks()
    try:
        window.use_fixed_mask = False  # threshold path, so the readout tracks the threshold
        window._update_current_brightness_display()
        before_text = window.brightness_display_label.text()

        window.threshold_spin.setValue(80.0)

        assert all(mask is None for mask in window.fixed_roi_masks)
        assert window.mask_status_label.text() == "Mask: cleared (manual threshold changed)"
        assert "Manual (80.00 L*)" in window.threshold_display_label.text()
        after_text = window.brightness_display_label.text()
        assert after_text != before_text
        assert "Manual threshold: L* 80.0" in after_text

        # Undo restores the threshold together with the masks captured under it.
        window.undo_last_action()
        assert window.manual_threshold == 60.0
        assert any(isinstance(mask, np.ndarray) for mask in window.fixed_roi_masks)
    finally:
        window.close()


def test_manual_threshold_change_keeps_masks_when_background_roi_rules(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _window_with_captured_masks()
    try:
        window.selected_rect_idx = 2
        window._set_background_roi()
        window._capture_fixed_masks(source_frame_idx=0)
        masks = list(window.fixed_roi_masks)

        window.threshold_spin.setValue(10.0)  # not the active rule

        assert all(a is b for a, b in zip(window.fixed_roi_masks, masks))
    finally:
        window.close()


def test_setting_background_roi_invalidates_masks_and_refreshes_preview(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _window_with_captured_masks()
    try:
        window._update_current_brightness_display()
        assert "Thr-Sub" in window.brightness_display_label.text()

        window.selected_rect_idx = 2
        window._set_background_roi()

        assert all(mask is None for mask in window.fixed_roi_masks)
        assert window.mask_status_label.text() == "Mask: cleared (background ROI changed)"
        readout = window.brightness_display_label.text()
        assert "BG-Sub" in readout
        assert "ROI 3:" not in readout  # the background ROI is no longer measured
        assert "Active Threshold: Background ROI 3" in window.threshold_display_label.text()

        # Re-selecting the same background ROI is not a rule change.
        window._capture_fixed_masks(source_frame_idx=0)
        window._set_background_roi()
        assert any(isinstance(mask, np.ndarray) for mask in window.fixed_roi_masks)
    finally:
        window.close()


def test_deleting_background_roi_refreshes_readout(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _window_with_captured_masks()
    try:
        window.selected_rect_idx = 2
        window._set_background_roi()
        assert "BG-Sub" in window.brightness_display_label.text()

        window.selected_rect_idx = 2
        window.delete_selected_rectangle()

        assert window.background_roi_idx is None
        readout = window.brightness_display_label.text()
        assert "BG-Sub" not in readout
        assert "Thr-Sub" in readout  # manual threshold (60) is the rule again
    finally:
        window.close()


class _CountedFrameCapture(_FrameCapture):
    """_FrameCapture that also reports a frame count, as load_video requires."""

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return super().get(prop)


def _load_frames(window: VideoAnalyzer, tmp_path, monkeypatch: pytest.MonkeyPatch, frames, name: str) -> None:
    video = tmp_path / name
    video.write_bytes(b"stub")
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: _CountedFrameCapture(frames))
    monkeypatch.setattr(window, "_add_recent_file", lambda path: None)
    window.video_path = str(video)
    window.load_video()


def test_reset_state_drops_cached_preview_analysis(
    qt_application: QtWidgets.QApplication,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = VideoAnalyzer()
    try:
        _load_frames(window, tmp_path, monkeypatch, [_graded_frame() for _ in range(3)], "clip.avi")
        window.rects = [((1, 1), (15, 15))]
        window._preview_frame_analysis()
        assert window._preview_analysis_cache is not None

        window._reset_state()

        # The closed video's frame and L* channel must not stay referenced.
        assert window._preview_analysis_cache is None
    finally:
        window.close()
