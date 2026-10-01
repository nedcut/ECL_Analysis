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
