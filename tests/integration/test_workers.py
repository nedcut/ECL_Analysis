from typing import Dict, List

import cv2
import numpy as np
import pytest

import ecl_analysis.analysis.frame as frame_module
from ecl_analysis.analysis.background import BackgroundComputationError
from ecl_analysis.analysis.brightness import compute_l_star_frame
from ecl_analysis.analysis.models import AnalysisRequest
from ecl_analysis.workers import (
    AnalysisWorker,
    BrightestFrameResult,
    BrightestFrameWorker,
    MaskScanRequest,
    PerRoiMaskCaptureResult,
    PerRoiMaskCaptureWorker,
)


class DummyVideoCapture:
    def __init__(self, frames: List[np.ndarray], fps: float = 30.0, seek_offset: int = 0):
        self._frames = frames
        self._index = 0
        self._fps = fps
        self._seek_offset = seek_offset

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._index = int(value) + self._seek_offset

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._index)
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0

    def read(self):
        if 0 <= self._index < len(self._frames):
            frame = self._frames[self._index].copy()
            self._index += 1
            return True, frame
        return False, None

    def release(self):
        pass


def test_analysis_worker_emits_structured_result(monkeypatch):
    frames = [
        np.full((6, 6, 3), 20, dtype=np.uint8),
        np.full((6, 6, 3), 40, dtype=np.uint8),
        np.full((6, 6, 3), 80, dtype=np.uint8),
    ]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: DummyVideoCapture(frames))

    request = AnalysisRequest(
        video_path="dummy.mp4",
        rects=[((0, 0), (6, 6))],
        background_roi_idx=None,
        start_frame=0,
        end_frame=2,
        use_fixed_mask=False,
        fixed_roi_masks=[],
        background_percentile=90.0,
        morphological_kernel_size=3,
        noise_floor_threshold=0.0,
    )

    worker = AnalysisWorker(request)
    captured: Dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.error.connect(lambda message: captured.setdefault("error", message))
    worker.run()

    assert "error" not in captured
    result = captured.get("result")
    assert result is not None
    assert result.frames_processed == 3
    assert result.truncated is False
    assert len(result.brightness_mean_data) == 1
    assert len(result.brightness_mean_data[0]) == 3


def _run_analysis_worker(frames, monkeypatch, capture_kwargs=None, **overrides):
    capture_kwargs = capture_kwargs or {}
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: DummyVideoCapture(frames, **capture_kwargs))
    request = AnalysisRequest(
        video_path="dummy.mp4",
        rects=overrides.pop("rects", [((0, 0), (6, 6))]),
        background_roi_idx=overrides.pop("background_roi_idx", None),
        start_frame=overrides.pop("start_frame", 0),
        end_frame=overrides.pop("end_frame", len(frames) - 1),
        use_fixed_mask=overrides.pop("use_fixed_mask", False),
        fixed_roi_masks=overrides.pop("fixed_roi_masks", []),
        background_percentile=90.0,
        morphological_kernel_size=3,
        noise_floor_threshold=0.0,
        **overrides,
    )
    worker = AnalysisWorker(request)
    captured: Dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.error.connect(lambda message: captured.setdefault("error", message))
    worker.run()
    assert "error" not in captured
    return captured["result"]


def _half_lit_frame() -> np.ndarray:
    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    frame[:, :3, :] = 10  # dark half, L* ~ 3
    frame[:, 3:, :] = 200  # bright half, L* ~ 81
    return frame


def test_analysis_worker_records_pixel_counts(monkeypatch):
    frames = [_half_lit_frame()]

    res_raw = _run_analysis_worker(frames, monkeypatch)
    assert res_raw.pixel_count_data == [[36]]

    res_thresh = _run_analysis_worker(frames, monkeypatch, manual_threshold=50.0)
    assert res_thresh.pixel_count_data == [[18]]


def test_analysis_worker_records_fps_and_request(monkeypatch):
    res = _run_analysis_worker([_half_lit_frame()], monkeypatch, capture_kwargs={"fps": 25.0})
    assert res.fps == pytest.approx(25.0)
    assert res.request is not None
    assert res.request.threshold_mode == "none"
    assert res.seek_warning is None


def test_analysis_worker_fps_none_when_unavailable(monkeypatch):
    res = _run_analysis_worker([_half_lit_frame()], monkeypatch, capture_kwargs={"fps": 0.0})
    assert res.fps is None


def test_analysis_worker_records_seek_mismatch(monkeypatch, caplog):
    frames = [_half_lit_frame() for _ in range(5)]
    with caplog.at_level("WARNING"):
        res = _run_analysis_worker(
            frames,
            monkeypatch,
            capture_kwargs={"seek_offset": 1},
            start_frame=2,
            end_frame=3,
        )
    assert res.seek_warning is not None
    assert "index 2" in res.seek_warning
    assert any("Seek to frame" in rec.message for rec in caplog.records)


def test_analysis_worker_records_fixed_mask_status(monkeypatch, caplog):
    frames = [_half_lit_frame()]
    rects = [((0, 0), (6, 6)), ((0, 0), (3, 3)), ((3, 3), (6, 6))]
    masks = [
        np.ones((6, 6), dtype=bool),  # matches ROI 1
        np.ones((5, 5), dtype=bool),  # wrong shape for ROI 2
        None,  # no mask for ROI 3
    ]
    with caplog.at_level("WARNING"):
        res = _run_analysis_worker(
            frames,
            monkeypatch,
            rects=rects,
            use_fixed_mask=True,
            fixed_roi_masks=masks,
        )

    assert res.mask_status == {0: "applied", 1: "dropped_shape_mismatch", 2: "missing"}
    assert any("mask dropped" in rec.message for rec in caplog.records)

    res_off = _run_analysis_worker(frames, monkeypatch, rects=rects, fixed_roi_masks=masks)
    assert res_off.mask_status == {0: "not_requested", 1: "not_requested", 2: "not_requested"}


def test_analysis_worker_flags_truncation_on_early_eof(monkeypatch):
    frames = [
        np.full((6, 6, 3), 20, dtype=np.uint8),
        np.full((6, 6, 3), 40, dtype=np.uint8),
    ]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: DummyVideoCapture(frames))

    request = AnalysisRequest(
        video_path="dummy.mp4",
        rects=[((0, 0), (6, 6))],
        background_roi_idx=None,
        start_frame=0,
        end_frame=4,
        use_fixed_mask=False,
        fixed_roi_masks=[],
        background_percentile=90.0,
        morphological_kernel_size=3,
        noise_floor_threshold=0.0,
    )

    worker = AnalysisWorker(request)
    captured: Dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.error.connect(lambda message: captured.setdefault("error", message))
    worker.run()

    assert "error" not in captured
    result = captured.get("result")
    assert result is not None
    assert result.truncated is True
    assert result.frames_processed == 2
    assert result.total_frames == 5
    assert len(result.brightness_mean_data[0]) == 2


def test_analysis_worker_aborts_on_brightness_computation_failure(monkeypatch):
    """An injected computation fault must surface as an error signal, not silently
    become fabricated zero-brightness rows in the emitted result."""
    frames = [
        np.full((6, 6, 3), 20, dtype=np.uint8),
        np.full((6, 6, 3), 40, dtype=np.uint8),
    ]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: DummyVideoCapture(frames))

    def boom(*args, **kwargs):
        raise cv2.error("synthetic brightness computation failure")

    monkeypatch.setattr(frame_module, "compute_brightness_stats_detailed", boom)

    request = AnalysisRequest(
        video_path="dummy.mp4",
        rects=[((0, 0), (6, 6))],
        background_roi_idx=None,
        start_frame=0,
        end_frame=1,
        use_fixed_mask=False,
        fixed_roi_masks=[],
        background_percentile=90.0,
        morphological_kernel_size=3,
        noise_floor_threshold=0.0,
    )

    worker = AnalysisWorker(request)
    captured: Dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.error.connect(lambda message: captured.setdefault("error", message))
    worker.run()

    assert "result" not in captured
    assert "error" in captured
    assert "synthetic brightness computation failure" in captured["error"]


def test_analysis_worker_aborts_on_background_computation_failure(monkeypatch):
    """A background-ROI computation fault must abort the run with an error rather
    than silently switching to raw (non-background-subtracted) measurements."""
    frames = [
        np.full((6, 6, 3), 20, dtype=np.uint8),
        np.full((6, 6, 3), 40, dtype=np.uint8),
    ]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: DummyVideoCapture(frames))

    def boom(*args, **kwargs):
        raise BackgroundComputationError("synthetic background computation failure")

    monkeypatch.setattr(frame_module, "compute_background_brightness", boom)

    request = AnalysisRequest(
        video_path="dummy.mp4",
        rects=[((0, 0), (3, 6)), ((3, 0), (6, 6))],
        background_roi_idx=1,
        start_frame=0,
        end_frame=1,
        use_fixed_mask=False,
        fixed_roi_masks=[],
        background_percentile=90.0,
        morphological_kernel_size=3,
        noise_floor_threshold=0.0,
    )

    worker = AnalysisWorker(request)
    captured: Dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.error.connect(lambda message: captured.setdefault("error", message))
    worker.run()

    assert "result" not in captured
    assert "error" in captured
    assert "synthetic background computation failure" in captured["error"]


def test_analysis_worker_aborts_on_degenerate_background_roi(monkeypatch):
    """A configured background ROI that has zero area inside the frame must abort
    the run instead of silently exporting raw (non-subtracted) values."""
    frames = [np.full((6, 6, 3), 40, dtype=np.uint8)]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: DummyVideoCapture(frames))

    request = AnalysisRequest(
        video_path="dummy.mp4",
        rects=[((0, 0), (6, 6)), ((20, 20), (30, 30))],
        background_roi_idx=1,
        start_frame=0,
        end_frame=0,
        use_fixed_mask=False,
        fixed_roi_masks=[],
        background_percentile=90.0,
        morphological_kernel_size=3,
        noise_floor_threshold=0.0,
    )

    worker = AnalysisWorker(request)
    captured: Dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.error.connect(lambda message: captured.setdefault("error", message))
    worker.run()

    assert "result" not in captured
    assert "Background ROI 2" in captured["error"]


def test_analysis_worker_manual_threshold_gates_pixels(monkeypatch):
    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    frame[:, :3, :] = 10  # dark half, L* ~ 3
    frame[:, 3:, :] = 200  # bright half, L* ~ 81
    frames = [frame]

    l_star = compute_l_star_frame(frame)
    bright_pixels = l_star[l_star > 50.0]

    # Default (manual_threshold omitted): raw mean over the whole ROI, legacy behavior.
    res_default = _run_analysis_worker(frames, monkeypatch)
    assert res_default.brightness_mean_data[0][0] == pytest.approx(float(np.mean(l_star)), abs=1e-4)
    assert res_default.background_values_per_frame == [0.0]

    # Manual threshold gates pixels and offsets background-subtracted stats.
    res_50 = _run_analysis_worker(frames, monkeypatch, manual_threshold=50.0)
    assert res_50.brightness_mean_data[0][0] == pytest.approx(float(np.mean(bright_pixels)) - 50.0, abs=1e-4)
    assert res_50.background_values_per_frame == [50.0]

    # Changing the manual threshold changes the computed stats.
    res_20 = _run_analysis_worker(frames, monkeypatch, manual_threshold=20.0)
    assert res_20.brightness_mean_data[0][0] == pytest.approx(float(np.mean(bright_pixels)) - 20.0, abs=1e-4)
    assert res_20.brightness_mean_data[0][0] - res_50.brightness_mean_data[0][0] == pytest.approx(30.0, abs=1e-4)


def test_analysis_worker_manual_threshold_ignored_with_background_roi(monkeypatch):
    frame = np.zeros((6, 12, 3), dtype=np.uint8)
    frame[:, :6, :] = 200  # target ROI, bright
    frame[:, 6:, :] = 10  # background ROI, dark
    frames = [frame]
    rects = [((0, 0), (6, 6)), ((6, 0), (12, 6))]

    res_no_manual = _run_analysis_worker(
        frames, monkeypatch, rects=rects, background_roi_idx=1, manual_threshold=0.0
    )
    res_manual = _run_analysis_worker(
        frames, monkeypatch, rects=rects, background_roi_idx=1, manual_threshold=99.0
    )

    assert res_manual.brightness_mean_data == res_no_manual.brightness_mean_data
    assert res_manual.brightness_median_data == res_no_manual.brightness_median_data
    assert res_manual.background_values_per_frame == res_no_manual.background_values_per_frame


def test_brightest_frame_worker_picks_max_frame(monkeypatch):
    frames = [
        np.full((4, 4, 3), 10, dtype=np.uint8),
        np.full((4, 4, 3), 200, dtype=np.uint8),
        np.full((4, 4, 3), 50, dtype=np.uint8),
    ]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: DummyVideoCapture(frames))

    request = MaskScanRequest(
        video_path="dummy.mp4",
        rects=[((0, 0), (4, 4))],
        background_roi_idx=None,
        start_frame=0,
        end_frame=2,
        step=1,
        background_percentile=90.0,
        morphological_kernel_size=3,
    )

    worker = BrightestFrameWorker(request)
    captured: Dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.run()

    result = captured.get("result")
    assert isinstance(result, BrightestFrameResult)
    assert result.brightest_frame_idx == 1


def test_brightest_frame_worker_handles_edge_touching_roi(monkeypatch):
    frame0 = np.zeros((4, 4, 3), dtype=np.uint8)
    frame1 = np.zeros((4, 4, 3), dtype=np.uint8)
    frame1[:, 3:4, :] = 255
    frames = [frame0, frame1]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: DummyVideoCapture(frames))

    request = MaskScanRequest(
        video_path="dummy.mp4",
        rects=[((3, 0), (4, 4))],
        background_roi_idx=None,
        start_frame=0,
        end_frame=1,
        step=1,
        background_percentile=90.0,
        morphological_kernel_size=3,
    )

    worker = BrightestFrameWorker(request)
    captured: Dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.run()

    result = captured.get("result")
    assert isinstance(result, BrightestFrameResult)
    assert result.brightest_frame_idx == 1


def test_per_roi_mask_capture_worker_returns_sources(monkeypatch):
    frame0 = np.zeros((4, 4, 3), dtype=np.uint8)
    frame1 = np.zeros((4, 4, 3), dtype=np.uint8)
    frame0[:, 0:2, :] = 210
    frame1[:, 2:4, :] = 220
    frames = [frame0, frame1]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: DummyVideoCapture(frames))

    request = MaskScanRequest(
        video_path="dummy.mp4",
        rects=[((0, 0), (2, 4)), ((2, 0), (4, 4))],
        background_roi_idx=None,
        start_frame=0,
        end_frame=1,
        step=1,
        background_percentile=90.0,
        morphological_kernel_size=3,
    )

    worker = PerRoiMaskCaptureWorker(request)
    captured: Dict[str, object] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.run()

    result = captured.get("result")
    assert isinstance(result, PerRoiMaskCaptureResult)
    assert result.sources[0] == 0
    assert result.sources[1] == 1
    assert result.masks[0] is not None
    assert result.masks[1] is not None


def test_per_roi_mask_capture_worker_applies_manual_threshold(monkeypatch):
    """With no background ROI, per-ROI auto-capture must gate on the manual
    threshold (like "Capture From Current" and the analysis) instead of
    producing all-ones masks."""
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, 3:, :] = 220  # bright right half
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: DummyVideoCapture([frame, frame]))

    def capture(manual_threshold):
        request = MaskScanRequest(
            video_path="dummy.mp4",
            rects=[((0, 0), (6, 4))],
            background_roi_idx=None,
            start_frame=0,
            end_frame=1,
            step=1,
            background_percentile=90.0,
            morphological_kernel_size=1,
            manual_threshold=manual_threshold,
        )
        worker = PerRoiMaskCaptureWorker(request)
        captured: Dict[str, object] = {}
        worker.finished.connect(lambda payload: captured.setdefault("result", payload))
        worker.run()
        return captured["result"].masks[0]

    gated = capture(50.0)
    assert not gated[:, :3].any()
    assert gated[:, 3:].all()

    # 0 disables the manual threshold: the analysis then uses the whole ROI.
    assert capture(0.0).all()
