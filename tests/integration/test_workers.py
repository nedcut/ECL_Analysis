from typing import Dict, List, Optional

import cv2
import numpy as np

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
    def __init__(self, frames: List[np.ndarray]):
        self._frames = frames
        self._index = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._index = int(value)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
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
    assert len(result.brightness_mean_data) == 1
    assert len(result.brightness_mean_data[0]) == 3


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
