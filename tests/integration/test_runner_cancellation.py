"""Cancellation releases decoders in all shared desktop/web runners."""

import cv2
import pytest

from ecl_analysis.analysis.models import AnalysisRequest
from ecl_analysis.analysis.runner import AnalysisCancelled, run_analysis
from ecl_analysis.analysis.scans import (
    MaskScanRequest,
    capture_per_roi_masks,
    find_brightest_frame,
)


@pytest.mark.parametrize("runner", [run_analysis, find_brightest_frame, capture_per_roi_masks])
def test_cancelled_shared_runner_releases_decoder(monkeypatch, runner):
    class Capture:
        released = False
        reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

        def set(self, *args):
            return True

        def read(self):
            self.reads += 1
            pytest.fail("Cancelled runs must not decode a frame")

        def release(self):
            self.released = True

    cap = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: cap)
    inputs = dict(
        video_path="cancelled.mp4", rects=[((0, 0), (2, 2))],
        background_roi_idx=None, start_frame=0, end_frame=1,
        background_percentile=90.0, morphological_kernel_size=3,
    )
    if runner is run_analysis:
        request = AnalysisRequest(**inputs, use_fixed_mask=False, fixed_roi_masks=[], noise_floor_threshold=0)
    else:
        request = MaskScanRequest(**inputs, step=1)
    with pytest.raises(AnalysisCancelled):
        runner(request, cancel_check=lambda: True)
    assert cap.released
    assert cap.reads == 0
