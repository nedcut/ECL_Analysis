import cv2
import numpy as np
import pytest

from ecl_analysis.analysis.background import compute_background_brightness
from ecl_analysis.analysis.brightness import compute_brightness_stats
from ecl_analysis.video_analyzer import VideoAnalyzer


def test_compute_brightness_stats_wrapper_uses_instance_parameters(video_analyzer_factory):
    analyzer: VideoAnalyzer = video_analyzer_factory(morphological_kernel_size=5, noise_floor_threshold=2.5)

    roi = np.array(
        [
            [[0, 0, 0], [250, 250, 250], [0, 0, 0]],
            [[250, 250, 250], [250, 250, 250], [250, 250, 250]],
            [[0, 0, 0], [250, 250, 250], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    roi_l_star = analyzer._compute_l_star_frame(roi)

    wrapped_stats = analyzer._compute_brightness_stats(
        roi,
        background_brightness=10.0,
        roi_l_star=roi_l_star,
    )
    direct_stats = compute_brightness_stats(
        roi_bgr=roi,
        background_brightness=10.0,
        roi_l_star=roi_l_star,
        morphological_kernel_size=analyzer.morphological_kernel_size,
        noise_floor_threshold=analyzer.noise_floor_threshold,
    )

    assert wrapped_stats == pytest.approx(direct_stats, rel=1e-6)


def test_compute_background_brightness_wrapper_uses_instance_state(video_analyzer_factory):
    analyzer: VideoAnalyzer = video_analyzer_factory(
        rects=[((0, 0), (2, 2))],
        background_roi_idx=0,
        background_percentile=50.0,
    )

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[0:2, 0:2, :] = np.array(
        [
            [[10, 10, 10], [20, 20, 20]],
            [[30, 30, 30], [40, 40, 40]],
        ],
        dtype=np.uint8,
    )

    frame_l_star = analyzer._compute_l_star_frame(frame)
    wrapped_result = analyzer._compute_background_brightness(frame, frame_l_star=frame_l_star)
    direct_result = compute_background_brightness(
        frame=frame,
        rects=analyzer.rects,
        background_roi_idx=analyzer.background_roi_idx,
        background_percentile=analyzer.background_percentile,
        frame_l_star=frame_l_star,
    )

    assert wrapped_result is not None
    assert direct_result is not None
    assert wrapped_result == pytest.approx(direct_result, rel=1e-6)


def test_validate_run_duration_confidence(video_analyzer_factory):
    class DummyCap:
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

    analyzer: VideoAnalyzer = video_analyzer_factory(cap=DummyCap())

    confidence = analyzer._validate_run_duration(start_frame=0, end_frame=59, expected_duration=2.0)
    assert pytest.approx(confidence, rel=1e-6) == 1.0
