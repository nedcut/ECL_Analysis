import cv2
import numpy as np
import pytest

from ecl_analysis.video_analyzer import VideoAnalyzer


def test_compute_brightness_stats_zero_roi(video_analyzer_factory):
    analyzer: VideoAnalyzer = video_analyzer_factory()
    roi = np.zeros((2, 2, 3), dtype=np.uint8)

    stats = analyzer._compute_brightness_stats(roi)

    assert stats == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_compute_brightness_stats_white_roi(video_analyzer_factory):
    analyzer: VideoAnalyzer = video_analyzer_factory()
    roi = np.full((2, 2, 3), 255, dtype=np.uint8)

    stats = analyzer._compute_brightness_stats(roi)

    l_raw_mean, l_raw_median, l_bg_mean, l_bg_median, b_raw_mean, b_raw_median, b_bg_mean, b_bg_median = stats

    assert pytest.approx(l_raw_mean, rel=1e-3) == 100.0
    assert pytest.approx(l_raw_median, rel=1e-3) == 100.0
    assert pytest.approx(l_bg_mean, rel=1e-3) == 100.0
    assert pytest.approx(l_bg_median, rel=1e-3) == 100.0
    assert pytest.approx(b_raw_mean, rel=1e-3) == 255.0
    assert pytest.approx(b_raw_median, rel=1e-3) == 255.0
    assert pytest.approx(b_bg_mean, rel=1e-3) == 255.0
    assert pytest.approx(b_bg_median, rel=1e-3) == 255.0


def test_compute_brightness_stats_with_mask_and_background(video_analyzer_factory):
    analyzer: VideoAnalyzer = video_analyzer_factory()
    roi = np.array(
        [
            [[50, 60, 70], [200, 210, 220]],
            [[50, 60, 70], [200, 210, 220]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[True, False], [True, False]])

    background_brightness = 10.0
    stats = analyzer._compute_brightness_stats(roi, background_brightness=background_brightness, roi_mask=mask)

    l_raw_mean, l_raw_median, l_bg_mean, l_bg_median, b_raw_mean, b_raw_median, b_bg_mean, b_bg_median = stats

    # Only masked pixels (the first column) should be considered
    assert l_raw_mean == pytest.approx(l_raw_median)
    assert b_raw_mean == pytest.approx(b_raw_median)
    assert l_bg_mean == pytest.approx(l_raw_mean - background_brightness)
    assert l_bg_median == pytest.approx(l_raw_median - background_brightness)
    assert b_bg_mean == pytest.approx(b_raw_mean)
    assert b_bg_median == pytest.approx(b_raw_median)


def test_compute_brightness_stats_with_morphology(video_analyzer_factory):
    analyzer: VideoAnalyzer = video_analyzer_factory(morphological_kernel_size=3)

    roi = np.zeros((7, 7, 3), dtype=np.uint8)
    roi[2:5, 2:5, :] = 200
    roi[2:5, 2:5, 0] = 220  # Blue-enriched center block

    l_star_frame = analyzer._compute_l_star_frame(roi)
    stats = analyzer._compute_brightness_stats(
        roi,
        background_brightness=5.0,
        roi_l_star=l_star_frame,
    )

    # Morphological filtering retains the dense bright block
    assert stats[2] > 0.0
    assert stats[4] > 0.0


def test_compute_background_brightness(video_analyzer_factory):
    analyzer: VideoAnalyzer = video_analyzer_factory(
        rects=[((0, 0), (2, 2))],
        background_roi_idx=0,
        background_percentile=50.0,
    )

    roi_values = np.array(
        [
            [[10, 10, 10], [20, 20, 20]],
            [[30, 30, 30], [40, 40, 40]],
        ],
        dtype=np.uint8,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[0:2, 0:2, :] = roi_values

    frame_l_star = analyzer._compute_l_star_frame(frame)
    expected = float(np.percentile(frame_l_star[0:2, 0:2], 50.0))

    result = analyzer._compute_background_brightness(frame, frame_l_star=frame_l_star)
    assert result is not None
    assert pytest.approx(result, rel=1e-6) == expected


def test_validate_run_duration_confidence(video_analyzer_factory):
    class DummyCap:
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

    analyzer: VideoAnalyzer = video_analyzer_factory(cap=DummyCap())

    confidence = analyzer._validate_run_duration(start_frame=0, end_frame=59, expected_duration=2.0)
    assert pytest.approx(confidence, rel=1e-6) == 1.0
