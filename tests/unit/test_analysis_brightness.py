import numpy as np
import pytest

from ecl_analysis.analysis.brightness import (
    compute_brightness,
    compute_brightness_stats,
    compute_l_star_frame,
)


def test_compute_l_star_frame_white_is_100():
    frame = np.full((3, 3, 3), 255, dtype=np.uint8)
    l_star = compute_l_star_frame(frame)

    assert l_star.shape == (3, 3)
    assert l_star.dtype == np.float32
    assert np.allclose(l_star, 100.0, atol=1e-3)


def test_compute_brightness_stats_empty_roi_returns_zeros():
    roi = np.zeros((0, 0, 3), dtype=np.uint8)
    stats = compute_brightness_stats(roi)
    assert stats == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_compute_brightness_stats_masked_background_subtraction():
    roi = np.array(
        [
            [[50, 60, 70], [200, 210, 220]],
            [[50, 60, 70], [200, 210, 220]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[True, False], [True, False]])

    background_brightness = 10.0
    stats = compute_brightness_stats(
        roi,
        background_brightness=background_brightness,
        roi_mask=mask,
    )

    l_raw_mean, l_raw_median, l_bg_mean, l_bg_median, b_raw_mean, b_raw_median, b_bg_mean, b_bg_median = stats

    assert l_raw_mean == pytest.approx(l_raw_median)
    assert b_raw_mean == pytest.approx(b_raw_median)
    assert l_bg_mean == pytest.approx(l_raw_mean - background_brightness)
    assert l_bg_median == pytest.approx(l_raw_median - background_brightness)
    assert b_bg_mean == pytest.approx(b_raw_mean)
    assert b_bg_median == pytest.approx(b_raw_median)


def test_compute_brightness_returns_mean_l_star():
    roi = np.full((4, 4, 3), 255, dtype=np.uint8)
    result = compute_brightness(roi)
    assert result == pytest.approx(100.0, rel=1e-3)
