import cv2
import numpy as np
import pytest

from ecl_analysis.analysis.brightness import (
    BrightnessStats,
    compute_brightness,
    compute_brightness_stats,
    compute_brightness_stats_detailed,
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


def test_compute_brightness_stats_propagates_computation_errors(monkeypatch):
    """A processing fault must abort loudly, not be masked as fabricated zero data."""
    import ecl_analysis.analysis.brightness as brightness_module

    def boom(*args, **kwargs):
        raise cv2.error("synthetic failure")

    monkeypatch.setattr(brightness_module.cv2, "cvtColor", boom)

    roi = np.full((4, 4, 3), 128, dtype=np.uint8)
    with pytest.raises(cv2.error):
        compute_brightness_stats(roi)


def _two_level_roi() -> np.ndarray:
    """6x6 ROI: left half dark (L* ~3), right half bright (L* ~81)."""
    roi = np.zeros((6, 6, 3), dtype=np.uint8)
    roi[:, :3, :] = 10
    roi[:, 3:, :] = 200
    return roi


def test_detailed_stats_match_legacy_tuple():
    roi = _two_level_roi()
    for kwargs in (
        {},
        {"background_brightness": 50.0},
        {"noise_floor_threshold": 50.0},
        {"background_brightness": 10.0, "roi_mask": np.ones((6, 6), dtype=bool)},
    ):
        detailed = compute_brightness_stats_detailed(roi, **kwargs)
        assert tuple(detailed[:8]) == compute_brightness_stats(roi, **kwargs)


def test_detailed_stats_pixel_count_without_threshold_is_whole_roi():
    stats = compute_brightness_stats_detailed(_two_level_roi())
    assert stats.raw_pixel_count == 36
    assert stats.analyzed_pixel_count == 36


def test_detailed_stats_pixel_count_threshold_path_counts_only_lit_pixels():
    roi = _two_level_roi()
    l_star = compute_l_star_frame(roi)
    stats = compute_brightness_stats_detailed(roi, background_brightness=50.0)

    assert stats.raw_pixel_count == 36
    assert stats.analyzed_pixel_count == 18
    # Threshold path averages only lit pixels, so it is >= 0 and excludes dark pixels.
    expected = float(np.mean(l_star[:, 3:])) - 50.0
    assert stats.l_bg_sub_mean == pytest.approx(expected, abs=1e-4)
    # Integrated above-background intensity is recoverable from count * mean.
    assert stats.analyzed_pixel_count * stats.l_bg_sub_mean == pytest.approx(
        float(np.sum(l_star[:, 3:] - 50.0)), rel=1e-5
    )


def test_detailed_stats_pixel_count_zero_when_nothing_above_threshold():
    stats = compute_brightness_stats_detailed(_two_level_roi(), background_brightness=99.0)
    assert stats.analyzed_pixel_count == 0
    assert stats.l_bg_sub_mean == 0.0


def test_detailed_stats_masked_path_counts_mask_pixels_and_can_be_negative():
    roi = _two_level_roi()
    mask = np.zeros((6, 6), dtype=bool)
    mask[:, :3] = True  # dark half only
    l_star = compute_l_star_frame(roi)

    stats = compute_brightness_stats_detailed(roi, background_brightness=50.0, roi_mask=mask)

    assert stats.raw_pixel_count == 18
    assert stats.analyzed_pixel_count == 18
    # Fixed-mask path subtracts background from every mask pixel without gating.
    assert stats.l_bg_sub_mean == pytest.approx(float(np.mean(l_star[:, :3])) - 50.0, abs=1e-4)
    assert stats.l_bg_sub_mean < 0


def test_detailed_stats_blue_is_not_background_subtracted():
    stats = compute_brightness_stats_detailed(_two_level_roi(), background_brightness=50.0)
    # Raw blue over the analyzed (bright) pixels.
    assert stats.b_analyzed_mean == pytest.approx(200.0)


def test_detailed_stats_mismatched_mask_returns_zero_counts():
    stats = compute_brightness_stats_detailed(_two_level_roi(), roi_mask=np.ones((3, 3), dtype=bool))
    assert stats == BrightnessStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
