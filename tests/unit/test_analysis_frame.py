import numpy as np
import pytest

from ecl_analysis.analysis.brightness import compute_brightness_stats_detailed, compute_l_star_frame
from ecl_analysis.analysis.frame import (
    FrameAnalysisSettings,
    analyze_frame,
    build_roi_mask,
    build_threshold_mask,
    resolve_frame_threshold,
)
from ecl_analysis.analysis.models import (
    MASK_STATUS_APPLIED,
    MASK_STATUS_MISSING,
    MASK_STATUS_SHAPE_MISMATCH,
    THRESHOLD_MODE_BACKGROUND_ROI,
    THRESHOLD_MODE_MANUAL,
    THRESHOLD_MODE_NONE,
)


def _frame() -> np.ndarray:
    frame = np.full((8, 8, 3), 10, dtype=np.uint8)  # dark, L* ~ 3
    frame[:, 4:, :] = 200  # bright right half, L* ~ 81
    return frame


def _settings(**overrides) -> FrameAnalysisSettings:
    values = dict(background_percentile=90.0, morphological_kernel_size=1, noise_floor_threshold=0.0)
    values.update(overrides)
    return FrameAnalysisSettings(**values)


def test_resolve_frame_threshold_modes():
    frame = _frame()
    rects = [((0, 0), (8, 8)), ((0, 0), (4, 8))]
    assert resolve_frame_threshold(frame, rects, None, 90.0, 0.0) is None
    assert resolve_frame_threshold(frame, rects, None, 90.0, 40.0) == 40.0
    # A background ROI wins over the manual threshold.
    bg = resolve_frame_threshold(frame, rects, 1, 90.0, 40.0)
    assert bg == pytest.approx(float(np.percentile(compute_l_star_frame(frame)[:, :4], 90.0)))


def test_analyze_frame_manual_threshold_gates_and_subtracts():
    frame = _frame()
    result = analyze_frame(frame, [((0, 0), (8, 8))], None, _settings(manual_threshold=40.0))

    assert result.threshold == 40.0
    assert result.threshold_mode == THRESHOLD_MODE_MANUAL
    (roi,) = result.rois
    expected = compute_brightness_stats_detailed(frame, background_brightness=40.0, morphological_kernel_size=1)
    assert roi.stats == expected
    assert roi.thresholded
    assert roi.mean == expected.l_bg_sub_mean
    assert roi.pixel_count == expected.analyzed_pixel_count == 32
    assert roi.mask_status is None


def test_analyze_frame_without_threshold_exports_raw_values():
    frame = _frame()
    result = analyze_frame(frame, [((0, 0), (8, 8))], None, _settings())

    assert result.threshold is None
    assert result.threshold_mode == THRESHOLD_MODE_NONE
    (roi,) = result.rois
    assert not roi.thresholded
    assert roi.mean == roi.stats.l_raw_mean
    assert roi.blue_mean == roi.stats.b_raw_mean
    assert roi.pixel_count == 64


def test_analyze_frame_skips_background_roi_and_zeroes_empty_rois():
    frame = _frame()
    rects = [((0, 0), (8, 8)), ((0, 0), (4, 8)), ((20, 20), (30, 30))]
    result = analyze_frame(frame, rects, 1, _settings())

    assert result.threshold_mode == THRESHOLD_MODE_BACKGROUND_ROI
    assert [roi.roi_idx for roi in result.rois] == [0, 2]
    empty = result.rois[1]
    assert empty.is_empty
    assert (empty.mean, empty.median, empty.blue_mean, empty.blue_median, empty.pixel_count) == (0.0, 0.0, 0.0, 0.0, 0)


def test_analyze_frame_fixed_mask_statuses():
    frame = _frame()
    rects = [((0, 0), (8, 8)), ((0, 0), (4, 4)), ((4, 4), (8, 8))]
    masks = [np.ones((8, 8), dtype=bool), np.ones((3, 3), dtype=bool), None]

    result = analyze_frame(frame, rects, None, _settings(use_fixed_mask=True), masks=masks)
    assert [roi.mask_status for roi in result.rois] == [
        MASK_STATUS_APPLIED,
        MASK_STATUS_SHAPE_MISMATCH,
        MASK_STATUS_MISSING,
    ]

    # Masks are ignored unless fixed-mask mode is on.
    result_off = analyze_frame(frame, rects, None, _settings(), masks=masks)
    assert [roi.mask_status for roi in result_off.rois] == [None, None, None]


def test_build_threshold_mask_rules():
    l_star = compute_l_star_frame(_frame())
    assert build_threshold_mask(l_star, None, 3).all()
    mask = build_threshold_mask(l_star, 40.0, 1)
    assert not mask[:, :4].any()
    assert mask[:, 4:].all()


def test_build_roi_mask_matches_analyzed_region_including_frame_edge():
    l_star = compute_l_star_frame(_frame())
    mask = build_roi_mask(l_star, ((8, 8), (4, 0)), 40.0, 1)  # reversed corners, on the edge
    assert mask is not None and mask.shape == (8, 4)
    assert build_roi_mask(l_star, ((9, 0), (12, 8)), 40.0, 1) is None
