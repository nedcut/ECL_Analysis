import numpy as np
import pytest

from ecl_analysis.analysis.background import compute_background_brightness
from ecl_analysis.analysis.brightness import compute_l_star_frame


def test_compute_background_brightness_missing_background_roi_returns_none():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    rects = [((0, 0), (2, 2))]

    result = compute_background_brightness(
        frame=frame,
        rects=rects,
        background_roi_idx=None,
        background_percentile=90.0,
    )
    assert result is None


def test_compute_background_brightness_with_precomputed_l_star():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[0:2, 0:2, :] = np.array(
        [
            [[10, 10, 10], [20, 20, 20]],
            [[30, 30, 30], [40, 40, 40]],
        ],
        dtype=np.uint8,
    )
    rects = [((0, 0), (2, 2))]
    frame_l_star = compute_l_star_frame(frame)

    result = compute_background_brightness(
        frame=frame,
        rects=rects,
        background_roi_idx=0,
        background_percentile=50.0,
        frame_l_star=frame_l_star,
    )

    expected = float(np.percentile(frame_l_star[0:2, 0:2], 50.0))
    assert result is not None
    assert result == pytest.approx(expected, rel=1e-6)


def test_compute_background_brightness_without_precomputed_l_star():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[1:3, 1:3, :] = 200
    rects = [((1, 1), (3, 3))]

    result = compute_background_brightness(
        frame=frame,
        rects=rects,
        background_roi_idx=0,
        background_percentile=90.0,
    )

    assert result is not None
    assert result > 0.0
