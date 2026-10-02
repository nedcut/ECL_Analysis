"""Brightness analysis functions with no UI dependencies.

Two pixel sets matter when reading these statistics:

* The **raw pixel set**: every pixel of the ROI, or every pixel selected by a
  fixed ROI mask when one is applied.
* The **analyzed pixel set**: the pixels that contribute to the
  ``l_bg_sub_*`` / ``b_analyzed_*`` statistics. Which pixels these are depends
  on the measurement method (see :func:`compute_brightness_stats_detailed`).
"""

from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np

ZERO_BRIGHTNESS_STATS: Tuple[float, float, float, float, float, float, float, float] = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


class BrightnessStats(NamedTuple):
    """Per-ROI, per-frame brightness statistics plus contributing pixel counts.

    The first eight fields are in the same order as the legacy 8-tuple returned
    by :func:`compute_brightness_stats`.

    Attributes:
        l_raw_mean / l_raw_median: L* (0-100) over the raw pixel set.
        l_bg_sub_mean / l_bg_sub_median: L* minus the background/threshold value
            over the analyzed pixel set (plain L* over the analyzed pixel set
            when no background/threshold value is supplied).
        b_raw_mean / b_raw_median: blue channel (0-255) over the raw pixel set.
        b_analyzed_mean / b_analyzed_median: raw (NOT background-subtracted)
            blue channel over the analyzed pixel set.
        raw_pixel_count: number of pixels in the raw pixel set.
        analyzed_pixel_count: number of pixels in the analyzed pixel set.
    """

    l_raw_mean: float
    l_raw_median: float
    l_bg_sub_mean: float
    l_bg_sub_median: float
    b_raw_mean: float
    b_raw_median: float
    b_analyzed_mean: float
    b_analyzed_median: float
    raw_pixel_count: int
    analyzed_pixel_count: int


ZERO_DETAILED_BRIGHTNESS_STATS = BrightnessStats(*ZERO_BRIGHTNESS_STATS, 0, 0)


def compute_l_star_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to its L* channel in the 0-100 range."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0].astype(np.float32)
    np.multiply(l_chan, 100.0 / 255.0, out=l_chan)
    return l_chan


def compute_threshold_pixel_mask(
    roi_l_star: np.ndarray,
    threshold: float,
    morphological_kernel_size: int,
    noise_floor_threshold: float = 0.0,
) -> np.ndarray:
    """Return the threshold pixels contributing to brightness statistics.

    Opening is applied before the absolute noise floor, matching the analysis
    measurement rule. Fixed masks and whole-ROI measurements bypass this rule.
    """
    mask = roi_l_star > threshold
    if np.any(mask):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphological_kernel_size, morphological_kernel_size),
        )
        mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel) > 0
    if noise_floor_threshold > 0:
        mask &= roi_l_star > noise_floor_threshold
    return mask


def compute_brightness_stats_detailed(
    roi_bgr: np.ndarray,
    background_brightness: Optional[float] = None,
    roi_mask: Optional[np.ndarray] = None,
    roi_l_star: Optional[np.ndarray] = None,
    morphological_kernel_size: int = 3,
    noise_floor_threshold: float = 0.0,
) -> BrightnessStats:
    """Calculate L* and blue-channel statistics plus the pixel counts behind them.

    Two measurement methods exist, and they define ``l_bg_sub_*`` differently:

    **Fixed mask** (``roi_mask`` given): the analyzed pixel set is exactly the
    mask. ``l_bg_sub_*`` is the mean/median of ``L* - background_brightness``
    over every mask pixel. No threshold gating, morphology, or noise floor is
    applied, so the value can be negative when mask pixels are darker than the
    background. A mask that does not match the ROI shape, or selects no pixels,
    yields all-zero stats with zero pixel counts.

    **Threshold** (no ``roi_mask``):

    * With ``background_brightness``: pixels with ``L* > background_brightness``
      are cleaned with a morphological opening (elliptical kernel of
      ``morphological_kernel_size``), then intersected with
      ``L* > noise_floor_threshold`` when that is > 0. ``l_bg_sub_*`` is
      ``L* - background_brightness`` over only those surviving pixels, so it is
      always >= 0 and is biased upward relative to a whole-ROI mean (it only
      describes lit pixels). If no pixel survives, the analyzed stats are 0.0
      with ``analyzed_pixel_count == 0``.
    * Without ``background_brightness`` but ``noise_floor_threshold > 0``: the
      analyzed set is ``L* > noise_floor_threshold`` after morphological
      opening, and ``l_bg_sub_*`` is plain L* over that set.
    * Otherwise the analyzed set equals the raw pixel set.

    ``b_analyzed_*`` is always the raw blue channel over the analyzed pixel set;
    blue is never background-subtracted.

    ``analyzed_pixel_count * l_bg_sub_mean`` is the integrated L* above
    background over the ROI for that frame.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return ZERO_DETAILED_BRIGHTNESS_STATS

    # Note: computation errors below are intentionally NOT caught here. Silently
    # converting an OpenCV/processing fault into a plausible-looking zero-brightness
    # measurement would corrupt the analysis run with fabricated data. Callers (the
    # analysis worker) are expected to let the exception propagate and abort the run
    # via the existing error-signal path rather than continuing with fake results.
    if roi_l_star is None:
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        l_star = l_chan * 100.0 / 255.0
    else:
        l_star = roi_l_star.astype(np.float32, copy=False)

    if roi_mask is not None:
        mask_bool = roi_mask.astype(bool)
        if mask_bool.shape[:2] != roi_bgr.shape[:2] or not np.any(mask_bool):
            return ZERO_DETAILED_BRIGHTNESS_STATS
        mask_pixel_count = int(np.count_nonzero(mask_bool))
        blue_chan = roi_bgr[:, :, 0].astype(np.float32)
        l_pixels = l_star[mask_bool]
        b_pixels = blue_chan[mask_bool]
        l_raw_mean = float(np.mean(l_pixels))
        l_raw_median = float(np.median(l_pixels))
        b_raw_mean = float(np.mean(b_pixels))
        b_raw_median = float(np.median(b_pixels))
        if background_brightness is not None:
            l_bg = l_pixels - background_brightness
            l_bg_sub_mean = float(np.mean(l_bg))
            l_bg_sub_median = float(np.median(l_bg))
        else:
            l_bg_sub_mean = l_raw_mean
            l_bg_sub_median = l_raw_median
        # Blue over the same mask pixels; blue is never background-subtracted.
        b_analyzed_mean = b_raw_mean
        b_analyzed_median = b_raw_median
        return BrightnessStats(
            l_raw_mean,
            l_raw_median,
            l_bg_sub_mean,
            l_bg_sub_median,
            b_raw_mean,
            b_raw_median,
            b_analyzed_mean,
            b_analyzed_median,
            mask_pixel_count,
            mask_pixel_count,
        )

    blue_chan = roi_bgr[:, :, 0].astype(np.float32)
    raw_pixel_count = int(l_star.size)
    l_raw_mean = float(np.mean(l_star))
    l_raw_median = float(np.median(l_star))
    b_raw_mean = float(np.mean(blue_chan))
    b_raw_median = float(np.median(blue_chan))

    if background_brightness is not None:
        above_background_mask = compute_threshold_pixel_mask(
            l_star, background_brightness, morphological_kernel_size
        )

        if np.any(above_background_mask):
            if noise_floor_threshold > 0:
                noise_floor_mask = l_star > noise_floor_threshold
                combined_mask = above_background_mask & noise_floor_mask
            else:
                combined_mask = above_background_mask

            if np.any(combined_mask):
                filtered_l_pixels = l_star[combined_mask]
                filtered_b_pixels = blue_chan[combined_mask]
            else:
                return BrightnessStats(
                    l_raw_mean,
                    l_raw_median,
                    0.0,
                    0.0,
                    b_raw_mean,
                    b_raw_median,
                    0.0,
                    0.0,
                    raw_pixel_count,
                    0,
                )

            bg_subtracted_l_pixels = filtered_l_pixels - background_brightness
            l_bg_sub_mean = float(np.mean(bg_subtracted_l_pixels))
            l_bg_sub_median = float(np.median(bg_subtracted_l_pixels))
            b_analyzed_mean = float(np.mean(filtered_b_pixels))
            b_analyzed_median = float(np.median(filtered_b_pixels))
            analyzed_pixel_count = int(filtered_l_pixels.size)
        else:
            l_bg_sub_mean = 0.0
            l_bg_sub_median = 0.0
            b_analyzed_mean = 0.0
            b_analyzed_median = 0.0
            analyzed_pixel_count = 0
    elif noise_floor_threshold > 0:
        noise_floor_mask = compute_threshold_pixel_mask(
            l_star, noise_floor_threshold, morphological_kernel_size
        )

        if np.any(noise_floor_mask):
            filtered_l_pixels = l_star[noise_floor_mask]
            filtered_b_pixels = blue_chan[noise_floor_mask]
            l_bg_sub_mean = float(np.mean(filtered_l_pixels))
            l_bg_sub_median = float(np.median(filtered_l_pixels))
            b_analyzed_mean = float(np.mean(filtered_b_pixels))
            b_analyzed_median = float(np.median(filtered_b_pixels))
            analyzed_pixel_count = int(filtered_l_pixels.size)
        else:
            l_bg_sub_mean = 0.0
            l_bg_sub_median = 0.0
            b_analyzed_mean = 0.0
            b_analyzed_median = 0.0
            analyzed_pixel_count = 0
    else:
        l_bg_sub_mean = l_raw_mean
        l_bg_sub_median = l_raw_median
        b_analyzed_mean = b_raw_mean
        b_analyzed_median = b_raw_median
        analyzed_pixel_count = raw_pixel_count

    return BrightnessStats(
        l_raw_mean,
        l_raw_median,
        l_bg_sub_mean,
        l_bg_sub_median,
        b_raw_mean,
        b_raw_median,
        b_analyzed_mean,
        b_analyzed_median,
        raw_pixel_count,
        analyzed_pixel_count,
    )


def compute_brightness_stats(
    roi_bgr: np.ndarray,
    background_brightness: Optional[float] = None,
    roi_mask: Optional[np.ndarray] = None,
    roi_l_star: Optional[np.ndarray] = None,
    morphological_kernel_size: int = 3,
    noise_floor_threshold: float = 0.0,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Calculate L* and blue-channel statistics with optional masking and background subtraction.

    Returns the legacy 8-tuple ``(l_raw_mean, l_raw_median, l_bg_sub_mean,
    l_bg_sub_median, b_raw_mean, b_raw_median, b_analyzed_mean,
    b_analyzed_median)``. See :func:`compute_brightness_stats_detailed` for the
    exact definition of each value and for the contributing pixel counts. The
    last two values (historically named ``b_bg_sub_*``) are raw blue over the
    analyzed pixel set; blue is never background-subtracted.
    """
    stats = compute_brightness_stats_detailed(
        roi_bgr,
        background_brightness=background_brightness,
        roi_mask=roi_mask,
        roi_l_star=roi_l_star,
        morphological_kernel_size=morphological_kernel_size,
        noise_floor_threshold=noise_floor_threshold,
    )
    return tuple(stats[:8])


def compute_brightness(
    roi_bgr: np.ndarray,
    morphological_kernel_size: int = 3,
    noise_floor_threshold: float = 0.0,
) -> float:
    """Return only the mean L* brightness for compatibility call sites."""
    l_raw_mean, _, _, _, _, _, _, _ = compute_brightness_stats(
        roi_bgr,
        morphological_kernel_size=morphological_kernel_size,
        noise_floor_threshold=noise_floor_threshold,
    )
    return l_raw_mean
