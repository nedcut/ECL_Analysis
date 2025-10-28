"""Core analysis helpers shared between UI and worker threads."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Rect = Tuple[Tuple[int, int], Tuple[int, int]]


def compute_l_star_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to its L* channel scaled to 0-100."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0].astype(np.float32)
    np.multiply(l_chan, 100.0 / 255.0, out=l_chan)
    return l_chan


def compute_background_brightness(
    frame: np.ndarray,
    rects: Sequence[Rect],
    background_roi_idx: Optional[int],
    background_percentile: float,
    frame_l_star: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Return the percentile L* brightness for the configured background ROI."""
    if background_roi_idx is None or frame is None:
        return None

    if not (0 <= background_roi_idx < len(rects)):
        return None

    pt1, pt2 = rects[background_roi_idx]
    if frame_l_star is not None:
        roi_l_star = frame_l_star[pt1[1] : pt2[1], pt1[0] : pt2[0]]
    else:
        roi = frame[pt1[1] : pt2[1], pt1[0] : pt2[0]]
        if roi.size == 0:
            return None
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        roi_l_star = l_chan * 100.0 / 255.0

    if roi_l_star.size == 0:
        return None

    return float(np.percentile(roi_l_star, background_percentile))


def _apply_morphology(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Run an opening operation to smooth binary masks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_uint8 = mask.astype(np.uint8) * 255
    cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    return cleaned_mask > 0


def compute_brightness_stats(
    roi_bgr: np.ndarray,
    morphological_kernel_size: int,
    noise_floor_threshold: float,
    background_brightness: Optional[float] = None,
    roi_mask: Optional[np.ndarray] = None,
    roi_l_star: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Calculates brightness statistics for an ROI with optional background subtraction.

    Returns tuple of (l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median,
    b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median).
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return (0.0,) * 8

    try:
        if roi_l_star is None:
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)
            l_star = l_chan * 100.0 / 255.0
        else:
            l_star = roi_l_star.astype(np.float32, copy=False)

        blue_chan = roi_bgr[:, :, 0].astype(np.float32)

        if roi_mask is not None:
            mask_bool = roi_mask.astype(bool)
            if mask_bool.shape[:2] != roi_bgr.shape[:2] or not np.any(mask_bool):
                return (0.0,) * 8

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
                b_bg_sub_mean = b_raw_mean
                b_bg_sub_median = b_raw_median
            else:
                l_bg_sub_mean = l_raw_mean
                l_bg_sub_median = l_raw_median
                b_bg_sub_mean = b_raw_mean
                b_bg_sub_median = b_raw_median

            return (
                l_raw_mean,
                l_raw_median,
                l_bg_sub_mean,
                l_bg_sub_median,
                b_raw_mean,
                b_raw_median,
                b_bg_sub_mean,
                b_bg_sub_median,
            )

        l_raw_mean = float(np.mean(l_star))
        l_raw_median = float(np.median(l_star))
        b_raw_mean = float(np.mean(blue_chan))
        b_raw_median = float(np.median(blue_chan))

        if background_brightness is None:
            return (
                l_raw_mean,
                l_raw_median,
                l_raw_mean,
                l_raw_median,
                b_raw_mean,
                b_raw_median,
                b_raw_mean,
                b_raw_median,
            )

        above_background_mask = l_star > background_brightness
        if np.any(above_background_mask):
            above_background_mask = _apply_morphology(above_background_mask, morphological_kernel_size)

        if np.any(above_background_mask):
            if noise_floor_threshold > 0:
                noise_floor_mask = l_star > noise_floor_threshold
                combined_mask = above_background_mask & noise_floor_mask
            else:
                combined_mask = above_background_mask

            if not np.any(combined_mask):
                return (
                    l_raw_mean,
                    l_raw_median,
                    0.0,
                    0.0,
                    b_raw_mean,
                    b_raw_median,
                    0.0,
                    0.0,
                )

            filtered_l_pixels = l_star[combined_mask]
            filtered_b_pixels = blue_chan[combined_mask]

            bg_subtracted_l_pixels = filtered_l_pixels - background_brightness
            l_bg_sub_mean = float(np.mean(bg_subtracted_l_pixels))
            l_bg_sub_median = float(np.median(bg_subtracted_l_pixels))
            b_bg_sub_mean = float(np.mean(filtered_b_pixels))
            b_bg_sub_median = float(np.median(filtered_b_pixels))

            return (
                l_raw_mean,
                l_raw_median,
                l_bg_sub_mean,
                l_bg_sub_median,
                b_raw_mean,
                b_raw_median,
                b_bg_sub_mean,
                b_bg_sub_median,
            )

        return (
            l_raw_mean,
            l_raw_median,
            0.0,
            0.0,
            b_raw_mean,
            b_raw_median,
            0.0,
            0.0,
        )

    except cv2.error:
        return (0.0,) * 8
    except Exception:
        return (0.0,) * 8
