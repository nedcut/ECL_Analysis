"""Brightness analysis functions with no UI dependencies."""

import logging
from typing import Optional, Tuple

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


def compute_l_star_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to its L* channel in the 0-100 range."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0].astype(np.float32)
    np.multiply(l_chan, 100.0 / 255.0, out=l_chan)
    return l_chan


def compute_brightness_stats(
    roi_bgr: np.ndarray,
    background_brightness: Optional[float] = None,
    roi_mask: Optional[np.ndarray] = None,
    roi_l_star: Optional[np.ndarray] = None,
    morphological_kernel_size: int = 3,
    noise_floor_threshold: float = 0.0,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Calculate L* and blue-channel statistics with optional masking and background subtraction.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return ZERO_BRIGHTNESS_STATS

    try:
        if roi_l_star is None:
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)
            l_star = l_chan * 100.0 / 255.0
        else:
            l_star = roi_l_star.astype(np.float32, copy=False)

        if roi_mask is not None:
            mask_bool = roi_mask.astype(bool)
            if mask_bool.shape[:2] != roi_bgr.shape[:2] or not np.any(mask_bool):
                return ZERO_BRIGHTNESS_STATS
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

        blue_chan = roi_bgr[:, :, 0].astype(np.float32)
        l_raw_mean = float(np.mean(l_star))
        l_raw_median = float(np.median(l_star))
        b_raw_mean = float(np.mean(blue_chan))
        b_raw_median = float(np.median(blue_chan))

        if background_brightness is not None:
            above_background_mask = l_star > background_brightness

            if np.any(above_background_mask):
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (morphological_kernel_size, morphological_kernel_size),
                )
                mask_uint8 = above_background_mask.astype(np.uint8) * 255
                cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
                above_background_mask = cleaned_mask > 0

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

                bg_subtracted_l_pixels = filtered_l_pixels - background_brightness
                l_bg_sub_mean = float(np.mean(bg_subtracted_l_pixels))
                l_bg_sub_median = float(np.median(bg_subtracted_l_pixels))
                b_bg_sub_mean = float(np.mean(filtered_b_pixels))
                b_bg_sub_median = float(np.median(filtered_b_pixels))
            else:
                l_bg_sub_mean = 0.0
                l_bg_sub_median = 0.0
                b_bg_sub_mean = 0.0
                b_bg_sub_median = 0.0
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
    except cv2.error:
        logging.exception("OpenCV error during brightness computation")
        return ZERO_BRIGHTNESS_STATS
    except Exception:
        logging.exception("Error during brightness computation")
        return ZERO_BRIGHTNESS_STATS


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
