from typing import Optional, Tuple, List
import cv2
import numpy as np

from .utils import MORPHOLOGICAL_KERNEL_SIZE


def compute_brightness_stats(
    roi_bgr: np.ndarray,
    background_brightness: Optional[float] = None,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Calculate brightness statistics for a region of interest."""
    if roi_bgr is None or roi_bgr.size == 0:
        return (0.0,) * 8

    try:
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        l_star = l_chan * 100.0 / 255.0
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
                    (MORPHOLOGICAL_KERNEL_SIZE, MORPHOLOGICAL_KERNEL_SIZE),
                )
                mask_uint8 = above_background_mask.astype(np.uint8) * 255
                cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
                above_background_mask = cleaned_mask > 0
            if np.any(above_background_mask):
                filtered_l = l_star[above_background_mask]
                filtered_b = blue_chan[above_background_mask]
                bg_sub_l = filtered_l - background_brightness
                l_bg_sub_mean = float(np.mean(bg_sub_l))
                l_bg_sub_median = float(np.median(bg_sub_l))
                b_bg_sub_mean = float(np.mean(filtered_b))
                b_bg_sub_median = float(np.median(filtered_b))
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

    except Exception:
        return (0.0,) * 8


def compute_background_brightness(
    frame: np.ndarray,
    rects: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    background_roi_idx: Optional[int],
) -> Optional[float]:
    if background_roi_idx is None or frame is None:
        return None
    if not (0 <= background_roi_idx < len(rects)):
        return None

    try:
        pt1, pt2 = rects[background_roi_idx]
        roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        if roi.size == 0:
            return None
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        l_star = l_chan * 100.0 / 255.0
        return float(np.percentile(l_star, 90))
    except Exception:
        return None


def compute_brightness(roi_bgr: np.ndarray) -> float:
    return compute_brightness_stats(roi_bgr)[0]
