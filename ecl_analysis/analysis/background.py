"""Background brightness analysis helpers with no UI dependencies."""

import logging
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .brightness import compute_l_star_frame

Point = Tuple[int, int]
RoiRect = Tuple[Point, Point]


def compute_background_brightness(
    frame: np.ndarray,
    rects: Sequence[RoiRect],
    background_roi_idx: Optional[int],
    background_percentile: float,
    frame_l_star: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Compute percentile L* brightness for a configured background ROI."""
    if background_roi_idx is None or frame is None:
        return None

    if not (0 <= background_roi_idx < len(rects)):
        return None

    try:
        pt1, pt2 = rects[background_roi_idx]
        if frame_l_star is not None:
            roi_l_star = frame_l_star[pt1[1] : pt2[1], pt1[0] : pt2[0]]
        else:
            roi = frame[pt1[1] : pt2[1], pt1[0] : pt2[0]]
            if roi.size == 0:
                return None
            roi_l_star = compute_l_star_frame(roi)

        if roi_l_star.size == 0:
            return None

        return float(np.percentile(roi_l_star, background_percentile))
    except cv2.error:
        logging.exception("OpenCV error computing background brightness")
        return None
    except Exception:
        logging.exception("Error computing background brightness")
        return None
