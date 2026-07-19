"""Background brightness analysis helpers with no UI dependencies."""

import logging
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .brightness import compute_l_star_frame

Point = Tuple[int, int]
RoiRect = Tuple[Point, Point]


class BackgroundComputationError(RuntimeError):
    """Raised when background brightness cannot be computed for a configured ROI.

    This is distinct from a `None` return, which means "no background ROI is
    configured" — an intentional, expected state. This error means a background
    ROI *is* configured but the computation itself failed, which must not be
    papered over by silently falling back to raw (non-background-subtracted)
    measurements.
    """


def compute_background_brightness(
    frame: np.ndarray,
    rects: Sequence[RoiRect],
    background_roi_idx: Optional[int],
    background_percentile: float,
    frame_l_star: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Compute percentile L* brightness for a configured background ROI.

    Returns None only when no background ROI is configured (or the configured
    index/ROI is degenerate). Raises BackgroundComputationError if a background
    ROI is configured but the underlying computation fails, so callers cannot
    mistake a computation fault for "background not configured".
    """
    if background_roi_idx is None or frame is None:
        return None

    if not (0 <= background_roi_idx < len(rects)):
        return None

    try:
        pt1, pt2 = rects[background_roi_idx]
        frame_height, frame_width = frame.shape[:2]

        left, right = sorted((int(pt1[0]), int(pt2[0])))
        top, bottom = sorted((int(pt1[1]), int(pt2[1])))

        x1 = max(0, min(left, frame_width))
        x2 = max(0, min(right, frame_width))
        y1 = max(0, min(top, frame_height))
        y2 = max(0, min(bottom, frame_height))

        if x2 <= x1 or y2 <= y1:
            return None

        if frame_l_star is not None:
            roi_l_star = frame_l_star[y1:y2, x1:x2]
        else:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            roi_l_star = compute_l_star_frame(roi)

        if roi_l_star.size == 0:
            return None

        return float(np.percentile(roi_l_star, background_percentile))
    except cv2.error as exc:
        logging.exception("OpenCV error computing background brightness")
        raise BackgroundComputationError(str(exc)) from exc
    except Exception as exc:
        logging.exception("Error computing background brightness")
        raise BackgroundComputationError(str(exc)) from exc
