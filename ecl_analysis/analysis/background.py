"""Background brightness analysis helpers with no UI dependencies."""

import logging
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..roi_geometry import roi_slice_bounds
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

    Returns None only when no background ROI is configured
    (``background_roi_idx is None``). Raises BackgroundComputationError when a
    background ROI is configured but unusable (index out of range, no frame,
    or zero area after clamping to the frame) or when the computation itself
    fails, so callers cannot mistake a broken background ROI for "background
    not configured" and silently export raw measurements.
    """
    if background_roi_idx is None:
        return None

    if frame is None:
        raise BackgroundComputationError(
            f"Background ROI {background_roi_idx + 1} is configured but no frame was provided."
        )

    if not (0 <= background_roi_idx < len(rects)):
        raise BackgroundComputationError(
            f"Background ROI index {background_roi_idx} is out of range ({len(rects)} ROIs defined)."
        )

    try:
        pt1, pt2 = rects[background_roi_idx]
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = roi_slice_bounds(pt1, pt2, frame_width, frame_height)

        if x2 <= x1 or y2 <= y1:
            raise BackgroundComputationError(
                f"Background ROI {background_roi_idx + 1} has zero area inside the "
                f"{frame_width}x{frame_height} frame (rect {pt1}-{pt2})."
            )

        if frame_l_star is not None:
            roi_l_star = frame_l_star[y1:y2, x1:x2]
        else:
            roi_l_star = compute_l_star_frame(frame[y1:y2, x1:x2])

        if roi_l_star.size == 0:
            raise BackgroundComputationError(
                f"Background ROI {background_roi_idx + 1} contains no pixels."
            )

        return float(np.percentile(roi_l_star, background_percentile))
    except BackgroundComputationError:
        raise
    except cv2.error as exc:
        logging.exception("OpenCV error computing background brightness")
        raise BackgroundComputationError(str(exc)) from exc
    except Exception as exc:
        logging.exception("Error computing background brightness")
        raise BackgroundComputationError(str(exc)) from exc
