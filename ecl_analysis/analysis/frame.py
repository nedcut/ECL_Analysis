"""Per-frame analysis shared by the analysis worker and the live preview.

:func:`analyze_frame` is the single definition of what one frame measures:
which threshold applies, which pixels each ROI covers, whether a fixed mask is
used, and which statistics are exported. The analysis worker calls it for every
frame and the live preview calls it for the displayed frame, so the two cannot
drift apart. :func:`build_roi_mask` is likewise the single rule for turning a
frame into a fixed ROI mask ("Capture From Current" and per-ROI auto-capture).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..roi_geometry import roi_slice_bounds
from .background import compute_background_brightness
from .brightness import (
    ZERO_DETAILED_BRIGHTNESS_STATS,
    BrightnessStats,
    compute_brightness_stats_detailed,
    compute_l_star_frame,
)
from .models import (
    MASK_STATUS_APPLIED,
    MASK_STATUS_MISSING,
    MASK_STATUS_SHAPE_MISMATCH,
    AnalysisRequest,
    RoiRect,
    resolve_threshold_mode,
)

RoiBounds = Tuple[int, int, int, int]


@dataclass(frozen=True)
class FrameAnalysisSettings:
    """Measurement settings that affect per-frame results."""

    background_percentile: float
    morphological_kernel_size: int
    noise_floor_threshold: float
    # Used only when no background ROI is configured; 0 disables it.
    manual_threshold: float = 0.0
    use_fixed_mask: bool = False

    @classmethod
    def from_request(cls, request: AnalysisRequest) -> "FrameAnalysisSettings":
        return cls(
            background_percentile=request.background_percentile,
            morphological_kernel_size=request.morphological_kernel_size,
            noise_floor_threshold=request.noise_floor_threshold,
            manual_threshold=request.manual_threshold,
            use_fixed_mask=request.use_fixed_mask,
        )


@dataclass(frozen=True)
class RoiFrameAnalysis:
    """One non-background ROI measured on one frame.

    ``mean``/``median``/``blue_mean``/``blue_median``/``pixel_count`` are the
    values the analysis exports for this ROI and frame.
    """

    roi_idx: int
    bounds: RoiBounds
    stats: BrightnessStats
    # True when a background/threshold value was applied to this frame.
    thresholded: bool
    # Fixed-mask outcome (MASK_STATUS_*); None when fixed masks are not in use
    # or the ROI has no area inside the frame.
    mask_status: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        x1, y1, x2, y2 = self.bounds
        return x2 <= x1 or y2 <= y1

    @property
    def mean(self) -> float:
        return self.stats.l_bg_sub_mean if self.thresholded else self.stats.l_raw_mean

    @property
    def median(self) -> float:
        return self.stats.l_bg_sub_median if self.thresholded else self.stats.l_raw_median

    @property
    def blue_mean(self) -> float:
        # Blue is never background-subtracted: with a threshold it is raw blue
        # over the same analyzed pixels as L*.
        return self.stats.b_analyzed_mean if self.thresholded else self.stats.b_raw_mean

    @property
    def blue_median(self) -> float:
        return self.stats.b_analyzed_median if self.thresholded else self.stats.b_raw_median

    @property
    def pixel_count(self) -> int:
        return self.stats.analyzed_pixel_count if self.thresholded else self.stats.raw_pixel_count


@dataclass(frozen=True)
class FrameAnalysis:
    """Result of :func:`analyze_frame` for one frame."""

    # Background/threshold L* applied to the frame (background-ROI percentile or
    # manual threshold); None when no threshold applies.
    threshold: Optional[float]
    threshold_mode: str
    # Non-background ROIs in index order.
    rois: List[RoiFrameAnalysis]


def resolve_frame_threshold(
    frame: np.ndarray,
    rects: Sequence[RoiRect],
    background_roi_idx: Optional[int],
    background_percentile: float,
    manual_threshold: float,
    frame_l_star: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Return the background/threshold L* the analysis applies to ``frame``.

    The background-ROI percentile when a background ROI is configured,
    otherwise the manual threshold when it is above zero, otherwise None.
    Raises BackgroundComputationError if a configured background ROI fails.
    """
    threshold = compute_background_brightness(
        frame=frame,
        rects=rects,
        background_roi_idx=background_roi_idx,
        background_percentile=background_percentile,
        frame_l_star=frame_l_star,
    )
    if background_roi_idx is None and manual_threshold > 0:
        # Manual threshold mode: no background ROI configured, so the
        # user-set manual threshold acts as the active threshold.
        threshold = manual_threshold
    return threshold


def analyze_frame(
    frame: np.ndarray,
    rects: Sequence[RoiRect],
    background_roi_idx: Optional[int],
    settings: FrameAnalysisSettings,
    masks: Sequence[Optional[np.ndarray]] = (),
    frame_l_star: Optional[np.ndarray] = None,
) -> FrameAnalysis:
    """Measure every non-background ROI on one BGR frame.

    ``masks`` is aligned with ``rects`` and is only consulted when
    ``settings.use_fixed_mask`` is set; a mask whose shape does not match the
    ROI is ignored (``MASK_STATUS_SHAPE_MISMATCH``) and the threshold method is
    used instead. ``frame_l_star`` may be passed to reuse a precomputed L*
    frame. Computation errors (including BackgroundComputationError) propagate.
    """
    l_star_frame = frame_l_star if frame_l_star is not None else compute_l_star_frame(frame)
    threshold = resolve_frame_threshold(
        frame,
        rects,
        background_roi_idx,
        settings.background_percentile,
        settings.manual_threshold,
        frame_l_star=l_star_frame,
    )
    thresholded = threshold is not None

    frame_height, frame_width = frame.shape[:2]
    rois: List[RoiFrameAnalysis] = []
    for roi_idx, (pt1, pt2) in enumerate(rects):
        if roi_idx == background_roi_idx:
            continue
        bounds = roi_slice_bounds(pt1, pt2, frame_width, frame_height)
        x1, y1, x2, y2 = bounds
        if x2 <= x1 or y2 <= y1:
            rois.append(RoiFrameAnalysis(roi_idx, bounds, ZERO_DETAILED_BRIGHTNESS_STATS, thresholded))
            continue

        roi = frame[y1:y2, x1:x2]
        roi_mask = None
        mask_status = None
        if settings.use_fixed_mask:
            candidate = masks[roi_idx] if roi_idx < len(masks) else None
            if isinstance(candidate, np.ndarray) and candidate.shape[:2] == roi.shape[:2]:
                roi_mask = candidate
                mask_status = MASK_STATUS_APPLIED
            elif isinstance(candidate, np.ndarray):
                mask_status = MASK_STATUS_SHAPE_MISMATCH
            else:
                mask_status = MASK_STATUS_MISSING

        stats = compute_brightness_stats_detailed(
            roi_bgr=roi,
            background_brightness=threshold,
            roi_mask=roi_mask,
            roi_l_star=l_star_frame[y1:y2, x1:x2],
            morphological_kernel_size=settings.morphological_kernel_size,
            noise_floor_threshold=settings.noise_floor_threshold,
        )
        rois.append(RoiFrameAnalysis(roi_idx, bounds, stats, thresholded, mask_status))

    return FrameAnalysis(
        threshold=threshold,
        threshold_mode=resolve_threshold_mode(background_roi_idx, settings.manual_threshold),
        rois=rois,
    )


def build_threshold_mask(
    roi_l_star: np.ndarray,
    threshold: Optional[float],
    morphological_kernel_size: int,
) -> np.ndarray:
    """Fixed-mask rule: ``L* > threshold`` cleaned by a morphological opening.

    With no threshold the analysis uses the whole ROI, so the mask covers it.
    """
    if threshold is None:
        return np.ones(roi_l_star.shape, dtype=bool)
    mask = roi_l_star > threshold
    if np.any(mask):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphological_kernel_size, morphological_kernel_size),
        )
        mask_uint8 = mask.astype(np.uint8) * 255
        cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask = cleaned > 0
    return mask


def build_roi_mask(
    l_star_frame: np.ndarray,
    rect: RoiRect,
    threshold: Optional[float],
    morphological_kernel_size: int,
) -> Optional[np.ndarray]:
    """Capture a fixed mask for one ROI from a frame's L* channel.

    The mask is shaped exactly like the region :func:`analyze_frame` measures
    for ``rect``. Returns None when the ROI has no area inside the frame.
    """
    frame_height, frame_width = l_star_frame.shape[:2]
    x1, y1, x2, y2 = roi_slice_bounds(rect[0], rect[1], frame_width, frame_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return build_threshold_mask(l_star_frame[y1:y2, x1:x2], threshold, morphological_kernel_size)
