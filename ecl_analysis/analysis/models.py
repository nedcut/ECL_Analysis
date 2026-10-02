"""Data contracts for analysis requests and results."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[int, int]
RoiRect = Tuple[Point, Point]

# Threshold modes recorded in export metadata.
THRESHOLD_MODE_BACKGROUND_ROI = "background_roi"
THRESHOLD_MODE_MANUAL = "manual_threshold"
THRESHOLD_MODE_NONE = "none"

# Per-ROI fixed-mask outcomes recorded by the analysis worker.
MASK_STATUS_NOT_REQUESTED = "not_requested"  # fixed masks disabled for the run
MASK_STATUS_APPLIED = "applied"
MASK_STATUS_MISSING = "missing"  # fixed masks enabled but none captured for this ROI
MASK_STATUS_SHAPE_MISMATCH = "dropped_shape_mismatch"  # mask ignored; threshold path used


def resolve_threshold_mode(background_roi_idx: Optional[int], manual_threshold: float) -> str:
    """Which per-frame threshold/background value the analysis applies."""
    if background_roi_idx is not None:
        return THRESHOLD_MODE_BACKGROUND_ROI
    if manual_threshold > 0:
        return THRESHOLD_MODE_MANUAL
    return THRESHOLD_MODE_NONE


def has_analyzable_rois(rects: Sequence[RoiRect], background_roi_idx: Optional[int]) -> bool:
    """Return True if at least one ROI is not designated as the background ROI."""
    return any(i != background_roi_idx for i in range(len(rects)))


@dataclass(frozen=True)
class AnalysisRequest:
    """Immutable snapshot of all inputs required for frame analysis."""

    video_path: str
    rects: Sequence[RoiRect]
    background_roi_idx: Optional[int]
    start_frame: int
    end_frame: int
    use_fixed_mask: bool
    fixed_roi_masks: Sequence[Optional[np.ndarray]]
    background_percentile: float
    morphological_kernel_size: int
    noise_floor_threshold: float
    # Manual threshold mode: used when no background ROI is configured.
    # A value > 0 gates pixel inclusion and offsets background-subtracted
    # stats exactly like a background-derived threshold; 0 disables it.
    manual_threshold: float = 0.0

    @property
    def threshold_mode(self) -> str:
        """Which per-frame threshold/background value the worker applies."""
        return resolve_threshold_mode(self.background_roi_idx, self.manual_threshold)


@dataclass
class AnalysisResult:
    """Structured payload returned from frame analysis execution.

    ``brightness_*``/``blue_*``/``pixel_count_data`` are indexed by position in
    ``non_background_rois``. ``background_values_per_frame`` holds the
    background/threshold L* applied to each frame (0.0 when none was applied).
    """

    brightness_mean_data: List[List[float]]
    brightness_median_data: List[List[float]]
    blue_mean_data: List[List[float]]
    blue_median_data: List[List[float]]
    background_values_per_frame: List[float]
    frames_processed: int
    total_frames: int
    non_background_rois: List[int]
    elapsed_seconds: float
    start_frame: int
    end_frame: int
    truncated: bool = False
    # Number of pixels contributing to the exported L* stats, per ROI per frame.
    pixel_count_data: List[List[int]] = field(default_factory=list)
    # Video frame rate reported by the decoder; None when unavailable.
    fps: Optional[float] = None
    # Snapshot of the inputs that produced this result (for export provenance).
    request: Optional[AnalysisRequest] = None
    # Fixed-mask outcome per ROI index (see MASK_STATUS_* constants).
    mask_status: Dict[int, str] = field(default_factory=dict)
    # Set when the decoder did not land on start_frame after seeking.
    seek_warning: Optional[str] = None
