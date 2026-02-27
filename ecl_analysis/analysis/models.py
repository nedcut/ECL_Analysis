"""Data contracts for analysis requests and results."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[int, int]
RoiRect = Tuple[Point, Point]


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


@dataclass
class AnalysisResult:
    """Structured payload returned from frame analysis execution."""

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
