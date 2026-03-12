"""Data contracts for analysis requests and results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[int, int]
RoiRect = Tuple[Point, Point]


@dataclass
class MaskCaptureMetadata:
    """Structured provenance for a captured fixed mask."""

    capture_mode: str
    source_frames: List[int] = field(default_factory=list)
    primary_source_frame: Optional[int] = None
    background_values: List[float] = field(default_factory=list)
    signal_scores: List[float] = field(default_factory=list)
    threshold_values: List[float] = field(default_factory=list)
    pixel_count: int = 0
    consensus_ratio: float = 0.0
    stability_ratio: float = 0.0
    confidence_label: str = "none"
    min_component_area: int = 0
    warnings: List[str] = field(default_factory=list)
    noise_floor_threshold: float = 0.0
    morphological_kernel_size: int = 0

    def clone(self) -> "MaskCaptureMetadata":
        """Return a detached copy for history snapshots and worker payloads."""
        return MaskCaptureMetadata(
            capture_mode=str(self.capture_mode),
            source_frames=[int(frame) for frame in self.source_frames],
            primary_source_frame=None if self.primary_source_frame is None else int(self.primary_source_frame),
            background_values=[float(value) for value in self.background_values],
            signal_scores=[float(value) for value in self.signal_scores],
            threshold_values=[float(value) for value in self.threshold_values],
            pixel_count=int(self.pixel_count),
            consensus_ratio=float(self.consensus_ratio),
            stability_ratio=float(self.stability_ratio),
            confidence_label=str(self.confidence_label),
            min_component_area=int(self.min_component_area),
            warnings=[str(value) for value in self.warnings],
            noise_floor_threshold=float(self.noise_floor_threshold),
            morphological_kernel_size=int(self.morphological_kernel_size),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize mask provenance for JSON export."""
        return {
            "capture_mode": self.capture_mode,
            "source_frames": list(self.source_frames),
            "primary_source_frame": self.primary_source_frame,
            "background_values": list(self.background_values),
            "signal_scores": list(self.signal_scores),
            "threshold_values": list(self.threshold_values),
            "pixel_count": self.pixel_count,
            "consensus_ratio": self.consensus_ratio,
            "stability_ratio": self.stability_ratio,
            "confidence_label": self.confidence_label,
            "min_component_area": self.min_component_area,
            "warnings": list(self.warnings),
            "noise_floor_threshold": self.noise_floor_threshold,
            "morphological_kernel_size": self.morphological_kernel_size,
        }


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
    mask_metadata: Sequence[Optional[MaskCaptureMetadata]] = field(default_factory=list)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


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
    use_fixed_mask: bool = False
    mask_metadata: List[Optional[MaskCaptureMetadata]] = field(default_factory=list)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
