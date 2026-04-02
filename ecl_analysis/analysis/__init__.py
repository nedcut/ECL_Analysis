"""Pure analysis helpers for brightness and background calculations."""

from .background import compute_background_brightness
from .brightness import compute_brightness, compute_brightness_stats, compute_l_star_frame
from .duration import validate_run_duration
from .masking import MASK_TOP_CANDIDATES, build_consensus_mask, build_signal_mask, evaluate_mask_candidate
from .models import AnalysisRequest, AnalysisResult, MaskCaptureMetadata

__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "MaskCaptureMetadata",
    "MASK_TOP_CANDIDATES",
    "build_consensus_mask",
    "build_signal_mask",
    "compute_background_brightness",
    "compute_brightness",
    "compute_brightness_stats",
    "compute_l_star_frame",
    "evaluate_mask_candidate",
    "validate_run_duration",
]
