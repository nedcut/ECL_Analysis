"""Mask scoring and consensus helpers for electrode-light analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .models import MaskCaptureMetadata

MASK_TOP_CANDIDATES = 3


@dataclass(frozen=True)
class MaskCandidate:
    """Single-frame mask candidate for one ROI."""

    frame_idx: int
    score: float
    background_brightness: Optional[float]
    mask: np.ndarray
    pixel_count: int
    signal_peak: float
    threshold_value: float
    min_component_area: int


def compute_min_component_area(
    mask_shape: Tuple[int, int],
    morphological_kernel_size: int,
) -> int:
    """Return a conservative connected-component floor scaled to ROI area."""
    roi_area = max(1, int(mask_shape[0] * mask_shape[1]))
    area_floor = int(round(roi_area * 0.002))
    return max(4, morphological_kernel_size, min(64, area_floor))


def filter_connected_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    """Remove connected components smaller than the requested area."""
    if mask.size == 0 or not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    filtered = np.zeros(mask.shape, dtype=bool)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            filtered |= labels == label_idx
    return filtered


def build_signal_mask(
    roi_l_star: np.ndarray,
    background_brightness: Optional[float],
    noise_floor_threshold: float,
    morphological_kernel_size: int,
    min_component_area: Optional[int] = None,
) -> Tuple[np.ndarray, float, int]:
    """Build a cleaned binary mask for pixels that plausibly belong to electrode light."""
    if roi_l_star.size == 0:
        return np.zeros(roi_l_star.shape, dtype=bool), float(noise_floor_threshold), 0

    threshold_value = float(noise_floor_threshold)
    if background_brightness is not None:
        threshold_value = max(threshold_value, float(background_brightness))

    mask = roi_l_star > threshold_value
    if np.any(mask):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morphological_kernel_size, morphological_kernel_size),
        )
        mask_uint8 = mask.astype(np.uint8) * 255
        cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask = cleaned > 0

    min_area = (
        compute_min_component_area(mask.shape, morphological_kernel_size)
        if min_component_area is None
        else max(1, int(min_component_area))
    )
    mask = filter_connected_components(mask, min_area)
    return mask, threshold_value, min_area


def evaluate_mask_candidate(
    roi_l_star: np.ndarray,
    background_brightness: Optional[float],
    noise_floor_threshold: float,
    morphological_kernel_size: int,
    frame_idx: int,
) -> Optional[MaskCandidate]:
    """Score a single-frame candidate using background-aware positive signal only."""
    mask, threshold_value, min_area = build_signal_mask(
        roi_l_star=roi_l_star,
        background_brightness=background_brightness,
        noise_floor_threshold=noise_floor_threshold,
        morphological_kernel_size=morphological_kernel_size,
    )
    if not np.any(mask):
        return None

    reference_value = float(background_brightness) if background_brightness is not None else threshold_value
    signal_values = np.maximum(roi_l_star[mask] - reference_value, 0.0)
    score = float(np.sum(signal_values))
    if score <= 0.0:
        return None

    return MaskCandidate(
        frame_idx=int(frame_idx),
        score=score,
        background_brightness=None if background_brightness is None else float(background_brightness),
        mask=mask,
        pixel_count=int(np.count_nonzero(mask)),
        signal_peak=float(np.max(signal_values)),
        threshold_value=float(threshold_value),
        min_component_area=int(min_area),
    )


def update_top_candidates(
    candidates: Sequence[MaskCandidate],
    candidate: Optional[MaskCandidate],
    limit: int = MASK_TOP_CANDIDATES,
) -> List[MaskCandidate]:
    """Return the top scored candidates with deterministic ordering."""
    if candidate is None:
        return list(candidates)

    ranked = list(candidates) + [candidate]
    ranked.sort(key=lambda item: (-item.score, item.frame_idx))
    return ranked[: max(1, int(limit))]


def _confidence_label(
    candidate_count: int,
    consensus_ratio: float,
    stability_ratio: float,
    pixel_count: int,
    min_component_area: int,
) -> str:
    if pixel_count <= 0:
        return "none"
    if candidate_count == 1 or consensus_ratio < 0.6 or stability_ratio < 0.4:
        return "low"
    if pixel_count <= (min_component_area * 2) or consensus_ratio < 0.8 or stability_ratio < 0.6:
        return "medium"
    return "high"


def build_consensus_mask(
    candidates: Sequence[MaskCandidate],
    capture_mode: str,
    noise_floor_threshold: float,
    morphological_kernel_size: int,
) -> Tuple[Optional[np.ndarray], MaskCaptureMetadata]:
    """Build a deterministic fixed mask from top candidates and summarize provenance."""
    ranked = sorted(candidates, key=lambda item: (-item.score, item.frame_idx))
    if not ranked:
        return (
            None,
            MaskCaptureMetadata(
                capture_mode=capture_mode,
                warnings=["no_signal"],
                noise_floor_threshold=float(noise_floor_threshold),
                morphological_kernel_size=int(morphological_kernel_size),
            ),
        )

    source_frames = [candidate.frame_idx for candidate in ranked]
    background_values = [
        0.0 if candidate.background_brightness is None else float(candidate.background_brightness)
        for candidate in ranked
    ]
    signal_scores = [float(candidate.score) for candidate in ranked]
    threshold_values = [float(candidate.threshold_value) for candidate in ranked]
    min_component_area = max(candidate.min_component_area for candidate in ranked)

    if len(ranked) == 1:
        consensus_mask = ranked[0].mask.copy()
        consensus_ratio = 1.0
        stability_ratio = 1.0
    else:
        mask_stack = np.stack([candidate.mask.astype(np.uint8) for candidate in ranked], axis=0)
        support_counts = np.sum(mask_stack, axis=0)
        required_votes = max(1, ceil(len(ranked) / 2))
        consensus_mask = support_counts >= required_votes
        consensus_mask = filter_connected_components(consensus_mask, min_component_area)
        if np.any(consensus_mask):
            consensus_ratio = float(np.mean(support_counts[consensus_mask] / len(ranked)))
        else:
            consensus_ratio = 0.0
        union_mask = np.any(mask_stack.astype(bool), axis=0)
        overlap = np.count_nonzero(consensus_mask)
        union = np.count_nonzero(union_mask)
        stability_ratio = float(overlap / union) if union else 0.0

    warnings: List[str] = []
    if len(ranked) == 1:
        warnings.append("single_frame_capture")
    if not np.any(consensus_mask):
        warnings.append("low_consensus")
    if np.count_nonzero(consensus_mask) <= min_component_area:
        warnings.append("small_mask")
    if consensus_ratio and consensus_ratio < 0.7:
        warnings.append("low_consensus")
    if stability_ratio and stability_ratio < 0.5:
        warnings.append("unstable_mask")

    confidence_label = _confidence_label(
        candidate_count=len(ranked),
        consensus_ratio=consensus_ratio,
        stability_ratio=stability_ratio,
        pixel_count=int(np.count_nonzero(consensus_mask)),
        min_component_area=min_component_area,
    )

    metadata = MaskCaptureMetadata(
        capture_mode=capture_mode,
        source_frames=source_frames,
        primary_source_frame=ranked[0].frame_idx,
        background_values=background_values,
        signal_scores=signal_scores,
        threshold_values=threshold_values,
        pixel_count=int(np.count_nonzero(consensus_mask)),
        consensus_ratio=float(consensus_ratio),
        stability_ratio=float(stability_ratio),
        confidence_label=confidence_label,
        min_component_area=int(min_component_area),
        warnings=sorted(set(warnings)),
        noise_floor_threshold=float(noise_floor_threshold),
        morphological_kernel_size=int(morphological_kernel_size),
    )
    if not np.any(consensus_mask):
        return None, metadata
    return consensus_mask.astype(bool), metadata
