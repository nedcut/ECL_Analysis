"""UI-free brightest-frame and per-ROI mask capture scans.

Like :mod:`.runner`, these functions own loops that were previously embedded in
Qt workers so the desktop app and the web server share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .background import compute_background_brightness
from .brightness import compute_l_star_frame
from .models import RoiRect
from .runner import (
    AnalysisCancelled,
    AnalysisRunError,
    CancelCheck,
    MessageCallback,
    ProgressCallback,
    normalized_slice_bounds,
)


@dataclass(frozen=True)
class MaskScanRequest:
    """Immutable scan inputs for brightest-frame mask workflows."""

    video_path: str
    rects: Sequence[RoiRect]
    background_roi_idx: Optional[int]
    start_frame: int
    end_frame: int
    step: int
    background_percentile: float
    morphological_kernel_size: int


@dataclass
class BrightestFrameResult:
    """Result payload for global brightest frame detection."""

    brightest_frame_idx: int
    max_brightness: float


@dataclass
class PerRoiMaskCaptureResult:
    """Result payload for per-ROI mask capture."""

    masks: List[Optional[np.ndarray]]
    sources: List[Optional[int]]
    max_brightness: Dict[int, float]


def _check_cancelled(cancel_check: Optional[CancelCheck]) -> None:
    if cancel_check is not None and cancel_check():
        raise AnalysisCancelled()


def find_brightest_frame(
    request: MaskScanRequest,
    progress_callback: Optional[ProgressCallback] = None,
    message_callback: Optional[MessageCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> BrightestFrameResult:
    """Find the single frame with the highest mean L* across non-background ROIs."""
    req = request
    frame_indices = list(range(req.start_frame, req.end_frame + 1, max(1, req.step)))
    if not frame_indices:
        raise AnalysisRunError("No frames available for brightest-frame scan.")

    non_background_rois = [i for i in range(len(req.rects)) if i != req.background_roi_idx]
    if not non_background_rois:
        raise AnalysisRunError("No non-background ROI available for brightest-frame scan.")

    cap = cv2.VideoCapture(req.video_path)
    if not cap.isOpened():
        raise AnalysisRunError(f"Could not open video file: {req.video_path}")

    brightest_frame_idx = frame_indices[0]
    max_brightness = float("-inf")

    try:
        total = len(frame_indices)
        for idx, frame_idx in enumerate(frame_indices):
            _check_cancelled(cancel_check)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            l_star_frame = compute_l_star_frame(frame)
            frame_height, frame_width = frame.shape[:2]
            brightness_sum = 0.0
            roi_count = 0

            for roi_idx in non_background_rois:
                pt1, pt2 = req.rects[roi_idx]
                x1, y1, x2, y2 = normalized_slice_bounds(pt1, pt2, frame_width, frame_height)
                if x2 > x1 and y2 > y1:
                    roi_l_star = l_star_frame[y1:y2, x1:x2]
                    if roi_l_star.size:
                        brightness_sum += float(np.mean(roi_l_star))
                        roi_count += 1

            if roi_count > 0:
                frame_brightness = brightness_sum / roi_count
                if frame_brightness > max_brightness:
                    max_brightness = frame_brightness
                    brightest_frame_idx = frame_idx

            if progress_callback is not None:
                progress_callback(idx + 1, total)
            if message_callback is not None and ((idx + 1) % 10 == 0 or idx + 1 == total):
                message_callback(f"Scanning frame {idx + 1}/{total} for global brightest mask source")

        if max_brightness == float("-inf"):
            max_brightness = 0.0

        return BrightestFrameResult(
            brightest_frame_idx=brightest_frame_idx,
            max_brightness=max_brightness,
        )
    finally:
        cap.release()


def capture_per_roi_masks(
    request: MaskScanRequest,
    progress_callback: Optional[ProgressCallback] = None,
    message_callback: Optional[MessageCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> PerRoiMaskCaptureResult:
    """Find each ROI's brightest frame and capture an above-background mask there."""
    req = request
    roi_indices = [i for i in range(len(req.rects)) if i != req.background_roi_idx]
    if not roi_indices:
        raise AnalysisRunError("No non-background ROI available.")

    frame_indices = list(range(req.start_frame, req.end_frame + 1, max(1, req.step)))
    if not frame_indices:
        raise AnalysisRunError("No frames available for per-ROI scan.")

    cap = cv2.VideoCapture(req.video_path)
    if not cap.isOpened():
        raise AnalysisRunError(f"Could not open video file: {req.video_path}")

    brightest_frames: Dict[int, int] = {idx: frame_indices[0] for idx in roi_indices}
    max_brightness: Dict[int, float] = {idx: float("-inf") for idx in roi_indices}

    scan_total = len(frame_indices)
    total = scan_total + len(roi_indices)

    try:
        for idx, frame_idx in enumerate(frame_indices):
            _check_cancelled(cancel_check)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                if progress_callback is not None:
                    progress_callback(idx + 1, total)
                continue

            l_star_frame = compute_l_star_frame(frame)
            frame_height, frame_width = frame.shape[:2]

            for roi_idx in roi_indices:
                pt1, pt2 = req.rects[roi_idx]
                x1, y1, x2, y2 = normalized_slice_bounds(pt1, pt2, frame_width, frame_height)
                if x2 > x1 and y2 > y1:
                    roi_l_star = l_star_frame[y1:y2, x1:x2]
                    if roi_l_star.size:
                        roi_mean = float(np.mean(roi_l_star))
                        if roi_mean > max_brightness[roi_idx]:
                            max_brightness[roi_idx] = roi_mean
                            brightest_frames[roi_idx] = frame_idx

            if progress_callback is not None:
                progress_callback(idx + 1, total)
            if message_callback is not None and ((idx + 1) % 10 == 0 or idx + 1 == scan_total):
                message_callback(f"Scanning frame {idx + 1}/{scan_total} for per-ROI brightest sources")

        masks: List[Optional[np.ndarray]] = [None] * len(req.rects)
        sources: List[Optional[int]] = [None] * len(req.rects)

        for idx, roi_idx in enumerate(roi_indices):
            _check_cancelled(cancel_check)

            frame_idx = brightest_frames[roi_idx]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                if progress_callback is not None:
                    progress_callback(scan_total + idx + 1, total)
                continue

            l_star_frame = compute_l_star_frame(frame)
            background = compute_background_brightness(
                frame=frame,
                rects=req.rects,
                background_roi_idx=req.background_roi_idx,
                background_percentile=req.background_percentile,
                frame_l_star=l_star_frame,
            )

            frame_height, frame_width = frame.shape[:2]
            pt1, pt2 = req.rects[roi_idx]
            x1, y1, x2, y2 = normalized_slice_bounds(pt1, pt2, frame_width, frame_height)

            if x2 > x1 and y2 > y1:
                roi_l_star = l_star_frame[y1:y2, x1:x2]
                if background is not None:
                    mask = roi_l_star > background
                    if np.any(mask):
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (req.morphological_kernel_size, req.morphological_kernel_size),
                        )
                        mask_uint8 = mask.astype(np.uint8) * 255
                        cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
                        mask = cleaned > 0
                else:
                    mask = np.ones(roi_l_star.shape, dtype=bool)
                masks[roi_idx] = mask
                sources[roi_idx] = frame_idx

            if progress_callback is not None:
                progress_callback(scan_total + idx + 1, total)
            if message_callback is not None:
                message_callback(f"Capturing mask {idx + 1}/{len(roi_indices)} from frame {frame_idx}")

        for roi_idx, value in max_brightness.items():
            if value == float("-inf"):
                max_brightness[roi_idx] = 0.0

        return PerRoiMaskCaptureResult(
            masks=masks,
            sources=sources,
            max_brightness=max_brightness,
        )
    finally:
        cap.release()
