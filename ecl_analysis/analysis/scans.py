"""Audited brightest-frame and per-ROI mask scans shared by both frontends."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence
import cv2
import numpy as np
from ..roi_geometry import roi_slice_bounds
from .brightness import compute_l_star_frame
from .frame import build_roi_mask, resolve_frame_threshold
from .models import RoiRect
from .runner import AnalysisCancelled, AnalysisRunError


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
    # Manual threshold mode (no background ROI): masks gate on L* > this value,
    # matching "Capture From Current" and the analysis; 0 disables it.
    manual_threshold: float = 0.0


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


def find_brightest_frame(
    req: MaskScanRequest,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    message_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> BrightestFrameResult:
    progress = progress_callback or (lambda *args: None)
    message = message_callback or (lambda *args: None)
    cancelled = cancel_check or (lambda: False)
    frame_indices = list(range(req.start_frame, req.end_frame + 1, max(1, req.step)))
    if not frame_indices:
        raise AnalysisRunError("No frames available for brightest-frame scan.")
    non_background_rois = [
        i for i in range(len(req.rects)) if i != req.background_roi_idx
    ]
    if not non_background_rois:
        raise AnalysisRunError(
            "No non-background ROI available for brightest-frame scan."
        )
    cap = cv2.VideoCapture(req.video_path)
    if not cap.isOpened():
        raise AnalysisRunError(f"Could not open video file: {req.video_path}")
    brightest_frame_idx = frame_indices[0]
    max_brightness = float("-inf")
    try:
        total = len(frame_indices)
        for idx, frame_idx in enumerate(frame_indices):
            if cancelled():
                raise AnalysisCancelled()
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
                x1, y1, x2, y2 = roi_slice_bounds(pt1, pt2, frame_width, frame_height)
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
            progress(idx + 1, total)
            if (idx + 1) % 10 == 0 or idx + 1 == total:
                message(
                    f"Scanning frame {idx + 1}/{total} for global brightest mask source"
                )
        if max_brightness == float("-inf"):
            max_brightness = 0.0
        return BrightestFrameResult(
            brightest_frame_idx=brightest_frame_idx, max_brightness=max_brightness
        )
    finally:
        cap.release()


def capture_per_roi_masks(
    req: MaskScanRequest,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    message_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> PerRoiMaskCaptureResult:
    progress = progress_callback or (lambda *args: None)
    message = message_callback or (lambda *args: None)
    cancelled = cancel_check or (lambda: False)
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
            if cancelled():
                raise AnalysisCancelled()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                progress(idx + 1, total)
                continue
            l_star_frame = compute_l_star_frame(frame)
            frame_height, frame_width = frame.shape[:2]
            for roi_idx in roi_indices:
                pt1, pt2 = req.rects[roi_idx]
                x1, y1, x2, y2 = roi_slice_bounds(pt1, pt2, frame_width, frame_height)
                if x2 > x1 and y2 > y1:
                    roi_l_star = l_star_frame[y1:y2, x1:x2]
                    if roi_l_star.size:
                        roi_mean = float(np.mean(roi_l_star))
                        if roi_mean > max_brightness[roi_idx]:
                            max_brightness[roi_idx] = roi_mean
                            brightest_frames[roi_idx] = frame_idx
            progress(idx + 1, total)
            if (idx + 1) % 10 == 0 or idx + 1 == scan_total:
                message(
                    f"Scanning frame {idx + 1}/{scan_total} for per-ROI brightest sources"
                )
        masks: List[Optional[np.ndarray]] = [None] * len(req.rects)
        sources: List[Optional[int]] = [None] * len(req.rects)
        for idx, roi_idx in enumerate(roi_indices):
            if cancelled():
                raise AnalysisCancelled()
            frame_idx = brightest_frames[roi_idx]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                progress(scan_total + idx + 1, total)
                continue
            l_star_frame = compute_l_star_frame(frame)
            threshold = resolve_frame_threshold(
                frame,
                req.rects,
                req.background_roi_idx,
                req.background_percentile,
                req.manual_threshold,
                frame_l_star=l_star_frame,
            )
            mask = build_roi_mask(
                l_star_frame,
                req.rects[roi_idx],
                threshold,
                req.morphological_kernel_size,
            )
            if mask is not None:
                masks[roi_idx] = mask
                sources[roi_idx] = frame_idx
            progress(scan_total + idx + 1, total)
            message(
                f"Capturing mask {idx + 1}/{len(roi_indices)} from frame {frame_idx}"
            )
        for roi_idx, value in max_brightness.items():
            if value == float("-inf"):
                max_brightness[roi_idx] = 0.0
        return PerRoiMaskCaptureResult(
            masks=masks, sources=sources, max_brightness=max_brightness
        )
    finally:
        cap.release()
