"""UI-free execution of frame analysis requests.

This module owns the core frame loop so that both the PyQt worker
(:class:`ecl_analysis.workers.AnalysisWorker`) and the local web API can run
identical analyses. It has no Qt or HTTP dependencies: callers provide plain
callables for progress reporting and cancellation checks.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .background import compute_background_brightness
from .brightness import compute_brightness_stats, compute_l_star_frame
from .models import AnalysisRequest, AnalysisResult

ProgressCallback = Callable[[int, int], None]
MessageCallback = Callable[[str], None]
CancelCheck = Callable[[], bool]


class AnalysisCancelled(Exception):
    """Raised when a cancel check reports True mid-run."""


class AnalysisRunError(RuntimeError):
    """Raised when an analysis run cannot start or produce valid results."""


def normalized_slice_bounds(
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    """Return clamped, normalized ROI bounds for NumPy slicing (exclusive end)."""
    left, right = sorted((int(pt1[0]), int(pt2[0])))
    top, bottom = sorted((int(pt1[1]), int(pt2[1])))

    x1 = max(0, min(left, frame_width))
    x2 = max(0, min(right, frame_width))
    y1 = max(0, min(top, frame_height))
    y2 = max(0, min(bottom, frame_height))

    return x1, y1, x2, y2


def run_analysis(
    request: AnalysisRequest,
    progress_callback: Optional[ProgressCallback] = None,
    message_callback: Optional[MessageCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> AnalysisResult:
    """Execute a frame analysis request and return its result.

    Raises AnalysisRunError when the video cannot be opened or the first frame
    fails to read, and AnalysisCancelled when ``cancel_check`` returns True.
    Computation errors (e.g. cv2.error) propagate to the caller so that a
    faulty run is never silently converted into fabricated measurements.
    """
    req = request
    total_frames = req.end_frame - req.start_frame + 1
    non_background_rois = [i for i in range(len(req.rects)) if i != req.background_roi_idx]

    brightness_mean_data: List[List[float]] = [[] for _ in non_background_rois]
    brightness_median_data: List[List[float]] = [[] for _ in non_background_rois]
    blue_mean_data: List[List[float]] = [[] for _ in non_background_rois]
    blue_median_data: List[List[float]] = [[] for _ in non_background_rois]
    background_values_per_frame: List[float] = []

    def cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    start_time = time.time()
    cap = cv2.VideoCapture(req.video_path)
    if not cap.isOpened():
        raise AnalysisRunError(f"Could not open video file: {req.video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, req.start_frame)
        frames_processed = 0
        truncated = False

        for _frame_idx in range(req.start_frame, req.end_frame + 1):
            if cancelled():
                raise AnalysisCancelled()

            ret, frame = cap.read()
            if not ret:
                if frames_processed == 0:
                    raise AnalysisRunError("Failed to read first frame during analysis.")
                brightness_mean_data = [lst[:frames_processed] for lst in brightness_mean_data]
                brightness_median_data = [lst[:frames_processed] for lst in brightness_median_data]
                blue_mean_data = [lst[:frames_processed] for lst in blue_mean_data]
                blue_median_data = [lst[:frames_processed] for lst in blue_median_data]
                truncated = True
                break

            l_star_frame = compute_l_star_frame(frame)
            background_value = compute_background_brightness(
                frame=frame,
                rects=req.rects,
                background_roi_idx=req.background_roi_idx,
                background_percentile=req.background_percentile,
                frame_l_star=l_star_frame,
            )
            if req.background_roi_idx is None and req.manual_threshold > 0:
                # Manual threshold mode: no background ROI configured, so the
                # user-set manual threshold acts as the active threshold.
                background_value = req.manual_threshold
            background_values_per_frame.append(background_value if background_value is not None else 0.0)

            frame_height, frame_width = frame.shape[:2]
            for data_idx, roi_idx in enumerate(non_background_rois):
                pt1, pt2 = req.rects[roi_idx]
                x1, y1, x2, y2 = normalized_slice_bounds(pt1, pt2, frame_width, frame_height)

                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    roi_l_star = l_star_frame[y1:y2, x1:x2]
                    roi_mask = None
                    if req.use_fixed_mask and roi_idx < len(req.fixed_roi_masks):
                        candidate_mask = req.fixed_roi_masks[roi_idx]
                        if isinstance(candidate_mask, np.ndarray) and candidate_mask.shape[:2] == roi.shape[:2]:
                            roi_mask = candidate_mask

                    (
                        l_raw_mean,
                        l_raw_median,
                        l_bg_sub_mean,
                        l_bg_sub_median,
                        b_raw_mean,
                        b_raw_median,
                        b_bg_sub_mean,
                        b_bg_sub_median,
                    ) = compute_brightness_stats(
                        roi_bgr=roi,
                        background_brightness=background_value,
                        roi_mask=roi_mask,
                        roi_l_star=roi_l_star,
                        morphological_kernel_size=req.morphological_kernel_size,
                        noise_floor_threshold=req.noise_floor_threshold,
                    )

                    if background_value is not None:
                        brightness_mean_data[data_idx].append(l_bg_sub_mean)
                        brightness_median_data[data_idx].append(l_bg_sub_median)
                        blue_mean_data[data_idx].append(b_bg_sub_mean)
                        blue_median_data[data_idx].append(b_bg_sub_median)
                    else:
                        brightness_mean_data[data_idx].append(l_raw_mean)
                        brightness_median_data[data_idx].append(l_raw_median)
                        blue_mean_data[data_idx].append(b_raw_mean)
                        blue_median_data[data_idx].append(b_raw_median)
                else:
                    brightness_mean_data[data_idx].append(0.0)
                    brightness_median_data[data_idx].append(0.0)
                    blue_mean_data[data_idx].append(0.0)
                    blue_median_data[data_idx].append(0.0)

            frames_processed += 1

            if message_callback is not None and frames_processed % 10 == 0:
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed if elapsed > 0 else 0.0
                remaining = total_frames - frames_processed
                eta_seconds = remaining / fps if fps > 0 else 0.0
                message_callback(
                    f"Analyzing frame {frames_processed}/{total_frames} • "
                    f"Speed: {fps:.1f} fps • ETA: {eta_seconds:.0f}s"
                )

            if progress_callback is not None:
                progress_callback(frames_processed, total_frames)
        else:
            frames_processed = total_frames

        if cancelled():
            raise AnalysisCancelled()

        elapsed_seconds = time.time() - start_time
        return AnalysisResult(
            brightness_mean_data=brightness_mean_data,
            brightness_median_data=brightness_median_data,
            blue_mean_data=blue_mean_data,
            blue_median_data=blue_median_data,
            background_values_per_frame=background_values_per_frame,
            frames_processed=frames_processed,
            total_frames=total_frames,
            non_background_rois=non_background_rois,
            elapsed_seconds=elapsed_seconds,
            start_frame=req.start_frame,
            end_frame=req.end_frame,
            truncated=truncated,
        )
    finally:
        cap.release()
