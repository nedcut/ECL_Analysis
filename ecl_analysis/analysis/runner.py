"""Frame analysis shared by desktop workers and the local web API.

The audited frame loop retains contributing counts, decoder timing, request
provenance, mask outcomes and seek warnings for every frontend.
"""

from __future__ import annotations
import logging
import math
import time
from typing import Callable, Dict, List, Optional
import cv2
from .frame import FrameAnalysisSettings, analyze_frame
from .models import (
    MASK_STATUS_MISSING,
    MASK_STATUS_NOT_REQUESTED,
    MASK_STATUS_SHAPE_MISMATCH,
    AnalysisRequest,
    AnalysisResult,
)


class AnalysisCancelled(Exception):
    """The caller cancelled a run before completion."""


class AnalysisRunError(RuntimeError):
    """A run could not open or read its input."""


def run_analysis(
    req: AnalysisRequest,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    message_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> AnalysisResult:
    progress = progress_callback or (lambda *args: None)
    message = message_callback or (lambda *args: None)
    cancelled = cancel_check or (lambda: False)
    total_frames = req.end_frame - req.start_frame + 1
    non_background_rois = [
        i for i in range(len(req.rects)) if i != req.background_roi_idx
    ]
    brightness_mean_data = [[] for _ in non_background_rois]
    brightness_median_data = [[] for _ in non_background_rois]
    blue_mean_data = [[] for _ in non_background_rois]
    blue_median_data = [[] for _ in non_background_rois]
    pixel_count_data: List[List[int]] = [[] for _ in non_background_rois]
    background_values_per_frame: List[float] = []
    mask_status: Dict[int, str] = {}
    settings = FrameAnalysisSettings.from_request(req)
    start_time = time.time()
    cap = cv2.VideoCapture(req.video_path)
    if not cap.isOpened():
        raise AnalysisRunError(f"Could not open video file: {req.video_path}")
    try:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        video_fps = (
            float(raw_fps)
            if raw_fps and math.isfinite(raw_fps) and (raw_fps > 0)
            else None
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, req.start_frame)
        seek_warning = None
        reported_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if (
            not math.isfinite(reported_pos)
            or int(round(reported_pos)) != req.start_frame
        ):
            seek_warning = f"Seek to frame {req.start_frame + 1} (0-based index {req.start_frame}) reported decoder position {reported_pos}; exported frame numbers may be offset from the decoded frames."
            logging.warning("AnalysisWorker: %s", seek_warning)
        frames_processed = 0
        truncated = False
        for _frame_idx in range(req.start_frame, req.end_frame + 1):
            if cancelled():
                raise AnalysisCancelled()
            ret, frame = cap.read()
            if not ret:
                if frames_processed == 0:
                    raise AnalysisRunError(
                        "Failed to read first frame during analysis."
                    )
                brightness_mean_data = [
                    lst[:frames_processed] for lst in brightness_mean_data
                ]
                brightness_median_data = [
                    lst[:frames_processed] for lst in brightness_median_data
                ]
                blue_mean_data = [lst[:frames_processed] for lst in blue_mean_data]
                blue_median_data = [lst[:frames_processed] for lst in blue_median_data]
                pixel_count_data = [lst[:frames_processed] for lst in pixel_count_data]
                truncated = True
                break
            analysis = analyze_frame(
                frame,
                req.rects,
                req.background_roi_idx,
                settings,
                masks=req.fixed_roi_masks,
            )
            background_value = analysis.threshold
            background_values_per_frame.append(
                background_value if background_value is not None else 0.0
            )
            for data_idx, roi_result in enumerate(analysis.rois):
                roi_idx = roi_result.roi_idx
                if (
                    roi_result.mask_status == MASK_STATUS_SHAPE_MISMATCH
                    and roi_idx not in mask_status
                ):
                    x1, y1, x2, y2 = roi_result.bounds
                    logging.warning(
                        "AnalysisWorker: fixed mask for ROI %d has shape %s but ROI is %s; mask dropped, threshold method used instead.",
                        roi_idx + 1,
                        req.fixed_roi_masks[roi_idx].shape[:2],
                        (y2 - y1, x2 - x1),
                    )
                if roi_result.mask_status is not None:
                    mask_status.setdefault(roi_idx, roi_result.mask_status)
                brightness_mean_data[data_idx].append(roi_result.mean)
                brightness_median_data[data_idx].append(roi_result.median)
                blue_mean_data[data_idx].append(roi_result.blue_mean)
                blue_median_data[data_idx].append(roi_result.blue_median)
                pixel_count_data[data_idx].append(roi_result.pixel_count)
            frames_processed += 1
            if frames_processed % 10 == 0:
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed if elapsed > 0 else 0.0
                remaining = total_frames - frames_processed
                eta_seconds = remaining / fps if fps > 0 else 0.0
                message(
                    f"Analyzing frame {frames_processed}/{total_frames} • Speed: {fps:.1f} fps • ETA: {eta_seconds:.0f}s"
                )
            progress(frames_processed, total_frames)
        else:
            frames_processed = total_frames
        if cancelled():
            raise AnalysisCancelled()
        elapsed_seconds = time.time() - start_time
        for roi_idx in non_background_rois:
            mask_status.setdefault(
                roi_idx,
                MASK_STATUS_MISSING
                if req.use_fixed_mask
                else MASK_STATUS_NOT_REQUESTED,
            )
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
            pixel_count_data=pixel_count_data,
            fps=video_fps,
            request=req,
            mask_status=mask_status,
            seek_warning=seek_warning,
        )
    finally:
        cap.release()
