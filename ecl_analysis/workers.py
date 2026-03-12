"""Background workers for long-running analysis and scan tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PyQt5 import QtCore

from .analysis.background import compute_background_brightness
from .analysis.brightness import compute_brightness_stats, compute_l_star_frame
from .analysis.masking import (
    MASK_TOP_CANDIDATES,
    build_consensus_mask,
    evaluate_mask_candidate,
    update_top_candidates,
)
from .analysis.models import AnalysisRequest, AnalysisResult, MaskCaptureMetadata, RoiRect
from .audio import AudioAnalyzer


def _normalized_slice_bounds(
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Return clamped, normalized ROI bounds for NumPy slicing (exclusive end)."""
    left, right = sorted((int(pt1[0]), int(pt2[0])))
    top, bottom = sorted((int(pt1[1]), int(pt2[1])))

    x1 = max(0, min(left, frame_width))
    x2 = max(0, min(right, frame_width))
    y1 = max(0, min(top, frame_height))
    y2 = max(0, min(bottom, frame_height))

    return x1, y1, x2, y2


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
    noise_floor_threshold: float


@dataclass
class BrightestFrameResult:
    """Result payload for global auto-capture."""

    masks: List[Optional[np.ndarray]]
    sources: List[Optional[int]]
    metadata: List[Optional[MaskCaptureMetadata]]
    candidate_frames: List[int]
    max_signal_score: float


@dataclass
class PerRoiMaskCaptureResult:
    """Result payload for per-ROI mask capture."""

    masks: List[Optional[np.ndarray]]
    sources: List[Optional[int]]
    max_brightness: Dict[int, float]
    metadata: List[Optional[MaskCaptureMetadata]]


class AnalysisWorker(QtCore.QObject):
    """Execute frame analysis outside the UI thread."""

    progress_changed = QtCore.pyqtSignal(int, int)
    progress_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, request: AnalysisRequest):
        super().__init__()
        self._request = request
        self._cancelled = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        req = self._request
        total_frames = req.end_frame - req.start_frame + 1
        non_background_rois = [i for i in range(len(req.rects)) if i != req.background_roi_idx]

        brightness_mean_data = [[] for _ in non_background_rois]
        brightness_median_data = [[] for _ in non_background_rois]
        blue_mean_data = [[] for _ in non_background_rois]
        blue_median_data = [[] for _ in non_background_rois]
        background_values_per_frame: List[float] = []

        start_time = time.time()
        cap = cv2.VideoCapture(req.video_path)
        if not cap.isOpened():
            self.error.emit(f"Could not open video file: {req.video_path}")
            return

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, req.start_frame)
            frames_processed = 0

            for _frame_idx in range(req.start_frame, req.end_frame + 1):
                if self._cancelled:
                    self.cancelled.emit()
                    return

                ret, frame = cap.read()
                if not ret:
                    if frames_processed == 0:
                        self.error.emit("Failed to read first frame during analysis.")
                        return
                    brightness_mean_data = [lst[:frames_processed] for lst in brightness_mean_data]
                    brightness_median_data = [lst[:frames_processed] for lst in brightness_median_data]
                    blue_mean_data = [lst[:frames_processed] for lst in blue_mean_data]
                    blue_median_data = [lst[:frames_processed] for lst in blue_median_data]
                    break

                l_star_frame = compute_l_star_frame(frame)
                background_value = compute_background_brightness(
                    frame=frame,
                    rects=req.rects,
                    background_roi_idx=req.background_roi_idx,
                    background_percentile=req.background_percentile,
                    frame_l_star=l_star_frame,
                )
                background_values_per_frame.append(background_value if background_value is not None else 0.0)

                frame_height, frame_width = frame.shape[:2]
                for data_idx, roi_idx in enumerate(non_background_rois):
                    pt1, pt2 = req.rects[roi_idx]
                    x1, y1, x2, y2 = _normalized_slice_bounds(pt1, pt2, frame_width, frame_height)

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

                if frames_processed % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frames_processed / elapsed if elapsed > 0 else 0.0
                    remaining = total_frames - frames_processed
                    eta_seconds = remaining / fps if fps > 0 else 0.0
                    self.progress_message.emit(
                        f"Analyzing frame {frames_processed}/{total_frames} • "
                        f"Speed: {fps:.1f} fps • ETA: {eta_seconds:.0f}s"
                    )

                self.progress_changed.emit(frames_processed, total_frames)
            else:
                frames_processed = total_frames

            if self._cancelled:
                self.cancelled.emit()
                return

            elapsed_seconds = time.time() - start_time
            self.finished.emit(
                AnalysisResult(
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
                    use_fixed_mask=req.use_fixed_mask,
                    mask_metadata=[
                        metadata.clone() if isinstance(metadata, MaskCaptureMetadata) else None
                        for metadata in req.mask_metadata
                    ],
                    analysis_metadata=dict(req.analysis_metadata),
                )
            )
        except cv2.error as exc:
            self.error.emit(f"OpenCV error during analysis: {exc}")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            cap.release()

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        self._cancelled = True


class AudioDetectionWorker(QtCore.QObject):
    """Extract completion beeps on a background thread."""

    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, video_path: str, expected_duration: float):
        super().__init__()
        self._video_path = video_path
        self._expected_duration = expected_duration
        self._cancelled = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        analyzer = AudioAnalyzer()
        if not analyzer.is_available():
            self.error.emit("Audio analysis not available. Please install librosa and soundfile.")
            return

        beeps = analyzer.find_completion_beeps(self._video_path, self._expected_duration)
        if self._cancelled:
            self.cancelled.emit()
            return
        self.finished.emit(beeps)

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        self._cancelled = True


class BrightestFrameWorker(QtCore.QObject):
    """Capture fixed masks from the strongest shared signal frames."""

    progress_changed = QtCore.pyqtSignal(int, int)
    progress_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, request: MaskScanRequest):
        super().__init__()
        self._request = request
        self._cancelled = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        req = self._request
        frame_indices = list(range(req.start_frame, req.end_frame + 1))
        if not frame_indices:
            self.error.emit("No frames available for brightest-frame scan.")
            return

        non_background_rois = [i for i in range(len(req.rects)) if i != req.background_roi_idx]
        if not non_background_rois:
            self.error.emit("No non-background ROI available for brightest-frame scan.")
            return

        cap = cv2.VideoCapture(req.video_path)
        if not cap.isOpened():
            self.error.emit(f"Could not open video file: {req.video_path}")
            return

        top_global_frames: List[tuple[float, int]] = []

        try:
            total = len(frame_indices)
            for idx, frame_idx in enumerate(frame_indices):
                if self._cancelled:
                    self.cancelled.emit()
                    return

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                l_star_frame = compute_l_star_frame(frame)
                background_value = compute_background_brightness(
                    frame=frame,
                    rects=req.rects,
                    background_roi_idx=req.background_roi_idx,
                    background_percentile=req.background_percentile,
                    frame_l_star=l_star_frame,
                )
                frame_height, frame_width = frame.shape[:2]
                frame_signal_score = 0.0

                for roi_idx in non_background_rois:
                    pt1, pt2 = req.rects[roi_idx]
                    x1, y1, x2, y2 = _normalized_slice_bounds(pt1, pt2, frame_width, frame_height)
                    if x2 > x1 and y2 > y1:
                        roi_l_star = l_star_frame[y1:y2, x1:x2]
                        candidate = evaluate_mask_candidate(
                            roi_l_star=roi_l_star,
                            background_brightness=background_value,
                            noise_floor_threshold=req.noise_floor_threshold,
                            morphological_kernel_size=req.morphological_kernel_size,
                            frame_idx=frame_idx,
                        )
                        if candidate is not None:
                            frame_signal_score += candidate.score

                if frame_signal_score > 0.0:
                    top_global_frames.append((float(frame_signal_score), int(frame_idx)))
                    top_global_frames.sort(key=lambda item: (-item[0], item[1]))
                    del top_global_frames[MASK_TOP_CANDIDATES:]

                self.progress_changed.emit(idx + 1, total)
                if (idx + 1) % 10 == 0 or idx + 1 == total:
                    self.progress_message.emit(
                        f"Scanning frame {idx + 1}/{total} for global brightest mask source"
                    )

            candidate_frames = [frame_idx for _score, frame_idx in top_global_frames]
            per_roi_candidates: Dict[int, List] = {roi_idx: [] for roi_idx in non_background_rois}

            for frame_idx in candidate_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                l_star_frame = compute_l_star_frame(frame)
                background_value = compute_background_brightness(
                    frame=frame,
                    rects=req.rects,
                    background_roi_idx=req.background_roi_idx,
                    background_percentile=req.background_percentile,
                    frame_l_star=l_star_frame,
                )
                frame_height, frame_width = frame.shape[:2]

                for roi_idx in non_background_rois:
                    pt1, pt2 = req.rects[roi_idx]
                    x1, y1, x2, y2 = _normalized_slice_bounds(pt1, pt2, frame_width, frame_height)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    roi_l_star = l_star_frame[y1:y2, x1:x2]
                    candidate = evaluate_mask_candidate(
                        roi_l_star=roi_l_star,
                        background_brightness=background_value,
                        noise_floor_threshold=req.noise_floor_threshold,
                        morphological_kernel_size=req.morphological_kernel_size,
                        frame_idx=frame_idx,
                    )
                    per_roi_candidates[roi_idx] = update_top_candidates(per_roi_candidates[roi_idx], candidate)

            masks: List[Optional[np.ndarray]] = [None] * len(req.rects)
            sources: List[Optional[int]] = [None] * len(req.rects)
            metadata: List[Optional[MaskCaptureMetadata]] = [None] * len(req.rects)
            for roi_idx in non_background_rois:
                mask, mask_metadata = build_consensus_mask(
                    candidates=per_roi_candidates[roi_idx],
                    capture_mode="global_auto",
                    noise_floor_threshold=req.noise_floor_threshold,
                    morphological_kernel_size=req.morphological_kernel_size,
                )
                masks[roi_idx] = mask
                sources[roi_idx] = mask_metadata.primary_source_frame
                metadata[roi_idx] = mask_metadata

            self.finished.emit(
                BrightestFrameResult(
                    masks=masks,
                    sources=sources,
                    metadata=metadata,
                    candidate_frames=candidate_frames,
                    max_signal_score=float(top_global_frames[0][0]) if top_global_frames else 0.0,
                )
            )
        except cv2.error as exc:
            self.error.emit(f"OpenCV error during brightest-frame scan: {exc}")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            cap.release()

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        self._cancelled = True


class PerRoiMaskCaptureWorker(QtCore.QObject):
    """Find brightest frame per ROI and capture masks from those frames."""

    progress_changed = QtCore.pyqtSignal(int, int)
    progress_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, request: MaskScanRequest):
        super().__init__()
        self._request = request
        self._cancelled = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        req = self._request
        roi_indices = [i for i in range(len(req.rects)) if i != req.background_roi_idx]
        if not roi_indices:
            self.error.emit("No non-background ROI available.")
            return

        frame_indices = list(range(req.start_frame, req.end_frame + 1))
        if not frame_indices:
            self.error.emit("No frames available for per-ROI scan.")
            return

        cap = cv2.VideoCapture(req.video_path)
        if not cap.isOpened():
            self.error.emit(f"Could not open video file: {req.video_path}")
            return

        top_candidates: Dict[int, List] = {idx: [] for idx in roi_indices}
        max_brightness: Dict[int, float] = {idx: 0.0 for idx in roi_indices}

        scan_total = len(frame_indices)
        total = scan_total

        try:
            for idx, frame_idx in enumerate(frame_indices):
                if self._cancelled:
                    self.cancelled.emit()
                    return

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.progress_changed.emit(idx + 1, total)
                    continue

                l_star_frame = compute_l_star_frame(frame)
                background_value = compute_background_brightness(
                    frame=frame,
                    rects=req.rects,
                    background_roi_idx=req.background_roi_idx,
                    background_percentile=req.background_percentile,
                    frame_l_star=l_star_frame,
                )
                frame_height, frame_width = frame.shape[:2]

                for roi_idx in roi_indices:
                    pt1, pt2 = req.rects[roi_idx]
                    x1, y1, x2, y2 = _normalized_slice_bounds(pt1, pt2, frame_width, frame_height)
                    if x2 > x1 and y2 > y1:
                        roi_l_star = l_star_frame[y1:y2, x1:x2]
                        candidate = evaluate_mask_candidate(
                            roi_l_star=roi_l_star,
                            background_brightness=background_value,
                            noise_floor_threshold=req.noise_floor_threshold,
                            morphological_kernel_size=req.morphological_kernel_size,
                            frame_idx=frame_idx,
                        )
                        top_candidates[roi_idx] = update_top_candidates(top_candidates[roi_idx], candidate)
                        if candidate is not None:
                            max_brightness[roi_idx] = max(max_brightness[roi_idx], candidate.score)

                self.progress_changed.emit(idx + 1, total)
                if (idx + 1) % 10 == 0 or idx + 1 == scan_total:
                    self.progress_message.emit(
                        f"Scanning frame {idx + 1}/{scan_total} for per-ROI brightest sources"
                    )

            masks: List[Optional[np.ndarray]] = [None] * len(req.rects)
            sources: List[Optional[int]] = [None] * len(req.rects)
            metadata: List[Optional[MaskCaptureMetadata]] = [None] * len(req.rects)

            for roi_idx in roi_indices:
                mask, mask_metadata = build_consensus_mask(
                    candidates=top_candidates[roi_idx],
                    capture_mode="per_roi_auto",
                    noise_floor_threshold=req.noise_floor_threshold,
                    morphological_kernel_size=req.morphological_kernel_size,
                )
                masks[roi_idx] = mask
                sources[roi_idx] = mask_metadata.primary_source_frame
                metadata[roi_idx] = mask_metadata

            self.finished.emit(
                PerRoiMaskCaptureResult(
                    masks=masks,
                    sources=sources,
                    max_brightness=max_brightness,
                    metadata=metadata,
                )
            )
        except cv2.error as exc:
            self.error.emit(f"OpenCV error during per-ROI scan: {exc}")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            cap.release()

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        self._cancelled = True
