"""Background workers for long-running analysis tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore

from .analysis_core import (
    compute_background_brightness,
    compute_brightness_stats,
    compute_l_star_frame,
)
from .audio import AudioAnalyzer

Rect = Tuple[Tuple[int, int], Tuple[int, int]]


@dataclass
class AnalysisRequest:
    video_path: str
    rects: Sequence[Rect]
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
    brightness_mean_data: List[List[float]]
    brightness_median_data: List[List[float]]
    blue_mean_data: List[List[float]]
    blue_median_data: List[List[float]]
    background_values_per_frame: List[float]
    frames_processed: int
    total_frames: int
    non_background_rois: List[int]
    elapsed_seconds: float


class AnalysisWorker(QtCore.QObject):
    """Execute frame analysis outside the UI thread."""

    progress_changed = QtCore.pyqtSignal(int, int)
    progress_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(AnalysisResult)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, request: AnalysisRequest):
        super().__init__()
        self._request = request
        self._cancelled = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        """Process frames and emit results when done."""
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

            for frame_idx in range(req.start_frame, req.end_frame + 1):
                if self._cancelled:
                    self.cancelled.emit()
                    return

                ret, frame = cap.read()
                if not ret:
                    if frames_processed == 0:
                        self.error.emit("Failed to read first frame during analysis.")
                        return
                    # Truncate data to processed frame count
                    brightness_mean_data = [lst[:frames_processed] for lst in brightness_mean_data]
                    brightness_median_data = [lst[:frames_processed] for lst in brightness_median_data]
                    blue_mean_data = [lst[:frames_processed] for lst in blue_mean_data]
                    blue_median_data = [lst[:frames_processed] for lst in blue_median_data]
                    break

                l_star_frame = compute_l_star_frame(frame)
                background_value = compute_background_brightness(
                    frame,
                    req.rects,
                    req.background_roi_idx,
                    req.background_percentile,
                    frame_l_star=l_star_frame,
                )
                background_values_per_frame.append(background_value if background_value is not None else 0.0)

                frame_height, frame_width = frame.shape[:2]
                for data_idx, roi_idx in enumerate(non_background_rois):
                    pt1, pt2 = req.rects[roi_idx]
                    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
                    x2, y2 = min(frame_width - 1, pt2[0]), min(frame_height - 1, pt2[1])

                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2]
                        roi_l_star = l_star_frame[y1:y2, x1:x2]
                        roi_mask = None
                        if req.use_fixed_mask and roi_idx < len(req.fixed_roi_masks):
                            potential_mask = req.fixed_roi_masks[roi_idx]
                            if isinstance(potential_mask, np.ndarray) and potential_mask.shape[:2] == roi.shape[:2]:
                                roi_mask = potential_mask

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
                            roi,
                            req.morphological_kernel_size,
                            req.noise_floor_threshold,
                            background_value,
                            roi_mask,
                            roi_l_star=roi_l_star,
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
                # Completed all frames
                frames_processed = total_frames

            if self._cancelled:
                self.cancelled.emit()
                return

            elapsed_seconds = time.time() - start_time
            result = AnalysisResult(
                brightness_mean_data=brightness_mean_data,
                brightness_median_data=brightness_median_data,
                blue_mean_data=blue_mean_data,
                blue_median_data=blue_median_data,
                background_values_per_frame=background_values_per_frame,
                frames_processed=frames_processed,
                total_frames=total_frames,
                non_background_rois=non_background_rois,
                elapsed_seconds=elapsed_seconds,
            )
            self.finished.emit(result)

        except cv2.error as exc:
            self.error.emit(f"OpenCV error during analysis: {exc}")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            cap.release()

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        """Signal the worker to abort processing."""
        self._cancelled = True


class AudioDetectionWorker(QtCore.QObject):
    """Extract beeps from audio on a background thread."""

    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()
    status = QtCore.pyqtSignal(str)

    def __init__(self, video_path: str, expected_duration: float):
        super().__init__()
        self._video_path = video_path
        self._expected_duration = expected_duration
        self._cancelled = False

    @QtCore.pyqtSlot()
    def run(self) -> None:
        analyzer = AudioAnalyzer()
        if not analyzer.available:
            self.error.emit("Audio analysis not available. Please install librosa and soundfile.")
            return

        self.status.emit("Extracting audio samples…")
        beeps = analyzer.find_completion_beeps(self._video_path, self._expected_duration)

        if self._cancelled:
            self.cancelled.emit()
            return

        self.finished.emit(beeps)

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        self._cancelled = True
