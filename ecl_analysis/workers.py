"""Background workers for long-running analysis and scan tasks."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PyQt5 import QtCore

from .analysis.background import compute_background_brightness
from .analysis.brightness import compute_l_star_frame
from .analysis.models import AnalysisRequest, RoiRect
from .analysis.runner import (
    AnalysisCancelled,
    AnalysisRunError,
    normalized_slice_bounds as _normalized_slice_bounds,
    run_analysis,
)
from .audio import AudioAnalyzer


class CancellationToken:
    """Thread-safe cancellation flag shared between the GUI and worker threads.

    Backed by :class:`threading.Event`, so it is safe to call :meth:`cancel`
    from the GUI thread while a worker thread polls :meth:`is_cancelled`.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


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
        self._cancel_token = CancellationToken()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            result = run_analysis(
                self._request,
                progress_callback=self.progress_changed.emit,
                message_callback=self.progress_message.emit,
                cancel_check=self._cancel_token.is_cancelled,
            )
        except AnalysisCancelled:
            self.cancelled.emit()
        except AnalysisRunError as exc:
            self.error.emit(str(exc))
        except cv2.error as exc:
            self.error.emit(f"OpenCV error during analysis: {exc}")
        except Exception as exc:
            self.error.emit(str(exc))
        else:
            self.finished.emit(result)

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        """Request cooperative cancellation; safe to call from any thread."""
        self._cancel_token.cancel()


class AudioDetectionWorker(QtCore.QObject):
    """Extract completion beeps on a background thread."""

    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, video_path: str, expected_duration: float):
        super().__init__()
        self._video_path = video_path
        self._expected_duration = expected_duration
        self._cancel_token = CancellationToken()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        if self._cancel_token.is_cancelled():
            self.cancelled.emit()
            return

        analyzer = AudioAnalyzer()
        if not analyzer.is_available():
            self.error.emit("Audio analysis not available. Please install librosa and soundfile.")
            return

        beeps = analyzer.find_completion_beeps(
            self._video_path,
            self._expected_duration,
            cancel_check=self._cancel_token.is_cancelled,
        )
        if self._cancel_token.is_cancelled():
            self.cancelled.emit()
            return
        self.finished.emit(beeps)

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        """Request cooperative cancellation; safe to call from any thread."""
        self._cancel_token.cancel()


class BrightestFrameWorker(QtCore.QObject):
    """Find one brightest frame averaged across non-background ROIs."""

    progress_changed = QtCore.pyqtSignal(int, int)
    progress_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, request: MaskScanRequest):
        super().__init__()
        self._request = request
        self._cancel_token = CancellationToken()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        req = self._request
        frame_indices = list(range(req.start_frame, req.end_frame + 1, max(1, req.step)))
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

        brightest_frame_idx = frame_indices[0]
        max_brightness = float("-inf")

        try:
            total = len(frame_indices)
            for idx, frame_idx in enumerate(frame_indices):
                if self._cancel_token.is_cancelled():
                    self.cancelled.emit()
                    return

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
                    x1, y1, x2, y2 = _normalized_slice_bounds(pt1, pt2, frame_width, frame_height)
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

                self.progress_changed.emit(idx + 1, total)
                if (idx + 1) % 10 == 0 or idx + 1 == total:
                    self.progress_message.emit(
                        f"Scanning frame {idx + 1}/{total} for global brightest mask source"
                    )

            if max_brightness == float("-inf"):
                max_brightness = 0.0

            self.finished.emit(
                BrightestFrameResult(
                    brightest_frame_idx=brightest_frame_idx,
                    max_brightness=max_brightness,
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
        """Request cooperative cancellation; safe to call from any thread."""
        self._cancel_token.cancel()


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
        self._cancel_token = CancellationToken()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        req = self._request
        roi_indices = [i for i in range(len(req.rects)) if i != req.background_roi_idx]
        if not roi_indices:
            self.error.emit("No non-background ROI available.")
            return

        frame_indices = list(range(req.start_frame, req.end_frame + 1, max(1, req.step)))
        if not frame_indices:
            self.error.emit("No frames available for per-ROI scan.")
            return

        cap = cv2.VideoCapture(req.video_path)
        if not cap.isOpened():
            self.error.emit(f"Could not open video file: {req.video_path}")
            return

        brightest_frames: Dict[int, int] = {idx: frame_indices[0] for idx in roi_indices}
        max_brightness: Dict[int, float] = {idx: float("-inf") for idx in roi_indices}

        scan_total = len(frame_indices)
        total = scan_total + len(roi_indices)

        try:
            for idx, frame_idx in enumerate(frame_indices):
                if self._cancel_token.is_cancelled():
                    self.cancelled.emit()
                    return

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.progress_changed.emit(idx + 1, total)
                    continue

                l_star_frame = compute_l_star_frame(frame)
                frame_height, frame_width = frame.shape[:2]

                for roi_idx in roi_indices:
                    pt1, pt2 = req.rects[roi_idx]
                    x1, y1, x2, y2 = _normalized_slice_bounds(pt1, pt2, frame_width, frame_height)
                    if x2 > x1 and y2 > y1:
                        roi_l_star = l_star_frame[y1:y2, x1:x2]
                        if roi_l_star.size:
                            roi_mean = float(np.mean(roi_l_star))
                            if roi_mean > max_brightness[roi_idx]:
                                max_brightness[roi_idx] = roi_mean
                                brightest_frames[roi_idx] = frame_idx

                self.progress_changed.emit(idx + 1, total)
                if (idx + 1) % 10 == 0 or idx + 1 == scan_total:
                    self.progress_message.emit(
                        f"Scanning frame {idx + 1}/{scan_total} for per-ROI brightest sources"
                    )

            masks: List[Optional[np.ndarray]] = [None] * len(req.rects)
            sources: List[Optional[int]] = [None] * len(req.rects)

            for idx, roi_idx in enumerate(roi_indices):
                if self._cancel_token.is_cancelled():
                    self.cancelled.emit()
                    return

                frame_idx = brightest_frames[roi_idx]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.progress_changed.emit(scan_total + idx + 1, total)
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
                x1, y1, x2, y2 = _normalized_slice_bounds(pt1, pt2, frame_width, frame_height)

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

                self.progress_changed.emit(scan_total + idx + 1, total)
                self.progress_message.emit(
                    f"Capturing mask {idx + 1}/{len(roi_indices)} from frame {frame_idx}"
                )

            for roi_idx, value in max_brightness.items():
                if value == float("-inf"):
                    max_brightness[roi_idx] = 0.0

            self.finished.emit(
                PerRoiMaskCaptureResult(
                    masks=masks,
                    sources=sources,
                    max_brightness=max_brightness,
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
        """Request cooperative cancellation; safe to call from any thread."""
        self._cancel_token.cancel()
