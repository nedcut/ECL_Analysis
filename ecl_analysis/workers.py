"""Background workers for long-running analysis and scan tasks."""

from __future__ import annotations

import threading

import cv2
from PyQt5 import QtCore

from .analysis.models import AnalysisRequest
from .analysis.runner import AnalysisCancelled, AnalysisRunError, run_analysis
from .analysis.scans import (
    BrightestFrameResult,
    MaskScanRequest,
    PerRoiMaskCaptureResult,
    capture_per_roi_masks,
    find_brightest_frame,
)
from .audio import AudioAnalyzer

__all__ = [
    "AnalysisWorker",
    "AudioDetectionWorker",
    "BrightestFrameResult",
    "BrightestFrameWorker",
    "CancellationToken",
    "MaskScanRequest",
    "PerRoiMaskCaptureResult",
    "PerRoiMaskCaptureWorker",
]


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
        try:
            result = find_brightest_frame(
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
            self.error.emit(f"OpenCV error during brightest-frame scan: {exc}")
        except Exception as exc:
            self.error.emit(str(exc))
        else:
            self.finished.emit(result)

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
        try:
            result = capture_per_roi_masks(
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
            self.error.emit(f"OpenCV error during per-ROI scan: {exc}")
        except Exception as exc:
            self.error.emit(str(exc))
        else:
            self.finished.emit(result)

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        """Request cooperative cancellation; safe to call from any thread."""
        self._cancel_token.cancel()
