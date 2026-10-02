from __future__ import annotations

import time

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ecl_analysis.video_analyzer import VideoAnalyzer


class _StubCapture:
    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        return None


class _ImmediateMaskWorker(QtCore.QObject):
    """Mask-worker stand-in that finishes as soon as its thread starts."""

    progress_changed = QtCore.pyqtSignal(int, int)
    progress_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.cancel_calls = 0

    @QtCore.pyqtSlot()
    def run(self) -> None:
        self.finished.emit(None)

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        self.cancel_calls += 1


def _wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met before timeout")
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)


def _loaded_window() -> VideoAnalyzer:
    window = VideoAnalyzer()
    window.cap = _StubCapture()
    window.frame = np.zeros((32, 32, 3), dtype=np.uint8)
    window.total_frames = 10
    window.rects = [((0, 0), (8, 8))]
    return window


def test_successful_task_does_not_trigger_cancellation(
    qt_application: QtWidgets.QApplication,
) -> None:
    """Closing the progress dialog on success must not fire its canceled signal."""
    window = _loaded_window()
    worker = _ImmediateMaskWorker()

    window._start_mask_worker(worker, "Working...", "Test Task", task_type="global")
    _wait_until(lambda: window._mask_thread is None)

    assert worker.cancel_calls == 0
    assert window.statusBar().currentMessage() != "Cancelling background task..."
    assert window.results_label.text() != "Cancellation requested..."
    assert window._mask_progress is None
    assert not window._analysis_in_progress

    window.close()


def test_user_cancel_from_progress_dialog_still_cancels(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = _loaded_window()
    worker = _ImmediateMaskWorker()
    window._mask_worker = worker
    progress = window._create_progress_dialog("Working...", "Test Task", 0, 100)

    progress.show()
    progress.findChild(QtWidgets.QPushButton).click()

    assert worker.cancel_calls == 1
    assert window.results_label.text() == "Cancellation requested..."

    window._mask_worker = None
    window._close_progress_dialog(progress)
    window.close()


class _StuckThread:
    """QThread stand-in whose shutdown is reported as failed."""

    def __init__(self) -> None:
        self.running = True

    def isRunning(self) -> bool:
        return self.running


def test_stalled_thread_is_parked_not_dropped(
    qt_application: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    window = _loaded_window()
    stuck_thread = _StuckThread()
    worker = _ImmediateMaskWorker()
    worker.finished.connect(window._on_global_brightest_finished)
    window._mask_thread = stuck_thread
    window._mask_worker = worker
    window._mask_task_type = "global"
    window._set_busy_state(True)
    monkeypatch.setattr(
        "ecl_analysis.video_analyzer.shutdown_worker_thread",
        lambda thread, name, *args, **kwargs: thread is None or not thread.isRunning(),
    )

    window._cleanup_mask_worker()

    # The running thread stays referenced, the UI is free for the next task,
    # and the stalled worker was told to stop.
    assert window._stalled_workers == [(stuck_thread, worker)]
    assert window._mask_thread is None and window._mask_worker is None
    assert not window._analysis_in_progress
    assert worker.cancel_calls == 1

    # A late result from the stalled worker must not reach the UI handlers.
    window.results_label.setText("sentinel")
    worker.finished.emit(None)
    assert window.results_label.text() == "sentinel"

    # Closing waits for the parked thread...
    assert window.close() is False
    stuck_thread.running = False
    # ...and proceeds (dropping the reference) once it has stopped.
    assert window.close() is True
    assert window._stalled_workers == []
