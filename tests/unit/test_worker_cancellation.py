"""Tests for cooperative worker cancellation and thread shutdown helpers."""

from __future__ import annotations

import threading
from typing import List

import numpy as np
import pytest
from PyQt5 import QtCore

from ecl_analysis.audio import AudioAnalyzer
from ecl_analysis.video_analyzer import shutdown_worker_thread
from ecl_analysis.workers import (
    AnalysisWorker,
    AudioDetectionWorker,
    BrightestFrameWorker,
    CancellationToken,
    MaskScanRequest,
    PerRoiMaskCaptureWorker,
)
from ecl_analysis.analysis.models import AnalysisRequest


def _mask_scan_request() -> MaskScanRequest:
    return MaskScanRequest(
        video_path="missing.mp4",
        rects=[((0, 0), (4, 4))],
        background_roi_idx=None,
        start_frame=0,
        end_frame=0,
        step=1,
        background_percentile=50.0,
        morphological_kernel_size=3,
    )


def _analysis_request() -> AnalysisRequest:
    return AnalysisRequest(
        video_path="missing.mp4",
        rects=[((0, 0), (4, 4))],
        background_roi_idx=None,
        start_frame=0,
        end_frame=0,
        use_fixed_mask=False,
        fixed_roi_masks=[],
        background_percentile=50.0,
        morphological_kernel_size=3,
        noise_floor_threshold=0.0,
    )


class TestCancellationToken:
    def test_starts_uncancelled(self):
        token = CancellationToken()
        assert not token.is_cancelled()

    def test_cancel_sets_flag(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled()

    def test_cancel_from_other_thread_is_visible(self):
        token = CancellationToken()
        worker = threading.Thread(target=token.cancel)
        worker.start()
        worker.join(timeout=5.0)
        assert token.is_cancelled()


class TestWorkerCancelSlots:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: AnalysisWorker(_analysis_request()),
            lambda: AudioDetectionWorker("missing.mp4", 1.0),
            lambda: BrightestFrameWorker(_mask_scan_request()),
            lambda: PerRoiMaskCaptureWorker(_mask_scan_request()),
        ],
        ids=["analysis", "audio", "brightest", "per_roi"],
    )
    def test_cancel_sets_event_token(self, qt_application, factory):
        worker = factory()
        assert not worker._cancel_token.is_cancelled()
        worker.cancel()
        assert worker._cancel_token.is_cancelled()


class _StubAnalyzer:
    """Records calls made by AudioDetectionWorker.run."""

    instances: List["_StubAnalyzer"] = []

    def __init__(self):
        self.find_calls = []
        _StubAnalyzer.instances.append(self)

    def is_available(self) -> bool:
        return True

    def find_completion_beeps(self, video_path, expected_duration, cancel_check=None):
        self.find_calls.append((video_path, expected_duration, cancel_check))
        return [(1.0, 30)]


class TestAudioDetectionWorker:
    @pytest.fixture(autouse=True)
    def _stub_analyzer(self, monkeypatch):
        _StubAnalyzer.instances = []
        monkeypatch.setattr("ecl_analysis.workers.AudioAnalyzer", _StubAnalyzer)

    def test_cancel_before_run_skips_analysis(self, qt_application):
        worker = AudioDetectionWorker("video.mp4", 2.0)
        cancelled_signals = []
        finished_signals = []
        worker.cancelled.connect(lambda: cancelled_signals.append(True))
        worker.finished.connect(finished_signals.append)

        worker.cancel()
        worker.run()

        assert cancelled_signals == [True]
        assert finished_signals == []
        assert _StubAnalyzer.instances == []

    def test_run_passes_live_cancel_check_to_analyzer(self, qt_application):
        worker = AudioDetectionWorker("video.mp4", 2.0)
        finished_signals = []
        worker.finished.connect(finished_signals.append)

        worker.run()

        assert finished_signals == [[(1.0, 30)]]
        (analyzer,) = _StubAnalyzer.instances
        ((_, _, cancel_check),) = analyzer.find_calls
        assert cancel_check is not None
        assert cancel_check() is False
        worker.cancel()
        assert cancel_check() is True

    def test_cancel_during_analysis_emits_cancelled(self, qt_application, monkeypatch):
        worker = AudioDetectionWorker("video.mp4", 2.0)
        cancelled_signals = []
        finished_signals = []
        worker.cancelled.connect(lambda: cancelled_signals.append(True))
        worker.finished.connect(finished_signals.append)

        def cancel_mid_run(self, video_path, expected_duration, cancel_check=None):
            worker.cancel()
            return []

        monkeypatch.setattr(_StubAnalyzer, "find_completion_beeps", cancel_mid_run)
        worker.run()

        assert cancelled_signals == [True]
        assert finished_signals == []


class TestFindCompletionBeepsCheckpoints:
    def _analyzer(self, monkeypatch) -> AudioAnalyzer:
        analyzer = AudioAnalyzer()
        monkeypatch.setattr(analyzer, "_ensure_backend", lambda: True)
        return analyzer

    def test_cancel_before_extraction_returns_empty(self, monkeypatch):
        analyzer = self._analyzer(monkeypatch)
        monkeypatch.setattr(
            analyzer,
            "extract_audio_from_video",
            lambda path: pytest.fail("extraction ran despite cancellation"),
        )
        assert analyzer.find_completion_beeps("video.mp4", cancel_check=lambda: True) == []

    def test_cancel_after_extraction_skips_beep_detection(self, monkeypatch):
        analyzer = self._analyzer(monkeypatch)
        cancelled = {"value": False}

        def fake_extract(path):
            cancelled["value"] = True
            return np.zeros(8), 44100.0

        monkeypatch.setattr(analyzer, "extract_audio_from_video", fake_extract)
        monkeypatch.setattr(
            analyzer,
            "detect_beeps",
            lambda *a, **kw: pytest.fail("beep detection ran despite cancellation"),
        )

        result = analyzer.find_completion_beeps(
            "video.mp4", cancel_check=lambda: cancelled["value"]
        )
        assert result == []

    def test_cancel_after_beep_detection_skips_video_probe(self, monkeypatch):
        import cv2

        analyzer = self._analyzer(monkeypatch)
        cancelled = {"value": False}

        monkeypatch.setattr(
            analyzer, "extract_audio_from_video", lambda path: (np.zeros(8), 44100.0)
        )

        def fake_detect(*args, **kwargs):
            cancelled["value"] = True
            return [1.0]

        monkeypatch.setattr(analyzer, "detect_beeps", fake_detect)
        monkeypatch.setattr(
            cv2,
            "VideoCapture",
            lambda *a, **kw: pytest.fail("video probe ran despite cancellation"),
        )

        result = analyzer.find_completion_beeps(
            "video.mp4", cancel_check=lambda: cancelled["value"]
        )
        assert result == []

    def test_uncancelled_run_still_returns_results(self, monkeypatch):
        analyzer = self._analyzer(monkeypatch)
        monkeypatch.setattr(
            analyzer, "extract_audio_from_video", lambda path: (np.zeros(8), 44100.0)
        )
        monkeypatch.setattr(analyzer, "detect_beeps", lambda *a, **kw: [2.0])

        class _FakeCap:
            def get(self, prop):
                import cv2

                return 30.0 if prop == cv2.CAP_PROP_FPS else 300

            def release(self):
                pass

        import cv2

        monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: _FakeCap())

        result = analyzer.find_completion_beeps("video.mp4", cancel_check=lambda: False)
        assert result == [(2.0, 60)]


class _FakeThread:
    """Duck-typed QThread stand-in with scripted wait() results."""

    def __init__(self, wait_results):
        self._wait_results = list(wait_results)
        self.quit_called = False
        self.waits = []

    def quit(self):
        self.quit_called = True

    def wait(self, timeout):
        self.waits.append(timeout)
        return self._wait_results.pop(0)


class TestShutdownWorkerThread:
    def test_none_thread_counts_as_stopped(self):
        assert shutdown_worker_thread(None, "test") is True

    def test_stops_within_first_wait(self):
        thread = _FakeThread([True])
        assert shutdown_worker_thread(thread, "test") is True
        assert thread.quit_called
        assert thread.waits == [1500]

    def test_escalates_wait_before_giving_up(self):
        thread = _FakeThread([False, True])
        assert shutdown_worker_thread(thread, "test") is True
        assert thread.waits == [1500, 5000]

    def test_reports_still_running_thread(self):
        thread = _FakeThread([False, False])
        assert shutdown_worker_thread(thread, "test") is False
        assert thread.waits == [1500, 5000]

    def test_real_qthread_shutdown(self, qt_application):
        thread = QtCore.QThread()
        thread.start()
        assert shutdown_worker_thread(thread, "real") is True
        assert thread.isFinished()
