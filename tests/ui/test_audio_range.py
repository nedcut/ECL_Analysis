from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt5 import QtWidgets

from ecl_analysis.video_analyzer import VideoAnalyzer


class _StubCapture:
    def __init__(self, fps: float) -> None:
        self.fps = fps

    def isOpened(self) -> bool:
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def release(self) -> None:
        return None


@pytest.fixture
def window(qt_application: QtWidgets.QApplication, monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"stub")
    win = VideoAnalyzer()
    win.cap = _StubCapture(fps=25.0)
    win.video_path = str(video)
    win.frame = np.zeros((32, 32, 3), dtype=np.uint8)
    win.total_frames = 1000
    win.playback_fps = 25.0
    win.start_frame = 0
    win.end_frame = 999
    win.frame_slider.setRange(0, 999)
    win.frame_spinbox.setRange(1, 1000)
    win._seek_to_frame = lambda frame_index: setattr(win, "current_frame_index", frame_index)
    win.run_detected_calls = []
    monkeypatch.setattr(win.audio_manager, "play_run_detected", lambda: win.run_detected_calls.append(True))
    # The range must come from the already-open capture, not a second one.
    monkeypatch.setattr(
        cv2, "VideoCapture", lambda *a, **kw: pytest.fail("opened a second VideoCapture")
    )
    yield win
    win.cap = None
    win.close()


def test_audio_window_spans_exactly_the_expected_duration(window: VideoAnalyzer) -> None:
    window._apply_audio_detection_results([(20.0, 500)], expected_duration=10.0)

    assert window.end_frame == 500
    assert window.start_frame == 251
    assert window.end_frame - window.start_frame + 1 == 250  # 10.0s at 25 fps
    assert window.current_frame_index == 251
    assert window.run_detected_calls == [True]
    assert [entry.label for entry in window._undo_history] == ["Auto-Detect Analysis Window"]


def test_audio_window_is_clamped_to_video_start(window: VideoAnalyzer) -> None:
    # Beep at 6s with a 10s expected run: only 151 frames exist before the beep.
    window._apply_audio_detection_results([(6.0, 150)], expected_duration=10.0)

    assert window.start_frame == 0
    assert window.end_frame == 150
    # 151 frames is 6.04s, 40% short of 10s -> outside the shared 20% tolerance.
    assert window.run_detected_calls == []


def test_audio_window_within_shared_tolerance_plays_run_detected(window: VideoAnalyzer) -> None:
    # 8.5s of footage before the beep: 15% short, inside the 20% tolerance of
    # analysis/duration.py (the old local check used 10%).
    window._apply_audio_detection_results([(8.5, 212)], expected_duration=10.0)

    assert window.start_frame == 0
    assert window.end_frame == 212
    assert window.run_detected_calls == [True]
