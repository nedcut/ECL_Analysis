import numpy as np

import cv2

from ecl_analysis.audio import AudioAnalyzer


def test_find_completion_beeps_converts_times_to_frames(monkeypatch):
    analyzer = AudioAnalyzer()
    analyzer.available = True

    monkeypatch.setattr(analyzer, "extract_audio_from_video", lambda path: (np.zeros(1000, dtype=np.float32), 44100))
    monkeypatch.setattr(analyzer, "detect_beeps", lambda *args, **kwargs: [0.0, 0.5, 1.0])

    class DummyCapture:
        def __init__(self):
            self.released = False

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 120
            return 0.0

        def release(self):
            self.released = True

    capture = DummyCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)

    results = analyzer.find_completion_beeps("dummy.mp4")

    assert results == [(0.0, 0), (0.5, 15), (1.0, 30)]


def test_find_completion_beeps_applies_duration_filter(monkeypatch):
    analyzer = AudioAnalyzer()
    analyzer.available = True

    monkeypatch.setattr(analyzer, "extract_audio_from_video", lambda path: (np.zeros(1000, dtype=np.float32), 44100))
    monkeypatch.setattr(analyzer, "detect_beeps", lambda *args, **kwargs: [0.2, 1.2, 2.0])

    class DummyCapture:
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 25.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 200
            return 0.0

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: DummyCapture())

    results = analyzer.find_completion_beeps("dummy.mp4", expected_run_duration=1.0)

    # Only beeps occurring after the expected duration should be retained
    assert results == [(1.2, 30), (2.0, 50)]
