"""Beep detection on synthetic audio (requires librosa)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("librosa")

from ecl_analysis.audio import AudioAnalyzer  # noqa: E402

SAMPLE_RATE = 44100
# Long enough that a 0.3s event is under 5% of the clip, as in real recordings.
CLIP_SECONDS = 10.0
BEEP_HZ = 7000.0


def _noise(sigma: float, seconds: float = CLIP_SECONDS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SAMPLE_RATE * seconds)) * sigma).astype(np.float32)


def _add_tone(audio: np.ndarray, start_s: float, end_s: float, amplitude: float) -> np.ndarray:
    out = audio.copy()
    start, end = int(start_s * SAMPLE_RATE), int(end_s * SAMPLE_RATE)
    t = np.arange(end - start) / SAMPLE_RATE
    out[start:end] += (amplitude * np.sin(2 * np.pi * BEEP_HZ * t)).astype(np.float32)
    return out


@pytest.fixture(scope="module")
def analyzer() -> AudioAnalyzer:
    analyzer = AudioAnalyzer()
    assert analyzer.is_available()
    return analyzer


def _percentile_only(analyzer: AudioAnalyzer, audio: np.ndarray):
    """The pre-fix behaviour: threshold = 95th percentile of the clip itself."""
    return analyzer.detect_beeps(audio, SAMPLE_RATE, min_prominence_ratio=0.0, min_amplitude=0.0)


def test_silent_clip_has_no_beeps(analyzer: AudioAnalyzer) -> None:
    silence = np.zeros(int(SAMPLE_RATE * CLIP_SECONDS), dtype=np.float32)
    assert analyzer.detect_beeps(silence, SAMPLE_RATE) == []


def test_near_silent_clip_with_inaudible_tone_has_no_beeps(analyzer: AudioAnalyzer) -> None:
    # -100 dBFS: far below anything audible, but it is the clip's loudest content.
    audio = _add_tone(_noise(1e-7), 2.0, 2.3, amplitude=1e-5)
    assert _percentile_only(analyzer, audio), "percentile-only threshold should fire here"
    assert analyzer.detect_beeps(audio, SAMPLE_RATE) == []


def test_single_clear_beep_is_found(analyzer: AudioAnalyzer) -> None:
    audio = _add_tone(_noise(0.01), 2.0, 2.3, amplitude=0.3)

    beeps = analyzer.detect_beeps(audio, SAMPLE_RATE)

    assert len(beeps) == 1
    assert beeps[0] == pytest.approx(2.15, abs=0.05)


def test_noise_only_clip_has_no_beeps(analyzer: AudioAnalyzer) -> None:
    # Broadband noise that gets 4x louder for 0.3s (e.g. a bump or a cough). That is
    # under 5% of the clip, so the whole swell tops the 95th percentile.
    audio = _noise(0.01)
    swell = slice(int(2.0 * SAMPLE_RATE), int(2.3 * SAMPLE_RATE))
    audio[swell] *= 4.0
    assert _percentile_only(analyzer, audio), "percentile-only threshold should fire here"

    assert analyzer.detect_beeps(audio, SAMPLE_RATE) == []


def test_unfiltered_fallback_is_flagged_and_logged(
    analyzer: AudioAnalyzer, monkeypatch, caplog
) -> None:
    import cv2

    monkeypatch.setattr(
        analyzer, "extract_audio_from_video", lambda path: (np.zeros(8, dtype=np.float32), SAMPLE_RATE)
    )
    monkeypatch.setattr(analyzer, "detect_beeps", lambda *a, **kw: [1.0, 2.0])

    class _Cap:
        def get(self, prop):
            return 25.0 if prop == cv2.CAP_PROP_FPS else 250

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: _Cap())

    with caplog.at_level(logging.WARNING):
        results = analyzer.find_completion_beeps("video.mp4", expected_run_duration=5.0)

    assert results == [(1.0, 25), (2.0, 50)]
    assert analyzer.last_results_unfiltered is True
    assert "unfiltered" in caplog.text

    results = analyzer.find_completion_beeps("video.mp4", expected_run_duration=1.5)
    assert results == [(2.0, 50)]
    assert analyzer.last_results_unfiltered is False
