"""Decoder-position tracking, frame-copy and per-frame overhead tests for the main window."""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
import pytest
from PyQt5 import QtWidgets

from ecl_analysis.analysis.background import BackgroundComputationError
from ecl_analysis.video_analyzer import VideoAnalyzer


class CountingCapture:
    """In-memory VideoCapture stand-in that records seeks and reads."""

    def __init__(self, frames: List[np.ndarray], fail_reads_at: Optional[set] = None):
        self.frames = frames
        self.position = 0
        self.set_calls: List[int] = []
        self.read_calls = 0
        self.fail_reads_at = set(fail_reads_at or ())

    def isOpened(self) -> bool:
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_FPS:
            return 24.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frames[0].shape[1]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0]
        return 0.0

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.set_calls.append(int(value))
        self.position = int(value)
        return True

    def read(self):
        self.read_calls += 1
        index = self.position
        if index in self.fail_reads_at:
            self.fail_reads_at.discard(index)
            return False, None
        if 0 <= index < len(self.frames):
            self.position += 1
            return True, self.frames[index].copy()
        return False, None

    def release(self) -> None:
        return None


def _make_frames(count: int = 30) -> List[np.ndarray]:
    return [np.full((24, 32, 3), fill_value=idx, dtype=np.uint8) for idx in range(count)]


@pytest.fixture
def loaded_window(tmp_path, qt_application: QtWidgets.QApplication, monkeypatch):
    """Window with a CountingCapture-backed video loaded through load_video()."""
    frames = _make_frames()
    captures: List[CountingCapture] = []

    def _factory(path):
        capture = CountingCapture(frames)
        captures.append(capture)
        return capture

    video_file = tmp_path / "dummy.avi"
    video_file.write_bytes(b"stub")
    monkeypatch.setattr(cv2, "VideoCapture", _factory)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    window = VideoAnalyzer()
    window.video_path = str(video_file)
    window.load_video()
    assert window.total_frames == len(frames)

    yield window, frames, captures

    window.close()


def test_sequential_stepping_reads_without_seeking(loaded_window) -> None:
    window, frames, captures = loaded_window
    capture = captures[-1]

    for index in range(1, len(frames)):
        window.frame_slider.setValue(index)
        assert window.current_frame_index == index
        assert np.array_equal(window.frame, frames[index])

    assert capture.set_calls == []
    assert capture.read_calls == len(frames)  # first frame on load + one per step


def test_playback_timer_path_reads_without_seeking(loaded_window) -> None:
    window, frames, captures = loaded_window
    capture = captures[-1]

    window.is_playing = True
    for _ in range(10):
        window.advance_frame()
    window.stop_playback()

    assert window.current_frame_index == 10
    assert np.array_equal(window.frame, frames[10])
    assert capture.set_calls == []


def test_jumps_still_seek_and_sequential_reads_resume_afterwards(loaded_window) -> None:
    window, frames, captures = loaded_window
    capture = captures[-1]

    window.frame_slider.setValue(15)  # forward jump
    assert capture.set_calls == [15]
    assert np.array_equal(window.frame, frames[15])

    window.frame_slider.setValue(16)  # next frame: no seek
    assert capture.set_calls == [15]

    window.frame_slider.setValue(4)  # backward jump to an uncached frame
    assert capture.set_calls == [15, 4]
    assert np.array_equal(window.frame, frames[4])

    window.frame_slider.setValue(5)
    assert capture.set_calls == [15, 4]
    assert np.array_equal(window.frame, frames[5])


def test_cache_hits_do_not_disturb_tracked_decoder_position(loaded_window) -> None:
    window, frames, captures = loaded_window
    capture = captures[-1]

    window.frame_slider.setValue(1)
    window.frame_slider.setValue(2)
    window.frame_slider.setValue(0)  # cached: served without touching the decoder
    reads_before = capture.read_calls
    window.frame_slider.setValue(1)  # cached as well
    assert capture.read_calls == reads_before

    # Decoder is still positioned at frame 3, so 3 is read without a seek.
    window.frame_slider.setValue(3)
    assert capture.set_calls == []
    assert np.array_equal(window.frame, frames[3])


def test_failed_read_forces_seek_on_next_request(loaded_window) -> None:
    window, frames, captures = loaded_window
    capture = captures[-1]
    capture.fail_reads_at = {1}

    window.frame_slider.setValue(1)  # read fails, frame stays at 0
    assert window.current_frame_index == 0
    assert capture.set_calls == []

    window._seek_to_frame(1)  # retry: position is unknown, so seek explicitly
    assert capture.set_calls == [1]
    assert window.current_frame_index == 1
    assert np.array_equal(window.frame, frames[1])


def test_reload_and_replaced_capture_reset_decoder_tracking(loaded_window) -> None:
    window, frames, captures = loaded_window
    window.frame_slider.setValue(1)
    window.frame_slider.setValue(2)

    # Reloading creates a new capture positioned after frame 0.
    window.load_video()
    reloaded = captures[-1]
    assert reloaded is not captures[0]
    window.frame_slider.setValue(1)
    assert reloaded.set_calls == []
    assert np.array_equal(window.frame, frames[1])

    # A capture swapped in without load_video() must not inherit the old position.
    replacement = CountingCapture(frames)
    replacement.position = 17
    window.cap = replacement
    window.frame_cache.clear()
    window.frame_slider.setValue(2)
    assert replacement.set_calls == [2]
    assert np.array_equal(window.frame, frames[2])


def test_displayed_frame_is_shared_with_cache_but_never_mutated(loaded_window) -> None:
    window, frames, _captures = loaded_window
    window.frame_slider.setValue(3)
    window.rects = [((2, 2), (20, 20))]
    window.selected_rect_idx = 0
    window.manual_threshold = 0.0
    window.show_pixel_mask = True

    window.show_frame()
    window.show_pixel_mask = False
    window.show_frame()

    assert np.array_equal(window.frame, frames[3])
    assert np.array_equal(window.frame_cache.get(3), frames[3])


def test_show_frame_copies_when_overlay_falls_back_to_input(loaded_window, monkeypatch) -> None:
    window, frames, _captures = loaded_window
    window.rects = [((2, 2), (20, 20))]
    window.background_roi_idx = None
    window.show_pixel_mask = True

    def _fail(*args, **kwargs):
        raise BackgroundComputationError("boom")

    monkeypatch.setattr(window, "_preview_frame_analysis", _fail)
    window.show_frame()

    assert np.array_equal(window.frame, frames[0])
    assert np.array_equal(window.frame_cache.get(0), frames[0])


def test_video_info_static_part_is_computed_once_per_load(loaded_window, monkeypatch) -> None:
    window, frames, _captures = loaded_window
    size_calls = []
    real_getsize = os.path.getsize

    def _counting_getsize(path):
        size_calls.append(path)
        return real_getsize(path)

    monkeypatch.setattr(os.path, "getsize", _counting_getsize)

    for index in range(1, 10):
        window.frame_slider.setValue(index)
    assert size_calls == []

    info = window.video_info_label.text()
    assert "<b>File:</b> dummy.avi<br>" in info
    assert "<b>Resolution:</b> 32 × 24<br>" in info
    assert f"<b>Frames:</b> {len(frames)}<br>" in info
    assert "<b>FPS:</b> 24.00<br>" in info
    assert info.endswith(f"<b>Analysis Range:</b> 1-{len(frames)}")
    assert window.file_info_label.text() == "Loaded: dummy.avi"

    # The range part still tracks edits.
    window.current_frame_index = 4
    window.set_start_frame()
    assert window.video_info_label.text().endswith(f"<b>Analysis Range:</b> 5-{len(frames)}")

    # A reload recomputes the static part.
    window.load_video()
    assert len(size_calls) == 1


def test_sequential_stepping_matches_real_decoder_output(tmp_path, qt_application, monkeypatch) -> None:
    """Skipping cap.set() for sequential frames must yield the same frames as a plain decode."""
    video_path = str(tmp_path / "synthetic.avi")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (48, 32))
    if not writer.isOpened():
        pytest.skip("MJPG VideoWriter unavailable")
    rng = np.random.default_rng(1)
    for _ in range(12):
        writer.write(rng.integers(0, 255, size=(32, 48, 3), dtype=np.uint8))
    writer.release()

    reference_capture = cv2.VideoCapture(video_path)
    expected = []
    while True:
        ok, frame = reference_capture.read()
        if not ok:
            break
        expected.append(frame)
    reference_capture.release()
    if len(expected) < 12:
        pytest.skip("Synthetic video could not be decoded")

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)
    window = VideoAnalyzer()
    window.video_path = video_path
    window.load_video()

    for index in list(range(1, 12)) + [7, 2, 3, 11]:
        window.frame_slider.setValue(index)
        assert window.current_frame_index == index
        assert np.array_equal(window.frame, expected[index]), f"frame {index} differs"

    window.close()
