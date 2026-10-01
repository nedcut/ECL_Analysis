"""Benchmark playback-like sequential frame stepping through the main window.

Writes a synthetic video with ``cv2.VideoWriter`` into a temporary directory,
loads it into an offscreen ``VideoAnalyzer`` and steps through every frame the
same way the playback timer does (``frame_slider.setValue(next_frame)``).

Run with::

    QT_QPA_PLATFORM=offscreen python -m tests.performance.benchmark_playback
"""

from __future__ import annotations

import os
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5 import QtWidgets

from ecl_analysis.video_analyzer import VideoAnalyzer


class _CountingCapture:
    """Wrap a real VideoCapture and count seek/read calls."""

    def __init__(self, capture):
        self._capture = capture
        self.set_calls = 0
        self.read_calls = 0

    def set(self, prop, value):
        self.set_calls += 1
        return self._capture.set(prop, value)

    def read(self):
        self.read_calls += 1
        return self._capture.read()

    def __getattr__(self, name):
        return getattr(self._capture, name)


def write_synthetic_video(path: str, num_frames: int, width: int, height: int, fps: float = 30.0) -> None:
    """Write a moving-gradient test video to ``path``."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open an mp4v writer")
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    for idx in range(num_frames):
        writer.write(np.roll(base, idx * 4, axis=1))
    writer.release()


def benchmark_sequential_stepping(
    num_frames: int = 300,
    width: int = 1280,
    height: int = 720,
    with_rois: bool = True,
) -> dict:
    """Return timing and decoder-call statistics for stepping through a video."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    original_capture = cv2.VideoCapture
    holder = {}

    def _capture_factory(*args, **kwargs):
        holder["cap"] = _CountingCapture(original_capture(*args, **kwargs))
        return holder["cap"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "synthetic.mp4")
        write_synthetic_video(video_path, num_frames, width, height)

        cv2.VideoCapture = _capture_factory
        try:
            window = VideoAnalyzer()
            window.video_path = video_path
            window.load_video()
            if with_rois:
                window.rects = [((100, 100), (400, 400)), ((600, 200), (900, 500))]
                window.update_rect_list()
            # Count only the stepping phase, not loading.
            cap = holder["cap"]
            cap.set_calls = 0
            cap.read_calls = 0

            start = time.perf_counter()
            for idx in range(1, window.total_frames):
                window.frame_slider.setValue(idx)
                app.processEvents()
            elapsed = time.perf_counter() - start
            stepped = window.total_frames - 1
            window.close()
        finally:
            cv2.VideoCapture = original_capture

    return {
        "frames": stepped,
        "elapsed_s": elapsed,
        "ms_per_frame": 1000.0 * elapsed / stepped if stepped else 0.0,
        "fps": stepped / elapsed if elapsed > 0 else 0.0,
        "set_calls": cap.set_calls,
        "read_calls": cap.read_calls,
    }


def main():
    for with_rois in (False, True):
        stats = benchmark_sequential_stepping(with_rois=with_rois)
        label = "2 ROIs" if with_rois else "no ROIs"
        print(
            f"Sequential stepping 1280x720 ({label}): {stats['frames']} frames in {stats['elapsed_s']:.2f}s "
            f"-> {stats['ms_per_frame']:.2f} ms/frame ({stats['fps']:.1f} fps); "
            f"cap.set calls={stats['set_calls']}, cap.read calls={stats['read_calls']}"
        )


if __name__ == "__main__":
    main()
