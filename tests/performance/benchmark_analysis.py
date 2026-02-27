"""Lightweight benchmark harness for analysis throughput and memory."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from ecl_analysis.analysis.brightness import compute_brightness_stats
from ecl_analysis.cache import FrameCache


def benchmark_brightness_stats(num_frames: int = 120, height: int = 720, width: int = 1280) -> float:
    """Return approximate frames/second for pure brightness stats over synthetic frames."""
    frames = np.random.randint(0, 255, size=(num_frames, height, width, 3), dtype=np.uint8)
    start = time.perf_counter()
    for frame in frames:
        _ = compute_brightness_stats(frame, morphological_kernel_size=3, noise_floor_threshold=0.0)
    elapsed = time.perf_counter() - start
    return num_frames / elapsed if elapsed > 0 else 0.0


def benchmark_cache_memory(max_size: int = 200, frame_shape=(480, 640, 3)) -> int:
    """Return peak allocated bytes while filling a frame cache."""
    cache = FrameCache(max_size=max_size)
    tracemalloc.start()
    for idx in range(max_size):
        frame = np.random.randint(0, 255, size=frame_shape, dtype=np.uint8)
        cache.put(idx, frame)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main():
    fps = benchmark_brightness_stats()
    peak_bytes = benchmark_cache_memory()
    print(f"Brightness stats throughput: {fps:.2f} frames/s")
    print(f"Cache fill peak memory: {peak_bytes / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
