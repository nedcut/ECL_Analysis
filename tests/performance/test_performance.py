import time

import numpy as np
import pytest

from ecl_analysis.video_analyzer import VideoAnalyzer


@pytest.mark.performance
def test_brightness_stats_handles_hd_frame_under_budget(video_analyzer_factory):
    analyzer: VideoAnalyzer = video_analyzer_factory()
    roi = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)

    start = time.perf_counter()
    stats = analyzer._compute_brightness_stats(roi)
    duration = time.perf_counter() - start

    assert duration < 3.0, f"Brightness stats took too long: {duration:.2f}s"
    assert all(isinstance(value, float) for value in stats)
