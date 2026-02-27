"""Shared pytest fixtures for the Brightness Sorcerer test suite."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict

import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from ecl_analysis.cache import FrameCache
from ecl_analysis.constants import FRAME_CACHE_SIZE
from ecl_analysis.video_analyzer import VideoAnalyzer


@pytest.fixture(scope="session")
def qt_application() -> QtWidgets.QApplication:
    """Provide a QApplication instance configured for offscreen rendering."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
        created = True
    else:
        created = False

    yield app

    if created:
        app.quit()


@pytest.fixture
def video_analyzer_factory() -> Callable[..., VideoAnalyzer]:
    """Factory that constructs lightweight VideoAnalyzer instances for algorithm tests."""

    def _factory(**overrides: Dict[str, Any]) -> VideoAnalyzer:
        instance: VideoAnalyzer = VideoAnalyzer.__new__(VideoAnalyzer)
        # Set analysis defaults
        instance.morphological_kernel_size = overrides.get("morphological_kernel_size", 1)
        instance.noise_floor_threshold = overrides.get("noise_floor_threshold", 0.0)
        instance.background_percentile = overrides.get("background_percentile", 90.0)
        instance.background_roi_idx = overrides.get("background_roi_idx")
        instance.rects = overrides.get("rects", [])
        instance.frame_cache = overrides.get("frame_cache", FrameCache(FRAME_CACHE_SIZE))
        instance.cap = overrides.get("cap")
        return instance

    return _factory
