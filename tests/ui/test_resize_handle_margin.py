from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ecl_analysis.constants import MOUSE_RESIZE_HANDLE_SENSITIVITY
from ecl_analysis.video_analyzer import VideoAnalyzer


def test_resize_margin_is_screen_pixels_converted_to_frame_units(
    qt_application: QtWidgets.QApplication,
) -> None:
    """A 4K frame shown 960px wide should get a 4x larger margin in frame units."""
    window = VideoAnalyzer()
    window.frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    pixmap_rect = QtCore.QRect(0, 0, 960, 540)
    window._get_pixmap_rect_in_label = lambda: pixmap_rect

    margin = window._scale_value_for_frame(MOUSE_RESIZE_HANDLE_SENSITIVITY)
    assert margin == MOUSE_RESIZE_HANDLE_SENSITIVITY * 4

    rect = ((1000, 1000), (2000, 1600))
    # 30 frame px == 7.5 screen px from the right edge: inside the 10 screen-px handle.
    assert window._get_resize_handle(2030, 1300, rect, margin) == "edge_right"
    assert window._get_resize_handle(2000 - 30, 1600 + 30, rect, margin) == "corner_br"
    # 60 frame px == 15 screen px: outside the handle.
    assert window._get_resize_handle(2060, 1300, rect, margin) is None

    window.close()


def test_resize_margin_shrinks_for_upscaled_small_videos(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    window.frame = np.zeros((240, 320, 3), dtype=np.uint8)
    pixmap_rect = QtCore.QRect(0, 0, 1280, 960)
    window._get_pixmap_rect_in_label = lambda: pixmap_rect

    assert window._scale_value_for_frame(MOUSE_RESIZE_HANDLE_SENSITIVITY) == MOUSE_RESIZE_HANDLE_SENSITIVITY / 4

    window.close()
