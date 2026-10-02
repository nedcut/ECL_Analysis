"""Geometry helpers for mapping between QLabel and frame coordinates."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from PyQt5 import QtCore


def roi_slice_bounds(
    pt1: Sequence[int],
    pt2: Sequence[int],
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    """Return the ``(x1, y1, x2, y2)`` frame region an ROI covers.

    This is the single definition of ROI pixel coverage shared by the analysis
    worker, background computation, mask capture, and the live preview. The
    corners are normalized (either corner order is accepted) and clamped to the
    frame, and the end coordinates are exclusive, i.e. the ROI covers
    ``frame[y1:y2, x1:x2]``. An ROI whose edge lies on the right/bottom frame
    border (``x2 == frame_width``) therefore includes the last column/row.

    Callers must treat ``x2 <= x1`` or ``y2 <= y1`` as an empty ROI.
    """
    left, right = sorted((int(pt1[0]), int(pt2[0])))
    top, bottom = sorted((int(pt1[1]), int(pt2[1])))

    x1 = max(0, min(left, frame_width))
    x2 = max(0, min(right, frame_width))
    y1 = max(0, min(top, frame_height))
    y2 = max(0, min(bottom, frame_height))

    return x1, y1, x2, y2


def get_pixmap_rect_in_label(
    label_size: QtCore.QSize,
    pixmap_size: QtCore.QSize,
) -> Optional[QtCore.QRect]:
    """Calculate the centered pixmap rectangle inside a label."""
    if (
        not label_size.isValid()
        or label_size.isEmpty()
        or not pixmap_size.isValid()
        or pixmap_size.isEmpty()
    ):
        return None

    offset_x = (label_size.width() - pixmap_size.width()) / 2
    offset_y = (label_size.height() - pixmap_size.height()) / 2
    return QtCore.QRect(int(offset_x), int(offset_y), pixmap_size.width(), pixmap_size.height())


def map_label_to_frame_point(
    label_pos: QtCore.QPoint,
    pixmap_rect: QtCore.QRect,
    frame_shape: Tuple[int, int],
) -> Tuple[Optional[int], Optional[int]]:
    """Map a QLabel-space point to frame coordinates."""
    if not pixmap_rect.contains(label_pos):
        return None, None

    frame_h, frame_w = frame_shape
    pixmap_w = pixmap_rect.width()
    pixmap_h = pixmap_rect.height()
    if pixmap_w == 0 or pixmap_h == 0:
        return None, None

    relative_x = label_pos.x() - pixmap_rect.left()
    relative_y = label_pos.y() - pixmap_rect.top()
    scale_w = frame_w / pixmap_w
    scale_h = frame_h / pixmap_h

    frame_x = int(relative_x * scale_w)
    frame_y = int(relative_y * scale_h)
    frame_x = max(0, min(frame_x, frame_w - 1))
    frame_y = max(0, min(frame_y, frame_h - 1))
    return frame_x, frame_y


def map_label_to_frame_rect(
    label_pt1: QtCore.QPoint,
    label_pt2: QtCore.QPoint,
    pixmap_rect: QtCore.QRect,
    frame_shape: Tuple[int, int],
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Map two QLabel-space points to a frame-space rectangle."""
    frame_pt1 = map_label_to_frame_point(label_pt1, pixmap_rect, frame_shape)
    frame_pt2 = map_label_to_frame_point(label_pt2, pixmap_rect, frame_shape)
    if frame_pt1[0] is None or frame_pt2[0] is None:
        return None, None
    return (frame_pt1[0], frame_pt1[1]), (frame_pt2[0], frame_pt2[1])


def map_frame_to_label_point(
    frame_pos: Tuple[int, int],
    pixmap_rect: QtCore.QRect,
    frame_shape: Tuple[int, int],
) -> Optional[QtCore.QPoint]:
    """Map a frame-space point into QLabel coordinates."""
    frame_h, frame_w = frame_shape
    pixmap_w = pixmap_rect.width()
    pixmap_h = pixmap_rect.height()
    if frame_w == 0 or frame_h == 0:
        return None

    scale_w = pixmap_w / frame_w
    scale_h = pixmap_h / frame_h
    label_x = int(pixmap_rect.left() + frame_pos[0] * scale_w)
    label_y = int(pixmap_rect.top() + frame_pos[1] * scale_h)
    return QtCore.QPoint(label_x, label_y)


def scale_value_for_pixmap(
    value_in_frame_coords: float,
    pixmap_rect: QtCore.QRect,
    frame_width: int,
) -> float:
    """Scale a frame-space distance into pixmap-space distance."""
    if frame_width <= 0 or pixmap_rect.width() <= 0:
        return value_in_frame_coords
    return value_in_frame_coords * (pixmap_rect.width() / frame_width)
