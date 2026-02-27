from PyQt5 import QtCore

from ecl_analysis.roi_geometry import (
    get_pixmap_rect_in_label,
    map_frame_to_label_point,
    map_label_to_frame_point,
    map_label_to_frame_rect,
    scale_value_for_pixmap,
)


def test_get_pixmap_rect_in_label_centers_pixmap():
    rect = get_pixmap_rect_in_label(QtCore.QSize(200, 100), QtCore.QSize(100, 50))
    assert rect is not None
    assert rect.left() == 50
    assert rect.top() == 25


def test_map_label_to_frame_point_roundtrip():
    pixmap_rect = QtCore.QRect(10, 20, 100, 100)
    frame_shape = (200, 200)

    frame_x, frame_y = map_label_to_frame_point(QtCore.QPoint(60, 70), pixmap_rect, frame_shape)
    assert frame_x is not None and frame_y is not None

    mapped_back = map_frame_to_label_point((frame_x, frame_y), pixmap_rect, frame_shape)
    assert mapped_back is not None
    assert abs(mapped_back.x() - 60) <= 1
    assert abs(mapped_back.y() - 70) <= 1


def test_map_label_to_frame_rect_returns_none_outside_pixmap():
    pixmap_rect = QtCore.QRect(10, 10, 100, 100)
    frame_shape = (100, 100)
    pt1, pt2 = map_label_to_frame_rect(
        QtCore.QPoint(0, 0),
        QtCore.QPoint(20, 20),
        pixmap_rect,
        frame_shape,
    )
    assert pt1 is None
    assert pt2 is None


def test_scale_value_for_pixmap_scales_with_width():
    pixmap_rect = QtCore.QRect(0, 0, 100, 50)
    assert scale_value_for_pixmap(10.0, pixmap_rect, frame_width=200) == 5.0
