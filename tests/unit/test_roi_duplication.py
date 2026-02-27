from ecl_analysis.video_analyzer import _offset_rect_within_bounds
from ecl_analysis.video_analyzer import VideoAnalyzer


def test_offset_rect_without_bounds():
    rect = ((10, 20), (30, 40))
    shifted = _offset_rect_within_bounds(rect, dx=12, dy=-5)
    assert shifted == ((22, 15), (42, 35))


def test_offset_rect_clamps_to_frame_bounds():
    rect = ((90, 40), (99, 60))
    shifted = _offset_rect_within_bounds(rect, dx=12, dy=12, frame_shape=(100, 100))
    assert shifted == ((90, 52), (99, 72))


def test_offset_rect_clamps_negative_shift_to_zero():
    rect = ((5, 5), (20, 20))
    shifted = _offset_rect_within_bounds(rect, dx=-12, dy=-12, frame_shape=(100, 100))
    assert shifted == ((0, 0), (15, 15))


def test_get_resize_handle_detects_corner_and_edge():
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    rect = ((10, 10), (30, 30))

    assert analyzer._get_resize_handle(10, 10, rect, margin=3) == "corner_tl"
    assert analyzer._get_resize_handle(30, 30, rect, margin=3) == "corner_br"
    assert analyzer._get_resize_handle(10, 20, rect, margin=3) == "edge_left"
    assert analyzer._get_resize_handle(20, 10, rect, margin=3) == "edge_top"
