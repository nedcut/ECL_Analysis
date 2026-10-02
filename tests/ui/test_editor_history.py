from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ecl_analysis.video_analyzer import VideoAnalyzer


class _StubCapture:
    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        return None


def _prepare_loaded_window(window: VideoAnalyzer, total_frames: int = 100) -> None:
    window.cap = _StubCapture()
    window.frame = np.zeros((120, 200, 3), dtype=np.uint8)
    window.total_frames = total_frames
    window.playback_fps = 25.0
    window.start_frame = 0
    window.end_frame = total_frames - 1
    window.current_frame_index = 0
    window.frame_slider.setRange(0, total_frames - 1)
    window.frame_spinbox.setRange(0, total_frames - 1)
    window._seek_to_frame = lambda frame_index: setattr(window, "current_frame_index", frame_index)
    window._sync_analysis_range_widgets()


def test_analysis_range_changes_support_undo_and_redo(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    _prepare_loaded_window(window, total_frames=120)

    window.current_frame_index = 24
    window.set_start_frame()

    assert window.start_frame == 24
    assert window.range_start_spinbox.value() == 25
    assert "Frames 25-120" in window.analysis_range_summary_label.text()

    window.undo_last_action()
    assert window.start_frame == 0
    assert window.range_start_spinbox.value() == 1

    window.redo_last_action()
    assert window.start_frame == 24
    assert window.range_start_spinbox.value() == 25

    window.close()


def test_roi_addition_supports_undo_and_redo(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    _prepare_loaded_window(window)

    initial_rects = list(window.rects)
    window.roi_width_spin.setValue(60)
    window.roi_height_spin.setValue(40)
    window.add_roi_by_size()

    assert len(window.rects) == len(initial_rects) + 1

    window.undo_last_action()
    assert window.rects == initial_rects

    window.redo_last_action()
    assert len(window.rects) == len(initial_rects) + 1
    assert window.selected_rect_idx == 0

    window.close()


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _window_with_clock(qt_application: QtWidgets.QApplication):
    window = VideoAnalyzer()
    _prepare_loaded_window(window)
    clock = _FakeClock()
    window._history_timestamp = clock
    return window, clock


def test_consecutive_threshold_ticks_coalesce_into_one_undo_entry(
    qt_application: QtWidgets.QApplication,
) -> None:
    window, clock = _window_with_clock(qt_application)
    original = window.manual_threshold

    for value in (6.0, 6.5, 7.0, 7.5):
        clock.now += 0.1
        window.threshold_spin.setValue(value)

    assert len(window._undo_history) == 1
    assert window._undo_history[-1].label == "Adjust Manual Threshold"

    window.undo_last_action()
    assert window.manual_threshold == original
    assert window.threshold_spin.value() == original

    window.redo_last_action()
    assert window.manual_threshold == 7.5
    window.close()


def test_slider_ticks_coalesce_per_control_and_respect_time_window(
    qt_application: QtWidgets.QApplication,
) -> None:
    window, clock = _window_with_clock(qt_application)

    for value in (91, 92, 93):
        clock.now += 0.2
        window.bg_percentile_slider.setValue(value)
    assert [entry.label for entry in window._undo_history] == ["Adjust Background Percentile"]

    # A different control starts its own entry...
    clock.now += 0.2
    window.noise_floor_slider.setValue(4)
    clock.now += 0.2
    window.noise_floor_slider.setValue(6)
    # ...and returning to the first control does not merge into the old entry.
    clock.now += 0.2
    window.bg_percentile_slider.setValue(94)
    assert [entry.label for entry in window._undo_history] == [
        "Adjust Background Percentile",
        "Adjust Noise Floor",
        "Adjust Background Percentile",
    ]

    # A pause longer than the coalescing window starts a new entry.
    clock.now += 5.0
    window.bg_percentile_slider.setValue(95)
    assert len(window._undo_history) == 4

    window.undo_last_action()
    assert window.background_percentile == 94.0
    window.undo_last_action()
    assert window.background_percentile == 93.0
    window.undo_last_action()
    assert window.noise_floor_threshold == 0.0
    window.undo_last_action()
    assert window.background_percentile == 90.0
    window.close()


def test_coalesced_ticks_returning_to_start_leave_no_entry(
    qt_application: QtWidgets.QApplication,
) -> None:
    window, clock = _window_with_clock(qt_application)
    start_kernel = window.morphological_kernel_size

    clock.now += 0.1
    window.kernel_size_slider.setValue(5)
    clock.now += 0.1
    window.kernel_size_slider.setValue(start_kernel)

    assert window._undo_history == []
    assert not window.undo_action.isEnabled()
    window.close()


def test_ticks_after_undo_do_not_merge_into_undone_entry(
    qt_application: QtWidgets.QApplication,
) -> None:
    window, clock = _window_with_clock(qt_application)

    clock.now += 0.1
    window.threshold_spin.setValue(10.0)
    clock.now += 0.1
    window.threshold_spin.setValue(12.0)
    clock.now += 0.1
    window.undo_last_action()
    assert window.manual_threshold == 5.0

    clock.now += 0.1
    window.threshold_spin.setValue(20.0)
    assert len(window._undo_history) == 1
    assert window._redo_history == []
    window.undo_last_action()
    assert window.manual_threshold == 5.0
    window.close()


def test_undo_history_is_capped(
    qt_application: QtWidgets.QApplication,
) -> None:
    from ecl_analysis.constants import MAX_UNDO_HISTORY

    window, clock = _window_with_clock(qt_application)
    total_changes = MAX_UNDO_HISTORY + 50
    for step in range(1, total_changes + 1):
        clock.now += 10.0  # never coalesce
        window.threshold_spin.setValue(step * 0.5)

    assert len(window._undo_history) == MAX_UNDO_HISTORY

    while window._undo_history:
        window.undo_last_action()
    # The oldest 50 changes were dropped, so undo bottoms out after change 50.
    assert window.manual_threshold == 50 * 0.5
    assert len(window._redo_history) == MAX_UNDO_HISTORY
    window.close()


def test_history_snapshots_share_masks_without_aliasing_live_list(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    _prepare_loaded_window(window)
    mask = np.ones((4, 4), dtype=bool)
    window.rects = [((0, 0), (4, 4))]
    window.fixed_roi_masks = [mask]
    window.mask_source_frames = [0]

    snapshot = window._capture_editor_snapshot()
    assert snapshot.fixed_roi_masks[0] is mask
    assert snapshot.fixed_roi_masks is not window.fixed_roi_masks

    del window.fixed_roi_masks[0]
    assert snapshot.fixed_roi_masks == [mask]

    window._restore_editor_snapshot(snapshot)
    assert window.fixed_roi_masks[0] is not mask
    assert np.array_equal(window.fixed_roi_masks[0], mask)
    window.close()


def _mouse_event(event_type, pos, button, buttons):
    return QtGui.QMouseEvent(event_type, pos, button, buttons, QtCore.Qt.NoModifier)


def test_roi_drag_updates_list_row_without_rebuilding_until_release(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    _prepare_loaded_window(window)
    window.rects = [((10, 10), (50, 40)), ((100, 60), (150, 100))]
    window.selected_rect_idx = 0
    window.update_rect_list(preferred_row=0)
    window._current_image_size = window.image_label.size()
    window.show_frame()

    rebuilds = []
    widget_state_updates = []
    original_update_rect_list = window.update_rect_list
    original_update_widget_states = window._update_widget_states

    def _counting_update_rect_list(*args, **kwargs):
        rebuilds.append(args)
        return original_update_rect_list(*args, **kwargs)

    def _counting_update_widget_states(*args, **kwargs):
        widget_state_updates.append(args)
        return original_update_widget_states(*args, **kwargs)

    window.update_rect_list = _counting_update_rect_list
    window._update_widget_states = _counting_update_widget_states

    start = window._map_frame_to_label_point((30, 25))
    assert start is not None
    window.image_mouse_press(
        _mouse_event(QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    )
    assert window.moving

    for step in range(1, 6):
        target = window._map_frame_to_label_point((30 + step * 4, 25 + step * 2))
        window.image_mouse_move(
            _mouse_event(QtCore.QEvent.MouseMove, target, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
        )
        (x1, y1), (x2, y2) = window.rects[0]
        assert window.rect_list.item(0).text() == f"ROI 1: ({x1},{y1})-({x2},{y2})"

    assert window.rects[0] != ((10, 10), (50, 40))
    assert rebuilds == []
    assert widget_state_updates == []

    window.image_mouse_release(
        _mouse_event(QtCore.QEvent.MouseButtonRelease, target, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
    )
    assert len(rebuilds) == 1
    assert window.rect_list.count() == 2
    assert window.rect_list.currentRow() == 0
    (x1, y1), (x2, y2) = window.rects[0]
    assert window.rect_list.item(0).text() == f"ROI 1: ({x1},{y1})-({x2},{y2})"
    assert window._undo_history[-1].label == "Move ROI"

    window.undo_last_action()
    assert window.rects[0] == ((10, 10), (50, 40))
    assert window.rect_list.item(0).text() == "ROI 1: (10,10)-(50,40)"
    window.close()


def test_clear_all_rectangles_records_a_single_history_entry(
    qt_application: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    window = VideoAnalyzer()
    _prepare_loaded_window(window)
    window.rects = [((10, 10), (50, 50)), ((60, 10), (100, 50))]
    window.fixed_roi_masks = [np.ones((40, 40), dtype=bool), None]
    window.mask_source_frames = [3, None]
    window._set_use_fixed_mask_silently(True)
    window.update_rect_list()
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Yes,
    )

    window.clear_all_rectangles()

    assert window.rects == []
    assert window.use_fixed_mask is False
    assert not window.use_fixed_mask_checkbox.isChecked()
    assert [entry.label for entry in window._undo_history] == ["Clear All ROIs"]

    window.undo_last_action()
    assert len(window.rects) == 2
    assert window.use_fixed_mask is True
    assert window.use_fixed_mask_checkbox.isChecked()
    assert window._undo_history == []

    window.close()


def test_user_toggle_of_fixed_mask_still_records_history(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    _prepare_loaded_window(window)
    window.rects = [((10, 10), (50, 50))]

    window.use_fixed_mask_checkbox.setChecked(True)

    assert window.use_fixed_mask is True
    assert [entry.label for entry in window._undo_history] == ["Toggle Fixed Mask"]

    window.close()


def test_roi_drag_release_readout_reflects_cleared_masks(
    qt_application: QtWidgets.QApplication,
) -> None:
    window = VideoAnalyzer()
    _prepare_loaded_window(window)
    window.frame = np.tile(np.arange(0, 256, 2, dtype=np.uint8)[None, :128, None], (120, 1, 3)).copy()
    window.rects = [((10, 10), (60, 50))]
    window.selected_rect_idx = 0
    window.update_rect_list(preferred_row=0)
    window.manual_threshold = 30.0
    window._capture_fixed_masks(0)
    window.use_fixed_mask_checkbox.setChecked(True)
    window._current_image_size = window.image_label.size()
    window.show_frame()

    start = window._map_frame_to_label_point((30, 25))
    assert start is not None
    window.image_mouse_press(
        _mouse_event(QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    )
    target = window._map_frame_to_label_point((42, 31))
    window.image_mouse_move(
        _mouse_event(QtCore.QEvent.MouseMove, target, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
    )
    window.image_mouse_release(
        _mouse_event(QtCore.QEvent.MouseButtonRelease, target, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
    )

    assert window.rects[0] != ((10, 10), (60, 50))
    assert all(mask is None for mask in window.fixed_roi_masks)
    # The readout must describe the analysis after the masks were invalidated.
    shown = window.brightness_display_label.text()
    window._update_current_brightness_display()
    assert shown == window.brightness_display_label.text()

    window.close()
