"""Video display widget for Brightness Sorcerer."""

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from typing import Optional, Callable, Tuple
import logging

from ..core.video_processor import VideoProcessor
from ..core.roi_manager import ROIManager


class VideoDisplayLabel(QtWidgets.QLabel):
    """Custom label for video display with mouse interaction."""
    
    # Signals
    mouse_pressed = QtCore.pyqtSignal(int, int, int)  # x, y, button
    mouse_moved = QtCore.pyqtSignal(int, int)  # x, y
    mouse_released = QtCore.pyqtSignal(int, int, int)  # x, y, button
    mouse_double_clicked = QtCore.pyqtSignal(int, int)  # x, y
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid #555; background-color: #1e1e1e;")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setScaledContents(False)  # Don't auto-scale contents
        self.setText("No video loaded\n\nDrag and drop a video file here,\nor use File → Open Video")
        
        # Mouse tracking
        self.setMouseTracking(True)
        self._last_mouse_pos = None
        
        # Scale factor for coordinate conversion
        self._scale_factor = 1.0
        self._offset_x = 0
        self._offset_y = 0
    
    def set_frame(self, frame: np.ndarray):
        """Display a video frame."""
        if frame is None:
            self.clear()
            self.setText("No frame")
            return
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Create QImage
            qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            
            # Calculate scale factor and offsets for coordinate conversion
            if not pixmap.isNull():
                pixmap_size = pixmap.size()
                widget_size = self.size()
                
                scale_x = widget_size.width() / pixmap_size.width()
                scale_y = widget_size.height() / pixmap_size.height()
                self._scale_factor = min(scale_x, scale_y)
            
            # Scale to fit widget while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            
            # Calculate offsets for coordinate conversion
            scaled_w = int(pixmap_size.width() * self._scale_factor)
            scaled_h = int(pixmap_size.height() * self._scale_factor)
            
            self._offset_x = (widget_size.width() - scaled_w) // 2
            self._offset_y = (widget_size.height() - scaled_h) // 2
            
            self.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logging.error(f"Error displaying frame: {e}")
            self.setText("Error displaying frame")
    
    def _convert_widget_to_frame_coords(self, widget_x: int, widget_y: int) -> Tuple[int, int]:
        """Convert widget coordinates to frame coordinates."""
        if self._scale_factor <= 0:
            return widget_x, widget_y
        
        # Adjust for centering offset
        frame_x = int((widget_x - self._offset_x) / self._scale_factor)
        frame_y = int((widget_y - self._offset_y) / self._scale_factor)
        
        return frame_x, frame_y
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        frame_x, frame_y = self._convert_widget_to_frame_coords(event.x(), event.y())
        self.mouse_pressed.emit(frame_x, frame_y, event.button())
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        frame_x, frame_y = self._convert_widget_to_frame_coords(event.x(), event.y())
        self.mouse_moved.emit(frame_x, frame_y)
        self._last_mouse_pos = (frame_x, frame_y)
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        frame_x, frame_y = self._convert_widget_to_frame_coords(event.x(), event.y())
        self.mouse_released.emit(frame_x, frame_y, event.button())
        super().mouseReleaseEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Handle mouse double-click events."""
        frame_x, frame_y = self._convert_widget_to_frame_coords(event.x(), event.y())
        self.mouse_double_clicked.emit(frame_x, frame_y)
        super().mouseDoubleClickEvent(event)


class VideoWidget(QtWidgets.QWidget):
    """Complete video display widget with controls."""
    
    # Signals
    frame_changed = QtCore.pyqtSignal(int)  # frame_index
    roi_interaction = QtCore.pyqtSignal(str, int, int)  # action, x, y
    
    def __init__(self, app_controller, parent=None):
        super().__init__(parent)
        self.app_controller = app_controller
        
        # Playback state
        self.is_playing = False
        self.fast_forward_mode = False
        self.playback_speed = 1.0
        self.playback_timer = QtCore.QTimer()
        self.playback_timer.timeout.connect(self._advance_frame)
        
        self._setup_ui()
        self._connect_signals()
        
        # Register UI callbacks with controller
        self.app_controller.register_ui_callback('frame_changed', self._on_frame_changed)
        self.app_controller.register_ui_callback('display_update_needed', self.update_display)
        self.app_controller.register_ui_callback('video_loaded', self._on_video_loaded)
        self.app_controller.register_ui_callback('roi_created', self._on_roi_created)
        self.app_controller.register_ui_callback('roi_selected', self._on_roi_selected)
    
    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Video display
        self.video_display = VideoDisplayLabel()
        self.video_display.setMinimumSize(400, 300)
        self.video_display.setMaximumSize(1200, 900)  # Prevent excessive resizing
        self.video_display.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.video_display)
        
        # Frame navigation controls
        nav_layout = QtWidgets.QHBoxLayout()
        
        # Frame slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        nav_layout.addWidget(self.frame_slider)
        
        # Frame info
        self.frame_info_label = QtWidgets.QLabel("No video")
        self.frame_info_label.setMinimumWidth(120)
        nav_layout.addWidget(self.frame_info_label)
        
        layout.addLayout(nav_layout)
        
        # Navigation buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        # Frame-by-frame controls
        self.prev_frame_btn = QtWidgets.QPushButton("◀")
        self.prev_frame_btn.setMaximumWidth(40)
        self.prev_frame_btn.setEnabled(False)
        self.prev_frame_btn.setToolTip("Previous frame")
        btn_layout.addWidget(self.prev_frame_btn)
        
        self.next_frame_btn = QtWidgets.QPushButton("▶")
        self.next_frame_btn.setMaximumWidth(40)
        self.next_frame_btn.setEnabled(False)
        self.next_frame_btn.setToolTip("Next frame")
        btn_layout.addWidget(self.next_frame_btn)
        
        btn_layout.addWidget(QtWidgets.QLabel("|"))
        
        # 10-frame skip controls
        self.prev_10_btn = QtWidgets.QPushButton("◀◀")
        self.prev_10_btn.setMaximumWidth(50)
        self.prev_10_btn.setEnabled(False)
        self.prev_10_btn.setToolTip("Skip back 10 frames")
        btn_layout.addWidget(self.prev_10_btn)
        
        self.next_10_btn = QtWidgets.QPushButton("▶▶")
        self.next_10_btn.setMaximumWidth(50)
        self.next_10_btn.setEnabled(False)
        self.next_10_btn.setToolTip("Skip forward 10 frames")
        btn_layout.addWidget(self.next_10_btn)
        
        btn_layout.addWidget(QtWidgets.QLabel("|"))
        
        # Playback controls
        self.play_pause_btn = QtWidgets.QPushButton("▶")
        self.play_pause_btn.setMaximumWidth(40)
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setToolTip("Play/Pause")
        btn_layout.addWidget(self.play_pause_btn)
        
        self.fast_forward_btn = QtWidgets.QPushButton("⏩")
        self.fast_forward_btn.setMaximumWidth(40)
        self.fast_forward_btn.setEnabled(False)
        self.fast_forward_btn.setToolTip("Fast forward")
        btn_layout.addWidget(self.fast_forward_btn)
        
        btn_layout.addStretch()
        
        # Speed control
        self.speed_label = QtWidgets.QLabel("1x")
        btn_layout.addWidget(self.speed_label)
        
        self.zoom_label = QtWidgets.QLabel("100%")
        btn_layout.addWidget(self.zoom_label)
        
        layout.addLayout(btn_layout)
    
    def _connect_signals(self):
        """Connect widget signals."""
        # Frame navigation
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.prev_frame_btn.clicked.connect(lambda: self._step_frames(-1))
        self.next_frame_btn.clicked.connect(lambda: self._step_frames(1))
        
        # 10-frame skip controls
        self.prev_10_btn.clicked.connect(lambda: self._step_frames(-10))
        self.next_10_btn.clicked.connect(lambda: self._step_frames(10))
        
        # Playback controls
        self.play_pause_btn.clicked.connect(self._toggle_playback)
        self.fast_forward_btn.clicked.connect(self._toggle_fast_forward)
        
        # Mouse interactions
        self.video_display.mouse_pressed.connect(self._on_mouse_pressed)
        self.video_display.mouse_moved.connect(self._on_mouse_moved)
        self.video_display.mouse_released.connect(self._on_mouse_released)
        self.video_display.mouse_double_clicked.connect(self._on_mouse_double_clicked)
    
    
    
    def _update_video_controls(self):
        """Update video controls based on loaded video."""
        if self.app_controller.video_processor.is_loaded():
            info = self.app_controller.video_processor.get_video_info()
            total_frames = info['total_frames']
            
            # Update slider
            self.frame_slider.setMaximum(total_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            
            # Enable all controls
            self.prev_frame_btn.setEnabled(True)
            self.next_frame_btn.setEnabled(True)
            self.prev_10_btn.setEnabled(True)
            self.next_10_btn.setEnabled(True)
            self.play_pause_btn.setEnabled(True)
            self.fast_forward_btn.setEnabled(True)
            
            # Update frame info
            self._update_frame_info()
        else:
            # Disable all controls
            self.frame_slider.setEnabled(False)
            self.prev_frame_btn.setEnabled(False)
            self.next_frame_btn.setEnabled(False)
            self.prev_10_btn.setEnabled(False)
            self.next_10_btn.setEnabled(False)
            self.play_pause_btn.setEnabled(False)
            self.fast_forward_btn.setEnabled(False)
            self.frame_info_label.setText("No video")
            
            # Stop playback if running
            if self.is_playing:
                self._pause_playback()
    
    def _update_frame_info(self):
        """Update frame information display."""
        if self.app_controller.video_processor.is_loaded():
            info = self.app_controller.video_processor.get_video_info()
            current = info['current_frame']
            total = info['total_frames']
            fps = info['fps']
            
            # Calculate time
            current_time = current / fps if fps > 0 else 0
            total_time = total / fps if fps > 0 else 0
            
            time_str = f"{current_time:.1f}s / {total_time:.1f}s"
            frame_str = f"Frame {current + 1}/{total}"
            
            self.frame_info_label.setText(f"{frame_str} ({time_str})")
        else:
            self.frame_info_label.setText("No video")
    
    def update_display(self):
        """Update the video display."""
        if not self.app_controller.video_processor.is_loaded():
            self.video_display.set_frame(None)
            return
        
        # Get current frame
        frame = self.app_controller.get_current_frame()
        if frame is None:
            return
        
        # Render ROIs on frame
        frame_with_rois = self.app_controller.roi_manager.render_rois(frame)
        
        # Display frame
        self.video_display.set_frame(frame_with_rois)
        
        # Update frame info
        self._update_frame_info()
        
        # Update slider position
        current_frame = self.app_controller.video_processor.current_frame_index
        if self.frame_slider.value() != current_frame:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(current_frame)
            self.frame_slider.blockSignals(False)
    
    def _step_frames(self, step: int):
        """Step forward or backward by frames."""
        if self.app_controller.step_frames(step):
            self.update_display()
            self.frame_changed.emit(self.app_controller.video_processor.current_frame_index)
    
    def _toggle_playback(self):
        """Toggle play/pause."""
        if not self.app_controller.video_processor.is_loaded():
            return
            
        if self.is_playing:
            self._pause_playback()
        else:
            self._start_playback()
    
    def _start_playback(self):
        """Start video playback."""
        if not self.app_controller.video_processor.is_loaded():
            return
            
        self.is_playing = True
        self.play_pause_btn.setText("⏸")
        self.play_pause_btn.setToolTip("Pause")
        
        # Calculate timer interval based on FPS and playback speed
        fps = self.app_controller.video_processor.fps
        if fps > 0:
            interval = int(1000 / (fps * self.playback_speed))
            self.playback_timer.start(interval)
    
    def _pause_playback(self):
        """Pause video playback."""
        self.is_playing = False
        self.playback_timer.stop()
        self.play_pause_btn.setText("▶")
        self.play_pause_btn.setToolTip("Play")
    
    def _toggle_fast_forward(self):
        """Toggle fast forward mode."""
        if not self.app_controller.video_processor.is_loaded():
            return
            
        if self.fast_forward_mode:
            self.fast_forward_mode = False
            self.playback_speed = 1.0
            self.fast_forward_btn.setText("⏩")
            self.speed_label.setText("1x")
        else:
            self.fast_forward_mode = True
            self.playback_speed = 4.0
            self.fast_forward_btn.setText("⏸")
            self.speed_label.setText("4x")
            
        # Update timer if playing
        if self.is_playing:
            self._pause_playback()
            self._start_playback()
    
    def _advance_frame(self):
        """Advance to next frame during playback."""
        if not self.app_controller.video_processor.is_loaded():
            self._pause_playback()
            return
            
        # Check if we've reached the end
        current_frame = self.app_controller.video_processor.current_frame_index
        total_frames = self.app_controller.video_processor.total_frames
        
        if current_frame >= total_frames - 1:
            self._pause_playback()
            return
            
        # Advance frame
        self._step_frames(1)
    
    def _on_frame_slider_changed(self, frame_index: int):
        """Handle frame slider changes."""
        if self.app_controller.seek_to_frame(frame_index):
            self.update_display()
            self.frame_changed.emit(frame_index)
    
    def _on_mouse_pressed(self, x: int, y: int, button: int):
        """Handle mouse press on video display."""
        self.app_controller.handle_mouse_press(x, y, button)
    
    def _on_mouse_moved(self, x: int, y: int):
        """Handle mouse move on video display."""
        self.app_controller.handle_mouse_move(x, y)
    
    def _on_mouse_released(self, x: int, y: int, button: int):
        """Handle mouse release on video display."""
        self.app_controller.handle_mouse_release(x, y, button)
    
    def _on_mouse_double_clicked(self, x: int, y: int):
        """Handle mouse double-click on video display."""
        # Could be used for ROI properties dialog
        pass
    
    def _on_frame_changed(self, frame_index: int):
        """Handle frame changes from controller."""
        self.update_display()
    
    def _on_video_loaded(self, video_data):
        """Handle video loaded from controller."""
        self._update_video_controls()
        self.update_display()
        self.update_display()
    
    def _on_roi_created(self, roi_index: int):
        """Handle ROI creation from controller."""
        self.roi_interaction.emit("created", roi_index, -1)
    
    def _on_roi_selected(self, roi_index: int):
        """Handle ROI selection from controller."""
        self.update_display()
    
    