"""
Video Player Widget

Modern video playback control widget with enhanced UI/UX.
"""

import os
from typing import Optional, Tuple
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np


class VideoPlayerWidget(QtWidgets.QWidget):
    """
    Enhanced video player widget with modern controls and responsive design.
    """
    
    # Signals
    frameChanged = QtCore.pyqtSignal(int)  # Emitted when frame changes
    videoLoaded = QtCore.pyqtSignal(str)   # Emitted when video loads
    playbackStateChanged = QtCore.pyqtSignal(bool)  # Emitted when play/pause state changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30.0
        self.is_playing = False
        
        # Playback timer
        self.playback_timer = QtCore.QTimer()
        self.playback_timer.timeout.connect(self._advance_frame)
        
        self._setup_ui()
        self._setup_styles()
        self._connect_signals()
    
    def _setup_ui(self):
        """Create and layout UI components."""
        self.setMinimumSize(400, 300)
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Video display area
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumHeight(200)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #3daee9;
                border-radius: 8px;
                background-color: #2b2b2b;
                color: #ffffff;
            }
        """)
        self.video_label.setText("Drop video file here or use Open Video button")
        
        # Enable drag and drop
        self.video_label.setAcceptDrops(True)
        self.video_label.dragEnterEvent = self._drag_enter_event
        self.video_label.dropEvent = self._drop_event
        
        layout.addWidget(self.video_label, 1)
        
        # Controls frame
        controls_frame = QtWidgets.QFrame()
        controls_layout = QtWidgets.QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        
        # Play/Pause button
        self.play_button = QtWidgets.QPushButton()
        self.play_button.setFixedSize(40, 40)
        self.play_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.play_button.setEnabled(False)
        
        # Frame position slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        
        # Frame counter
        self.frame_label = QtWidgets.QLabel("0 / 0")
        self.frame_label.setMinimumWidth(80)
        self.frame_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # Speed control
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.setFixedWidth(60)
        
        # Add controls to layout
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.frame_slider, 1)
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(self.speed_combo)
        
        layout.addWidget(controls_frame)
        
        # File operations frame
        file_frame = QtWidgets.QFrame()
        file_layout = QtWidgets.QHBoxLayout(file_frame)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(8)
        
        # Open video button
        self.open_button = QtWidgets.QPushButton("Open Video")
        self.open_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        
        # Video info label
        self.info_label = QtWidgets.QLabel("No video loaded")
        self.info_label.setStyleSheet("color: #888888;")
        
        file_layout.addWidget(self.open_button)
        file_layout.addWidget(self.info_label, 1)
        
        layout.addWidget(file_frame)
    
    def _setup_styles(self):
        """Apply modern styling to the widget."""
        self.setStyleSheet("""
            VideoPlayerWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3daee9;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #5cbef4;
            }
            QPushButton:pressed {
                background-color: #2a9dd9;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #2b2b2b;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3daee9;
                border: 1px solid #3daee9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #5cbef4;
            }
            QComboBox {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px 8px;
                color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
        """)
    
    def _connect_signals(self):
        """Connect widget signals to their handlers."""
        self.play_button.clicked.connect(self._toggle_playback)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        self.frame_slider.sliderPressed.connect(self._on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)
        self.open_button.clicked.connect(self._open_video_dialog)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
    
    def load_video(self, video_path: str) -> bool:
        """
        Load a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(video_path):
            return False
        
        # Release existing video
        if self.cap is not None:
            self.cap.release()
        
        # Load new video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            return False
        
        # Get video properties
        self.video_path = video_path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Update UI
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        
        # Update info label
        filename = os.path.basename(video_path)
        self.info_label.setText(f"{filename} ({self.total_frames} frames, {self.fps:.1f} fps)")
        
        # Load first frame
        self.seek_to_frame(0)
        
        self.videoLoaded.emit(video_path)
        return True
    
    def seek_to_frame(self, frame_number: int):
        """
        Seek to a specific frame.
        
        Args:
            frame_number: Frame to seek to (0-based)
        """
        if self.cap is None or frame_number < 0 or frame_number >= self.total_frames:
            return
        
        self.current_frame = frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read and display frame
        ret, frame = self.cap.read()
        if ret:
            self._display_frame(frame)
        
        # Update UI
        self.frame_slider.setValue(frame_number)
        self.frame_label.setText(f"{frame_number + 1} / {self.total_frames}")
        
        self.frameChanged.emit(frame_number)
    
    def _display_frame(self, frame: np.ndarray):
        """Display a frame in the video label."""
        if frame is None:
            return
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage
        qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            QtCore.Qt.KeepAspectRatio, 
            QtCore.Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def _toggle_playback(self):
        """Toggle between play and pause."""
        if self.cap is None:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
            interval = int(1000 / (self.fps * self._get_speed_multiplier()))
            self.playback_timer.start(interval)
        else:
            self.play_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.playback_timer.stop()
        
        self.playbackStateChanged.emit(self.is_playing)
    
    def _advance_frame(self):
        """Advance to the next frame during playback."""
        if self.current_frame < self.total_frames - 1:
            self.seek_to_frame(self.current_frame + 1)
        else:
            # End of video, stop playback
            self._toggle_playback()
    
    def _get_speed_multiplier(self) -> float:
        """Get the current playback speed multiplier."""
        speed_text = self.speed_combo.currentText()
        return float(speed_text.replace('x', ''))
    
    def _on_speed_changed(self, speed_text: str):
        """Handle playback speed change."""
        if self.is_playing:
            # Update timer interval
            interval = int(1000 / (self.fps * self._get_speed_multiplier()))
            self.playback_timer.setInterval(interval)
    
    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        if not hasattr(self, '_slider_pressed') or not self._slider_pressed:
            self.seek_to_frame(value)
    
    def _on_slider_pressed(self):
        """Handle slider press (start dragging)."""
        self._slider_pressed = True
        if self.is_playing:
            self.playback_timer.stop()
    
    def _on_slider_released(self):
        """Handle slider release (end dragging)."""
        self._slider_pressed = False
        self.seek_to_frame(self.frame_slider.value())
        if self.is_playing:
            interval = int(1000 / (self.fps * self._get_speed_multiplier()))
            self.playback_timer.start(interval)
    
    def _open_video_dialog(self):
        """Open file dialog to select video."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def _drag_enter_event(self, event: QtGui.QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def _drop_event(self, event: QtGui.QDropEvent):
        """Handle drop event."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path and file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                self.load_video(file_path)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame as a numpy array.
        
        Returns:
            Current frame or None if no video loaded
        """
        if self.cap is None:
            return None
            
        # Store current position
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Seek to current frame and read
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        # Restore position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        return frame if ret else None
    
    def closeEvent(self, event):
        """Clean up resources when widget is closed."""
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)