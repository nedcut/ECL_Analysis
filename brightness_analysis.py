import sys
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore

# --- Constants ---
DEFAULT_FONT_FAMILY = "Segoe UI, Arial, sans-serif" # More standard font
COLOR_BACKGROUND = "#2d2d2d"
COLOR_FOREGROUND = "#cccccc"
COLOR_ACCENT = "#5a9bd5" # A professional blue accent
COLOR_ACCENT_HOVER = "#7ab3e0"
COLOR_SECONDARY = "#404040"
COLOR_SECONDARY_LIGHT = "#555555"
COLOR_SUCCESS = "#70ad47"
COLOR_WARNING = "#ed7d31"
COLOR_ERROR = "#ff0000"
COLOR_INFO = "#ffc000"
COLOR_BRIGHTNESS_LABEL = "#ffeb3b" # Keep the yellow for brightness

ROI_COLORS = [
    (255, 50, 50), (50, 200, 50), (50, 150, 255), (255, 150, 50),
    (255, 50, 255), (50, 200, 200), (150, 50, 255), (255, 255, 50)
]
ROI_THICKNESS_DEFAULT = 2
ROI_THICKNESS_SELECTED = 4
ROI_LABEL_FONT_SCALE = 0.8
ROI_LABEL_THICKNESS = 2

AUTO_DETECT_BASELINE_PERCENTILE = 5
AUTO_DETECT_THRESHOLD_L_UNITS = 5.0
BRIGHTNESS_NOISE_FLOOR_PERCENTILE = 2

MOUSE_RESIZE_HANDLE_SENSITIVITY = 10 # Pixels

class VideoAnalyzer(QtWidgets.QWidget):
    """Main application window for video brightness analysis."""
    def __init__(self):
        """Initializes the application window and UI elements."""
        super().__init__()
        self._init_vars()
        self._init_ui()

    def _init_vars(self):
        """Initialize instance variables."""
        self.video_path = None
        self.frame = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.cap = None
        self.out_paths = [] # Store paths of saved CSV files

        # ROI management
        self.rects = []
        self.selected_rect_idx = None
        self.drawing = False
        self.moving = False
        self.resizing = False
        self.start_point = None
        self.end_point = None
        self.move_offset = None
        self.resize_corner = None
        self._current_image_size = None # Store the size of the image label for scaling

        # Frame range
        self.start_frame = 0
        self.end_frame = None

    def _init_ui(self):
        """Set up the main UI layout and widgets."""
        self.setWindowTitle('Brightness Sorcerer')
        self.setGeometry(100, 100, 1200, 800) # Initial size and position
        self.setAcceptDrops(True)
        self._apply_stylesheet()

        self.main_layout = QtWidgets.QHBoxLayout(self)
        self._create_layouts()
        self._create_widgets()
        self._connect_signals()
        self._update_widget_states() # Set initial enabled/disabled states

    def _apply_stylesheet(self):
        """Apply a modern, clean stylesheet to the application."""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLOR_BACKGROUND};
                color: {COLOR_FOREGROUND};
                font-family: {DEFAULT_FONT_FAMILY};
                font-size: 14px;
            }}
            QLabel#titleLabel {{
                font-size: 24px;
                font-weight: bold;
                color: {COLOR_ACCENT};
                padding-bottom: 10px;
                qproperty-alignment: AlignCenter;
            }}
            QLabel#imageLabel {{
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                background: #1e1e1e; /* Slightly darker for contrast */
                border-radius: 6px;
            }}
            QLabel#resultsLabel {{
                font-size: 13px;
                color: {COLOR_INFO};
                background: {COLOR_SECONDARY};
                border-radius: 4px;
                padding: 8px;
                border: 1px solid {COLOR_SECONDARY_LIGHT};
            }}
            QLabel#brightnessDisplayLabel {{
                font-size: 28px;
                font-weight: bold;
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                padding: 10px;
                color: {COLOR_BRIGHTNESS_LABEL};
                background: {COLOR_SECONDARY};
                border-radius: 6px;
                qproperty-alignment: AlignCenter;
            }}
            QGroupBox {{
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 6px;
                margin-top: 10px;
                background: {COLOR_SECONDARY};
                font-weight: bold;
                font-size: 15px;
                padding-top: 10px; /* Add padding inside the box */
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left; /* Position title at top left */
                left: 10px;
                padding: 2px 5px; /* Adjust padding */
                color: {COLOR_ACCENT};
                background-color: {COLOR_BACKGROUND}; /* Match window background */
                border-radius: 3px;
            }}
            QPushButton {{
                background-color: {COLOR_SECONDARY_LIGHT};
                color: {COLOR_FOREGROUND};
                border: 1px solid {COLOR_SECONDARY};
                border-radius: 4px;
                padding: 8px 15px;
                font-size: 14px;
                min-height: 20px; /* Ensure buttons have a minimum height */
            }}
            QPushButton:hover {{
                background-color: {COLOR_ACCENT};
                color: {COLOR_BACKGROUND};
                border: 1px solid {COLOR_ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {COLOR_ACCENT_HOVER};
            }}
            QPushButton:disabled {{
                background-color: {COLOR_SECONDARY};
                color: #888888;
                border: 1px solid {COLOR_SECONDARY};
            }}
            QPushButton:checked {{ /* Style for checkable buttons like 'Add Rect' */
                background-color: {COLOR_ACCENT};
                color: {COLOR_BACKGROUND};
                border: 1px solid {COLOR_ACCENT_HOVER};
            }}
            QListWidget {{
                background: {COLOR_BACKGROUND}; /* Match main background */
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                color: {COLOR_FOREGROUND};
                font-size: 13px;
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background: {COLOR_ACCENT};
                color: {COLOR_BACKGROUND};
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {COLOR_SECONDARY};
                height: 6px; /* Thinner slider */
                background: {COLOR_SECONDARY};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLOR_ACCENT};
                border: 1px solid {COLOR_ACCENT_HOVER};
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLOR_SUCCESS}; /* Use success color for progress */
                border-radius: 3px;
            }}
            QSlider::add-page:horizontal {{
                background: {COLOR_SECONDARY};
                border-radius: 3px;
            }}
            QLineEdit, QSpinBox {{
                background-color: {COLOR_BACKGROUND};
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                padding: 4px;
                border-radius: 4px;
                min-height: 20px;
            }}
            QLineEdit:focus, QSpinBox:focus {{
                border: 1px solid {COLOR_ACCENT};
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                subcontrol-origin: border;
                width: 16px;
                border-left: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 2px;
            }}
            QSpinBox::up-button {{
                subcontrol-position: top right;
            }}
            QSpinBox::down-button {{
                subcontrol-position: bottom right;
            }}
            QSpinBox::up-arrow {{
                image: url(./icons/arrow_up.png); /* Requires icon files */
                width: 10px; height: 10px;
            }}
            QSpinBox::down-arrow {{
                image: url(./icons/arrow_down.png); /* Requires icon files */
                 width: 10px; height: 10px;
            }}
            QProgressDialog {{
                 font-size: 14px;
            }}
            QProgressDialog QLabel {{ /* Style label inside progress dialog */
                 color: {COLOR_FOREGROUND};
            }}
            QProgressBar {{
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 4px;
                text-align: center;
                color: {COLOR_FOREGROUND};
            }}
            QProgressBar::chunk {{
                background-color: {COLOR_SUCCESS};
                border-radius: 3px;
            }}
        """)

    def _create_layouts(self):
        """Create the main horizontal and vertical layouts."""
        self.left_layout = QtWidgets.QVBoxLayout()
        self.right_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.left_layout, stretch=3) # Give more space to video
        self.main_layout.addLayout(self.right_layout, stretch=1)

    def _create_widgets(self):
        """Create all the widgets and add them to layouts."""
        # --- Left Layout Widgets ---
        self.title_label = QtWidgets.QLabel("Brightness Sorcerer", self)
        self.title_label.setObjectName("titleLabel")
        self.left_layout.addWidget(self.title_label)

        # Open-file button
        self.open_btn = QtWidgets.QPushButton("Open Video…")    
        self.open_btn.setToolTip("Choose a video file from disk")
        self.left_layout.addWidget(self.open_btn)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image_label.setText("Drag & Drop Video File Here") # Initial text
        self.left_layout.addWidget(self.image_label, stretch=1) # Give it stretch factor

        # Slider and Frame Label Layout
        slider_frame_layout = QtWidgets.QHBoxLayout()
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_frame_layout.addWidget(self.frame_slider)

        self.frame_label = QtWidgets.QLabel("Frame: 0 / 0")
        self.frame_label.setMinimumWidth(100) # Adjust width as needed
        slider_frame_layout.addWidget(self.frame_label)

        self.frame_spinbox = QtWidgets.QSpinBox()
        self.frame_spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus) # Use +/- symbols
        slider_frame_layout.addWidget(self.frame_spinbox)
        self.left_layout.addLayout(slider_frame_layout)

        # Frame Control Buttons Layout
        frame_control_layout = QtWidgets.QHBoxLayout()
        self.prev_frame_btn = QtWidgets.QPushButton("<")
        self.prev_frame_btn.setToolTip("Previous Frame (Left Arrow)")
        self.prev_frame_btn.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left))
        frame_control_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QtWidgets.QPushButton(">")
        self.next_frame_btn.setToolTip("Next Frame (Right Arrow)")
        self.next_frame_btn.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right))
        frame_control_layout.addWidget(self.next_frame_btn)

        self.set_start_btn = QtWidgets.QPushButton("Set Start")
        self.set_start_btn.setToolTip("Set current frame as analysis start")
        frame_control_layout.addWidget(self.set_start_btn)

        self.set_end_btn = QtWidgets.QPushButton("Set End")
        self.set_end_btn.setToolTip("Set current frame as analysis end")
        frame_control_layout.addWidget(self.set_end_btn)

        self.auto_detect_btn = QtWidgets.QPushButton("Auto-Detect")
        self.auto_detect_btn.setToolTip("Automatically detect start/end frames based on ROI brightness")
        frame_control_layout.addWidget(self.auto_detect_btn)
        self.left_layout.addLayout(frame_control_layout)

        # Analysis Name Layout
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Analysis Name:"))
        self.analysis_name_input = QtWidgets.QLineEdit()
        self.analysis_name_input.setPlaceholderText("DefaultAnalysis")
        name_layout.addWidget(self.analysis_name_input)
        self.left_layout.addLayout(name_layout)

        # Action Buttons Layout
        action_layout = QtWidgets.QHBoxLayout()
        self.analyze_btn = QtWidgets.QPushButton('Analyze Brightness')
        self.analyze_btn.setToolTip("Run brightness analysis on the selected frame range and ROIs")
        action_layout.addWidget(self.analyze_btn)

        self.plot_btn = QtWidgets.QPushButton('Plot Results')
        self.plot_btn.setToolTip("Generate and show plots for the last analysis run")
        action_layout.addWidget(self.plot_btn)
        self.left_layout.addLayout(action_layout)

        # --- Right Layout Widgets ---
        # Brightness Display
        self.brightness_groupbox = QtWidgets.QGroupBox("Avg. ROI Brightness (Current Frame)")
        brightness_groupbox_layout = QtWidgets.QVBoxLayout()
        self.brightness_display_label = QtWidgets.QLabel("N/A")
        self.brightness_display_label.setObjectName("brightnessDisplayLabel")
        brightness_groupbox_layout.addWidget(self.brightness_display_label)
        self.brightness_groupbox.setLayout(brightness_groupbox_layout)
        self.right_layout.addWidget(self.brightness_groupbox)

        # Rectangle Controls
        self.rect_groupbox = QtWidgets.QGroupBox("Regions of Interest (ROI)")
        rect_groupbox_layout = QtWidgets.QVBoxLayout()
        self.rect_list = QtWidgets.QListWidget()
        self.rect_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        rect_groupbox_layout.addWidget(self.rect_list)

        rect_btn_layout = QtWidgets.QHBoxLayout()
        self.add_rect_btn = QtWidgets.QPushButton("Add ROI")
        self.add_rect_btn.setCheckable(True)
        self.add_rect_btn.setToolTip("Click then draw a rectangle on the video frame")
        rect_btn_layout.addWidget(self.add_rect_btn)

        self.del_rect_btn = QtWidgets.QPushButton("Delete ROI")
        self.del_rect_btn.setToolTip("Delete the selected ROI from the list")
        rect_btn_layout.addWidget(self.del_rect_btn)

        self.clear_rect_btn = QtWidgets.QPushButton("Clear All")
        self.clear_rect_btn.setToolTip("Remove all ROIs")
        rect_btn_layout.addWidget(self.clear_rect_btn)
        rect_groupbox_layout.addLayout(rect_btn_layout)
        self.rect_groupbox.setLayout(rect_groupbox_layout)
        self.right_layout.addWidget(self.rect_groupbox)

        # Results/Status Label
        self.results_label = QtWidgets.QLabel("Load a video to begin analysis.")
        self.results_label.setObjectName("resultsLabel")
        self.results_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.results_label.setWordWrap(True)
        self.results_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding) # Allow expansion
        self.right_layout.addWidget(self.results_label, stretch=1) # Give stretch factor

    def _connect_signals(self):
        """Connect widget signals to their corresponding slots."""
        self.open_btn.clicked.connect(self.open_video_dialog)

        self.frame_slider.valueChanged.connect(self.slider_frame_changed)
        self.frame_spinbox.valueChanged.connect(self.spinbox_frame_changed)

        self.prev_frame_btn.clicked.connect(lambda: self.step_frames(-1))
        self.next_frame_btn.clicked.connect(lambda: self.step_frames(1))
        self.set_start_btn.clicked.connect(self.set_start_frame)
        self.set_end_btn.clicked.connect(self.set_end_frame)
        self.auto_detect_btn.clicked.connect(self.auto_detect_range)

        self.analyze_btn.clicked.connect(self.analyze_video)
        self.plot_btn.clicked.connect(self.plot_results)

        self.rect_list.currentRowChanged.connect(self.select_rectangle_from_list)
        self.add_rect_btn.clicked.connect(self.toggle_add_rectangle_mode)
        self.del_rect_btn.clicked.connect(self.delete_selected_rectangle)
        self.clear_rect_btn.clicked.connect(self.clear_all_rectangles)

        # Connect mouse events directly (overriding QWidget methods)
        self.image_label.mousePressEvent = self.image_mouse_press
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release

    def _update_widget_states(self, video_loaded=False, analysis_done=False, rois_exist=False):
        """Enable/disable widgets based on application state."""
        self.frame_slider.setEnabled(video_loaded)
        self.frame_spinbox.setEnabled(video_loaded)
        self.prev_frame_btn.setEnabled(video_loaded)
        self.next_frame_btn.setEnabled(video_loaded)
        self.set_start_btn.setEnabled(video_loaded)
        self.set_end_btn.setEnabled(video_loaded)
        self.auto_detect_btn.setEnabled(video_loaded and rois_exist)
        self.analyze_btn.setEnabled(video_loaded and rois_exist)
        self.plot_btn.setEnabled(analysis_done)
        self.add_rect_btn.setEnabled(video_loaded)
        self.del_rect_btn.setEnabled(video_loaded and self.selected_rect_idx is not None)
        self.clear_rect_btn.setEnabled(video_loaded and rois_exist)

    # --- Event Handling ---

    # File-picker slot
    def open_video_dialog(self):
        """
        Present a Finder/Explorer dialog, let the user pick a video file,
        and load it just as if it had been dragged in.
        """
        # Start in the same folder as the last-opened clip if we have one
        initial_dir = os.path.dirname(self.video_path) if self.video_path else ""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            initial_dir,
            "Video Files (*.mp4 *.mov *.avi *.mkv *.wmv);;All Files (*)"
        )
        if path:
            self.video_path = path
            self.load_video()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """Accept drag events if they contain URLs (files)."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction() # Use acceptProposedAction for clarity
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        """Handle dropped files, attempting to load the first valid video."""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            # Basic check for video file extensions (can be improved)
            if os.path.splitext(path)[1].lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                self.video_path = path
                self.load_video() # Changed from load_first_frame
            else:
                QtWidgets.QMessageBox.warning(self, 'Invalid File',
                                              f'Unsupported file type: {os.path.basename(path)}')
            event.acceptProposedAction()
        else:
            event.ignore()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Release video capture resources when the window closes."""
        if self.cap:
            self.cap.release()
        super().closeEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        """
        Update cached label size *without* immediately redrawing the frame.
        Calling show_frame() synchronously inside resizeEvent can create
        a feedback loop: the freshly scaled pixmap changes the label’s
        sizeHint, Qt recalculates the layout, and another resizeEvent fires.
        By deferring the redraw with QTimer.singleShot(0, …) we let the
        resize settle first and repaint exactly once.
        """
        if hasattr(self, "image_label") and self.image_label.size().isValid():
            self._current_image_size = self.image_label.size()

            # Schedule a one‑shot repaint after the event loop returns.
            if self.frame is not None:
                QtCore.QTimer.singleShot(0, self.show_frame)

        # Call base-class handler last (standard Qt practice)
        super().resizeEvent(event)

    # --- Video Loading and Frame Display ---

    def load_video(self):
        """Loads the video specified by self.video_path."""
        if self.cap:
            self.cap.release()
            self.cap = None

        if not self.video_path or not os.path.exists(self.video_path):
            QtWidgets.QMessageBox.critical(self, 'Error', 'Video path is invalid or file does not exist.')
            self._reset_state()
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, 'Error', f'Could not open video file: {os.path.basename(self.video_path)}')
            self._reset_state()
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
             QtWidgets.QMessageBox.warning(self, 'Warning', 'Video file appears to have no frames.')
             self._reset_state()
             return

        self.current_frame_index = 0
        self.start_frame = 0
        self.end_frame = self.total_frames - 1 # Default end frame

        # Load the first frame
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            # Update UI elements for the loaded video
            self.frame_slider.setRange(0, self.total_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_spinbox.setRange(0, self.total_frames - 1)
            self.frame_spinbox.setValue(0)
            self.update_frame_label()
            self.show_frame()
            self.results_label.setText(f"Loaded: {os.path.basename(self.video_path)}\n"
                                       f"Frames: {self.total_frames}\n"
                                       "Draw ROIs or use Auto-Detect.")
            self._update_widget_states(video_loaded=True, rois_exist=bool(self.rects))

            # Attempt auto-detection if ROIs already exist (e.g., loaded state later)
            if self.rects:
                self.auto_detect_range()
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Could not read the first frame of the video.')
            self._reset_state()

    def _reset_state(self):
        """Resets the application state when a video fails to load or is closed."""
        if self.cap:
            self.cap.release()
        self.video_path = None
        self.frame = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.cap = None
        self.rects = []
        self.selected_rect_idx = None
        self.start_frame = 0
        self.end_frame = None
        self.out_paths = []
        self.image_label.setText("Drag & Drop Video File Here") # Reset text
        self.image_label.setPixmap(QtGui.QPixmap()) # Clear image
        self.update_frame_label(reset=True)
        self.update_rect_list()
        self.brightness_display_label.setText("N/A")
        self.results_label.setText("Load a video to begin analysis.")
        self.frame_slider.setRange(0, 0)
        self.frame_spinbox.setRange(0, 0)
        self._update_widget_states(video_loaded=False, analysis_done=False, rois_exist=False)


    def slider_frame_changed(self, value: int):
        """Handles frame changes initiated by the slider."""
        if self.cap and self.cap.isOpened() and value != self.current_frame_index:
            self._seek_to_frame(value)
            # Sync spinbox without triggering its signal
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(value)
            self.frame_spinbox.blockSignals(False)

    def spinbox_frame_changed(self, value: int):
        """Handles frame changes initiated by the spinbox."""
        if self.cap and self.cap.isOpened() and value != self.current_frame_index:
            # Sync slider, which will trigger slider_frame_changed -> _seek_to_frame
            self.frame_slider.setValue(value)

    def step_frames(self, delta: int):
        """Moves forward or backward by a specified number of frames."""
        if not self.cap or not self.cap.isOpened() or self.total_frames == 0:
            return
        new_idx = max(0, min(self.total_frames - 1, self.current_frame_index + delta))
        if new_idx != self.current_frame_index:
            self.frame_slider.setValue(new_idx) # Let slider signal handle the update

    def _seek_to_frame(self, frame_index: int):
        """Reads and displays the specified frame index."""
        if not self.cap or not self.cap.isOpened():
            return
        if frame_index < 0 or frame_index >= self.total_frames:
            print(f"Warning: Attempted to seek to invalid frame index {frame_index}")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self.current_frame_index = frame_index
            self.show_frame()
            self.update_frame_label()
            self._update_current_brightness_display() # Update brightness for the new frame
        else:
            # Error reading frame, maybe log this?
            print(f"Warning: Failed to read frame at index {frame_index}")
            # Optionally disable controls or show an error?
            # For now, just don't update the display
            pass

    def update_frame_label(self, reset=False):
        """Updates the frame counter label (e.g., "Frame: 10 / 100")."""
        if reset or self.total_frames == 0:
            self.frame_label.setText("Frame: 0 / 0")
        else:
            # Display 1-based index for user-friendliness
            self.frame_label.setText(f"Frame: {self.current_frame_index + 1} / {self.total_frames}")

    def show_frame(self):
        """Displays the current self.frame in the image_label, drawing ROIs."""
        if self.frame is None:
            # Keep the placeholder text if no frame is loaded
            # self.image_label.setText("No video loaded")
            return

        frame_copy = self.frame.copy()
        self._draw_rois(frame_copy)

        # Convert to QPixmap for display
        rgb_image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)

        # Scale pixmap to fit label while maintaining aspect ratio
        target_size = self._current_image_size if self._current_image_size and self._current_image_size.isValid() else self.image_label.size()
        if target_size.isValid() and not target_size.isEmpty():
            scaled_pixmap = pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            # Fallback if target size is invalid (e.g., during init)
            self.image_label.setPixmap(pixmap)


    def _draw_rois(self, frame_to_draw_on):
        """Draws all defined ROIs and the currently drawing ROI onto the frame."""
        # Draw existing rectangles
        for idx, (pt1, pt2) in enumerate(self.rects):
            color = ROI_COLORS[idx % len(ROI_COLORS)]
            thickness = ROI_THICKNESS_SELECTED if idx == self.selected_rect_idx else ROI_THICKNESS_DEFAULT
            cv2.rectangle(frame_to_draw_on, pt1, pt2, color, thickness)
            # Draw index label near the top-left corner
            label = f"{idx+1}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, ROI_LABEL_FONT_SCALE, ROI_LABEL_THICKNESS)
            label_pos = (pt1[0] + 5, pt1[1] + text_height + 5)
            # Simple background for label visibility
            cv2.rectangle(frame_to_draw_on, (pt1[0], pt1[1]), (pt1[0] + text_width + 10, pt1[1] + text_height + 10), (0,0,0), cv2.FILLED)
            cv2.putText(frame_to_draw_on, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, ROI_LABEL_FONT_SCALE, color, ROI_LABEL_THICKNESS, cv2.LINE_AA)

        # Draw rectangle currently being drawn
        if self.drawing and self.start_point and self.end_point:
            # Map points from label coordinates to frame coordinates
            pt1_frame, pt2_frame = self._map_label_to_frame_rect(self.start_point, self.end_point)
            if pt1_frame and pt2_frame:
                cv2.rectangle(frame_to_draw_on, pt1_frame, pt2_frame, (0, 255, 255), ROI_THICKNESS_DEFAULT) # Use a distinct color (cyan)

    def _update_current_brightness_display(self):
        """Calculates and displays the average brightness for the current frame's ROIs."""
        if self.frame is None or not self.rects:
            self.brightness_display_label.setText("N/A")
            return

        brightness_values = []
        fh, fw = self.frame.shape[:2]
        for pt1, pt2 in self.rects:
            # Ensure ROI coordinates are valid within the frame
            x1 = max(0, min(pt1[0], fw - 1))
            y1 = max(0, min(pt1[1], fh - 1))
            x2 = max(0, min(pt2[0], fw - 1))
            y2 = max(0, min(pt2[1], fh - 1))

            if x2 > x1 and y2 > y1: # Check for valid ROI area
                roi = self.frame[y1:y2, x1:x2]
                brightness_values.append(self._compute_brightness(roi))
            else:
                brightness_values.append(0.0) # Append 0 if ROI is invalid/empty

        if brightness_values:
            # Display brightness for each ROI individually
            display_text = ", ".join([f"R{i+1}: {val:.1f}" for i, val in enumerate(brightness_values)])
            self.brightness_display_label.setText(display_text)
        else:
            self.brightness_display_label.setText("N/A")


    # --- ROI Management ---

    def update_rect_list(self):
        """Updates the QListWidget displaying the ROIs."""
        self.rect_list.blockSignals(True) # Prevent selection signals during update
        current_row = self.rect_list.currentRow() # Remember selection
        self.rect_list.clear()
        for idx, (pt1, pt2) in enumerate(self.rects):
            x1, y1 = pt1
            x2, y2 = pt2
            # Ensure coordinates are ordered correctly for display
            disp_x1, disp_y1 = min(x1, x2), min(y1, y2)
            disp_x2, disp_y2 = max(x1, x2), max(y1, y2)
            self.rect_list.addItem(f"ROI {idx+1}: ({disp_x1},{disp_y1}) - ({disp_x2},{disp_y2})")

        # Restore selection if possible
        if 0 <= current_row < len(self.rects):
             self.rect_list.setCurrentRow(current_row)
        elif len(self.rects) > 0:
             # If previous selection invalid, select the last one if available
             self.rect_list.setCurrentRow(len(self.rects) - 1)
        else:
             self.selected_rect_idx = None # No items, no selection

        self.rect_list.blockSignals(False)
        self._update_widget_states(video_loaded=bool(self.cap), rois_exist=bool(self.rects))


    def toggle_add_rectangle_mode(self, checked: bool):
        """Enters or exits the mode for drawing a new ROI."""
        self.drawing = checked
        if checked:
            self.selected_rect_idx = None # Deselect any existing rectangle
            self.update_rect_list() # Update list to show no selection
            self.image_label.setCursor(QtCore.Qt.CrossCursor) # Change cursor
            self.results_label.setText("Click and drag on the frame to draw a new ROI.")
        else:
            self.image_label.unsetCursor() # Restore default cursor
            # Optionally clear the results label or set it back
            # self.results_label.setText("...")
        self.show_frame() # Redraw to potentially remove selection highlight

    def select_rectangle_from_list(self, row: int):
        """Handles selection changes in the ROI list widget."""
        if 0 <= row < len(self.rects):
            self.selected_rect_idx = row
            self.drawing = False # Exit drawing mode if active
            self.add_rect_btn.setChecked(False)
            self.image_label.unsetCursor()
        else:
            self.selected_rect_idx = None
        self._update_widget_states(video_loaded=bool(self.cap), rois_exist=bool(self.rects))
        self.show_frame() # Redraw to highlight selected rectangle

    def delete_selected_rectangle(self):
        """Deletes the currently selected ROI."""
        if self.selected_rect_idx is not None and 0 <= self.selected_rect_idx < len(self.rects):
            del self.rects[self.selected_rect_idx]
            # Adjust selection if the deleted item wasn't the last one
            if self.selected_rect_idx >= len(self.rects) and len(self.rects) > 0:
                 self.selected_rect_idx = len(self.rects) - 1
            elif len(self.rects) == 0:
                 self.selected_rect_idx = None
            # No need to explicitly set selection index otherwise, update_rect_list handles it
            self.update_rect_list()
            self.show_frame()

    def clear_all_rectangles(self):
        """Removes all defined ROIs."""
        if not self.rects: return # Nothing to clear
        reply = QtWidgets.QMessageBox.question(self, 'Confirm Clear',
                                               'Are you sure you want to delete all ROIs?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.rects.clear()
            self.selected_rect_idx = None
            self.update_rect_list()
            self.show_frame()

    # --- Mouse Interaction on Image Label ---

    def image_mouse_press(self, event: QtGui.QMouseEvent):
        """Handles mouse clicks on the video display area."""
        if self.frame is None or event.button() != QtCore.Qt.LeftButton:
            return # Ignore if no video or not left click

        pos_in_label = event.pos() # Position relative to the image_label widget

        # Check if the click is within the actual displayed pixmap area
        pixmap_rect = self._get_pixmap_rect_in_label()
        if not pixmap_rect or not pixmap_rect.contains(pos_in_label):
             return # Click was outside the image area (in borders/empty space)

        # Convert click position to frame coordinates
        frame_x, frame_y = self._map_label_to_frame_point(pos_in_label)
        if frame_x is None: return # Mapping failed

        if self.drawing:
            # Start drawing a new rectangle
            self.start_point = pos_in_label # Store label coordinates for drawing feedback
            self.end_point = pos_in_label
            self.moving = False
            self.resizing = False
        elif self.selected_rect_idx is not None:
            # Check if clicking near a corner of the selected rectangle for resizing
            pt1, pt2 = self.rects[self.selected_rect_idx]
            corners = [
                (pt1[0], pt1[1]), (pt2[0], pt1[1]), # Top-left, Top-right
                (pt1[0], pt2[1]), (pt2[0], pt2[1])  # Bottom-left, Bottom-right
            ]
            resize_margin = self._scale_value_for_pixmap(MOUSE_RESIZE_HANDLE_SENSITIVITY) # Scale sensitivity

            for i, (cx, cy) in enumerate(corners):
                if abs(frame_x - cx) < resize_margin and abs(frame_y - cy) < resize_margin:
                    self.resizing = True
                    self.resize_corner = i # Store which corner (0=TL, 1=TR, 2=BL, 3=BR)
                    # Determine the opposite corner, which stays fixed during resize
                    fixed_corner_index = 3 - i
                    self.start_point = self._map_frame_to_label_point(corners[fixed_corner_index]) # Store fixed corner in label coords
                    self.end_point = pos_in_label # Moving point is the current mouse pos
                    self.moving = False
                    self.drawing = False
                    self.image_label.setCursor(self._get_resize_cursor(i)) # Set resize cursor
                    self.show_frame()
                    return # Resizing initiated

            # Check if clicking inside the selected rectangle for moving
            pt1, pt2 = self.rects[self.selected_rect_idx]
            rect_x1, rect_y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
            rect_x2, rect_y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
            if rect_x1 <= frame_x <= rect_x2 and rect_y1 <= frame_y <= rect_y2:
                self.moving = True
                # Calculate offset from top-left corner of the rect
                self.move_offset = (frame_x - rect_x1, frame_y - rect_y1)
                self.start_point = pos_in_label # Store initial position for smooth dragging
                self.resizing = False
                self.drawing = False
                self.image_label.setCursor(QtCore.Qt.SizeAllCursor) # Set move cursor
                self.show_frame()
                return # Moving initiated

        # If click wasn't for drawing, resizing, or moving a selected rect,
        # check if it hit any *other* rectangle to select it.
        clicked_on_rect_idx = -1
        for idx, (pt1, pt2) in enumerate(self.rects):
             rect_x1, rect_y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
             rect_x2, rect_y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
             if rect_x1 <= frame_x <= rect_x2 and rect_y1 <= frame_y <= rect_y2:
                  clicked_on_rect_idx = idx
                  break # Found a rect, stop checking

        if clicked_on_rect_idx != -1 and clicked_on_rect_idx != self.selected_rect_idx:
             # Clicked on a different rectangle, select it
             self.selected_rect_idx = clicked_on_rect_idx
             self.drawing = False # Ensure not in drawing mode
             self.add_rect_btn.setChecked(False)
             self.image_label.unsetCursor()
             self.update_rect_list() # Update list selection
             self.show_frame() # Redraw to highlight new selection
        elif clicked_on_rect_idx == -1 and not self.drawing:
             # Clicked outside any rectangle and not drawing, deselect
             self.selected_rect_idx = None
             self.update_rect_list()
             self.show_frame()


    def image_mouse_move(self, event: QtGui.QMouseEvent):
        """Handles mouse movement over the video display area."""
        if self.frame is None: return

        pos_in_label = event.pos()
        frame_h, frame_w = self.frame.shape[:2]

        if self.drawing and self.start_point:
            # Update the end point for drawing feedback
            # Clamp position to within the label bounds
            clamped_x = max(0, min(pos_in_label.x(), self.image_label.width() - 1))
            clamped_y = max(0, min(pos_in_label.y(), self.image_label.height() - 1))
            self.end_point = QtCore.QPoint(clamped_x, clamped_y)
            self.show_frame() # Redraw to show the rectangle being drawn

        elif self.moving and self.selected_rect_idx is not None and self.start_point:
            # Move the selected rectangle
            frame_x, frame_y = self._map_label_to_frame_point(pos_in_label)
            if frame_x is None: return # Mapping failed

            pt1, pt2 = self.rects[self.selected_rect_idx]
            orig_w = abs(pt2[0] - pt1[0])
            orig_h = abs(pt2[1] - pt1[1])

            # Calculate new top-left based on mouse position and initial offset
            new_x1 = frame_x - self.move_offset[0]
            new_y1 = frame_y - self.move_offset[1]

            # Clamp new position to stay within frame boundaries
            new_x1 = max(0, min(new_x1, frame_w - orig_w))
            new_y1 = max(0, min(new_y1, frame_h - orig_h))
            new_x2 = new_x1 + orig_w
            new_y2 = new_y1 + orig_h

            self.rects[self.selected_rect_idx] = ((new_x1, new_y1), (new_x2, new_y2))
            self.update_rect_list() # Update coordinates in the list
            self.show_frame()

        elif self.resizing and self.selected_rect_idx is not None and self.start_point:
            # Resize the selected rectangle
            frame_x, frame_y = self._map_label_to_frame_point(pos_in_label)
            if frame_x is None: return # Mapping failed

            # Clamp mouse position to frame boundaries before calculating new rect
            frame_x = max(0, min(frame_x, frame_w - 1))
            frame_y = max(0, min(frame_y, frame_h - 1))

            fixed_corner_frame = self._map_label_to_frame_point(self.start_point)
            if fixed_corner_frame[0] is None: return # Mapping failed

            # New rectangle is defined by the fixed corner and the current mouse pos
            new_x1 = min(fixed_corner_frame[0], frame_x)
            new_y1 = min(fixed_corner_frame[1], frame_y)
            new_x2 = max(fixed_corner_frame[0], frame_x)
            new_y2 = max(fixed_corner_frame[1], frame_y)

            # Prevent zero-size rectangles (optional, but good practice)
            if new_x1 < new_x2 and new_y1 < new_y2:
                self.rects[self.selected_rect_idx] = ((new_x1, new_y1), (new_x2, new_y2))
                self.update_rect_list()
                self.show_frame()
        else:
            # Update cursor if hovering over a resize handle or a selectable rectangle
            self._update_hover_cursor(pos_in_label)


    def image_mouse_release(self, event: QtGui.QMouseEvent):
        """Handles mouse button releases on the video display area."""
        if self.frame is None or event.button() != QtCore.Qt.LeftButton:
            return

        pos_in_label = event.pos()
        self.image_label.unsetCursor() # Reset cursor after action

        if self.drawing and self.start_point and self.end_point:
            # Finalize drawing a new rectangle
            pt1_frame, pt2_frame = self._map_label_to_frame_rect(self.start_point, self.end_point)

            # Add rectangle only if it has a valid size
            if pt1_frame and pt2_frame and abs(pt1_frame[0] - pt2_frame[0]) > 1 and abs(pt1_frame[1] - pt2_frame[1]) > 1:
                # Ensure pt1 is top-left and pt2 is bottom-right
                final_x1 = min(pt1_frame[0], pt2_frame[0])
                final_y1 = min(pt1_frame[1], pt2_frame[1])
                final_x2 = max(pt1_frame[0], pt2_frame[0])
                final_y2 = max(pt1_frame[1], pt2_frame[1])
                self.rects.append(((final_x1, final_y1), (final_x2, final_y2)))
                self.selected_rect_idx = len(self.rects) - 1 # Select the new rectangle
                self.update_rect_list() # Update list and selection

            # Reset drawing state
            self.drawing = False
            self.start_point = None
            self.end_point = None
            self.add_rect_btn.setChecked(False) # Uncheck button
            self.show_frame() # Redraw

        elif self.moving:
            # Finalize moving
            self.moving = False
            self.move_offset = None
            self.start_point = None
            # Optional: Recalculate brightness display for the final position
            self._update_current_brightness_display()
            self.show_frame() # Redraw in final state

        elif self.resizing:
            # Finalize resizing
            self.resizing = False
            self.resize_corner = None
            self.start_point = None
            self.end_point = None
            # Optional: Recalculate brightness display for the final size
            self._update_current_brightness_display()
            self.show_frame() # Redraw in final state

    def _get_pixmap_rect_in_label(self) -> QtCore.QRect | None:
        """Calculates the QRect occupied by the scaled pixmap within the image label."""
        if not hasattr(self, 'image_label') or self.frame is None:
            return None

        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap() # Get the currently displayed pixmap

        if not pixmap or pixmap.isNull() or not label_size.isValid() or label_size.isEmpty():
            return None # No pixmap or invalid label size

        pixmap_size = pixmap.size() # Size of the scaled pixmap

        # Calculate top-left corner of the pixmap within the label (centered)
        offset_x = (label_size.width() - pixmap_size.width()) / 2
        offset_y = (label_size.height() - pixmap_size.height()) / 2

        return QtCore.QRect(int(offset_x), int(offset_y), pixmap_size.width(), pixmap_size.height())

    def _map_label_to_frame_point(self, label_pos: QtCore.QPoint) -> tuple[int | None, int | None]:
        """Maps a point from image label coordinates to original frame coordinates."""
        if self.frame is None: return None, None

        pixmap_rect = self._get_pixmap_rect_in_label()
        if not pixmap_rect: return None, None

        # Check if the point is actually within the pixmap area
        if not pixmap_rect.contains(label_pos):
            return None, None

        # Point relative to the top-left of the pixmap
        relative_x = label_pos.x() - pixmap_rect.left()
        relative_y = label_pos.y() - pixmap_rect.top()

        # Scale factors from pixmap size to original frame size
        frame_h, frame_w = self.frame.shape[:2]
        pixmap_w = pixmap_rect.width()
        pixmap_h = pixmap_rect.height()

        if pixmap_w == 0 or pixmap_h == 0: return None, None # Avoid division by zero

        scale_w = frame_w / pixmap_w
        scale_h = frame_h / pixmap_h

        # Calculate corresponding point in the original frame
        frame_x = int(relative_x * scale_w)
        frame_y = int(relative_y * scale_h)

        # Clamp to frame boundaries (shouldn't be necessary if logic is correct, but safe)
        frame_x = max(0, min(frame_x, frame_w - 1))
        frame_y = max(0, min(frame_y, frame_h - 1))

        return frame_x, frame_y

    def _map_label_to_frame_rect(self, label_pt1: QtCore.QPoint, label_pt2: QtCore.QPoint) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
         """Maps a rectangle defined by two points in label coordinates to frame coordinates."""
         frame_pt1 = self._map_label_to_frame_point(label_pt1)
         frame_pt2 = self._map_label_to_frame_point(label_pt2)

         if frame_pt1[0] is None or frame_pt2[0] is None:
              return None, None
         return frame_pt1, frame_pt2

    def _map_frame_to_label_point(self, frame_pos: tuple[int, int]) -> QtCore.QPoint | None:
        """Maps a point from original frame coordinates back to image label coordinates."""
        if self.frame is None: return None

        pixmap_rect = self._get_pixmap_rect_in_label()
        if not pixmap_rect: return None

        frame_h, frame_w = self.frame.shape[:2]
        pixmap_w = pixmap_rect.width()
        pixmap_h = pixmap_rect.height()

        if frame_w == 0 or frame_h == 0: return None # Avoid division by zero

        # Scale factors from frame size to pixmap size
        scale_w = pixmap_w / frame_w
        scale_h = pixmap_h / frame_h

        # Calculate position relative to pixmap top-left
        relative_x = frame_pos[0] * scale_w
        relative_y = frame_pos[1] * scale_h

        # Calculate position within the label widget
        label_x = int(pixmap_rect.left() + relative_x)
        label_y = int(pixmap_rect.top() + relative_y)

        return QtCore.QPoint(label_x, label_y)

    def _scale_value_for_pixmap(self, value_in_frame_coords: float) -> float:
        """Scales a value (like a distance) from frame coordinates to pixmap coordinates."""
        if self.frame is None: return value_in_frame_coords # No scaling if no frame

        pixmap_rect = self._get_pixmap_rect_in_label()
        if not pixmap_rect or pixmap_rect.width() == 0: return value_in_frame_coords

        frame_w = self.frame.shape[1]
        scale_w = pixmap_rect.width() / frame_w
        return value_in_frame_coords * scale_w # Assume uniform scaling for simplicity

    def _get_resize_cursor(self, corner_index: int) -> QtGui.QCursor:
        """Returns the appropriate resize cursor based on the corner index."""
        if corner_index == 0 or corner_index == 3: # Top-left or Bottom-right
            return QtGui.QCursor(QtCore.Qt.SizeFDiagCursor)
        elif corner_index == 1 or corner_index == 2: # Top-right or Bottom-left
            return QtGui.QCursor(QtCore.Qt.SizeBDiagCursor)
        else:
            return QtGui.QCursor(QtCore.Qt.ArrowCursor) # Default

    def _update_hover_cursor(self, pos_in_label: QtCore.QPoint):
        """Sets the cursor shape based on what the mouse is hovering over."""
        if self.frame is None or self.drawing or self.moving or self.resizing:
            # Don't change cursor if an action is in progress or no video
            return

        frame_x, frame_y = self._map_label_to_frame_point(pos_in_label)
        if frame_x is None:
            self.image_label.unsetCursor()
            return # Outside pixmap

        cursor_set = False

        # Check for resize handles on the selected rectangle first
        if self.selected_rect_idx is not None:
            pt1, pt2 = self.rects[self.selected_rect_idx]
            corners = [(pt1[0], pt1[1]), (pt2[0], pt1[1]), (pt1[0], pt2[1]), (pt2[0], pt2[1])]
            resize_margin = self._scale_value_for_pixmap(MOUSE_RESIZE_HANDLE_SENSITIVITY)

            for i, (cx, cy) in enumerate(corners):
                if abs(frame_x - cx) < resize_margin and abs(frame_y - cy) < resize_margin:
                    self.image_label.setCursor(self._get_resize_cursor(i))
                    cursor_set = True
                    break

        # If not hovering over a resize handle, check if hovering over any rectangle
        if not cursor_set:
            hovering_over_rect = False
            for idx, (pt1, pt2) in enumerate(self.rects):
                rect_x1, rect_y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
                rect_x2, rect_y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
                if rect_x1 <= frame_x <= rect_x2 and rect_y1 <= frame_y <= rect_y2:
                    hovering_over_rect = True
                    if idx == self.selected_rect_idx:
                        # Hovering over the selected rectangle (not on a handle) -> Move cursor
                        self.image_label.setCursor(QtCore.Qt.SizeAllCursor)
                    else:
                        # Hovering over a non-selected rectangle -> Pointer cursor
                        self.image_label.setCursor(QtCore.Qt.PointingHandCursor)
                    cursor_set = True
                    break

        # If not hovering over anything specific, reset to default
        if not cursor_set:
            self.image_label.unsetCursor()


    # --- Frame Range Selection ---

    def set_start_frame(self):
        """Sets the current frame as the start frame for analysis."""
        if self.cap and self.cap.isOpened():
            self.start_frame = self.current_frame_index
            self.results_label.setText(f"Start frame set to {self.start_frame + 1}") # Display 1-based index
            # Ensure end frame is not before start frame
            if self.end_frame is not None and self.end_frame < self.start_frame:
                self.end_frame = self.start_frame
                self.results_label.setText(f"Start frame set to {self.start_frame + 1}. End frame adjusted.")

    def set_end_frame(self):
        """Sets the current frame as the end frame for analysis."""
        if self.cap and self.cap.isOpened():
            self.end_frame = self.current_frame_index
            self.results_label.setText(f"End frame set to {self.end_frame + 1}") # Display 1-based index
            # Ensure start frame is not after end frame
            if self.start_frame > self.end_frame:
                self.start_frame = self.end_frame
                self.results_label.setText(f"End frame set to {self.end_frame + 1}. Start frame adjusted.")

    def auto_detect_range(self):
        """
        Scans the video to automatically find the first and last frames
        where ROI brightness significantly exceeds a baseline.
        """
        if not self.video_path:
             self.results_label.setText("Load a video first.")
             return
        if not self.rects:
            self.results_label.setText("Draw at least one ROI before using Auto-Detect.")
            return

        # Use a separate capture object for scanning to avoid interfering with main display
        scan_cap = cv2.VideoCapture(self.video_path)
        if not scan_cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Could not open video for auto-detection scan.")
            return

        total = int(scan_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
             QtWidgets.QMessageBox.warning(self, "Auto-Detect", "Video appears to have no frames for scanning.")
             scan_cap.release()
             return

        brightness_per_frame = np.zeros(total, dtype=np.float32)

        # Progress Dialog
        progress = QtWidgets.QProgressDialog("Scanning video for brightness changes...", "Cancel", 0, total, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Auto-Detecting Range")
        progress.setValue(0)
        progress.show()
        QtWidgets.QApplication.processEvents() # Ensure dialog shows up

        scan_cancelled = False
        for idx in range(total):
            if progress.wasCanceled():
                scan_cancelled = True
                break

            ret, frame = scan_cap.read()
            if not ret:
                total = idx # Adjust total if read failed early
                brightness_per_frame = brightness_per_frame[:total] # Trim array
                print(f"Warning: Auto-detect scan stopped early at frame {idx} due to read error.")
                break

            # Calculate average brightness across *all* defined ROIs for this frame
            current_frame_roi_brightness = []
            fh, fw = frame.shape[:2]
            for pt1, pt2 in self.rects:
                x1, y1 = max(0, pt1[0]), max(0, pt1[1])
                x2, y2 = min(fw - 1, pt2[0]), min(fh - 1, pt2[1])
                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    current_frame_roi_brightness.append(self._compute_brightness(roi))

            # Store the mean brightness of all ROIs for the current frame
            brightness_per_frame[idx] = np.mean(current_frame_roi_brightness) if current_frame_roi_brightness else 0.0

            progress.setValue(idx + 1)
            if idx % 10 == 0: # Update UI periodically to keep it responsive
                 QtWidgets.QApplication.processEvents()

        scan_cap.release()
        progress.close()

        if scan_cancelled:
            self.results_label.setText("Auto-detect scan cancelled.")
            return

        if brightness_per_frame.size == 0:
            QtWidgets.QMessageBox.warning(self, "Auto-Detect", "Scan completed, but no brightness data was gathered.")
            return

        # --- Analyze Brightness Data ---
        try:
            # Calculate baseline (e.g., 5th percentile)
            baseline = np.percentile(brightness_per_frame, AUTO_DETECT_BASELINE_PERCENTILE)
            # Set threshold slightly above baseline
            threshold = baseline + AUTO_DETECT_THRESHOLD_L_UNITS
        except IndexError:
             QtWidgets.QMessageBox.warning(self, "Auto-Detect", "Not enough frame data to determine brightness baseline.")
             return

        # Find indices where brightness is above the threshold
        above_threshold_indices = np.where(brightness_per_frame >= threshold)[0]

        if above_threshold_indices.size == 0:
            QtWidgets.QMessageBox.information(self, "Auto-Detect",
                                          f"No frames found significantly brighter than baseline (Threshold={threshold:.1f} L*). Frame range not changed.")
            self.results_label.setText(f"Auto-Detect: No bright frames found (Threshold={threshold:.1f}).")
            return

        # Get the first and last frame index that met the criteria
        detected_start_frame = int(above_threshold_indices[0])
        detected_end_frame = int(above_threshold_indices[-1])

        # --- Update UI ---
        self.start_frame = detected_start_frame
        self.end_frame = detected_end_frame

        # Update slider and spinbox to the new start frame, then seek
        self.frame_slider.blockSignals(True)
        self.frame_spinbox.blockSignals(True)
        self.frame_slider.setValue(self.start_frame)
        self.frame_spinbox.setValue(self.start_frame)
        self.frame_slider.blockSignals(False)
        self.frame_spinbox.blockSignals(False)

        self._seek_to_frame(self.start_frame) # Go to the detected start frame
        self.update_frame_label() # Update label after seeking

        self.results_label.setText(f"✅ Auto-detected range: Frame {self.start_frame + 1} to {self.end_frame + 1}")


    # --- Analysis and Plotting ---

    def analyze_video(self):
        """
        Performs brightness analysis over the selected frame range for each ROI
        and saves the results to CSV files.
        """
        # --- Preconditions ---
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, "Analysis Error", "Please load a video file first.")
            return
        if not self.rects:
            QtWidgets.QMessageBox.warning(self, "Analysis Error", "Please define at least one ROI.")
            return
        if self.start_frame is None or self.end_frame is None or self.start_frame > self.end_frame:
             QtWidgets.QMessageBox.warning(self, "Analysis Error", "Invalid start/end frame range selected.")
             return

        # --- Get Save Directory ---
        # Suggest the directory of the video file as a starting point
        initial_dir = os.path.dirname(self.video_path) if self.video_path else ""
        save_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Directory to Save Analysis Results", initial_dir)
        if not save_dir:
            self.results_label.setText("Analysis cancelled (no save directory chosen).")
            return

        # --- Setup ---
        self.results_label.setText("Starting analysis...")
        self.brightness_display_label.setText("Analyzing...") # Indicate analysis in progress
        QtWidgets.QApplication.processEvents() # Update UI

        analysis_cap = cv2.VideoCapture(self.video_path)
        if not analysis_cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not open video file for analysis: {os.path.basename(self.video_path)}")
            self.results_label.setText("Error opening video for analysis.")
            self.brightness_display_label.setText("Error")
            return

        # Frame range for analysis (inclusive)
        start = self.start_frame
        end = self.end_frame
        num_frames_to_analyze = end - start + 1

        # Data structure: list of lists, one inner list per ROI
        brightness_data = [[] for _ in self.rects]

        # --- Progress Dialog ---
        progress = QtWidgets.QProgressDialog("Analyzing video frames...", "Cancel", 0, num_frames_to_analyze, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Analyzing Brightness")
        progress.setValue(0)
        progress.show()
        QtWidgets.QApplication.processEvents()

        # --- Frame Processing Loop ---
        analysis_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        analysis_cancelled = False
        frames_processed = 0

        for f_idx in range(start, end + 1):
            if progress.wasCanceled():
                analysis_cancelled = True
                break

            ret, frame = analysis_cap.read()
            if not ret:
                print(f"Warning: Could not read frame {f_idx} during analysis. Stopping analysis.")
                # Adjust the number of frames if we stopped early
                num_frames_to_analyze = frames_processed
                # Trim data lists to match processed frames
                brightness_data = [lst[:frames_processed] for lst in brightness_data]
                break

            fh, fw = frame.shape[:2]
            for r_idx, (pt1, pt2) in enumerate(self.rects):
                # Ensure ROI coords are valid for this frame
                x1, y1 = max(0, pt1[0]), max(0, pt1[1])
                x2, y2 = min(fw - 1, pt2[0]), min(fh - 1, pt2[1])

                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    brightness_data[r_idx].append(self._compute_brightness(roi))
                else:
                    brightness_data[r_idx].append(0.0) # Append 0 for invalid ROI size

            frames_processed += 1
            progress.setValue(frames_processed)
            if frames_processed % 10 == 0: # Keep UI responsive
                 QtWidgets.QApplication.processEvents()

        analysis_cap.release()
        progress.close()

        if analysis_cancelled:
            self.results_label.setText("Analysis cancelled by user.")
            self._update_current_brightness_display() # Restore brightness display
            return

        if frames_processed == 0:
             QtWidgets.QMessageBox.warning(self, "Analysis", "No frames were processed during analysis.")
             self.results_label.setText("Analysis completed, but no frames processed.")
             self._update_current_brightness_display()
             return

        # --- Save Results ---
        self.out_paths = [] # Clear previous paths
        base_video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        analysis_name = self.analysis_name_input.text().strip() or "DefaultAnalysis"
        analysis_name = "".join(c for c in analysis_name if c.isalnum() or c in ('_', '-')).rstrip() # Sanitize name

        summary_lines = [f"Analysis Complete ({frames_processed} frames analyzed):"]
        avg_brightness_summary = []

        for r_idx, data in enumerate(brightness_data):
            if not data: continue # Skip if no data for this ROI

            # Create DataFrame
            frame_numbers = range(start, start + len(data)) # Use actual frame numbers
            df = pd.DataFrame({"frame": frame_numbers, "brightness": data})

            # Calculate average for summary
            avg_val = np.mean(data)
            avg_brightness_summary.append(f"ROI {r_idx+1} Avg: {avg_val:.2f}")

            # Construct filename and save CSV
            out_file = f"{analysis_name}_{base_video_name}_ROI{r_idx+1}_frames{start+1}-{start+len(data)}_brightness.csv"
            out_path = os.path.join(save_dir, out_file)
            try:
                df.to_csv(out_path, index=False)
                self.out_paths.append(out_path)
                summary_lines.append(f" - Saved: {out_file}")
            except Exception as e:
                error_msg = f"Failed to save CSV for ROI {r_idx+1}:\n{e}"
                QtWidgets.QMessageBox.critical(self, "Error Saving File", error_msg)
                summary_lines.append(f" - FAILED to save ROI {r_idx+1} data.")

        # --- Update UI ---
        self.results_label.setText("\n".join(summary_lines))
        self.brightness_display_label.setText(", ".join(avg_brightness_summary) if avg_brightness_summary else "N/A")
        self._update_widget_states(video_loaded=True, analysis_done=bool(self.out_paths), rois_exist=bool(self.rects))


    def plot_results(self):
        """
        Generates and displays plots for the saved analysis results (CSV files).
        Prompts the user for a directory to save the plot images.
        """
        if not hasattr(self, 'out_paths') or not self.out_paths:
            QtWidgets.QMessageBox.information(self, "Plotting", "No analysis results found to plot. Run analysis first.")
            return

        # --- Get Save Directory for Plots ---
        initial_dir = os.path.dirname(self.out_paths[0]) if self.out_paths else ""
        plot_save_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Directory to Save Plots", initial_dir)
        if not plot_save_dir:
            self.results_label.setText("Plotting cancelled (no save directory chosen).")
            return

        self.results_label.setText("Generating plots...")
        QtWidgets.QApplication.processEvents()

        plot_paths = []
        plot_failed = False
        for csv_path in self.out_paths:
            try:
                df = pd.read_csv(csv_path)
                if 'frame' not in df.columns or 'brightness' not in df.columns:
                     print(f"Warning: Skipping plot for {os.path.basename(csv_path)} - missing required columns.")
                     continue

                frames = df['frame']
                brightness = df['brightness']

                if brightness.empty:
                     print(f"Warning: Skipping plot for {os.path.basename(csv_path)} - no brightness data.")
                     continue

                # Find peak and mean
                idx_peak = brightness.idxmax()
                frame_peak, val_peak = frames.iloc[idx_peak], brightness.iloc[idx_peak]
                mean_brightness = brightness.mean()

                plt.style.use('seaborn-v0_8-darkgrid') # Use a nice plotting style
                fig, ax = plt.subplots(figsize=(10, 5)) # Create figure and axes

                ax.plot(frames, brightness, label='Avg. Brightness', color=COLOR_ACCENT, linewidth=1.5)
                ax.axhline(mean_brightness, color=COLOR_WARNING, linestyle='--', label=f'Mean ({mean_brightness:.1f})')
                ax.scatter([frame_peak], [val_peak], color=COLOR_ERROR, zorder=5, label=f'Peak ({val_peak:.1f})')

                # Annotate peak point
                ax.annotate(f'Peak\nFrame {frame_peak}\n({val_peak:.1f})',
                            xy=(frame_peak, val_peak),
                            xytext=(0, 15), textcoords='offset points',
                            ha='center', va='bottom', color=COLOR_ERROR,
                            arrowprops=dict(arrowstyle="->", color=COLOR_ERROR))

                # Customize plot
                plot_title = os.path.basename(csv_path).replace('_brightness.csv', '')
                ax.set_title(plot_title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Frame Number', fontsize=12)
                ax.set_ylabel('Average L* Brightness', fontsize=12)
                ax.legend(fontsize=10)
                ax.tick_params(axis='both', which='major', labelsize=10)
                fig.tight_layout() # Adjust layout

                # Save plot
                plot_filename = os.path.basename(csv_path).replace('.csv', '_plot.png')
                plot_save_path = os.path.join(plot_save_dir, plot_filename)
                plt.savefig(plot_save_path)
                plt.close(fig) # Close the figure to free memory
                plot_paths.append(plot_save_path)

            except Exception as e:
                plot_failed = True
                error_msg = f"Failed to generate plot for {os.path.basename(csv_path)}:\n{e}"
                print(error_msg) # Also print to console
                QtWidgets.QMessageBox.warning(self, "Plotting Error", error_msg)

        # --- Update UI ---
        if plot_paths:
             summary = f"Plots generated and saved to:\n{plot_save_dir}"
             if plot_failed:
                  summary += "\n(Some plots failed to generate - check console/messages for details)"
             self.results_label.setText(summary)
             # Optionally open the plot directory?
             # os.startfile(plot_save_dir) # Windows
             # subprocess.call(['open', plot_save_dir]) # macOS
        elif plot_failed:
             self.results_label.setText("Plotting failed. Check console or messages for errors.")
        else:
             self.results_label.setText("Plotting finished, but no valid data found to plot.")


    # --- Utility Methods ---

    def _compute_brightness(self, roi_bgr: np.ndarray) -> float:
        """
        Calculates a perceptually meaningful brightness (L*) for an ROI.

        Converts BGR to CIE LAB color space and uses the L* channel.
        Discards the darkest pixels (noise floor) before averaging.

        Args:
            roi_bgr: The region of interest as a NumPy array (BGR format).

        Returns:
            The mean L* value (0-100 range, scaled from 0-255 OpenCV output)
            or 0.0 if the ROI is invalid or calculation fails.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0

        try:
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)

            # Convert raw L to L* scale (0–100)
            l_star = l_chan * 100.0 / 255.0

            # Keep only pixels with brightness > 10
            mask = l_star > 10
            if not np.any(mask):
                return 0.0

            # Return the mean of the remaining pixels
            return float(np.mean(l_star[mask]))

        except cv2.error as e:
            print(f"OpenCV error during brightness computation: {e}")
            return 0.0
        except Exception as e:
            print(f"Error during brightness computation: {e}")
            return 0.0


# --- Main Execution ---
if __name__ == '__main__':
    # Ensure high DPI scaling is handled correctly
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    analyzer_window = VideoAnalyzer()
    analyzer_window.show()
    sys.exit(app.exec_())