import sys
import pandas as pd
import numpy as np
import cv2
import os
import json
import time
from typing import Optional, Tuple, List, Union, Dict
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
import logging
from collections import OrderedDict

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_FONT_FAMILY = "Segoe UI, Arial, sans-serif"
COLOR_BACKGROUND = "#2d2d2d"
COLOR_FOREGROUND = "#cccccc"
COLOR_ACCENT = "#5a9bd5"
COLOR_ACCENT_HOVER = "#7ab3e0"
COLOR_SECONDARY = "#404040"
COLOR_SECONDARY_LIGHT = "#555555"
COLOR_SUCCESS = "#70ad47"
COLOR_WARNING = "#ed7d31"
COLOR_ERROR = "#ff0000"
COLOR_INFO = "#ffc000"
COLOR_BRIGHTNESS_LABEL = "#ffeb3b"

ROI_COLORS = [
    (255, 50, 50), (50, 200, 50), (50, 150, 255), (255, 150, 50),
    (255, 50, 255), (50, 200, 200), (150, 50, 255), (255, 255, 50)
]
ROI_THICKNESS_DEFAULT = 2
ROI_THICKNESS_SELECTED = 4
ROI_LABEL_FONT_SCALE = 0.8
ROI_LABEL_THICKNESS = 2

AUTO_DETECT_BASELINE_PERCENTILE = 5
BRIGHTNESS_NOISE_FLOOR_PERCENTILE = 2
DEFAULT_MANUAL_THRESHOLD = 5.0
MORPHOLOGICAL_KERNEL_SIZE = 3

MOUSE_RESIZE_HANDLE_SENSITIVITY = 10

# New constants for improvements
DEFAULT_SETTINGS_FILE = "brightness_analyzer_settings.json"
MAX_RECENT_FILES = 10
FRAME_CACHE_SIZE = 100
JUMP_FRAMES = 10  # Number of frames to jump with Page Up/Down

class FrameCache:
    """Efficient frame caching system for better performance."""
    
    def __init__(self, max_size: int = FRAME_CACHE_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
    
    def get(self, frame_index: int) -> Optional[np.ndarray]:
        """Get frame from cache, moving it to end (most recently used)."""
        if frame_index in self._cache:
            # Move to end (most recently used)
            frame = self._cache.pop(frame_index)
            self._cache[frame_index] = frame
            return frame.copy()  # Return copy to prevent modifications
        return None
    
    def put(self, frame_index: int, frame: np.ndarray):
        """Add frame to cache, removing oldest if necessary."""
        if frame_index in self._cache:
            self._cache.pop(frame_index)
        
        self._cache[frame_index] = frame.copy()
        
        # Remove oldest items if cache is full
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
    
    def clear(self):
        """Clear all cached frames."""
        self._cache.clear()
    
    def get_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

class VideoAnalyzer(QtWidgets.QMainWindow):  # Changed to QMainWindow for better menu support
    """Main application window for video brightness analysis."""
    
    def __init__(self):
        """Initializes the application window and UI elements."""
        super().__init__()
        self._init_vars()
        self._load_settings()
        self._init_ui()
        self._create_menus()

    def _init_vars(self):
        """Initialize instance variables."""
        self.video_path = None
        self.frame = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.cap = None
        self.out_paths = []
        
        # Frame caching
        self.frame_cache = FrameCache(FRAME_CACHE_SIZE)
        
        # Recent files
        self.recent_files = []
        
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
        self._current_image_size = None
        
        # Frame range
        self.start_frame = 0
        self.end_frame = None
        
        # Analysis state
        self._analysis_in_progress = False
        
        # Settings
        self.settings = {}
        
        # Threshold / background
        self.manual_threshold = DEFAULT_MANUAL_THRESHOLD
        self.background_roi_idx = None       # index into self.rects
        
        # Pixel visualization
        self.show_pixel_mask = False

    def _load_settings(self):
        """Load application settings from file."""
        try:
            if os.path.exists(DEFAULT_SETTINGS_FILE):
                with open(DEFAULT_SETTINGS_FILE, 'r') as f:
                    self.settings = json.load(f)
                    self.recent_files = self.settings.get('recent_files', [])
        except Exception as e:
            logging.warning(f"Could not load settings: {e}")
            self.settings = {}
            self.recent_files = []

    def _save_settings(self):
        """Save application settings to file."""
        try:
            self.settings['recent_files'] = self.recent_files
            with open(DEFAULT_SETTINGS_FILE, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save settings: {e}")

    def _add_recent_file(self, file_path: str):
        """Add file to recent files list."""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        self.recent_files = self.recent_files[:MAX_RECENT_FILES]
        self._update_recent_files_menu()

    def _init_ui(self):
        """Set up the main UI layout and widgets."""
        self.setWindowTitle('Brightness Sorcerer v2.0')
        self.setGeometry(100, 100, 1400, 900)  # Larger default size
        self.setAcceptDrops(True)
        self._apply_stylesheet()

        # Create central widget and main layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        self._create_layouts()
        self._create_widgets()
        self._connect_signals()
        self._setup_shortcuts()
        self._update_widget_states()

    def _create_menus(self):
        """Create application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        open_action = QtWidgets.QAction('&Open Video...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open a video file')
        open_action.triggered.connect(self.open_video_dialog)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Recent files submenu
        self.recent_files_menu = file_menu.addMenu('Recent Files')
        self._update_recent_files_menu()
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit the application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('&Analysis')
        
        analyze_action = QtWidgets.QAction('&Run Analysis', self)
        analyze_action.setShortcut('F5')
        analyze_action.setStatusTip('Run brightness analysis')
        analyze_action.triggered.connect(self.analyze_video)
        analysis_menu.addAction(analyze_action)
        
        auto_detect_action = QtWidgets.QAction('&Auto-Detect Range', self)
        auto_detect_action.setShortcut('Ctrl+D')
        auto_detect_action.setStatusTip('Automatically detect frame range')
        auto_detect_action.triggered.connect(self.auto_detect_range)
        analysis_menu.addAction(auto_detect_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        shortcuts_action = QtWidgets.QAction('&Keyboard Shortcuts', self)
        shortcuts_action.triggered.connect(self._show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)
        
        about_action = QtWidgets.QAction('&About', self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)
        
        # Status bar
        self.statusBar().showMessage('Ready - Load a video to begin')

    def _update_recent_files_menu(self):
        """Update the recent files menu."""
        self.recent_files_menu.clear()
        
        if not self.recent_files:
            no_recent_action = QtWidgets.QAction('No recent files', self)
            no_recent_action.setEnabled(False)
            self.recent_files_menu.addAction(no_recent_action)
            return
        
        for file_path in self.recent_files:
            if os.path.exists(file_path):
                action = QtWidgets.QAction(os.path.basename(file_path), self)
                action.setStatusTip(file_path)
                action.triggered.connect(lambda checked, path=file_path: self._open_recent_file(path))
                self.recent_files_menu.addAction(action)

    def _open_recent_file(self, file_path: str):
        """Open a file from the recent files list."""
        if os.path.exists(file_path):
            self.video_path = file_path
            self.load_video()
        else:
            QtWidgets.QMessageBox.warning(self, 'File Not Found', 
                                        f'The file {file_path} no longer exists.')
            self.recent_files.remove(file_path)
            self._update_recent_files_menu()

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Frame navigation shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, 
                          lambda: self.step_frames(1))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self, 
                          lambda: self.step_frames(-1))
        
        # Jump navigation
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_PageDown), self, 
                          lambda: self.step_frames(JUMP_FRAMES))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_PageUp), self, 
                          lambda: self.step_frames(-JUMP_FRAMES))
        
        # Go to start/end
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Home), self, 
                          lambda: self.frame_slider.setValue(0))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_End), self, 
                          lambda: self.frame_slider.setValue(self.total_frames - 1))
        
        # ROI shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self, 
                          self.delete_selected_rectangle)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, 
                          self._cancel_current_action)

    def _cancel_current_action(self):
        """Cancel current drawing/moving/resizing action."""
        if self.drawing:
            self.add_rect_btn.setChecked(False)
            self.toggle_add_rectangle_mode(False)
        elif self.moving or self.resizing:
            self.moving = False
            self.resizing = False
            self.start_point = None
            self.end_point = None
            self.move_offset = None
            self.resize_corner = None
            self.image_label.unsetCursor()
            self.show_frame()

    def _show_shortcuts_dialog(self):
        """Show keyboard shortcuts dialog."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Keyboard Shortcuts')
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        
        shortcuts_text = """
<h3>Navigation Shortcuts:</h3>
<b>Left/Right Arrow:</b> Previous/Next frame<br>
<b>Space:</b> Next frame<br>
<b>Backspace:</b> Previous frame<br>
<b>Page Down/Up:</b> Jump 10 frames<br>
<b>Home/End:</b> Go to first/last frame<br>

<h3>Analysis Shortcuts:</h3>
<b>F5:</b> Run analysis<br>
<b>Ctrl+D:</b> Auto-detect range<br>

<h3>ROI Shortcuts:</h3>
<b>Delete:</b> Delete selected ROI<br>
<b>Escape:</b> Cancel current action<br>

<h3>File Shortcuts:</h3>
<b>Ctrl+O:</b> Open video<br>
<b>Ctrl+Q:</b> Exit application<br>
        """
        
        label = QtWidgets.QLabel(shortcuts_text)
        label.setWordWrap(True)
        layout.addWidget(label)
        
        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()

    def _show_about_dialog(self):
        """Show about dialog."""
        QtWidgets.QMessageBox.about(self, 'About Brightness Sorcerer',
            """<h2>Brightness Sorcerer v2.0</h2>
            <p>Advanced video brightness analysis tool</p>
            <p>Analyze brightness changes in video regions of interest (ROIs) 
            with automatic detection and comprehensive plotting.</p>
            <p><b>Features:</b></p>
            <ul>
            <li>Interactive ROI selection and editing</li>
            <li>Automatic frame range detection</li>
            <li>Statistical analysis with mean and median</li>
            <li>High-quality plot generation</li>
            <li>Frame caching for smooth navigation</li>
            </ul>""")

    def _apply_stylesheet(self):
        """Apply a modern, clean stylesheet to the application."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLOR_BACKGROUND};
                color: {COLOR_FOREGROUND};
                font-family: {DEFAULT_FONT_FAMILY};
                font-size: 14px;
            }}
            QMenuBar {{
                background-color: {COLOR_SECONDARY};
                color: {COLOR_FOREGROUND};
                border-bottom: 1px solid {COLOR_SECONDARY_LIGHT};
            }}
            QMenuBar::item {{
                background: transparent;
                padding: 4px 8px;
            }}
            QMenuBar::item:selected {{
                background-color: {COLOR_ACCENT};
            }}
            QMenu {{
                background-color: {COLOR_SECONDARY};
                color: {COLOR_FOREGROUND};
                border: 1px solid {COLOR_SECONDARY_LIGHT};
            }}
            QMenu::item:selected {{
                background-color: {COLOR_ACCENT};
            }}
            QStatusBar {{
                background-color: {COLOR_SECONDARY};
                color: {COLOR_FOREGROUND};
                border-top: 1px solid {COLOR_SECONDARY_LIGHT};
            }}
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
                background: #1e1e1e;
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
            QLabel#statusLabel {{
                font-size: 12px;
                color: {COLOR_INFO};
                padding: 4px;
            }}
            QGroupBox {{
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 6px;
                margin-top: 10px;
                background: {COLOR_SECONDARY};
                font-weight: bold;
                font-size: 15px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 2px 5px;
                color: {COLOR_ACCENT};
                background-color: {COLOR_BACKGROUND};
                border-radius: 3px;
            }}
            QPushButton {{
                background-color: {COLOR_SECONDARY_LIGHT};
                color: {COLOR_FOREGROUND};
                border: 1px solid {COLOR_SECONDARY};
                border-radius: 4px;
                padding: 8px 15px;
                font-size: 14px;
                min-height: 20px;
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
            QPushButton:checked {{
                background-color: {COLOR_ACCENT};
                color: {COLOR_BACKGROUND};
                border: 1px solid {COLOR_ACCENT_HOVER};
            }}
            QListWidget {{
                background: {COLOR_BACKGROUND};
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
                height: 6px;
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
                background: {COLOR_SUCCESS};
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
            QProgressDialog QLabel {{
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
        self.main_layout.addLayout(self.left_layout, stretch=3)
        self.main_layout.addLayout(self.right_layout, stretch=1)

    def _create_widgets(self):
        """Create all the widgets and add them to layouts."""
        # --- Left Layout Widgets ---
        self.title_label = QtWidgets.QLabel("Brightness Sorcerer", self)
        self.title_label.setObjectName("titleLabel")
        self.left_layout.addWidget(self.title_label)

        # File info label
        self.file_info_label = QtWidgets.QLabel("No video loaded")
        self.file_info_label.setObjectName("statusLabel")
        self.left_layout.addWidget(self.file_info_label)

        # Open-file button
        self.open_btn = QtWidgets.QPushButton("Open Video… (Ctrl+O)")    
        self.open_btn.setToolTip("Choose a video file from disk")
        self.left_layout.addWidget(self.open_btn)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image_label.setText("Drag & Drop Video File Here")
        self.left_layout.addWidget(self.image_label, stretch=1)

        # Slider and Frame Label Layout
        slider_frame_layout = QtWidgets.QHBoxLayout()
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setToolTip("Drag to navigate frames")
        slider_frame_layout.addWidget(self.frame_slider)

        self.frame_label = QtWidgets.QLabel("Frame: 0 / 0")
        self.frame_label.setMinimumWidth(120)
        slider_frame_layout.addWidget(self.frame_label)

        self.frame_spinbox = QtWidgets.QSpinBox()
        self.frame_spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.frame_spinbox.setToolTip("Enter frame number directly")
        slider_frame_layout.addWidget(self.frame_spinbox)
        self.left_layout.addLayout(slider_frame_layout)

        # Frame Control Buttons Layout
        frame_control_layout = QtWidgets.QHBoxLayout()
        self.prev_frame_btn = QtWidgets.QPushButton("◀")
        self.prev_frame_btn.setToolTip("Previous Frame (Left Arrow, Backspace)")
        frame_control_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QtWidgets.QPushButton("▶")
        self.next_frame_btn.setToolTip("Next Frame (Right Arrow, Space)")
        frame_control_layout.addWidget(self.next_frame_btn)

        self.jump_back_btn = QtWidgets.QPushButton(f"◀◀ {JUMP_FRAMES}")
        self.jump_back_btn.setToolTip(f"Jump back {JUMP_FRAMES} frames (Page Up)")
        frame_control_layout.addWidget(self.jump_back_btn)

        self.jump_forward_btn = QtWidgets.QPushButton(f"{JUMP_FRAMES} ▶▶")
        self.jump_forward_btn.setToolTip(f"Jump forward {JUMP_FRAMES} frames (Page Down)")
        frame_control_layout.addWidget(self.jump_forward_btn)

        self.set_start_btn = QtWidgets.QPushButton("Set Start")
        self.set_start_btn.setToolTip("Set current frame as analysis start")
        frame_control_layout.addWidget(self.set_start_btn)

        self.set_end_btn = QtWidgets.QPushButton("Set End")
        self.set_end_btn.setToolTip("Set current frame as analysis end")
        frame_control_layout.addWidget(self.set_end_btn)

        self.auto_detect_btn = QtWidgets.QPushButton("Auto-Detect (Ctrl+D)")
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
        self.analyze_btn = QtWidgets.QPushButton('Analyze Brightness (F5)')
        self.analyze_btn.setToolTip("Run brightness analysis on the selected frame range and ROIs")
        action_layout.addWidget(self.analyze_btn)
        self.left_layout.addLayout(action_layout)

        # --- Right Layout Widgets ---
        # Video info group
        self.video_info_groupbox = QtWidgets.QGroupBox("Video Information")
        video_info_layout = QtWidgets.QVBoxLayout()
        self.video_info_label = QtWidgets.QLabel("No video loaded")
        self.video_info_label.setWordWrap(True)
        video_info_layout.addWidget(self.video_info_label)
        self.video_info_groupbox.setLayout(video_info_layout)
        self.right_layout.addWidget(self.video_info_groupbox)

        # Brightness Display
        self.brightness_groupbox = QtWidgets.QGroupBox("ROI Brightness: Mean±Median (Current Frame)")
        brightness_groupbox_layout = QtWidgets.QVBoxLayout()
        self.brightness_display_label = QtWidgets.QLabel("N/A")
        self.brightness_display_label.setObjectName("brightnessDisplayLabel")
        brightness_groupbox_layout.addWidget(self.brightness_display_label)
        self.brightness_groupbox.setLayout(brightness_groupbox_layout)
        self.right_layout.addWidget(self.brightness_groupbox)

        # -- Threshold groupbox
        self.threshold_groupbox = QtWidgets.QGroupBox("Threshold Settings")
        th_layout = QtWidgets.QVBoxLayout()
        
        # Manual threshold controls
        manual_layout = QtWidgets.QHBoxLayout()
        manual_layout.addWidget(QtWidgets.QLabel("Manual ΔL*:"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.setRange(0.0, 100.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setValue(self.manual_threshold)
        manual_layout.addWidget(self.threshold_spin)
        th_layout.addLayout(manual_layout)
        
        # Background ROI controls
        bg_layout = QtWidgets.QHBoxLayout()
        self.set_bg_btn = QtWidgets.QPushButton("Set Selected ROI as Background")
        bg_layout.addWidget(self.set_bg_btn)
        th_layout.addLayout(bg_layout)
        
        # Current threshold display
        self.threshold_display_label = QtWidgets.QLabel("Active Threshold: Manual (5.0 L*)")
        self.threshold_display_label.setStyleSheet("color: #ffc000; font-weight: bold; padding: 4px;")
        th_layout.addWidget(self.threshold_display_label)
        
        self.threshold_groupbox.setLayout(th_layout)
        self.right_layout.addWidget(self.threshold_groupbox)

        # Visualization Controls
        self.viz_groupbox = QtWidgets.QGroupBox("Visualization")
        viz_layout = QtWidgets.QVBoxLayout()
        
        self.show_mask_checkbox = QtWidgets.QCheckBox("Show Pixel Mask")
        self.show_mask_checkbox.setToolTip("Highlight analyzed pixels in red overlay")
        self.show_mask_checkbox.setChecked(self.show_pixel_mask)
        viz_layout.addWidget(self.show_mask_checkbox)
        
        self.viz_groupbox.setLayout(viz_layout)
        self.right_layout.addWidget(self.viz_groupbox)

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
        self.del_rect_btn.setToolTip("Delete the selected ROI from the list (Delete key)")
        rect_btn_layout.addWidget(self.del_rect_btn)

        self.clear_rect_btn = QtWidgets.QPushButton("Clear All")
        self.clear_rect_btn.setToolTip("Remove all ROIs")
        rect_btn_layout.addWidget(self.clear_rect_btn)
        rect_groupbox_layout.addLayout(rect_btn_layout)
        self.rect_groupbox.setLayout(rect_groupbox_layout)
        self.right_layout.addWidget(self.rect_groupbox)

        # Cache status
        self.cache_status_label = QtWidgets.QLabel("Cache: 0 frames")
        self.cache_status_label.setObjectName("statusLabel")
        self.right_layout.addWidget(self.cache_status_label)

        # Results/Status Label
        self.results_label = QtWidgets.QLabel("Load a video to begin analysis.")
        self.results_label.setObjectName("resultsLabel")
        self.results_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.results_label.setWordWrap(True)
        self.results_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.right_layout.addWidget(self.results_label, stretch=1)

    def _connect_signals(self):
        """Connect widget signals to their corresponding slots."""
        self.open_btn.clicked.connect(self.open_video_dialog)

        self.frame_slider.valueChanged.connect(self.slider_frame_changed)
        self.frame_spinbox.valueChanged.connect(self.spinbox_frame_changed)

        self.prev_frame_btn.clicked.connect(lambda: self.step_frames(-1))
        self.next_frame_btn.clicked.connect(lambda: self.step_frames(1))
        self.jump_back_btn.clicked.connect(lambda: self.step_frames(-JUMP_FRAMES))
        self.jump_forward_btn.clicked.connect(lambda: self.step_frames(JUMP_FRAMES))
        self.set_start_btn.clicked.connect(self.set_start_frame)
        self.set_end_btn.clicked.connect(self.set_end_frame)
        self.auto_detect_btn.clicked.connect(self.auto_detect_range)

        self.analyze_btn.clicked.connect(self.analyze_video)

        self.rect_list.currentRowChanged.connect(self.select_rectangle_from_list)
        self.add_rect_btn.clicked.connect(self.toggle_add_rectangle_mode)
        self.del_rect_btn.clicked.connect(self.delete_selected_rectangle)
        self.clear_rect_btn.clicked.connect(self.clear_all_rectangles)

        # Connect mouse events directly
        self.image_label.mousePressEvent = self.image_mouse_press
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release

        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self.set_bg_btn.clicked.connect(self._set_background_roi)
        self.show_mask_checkbox.toggled.connect(self._on_mask_checkbox_toggled)

    def _update_widget_states(self, video_loaded=False, rois_exist=False):
        """Enable/disable widgets based on application state."""
        self.frame_slider.setEnabled(video_loaded and not self._analysis_in_progress)
        self.frame_spinbox.setEnabled(video_loaded and not self._analysis_in_progress)
        self.prev_frame_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.next_frame_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.jump_back_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.jump_forward_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.set_start_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.set_end_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.auto_detect_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        self.analyze_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        self.add_rect_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.del_rect_btn.setEnabled(video_loaded and self.selected_rect_idx is not None and not self._analysis_in_progress)
        self.clear_rect_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        self.set_bg_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        self.threshold_spin.setEnabled(not self._analysis_in_progress)

    def _update_cache_status(self):
        """Update cache status display."""
        cache_size = self.frame_cache.get_size()
        self.cache_status_label.setText(f"Cache: {cache_size}/{FRAME_CACHE_SIZE} frames")

    def _update_video_info(self):
        """Update video information display."""
        if not self.video_path or not self.cap:
            self.video_info_label.setText("No video loaded")
            self.file_info_label.setText("No video loaded")
            return
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = self.total_frames / fps if fps > 0 else 0
        
        file_name = os.path.basename(self.video_path)
        file_size = os.path.getsize(self.video_path) / (1024 * 1024)  # MB
        
        info_text = f"""
<b>File:</b> {file_name}<br>
<b>Size:</b> {file_size:.1f} MB<br>
<b>Resolution:</b> {width} × {height}<br>
<b>Frames:</b> {self.total_frames}<br>
<b>FPS:</b> {fps:.2f}<br>
<b>Duration:</b> {duration_sec:.1f}s<br>
<b>Analysis Range:</b> {self.start_frame + 1}-{(self.end_frame or 0) + 1}
        """.strip()
        
        self.video_info_label.setText(info_text)
        self.file_info_label.setText(f"Loaded: {file_name}")

    # --- Event Handling ---

    # File-picker slot
    def open_video_dialog(self):
        """Present a file dialog to select a video file."""
        initial_dir = os.path.dirname(self.video_path) if self.video_path else ""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            initial_dir,
            "Video Files (*.mp4 *.mov *.avi *.mkv *.wmv *.m4v *.flv);;All Files (*)"
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
        """Release resources and save settings when the window closes."""
        if self.cap:
            self.cap.release()
        self._save_settings()
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

        # Clear cache when loading new video
        self.frame_cache.clear()
        
        self.current_frame_index = 0
        self.start_frame = 0
        self.end_frame = self.total_frames - 1

        # Load the first frame
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self.frame_cache.put(0, frame)  # Cache first frame

            # Ensure the pixmap scales to the label’s current size the first time we draw it
            self._current_image_size = self.image_label.size()
            
            # Update UI elements for the loaded video
            self.frame_slider.setRange(0, self.total_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_spinbox.setRange(0, self.total_frames - 1)
            self.frame_spinbox.setValue(0)
            self.update_frame_label()
            self.show_frame()
            self._update_video_info()
            self._update_cache_status()
            self._update_threshold_display()
            
            # Add to recent files
            self._add_recent_file(self.video_path)
            
            self.results_label.setText(f"Loaded: {os.path.basename(self.video_path)}\n"
                                       f"Frames: {self.total_frames}\n"
                                       "Draw ROIs or use Auto-Detect.")
            self._update_widget_states(video_loaded=True, rois_exist=bool(self.rects))
            self.statusBar().showMessage(f"Loaded: {os.path.basename(self.video_path)}")

            # Attempt auto-detection if ROIs already exist
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
        self.frame_cache.clear()
        
        self.image_label.setText("Drag & Drop Video File Here")
        self.image_label.setPixmap(QtGui.QPixmap())
        self.update_frame_label(reset=True)
        self.update_rect_list()
        self.brightness_display_label.setText("N/A")
        self.results_label.setText("Load a video to begin analysis.")
        self.file_info_label.setText("No video loaded")
        self.video_info_label.setText("No video loaded")
        self.frame_slider.setRange(0, 0)
        self.frame_spinbox.setRange(0, 0)
        self._update_cache_status()
        self._update_widget_states(video_loaded=False, rois_exist=False)
        self.statusBar().showMessage("Ready - Load a video to begin")

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
        """Reads and displays the specified frame index with caching."""
        if not self.cap or not self.cap.isOpened():
            return
        if frame_index < 0 or frame_index >= self.total_frames:
            logging.warning(f"Attempted to seek to invalid frame index {frame_index}")
            return

        # Check cache first
        cached_frame = self.frame_cache.get(frame_index)
        if cached_frame is not None:
            self.frame = cached_frame
            self.current_frame_index = frame_index
            self.show_frame()
            self.update_frame_label()
            self._update_current_brightness_display()
            self._update_threshold_display()
            return

        # Not in cache, read from video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self.current_frame_index = frame_index
            
            # Cache the frame
            self.frame_cache.put(frame_index, frame)
            self._update_cache_status()
            
            self.show_frame()
            self.update_frame_label()
            self._update_current_brightness_display()
            self._update_threshold_display()
        else:
            logging.warning(f"Failed to read frame at index {frame_index}")

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
        
        # Apply pixel mask visualization if enabled
        if self.show_pixel_mask and len(self.rects) > 0:
            frame_copy = self._apply_pixel_mask_overlay(frame_copy)
            
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
        """Calculates and displays comprehensive brightness information for the current frame's ROIs."""
        if self.frame is None or not self.rects:
            self.brightness_display_label.setText("N/A")
            return

        roi_data = []
        background_brightness = None
        fh, fw = self.frame.shape[:2]
        
        # Calculate background brightness if background ROI is defined
        if self.background_roi_idx is not None:
            background_brightness = self._compute_background_brightness(self.frame)
        
        for idx, (pt1, pt2) in enumerate(self.rects):
            # Handle background ROI separately
            if idx == self.background_roi_idx:
                continue
                
            # Ensure ROI coordinates are valid within the frame
            x1 = max(0, min(pt1[0], fw - 1))
            y1 = max(0, min(pt1[1], fh - 1))
            x2 = max(0, min(pt2[0], fw - 1))
            y2 = max(0, min(pt2[1], fh - 1))

            if x2 > x1 and y2 > y1: # Check for valid ROI area
                roi = self.frame[y1:y2, x1:x2]
                l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median = self._compute_brightness_stats(roi, background_brightness)
                roi_data.append((idx, l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median))
            else:
                roi_data.append((idx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) # Append zeros if ROI is invalid/empty

        if roi_data:
            # Build comprehensive display
            display_lines = ["Current Brightness:"]
            
            for idx, l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median in roi_data:
                if self.background_roi_idx is not None:
                    # Show L* with background subtraction and blue channel
                    display_lines.append(f"ROI {idx+1}: L* {l_raw_mean:.1f} (BG-Sub: {l_bg_sub_mean:.1f}) | Blue: {b_raw_mean:.0f}")
                else:
                    # Show raw L* and blue channel when no background ROI
                    display_lines.append(f"ROI {idx+1}: L* {l_raw_mean:.1f} | Blue: {b_raw_mean:.0f}")
            
            # Add background ROI info if defined
            if self.background_roi_idx is not None and background_brightness is not None:
                # Calculate blue channel for background ROI
                bg_pt1, bg_pt2 = self.rects[self.background_roi_idx]
                bg_x1 = max(0, min(bg_pt1[0], fw - 1))
                bg_y1 = max(0, min(bg_pt1[1], fh - 1))
                bg_x2 = max(0, min(bg_pt2[0], fw - 1))
                bg_y2 = max(0, min(bg_pt2[1], fh - 1))
                
                if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                    bg_roi = self.frame[bg_y1:bg_y2, bg_x1:bg_x2]
                    _, _, _, _, bg_b_mean, _, _, _ = self._compute_brightness_stats(bg_roi)
                    display_lines.append(f"Background: L* {background_brightness:.1f} | Blue: {bg_b_mean:.0f}")
            
            self.brightness_display_label.setText("\n".join(display_lines))
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
            prefix = "* " if idx == self.background_roi_idx else ""
            self.rect_list.addItem(f"{prefix}ROI {idx+1}: ({disp_x1},{disp_y1})-({disp_x2},{disp_y2})")

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
        self._update_threshold_display()


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
            
            # Handle background ROI index adjustment
            if self.background_roi_idx == self.selected_rect_idx:
                self.background_roi_idx = None
            elif self.background_roi_idx is not None and self.selected_rect_idx < self.background_roi_idx:
                self.background_roi_idx -= 1
            
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
            self.background_roi_idx = None
            self.update_rect_list()
            self.show_frame()

    def _set_background_roi(self):
        """Mark current ROI as background reference for auto-detect."""
        if self.selected_rect_idx is None:
            QtWidgets.QMessageBox.information(self, "Background ROI",
                                              "Select an ROI first.")
            return
        
        self.background_roi_idx = self.selected_rect_idx
        
        # Calculate background threshold for display
        if self.frame is not None:
            threshold_value = self._calculate_background_threshold()
            if threshold_value is not None:
                self.results_label.setText(f"Background ROI set to ROI {self.selected_rect_idx + 1}\n"
                                         f"Current background threshold: {threshold_value:.2f} L*")
            else:
                self.results_label.setText(f"Background ROI set to ROI {self.selected_rect_idx + 1}\n"
                                         f"(Threshold will be calculated during full video scan)")
        else:
            self.results_label.setText(f"Background ROI set to ROI {self.selected_rect_idx + 1}")
        
        self.update_rect_list()

    def _calculate_background_threshold(self) -> Optional[float]:
        """Calculate the current background threshold based on background ROI or manual setting."""
        if self.background_roi_idx is not None and self.frame is not None:
            # Calculate threshold from current frame's background ROI
            if 0 <= self.background_roi_idx < len(self.rects):
                pt1, pt2 = self.rects[self.background_roi_idx]
                fh, fw = self.frame.shape[:2]
                
                # Ensure ROI coordinates are valid within the frame
                x1 = max(0, min(pt1[0], fw - 1))
                y1 = max(0, min(pt1[1], fh - 1))
                x2 = max(0, min(pt2[0], fw - 1))
                y2 = max(0, min(pt2[1], fh - 1))
                
                if x2 > x1 and y2 > y1:
                    roi = self.frame[y1:y2, x1:x2]
                    l_raw_mean, _, _, _, _, _, _, _ = self._compute_brightness_stats(roi)
                    return l_raw_mean
        
        # If no background ROI or calculation failed, return manual threshold
        return None

    def _update_threshold_display(self):
        """Update the threshold display label with current active threshold."""
        if self.background_roi_idx is not None:
            # Background ROI mode
            if self.frame is not None:
                threshold_value = self._calculate_background_threshold()
                if threshold_value is not None:
                    self.threshold_display_label.setText(f"Active Threshold: Background ROI {self.background_roi_idx + 1} ({threshold_value:.2f} L*)")
                else:
                    self.threshold_display_label.setText(f"Active Threshold: Background ROI {self.background_roi_idx + 1} (calculating...)")
            else:
                self.threshold_display_label.setText(f"Active Threshold: Background ROI {self.background_roi_idx + 1} (no frame)")
        else:
            # Manual threshold mode
            self.threshold_display_label.setText(f"Active Threshold: Manual ({self.manual_threshold:.2f} L*)")

    def _on_mask_checkbox_toggled(self, checked: bool):
        """Handle pixel mask visualization checkbox toggle."""
        self.show_pixel_mask = checked
        if self.frame is not None:
            self.show_frame()

    def _apply_pixel_mask_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply red overlay to show which pixels are being analyzed in each ROI.
        
        Args:
            frame: BGR frame to apply overlay to
            
        Returns:
            Frame with red mask overlay applied
        """
        overlay = frame.copy()
        
        # Get background brightness for current frame
        background_brightness = self._compute_background_brightness(frame)
        
        for roi_idx, (pt1, pt2) in enumerate(self.rects):
            # Skip background ROI
            if roi_idx == self.background_roi_idx:
                continue
                
            # Extract ROI bounds
            fh, fw = frame.shape[:2]
            x1 = max(0, min(pt1[0], fw - 1))
            y1 = max(0, min(pt1[1], fh - 1))
            x2 = max(0, min(pt2[0], fw - 1))
            y2 = max(0, min(pt2[1], fh - 1))
            
            if x2 > x1 and y2 > y1:
                # Extract ROI and convert to LAB
                roi = frame[y1:y2, x1:x2]
                try:
                    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                    l_chan = lab[:, :, 0].astype(np.float32)
                    l_star = l_chan * 100.0 / 255.0
                    
                    # Create mask for analyzed pixels
                    if background_brightness is not None:
                        # Show only pixels above background threshold
                        mask = l_star > background_brightness
                    else:
                        # Show all pixels (unthresholded analysis)
                        mask = np.ones_like(l_star, dtype=bool)
                    
                    # Apply red overlay to analyzed pixels
                    roi_overlay = roi.copy()
                    roi_overlay[mask] = roi_overlay[mask] * 0.7 + np.array([0, 0, 255]) * 0.3  # Red tint
                    
                    # Apply overlay back to main frame
                    overlay[y1:y2, x1:x2] = roi_overlay
                    
                except Exception as e:
                    print(f"Error creating pixel mask for ROI {roi_idx+1}: {e}")
                    continue
        
        return overlay

    def _on_threshold_changed(self, value: float):
        """Handle changes to the manual threshold spinbox."""
        self.manual_threshold = value
        self._update_threshold_display()

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
            if frame_x is None or frame_y is None: 
                return # Mapping failed

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
            if frame_x is None or frame_y is None: 
                return # Mapping failed

            # Clamp mouse position to frame boundaries before calculating new rect
            frame_x = max(0, min(frame_x, frame_w - 1))
            frame_y = max(0, min(frame_y, frame_h - 1))

            fixed_corner_frame = self._map_label_to_frame_point(self.start_point)
            if fixed_corner_frame[0] is None or fixed_corner_frame[1] is None: 
                return # Mapping failed

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

    def _get_pixmap_rect_in_label(self) -> Optional[QtCore.QRect]:
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

    def _map_label_to_frame_point(self, label_pos: QtCore.QPoint) -> Tuple[Optional[int], Optional[int]]:
        """Maps a point from image label coordinates to original frame coordinates."""
        if self.frame is None: 
            return None, None

        pixmap_rect = self._get_pixmap_rect_in_label()
        if not pixmap_rect: 
            return None, None

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

        if pixmap_w == 0 or pixmap_h == 0: 
            return None, None  # Avoid division by zero

        scale_w = frame_w / pixmap_w
        scale_h = frame_h / pixmap_h

        # Calculate corresponding point in the original frame
        frame_x = int(relative_x * scale_w)
        frame_y = int(relative_y * scale_h)

        # Clamp to frame boundaries (shouldn't be necessary if logic is correct, but safe)
        frame_x = max(0, min(frame_x, frame_w - 1))
        frame_y = max(0, min(frame_y, frame_h - 1))

        return frame_x, frame_y

    def _map_label_to_frame_rect(self, label_pt1: QtCore.QPoint, label_pt2: QtCore.QPoint) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
         """Maps a rectangle defined by two points in label coordinates to frame coordinates."""
         frame_pt1 = self._map_label_to_frame_point(label_pt1)
         frame_pt2 = self._map_label_to_frame_point(label_pt2)

         if frame_pt1[0] is None or frame_pt2[0] is None:
              return None, None
         
         # Convert the tuples to the expected format
         pt1 = (frame_pt1[0], frame_pt1[1])
         pt2 = (frame_pt2[0], frame_pt2[1]) 
         return pt1, pt2

    def _map_frame_to_label_point(self, frame_pos: Tuple[int, int]) -> Optional[QtCore.QPoint]:
        """Maps a point from original frame coordinates back to image label coordinates."""
        if self.frame is None: 
            return None

        pixmap_rect = self._get_pixmap_rect_in_label()
        if not pixmap_rect: 
            return None

        frame_h, frame_w = self.frame.shape[:2]
        pixmap_w = pixmap_rect.width()
        pixmap_h = pixmap_rect.height()

        if frame_w == 0 or frame_h == 0: 
            return None  # Avoid division by zero

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
        self._update_threshold_display() # Update threshold display after setting frames

    def set_end_frame(self):
        """Sets the current frame as the end frame for analysis."""
        if self.cap and self.cap.isOpened():
            self.end_frame = self.current_frame_index
            self.results_label.setText(f"End frame set to {self.end_frame + 1}") # Display 1-based index
            # Ensure start frame is not after end frame
            if self.start_frame > self.end_frame:
                self.start_frame = self.end_frame
                self.results_label.setText(f"End frame set to {self.end_frame + 1}. Start frame adjusted.")
        self._update_threshold_display() # Update threshold display after setting frames

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
        
        brightness_per_frame = np.zeros(total)
        background_brightness_per_frame = np.zeros(total) if self.background_roi_idx is not None else None
        
        # --- Progress Dialog ---
        progress = QtWidgets.QProgressDialog("Scanning video for auto-detection...", "Cancel", 0, total, self)
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
                if background_brightness_per_frame is not None:
                    background_brightness_per_frame = background_brightness_per_frame[:total]
                logging.warning(f"Auto-detect scan stopped early at frame {idx} due to read error.")
                break

            # Calculate average brightness across ROIs for this frame
            current_frame_measurement_roi_brightness = []
            fh, fw = frame.shape[:2]
            for roi_idx, (pt1, pt2) in enumerate(self.rects):
                x1, y1 = max(0, pt1[0]), max(0, pt1[1])
                x2, y2 = min(fw - 1, pt2[0]), min(fh - 1, pt2[1])
                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    brightness = self._compute_brightness(roi)
                    
                    if roi_idx == self.background_roi_idx:
                        if background_brightness_per_frame is not None:
                            background_brightness_per_frame[idx] = brightness
                    else:
                        current_frame_measurement_roi_brightness.append(brightness)

            # Store the mean brightness of all measurement ROIs for the current frame
            brightness_per_frame[idx] = np.mean(current_frame_measurement_roi_brightness) if current_frame_measurement_roi_brightness else 0.0

            progress.setValue(idx + 1)
            if idx % 10 == 0: # Update UI periodically to keep it responsive
                 QtWidgets.QApplication.processEvents()

        scan_cap.release()
        progress.close()

        if scan_cancelled:
            self.results_label.setText("Auto-detect scan cancelled.")
            return

        if brightness_per_frame.size == 0 and (background_brightness_per_frame is None or background_brightness_per_frame.size == 0):
            QtWidgets.QMessageBox.warning(self, "Auto-Detect", "Scan completed, but no brightness data was gathered.")
            return

        # --- Analyze Brightness Data ---
        try:
            if self.background_roi_idx is not None and background_brightness_per_frame is not None:
                # Threshold = mean brightness of background ROI over scan + manual delta
                valid_bg_frames = background_brightness_per_frame[background_brightness_per_frame > 0]
                background_baseline = np.mean(valid_bg_frames) if valid_bg_frames.size > 0 else 0
                threshold = background_baseline + self.manual_threshold
            else:
                baseline = np.percentile(brightness_per_frame, AUTO_DETECT_BASELINE_PERCENTILE)
                threshold = baseline + self.manual_threshold
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
        """Performs brightness analysis with enhanced progress tracking and error handling."""
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

        # Set analysis state
        self._analysis_in_progress = True
        self._update_widget_states(video_loaded=True, rois_exist=True)
        
        # --- Get Save Directory ---
        initial_dir = os.path.dirname(self.video_path) if self.video_path else ""
        save_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose Directory to Save Analysis Results and Plots", initial_dir)
        if not save_dir:
            self.results_label.setText("Analysis cancelled (no save directory chosen).")
            self._analysis_in_progress = False
            self._update_widget_states(video_loaded=True, rois_exist=True)
            return

        # --- Setup ---
        self.results_label.setText("Starting analysis...")
        self.brightness_display_label.setText("Analyzing...")
        self.statusBar().showMessage("Running brightness analysis...")
        QtWidgets.QApplication.processEvents()

        try:
            analysis_cap = cv2.VideoCapture(self.video_path)
            if not analysis_cap.isOpened():
                raise Exception(f"Could not open video file for analysis: {os.path.basename(self.video_path)}")

            # Frame range for analysis (inclusive)
            start = self.start_frame
            end = self.end_frame
            num_frames_to_analyze = end - start + 1

            # Data structure: list of lists for mean and median, one inner list per non-background ROI
            non_background_rois = [i for i in range(len(self.rects)) if i != self.background_roi_idx]
            brightness_mean_data = [[] for _ in non_background_rois]
            brightness_median_data = [[] for _ in non_background_rois]
            blue_mean_data = [[] for _ in non_background_rois]
            blue_median_data = [[] for _ in non_background_rois]
            background_values_per_frame = []  # Track background brightness for each frame

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
            start_time = time.time()

            for f_idx in range(start, end + 1):
                if progress.wasCanceled():
                    analysis_cancelled = True
                    break

                ret, frame = analysis_cap.read()
                if not ret:
                    logging.warning(f"Could not read frame {f_idx} during analysis. Stopping analysis.")
                    num_frames_to_analyze = frames_processed
                    brightness_mean_data = [lst[:frames_processed] for lst in brightness_mean_data]
                    brightness_median_data = [lst[:frames_processed] for lst in brightness_median_data]
                    blue_mean_data = [lst[:frames_processed] for lst in blue_mean_data]
                    blue_median_data = [lst[:frames_processed] for lst in blue_median_data]
                    break

                # Calculate background brightness for this frame
                background_brightness = self._compute_background_brightness(frame)
                background_values_per_frame.append(background_brightness if background_brightness is not None else 0.0)
                
                fh, fw = frame.shape[:2]
                for data_idx, roi_idx in enumerate(non_background_rois):
                    pt1, pt2 = self.rects[roi_idx]
                    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
                    x2, y2 = min(fw - 1, pt2[0]), min(fh - 1, pt2[1])

                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2]
                        l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median = self._compute_brightness_stats(roi, background_brightness)
                        # Store background-subtracted values if background ROI defined, otherwise raw values
                        if background_brightness is not None:
                            brightness_mean_data[data_idx].append(l_bg_sub_mean)
                            brightness_median_data[data_idx].append(l_bg_sub_median)
                            blue_mean_data[data_idx].append(b_bg_sub_mean)
                            blue_median_data[data_idx].append(b_bg_sub_median)
                        else:
                            brightness_mean_data[data_idx].append(l_raw_mean)
                            brightness_median_data[data_idx].append(l_raw_median)
                            blue_mean_data[data_idx].append(b_raw_mean)
                            blue_median_data[data_idx].append(b_raw_median)
                    else:
                        brightness_mean_data[data_idx].append(0.0)
                        brightness_median_data[data_idx].append(0.0)
                        blue_mean_data[data_idx].append(0.0)
                        blue_median_data[data_idx].append(0.0)

                frames_processed += 1
                
                # Update progress with time estimate
                if frames_processed % 10 == 0:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        fps = frames_processed / elapsed_time
                        remaining_frames = num_frames_to_analyze - frames_processed
                        eta_seconds = remaining_frames / fps if fps > 0 else 0
                        progress.setLabelText(f"Analyzing frame {frames_processed}/{num_frames_to_analyze}\n"
                                            f"Speed: {fps:.1f} fps, ETA: {eta_seconds:.0f}s")
                    
                progress.setValue(frames_processed)
                QtWidgets.QApplication.processEvents()

            analysis_cap.release()
            progress.close()

            if analysis_cancelled:
                self.results_label.setText("Analysis cancelled by user.")
                self._update_current_brightness_display()
                return

            if frames_processed == 0:
                 QtWidgets.QMessageBox.warning(self, "Analysis", "No frames were processed during analysis.")
                 self.results_label.setText("Analysis completed, but no frames processed.")
                 self._update_current_brightness_display()
                 return

            # --- Save Results and Generate Plots ---
            self._save_analysis_results(brightness_mean_data, brightness_median_data, blue_mean_data, blue_median_data, save_dir, frames_processed, non_background_rois, background_values_per_frame)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis:\n{str(e)}")
            self.results_label.setText(f"Analysis failed: {str(e)}")
            logging.error(f"Analysis error: {e}")
        finally:
            self._analysis_in_progress = False
            self._update_widget_states(video_loaded=True, rois_exist=True)
            self.statusBar().showMessage("Analysis complete")

    def _save_analysis_results(self, brightness_mean_data, brightness_median_data, blue_mean_data, blue_median_data, save_dir, frames_processed, non_background_rois, background_values_per_frame):
        """Save analysis results and generate plots."""
        self.out_paths = []
        plot_paths = []
        base_video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        analysis_name = self.analysis_name_input.text().strip() or "DefaultAnalysis"
        analysis_name = "".join(c for c in analysis_name if c.isalnum() or c in ('_', '-')).rstrip()

        summary_lines = [f"Analysis Complete ({frames_processed} frames analyzed):"]
        avg_brightness_summary = []

        # Update progress dialog for plot generation
        progress = QtWidgets.QProgressDialog("Saving results and generating plots...", "Cancel", 0, len(brightness_mean_data), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Saving Results")
        progress.setValue(0)
        progress.show()
        QtWidgets.QApplication.processEvents()

        plot_failed = False
        for data_idx in range(len(brightness_mean_data)):
            if progress.wasCanceled():
                break
                
            actual_roi_idx = non_background_rois[data_idx]  # Get the actual ROI index
            mean_data = brightness_mean_data[data_idx]
            median_data = brightness_median_data[data_idx]
            blue_mean = blue_mean_data[data_idx]
            blue_median = blue_median_data[data_idx]
            
            if not mean_data: 
                progress.setValue(data_idx + 1)
                continue

            # Create DataFrame with L* and blue channel data
            frame_numbers = range(self.start_frame, self.start_frame + len(mean_data))
            df = pd.DataFrame({
                "frame": frame_numbers, 
                "brightness_mean": mean_data,
                "brightness_median": median_data,
                "blue_mean": blue_mean,
                "blue_median": blue_median
            })

            # Calculate averages for summary
            avg_mean = np.mean(mean_data)
            avg_median = np.mean(median_data)
            avg_blue_mean = np.mean(blue_mean)
            avg_blue_median = np.mean(blue_median)
            avg_brightness_summary.append(f"ROI {actual_roi_idx+1} L*: {avg_mean:.2f}±{avg_median:.2f}, Blue: {avg_blue_mean:.1f}±{avg_blue_median:.1f}")

            # Construct filename and save CSV using actual ROI number
            base_filename = f"{analysis_name}_{base_video_name}_ROI{actual_roi_idx+1}_frames{self.start_frame+1}-{self.start_frame+len(mean_data)}"
            csv_file = f"{base_filename}_brightness.csv"
            csv_path = os.path.join(save_dir, csv_file)
            
            try:
                df.to_csv(csv_path, index=False)
                self.out_paths.append(csv_path)
                summary_lines.append(f" - Saved CSV: {csv_file}")
                
                # Generate enhanced plot for this ROI
                self._generate_enhanced_plot(df, base_filename, save_dir, actual_roi_idx, analysis_name, base_video_name, background_values_per_frame)
                summary_lines.append(f" - Saved Plot: {base_filename}_plot.png")

            except Exception as e:
                plot_failed = True
                error_msg = f"Failed to save/plot ROI {actual_roi_idx+1}: {e}"
                logging.error(error_msg)
                summary_lines.append(f" - FAILED: ROI {actual_roi_idx+1}")

            progress.setValue(data_idx + 1)
            QtWidgets.QApplication.processEvents()

        progress.close()

        # --- Update UI ---
        if plot_failed:
            summary_lines.append("Note: Some plots failed to generate - check console for details")
            
        self.results_label.setText("\n".join(summary_lines))
        self.brightness_display_label.setText(", ".join(avg_brightness_summary) if avg_brightness_summary else "N/A")

    def _generate_enhanced_plot(self, df, base_filename, save_dir, r_idx, analysis_name, base_video_name, background_values_per_frame):
        """Generate enhanced plots with better styling and information."""
        try:
            frames = df['frame']
            brightness_mean = df['brightness_mean']
            brightness_median = df['brightness_median']
            blue_mean = df['blue_mean']
            blue_median = df['blue_median']

            if brightness_mean.empty:
                return

            # Statistics
            idx_peak_mean = brightness_mean.idxmax()
            frame_peak_mean, val_peak_mean = frames.iloc[idx_peak_mean], brightness_mean.iloc[idx_peak_mean]
            mean_of_means = brightness_mean.mean()
            std_of_means = brightness_mean.std()
            
            idx_peak_median = brightness_median.idxmax()
            frame_peak_median, val_peak_median = frames.iloc[idx_peak_median], brightness_median.iloc[idx_peak_median]
            mean_of_medians = brightness_median.mean()
            std_of_medians = brightness_median.std()
            
            # Blue channel statistics
            idx_peak_blue_mean = blue_mean.idxmax()
            frame_peak_blue_mean, val_peak_blue_mean = frames.iloc[idx_peak_blue_mean], blue_mean.iloc[idx_peak_blue_mean]
            mean_of_blue_means = blue_mean.mean()
            std_of_blue_means = blue_mean.std()
            
            idx_peak_blue_median = blue_median.idxmax()
            frame_peak_blue_median, val_peak_blue_median = frames.iloc[idx_peak_blue_median], blue_median.iloc[idx_peak_blue_median]
            mean_of_blue_medians = blue_median.mean()
            std_of_blue_medians = blue_median.std()

            # Create enhanced plot with dual subplots
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Main brightness plot
            ax1.plot(frames, brightness_mean, label='Mean Brightness', color='#5a9bd5', linewidth=2, alpha=0.8)
            ax1.plot(frames, brightness_median, label='Median Brightness', color='#70ad47', linewidth=2, alpha=0.8)
            
            # Add background line if background values are available
            if background_values_per_frame and len(background_values_per_frame) == len(frames):
                # Filter out zero values (frames where background ROI wasn't available)
                background_array = np.array(background_values_per_frame)
                valid_background_mask = background_array > 0
                if np.any(valid_background_mask):
                    ax1.plot(frames, background_array, label='Background Level', color='#808080', 
                           linewidth=1.5, linestyle=':', alpha=0.9)
            
            # Add confidence bands (mean ± std)
            ax1.fill_between(frames, brightness_mean - std_of_means, brightness_mean + std_of_means, 
                           alpha=0.2, color='#5a9bd5', label=f'Mean ±1σ ({std_of_means:.1f})')
            ax1.fill_between(frames, brightness_median - std_of_medians, brightness_median + std_of_medians, 
                           alpha=0.2, color='#70ad47', label=f'Median ±1σ ({std_of_medians:.1f})')
            
            # Add horizontal lines for averages
            ax1.axhline(mean_of_means, color='#5a9bd5', linestyle='--', alpha=0.7, 
                       label=f'Avg Mean ({mean_of_means:.1f})')
            ax1.axhline(mean_of_medians, color='#70ad47', linestyle='--', alpha=0.7, 
                       label=f'Avg Median ({mean_of_medians:.1f})')
            
            # Mark peak points
            ax1.scatter([frame_peak_mean], [val_peak_mean], color='#ff0000', zorder=5, s=100, 
                       marker='^', label=f'Peak Mean ({val_peak_mean:.1f})')
            ax1.scatter([frame_peak_median], [val_peak_median], color='#ed7d31', zorder=5, s=100, 
                       marker='v', label=f'Peak Median ({val_peak_median:.1f})')

            ax1.set_title(f"{analysis_name} - {base_video_name} - ROI {r_idx+1}", fontsize=16, fontweight='bold')
            ax1.set_ylabel('L* Brightness', fontsize=12)
            ax1.legend(fontsize=10, loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Adjust y-axis limits to provide more space at the top for statistics panel
            y_min, y_max = ax1.get_ylim()
            y_range = y_max - y_min
            # Add 15% padding at the top to accommodate the statistics panel
            ax1.set_ylim(y_min, y_max + 0.15 * y_range)

            # Add statistics text box - positioned in top-right to avoid interference
            stats_text = f"""Statistics:
Mean: {mean_of_means:.2f} ± {std_of_means:.2f}
Median: {mean_of_medians:.2f} ± {std_of_medians:.2f}
Peak Mean: {val_peak_mean:.2f} @ Frame {frame_peak_mean}
Peak Median: {val_peak_median:.2f} @ Frame {frame_peak_median}
Frames Analyzed: {len(frames)}"""
            
            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Blue channel plot
            ax2.plot(frames, blue_mean, label='Blue Mean', color='#0066cc', linewidth=2, alpha=0.8)
            ax2.plot(frames, blue_median, label='Blue Median', color='#3399ff', linewidth=2, alpha=0.8)
            
            # Add confidence bands for blue channel
            ax2.fill_between(frames, blue_mean - std_of_blue_means, blue_mean + std_of_blue_means, 
                           alpha=0.2, color='#0066cc', label=f'Blue Mean ±1σ ({std_of_blue_means:.1f})')
            ax2.fill_between(frames, blue_median - std_of_blue_medians, blue_median + std_of_blue_medians, 
                           alpha=0.2, color='#3399ff', label=f'Blue Median ±1σ ({std_of_blue_medians:.1f})')
            
            # Add horizontal lines for blue averages
            ax2.axhline(mean_of_blue_means, color='#0066cc', linestyle='--', alpha=0.7, 
                       label=f'Avg Blue Mean ({mean_of_blue_means:.1f})')
            ax2.axhline(mean_of_blue_medians, color='#3399ff', linestyle='--', alpha=0.7, 
                       label=f'Avg Blue Median ({mean_of_blue_medians:.1f})')
            
            # Mark blue peak points
            ax2.scatter([frame_peak_blue_mean], [val_peak_blue_mean], color='#ff0000', zorder=5, s=100, 
                       marker='^', label=f'Peak Blue Mean ({val_peak_blue_mean:.1f})')
            ax2.scatter([frame_peak_blue_median], [val_peak_blue_median], color='#ed7d31', zorder=5, s=100, 
                       marker='v', label=f'Peak Blue Median ({val_peak_blue_median:.1f})')
            
            ax2.set_xlabel('Frame Number', fontsize=12)
            ax2.set_ylabel('Blue Channel Value', fontsize=12)
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, alpha=0.3)
            
            # Add blue channel statistics text box
            blue_stats_text = f"""Blue Channel Statistics:
Mean: {mean_of_blue_means:.1f} ± {std_of_blue_means:.1f}
Median: {mean_of_blue_medians:.1f} ± {std_of_blue_medians:.1f}
Peak Mean: {val_peak_blue_mean:.1f} @ Frame {frame_peak_blue_mean}
Peak Median: {val_peak_blue_median:.1f} @ Frame {frame_peak_blue_median}"""
            
            ax2.text(0.98, 0.98, blue_stats_text, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            plt.tight_layout()

            # Save plot
            plot_filename = f"{base_filename}_plot.png"
            plot_save_path = os.path.join(save_dir, plot_filename)
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            
            # Automatically open the generated PNG file
            try:
                import subprocess
                subprocess.run(['open', plot_save_path], check=True)
            except Exception as e:
                logging.warning(f"Could not automatically open plot file {plot_save_path}: {e}")

        except Exception as e:
            logging.error(f"Failed to generate plot for ROI {r_idx+1}: {e}")
            raise

    # --- Utility Methods ---

    def _compute_brightness_stats(self, roi_bgr: np.ndarray, background_brightness: Optional[float] = None) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Calculates brightness statistics for an ROI with optional background subtraction.

        Converts BGR to CIE LAB color space and uses the L* channel.
        Also extracts blue channel statistics for blue light analysis.

        Args:
            roi_bgr: The region of interest as a NumPy array (BGR format).
            background_brightness: Optional background L* value to subtract from all pixels.

        Returns:
            Tuple of (l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, 
                     b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median)
            L* values in 0-100 range, Blue values in 0-255 range
            or (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if the ROI is invalid or calculation fails.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)

            # Convert raw L to L* scale (0–100)
            l_star = l_chan * 100.0 / 255.0
            
            # Extract blue channel (BGR format, so blue is index 0)
            blue_chan = roi_bgr[:, :, 0].astype(np.float32)

            # Calculate raw L* statistics (unthresholded)
            l_raw_mean = float(np.mean(l_star))
            l_raw_median = float(np.median(l_star))
            
            # Calculate raw blue statistics (unthresholded)
            b_raw_mean = float(np.mean(blue_chan))
            b_raw_median = float(np.median(blue_chan))
            
            # Calculate background-subtracted statistics if background provided
            if background_brightness is not None:
                # Filter pixels above background threshold, then subtract background
                above_background_mask = l_star > background_brightness
                
                # Apply morphological operations to clean up the mask (remove noise/stray pixels)
                if np.any(above_background_mask):
                    # Create structuring element for morphological operations
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (MORPHOLOGICAL_KERNEL_SIZE, MORPHOLOGICAL_KERNEL_SIZE))
                    
                    # Convert boolean mask to uint8 for morphological operations
                    mask_uint8 = above_background_mask.astype(np.uint8) * 255
                    
                    # Apply opening (erosion followed by dilation) to remove small noise
                    cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
                    
                    # Convert back to boolean mask
                    above_background_mask = cleaned_mask > 0
                
                if np.any(above_background_mask):
                    # Only analyze pixels above background threshold
                    filtered_l_pixels = l_star[above_background_mask]
                    filtered_b_pixels = blue_chan[above_background_mask]
                    
                    # Background-subtracted L* statistics
                    bg_subtracted_l_pixels = filtered_l_pixels - background_brightness
                    l_bg_sub_mean = float(np.mean(bg_subtracted_l_pixels))
                    l_bg_sub_median = float(np.median(bg_subtracted_l_pixels))
                    
                    # Blue channel statistics for masked pixels (no background subtraction for blue)
                    b_bg_sub_mean = float(np.mean(filtered_b_pixels))
                    b_bg_sub_median = float(np.median(filtered_b_pixels))
                else:
                    # No pixels above background - return 0
                    l_bg_sub_mean = 0.0
                    l_bg_sub_median = 0.0
                    b_bg_sub_mean = 0.0
                    b_bg_sub_median = 0.0
            else:
                l_bg_sub_mean = l_raw_mean
                l_bg_sub_median = l_raw_median
                b_bg_sub_mean = b_raw_mean
                b_bg_sub_median = b_raw_median
            
            return l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median

        except cv2.error as e:
            print(f"OpenCV error during brightness computation: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        except Exception as e:
            print(f"Error during brightness computation: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def _compute_background_brightness(self, frame: np.ndarray) -> Optional[float]:
        """
        Calculate background ROI brightness for current frame.
        
        Args:
            frame: Current video frame in BGR format
            
        Returns:
            90th percentile L* brightness of background ROI, or None if no background ROI defined
        """
        if self.background_roi_idx is None or frame is None:
            return None
            
        if not (0 <= self.background_roi_idx < len(self.rects)):
            return None
            
        try:
            pt1, pt2 = self.rects[self.background_roi_idx]
            roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            if roi.size == 0:
                return None
                
            # Convert to LAB and get L* channel
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)
            l_star = l_chan * 100.0 / 255.0
            
            return float(np.percentile(l_star, 90))
            
        except Exception as e:
            print(f"Error computing background brightness: {e}")
            return None

    def _compute_brightness(self, roi_bgr: np.ndarray) -> float:
        """
        Legacy method for backward compatibility.
        Returns only the mean brightness for existing code that expects a single value.
        """
        l_raw_mean, _, _, _, _, _, _, _ = self._compute_brightness_stats(roi_bgr)
        return l_raw_mean
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create and show the main window
    win = VideoAnalyzer()
    win.show()

    sys.exit(app.exec_())