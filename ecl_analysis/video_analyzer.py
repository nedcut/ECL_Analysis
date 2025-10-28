"""Main GUI for the Brightness Sorcerer application."""

import json
import logging
import os
import time
from string import Template
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from .audio import AudioAnalyzer, AudioManager
from .cache import FrameCache
from .constants import (
    AUTO_DETECT_BASELINE_PERCENTILE,
    BRIGHTNESS_NOISE_FLOOR_PERCENTILE,
    COLOR_ACCENT,
    COLOR_ACCENT_HOVER,
    COLOR_BACKGROUND,
    COLOR_BRIGHTNESS_LABEL,
    COLOR_ERROR,
    COLOR_FOREGROUND,
    COLOR_INFO,
    COLOR_SECONDARY,
    COLOR_SECONDARY_LIGHT,
    COLOR_SUCCESS,
    COLOR_WARNING,
    DEFAULT_FONT_FAMILY,
    DEFAULT_MANUAL_THRESHOLD,
    DEFAULT_SETTINGS_FILE,
    FRAME_CACHE_SIZE,
    JUMP_FRAMES,
    MAX_RECENT_FILES,
    MORPHOLOGICAL_KERNEL_SIZE,
    MOUSE_RESIZE_HANDLE_SENSITIVITY,
    ROI_COLORS,
    ROI_LABEL_FONT_SCALE,
    ROI_LABEL_THICKNESS,
    ROI_THICKNESS_DEFAULT,
    ROI_THICKNESS_SELECTED,
)
from .dependencies import PLOTLY_AVAILABLE, go, make_subplots


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a hex color string like '#RRGGBB' to an rgba() string."""
    stripped = color.lstrip('#')
    if len(stripped) != 6:
        raise ValueError("Expected a 6-character hex color.")
    r = int(stripped[0:2], 16)
    g = int(stripped[2:4], 16)
    b = int(stripped[4:6], 16)
    clamped_alpha = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{clamped_alpha})"

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
        
        # Audio system
        self.audio_manager = AudioManager()
        self.audio_analyzer = AudioAnalyzer()
        
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
        # Fixed mask across frames
        self.use_fixed_mask = False
        self.fixed_roi_masks: List[Optional[np.ndarray]] = []  # aligned with self.rects

        # Noise filtering parameters (adjustable via UI)
        self.morphological_kernel_size = MORPHOLOGICAL_KERNEL_SIZE
        self.background_percentile = 90.0  # For background ROI threshold calculation
        self.noise_floor_threshold = 0.0   # Additional noise floor filtering

        # Video playback
        self.is_playing = False
        self.playback_timer = QtCore.QTimer()
        self.playback_fps = 30.0  # Default playback FPS
        self.playback_speed = 1.0  # Playback speed multiplier

    def _load_settings(self):
        """Load application settings from file."""
        try:
            if os.path.exists(DEFAULT_SETTINGS_FILE):
                with open(DEFAULT_SETTINGS_FILE, 'r') as f:
                    self.settings = json.load(f)
                    self.recent_files = self.settings.get('recent_files', [])
                    
                    # Load audio settings
                    audio_enabled = self.settings.get('audio_enabled', True)
                    audio_volume = self.settings.get('audio_volume', 0.7)
                    self.audio_manager.set_enabled(audio_enabled)
                    self.audio_manager.set_volume(audio_volume)

        except Exception as e:
            logging.warning(f"Could not load settings: {e}")
            self.settings = {}
            self.recent_files = []

    def _save_settings(self):
        """Save application settings to file."""
        try:
            self.settings['recent_files'] = self.recent_files
            self.settings['audio_enabled'] = self.audio_manager.enabled
            self.settings['audio_volume'] = self.audio_manager.volume
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
        # After widgets exist, set a sensible initial split
        self._set_default_splitter_sizes()

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
        
        auto_detect_action = QtWidgets.QAction('&Detect from Audio', self)
        auto_detect_action.setShortcut('Ctrl+D')
        auto_detect_action.setStatusTip('Detect completion beeps in audio and calculate frame ranges')
        auto_detect_action.triggered.connect(self.auto_detect_range)
        analysis_menu.addAction(auto_detect_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        reset_layout_action = QtWidgets.QAction('&Reset Layout', self)
        reset_layout_action.setStatusTip('Reset splitter sizes and panel widths')
        reset_layout_action.triggered.connect(self._set_default_splitter_sizes)
        view_menu.addAction(reset_layout_action)

        # Settings menu
        settings_menu = menubar.addMenu('&Settings')
        
        audio_settings_action = QtWidgets.QAction('&Audio Settings...', self)
        audio_settings_action.setStatusTip('Configure audio feedback settings')
        audio_settings_action.triggered.connect(self._show_audio_settings_dialog)
        settings_menu.addAction(audio_settings_action)
        
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
                action.triggered.connect(lambda _checked, path=file_path: self._open_recent_file(path))
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
        # Playback shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, 
                          self.toggle_playback)
        
        # Frame navigation shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, 
                          lambda: self.step_frames(-1))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, 
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
<h3>Playback Shortcuts:</h3>
<b>Space:</b> Play/Pause video<br>

<h3>Navigation Shortcuts:</h3>
<b>Left/Right Arrow:</b> Previous/Next frame<br>
<b>Backspace:</b> Previous frame<br>
<b>Page Down/Up:</b> Jump 10 frames<br>
<b>Home/End:</b> Go to first/last frame<br>

<h3>Analysis Shortcuts:</h3>
<b>F5:</b> Run analysis<br>
<b>Ctrl+D:</b> Detect from audio<br>

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

    def _show_audio_settings_dialog(self):
        """Show audio settings dialog."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Audio Settings')
        dialog.setModal(True)
        dialog.resize(350, 200)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Audio enabled checkbox
        enabled_checkbox = QtWidgets.QCheckBox("Enable Audio Feedback")
        enabled_checkbox.setChecked(self.audio_manager.enabled)
        layout.addWidget(enabled_checkbox)
        
        # Volume slider
        volume_group = QtWidgets.QGroupBox("Volume")
        volume_layout = QtWidgets.QVBoxLayout(volume_group)
        
        volume_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        volume_slider.setRange(0, 100)
        volume_slider.setValue(int(self.audio_manager.volume * 100))
        volume_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        volume_slider.setTickInterval(25)
        
        volume_label = QtWidgets.QLabel(f"Volume: {int(self.audio_manager.volume * 100)}%")
        
        def update_volume_label(value):
            volume_label.setText(f"Volume: {value}%")
        
        volume_slider.valueChanged.connect(update_volume_label)
        
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(volume_slider)
        layout.addWidget(volume_group)
        
        # Test button
        test_layout = QtWidgets.QHBoxLayout()
        test_btn = QtWidgets.QPushButton("Test Audio")
        test_btn.clicked.connect(lambda: self.audio_manager.play_analysis_start())
        test_layout.addWidget(test_btn)
        test_layout.addStretch()
        layout.addLayout(test_layout)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        def accept_settings():
            self.audio_manager.set_enabled(enabled_checkbox.isChecked())
            self.audio_manager.set_volume(volume_slider.value() / 100.0)
            self._save_settings()
            dialog.accept()
        
        ok_btn.clicked.connect(accept_settings)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec_()

    def _apply_stylesheet(self):
        """Apply a modern, clean stylesheet to the application."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLOR_BACKGROUND};
                color: {COLOR_FOREGROUND};
                font-family: {DEFAULT_FONT_FAMILY};
                font-size: 14px;
            }}
            QSplitter::handle {{ background: {COLOR_SECONDARY_LIGHT}; }}
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
            QScrollArea, QScrollArea > QWidget > QWidget {{
                background-color: {COLOR_BACKGROUND};
            }}
            QTabWidget::pane {{ border: 1px solid {COLOR_SECONDARY_LIGHT}; border-radius: 6px; }}
            QTabBar::tab {{ padding: 6px 10px; }}
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
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SECONDARY_LIGHT}, stop: 1 {COLOR_SECONDARY});
                color: {COLOR_FOREGROUND};
                border: 1px solid {COLOR_SECONDARY};
                border-radius: 6px;
                padding: 8px 15px;
                font-size: 14px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_ACCENT}, stop: 1 #3c82c4);
                color: white;
                border: 1px solid {COLOR_ACCENT_HOVER};
                padding: 9px 16px;
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 #3170a8, stop: 1 {COLOR_ACCENT});
                padding: 7px 14px;
                border: 1px solid {COLOR_ACCENT};
            }}
            QPushButton:disabled {{
                background: {COLOR_SECONDARY};
                color: #888888;
                border: 1px solid {COLOR_SECONDARY};
            }}
            QPushButton:checked {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_ACCENT}, stop: 1 #3c82c4);
                color: white;
                border: 1px solid {COLOR_ACCENT_HOVER};
            }}
            QListWidget {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_BACKGROUND}, stop: 1 {COLOR_SECONDARY});
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                color: {COLOR_FOREGROUND};
                font-size: 13px;
                border-radius: 6px;
                padding: 4px;
            }}
            QListWidget::item {{
                border-radius: 4px;
                padding: 4px 8px;
                margin: 1px;
            }}
            QListWidget::item:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SECONDARY_LIGHT}, stop: 1 {COLOR_SECONDARY});
                border: 1px solid {COLOR_ACCENT};
            }}
            QListWidget::item:selected {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_ACCENT}, stop: 1 #3c82c4);
                color: white;
                border: 1px solid {COLOR_ACCENT_HOVER};
            }}
            QListWidget::item:selected:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_ACCENT_HOVER}, stop: 1 {COLOR_ACCENT});
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
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_BACKGROUND}, stop: 1 {COLOR_SECONDARY});
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                padding: 6px 8px;
                border-radius: 6px;
                min-height: 20px;
                color: {COLOR_FOREGROUND};
            }}
            QLineEdit:hover, QSpinBox:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SECONDARY}, stop: 1 {COLOR_SECONDARY_LIGHT});
                border: 1px solid {COLOR_ACCENT};
            }}
            QLineEdit:focus, QSpinBox:focus {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 white, stop: 1 #f8f9fa);
                border: 2px solid {COLOR_ACCENT};
                color: #1a1a1a;
                padding: 5px 7px;
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

            /* Modern Video Control Styling */
            QPushButton#playButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_ACCENT}, stop: 1 #3c82c4);
                color: white;
                border: 2px solid {COLOR_ACCENT_HOVER};
                border-radius: 20px;
                font-size: 16px;
                font-weight: bold;
                min-width: 44px;
                min-height: 36px;
            }}
            QPushButton#playButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_ACCENT_HOVER}, stop: 1 {COLOR_ACCENT});
                border: 2px solid #8fc8ff;
                transform: scale(1.05);
            }}
            QPushButton#playButton:pressed {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 #3170a8, stop: 1 {COLOR_ACCENT});
                border: 2px solid {COLOR_ACCENT};
            }}

            QPushButton#mediaButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SECONDARY_LIGHT}, stop: 1 {COLOR_SECONDARY});
                color: {COLOR_FOREGROUND};
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
                min-width: 36px;
                min-height: 36px;
            }}
            QPushButton#mediaButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_ACCENT}, stop: 1 #3c82c4);
                color: white;
                border: 1px solid {COLOR_ACCENT_HOVER};
            }}
            QPushButton#mediaButton:pressed {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 #3170a8, stop: 1 {COLOR_ACCENT});
            }}

            QPushButton#jumpButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SECONDARY_LIGHT}, stop: 1 {COLOR_SECONDARY});
                color: {COLOR_FOREGROUND};
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
                padding: 6px 12px;
            }}
            QPushButton#jumpButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_INFO}, stop: 1 #059aa8);
                color: white;
                border: 1px solid {COLOR_INFO};
            }}

            QPushButton#analysisButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SECONDARY_LIGHT}, stop: 1 {COLOR_SECONDARY});
                color: {COLOR_FOREGROUND};
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
                padding: 6px 12px;
            }}
            QPushButton#analysisButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SUCCESS}, stop: 1 #0d9488);
                color: white;
                border: 1px solid {COLOR_SUCCESS};
            }}

            QSlider#timelineSlider::groove:horizontal {{
                border: none;
                height: 8px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SECONDARY}, stop: 1 {COLOR_BACKGROUND});
                border-radius: 4px;
            }}
            QSlider#timelineSlider::handle:horizontal {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 white, stop: 1 {COLOR_ACCENT});
                border: 2px solid {COLOR_ACCENT_HOVER};
                width: 20px;
                margin: -8px 0;
                border-radius: 10px;
            }}
            QSlider#timelineSlider::handle:horizontal:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 white, stop: 1 {COLOR_ACCENT_HOVER});
                border: 2px solid #8fc8ff;
                width: 24px;
                margin: -10px 0;
            }}
            QSlider#timelineSlider::sub-page:horizontal {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SUCCESS}, stop: 1 #059669);
                border-radius: 4px;
            }}

            /* Enhanced ComboBox Styling */
            QComboBox {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SECONDARY_LIGHT}, stop: 1 {COLOR_SECONDARY});
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 20px;
                color: {COLOR_FOREGROUND};
            }}
            QComboBox:hover {{
                border: 1px solid {COLOR_ACCENT};
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_ACCENT}, stop: 1 #3c82c4);
                color: white;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid {COLOR_SECONDARY_LIGHT};
            }}
            QComboBox::down-arrow {{
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {COLOR_FOREGROUND};
                margin: 0px 6px;
            }}
            QComboBox:hover::down-arrow {{
                border-top: 6px solid white;
            }}

            /* Enhanced Frame Separator */
            QFrame[frameShape="5"] {{ /* VLine */
                color: {COLOR_SECONDARY_LIGHT};
                background-color: {COLOR_SECONDARY_LIGHT};
                max-width: 1px;
                margin: 4px 8px;
            }}

            /* Improved GroupBox styling */
            QGroupBox {{
                border: 2px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 8px;
                margin-top: 12px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SECONDARY}, stop: 1 {COLOR_BACKGROUND});
                font-weight: bold;
                font-size: 14px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 4px 8px;
                color: {COLOR_ACCENT};
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_BACKGROUND}, stop: 1 {COLOR_SECONDARY});
                border: 1px solid {COLOR_ACCENT};
                border-radius: 4px;
            }}

            /* Primary Button Styles */
            QPushButton#primaryButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_ACCENT}, stop: 1 #3c82c4);
                color: white;
                border: 1px solid {COLOR_ACCENT_HOVER};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
                min-height: 28px;
            }}
            QPushButton#primaryButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_ACCENT_HOVER}, stop: 1 {COLOR_ACCENT});
                border: 1px solid #8fc8ff;
            }}
            QPushButton#primaryButton:pressed {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 #3170a8, stop: 1 {COLOR_ACCENT});
            }}

            QPushButton#primaryActionButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SUCCESS}, stop: 1 #059669);
                color: white;
                border: 2px solid {COLOR_SUCCESS};
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
                min-height: 32px;
            }}
            QPushButton#primaryActionButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 #22c55e, stop: 1 {COLOR_SUCCESS});
                border: 2px solid #22c55e;
                transform: scale(1.02);
            }}
            QPushButton#primaryActionButton:pressed {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 #15803d, stop: 1 #059669);
            }}
            QPushButton#primaryActionButton:disabled {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                           stop: 0 {COLOR_SECONDARY}, stop: 1 {COLOR_BACKGROUND});
                color: #888888;
                border: 2px solid {COLOR_SECONDARY};
            }}
        """)

    def _create_layouts(self):
        """Create resizable panes with a splitter and add scrollable side panel."""
        # Main horizontal splitter between left (video + controls) and right (side panel)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Left container widget (keeps existing left_layout semantics)
        self.left_container = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QVBoxLayout(self.left_container)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(8)
        self.splitter.addWidget(self.left_container)

        # Right container inside a scroll area (prevents off‑screen controls)
        self.right_scroll = QtWidgets.QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.right_content = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout(self.right_content)
        self.right_layout.setContentsMargins(8, 8, 8, 8)
        self.right_layout.setSpacing(10)
        self.right_scroll.setWidget(self.right_content)
        self.splitter.addWidget(self.right_scroll)

        # Enhanced responsive sizing
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.right_scroll.setMinimumWidth(320)  # Slightly reduced for smaller screens
        self.right_scroll.setMaximumWidth(480)  # Prevent right panel from being too wide

        # Prevent either pane from collapsing to zero
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)

        # Set minimum size for main window to ensure usability
        self.setMinimumSize(800, 600)

    def _set_default_splitter_sizes(self):
        """Set responsive splitter sizes based on current window width and screen size."""
        try:
            total = max(1, self.width())

            # Adaptive split ratio based on window size
            if total <= 1000:
                # Smaller screens: give more space to video
                ratio = 0.72
            elif total <= 1400:
                # Medium screens: balanced layout
                ratio = 0.68
            else:
                # Large screens: slightly more space for side panel
                ratio = 0.65

            left = int(total * ratio)
            right = max(320, min(480, total - left))  # Clamp right panel size
            self.splitter.setSizes([left, right])
        except Exception:
            # Fallback sizes for different screen types
            total = max(1, self.width())
            if total <= 1000:
                self.splitter.setSizes([720, 320])
            else:
                self.splitter.setSizes([900, 420])

    def _create_widgets(self):
        """Create all the widgets and add them to layouts."""
        # --- Left Layout Widgets ---
        # Header section with improved spacing
        header_layout = QtWidgets.QVBoxLayout()
        header_layout.setSpacing(8)
        header_layout.setContentsMargins(8, 8, 8, 16)

        self.title_label = QtWidgets.QLabel("Brightness Sorcerer", self)
        self.title_label.setObjectName("titleLabel")
        header_layout.addWidget(self.title_label)

        # File info label
        self.file_info_label = QtWidgets.QLabel("No video loaded")
        self.file_info_label.setObjectName("statusLabel")
        header_layout.addWidget(self.file_info_label)

        # Open-file button with better styling
        self.open_btn = QtWidgets.QPushButton("📁 Open Video… (Ctrl+O)")
        self.open_btn.setObjectName("primaryButton")
        self.open_btn.setToolTip("Choose a video file from disk")
        self.open_btn.setFixedHeight(36)
        header_layout.addWidget(self.open_btn)

        self.left_layout.addLayout(header_layout)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        # Let image scale down responsively but not push controls off-screen
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_label.setMinimumHeight(240)
        self.image_label.setText("Drag & Drop Video File Here")
        self.left_layout.addWidget(self.image_label, stretch=1)

        # Modern Video Controls with consolidated layout
        self.video_controls_groupbox = QtWidgets.QGroupBox("Video Controls")
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setContentsMargins(12, 16, 12, 16)
        controls_layout.setSpacing(12)

        # Main Timeline Row - consolidated playback controls
        main_timeline_layout = QtWidgets.QHBoxLayout()
        main_timeline_layout.setSpacing(8)

        # Previous frame button
        self.prev_frame_btn = QtWidgets.QPushButton("⏮")
        self.prev_frame_btn.setToolTip("Previous Frame (Left Arrow)")
        self.prev_frame_btn.setFixedSize(40, 40)
        self.prev_frame_btn.setObjectName("mediaButton")
        main_timeline_layout.addWidget(self.prev_frame_btn)

        # Play/Pause button - larger and more prominent
        self.play_pause_btn = QtWidgets.QPushButton("⏵")
        self.play_pause_btn.setToolTip("Play/Pause video (Spacebar)")
        self.play_pause_btn.setFixedSize(48, 40)
        self.play_pause_btn.setObjectName("playButton")
        main_timeline_layout.addWidget(self.play_pause_btn)

        # Next frame button
        self.next_frame_btn = QtWidgets.QPushButton("⏭")
        self.next_frame_btn.setToolTip("Next Frame (Right Arrow)")
        self.next_frame_btn.setFixedSize(40, 40)
        self.next_frame_btn.setObjectName("mediaButton")
        main_timeline_layout.addWidget(self.next_frame_btn)

        # Timeline slider with frame info
        timeline_container = QtWidgets.QVBoxLayout()
        timeline_container.setSpacing(4)

        # Frame slider (main timeline)
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setToolTip("Drag to navigate frames or click to seek")
        self.frame_slider.setMinimumHeight(24)
        self.frame_slider.setObjectName("timelineSlider")
        timeline_container.addWidget(self.frame_slider)

        # Frame info row
        frame_info_layout = QtWidgets.QHBoxLayout()
        frame_info_layout.setSpacing(8)

        self.frame_label = QtWidgets.QLabel("Frame: 0 / 0")
        self.frame_label.setMinimumWidth(120)
        self.frame_label.setAlignment(QtCore.Qt.AlignLeft)
        frame_info_layout.addWidget(self.frame_label)

        frame_info_layout.addStretch()

        # Frame input
        frame_input_layout = QtWidgets.QHBoxLayout()
        frame_input_layout.setSpacing(4)
        frame_input_layout.addWidget(QtWidgets.QLabel("Go to:"))
        self.frame_spinbox = QtWidgets.QSpinBox()
        self.frame_spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.frame_spinbox.setToolTip("Enter frame number directly")
        self.frame_spinbox.setAlignment(QtCore.Qt.AlignCenter)
        self.frame_spinbox.setFixedWidth(80)
        frame_input_layout.addWidget(self.frame_spinbox)

        frame_info_layout.addLayout(frame_input_layout)
        timeline_container.addLayout(frame_info_layout)

        main_timeline_layout.addLayout(timeline_container)

        # Speed control - compact design
        speed_container = QtWidgets.QVBoxLayout()
        speed_container.setSpacing(4)
        speed_label = QtWidgets.QLabel("Speed")
        speed_label.setAlignment(QtCore.Qt.AlignCenter)
        speed_container.addWidget(speed_label)
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.speed_combo.setCurrentText("1×")
        self.speed_combo.setToolTip("Playback speed")
        self.speed_combo.setFixedWidth(60)
        speed_container.addWidget(self.speed_combo)
        main_timeline_layout.addLayout(speed_container)

        controls_layout.addLayout(main_timeline_layout)

        # Secondary Controls Row - Jump and Analysis
        secondary_layout = QtWidgets.QHBoxLayout()
        secondary_layout.setSpacing(12)

        # Jump controls group
        jump_group = QtWidgets.QHBoxLayout()
        jump_group.setSpacing(6)

        self.jump_back_btn = QtWidgets.QPushButton(f"⏪ {JUMP_FRAMES}")
        self.jump_back_btn.setToolTip(f"Jump back {JUMP_FRAMES} frames (Page Up)")
        self.jump_back_btn.setFixedHeight(32)
        self.jump_back_btn.setObjectName("jumpButton")
        jump_group.addWidget(self.jump_back_btn)

        self.jump_forward_btn = QtWidgets.QPushButton(f"{JUMP_FRAMES} ⏩")
        self.jump_forward_btn.setToolTip(f"Jump forward {JUMP_FRAMES} frames (Page Down)")
        self.jump_forward_btn.setFixedHeight(32)
        self.jump_forward_btn.setObjectName("jumpButton")
        jump_group.addWidget(self.jump_forward_btn)

        secondary_layout.addLayout(jump_group)

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        secondary_layout.addWidget(separator)

        # Analysis controls group
        analysis_group = QtWidgets.QHBoxLayout()
        analysis_group.setSpacing(6)

        self.set_start_btn = QtWidgets.QPushButton("Set Start")
        self.set_start_btn.setToolTip("Set current frame as analysis start")
        self.set_start_btn.setFixedHeight(32)
        self.set_start_btn.setObjectName("analysisButton")
        analysis_group.addWidget(self.set_start_btn)

        self.set_end_btn = QtWidgets.QPushButton("Set End")
        self.set_end_btn.setToolTip("Set current frame as analysis end")
        self.set_end_btn.setFixedHeight(32)
        self.set_end_btn.setObjectName("analysisButton")
        analysis_group.addWidget(self.set_end_btn)

        self.auto_detect_btn = QtWidgets.QPushButton("🎵 Detect from Audio")
        self.auto_detect_btn.setToolTip("Detect completion beeps in audio and calculate frame ranges (Ctrl+D)")
        self.auto_detect_btn.setFixedHeight(32)
        self.auto_detect_btn.setObjectName("analysisButton")
        analysis_group.addWidget(self.auto_detect_btn)

        secondary_layout.addLayout(analysis_group)
        secondary_layout.addStretch()

        controls_layout.addLayout(secondary_layout)
        
        self.video_controls_groupbox.setLayout(controls_layout)
        # Keep controls compact to avoid crowding
        self.video_controls_groupbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.video_controls_groupbox.setMaximumHeight(210)
        self.left_layout.addWidget(self.video_controls_groupbox)

        # Analysis Section with improved grouping and spacing
        analysis_section = QtWidgets.QVBoxLayout()
        analysis_section.setSpacing(12)
        analysis_section.setContentsMargins(8, 16, 8, 8)

        # Analysis Name Layout with better spacing
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.setSpacing(8)
        name_label = QtWidgets.QLabel("Analysis Name:")
        name_label.setMinimumWidth(100)
        name_layout.addWidget(name_label)
        self.analysis_name_input = QtWidgets.QLineEdit()
        self.analysis_name_input.setPlaceholderText("DefaultAnalysis")
        self.analysis_name_input.setFixedHeight(32)
        name_layout.addWidget(self.analysis_name_input)
        analysis_section.addLayout(name_layout)

        # Action Buttons Layout with prominent styling
        action_layout = QtWidgets.QHBoxLayout()
        self.analyze_btn = QtWidgets.QPushButton('🔍 Analyze Brightness (F5)')
        self.analyze_btn.setObjectName("primaryActionButton")
        self.analyze_btn.setToolTip("Run brightness analysis on the selected frame range and ROIs")
        self.analyze_btn.setFixedHeight(40)
        action_layout.addWidget(self.analyze_btn)
        analysis_section.addLayout(action_layout)

        self.left_layout.addLayout(analysis_section)

        # --- Right Panel Widgets (not yet added to layout; placed into tabs below) ---
        # Video info group
        self.video_info_groupbox = QtWidgets.QGroupBox("Video Information")
        video_info_layout = QtWidgets.QVBoxLayout()
        self.video_info_label = QtWidgets.QLabel("No video loaded")
        self.video_info_label.setWordWrap(True)
        video_info_layout.addWidget(self.video_info_label)
        self.video_info_groupbox.setLayout(video_info_layout)

        # Brightness Display
        self.brightness_groupbox = QtWidgets.QGroupBox("ROI Brightness: Mean±Median (Current Frame)")
        brightness_groupbox_layout = QtWidgets.QVBoxLayout()
        self.brightness_display_label = QtWidgets.QLabel("N/A")
        self.brightness_display_label.setObjectName("brightnessDisplayLabel")
        brightness_groupbox_layout.addWidget(self.brightness_display_label)
        self.brightness_groupbox.setLayout(brightness_groupbox_layout)

        # -- Run Duration Settings
        self.run_duration_groupbox = QtWidgets.QGroupBox("Run Duration (Optional)")
        run_duration_layout = QtWidgets.QVBoxLayout()
        
        duration_input_layout = QtWidgets.QHBoxLayout()
        duration_input_layout.addWidget(QtWidgets.QLabel("Expected Duration (sec):"))
        self.run_duration_spin = QtWidgets.QDoubleSpinBox()
        self.run_duration_spin.setDecimals(1)
        self.run_duration_spin.setRange(0.0, 3600.0)  # 0 to 1 hour
        self.run_duration_spin.setSingleStep(0.5)
        self.run_duration_spin.setValue(0.0)  # 0 = disabled
        self.run_duration_spin.setToolTip("Expected run duration for validation (0 = disabled)")
        duration_input_layout.addWidget(self.run_duration_spin)
        run_duration_layout.addLayout(duration_input_layout)
        
        self.run_duration_groupbox.setLayout(run_duration_layout)
        self.right_layout.addWidget(self.run_duration_groupbox)

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

        # Visualization Controls
        self.viz_groupbox = QtWidgets.QGroupBox("Visualization")
        viz_layout = QtWidgets.QVBoxLayout()
        
        self.show_mask_checkbox = QtWidgets.QCheckBox("Show Pixel Mask")
        self.show_mask_checkbox.setToolTip("Highlight analyzed pixels in red overlay")
        self.show_mask_checkbox.setChecked(self.show_pixel_mask)
        viz_layout.addWidget(self.show_mask_checkbox)

        # Fixed mask controls
        self.use_fixed_mask_checkbox = QtWidgets.QCheckBox("Use Fixed Mask (from frame)")
        self.use_fixed_mask_checkbox.setToolTip("Apply a pixel mask captured on a specific frame to all frames during analysis and visualization")
        viz_layout.addWidget(self.use_fixed_mask_checkbox)

        mask_btn_layout = QtWidgets.QHBoxLayout()
        self.capture_mask_btn = QtWidgets.QPushButton("Capture Mask From Current Frame")
        self.capture_mask_btn.setToolTip("Compute the analyzed-pixel mask for each ROI based on the current frame and reuse it for all frames")
        mask_btn_layout.addWidget(self.capture_mask_btn)

        self.auto_brightest_mask_btn = QtWidgets.QPushButton("Auto-Capture from Brightest Frame")
        self.auto_brightest_mask_btn.setToolTip("Automatically find the brightest frame in the current range and capture masks from it")
        mask_btn_layout.addWidget(self.auto_brightest_mask_btn)

        mask_btn_layout.addStretch()
        viz_layout.addLayout(mask_btn_layout)

        self.mask_status_label = QtWidgets.QLabel("Mask: none")
        self.mask_status_label.setObjectName("statusLabel")
        viz_layout.addWidget(self.mask_status_label)

        self.mask_pixel_count_label = QtWidgets.QLabel("Mask Pixels: n/a")
        self.mask_pixel_count_label.setObjectName("statusLabel")
        viz_layout.addWidget(self.mask_pixel_count_label)

        # Noise filtering controls
        noise_groupbox = QtWidgets.QGroupBox("Noise Filtering")
        noise_layout = QtWidgets.QVBoxLayout()

        # Morphological kernel size
        kernel_layout = QtWidgets.QHBoxLayout()
        kernel_layout.addWidget(QtWidgets.QLabel("Morphological Kernel:"))
        self.kernel_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.kernel_size_slider.setRange(1, 15)
        self.kernel_size_slider.setValue(self.morphological_kernel_size)
        self.kernel_size_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.kernel_size_slider.setTickInterval(2)
        self.kernel_size_label = QtWidgets.QLabel(f"{self.morphological_kernel_size}×{self.morphological_kernel_size}")
        kernel_layout.addWidget(self.kernel_size_slider)
        kernel_layout.addWidget(self.kernel_size_label)
        noise_layout.addLayout(kernel_layout)

        # Background percentile
        bg_percentile_layout = QtWidgets.QHBoxLayout()
        bg_percentile_layout.addWidget(QtWidgets.QLabel("Background Percentile:"))
        self.bg_percentile_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bg_percentile_slider.setRange(50, 99)
        self.bg_percentile_slider.setValue(int(self.background_percentile))
        self.bg_percentile_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.bg_percentile_slider.setTickInterval(10)
        self.bg_percentile_label = QtWidgets.QLabel(f"{self.background_percentile:.0f}%")
        bg_percentile_layout.addWidget(self.bg_percentile_slider)
        bg_percentile_layout.addWidget(self.bg_percentile_label)
        noise_layout.addLayout(bg_percentile_layout)

        # Noise floor threshold
        noise_floor_layout = QtWidgets.QHBoxLayout()
        noise_floor_layout.addWidget(QtWidgets.QLabel("Noise Floor (L*):"))
        self.noise_floor_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.noise_floor_slider.setRange(0, 20)
        self.noise_floor_slider.setValue(int(self.noise_floor_threshold * 2))  # 0.5 precision
        self.noise_floor_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.noise_floor_slider.setTickInterval(4)
        self.noise_floor_label = QtWidgets.QLabel(f"{self.noise_floor_threshold:.1f}")
        noise_floor_layout.addWidget(self.noise_floor_slider)
        noise_floor_layout.addWidget(self.noise_floor_label)
        noise_layout.addLayout(noise_floor_layout)

        noise_groupbox.setLayout(noise_layout)
        viz_layout.addWidget(noise_groupbox)

        self.viz_groupbox.setLayout(viz_layout)

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

        # Second row of buttons
        rect_btn_layout2 = QtWidgets.QHBoxLayout()
        self.set_bg_roi_btn = QtWidgets.QPushButton("Set as Background")
        self.set_bg_roi_btn.setToolTip("Set the selected ROI as the background reference for threshold calculation")
        rect_btn_layout2.addWidget(self.set_bg_roi_btn)
        rect_btn_layout2.addStretch()  # Push button to the left

        rect_groupbox_layout.addLayout(rect_btn_layout)
        rect_groupbox_layout.addLayout(rect_btn_layout2)
        self.rect_groupbox.setLayout(rect_groupbox_layout)
        # Cache status
        self.cache_status_label = QtWidgets.QLabel("Cache: 0 frames")
        self.cache_status_label.setObjectName("statusLabel")

        # Results/Status Label
        self.results_label = QtWidgets.QLabel("Load a video to begin analysis.")
        self.results_label.setObjectName("resultsLabel")
        self.results_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.results_label.setWordWrap(True)
        self.results_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

        # Organize right panel into tabs for clarity and prevent overflow with a scroll area (added in _create_layouts)
        self.side_tabs = QtWidgets.QTabWidget()
        self.side_tabs.setDocumentMode(True)
        self.side_tabs.setTabPosition(QtWidgets.QTabWidget.North)
        self.side_tabs.setMinimumWidth(360)

        # Info tab
        info_tab = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_tab)
        info_layout.addWidget(self.video_info_groupbox)
        info_layout.addWidget(self.brightness_groupbox)
        info_layout.addWidget(self.cache_status_label)
        info_layout.addStretch()
        self.side_tabs.addTab(info_tab, "Info")

        # Analysis tab
        analysis_tab = QtWidgets.QWidget()
        analysis_tab_layout = QtWidgets.QVBoxLayout(analysis_tab)
        analysis_tab_layout.addWidget(self.run_duration_groupbox)
        analysis_tab_layout.addWidget(self.threshold_groupbox)
        analysis_tab_layout.addStretch()
        self.side_tabs.addTab(analysis_tab, "Analysis")

        # Visualization tab
        viz_tab = QtWidgets.QWidget()
        viz_tab_layout = QtWidgets.QVBoxLayout(viz_tab)
        viz_tab_layout.addWidget(self.viz_groupbox)
        viz_tab_layout.addStretch()
        self.side_tabs.addTab(viz_tab, "Visualization")

        # ROIs tab
        roi_tab = QtWidgets.QWidget()
        roi_tab_layout = QtWidgets.QVBoxLayout(roi_tab)
        roi_tab_layout.addWidget(self.rect_groupbox)
        roi_tab_layout.addStretch()
        self.side_tabs.addTab(roi_tab, "ROIs")

        # Status tab
        status_tab = QtWidgets.QWidget()
        status_layout = QtWidgets.QVBoxLayout(status_tab)
        status_layout.addWidget(self.results_label)
        self.side_tabs.addTab(status_tab, "Status")

        # Add the tab widget to the right layout (scroll area content)
        self.right_layout.addWidget(self.side_tabs)

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
        
        # Playback controls
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        self.playback_timer.timeout.connect(self.advance_frame)

        self.analyze_btn.clicked.connect(self.analyze_video)

        self.rect_list.currentRowChanged.connect(self.select_rectangle_from_list)
        self.add_rect_btn.clicked.connect(self.toggle_add_rectangle_mode)
        self.del_rect_btn.clicked.connect(self.delete_selected_rectangle)
        self.clear_rect_btn.clicked.connect(self.clear_all_rectangles)
        self.set_bg_roi_btn.clicked.connect(self._set_background_roi)

        # Connect mouse events directly
        self.image_label.mousePressEvent = self.image_mouse_press
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release

        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self.set_bg_btn.clicked.connect(self._set_background_roi)
        self.show_mask_checkbox.toggled.connect(self._on_mask_checkbox_toggled)
        self.use_fixed_mask_checkbox.toggled.connect(self._on_use_fixed_mask_toggled)
        self.capture_mask_btn.clicked.connect(self._capture_fixed_masks)
        self.auto_brightest_mask_btn.clicked.connect(self._auto_capture_brightest_frame_masks)

        # Noise filtering slider connections
        self.kernel_size_slider.valueChanged.connect(self._on_kernel_size_changed)
        self.bg_percentile_slider.valueChanged.connect(self._on_bg_percentile_changed)
        self.noise_floor_slider.valueChanged.connect(self._on_noise_floor_changed)

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
        self.auto_detect_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.analyze_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        
        # Playback controls
        self.play_pause_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.speed_combo.setEnabled(video_loaded and not self._analysis_in_progress)
        self.add_rect_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.del_rect_btn.setEnabled(video_loaded and self.selected_rect_idx is not None and not self._analysis_in_progress)
        self.clear_rect_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        self.set_bg_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        self.set_bg_roi_btn.setEnabled(video_loaded and self.selected_rect_idx is not None and not self._analysis_in_progress)
        self.threshold_spin.setEnabled(not self._analysis_in_progress)
        # Fixed mask controls
        if hasattr(self, 'use_fixed_mask_checkbox'):
            self.use_fixed_mask_checkbox.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        if hasattr(self, 'capture_mask_btn'):
            self.capture_mask_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        if hasattr(self, 'auto_brightest_mask_btn'):
            self.auto_brightest_mask_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)

    def _update_cache_status(self):
        """Update cache status display."""
        cache_size = self.frame_cache.get_size()
        self.cache_status_label.setText(f"Cache: {cache_size}/{FRAME_CACHE_SIZE} frames")

    def _update_mask_pixel_count_display(self):
        """Update sidebar label with total fixed mask pixels."""
        if not hasattr(self, "mask_pixel_count_label"):
            return

        count = 0
        mask_found = False
        for mask in getattr(self, "fixed_roi_masks", []):
            if isinstance(mask, np.ndarray):
                mask_found = True
                count += int(np.count_nonzero(mask))

        if mask_found:
            self.mask_pixel_count_label.setText(f"Mask Pixels: {count:,}")
        else:
            self.mask_pixel_count_label.setText("Mask Pixels: n/a")

    def _invalidate_fixed_masks(self, reason: str = ""):
        """Clear captured fixed masks when ROIs change or become invalid.
        Optionally provide a reason for UI feedback.
        """
        if self.fixed_roi_masks:
            self.fixed_roi_masks = [None for _ in self.rects]
            if reason:
                self.mask_status_label.setText(f"Mask: cleared ({reason})")
            else:
                self.mask_status_label.setText("Mask: cleared")
        self._update_mask_pixel_count_display()

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
        Enhanced resize event handler with responsive layout updates.
        Update cached label size *without* immediately redrawing the frame.
        Calling show_frame() synchronously inside resizeEvent can create
        a feedback loop: the freshly scaled pixmap changes the label's
        sizeHint, Qt recalculates the layout, and another resizeEvent fires.
        By deferring the redraw with QTimer.singleShot(0, …) we let the
        resize settle first and repaint exactly once.
        """
        if hasattr(self, "image_label") and self.image_label.size().isValid():
            self._current_image_size = self.image_label.size()

            # Schedule a one‑shot repaint after the event loop returns.
            if self.frame is not None:
                QtCore.QTimer.singleShot(0, self.show_frame)

        # Update splitter sizes responsively when window is resized significantly
        if hasattr(self, "splitter") and event:
            old_size = event.oldSize()
            new_size = event.size()
            if old_size.isValid():
                # Only update if resize is significant (more than 100px change)
                width_change = abs(new_size.width() - old_size.width())
                if width_change > 100:
                    QtCore.QTimer.singleShot(100, self._set_default_splitter_sizes)

        # Call base-class handler last (standard Qt practice)
        super().resizeEvent(event)

    # --- Video Loading and Frame Display ---

    def load_video(self):
        """Loads the video specified by self.video_path."""
        # Show loading feedback
        self.statusBar().showMessage("📂 Loading video...")
        self.file_info_label.setText("📂 Loading video file...")
        QtWidgets.QApplication.processEvents()

        if self.cap:
            self.cap.release()
            self.cap = None

        if not self.video_path or not os.path.exists(self.video_path):
            QtWidgets.QMessageBox.critical(self, 'Error', 'Video path is invalid or file does not exist.')
            self._reset_state()
            return

        self.statusBar().showMessage("⚙️ Initializing video decoder...")
        QtWidgets.QApplication.processEvents()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, 'Error', f'Could not open video file: {os.path.basename(self.video_path)}')
            self._reset_state()
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.playback_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.playback_fps <= 0:
            self.playback_fps = 30.0  # Default fallback
            
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
            
            self.results_label.setText(f"✅ Loaded: {os.path.basename(self.video_path)}\n"
                                       f"📊 Frames: {self.total_frames:,} | ⏱️ FPS: {self.playback_fps:.1f}\n"
                                       f"⏳ Duration: {self.total_frames/self.playback_fps:.1f}s\n"
                                       "🎯 Draw ROIs or use Auto-Detect to begin!")
            self._update_widget_states(video_loaded=True, rois_exist=bool(self.rects))
            self.statusBar().showMessage(f"✅ Successfully loaded: {os.path.basename(self.video_path)}")

            # Reset fixed masks on new video load
            self.fixed_roi_masks = [None for _ in self.rects]
            self.use_fixed_mask_checkbox.setChecked(False)
            self.mask_status_label.setText("Mask: none")
            self._update_mask_pixel_count_display()

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
        
        # Stop playback and reset controls
        self.stop_playback()
        self.speed_combo.setCurrentText("1x")
        self.playback_speed = 1.0
        
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

    def toggle_playback(self):
        """Toggle video playback on/off."""
        if not self.cap or not self.cap.isOpened() or self.total_frames == 0:
            return
            
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start video playback."""
        if not self.cap or not self.cap.isOpened() or self.total_frames == 0:
            return
        
        self.is_playing = True
        self.play_pause_btn.setText("⏸")
        
        # Calculate timer interval based on FPS and speed
        if hasattr(self, 'playback_fps') and self.playback_fps > 0:
            interval = int(1000 / (self.playback_fps * self.playback_speed))
        else:
            interval = int(1000 / (30.0 * self.playback_speed))  # Default 30 FPS
        
        self.playback_timer.start(interval)
    
    def stop_playback(self):
        """Stop video playback."""
        self.is_playing = False
        self.play_pause_btn.setText("⏵")
        self.playback_timer.stop()
    
    def advance_frame(self):
        """Advance to next frame during playback."""
        if not self.is_playing or not self.cap or not self.cap.isOpened():
            return
        
        next_frame = self.current_frame_index + 1
        if next_frame >= self.total_frames:
            # Reached end of video, stop playback
            self.stop_playback()
            return
        
        self.frame_slider.setValue(next_frame)
    
    def on_speed_changed(self, speed_text: str):
        """Handle playback speed change."""
        try:
            speed_value = float(speed_text.replace('x', ''))
            self.playback_speed = speed_value
            
            # If currently playing, restart timer with new interval
            if self.is_playing:
                self.stop_playback()
                self.start_playback()
        except ValueError:
            logging.warning(f"Invalid speed value: {speed_text}")

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
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, ROI_LABEL_FONT_SCALE, ROI_LABEL_THICKNESS)
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
                roi_mask = None
                if self.use_fixed_mask and idx < len(self.fixed_roi_masks):
                    roi_mask = self.fixed_roi_masks[idx]
                    if isinstance(roi_mask, np.ndarray) and roi_mask.shape[:2] != roi.shape[:2]:
                        roi_mask = None
                brightness_stats = self._compute_brightness_stats(roi, background_brightness, roi_mask)
                l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median = brightness_stats[:6]
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
        # Keep fixed masks list aligned with rects
        if len(self.fixed_roi_masks) != len(self.rects):
            # Resize/realign masks; default to None for new or mismatched entries
            new_masks: List[Optional[np.ndarray]] = []
            for i in range(len(self.rects)):
                if i < len(self.fixed_roi_masks):
                    new_masks.append(self.fixed_roi_masks[i])
                else:
                    new_masks.append(None)
            self.fixed_roi_masks = new_masks
        self._update_mask_pixel_count_display()

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
            self.image_label.unsetCursor()
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
            # Remove corresponding fixed mask and invalidate
            if self.selected_rect_idx is not None and self.selected_rect_idx < len(self.fixed_roi_masks):
                del self.fixed_roi_masks[self.selected_rect_idx]
            self._invalidate_fixed_masks("ROI deleted")
            
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
            self.fixed_roi_masks = []
            self.use_fixed_mask_checkbox.setChecked(False)
            self.mask_status_label.setText("Mask: none")
            self._update_mask_pixel_count_display()
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

        # Precompute L* for current frame and derive background brightness
        l_star_frame = self._compute_l_star_frame(frame)
        background_brightness = self._compute_background_brightness(frame, frame_l_star=l_star_frame)
        
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
                roi = frame[y1:y2, x1:x2]
                roi_l_star = l_star_frame[y1:y2, x1:x2]
                try:
                    use_fixed = self.use_fixed_mask and roi_idx < len(self.fixed_roi_masks) and isinstance(self.fixed_roi_masks[roi_idx], np.ndarray)
                    mask = None
                    if use_fixed:
                        fixed_mask = self.fixed_roi_masks[roi_idx]
                        if fixed_mask is not None and fixed_mask.shape[:2] == roi.shape[:2]:
                            mask = fixed_mask.astype(bool)
                        else:
                            # Shape mismatch - ignore fixed mask
                            mask = None

                    if mask is None:
                        # Derive mask from current frame using cached L* channel
                        if background_brightness is not None:
                            mask = roi_l_star > background_brightness
                        else:
                            mask = np.ones_like(roi_l_star, dtype=bool)
                    
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

    def _on_use_fixed_mask_toggled(self, checked: bool):
        """Enable/disable using a fixed mask across frames."""
        self.use_fixed_mask = checked
        if checked and (not self.fixed_roi_masks or all(m is None for m in self.fixed_roi_masks)):
            self.mask_status_label.setText("Mask: none (capture from a frame)")
        elif checked:
            self.mask_status_label.setText("Mask: active")
        else:
            self.mask_status_label.setText("Mask: disabled")
        if self.frame is not None:
            self.show_frame()

    def _capture_fixed_masks(self):
        """Capture analyzed-pixel masks for all ROIs based on the current frame."""
        if self.frame is None or not self.rects:
            QtWidgets.QMessageBox.information(self, "Capture Mask", "Load a video and define at least one ROI.")
            return
        
        frame = self.frame
        fh, fw = frame.shape[:2]
        # Determine background brightness once from current frame
        l_star_frame = self._compute_l_star_frame(frame)
        background_brightness = self._compute_background_brightness(frame, frame_l_star=l_star_frame)
        
        masks: List[Optional[np.ndarray]] = []
        created_any = False
        for roi_idx, (pt1, pt2) in enumerate(self.rects):
            if roi_idx == self.background_roi_idx:
                masks.append(None)
                continue
            x1 = max(0, min(pt1[0], fw - 1))
            y1 = max(0, min(pt1[1], fh - 1))
            x2 = max(0, min(pt2[0], fw - 1))
            y2 = max(0, min(pt2[1], fh - 1))
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                roi_l_star = l_star_frame[y1:y2, x1:x2]
                try:
                    if background_brightness is not None:
                        mask = roi_l_star > background_brightness
                        # Morphological cleanup similar to analysis
                        if np.any(mask):
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morphological_kernel_size, self.morphological_kernel_size))
                            mask_uint8 = mask.astype(np.uint8) * 255
                            cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
                            mask = cleaned > 0
                    else:
                        # If no background brightness, default to full ROI
                        mask = np.ones(roi_l_star.shape, dtype=bool)
                    masks.append(mask)
                    created_any = True
                except Exception as e:
                    logging.warning(f"Failed to capture mask for ROI {roi_idx+1}: {e}")
                    masks.append(None)
            else:
                masks.append(None)
        
        self.fixed_roi_masks = masks
        if created_any:
            self.mask_status_label.setText("Mask: captured from current frame")
            if not self.use_fixed_mask:
                # Auto-enable usage for convenience
                self.use_fixed_mask_checkbox.setChecked(True)
        else:
            self.mask_status_label.setText("Mask: none (could not capture)")
        self._update_mask_pixel_count_display()
        if self.frame is not None:
            self.show_frame()

    def _auto_capture_brightest_frame_masks(self):
        """Find the brightest frame in the current range and capture masks from it."""
        if self.cap is None or not self.rects:
            QtWidgets.QMessageBox.information(self, "Auto-Capture Masks",
                                            "Load a video and define at least one ROI first.")
            return

        if not self.rects:
            QtWidgets.QMessageBox.information(self, "Auto-Capture Masks",
                                            "Define at least one ROI before capturing masks.")
            return

        # Use current frame range or full video if not set
        start_frame = max(0, self.start_frame)
        end_frame = min(self.total_frames - 1, self.end_frame if self.end_frame is not None else self.total_frames - 1)

        if start_frame >= end_frame:
            QtWidgets.QMessageBox.information(self, "Auto-Capture Masks",
                                            "Invalid frame range for analysis.")
            return

        # Show progress dialog
        progress = QtWidgets.QProgressDialog("Finding brightest frame...", "Cancel", 0, end_frame - start_frame, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()

        brightest_frame_idx = start_frame
        max_brightness = 0.0

        # Sample every 10th frame for performance, or all frames if range is small
        step = max(1, (end_frame - start_frame) // 100)

        try:
            for frame_idx in range(start_frame, end_frame + 1, step):
                if progress.wasCanceled():
                    return

                progress.setValue(frame_idx - start_frame)
                QtWidgets.QApplication.processEvents()

                # Get frame from cache or video
                frame = self.frame_cache.get(frame_idx)
                if frame is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        continue
                    self.frame_cache.put(frame_idx, frame)
                l_star_frame = self._compute_l_star_frame(frame)

                # Calculate average brightness of all non-background ROIs
                frame_brightness = 0.0
                roi_count = 0

                for roi_idx, (pt1, pt2) in enumerate(self.rects):
                    if roi_idx == self.background_roi_idx:
                        continue

                    fh, fw = frame.shape[:2]
                    x1 = max(0, min(pt1[0], fw - 1))
                    y1 = max(0, min(pt1[1], fh - 1))
                    x2 = max(0, min(pt2[0], fw - 1))
                    y2 = max(0, min(pt2[1], fh - 1))

                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2]
                        roi_l_star = l_star_frame[y1:y2, x1:x2]
                        l_raw_mean, _, _, _, _, _, _, _ = self._compute_brightness_stats(roi, roi_l_star=roi_l_star)
                        frame_brightness += l_raw_mean
                        roi_count += 1

                if roi_count > 0:
                    frame_brightness /= roi_count
                    if frame_brightness > max_brightness:
                        max_brightness = frame_brightness
                        brightest_frame_idx = frame_idx

        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.warning(self, "Error", f"Error finding brightest frame: {str(e)}")
            return

        progress.close()

        # Navigate to the brightest frame and capture masks
        self.frame_slider.setValue(brightest_frame_idx)

        # Capture masks from the brightest frame
        self._capture_fixed_masks()

        # Show result to user
        QtWidgets.QMessageBox.information(self, "Auto-Capture Complete",
                                        f"Captured masks from frame {brightest_frame_idx} "
                                        f"(brightness: {max_brightness:.1f} L*)")

    def _on_kernel_size_changed(self, value: int):
        """Handle morphological kernel size slider change."""
        self.morphological_kernel_size = max(1, value if value % 2 == 1 else value - 1)  # Ensure odd value
        self.kernel_size_label.setText(f"{self.morphological_kernel_size}×{self.morphological_kernel_size}")
        # Update display and invalidate cached masks since filtering changed
        self._invalidate_fixed_masks("noise parameters changed")
        if self.frame is not None:
            self._update_current_brightness_display()
            self.show_frame()

    def _on_bg_percentile_changed(self, value: int):
        """Handle background percentile slider change."""
        self.background_percentile = float(value)
        self.bg_percentile_label.setText(f"{self.background_percentile:.0f}%")
        # Update display since background calculation changed
        if self.frame is not None:
            self._update_current_brightness_display()
            self._update_threshold_display()
            self.show_frame()

    def _on_noise_floor_changed(self, value: int):
        """Handle noise floor threshold slider change."""
        self.noise_floor_threshold = value / 2.0  # Convert back from 0.5 precision
        self.noise_floor_label.setText(f"{self.noise_floor_threshold:.1f}")
        # Update display since noise filtering changed
        self._invalidate_fixed_masks("noise parameters changed")
        if self.frame is not None:
            self._update_current_brightness_display()
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

        self.image_label.unsetCursor()  # Reset cursor after action

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
            # Moving ROI invalidates any captured masks (shape/position may change)
            self._invalidate_fixed_masks("ROI moved")
            self.show_frame() # Redraw in final state

        elif self.resizing:
            # Finalize resizing
            self.resizing = False
            self.resize_corner = None
            self.start_point = None
            self.end_point = None
            # Optional: Recalculate brightness display for the final size
            self._update_current_brightness_display()
            # Resizing ROI invalidates any captured masks (shape changed)
            self._invalidate_fixed_masks("ROI resized")
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
            for idx, (pt1, pt2) in enumerate(self.rects):
                rect_x1, rect_y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
                rect_x2, rect_y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
                if rect_x1 <= frame_x <= rect_x2 and rect_y1 <= frame_y <= rect_y2:
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
        Analyzes video audio to detect completion beeps and calculates frame ranges 
        using the expected run duration.
        """
        if not self.video_path:
            self.results_label.setText("Load a video first.")
            return
        
        if not self.audio_analyzer.available:
            QtWidgets.QMessageBox.warning(self, "Audio Detection", 
                "Audio analysis not available. Please install librosa:\npip install librosa soundfile")
            return
        
        # Get expected run duration
        expected_duration = self.run_duration_spin.value()
        if expected_duration <= 0.0:
            QtWidgets.QMessageBox.warning(self, "Audio Detection", 
                "Please set an expected run duration (> 0 seconds) to calculate start frames from detected completion beeps.")
            return
        
        # Show progress dialog
        progress = QtWidgets.QProgressDialog("Analyzing audio for completion beeps...", "Cancel", 0, 0, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Audio Detection")
        progress.setRange(0, 0)  # Indeterminate progress
        progress.show()
        QtWidgets.QApplication.processEvents()
        
        try:
            # Find completion beeps in the audio
            completion_beeps = self.audio_analyzer.find_completion_beeps(self.video_path, expected_duration)
            
            progress.close()
            
            if not completion_beeps:
                QtWidgets.QMessageBox.information(self, "Audio Detection", 
                    "No completion beeps detected in the audio. Try adjusting the audio detection parameters or check if the video has audio.")
                self.results_label.setText("Audio Detection: No completion beeps found.")
                return
            
            # Show detected beeps to user for selection
            if len(completion_beeps) == 1:
                # Only one beep found, use it automatically
                selected_beep_time, selected_end_frame = completion_beeps[0]
            else:
                # Multiple beeps found, let user choose
                beep_options = []
                for i, (beep_time, frame_num) in enumerate(completion_beeps):
                    beep_options.append(f"Beep {i+1}: {beep_time:.1f}s (Frame {frame_num+1})")
                
                selected_option, ok = QtWidgets.QInputDialog.getItem(
                    self, "Audio Detection", 
                    f"Found {len(completion_beeps)} completion beeps. Select which one to use:",
                    beep_options, 0, False)
                
                if not ok:
                    self.results_label.setText("Audio Detection: User cancelled selection.")
                    return
                
                # Get the selected beep
                selected_index = beep_options.index(selected_option)
                selected_beep_time, selected_end_frame = completion_beeps[selected_index]
            
            # Calculate start frame using run duration and video FPS
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps <= 0:
                QtWidgets.QMessageBox.critical(self, "Error", "Could not determine video frame rate.")
                return
            
            # Calculate start frame (work backwards from completion beep)
            start_time = selected_beep_time - expected_duration
            calculated_start_frame = max(0, int(start_time * fps))
            
            # Ensure we don't go beyond video bounds
            calculated_end_frame = min(selected_end_frame, total_frames - 1)
            calculated_start_frame = min(calculated_start_frame, calculated_end_frame)
            
            # Update UI with detected range
            self.start_frame = calculated_start_frame
            self.end_frame = calculated_end_frame
            
            # Update slider and spinbox to the new start frame, then seek
            self.frame_slider.blockSignals(True)
            self.frame_spinbox.blockSignals(True)
            self.frame_slider.setValue(self.start_frame)
            self.frame_spinbox.setValue(self.start_frame)
            self.frame_slider.blockSignals(False)
            self.frame_spinbox.blockSignals(False)
            
            self._seek_to_frame(self.start_frame)
            self.update_frame_label()
            
            # Calculate actual duration for verification
            actual_duration = (self.end_frame - self.start_frame + 1) / fps
            
            self.results_label.setText(
                f"✅ Audio-detected range: Frame {self.start_frame + 1} to {self.end_frame + 1} "
                f"(Duration: {actual_duration:.1f}s, Expected: {expected_duration:.1f}s)")
            
            # Play success sound if duration is close to expected
            duration_difference = abs(actual_duration - expected_duration)
            if duration_difference <= expected_duration * 0.1:  # Within 10%
                self.audio_manager.play_run_detected()
                
        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.critical(self, "Audio Detection Error", 
                f"An error occurred during audio analysis:\n{str(e)}")
            self.results_label.setText(f"Audio Detection failed: {str(e)}")
            logging.error(f"Audio detection error: {e}")


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
        self.stop_playback()  # Stop playback during analysis
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
        self.results_label.setText("⚙️ Initializing analysis...")
        self.brightness_display_label.setText("📊 Preparing...")

        # Play analysis start sound
        self.audio_manager.play_analysis_start()
        self.statusBar().showMessage("🔍 Starting brightness analysis...")
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

            # --- Enhanced Progress Dialog ---
            progress = QtWidgets.QProgressDialog(f"🔍 Analyzing {num_frames_to_analyze} video frames...\nROIs: {len(non_background_rois)} | Frames: {start+1}-{end+1}",
                                                "Cancel", 0, num_frames_to_analyze, self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setWindowTitle("📊 Brightness Analysis")
            progress.setMinimumWidth(400)
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

                # Pre-compute L* channel once per frame for reuse
                l_star_frame = self._compute_l_star_frame(frame)

                # Calculate background brightness for this frame
                background_brightness = self._compute_background_brightness(frame, frame_l_star=l_star_frame)
                background_values_per_frame.append(background_brightness if background_brightness is not None else 0.0)
                
                fh, fw = frame.shape[:2]
                for data_idx, roi_idx in enumerate(non_background_rois):
                    pt1, pt2 = self.rects[roi_idx]
                    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
                    x2, y2 = min(fw - 1, pt2[0]), min(fh - 1, pt2[1])

                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2]
                        roi_l_star = l_star_frame[y1:y2, x1:x2]
                        roi_mask = None
                        if self.use_fixed_mask and roi_idx < len(self.fixed_roi_masks):
                            roi_mask = self.fixed_roi_masks[roi_idx]
                            # Ensure shape matches current ROI, else ignore
                            if isinstance(roi_mask, np.ndarray) and roi_mask.shape[:2] != roi.shape[:2]:
                                roi_mask = None
                        l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median = self._compute_brightness_stats(
                            roi,
                            background_brightness,
                            roi_mask,
                            roi_l_star=roi_l_star,
                        )
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
                        progress.setLabelText(f"🔍 Analyzing frame {frames_processed}/{num_frames_to_analyze}\n"
                                            f"⚡ Speed: {fps:.1f} fps | ⏱️ ETA: {eta_seconds:.0f}s | 📊 ROIs: {len(non_background_rois)}")
                    
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
            
            # Play analysis completion sound
            self.audio_manager.play_analysis_complete()
            
            # Check run duration if specified
            expected_duration = self.run_duration_spin.value()
            if expected_duration > 0.0 and self.start_frame is not None and self.end_frame is not None:
                confidence = self._validate_run_duration(self.start_frame, self.end_frame, expected_duration)
                if confidence >= 0.8:  # High confidence threshold
                    self.audio_manager.play_run_detected()

    def _save_analysis_results(self, brightness_mean_data, brightness_median_data, blue_mean_data, blue_median_data, save_dir, frames_processed, non_background_rois, background_values_per_frame):
        """Save analysis results and generate plots."""
        self.out_paths = []
        if self.video_path is None:
            return
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
                png_path, interactive_path = self._generate_enhanced_plot(
                    df,
                    base_filename,
                    save_dir,
                    actual_roi_idx,
                    analysis_name,
                    base_video_name,
                    background_values_per_frame
                )
                if png_path:
                    summary_lines.append(f" - Saved Plot: {os.path.basename(png_path)}")
                if interactive_path:
                    summary_lines.append(f" - Saved Interactive Plot: {os.path.basename(interactive_path)}")

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
        """Generate enhanced plots and an interactive visualization for the ROI."""
        png_path: Optional[str] = None
        interactive_path: Optional[str] = None
        try:
            frames = df['frame']
            brightness_mean = df['brightness_mean']
            brightness_median = df['brightness_median']
            blue_mean = df['blue_mean']
            blue_median = df['blue_median']

            if brightness_mean.empty:
                return png_path, interactive_path

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

            frame_list = frames.tolist()
            brightness_mean_values = brightness_mean.tolist()
            brightness_median_values = brightness_median.tolist()
            blue_mean_values = blue_mean.tolist()
            blue_median_values = blue_median.tolist()

            background_array: Optional[np.ndarray] = None
            if background_values_per_frame and len(background_values_per_frame) == len(frames):
                candidate_background = np.array(background_values_per_frame)
                valid_background_mask = candidate_background > 0
                if np.any(valid_background_mask):
                    background_array = candidate_background

            # Create enhanced plot with dual subplots
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Main brightness plot
            ax1.plot(frames, brightness_mean, label='Mean Brightness', color='#5a9bd5', linewidth=2, alpha=0.8)
            ax1.plot(frames, brightness_median, label='Median Brightness', color='#70ad47', linewidth=2, alpha=0.8)
            
            # Add background line if background values are available
            if background_array is not None:
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
            ax1.set_ylim(y_min, y_max + 0.15 * y_range)

            # Add statistics text box
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
            png_path = plot_save_path
            self.out_paths.append(plot_save_path)
            
            # Automatically open the generated PNG file
            try:
                import subprocess
                subprocess.run(['open', plot_save_path], check=True)
            except Exception as e:
                logging.warning(f"Could not automatically open plot file {plot_save_path}: {e}")

            if PLOTLY_AVAILABLE:
                try:
                    interactive_filename = f"{base_filename}_interactive.html"
                    interactive_save_path = os.path.join(save_dir, interactive_filename)

                    fig_interactive = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("L* Brightness", "Blue Channel")
                    )
                    selection_fill = _hex_to_rgba(COLOR_ACCENT, 0.18)

                    # L* mean confidence band
                    upper_mean_band = (brightness_mean + std_of_means).tolist()
                    lower_mean_band = (brightness_mean - std_of_means).tolist()
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=upper_mean_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=lower_mean_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=True,
                            name=f"Mean ±1σ ({std_of_means:.1f})",
                            fill='tonexty',
                            fillcolor='rgba(90,155,213,0.25)',
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )

                    # Median confidence band
                    upper_median_band = (brightness_median + std_of_medians).tolist()
                    lower_median_band = (brightness_median - std_of_medians).tolist()
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=upper_median_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=lower_median_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=True,
                            name=f"Median ±1σ ({std_of_medians:.1f})",
                            fill='tonexty',
                            fillcolor='rgba(112,173,71,0.25)',
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )

                    # Brightness lines
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=brightness_mean_values,
                            mode='lines',
                            name='Mean Brightness',
                            line=dict(color='#5a9bd5', width=2),
                            hovertemplate="Frame %{x}<br>Mean L*: %{y:.2f}<extra></extra>"
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=brightness_median_values,
                            mode='lines',
                            name='Median Brightness',
                            line=dict(color='#70ad47', width=2),
                            hovertemplate="Frame %{x}<br>Median L*: %{y:.2f}<extra></extra>"
                        ),
                        row=1,
                        col=1
                    )

                    # Background level
                    if background_array is not None:
                        fig_interactive.add_trace(
                            go.Scatter(
                                x=frame_list,
                                y=background_array.tolist(),
                                mode='lines',
                                name='Background Level',
                                line=dict(color='#808080', width=1.5, dash='dot'),
                                hovertemplate="Frame %{x}<br>Background L*: %{y:.2f}<extra></extra>"
                            ),
                            row=1,
                            col=1
                        )

                    # Peak annotations
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=[frame_peak_mean],
                            y=[val_peak_mean],
                            mode='markers',
                            name=f'Peak Mean ({val_peak_mean:.1f})',
                            marker=dict(color='#ff0000', size=10, symbol='triangle-up'),
                            hovertemplate="Frame %{x}<br>Peak Mean L*: %{y:.2f}<extra></extra>"
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=[frame_peak_median],
                            y=[val_peak_median],
                            mode='markers',
                            name=f'Peak Median ({val_peak_median:.1f})',
                            marker=dict(color='#ed7d31', size=10, symbol='triangle-down'),
                            hovertemplate="Frame %{x}<br>Peak Median L*: %{y:.2f}<extra></extra>"
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=[frame_peak_mean],
                            y=[val_peak_mean],
                            mode='markers',
                            name='Selected Range Peak (L*)',
                            marker=dict(
                                color=COLOR_WARNING,
                                size=14,
                                symbol='star',
                                line=dict(color='#92400e', width=1.2)
                            ),
                            hovertemplate="Frame %{x}<br>Selected L* Peak: %{y:.2f}<extra></extra>"
                        ),
                        row=1,
                        col=1
                    )

                    # Horizontal averages
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=[mean_of_means] * len(frame_list),
                            mode='lines',
                            name=f'Avg Mean ({mean_of_means:.1f})',
                            line=dict(color='#5a9bd5', dash='dash'),
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=[mean_of_medians] * len(frame_list),
                            mode='lines',
                            name=f'Avg Median ({mean_of_medians:.1f})',
                            line=dict(color='#70ad47', dash='dash'),
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )

                    # Blue channel confidence bands
                    upper_blue_mean_band = (blue_mean + std_of_blue_means).tolist()
                    lower_blue_mean_band = (blue_mean - std_of_blue_means).tolist()
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=upper_blue_mean_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=lower_blue_mean_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=True,
                            name=f'Blue Mean ±1σ ({std_of_blue_means:.1f})',
                            fill='tonexty',
                            fillcolor='rgba(0,102,204,0.25)',
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )

                    upper_blue_median_band = (blue_median + std_of_blue_medians).tolist()
                    lower_blue_median_band = (blue_median - std_of_blue_medians).tolist()
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=upper_blue_median_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=lower_blue_median_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=True,
                            name=f'Blue Median ±1σ ({std_of_blue_medians:.1f})',
                            fill='tonexty',
                            fillcolor='rgba(51,153,255,0.25)',
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )

                    # Blue channel lines
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=blue_mean_values,
                            mode='lines',
                            name='Blue Mean',
                            line=dict(color='#0066cc', width=2),
                            hovertemplate="Frame %{x}<br>Blue Mean: %{y:.2f}<extra></extra>"
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=blue_median_values,
                            mode='lines',
                            name='Blue Median',
                            line=dict(color='#3399ff', width=2),
                            hovertemplate="Frame %{x}<br>Blue Median: %{y:.2f}<extra></extra>"
                        ),
                        row=2,
                        col=1
                    )

                    # Blue channel peaks
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=[frame_peak_blue_mean],
                            y=[val_peak_blue_mean],
                            mode='markers',
                            name=f'Peak Blue Mean ({val_peak_blue_mean:.1f})',
                            marker=dict(color='#ff0000', size=10, symbol='triangle-up'),
                            hovertemplate="Frame %{x}<br>Peak Blue Mean: %{y:.2f}<extra></extra>"
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=[frame_peak_blue_median],
                            y=[val_peak_blue_median],
                            mode='markers',
                            name=f'Peak Blue Median ({val_peak_blue_median:.1f})',
                            marker=dict(color='#ed7d31', size=10, symbol='triangle-down'),
                            hovertemplate="Frame %{x}<br>Peak Blue Median: %{y:.2f}<extra></extra>"
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=[frame_peak_blue_mean],
                            y=[val_peak_blue_mean],
                            mode='markers',
                            name='Selected Range Peak (Blue)',
                            marker=dict(
                                color=COLOR_INFO,
                                size=14,
                                symbol='star',
                                line=dict(color='#0e7490', width=1.2)
                            ),
                            hovertemplate="Frame %{x}<br>Selected Blue Peak: %{y:.2f}<extra></extra>"
                        ),
                        row=2,
                        col=1
                    )

                    # Blue channel averages
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=[mean_of_blue_means] * len(frame_list),
                            mode='lines',
                            name=f'Avg Blue Mean ({mean_of_blue_means:.1f})',
                            line=dict(color='#0066cc', dash='dash'),
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        go.Scatter(
                            x=frame_list,
                            y=[mean_of_blue_medians] * len(frame_list),
                            mode='lines',
                            name=f'Avg Blue Median ({mean_of_blue_medians:.1f})',
                            line=dict(color='#3399ff', dash='dash'),
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )

                    # Layout
                    fig_interactive.update_layout(
                        title=f"{analysis_name} - {base_video_name} - ROI {r_idx+1}",
                        height=820,
                        dragmode='select',
                        selectdirection='h',
                        hovermode='x unified',
                        template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
                        margin=dict(t=80, b=60, l=60, r=30),
                        newselection=dict(
                            line=dict(color=COLOR_ACCENT, width=2)
                        )
                    )
                    fig_interactive.update_xaxes(title_text="Frame Number", row=2, col=1)
                    fig_interactive.update_yaxes(title_text="L* Brightness", row=1, col=1)
                    fig_interactive.update_yaxes(title_text="Blue Channel Value", row=2, col=1)

                    # Summary annotation
                    fig_interactive.add_annotation(
                        text=(
                            f"Mean: {mean_of_means:.2f} ± {std_of_means:.2f} | "
                            f"Median: {mean_of_medians:.2f} ± {std_of_medians:.2f}<br>"
                            f"Blue Mean: {mean_of_blue_means:.1f} ± {std_of_blue_means:.1f} | "
                            f"Blue Median: {mean_of_blue_medians:.1f} ± {std_of_blue_medians:.1f}"
                        ),
                        xref="paper",
                        yref="paper",
                        x=0.0,
                        y=1.12,
                        showarrow=False,
                        align='left',
                        font=dict(size=12),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#cccccc',
                        borderwidth=1,
                        borderpad=6
                    )

                    div_id = f"roi-interactive-{r_idx+1}"
                    selection_script = self._build_selection_post_script(
                        div_id=div_id,
                        frames=frame_list,
                        brightness_values=brightness_mean_values,
                        blue_values=blue_mean_values,
                        accent_color=COLOR_ACCENT,
                        selection_fill=selection_fill,
                    )
                    fig_interactive.write_html(
                        interactive_save_path,
                        include_plotlyjs='cdn',
                        div_id=div_id,
                        post_script=selection_script,
                    )
                    interactive_path = interactive_save_path
                    self.out_paths.append(interactive_save_path)

                    try:
                        import subprocess
                        subprocess.run(['open', interactive_save_path], check=True)
                    except Exception as e:
                        logging.warning(f"Could not automatically open interactive plot {interactive_save_path}: {e}")
                except Exception as plotly_error:
                    logging.warning(f"Failed to generate interactive plot for ROI {r_idx+1}: {plotly_error}")
            else:
                logging.info("Plotly not available - skipping interactive plot generation.")

            return png_path, interactive_path

        except Exception as e:
            logging.error(f"Failed to generate plot for ROI {r_idx+1}: {e}")
            raise

    def _build_selection_post_script(
        self,
        div_id: str,
        frames: List[float],
        brightness_values: List[float],
        blue_values: List[float],
        accent_color: str,
        selection_fill: str,
    ) -> str:
        """Generate a JS snippet that highlights peaks inside the active brush selection."""
        frames_json = json.dumps(frames)
        brightness_json = json.dumps(brightness_values)
        blue_json = json.dumps(blue_values)
        font_family = DEFAULT_FONT_FAMILY.replace("\\", "\\\\").replace("'", "\\'")
        accent_color = accent_color.replace("\\", "\\\\").replace("'", "\\'")
        selection_fill = selection_fill.replace("\\", "\\\\").replace("'", "\\'")

        script = Template(
            "(function() {\n"
            "  const divId = '$div_id';\n"
            "  const frames = $frames_json;\n"
            "  const brightnessValues = $brightness_json;\n"
            "  const blueValues = $blue_json;\n"
            "  const accentColor = '$accent_color';\n"
            "  const selectionFill = '$selection_fill';\n"
            "  const fontFamily = '$font_family';\n"
            "\n"
            "  const createSelectionEnhancements = () => {\n"
            "    const gd = document.getElementById(divId);\n"
            "    if (!gd || gd.__selectionInitialized) {\n"
            "      return;\n"
            "    }\n"
            "\n"
            "    const initialiseWhenReady = () => {\n"
            "      if (typeof Plotly === 'undefined') {\n"
            "        return false;\n"
            "      }\n"
            "      if (!gd.data || !gd.data.length) {\n"
            "        return false;\n"
            "      }\n"
            "\n"
            "      const selectedMeanIndex = gd.data.findIndex(trace => trace.name === 'Selected Range Peak (L*)');\n"
            "      const selectedBlueIndex = gd.data.findIndex(trace => trace.name === 'Selected Range Peak (Blue)');\n"
            "      if (selectedMeanIndex === -1 || selectedBlueIndex === -1) {\n"
            "        return false;\n"
            "      }\n"
            "\n"
            "      gd.__selectionInitialized = true;\n"
            "\n"
            "      const infoPanel = document.createElement('div');\n"
            "      infoPanel.className = 'selection-info';\n"
            "      infoPanel.style.marginTop = '12px';\n"
            "      infoPanel.style.fontFamily = fontFamily;\n"
            "      infoPanel.style.fontSize = '14px';\n"
            "      infoPanel.style.color = '#1f2937';\n"
            "      infoPanel.style.background = 'rgba(255,255,255,0.92)';\n"
            "      infoPanel.style.border = '1px solid #e5e7eb';\n"
            "      infoPanel.style.borderRadius = '8px';\n"
            "      infoPanel.style.padding = '8px 12px';\n"
            "      infoPanel.style.boxShadow = '0 1px 3px rgba(15,23,42,0.12)';\n"
            "      infoPanel.style.display = 'inline-block';\n"
            "      if (gd.parentNode) {\n"
            "        gd.parentNode.insertBefore(infoPanel, gd.nextSibling);\n"
            "      }\n"
            "\n"
            "      const domainStart = frames[0];\n"
            "      const domainEnd = frames[frames.length - 1];\n"
            "      const nearlyEqual = (a, b) => Math.abs(a - b) <= Math.max(1, Math.abs(domainEnd - domainStart)) * 1e-6;\n"
            "\n"
            "      const findPeakIndex = (range, values) => {\n"
            "        let candidate = -1;\n"
            "        let maxValue = -Infinity;\n"
            "        for (let i = 0; i < frames.length; i += 1) {\n"
            "          const frame = frames[i];\n"
            "          if (frame >= range[0] && frame <= range[1]) {\n"
            "            const value = values[i];\n"
            "            if (value > maxValue) {\n"
            "              maxValue = value;\n"
            "              candidate = i;\n"
            "            }\n"
            "          }\n"
            "        }\n"
            "        if (candidate !== -1) {\n"
            "          return candidate;\n"
            "        }\n"
            "        let bestDistance = Infinity;\n"
            "        for (let i = 0; i < frames.length; i += 1) {\n"
            "          const frame = frames[i];\n"
            "          const distance = frame < range[0] ? range[0] - frame : frame > range[1] ? frame - range[1] : 0;\n"
            "          if (distance < bestDistance) {\n"
            "            bestDistance = distance;\n"
            "            candidate = i;\n"
            "          }\n"
            "        }\n"
            "        return candidate;\n"
            "      };\n"
            "\n"
            "      const extractRange = (rangeLike) => {\n"
            "        if (Array.isArray(rangeLike) && rangeLike.length >= 2) {\n"
            "          return [Number(rangeLike[0]), Number(rangeLike[1])];\n"
            "        }\n"
            "        if (rangeLike && Array.isArray(rangeLike.x) && rangeLike.x.length >= 2) {\n"
            "          return [Number(rangeLike.x[0]), Number(rangeLike.x[1])];\n"
            "        }\n"
            "        return [domainStart, domainEnd];\n"
            "      };\n"
            "\n"
            "      const restyleMarker = (traceIndex, frameIndex, values) => {\n"
            "        if (traceIndex === -1 || frameIndex < 0 || frameIndex >= frames.length) {\n"
            "          return;\n"
            "        }\n"
            "        try {\n"
            "          Plotly.restyle(gd, {\n"
            "            x: [[frames[frameIndex]]],\n"
            "            y: [[values[frameIndex]]]\n"
            "          }, [traceIndex]);\n"
            "        } catch (err) {\n"
            "          /* ignore restyle issues */\n"
            "        }\n"
            "      };\n"
            "\n"
            "      const applySelectionShape = (start, end) => {\n"
            "        const coversAll = nearlyEqual(start, domainStart) && nearlyEqual(end, domainEnd);\n"
            "        const shapes = coversAll ? [] : [{\n"
            "          type: 'rect',\n"
            "          xref: 'x',\n"
            "          x0: start,\n"
            "          x1: end,\n"
            "          yref: 'paper',\n"
            "          y0: 0,\n"
            "          y1: 1,\n"
            "          fillcolor: selectionFill,\n"
            "          line: { color: accentColor, width: 1, dash: 'dot' },\n"
            "          layer: 'below'\n"
            "        }];\n"
            "        try {\n"
            "          Plotly.relayout(gd, { shapes });\n"
            "        } catch (err) {\n"
            "          /* ignore relayout issues */\n"
            "        }\n"
            "      };\n"
            "\n"
            "      const updateInfoPanel = (start, end, brightnessIndex, blueIndex) => {\n"
            "        const formatValue = (value, digits = 2) => {\n"
            "          const numeric = Number.parseFloat(value);\n"
            "          return Number.isFinite(numeric) ? numeric.toFixed(digits) : 'n/a';\n"
            "        };\n"
            "        const parts = [];\n"
            "        const framesLabel = 'Frames ' + Math.round(start) + '–' + Math.round(end);\n"
            "        parts.push('<div style=\"font-weight:600;color:' + accentColor + '\">' + framesLabel + '</div>');\n"
            "        if (brightnessIndex >= 0) {\n"
            "          const frame = frames[brightnessIndex];\n"
            "          const value = formatValue(brightnessValues[brightnessIndex]);\n"
            "          parts.push('<div><strong>L*</strong> frame ' + frame + ' (' + value + ')</div>');\n"
            "        } else {\n"
            "          parts.push('<div><strong>L*</strong> peak n/a</div>');\n"
            "        }\n"
            "        if (blueIndex >= 0) {\n"
            "          const frame = frames[blueIndex];\n"
            "          const value = formatValue(blueValues[blueIndex]);\n"
            "          parts.push('<div><strong>Blue</strong> frame ' + frame + ' (' + value + ')</div>');\n"
            "        } else {\n"
            "          parts.push('<div><strong>Blue</strong> peak n/a</div>');\n"
            "        }\n"
            "        infoPanel.innerHTML = parts.join('');\n"
            "      };\n"
            "\n"
            "      const updateSelection = (rangeLike) => {\n"
            "        const [rawStart, rawEnd] = extractRange(rangeLike);\n"
            "        const clamp = (value) => Math.min(Math.max(value, domainStart), domainEnd);\n"
            "        const start = clamp(Math.min(rawStart, rawEnd));\n"
            "        const end = clamp(Math.max(rawStart, rawEnd));\n"
            "        const brightnessIndex = findPeakIndex([start, end], brightnessValues);\n"
            "        const blueIndex = findPeakIndex([start, end], blueValues);\n"
            "        restyleMarker(selectedMeanIndex, brightnessIndex, brightnessValues);\n"
            "        restyleMarker(selectedBlueIndex, blueIndex, blueValues);\n"
            "        applySelectionShape(start, end);\n"
            "        updateInfoPanel(start, end, brightnessIndex, blueIndex);\n"
            "      };\n"
            "\n"
            "      const reset = () => {\n"
            "        updateSelection([domainStart, domainEnd]);\n"
            "      };\n"
            "\n"
            "      gd.on('plotly_selected', (eventData) => {\n"
            "        if (eventData && eventData.range && eventData.range.x) {\n"
            "          updateSelection(eventData.range);\n"
            "        }\n"
            "      });\n"
            "      gd.on('plotly_selecting', (eventData) => {\n"
            "        if (eventData && eventData.range && eventData.range.x) {\n"
            "          updateSelection(eventData.range);\n"
            "        }\n"
            "      });\n"
            "      gd.on('plotly_doubleclick', reset);\n"
            "      gd.on('plotly_deselect', reset);\n"
            "\n"
            "      reset();\n"
            "      return true;\n"
            "    };\n"
            "\n"
            "    if (!initialiseWhenReady()) {\n"
            "      const handler = () => {\n"
            "        if (initialiseWhenReady() && gd.removeListener) {\n"
            "          gd.removeListener('plotly_afterplot', handler);\n"
            "        }\n"
            "      };\n"
            "      if (gd.on) {\n"
            "        gd.on('plotly_afterplot', handler);\n"
            "      } else {\n"
            "        setTimeout(handler, 60);\n"
            "      }\n"
            "    }\n"
            "  };\n"
            "\n"
            "  if (document.readyState === 'loading') {\n"
            "    document.addEventListener('DOMContentLoaded', createSelectionEnhancements, { once: true });\n"
            "  } else {\n"
            "    createSelectionEnhancements();\n"
            "  }\n"
            "})();"
        )

        return script.substitute(
            div_id=div_id,
            frames_json=frames_json,
            brightness_json=brightness_json,
            blue_json=blue_json,
            accent_color=accent_color,
            selection_fill=selection_fill,
            font_family=font_family,
        )

    # --- Utility Methods ---

    def _validate_run_duration(self, start_frame: int, end_frame: int, expected_duration: float) -> float:
        """
        Calculate confidence score for detected run vs expected duration.
        
        Args:
            start_frame: Start frame of detected run
            end_frame: End frame of detected run
            expected_duration: Expected duration in seconds
            
        Returns:
            Confidence score from 0.0 to 1.0 (1.0 = perfect match)
        """
        if expected_duration <= 0.0 or not self.cap:
            return 1.0  # No validation if duration not set or video not loaded
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return 1.0  # No validation if fps unknown
        
        actual_duration = (end_frame - start_frame + 1) / fps
        duration_difference = abs(actual_duration - expected_duration)
        
        # Calculate confidence - perfect match = 1.0, large difference = 0.0
        # Allow 20% tolerance before confidence starts dropping
        tolerance = expected_duration * 0.2
        if duration_difference <= tolerance:
            confidence = 1.0
        else:
            # Linear decay from 1.0 to 0.0 over the next 80% of expected duration
            max_difference = expected_duration * 0.8
            confidence = max(0.0, 1.0 - (duration_difference - tolerance) / max_difference)
        
        return confidence

    def _compute_l_star_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert a BGR frame to its L* channel (0-100) once for reuse across ROIs.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        np.multiply(l_chan, 100.0 / 255.0, out=l_chan)
        return l_chan

    def _compute_brightness_stats(
        self,
        roi_bgr: np.ndarray,
        background_brightness: Optional[float] = None,
        roi_mask: Optional[np.ndarray] = None,
        roi_l_star: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Calculates brightness statistics for an ROI with optional background subtraction.

        Converts BGR to CIE LAB color space and uses the L* channel.
        Also extracts blue channel statistics for blue light analysis.

        Args:
            roi_bgr: The region of interest as a NumPy array (BGR format).
            background_brightness: Optional background L* value to subtract from all pixels.
            roi_mask: Optional boolean mask selecting pixels to analyze within ROI.
            roi_l_star: Optional precomputed L* slice aligned with roi_bgr to avoid redundant conversions.

        Returns:
            Tuple of (l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, 
                     b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median)
            L* values in 0-100 range, Blue values in 0-255 range
            or (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if the ROI is invalid or calculation fails.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            if roi_l_star is None:
                lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
                l_chan = lab[:, :, 0].astype(np.float32)
                l_star = l_chan * 100.0 / 255.0
            else:
                l_star = roi_l_star.astype(np.float32, copy=False)

            # If a fixed ROI mask is provided, use it directly
            if roi_mask is not None:
                mask_bool = roi_mask.astype(bool)
                if mask_bool.shape[:2] != roi_bgr.shape[:2] or not np.any(mask_bool):
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                blue_chan = roi_bgr[:, :, 0].astype(np.float32)
                l_pixels = l_star[mask_bool]
                b_pixels = blue_chan[mask_bool]
                l_raw_mean = float(np.mean(l_pixels))
                l_raw_median = float(np.median(l_pixels))
                b_raw_mean = float(np.mean(b_pixels))
                b_raw_median = float(np.median(b_pixels))
                if background_brightness is not None:
                    l_bg = l_pixels - background_brightness
                    l_bg_sub_mean = float(np.mean(l_bg))
                    l_bg_sub_median = float(np.median(l_bg))
                    b_bg_sub_mean = b_raw_mean
                    b_bg_sub_median = b_raw_median
                else:
                    l_bg_sub_mean = l_raw_mean
                    l_bg_sub_median = l_raw_median
                    b_bg_sub_mean = b_raw_mean
                    b_bg_sub_median = b_raw_median
                return l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median

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
                                                     (self.morphological_kernel_size, self.morphological_kernel_size))
                    
                    # Convert boolean mask to uint8 for morphological operations
                    mask_uint8 = above_background_mask.astype(np.uint8) * 255
                    
                    # Apply opening (erosion followed by dilation) to remove small noise
                    cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
                    
                    # Convert back to boolean mask
                    above_background_mask = cleaned_mask > 0
                
                if np.any(above_background_mask):
                    # Apply additional noise floor filtering if enabled
                    if self.noise_floor_threshold > 0:
                        noise_floor_mask = l_star > self.noise_floor_threshold
                        combined_mask = above_background_mask & noise_floor_mask
                    else:
                        combined_mask = above_background_mask

                    if np.any(combined_mask):
                        # Only analyze pixels above both background and noise floor thresholds
                        filtered_l_pixels = l_star[combined_mask]
                        filtered_b_pixels = blue_chan[combined_mask]
                    else:
                        # No pixels pass both filters
                        l_bg_sub_mean = 0.0
                        l_bg_sub_median = 0.0
                        b_bg_sub_mean = 0.0
                        b_bg_sub_median = 0.0
                        return l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median
                    
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

    def _compute_background_brightness(
        self,
        frame: np.ndarray,
        frame_l_star: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """
        Calculate background ROI brightness for current frame.
        
        Args:
            frame: Current video frame in BGR format
            frame_l_star: Optional precomputed L* channel for the frame to avoid recomputation.
            
        Returns:
            90th percentile L* brightness of background ROI, or None if no background ROI defined
        """
        if self.background_roi_idx is None or frame is None:
            return None
            
        if not (0 <= self.background_roi_idx < len(self.rects)):
            return None
            
        try:
            pt1, pt2 = self.rects[self.background_roi_idx]
            if frame_l_star is not None:
                roi_l_star = frame_l_star[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            else:
                roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                if roi.size == 0:
                    return None
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                l_chan = lab[:, :, 0].astype(np.float32)
                roi_l_star = l_chan * 100.0 / 255.0

            if roi_l_star.size == 0:
                return None

            return float(np.percentile(roi_l_star, self.background_percentile))

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
