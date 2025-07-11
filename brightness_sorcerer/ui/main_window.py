"""Main window for Brightness Sorcerer."""

import sys
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Optional
import logging

from ..core.application_controller import ApplicationController
from ..utils.file_utils import get_video_filter_string, validate_video_file
from .video_widget import VideoWidget
from .controls_panel import ControlsPanel
from .styles.theme import apply_dark_theme


class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize application controller
        self.app_controller = ApplicationController()
        
        # UI components
        self.video_widget: Optional[VideoWidget] = None
        self.controls_panel: Optional[ControlsPanel] = None
        
        self._setup_ui()
        self._setup_menus()
        self._setup_shortcuts()
        self._connect_signals()
        self._load_settings()
        
        # Register UI callbacks with controller
        self.app_controller.register_ui_callback('video_load_failed', self._on_video_load_failed)
        self.app_controller.register_ui_callback('session_saved', self._on_session_saved)
        self.app_controller.register_ui_callback('session_loaded', self._on_session_loaded)
        
        logging.info("Main window initialized")
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        # Set window properties
        self.setWindowTitle("Brightness Sorcerer v3.0")
        self.setMinimumSize(1000, 700)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Create central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        
        # Video widget
        self.video_widget = VideoWidget(self.app_controller)
        main_layout.addWidget(self.video_widget, 1)  # Give it more space
        
        # Controls panel
        self.controls_panel = ControlsPanel(self.app_controller)
        main_layout.addWidget(self.controls_panel, 0)  # Fixed width
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Load a video to begin")
        
        # Progress widget in status bar
        self.progress_widget = QtWidgets.QWidget()
        progress_layout = QtWidgets.QHBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setVisible(False)
        progress_layout.addWidget(self.progress_label)
        
        self.status_bar.addPermanentWidget(self.progress_widget)
    
    def _setup_menus(self):
        """Setup application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Open video
        open_action = QtWidgets.QAction('&Open Video...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open a video file')
        open_action.triggered.connect(self._open_video_dialog)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Recent files
        self.recent_files_menu = file_menu.addMenu('Recent Files')
        self._update_recent_files_menu()
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QtWidgets.QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit the application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('&Analysis')
        
        # Run analysis
        analyze_action = QtWidgets.QAction('&Run Analysis', self)
        analyze_action.setShortcut('F5')
        analyze_action.setStatusTip('Run brightness analysis')
        analyze_action.triggered.connect(self._run_analysis)
        analysis_menu.addAction(analyze_action)
        
        # Auto-detect range
        auto_detect_action = QtWidgets.QAction('&Auto-Detect Range', self)
        auto_detect_action.setShortcut('Ctrl+D')
        auto_detect_action.setStatusTip('Automatically detect frame range')
        auto_detect_action.triggered.connect(self._auto_detect_range)
        analysis_menu.addAction(auto_detect_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Theme submenu
        theme_menu = view_menu.addMenu('Theme')
        
        dark_theme_action = QtWidgets.QAction('Dark', self)
        dark_theme_action.triggered.connect(lambda: self._set_theme('dark'))
        theme_menu.addAction(dark_theme_action)
        
        light_theme_action = QtWidgets.QAction('Light', self)
        light_theme_action.triggered.connect(lambda: self._set_theme('light'))
        theme_menu.addAction(light_theme_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        shortcuts_action = QtWidgets.QAction('&Keyboard Shortcuts', self)
        shortcuts_action.triggered.connect(self._show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)
        
        about_action = QtWidgets.QAction('&About', self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Video navigation shortcuts are handled by the video widget
        
        # ROI shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self, self._add_roi)
        
        # Analysis shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence('F5'), self, self._run_analysis)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+D'), self, self._auto_detect_range)
    
    def _connect_signals(self):
        """Connect widget signals."""
        # Video widget signals
        self.video_widget.frame_changed.connect(self._on_frame_changed)
        self.video_widget.roi_interaction.connect(self._on_roi_interaction)
        
        # Controls panel signals
        self.controls_panel.analysis_widget.analyze_requested.connect(self._start_analysis)
        self.controls_panel.analysis_widget.auto_detect_requested.connect(self._auto_detect_range)
        
        # Application controller will handle progress callbacks
    
    def _load_settings(self):
        """Load application settings."""
        # Window geometry
        geometry = self.app_controller.settings_manager.get_window_geometry()
        if geometry:
            self.setGeometry(geometry['x'], geometry['y'], geometry['width'], geometry['height'])
        
        if self.app_controller.settings_manager.is_window_maximized():
            self.showMaximized()
        
        # Theme
        theme = self.app_controller.settings_manager.get_setting('theme', 'dark')
        self._set_theme(theme)
    
    def _save_settings(self):
        """Save application settings."""
        # Window geometry
        if not self.isMaximized():
            rect = self.geometry()
            self.app_controller.settings_manager.set_window_geometry(rect.x(), rect.y(), rect.width(), rect.height())
        
        self.app_controller.settings_manager.set_window_maximized(self.isMaximized())
        self.app_controller.settings_manager.save_settings()
    
    def _set_theme(self, theme_name: str):
        """Set application theme."""
        if theme_name == 'dark':
            apply_dark_theme(QtWidgets.QApplication.instance())
        # Light theme would be implemented similarly
        
        self.app_controller.settings_manager.set_setting('theme', theme_name)
    
    def _open_video_dialog(self):
        """Open video file dialog."""
        file_filter = get_video_filter_string()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video File", "", file_filter
        )
        
        if file_path:
            self._load_video(file_path)
    
    def _load_video(self, file_path: str):
        """Load a video file."""
        if not validate_video_file(file_path):
            QtWidgets.QMessageBox.warning(
                self, "Invalid Video File",
                f"The file '{os.path.basename(file_path)}' is not a supported video format."
            )
            return
        
        if self.video_widget.load_video(file_path):
            self.app_controller.settings_manager.add_recent_file(file_path)
            self._update_recent_files_menu()
            
            # Update status
            video_info = self.app_controller.video_processor.get_video_info()
            filename = os.path.basename(file_path)
            self.status_bar.showMessage(
                f"Loaded: {filename} ({video_info['total_frames']} frames, {video_info['fps']:.1f} fps)"
            )
            
            logging.info(f"Video loaded: {file_path}")
        else:
            QtWidgets.QMessageBox.critical(
                self, "Video Load Error",
                f"Failed to load video file: {os.path.basename(file_path)}"
            )
    
    def _update_recent_files_menu(self):
        """Update recent files menu."""
        self.recent_files_menu.clear()
        
        recent_files = self.settings_manager.get_recent_files()
        if not recent_files:
            no_recent_action = QtWidgets.QAction('No recent files', self)
            no_recent_action.setEnabled(False)
            self.recent_files_menu.addAction(no_recent_action)
            return
        
        for file_path in recent_files:
            action = QtWidgets.QAction(os.path.basename(file_path), self)
            action.setStatusTip(file_path)
            action.triggered.connect(lambda checked, path=file_path: self._load_video(path))
            self.recent_files_menu.addAction(action)
    
    def _add_roi(self):
        """Add a new ROI."""
        if not self.video_processor.is_loaded():
            return
        
        # Add ROI at center of current frame
        frame_size = self.video_processor.frame_size
        center_x = frame_size[0] // 2 - 100
        center_y = frame_size[1] // 2 - 75
        
        self.roi_manager.add_roi(center_x, center_y, 200, 150)
        self.video_widget.update_display()
    
    def _run_analysis(self):
        """Run brightness analysis."""
        if not self.video_processor.is_loaded():
            QtWidgets.QMessageBox.information(
                self, "No Video", "Please load a video file first."
            )
            return
        
        if not self.roi_manager.rois:
            QtWidgets.QMessageBox.information(
                self, "No ROIs", "Please add at least one ROI before running analysis."
            )
            return
        
        # This would trigger the controls panel to show its analysis dialog
        # For now, just show a placeholder
        QtWidgets.QMessageBox.information(
            self, "Analysis", "Analysis feature will be completed in the next phase."
        )
    
    def _auto_detect_range(self):
        """Auto-detect frame range."""
        if not self.video_processor.is_loaded():
            return
        
        if not self.roi_manager.rois:
            QtWidgets.QMessageBox.information(
                self, "No ROIs", "Please add at least one ROI before auto-detection."
            )
            return
        
        # Run auto-detection
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            background_roi = self.roi_manager.get_background_roi()
            result = self.brightness_analyzer.auto_detect_frame_range(
                self.roi_manager.rois, background_roi
            )
            
            if result:
                start_frame, end_frame = result
                self.controls_panel.analysis_widget.set_auto_detected_range(start_frame, end_frame)
                self.status_bar.showMessage(
                    f"Auto-detected range: frames {start_frame} - {end_frame}", 3000
                )
            else:
                QtWidgets.QMessageBox.information(
                    self, "Auto-Detection", 
                    "Could not detect a suitable frame range. Please set manually."
                )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    
    def _start_analysis(self, start_frame: int, end_frame: int, output_dir: str):
        """Start brightness analysis."""
        if self._analysis_in_progress:
            return
        
        self._analysis_in_progress = True
        
        # Get background ROI
        background_roi = self.roi_manager.get_background_roi()
        
        # Start analysis in a separate thread (would be implemented with QThread)
        # For now, just show completion message
        QtWidgets.QMessageBox.information(
            self, "Analysis Started", 
            f"Analysis would run from frame {start_frame} to {end_frame}\n"
            f"Output directory: {output_dir}"
        )
        
        self._analysis_in_progress = False
    
    def _on_analysis_progress(self, current: int, total: int, message: str):
        """Handle analysis progress updates."""
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)
        
        # Update controls panel progress too
        self.controls_panel.analysis_widget.show_progress(current, total, message)
    
    def _on_frame_changed(self, frame_index: int):
        """Handle frame navigation."""
        # Update status bar with current frame info
        if self.video_processor.is_loaded():
            video_info = self.video_processor.get_video_info()
            fps = video_info['fps']
            current_time = frame_index / fps if fps > 0 else 0
            self.status_bar.showMessage(
                f"Frame {frame_index + 1} ({current_time:.2f}s)", 2000
            )
    
    def _on_roi_interaction(self, action: str, roi_index: int, extra: int):
        """Handle ROI interactions."""
        if action == "created":
            self.status_bar.showMessage(f"ROI {roi_index + 1} created", 2000)
        elif action == "deleted":
            self.status_bar.showMessage(f"ROI deleted", 2000)
    
    def _show_shortcuts_dialog(self):
        """Show keyboard shortcuts dialog."""
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        
        <h4>Video Navigation</h4>
        <b>← / Backspace</b> - Previous frame<br>
        <b>→ / Space</b> - Next frame<br>
        <b>Page Up</b> - Jump back 10 frames<br>
        <b>Page Down</b> - Jump forward 10 frames<br>
        <b>Home</b> - Go to first frame<br>
        <b>End</b> - Go to last frame<br>
        
        <h4>ROI Management</h4>
        <b>Ctrl+R</b> - Add new ROI<br>
        <b>Delete</b> - Delete selected ROI<br>
        <b>Escape</b> - Cancel current action<br>
        
        <h4>Analysis</h4>
        <b>F5</b> - Run analysis<br>
        <b>Ctrl+D</b> - Auto-detect frame range<br>
        
        <h4>File Operations</h4>
        <b>Ctrl+O</b> - Open video file<br>
        <b>Ctrl+Q</b> - Exit application<br>
        """
        
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setTextFormat(QtCore.Qt.RichText)
        dialog.setText(shortcuts_text)
        dialog.exec_()
    
    def _show_about_dialog(self):
        """Show about dialog."""
        about_text = """
        <h2>Brightness Sorcerer v3.0</h2>
        <p>Professional Video Brightness Analysis Tool</p>
        
        <p>A modular, maintainable application for analyzing brightness changes 
        in video regions of interest with automatic detection and comprehensive plotting.</p>
        
        <h3>Features</h3>
        <ul>
        <li>Interactive ROI management</li>
        <li>CIE LAB color space analysis</li>
        <li>Automatic frame range detection</li>
        <li>High-quality data export and plotting</li>
        <li>Professional modular architecture</li>
        </ul>
        
        <p><b>Built with:</b> Python, PyQt5, OpenCV, NumPy, Matplotlib</p>
        """
        
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("About Brightness Sorcerer")
        dialog.setTextFormat(QtCore.Qt.RichText)
        dialog.setText(about_text)
        dialog.exec_()
    
    # Drag and drop support
    def dragEnterEvent(self, event):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and validate_video_file(urls[0].toLocalFile()):
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """Handle drop events."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                file_path = urls[0].toLocalFile()
                self._load_video(file_path)
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()
    
    def closeEvent(self, event):
        """Handle window close event."""
        self._save_settings()
        
        # Cancel any ongoing analysis
        if self._analysis_in_progress:
            self.brightness_analyzer.cancel_analysis()
        
        # Clean up resources
        self.video_processor.release()
        
        event.accept()
        logging.info("Application closed")