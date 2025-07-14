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
        self.app_controller.register_ui_callback('analysis_progress', self._on_analysis_progress)
        self.app_controller.register_ui_callback('analysis_completed', self._on_analysis_completed)
        self.app_controller.register_ui_callback('analysis_failed', self._on_analysis_failed)
        self.app_controller.register_ui_callback('display_update_needed', self._on_display_update_needed)
        
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
        self.status_bar.showMessage("Ready - Load a video to begin (Ctrl+O) or drag & drop a video file")
        
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
        
        # Save session
        save_action = QtWidgets.QAction('&Save Session...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save current analysis session')
        save_action.triggered.connect(self._save_session_dialog)
        file_menu.addAction(save_action)
        
        # Load session
        load_action = QtWidgets.QAction('&Load Session...', self)
        load_action.setShortcut('Ctrl+L')
        load_action.setStatusTip('Load an analysis session')
        load_action.triggered.connect(self._load_session_dialog)
        file_menu.addAction(load_action)
        
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
        # Video navigation shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, lambda: self._step_video_frames(-1))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, lambda: self._step_video_frames(1))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_PageUp), self, lambda: self._step_video_frames(-10))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_PageDown), self, lambda: self._step_video_frames(10))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Home), self, self._go_to_first_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_End), self, self._go_to_last_frame)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, self._toggle_playback)
        
        # ROI shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self, self._add_roi)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self, self._delete_selected_roi)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, self._cancel_roi_interaction)
        
        # Analysis shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence('F5'), self, self._run_analysis)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+D'), self, self._auto_detect_range)
    
    def _connect_signals(self):
        """Connect widget signals."""
        # Video widget signals
        if self.video_widget:
            self.video_widget.frame_changed.connect(self._on_frame_changed)
            self.video_widget.roi_interaction.connect(self._on_roi_interaction)

        # Controls panel signals
        if self.controls_panel:
            self.controls_panel.analysis_widget.analyze_requested.connect(self._start_analysis)
            self.controls_panel.analysis_widget.auto_detect_requested.connect(self._auto_detect_range)
            self.controls_panel.roi_widget.roi_selected.connect(self.app_controller.select_roi)
            self.controls_panel.roi_widget.roi_deleted.connect(self.app_controller.delete_roi)
            self.controls_panel.roi_widget.background_roi_changed.connect(self.app_controller.set_background_roi)
            self.controls_panel.roi_widget.add_roi_requested.connect(self._add_roi)
        
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
        elif theme_name == 'light':
            apply_light_theme(QtWidgets.QApplication.instance())
        
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
        
        result = self.app_controller.load_video(file_path)
        if result.is_success():
            self.app_controller.settings_manager.add_recent_file(file_path)
            self._update_recent_files_menu()
            
            # Update video display and controls
            self.video_widget.update_display()
            
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
        
        recent_files = self.app_controller.settings_manager.get_recent_files()
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
        if not self.app_controller.video_processor.is_loaded():
            return
        
        # Add ROI at center of current frame
        frame_size = self.app_controller.video_processor.frame_size
        center_x = frame_size[0] // 2 - 100
        center_y = frame_size[1] // 2 - 75
        
        # Use the application controller's add_roi method which triggers UI notifications
        result = self.app_controller.add_roi(center_x, center_y, 200, 150)
        if result.is_success():
            self.video_widget.update_display()
    
    def _run_analysis(self):
        """Run brightness analysis."""
        if not self.app_controller.video_processor.is_loaded():
            QtWidgets.QMessageBox.information(
                self, "No Video", "Please load a video file first."
            )
            return
        
        if not self.app_controller.roi_manager.rois:
            QtWidgets.QMessageBox.information(
                self, "No ROIs", "Please add at least one ROI before running analysis."
            )
            return
        
        # Trigger the controls panel to start analysis
        if self.controls_panel:
            self.controls_panel.analysis_widget.start_analysis()
    
    def _auto_detect_range(self):
        """Auto-detect frame range."""
        if not self.app_controller.video_processor.is_loaded():
            return
        
        if not self.app_controller.roi_manager.rois:
            QtWidgets.QMessageBox.information(
                self, "No ROIs", "Please add at least one ROI before auto-detection."
            )
            return
        
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.app_controller.auto_detect_frame_range()
        QtWidgets.QApplication.restoreOverrideCursor()
    
    def _start_analysis(self, start_frame: int, end_frame: int, output_dir: str):
        """Start brightness analysis."""
        self.app_controller.start_analysis(start_frame, end_frame, output_dir)
    
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
        if self.app_controller.video_processor.is_loaded():
            video_info = self.app_controller.video_processor.get_video_info()
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
    
    def _cancel_roi_interaction(self):
        """Cancel any ongoing ROI interaction."""
        self.app_controller.roi_manager.cancel_all_interactions()
        self.video_widget.update_display()

    def _delete_selected_roi(self):
        """Delete the currently selected ROI."""
        if self.app_controller.roi_manager.selected_roi_index is not None:
            roi_index = self.app_controller.roi_manager.selected_roi_index
            self.app_controller.delete_roi(roi_index)

    def _step_video_frames(self, step: int):
        """Step video frames using keyboard shortcuts."""
        if self.video_widget and self.app_controller.video_processor.is_loaded():
            self.video_widget._step_frames(step)

    def _go_to_first_frame(self):
        """Go to the first frame of the video."""
        if self.video_widget and self.app_controller.video_processor.is_loaded():
            self.app_controller.seek_to_frame(0)
            self.video_widget.update_display()

    def _go_to_last_frame(self):
        """Go to the last frame of the video."""
        if self.video_widget and self.app_controller.video_processor.is_loaded():
            total_frames = self.app_controller.video_processor.total_frames
            self.app_controller.seek_to_frame(total_frames - 1)
            self.video_widget.update_display()

    def _toggle_playback(self):
        """Toggle video playback using keyboard shortcut."""
        if self.video_widget and self.app_controller.video_processor.is_loaded():
            self.video_widget._toggle_playback()

    def _on_display_update_needed(self):
        """Handle display update requests from the application controller."""
        if self.video_widget:
            self.video_widget.update_display()

    def _on_video_load_failed(self, error_message: str):
        """Handle video load failure."""
        QtWidgets.QMessageBox.critical(self, "Video Load Error", f"Failed to load video: {error_message}")
        self.status_bar.showMessage("Failed to load video", 3000)

    def _on_session_loaded(self, session):
        """Handle session load success."""
        self.status_bar.showMessage(f"Session loaded: {session.session_id}", 3000)

    def _on_analysis_completed(self, success: bool):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        if success:
            self.status_bar.showMessage("Analysis completed successfully", 3000)
        else:
            self.status_bar.showMessage("Analysis completed with issues", 3000)

    def _on_analysis_failed(self, error_message: str):
        """Handle analysis failure."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        QtWidgets.QMessageBox.critical(self, "Analysis Error", f"Analysis failed: {error_message}")
        self.status_bar.showMessage("Analysis failed", 3000)

    def _on_session_saved(self, file_path: str):
        """Handle session save success."""
        self.status_bar.showMessage(f"Session saved to {file_path}", 3000)
    




    def _on_session_save_failed(self, error_message: str):
        """Handle session save failure."""
        QtWidgets.QMessageBox.critical(self, "Session Save Error", f"Failed to save session: {error_message}")

    def _on_session_load_failed(self, error_message: str):
        """Handle session load failure."""
        QtWidgets.QMessageBox.critical(self, "Session Load Error", f"Failed to load session: {error_message}")

    def _on_roi_add_failed(self, error_message: str):
        """Handle ROI add failure."""
        QtWidgets.QMessageBox.critical(self, "ROI Add Error", f"Failed to add ROI: {error_message}")

    def _on_roi_delete_failed(self, error_message: str):
        """Handle ROI delete failure."""
        QtWidgets.QMessageBox.critical(self, "ROI Delete Error", f"Failed to delete ROI: {error_message}")

    def _on_roi_rename_failed(self, error_message: str):
        """Handle ROI rename failure."""
        QtWidgets.QMessageBox.critical(self, "ROI Rename Error", f"Failed to rename ROI: {error_message}")

    def _on_background_roi_change_failed(self, error_message: str):
        """Handle background ROI change failure."""
        QtWidgets.QMessageBox.critical(self, "Background ROI Error", f"Failed to set background ROI: {error_message}")

    def _on_frame_range_detection_failed(self, error_message: str):
        """Handle frame range detection failure."""
        QtWidgets.QMessageBox.critical(self, "Auto-Detection Error", f"Failed to auto-detect frame range: {error_message}")

    def _on_threshold_change_failed(self, error_message: str):
        """Handle threshold change failure."""
        QtWidgets.QMessageBox.critical(self, "Threshold Error", f"Failed to change threshold: {error_message}")

    def _save_session_dialog(self):
        """Open dialog to save current session."""
        if not self.app_controller.current_session:
            QtWidgets.QMessageBox.information(self, "No Session", "No active session to save.")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Analysis Session", "", "JSON Session Files (*.json)"
        )
        if file_path:
            self.app_controller.save_session(file_path)

    def _load_session_dialog(self):
        """Open dialog to load an analysis session."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Analysis Session", "", "JSON Session Files (*.json)"
        )
        if file_path:
            self.app_controller.load_session(file_path)

    def _show_shortcuts_dialog(self):
        """Show keyboard shortcuts dialog."""
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        
        <h4>Video Navigation</h4>
        <b>← / Backspace</b> - Previous frame<br>
        <b>→</b> - Next frame<br>
        <b>Page Up</b> - Jump back 10 frames<br>
        <b>Page Down</b> - Jump forward 10 frames<br>
        <b>Home</b> - Go to first frame<br>
        <b>End</b> - Go to last frame<br>
        <b>Space</b> - Play/Pause video<br>
        
        <h4>ROI Management</h4>
        <b>Ctrl+R</b> - Add new ROI<br>
        <b>Delete</b> - Delete selected ROI<br>
        <b>Escape</b> - Cancel current action<br>
        
        <h4>Analysis</h4>
        <b>F5</b> - Run analysis<br>
        <b>Ctrl+D</b> - Auto-detect frame range<br>
        
        <h4>File Operations</h4>
        <b>Ctrl+O</b> - Open video file<br>
        <b>Ctrl+S</b> - Save session<br>
        <b>Ctrl+L</b> - Load session<br>
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
        
        self.app_controller.cleanup()
        
        event.accept()
        logging.info("Application closed")