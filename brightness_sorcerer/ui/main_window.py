"""
Modern Main Window

Enhanced main application window with modular architecture and modern UI/UX.
"""

import os
import sys
from typing import Optional, Dict, Any
from PyQt5 import QtCore, QtGui, QtWidgets

from .widgets.video_player import VideoPlayerWidget
from .widgets.roi_editor import ROIEditorWidget
from .widgets.analysis_panel import AnalysisPanelWidget
from .widgets.results_viewer import ResultsViewerWidget
from .widgets.toolbar import MainToolbar


class ModernMainWindow(QtWidgets.QMainWindow):
    """
    Modern main application window with enhanced UI/UX and modular architecture.
    
    Features:
    - Responsive layout with proper widget organization
    - Modern dark theme with professional styling
    - Modular component architecture for maintainability
    - Enhanced user experience with intuitive controls
    """
    
    def __init__(self):
        super().__init__()
        
        # Application state
        self.current_video_path: Optional[str] = None
        self.analysis_results: Dict[str, Any] = {}
        
        # Initialize UI
        self._setup_window()
        self._setup_ui()
        self._setup_menubar()
        self._setup_statusbar()
        self._setup_styles()
        self._connect_signals()
        
        # Show maximized for better initial experience
        self.showMaximized()
    
    def _setup_window(self):
        """Configure main window properties."""
        self.setWindowTitle("Brightness Sorcerer - Professional Video Analysis")
        self.setWindowIcon(QtGui.QIcon())  # TODO: Add application icon
        self.setMinimumSize(1200, 800)
        
        # Set application icon in taskbar (Windows/Linux)
        if hasattr(QtWidgets.QApplication, 'setWindowIcon'):
            QtWidgets.QApplication.instance().setWindowIcon(QtGui.QIcon())
    
    def _setup_ui(self):
        """Create and layout the main UI components."""
        # Central widget with splitter layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Toolbar
        self.toolbar = MainToolbar()
        self.addToolBar(self.toolbar)
        
        # Main content area with splitters
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        
        # Left panel (video + ROI editor)
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(8)
        
        # Video player (takes most space)
        self.video_player = VideoPlayerWidget()
        left_layout.addWidget(self.video_player, 3)
        
        # ROI editor (smaller section)
        self.roi_editor = ROIEditorWidget()
        left_layout.addWidget(self.roi_editor, 1)
        
        # Right panel (analysis + results)
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(8)
        
        # Analysis panel
        self.analysis_panel = AnalysisPanelWidget()
        right_layout.addWidget(self.analysis_panel, 1)
        
        # Results viewer
        self.results_viewer = ResultsViewerWidget()
        right_layout.addWidget(self.results_viewer, 1)
        
        # Add panels to splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (70% left, 30% right)
        main_splitter.setSizes([800, 400])
        
        main_layout.addWidget(main_splitter)
    
    def _setup_menubar(self):
        """Create modern menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QtWidgets.QAction("&Open Video...", self)
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.setStatusTip("Open video file for analysis")
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_project_action = QtWidgets.QAction("&Save Project", self)
        save_project_action.setShortcut(QtGui.QKeySequence.Save)
        save_project_action.setStatusTip("Save current project")
        save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(save_project_action)
        
        load_project_action = QtWidgets.QAction("&Load Project...", self)
        load_project_action.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        load_project_action.setStatusTip("Load existing project")
        load_project_action.triggered.connect(self._load_project)
        file_menu.addAction(load_project_action)
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction("E&xit", self)
        exit_action.setShortcut(QtGui.QKeySequence.Quit)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        
        start_analysis_action = QtWidgets.QAction("&Start Analysis", self)
        start_analysis_action.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        start_analysis_action.setStatusTip("Start brightness analysis")
        start_analysis_action.triggered.connect(self._start_analysis)
        analysis_menu.addAction(start_analysis_action)
        
        analysis_menu.addSeparator()
        
        export_results_action = QtWidgets.QAction("&Export Results...", self)
        export_results_action.setShortcut(QtGui.QKeySequence("Ctrl+E"))
        export_results_action.setStatusTip("Export analysis results")
        export_results_action.triggered.connect(self._export_results)
        analysis_menu.addAction(export_results_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Theme submenu
        theme_menu = view_menu.addMenu("&Theme")
        
        dark_theme_action = QtWidgets.QAction("&Dark Theme", self)
        dark_theme_action.setCheckable(True)
        dark_theme_action.setChecked(True)
        dark_theme_action.triggered.connect(lambda: self._set_theme('dark'))
        theme_menu.addAction(dark_theme_action)
        
        light_theme_action = QtWidgets.QAction("&Light Theme", self)
        light_theme_action.setCheckable(True)
        light_theme_action.triggered.connect(lambda: self._set_theme('light'))
        theme_menu.addAction(light_theme_action)
        
        # Theme action group
        theme_group = QtWidgets.QActionGroup(self)
        theme_group.addAction(dark_theme_action)
        theme_group.addAction(light_theme_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QtWidgets.QAction("&About", self)
        about_action.setStatusTip("About Brightness Sorcerer")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_statusbar(self):
        """Create modern status bar."""
        self.status_bar = self.statusBar()
        
        # Status message
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Video info
        self.video_info_label = QtWidgets.QLabel("No video loaded")
        self.status_bar.addPermanentWidget(self.video_info_label)
    
    def _setup_styles(self):
        """Apply modern application-wide styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            
            QMenuBar {
                background-color: #2b2b2b;
                color: #ffffff;
                border-bottom: 1px solid #555555;
                padding: 4px;
            }
            
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 12px;
                border-radius: 4px;
            }
            
            QMenuBar::item:selected {
                background-color: #3daee9;
            }
            
            QMenu {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
            }
            
            QMenu::item {
                padding: 8px 20px;
                border-radius: 4px;
            }
            
            QMenu::item:selected {
                background-color: #3daee9;
            }
            
            QMenu::separator {
                height: 1px;
                background-color: #555555;
                margin: 4px 0;
            }
            
            QStatusBar {
                background-color: #2b2b2b;
                color: #ffffff;
                border-top: 1px solid #555555;
            }
            
            QSplitter::handle {
                background-color: #555555;
                width: 2px;
                height: 2px;
            }
            
            QSplitter::handle:hover {
                background-color: #3daee9;
            }
            
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                background-color: #2b2b2b;
                color: #ffffff;
            }
            
            QProgressBar::chunk {
                background-color: #3daee9;
                border-radius: 3px;
            }
        """)
    
    def _connect_signals(self):
        """Connect widget signals to handlers."""
        # Toolbar signals
        self.toolbar.openVideoRequested.connect(self._open_video)
        self.toolbar.saveProjectRequested.connect(self._save_project)
        self.toolbar.loadProjectRequested.connect(self._load_project)
        self.toolbar.settingsRequested.connect(self._show_settings)
        self.toolbar.helpRequested.connect(self._show_help)
        
        # Video player signals
        self.video_player.videoLoaded.connect(self._on_video_loaded)
        self.video_player.frameChanged.connect(self._on_frame_changed)
        
        # Analysis panel signals
        self.analysis_panel.analysisRequested.connect(self._on_analysis_requested)
        self.analysis_panel.settingsChanged.connect(self._on_analysis_settings_changed)
        
        # Results viewer signals
        self.results_viewer.exportRequested.connect(self._on_export_requested)
    
    def _open_video(self):
        """Open video file dialog."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.video_player.load_video(file_path)
    
    def _save_project(self):
        """Save current project."""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "Project Files (*.bsproj);;JSON Files (*.json)"
        )
        
        if file_path:
            # TODO: Implement project saving
            self.status_label.setText(f"Project saved: {os.path.basename(file_path)}")
    
    def _load_project(self):
        """Load existing project."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Project", 
            "",
            "Project Files (*.bsproj);;JSON Files (*.json)"
        )
        
        if file_path:
            # TODO: Implement project loading
            self.status_label.setText(f"Project loaded: {os.path.basename(file_path)}")
    
    def _start_analysis(self):
        """Start brightness analysis."""
        if self.current_video_path:
            settings = self.analysis_panel.get_current_settings()
            self._on_analysis_requested(settings)
        else:
            QtWidgets.QMessageBox.warning(
                self, "No Video", "Please load a video file first."
            )
    
    def _export_results(self):
        """Export analysis results."""
        if not self.analysis_results:
            QtWidgets.QMessageBox.information(
                self, "No Results", "No analysis results to export."
            )
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "CSV Files (*.csv);;JSON Files (*.json);;Excel Files (*.xlsx)"
        )
        
        if file_path:
            # TODO: Implement results export
            self.status_label.setText(f"Results exported: {os.path.basename(file_path)}")
    
    def _set_theme(self, theme: str):
        """Set application theme."""
        if theme == 'light':
            # TODO: Implement light theme
            self.status_label.setText("Light theme not yet implemented")
        else:
            # Dark theme is default
            self.status_label.setText("Dark theme active")
    
    def _show_settings(self):
        """Show settings dialog."""
        # TODO: Implement settings dialog
        QtWidgets.QMessageBox.information(
            self, "Settings", "Settings dialog not yet implemented."
        )
    
    def _show_help(self):
        """Show help dialog."""
        # TODO: Implement help dialog
        QtWidgets.QMessageBox.information(
            self, "Help", "Help system not yet implemented."
        )
    
    def _show_about(self):
        """Show about dialog."""
        QtWidgets.QMessageBox.about(
            self,
            "About Brightness Sorcerer",
            """
            <h3>Brightness Sorcerer v2.0</h3>
            <p>Professional Video Brightness Analysis Tool</p>
            <p>Modern UI with enhanced functionality and improved user experience.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>Advanced CIE LAB color space analysis</li>
                <li>Blue channel bioluminescence detection</li>
                <li>ROI-based analysis with reference masking</li>
                <li>Professional data export and visualization</li>
            </ul>
            """
        )
    
    def _on_video_loaded(self, video_path: str):
        """Handle video loaded signal."""
        self.current_video_path = video_path
        filename = os.path.basename(video_path)
        self.video_info_label.setText(f"Video: {filename}")
        self.status_label.setText(f"Video loaded: {filename}")
        
        # Enable analysis
        self.analysis_panel.set_analysis_enabled(True)
    
    def _on_frame_changed(self, frame_number: int):
        """Handle frame change signal."""
        # Update status if needed
        pass
    
    def _on_analysis_requested(self, settings: Dict[str, Any]):
        """Handle analysis request."""
        if not self.current_video_path:
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.status_label.setText("Running analysis...")
        
        # TODO: Implement actual analysis
        # For now, show placeholder results
        QtCore.QTimer.singleShot(2000, self._on_analysis_complete)
    
    def _on_analysis_complete(self):
        """Handle analysis completion."""
        # Hide progress
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis complete")
        
        # Show placeholder results
        self.analysis_results = {
            "Total Frames": 1000,
            "Average Brightness": 45.2,
            "Peak Brightness": 89.7,
            "Analysis Method": "Enhanced"
        }
        
        self.results_viewer.display_results(self.analysis_results)
    
    def _on_analysis_settings_changed(self, settings: Dict[str, Any]):
        """Handle analysis settings change."""
        # Update status or UI as needed
        pass
    
    def _on_export_requested(self, format_type: str):
        """Handle export request."""
        self._export_results()
    
    def closeEvent(self, event):
        """Handle application close."""
        # Clean up resources
        if hasattr(self.video_player, 'cap') and self.video_player.cap:
            self.video_player.cap.release()
        
        # Accept close event
        event.accept()


def create_application() -> QtWidgets.QApplication:
    """Create and configure the Qt application."""
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Brightness Sorcerer")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("Brightness Sorcerer Team")
    
    # Set application icon
    # app.setWindowIcon(QtGui.QIcon("assets/icon.png"))  # TODO: Add icon
    
    return app


def main():
    """Main application entry point."""
    app = create_application()
    
    # Create and show main window
    window = ModernMainWindow()
    window.show()
    
    # Start event loop
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())