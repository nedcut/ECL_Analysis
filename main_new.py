#!/usr/bin/env python3
"""
Brightness Sorcerer v3.0 - Main Entry Point
Professional Video Brightness Analysis Tool

This is the main entry point for the new modular version of Brightness Sorcerer.
It initializes the PyQt5 application and launches the main window.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path so we can import brightness_sorcerer
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5 import QtWidgets, QtCore, QtGui
from brightness_sorcerer.ui.main_window import MainWindow
from brightness_sorcerer.ui.styles.theme import apply_dark_theme


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('brightness_sorcerer.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logging.info("Starting Brightness Sorcerer v3.0")
    
    # Create QApplication
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Brightness Sorcerer")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("ECL Analysis")
    app.setOrganizationDomain("ecl-analysis.com")
    
    # Set application icon (if available)
    try:
        icon_path = Path(__file__).parent / "assets" / "icon.ico"
        if icon_path.exists():
            app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    except Exception:
        pass  # Icon not critical for functionality
    
    # Apply dark theme by default
    apply_dark_theme(app)
    
    # Create and show main window
    try:
        main_window = MainWindow()
        main_window.show()
        
        # Handle command line arguments for video file
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
            if Path(video_path).exists():
                main_window._load_video(video_path)
                logging.info(f"Loaded video from command line: {video_path}")
        
        logging.info("Application started successfully")
        
        # Run the application
        return app.exec_()
        
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        QtWidgets.QMessageBox.critical(
            None, "Startup Error", 
            f"Failed to start Brightness Sorcerer:\n{str(e)}"
        )
        return 1


if __name__ == '__main__':
    sys.exit(main())