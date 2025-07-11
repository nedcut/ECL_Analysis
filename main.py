#!/usr/bin/env python3
"""
Brightness Sorcerer v3.0 - Professional Video Brightness Analysis Tool

Entry point for the modular, refactored application.
"""

import sys
import os
import logging
from PyQt5 import QtWidgets

# Add the brightness_sorcerer package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brightness_sorcerer.ui.main_window import MainWindow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('brightness_sorcerer.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class BrightnessSorcererApp:
    """Main application controller."""
    
    def __init__(self):
        """Initialize the application."""
        logger.info("Initializing Brightness Sorcerer v3.0")
        self.main_window = None
        logger.info("Application initialized successfully")
    
    def run(self):
        """Run the application."""
        try:
            # Create Qt application
            app = QtWidgets.QApplication(sys.argv)
            app.setApplicationName("Brightness Sorcerer")
            app.setApplicationVersion("3.0.0")
            app.setOrganizationName("ECL Analysis")
            
            # Create and show main window
            self.main_window = MainWindow()
            self.main_window.show()
            
            logger.info("Main window created and shown")
            
            # Run the application
            return app.exec_()
            
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            
            # Show error dialog
            error_dialog = QtWidgets.QMessageBox()
            error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
            error_dialog.setWindowTitle("Application Error")
            error_dialog.setText("An error occurred while running the application.")
            error_dialog.setDetailedText(str(e))
            error_dialog.exec_()
            
            return 1
    
    def cleanup(self):
        """Cleanup application resources."""
        logger.info("Cleaning up application resources")
        
        if self.main_window:
            # Cleanup is handled by the main window's closeEvent
            pass
        
        logger.info("Cleanup completed")


def main():
    """Main entry point."""
    try:
        # Create and run application
        app = BrightnessSorcererApp()
        exit_code = app.run()
        app.cleanup()
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())