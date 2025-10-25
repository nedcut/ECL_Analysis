"""Application entry point helpers."""

import logging
import sys

from PyQt5 import QtWidgets

from .video_analyzer import VideoAnalyzer


def run_app():
    """Launch the Brightness Sorcerer GUI."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    app = QtWidgets.QApplication(sys.argv)
    window = VideoAnalyzer()
    window.show()
    sys.exit(app.exec_())
