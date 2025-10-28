from PyQt5 import QtWidgets

from ecl_analysis.constants import APP_WINDOW_TITLE
from ecl_analysis.video_analyzer import VideoAnalyzer


def test_video_analyzer_initialization(qt_application: QtWidgets.QApplication):
    window = VideoAnalyzer()

    assert window.windowTitle() == APP_WINDOW_TITLE
    assert window.menuBar() is not None
    assert window.statusBar() is not None

    window.close()
