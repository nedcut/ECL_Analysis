"""Application entry point helpers."""

import logging
import sys
from typing import Callable

from PyQt5 import QtCore, QtWidgets

from .video_analyzer import VideoAnalyzer

_QT_ATTRIBUTES_CONFIGURED = False


def _configure_qt_attributes():
    """Apply high-DPI friendly Qt attributes once before QApplication exists."""
    global _QT_ATTRIBUTES_CONFIGURED
    if _QT_ATTRIBUTES_CONFIGURED:
        return

    attribute_names = ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps")
    for name in attribute_names:
        attribute = getattr(QtCore.Qt, name, None)
        if attribute is not None:
            QtCore.QCoreApplication.setAttribute(attribute)  # type: ignore[arg-type]

    _QT_ATTRIBUTES_CONFIGURED = True


def _exec_app(app: QtWidgets.QApplication) -> int:
    """Call the correct exec variant for the current Qt version."""
    exec_fn: Callable[[], int] = getattr(app, "exec", app.exec_)
    return exec_fn()


def run_app():
    """Launch the Brightness Sorcerer GUI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    _configure_qt_attributes()
    app = QtWidgets.QApplication(sys.argv)

    try:
        window = VideoAnalyzer()
    except Exception as exc:  # pragma: no cover - guard during GUI startup
        logging.exception("Failed to initialize Brightness Sorcerer UI")
        QtWidgets.QMessageBox.critical(
            None,
            "Initialization Error",
            f"Brightness Sorcerer could not start:\n{exc}",
        )
        sys.exit(1)

    window.show()
    sys.exit(_exec_app(app))
