"""UI stylesheet definitions for the application."""

from ..constants import (
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
)


def get_application_stylesheet() -> str:
    """
    Get the complete application stylesheet.

    Returns:
        QSS stylesheet string
    """
    return f"""
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
    """