"""Theme and styling system for Brightness Sorcerer UI."""

from typing import Dict, Any
from PyQt5 import QtCore

# Color palette
class AppColors:
    """Application color scheme."""
    
    # Dark theme colors
    DARK_BACKGROUND = "#2d2d2d"
    DARK_FOREGROUND = "#cccccc"
    DARK_ACCENT = "#5a9bd5"
    DARK_ACCENT_HOVER = "#7ab3e0"
    DARK_SECONDARY = "#404040"
    DARK_SECONDARY_LIGHT = "#555555"
    DARK_SUCCESS = "#70ad47"
    DARK_WARNING = "#ed7d31"
    DARK_ERROR = "#ff0000"
    DARK_INFO = "#ffc000"
    DARK_BRIGHTNESS_LABEL = "#ffeb3b"
    
    # Light theme colors
    LIGHT_BACKGROUND = "#ffffff"
    LIGHT_FOREGROUND = "#333333"
    LIGHT_ACCENT = "#0078d4"
    LIGHT_ACCENT_HOVER = "#106ebe"
    LIGHT_SECONDARY = "#f3f2f1"
    LIGHT_SECONDARY_LIGHT = "#faf9f8"
    LIGHT_SUCCESS = "#107c10"
    LIGHT_WARNING = "#ff8c00"
    LIGHT_ERROR = "#d13438"
    LIGHT_INFO = "#ffb900"
    LIGHT_BRIGHTNESS_LABEL = "#8a8886"


class AppTheme:
    """Application theme configuration."""
    
    def __init__(self, theme_name: str = "dark"):
        self.theme_name = theme_name
        self._setup_theme()
    
    def _setup_theme(self):
        """Setup theme colors and properties."""
        if self.theme_name == "dark":
            self.colors = {
                'background': AppColors.DARK_BACKGROUND,
                'foreground': AppColors.DARK_FOREGROUND,
                'accent': AppColors.DARK_ACCENT,
                'accent_hover': AppColors.DARK_ACCENT_HOVER,
                'secondary': AppColors.DARK_SECONDARY,
                'secondary_light': AppColors.DARK_SECONDARY_LIGHT,
                'success': AppColors.DARK_SUCCESS,
                'warning': AppColors.DARK_WARNING,
                'error': AppColors.DARK_ERROR,
                'info': AppColors.DARK_INFO,
                'brightness_label': AppColors.DARK_BRIGHTNESS_LABEL
            }
        else:  # light theme
            self.colors = {
                'background': AppColors.LIGHT_BACKGROUND,
                'foreground': AppColors.LIGHT_FOREGROUND,
                'accent': AppColors.LIGHT_ACCENT,
                'accent_hover': AppColors.LIGHT_ACCENT_HOVER,
                'secondary': AppColors.LIGHT_SECONDARY,
                'secondary_light': AppColors.LIGHT_SECONDARY_LIGHT,
                'success': AppColors.LIGHT_SUCCESS,
                'warning': AppColors.LIGHT_WARNING,
                'error': AppColors.LIGHT_ERROR,
                'info': AppColors.LIGHT_INFO,
                'brightness_label': AppColors.LIGHT_BRIGHTNESS_LABEL
            }
    
    def get_stylesheet(self) -> str:
        """Get complete application stylesheet."""
        return f"""
        /* Main application styling */
        QMainWindow {{
            background-color: {self.colors['background']};
            color: {self.colors['foreground']};
            font-family: "Segoe UI", Arial, sans-serif;
        }}
        
        /* Buttons */
        QPushButton {{
            background-color: {self.colors['accent']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: 500;
            min-width: 80px;
        }}
        
        QPushButton:hover {{
            background-color: {self.colors['accent_hover']};
        }}
        
        QPushButton:pressed {{
            background-color: {self.colors['accent']};
            margin: 1px;
        }}
        
        QPushButton:disabled {{
            background-color: {self.colors['secondary']};
            color: #888888;
        }}
        
        /* Secondary buttons */
        QPushButton[secondary="true"] {{
            background-color: {self.colors['secondary']};
            color: {self.colors['foreground']};
        }}
        
        QPushButton[secondary="true"]:hover {{
            background-color: {self.colors['secondary_light']};
        }}
        
        /* Panels and containers */
        QWidget[panel="true"] {{
            background-color: {self.colors['secondary']};
            border: 1px solid {self.colors['secondary_light']};
            border-radius: 6px;
            padding: 8px;
        }}
        
        QGroupBox {{
            font-weight: bold;
            border: 2px solid {self.colors['secondary_light']};
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}
        
        /* Labels */
        QLabel {{
            color: {self.colors['foreground']};
        }}
        
        QLabel[brightness="true"] {{
            color: {self.colors['brightness_label']};
            font-weight: bold;
        }}
        
        QLabel[status="success"] {{
            color: {self.colors['success']};
        }}
        
        QLabel[status="warning"] {{
            color: {self.colors['warning']};
        }}
        
        QLabel[status="error"] {{
            color: {self.colors['error']};
        }}
        
        QLabel[status="info"] {{
            color: {self.colors['info']};
        }}
        
        /* Input fields */
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {self.colors['background']};
            border: 1px solid {self.colors['secondary_light']};
            border-radius: 4px;
            padding: 6px;
            color: {self.colors['foreground']};
        }}
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {self.colors['accent']};
        }}
        
        /* Sliders */
        QSlider::groove:horizontal {{
            border: 1px solid {self.colors['secondary_light']};
            height: 6px;
            background: {self.colors['secondary']};
            border-radius: 3px;
        }}
        
        QSlider::handle:horizontal {{
            background: {self.colors['accent']};
            border: 1px solid {self.colors['accent']};
            width: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background: {self.colors['accent_hover']};
        }}
        
        /* Progress bars */
        QProgressBar {{
            border: 1px solid {self.colors['secondary_light']};
            border-radius: 4px;
            text-align: center;
            background-color: {self.colors['secondary']};
        }}
        
        QProgressBar::chunk {{
            background-color: {self.colors['accent']};
            border-radius: 3px;
        }}
        
        /* Menu and menu bar */
        QMenuBar {{
            background-color: {self.colors['background']};
            color: {self.colors['foreground']};
            border-bottom: 1px solid {self.colors['secondary_light']};
        }}
        
        QMenuBar::item {{
            background-color: transparent;
            padding: 4px 8px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {self.colors['accent']};
        }}
        
        QMenu {{
            background-color: {self.colors['secondary']};
            border: 1px solid {self.colors['secondary_light']};
            border-radius: 4px;
        }}
        
        QMenu::item {{
            padding: 6px 20px;
        }}
        
        QMenu::item:selected {{
            background-color: {self.colors['accent']};
        }}
        
        /* Status bar */
        QStatusBar {{
            background-color: {self.colors['secondary']};
            border-top: 1px solid {self.colors['secondary_light']};
        }}
        
        /* Tooltips */
        QToolTip {{
            background-color: {self.colors['secondary']};
            color: {self.colors['foreground']};
            border: 1px solid {self.colors['secondary_light']};
            border-radius: 4px;
            padding: 4px;
        }}
        
        /* Scroll bars */
        QScrollBar:vertical {{
            background-color: {self.colors['secondary']};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {self.colors['accent']};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {self.colors['accent_hover']};
        }}
        
        /* Splitters */
        QSplitter::handle {{
            background-color: {self.colors['secondary_light']};
        }}
        
        QSplitter::handle:horizontal {{
            width: 2px;
        }}
        
        QSplitter::handle:vertical {{
            height: 2px;
        }}
        """
    
    def get_color(self, color_name: str) -> str:
        """Get specific color from theme."""
        return self.colors.get(color_name, self.colors['foreground'])


def apply_dark_theme(app):
    """Apply dark theme to the application."""
    theme = AppTheme("dark")
    app.setStyleSheet(theme.get_stylesheet())
    return theme


def apply_light_theme(app):
    """Apply light theme to the application."""
    theme = AppTheme("light")
    app.setStyleSheet(theme.get_stylesheet())
    return theme


# ROI colors for drawing
ROI_COLORS = [
    (255, 50, 50), (50, 200, 50), (50, 150, 255), (255, 150, 50),
    (255, 50, 255), (50, 200, 200), (150, 50, 255), (255, 255, 50)
]

# ROI drawing constants
ROI_THICKNESS_DEFAULT = 2
ROI_THICKNESS_SELECTED = 4
ROI_LABEL_FONT_SCALE = 0.8
ROI_LABEL_THICKNESS = 2
MOUSE_RESIZE_HANDLE_SENSITIVITY = 10