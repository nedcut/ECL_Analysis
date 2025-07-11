"""
Brightness Sorcerer v3.0 - Professional Video Brightness Analysis Tool

A modular, maintainable application for analyzing brightness changes in video regions of interest.
"""

__version__ = "3.0.0"
__author__ = "ECL Analysis Team"

from .core.video_processor import VideoProcessor
from .core.roi_manager import ROIManager  
from .core.brightness_analyzer import BrightnessAnalyzer
from .core.settings_manager import SettingsManager

__all__ = [
    "VideoProcessor",
    "ROIManager", 
    "BrightnessAnalyzer",
    "SettingsManager"
]