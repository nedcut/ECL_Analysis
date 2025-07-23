"""
Core functionality for Brightness Sorcerer.

This package contains the core video processing, analysis algorithms,
and data structures used throughout the application.
"""

from .exceptions import (
    BrightnessSorcererError,
    VideoLoadError,
    AnalysisError,
    ConfigurationError,
    ValidationError
)

# Import core business logic classes (will be created)
try:
    from .video_processor import VideoProcessor
    from .brightness_analyzer import BrightnessAnalyzer
    from .roi_manager import ROIManager
    VIDEO_CLASSES_AVAILABLE = True
except ImportError:
    # Classes not yet created - will be available after refactoring
    VIDEO_CLASSES_AVAILABLE = False

__all__ = [
    'BrightnessSorcererError',
    'VideoLoadError',
    'AnalysisError', 
    'ConfigurationError',
    'ValidationError'
]

if VIDEO_CLASSES_AVAILABLE:
    __all__.extend([
        'VideoProcessor',
        'BrightnessAnalyzer',
        'ROIManager'
    ])