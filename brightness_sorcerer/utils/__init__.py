"""
Utility functions and constants for Brightness Sorcerer.

Contains validation functions, constants, logging configuration,
and other utilities used throughout the application.
"""

from .constants import *
from .validation import *

__all__ = [
    # Constants
    'DEFAULT_FONT_FAMILY',
    'COLOR_BACKGROUND',
    'COLOR_FOREGROUND', 
    'COLOR_ACCENT',
    'ROI_COLORS',
    'SUPPORTED_VIDEO_FORMATS',
    'FRAME_CACHE_SIZE',
    'DEFAULT_MANUAL_THRESHOLD',
    'APP_NAME',
    'APP_VERSION',
    'DEFAULT_SETTINGS_FILE',
    
    # Validation functions
    'validate_video_file',
    'validate_roi_coordinates',
    'validate_frame_range',
    'safe_float_conversion',
    'safe_int_conversion',
]