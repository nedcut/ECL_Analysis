"""
Brightness Sorcerer v2.0 - Professional Video Brightness Analysis Tool

A PyQt5-based desktop application for advanced video brightness analysis
using CIE LAB color space for perceptually uniform measurements.
"""

__version__ = "2.0.0"
__author__ = "Brightness Sorcerer Development Team"
__description__ = "Professional Video Brightness Analysis Tool"

# Import main classes for convenience
from .core.exceptions import (
    BrightnessSorcererError,
    VideoLoadError,
    AnalysisError,
    ConfigurationError,
    ValidationError
)

__all__ = [
    'BrightnessSorcererError',
    'VideoLoadError', 
    'AnalysisError',
    'ConfigurationError',
    'ValidationError'
]