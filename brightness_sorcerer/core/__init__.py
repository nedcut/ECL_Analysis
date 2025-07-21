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

__all__ = [
    'BrightnessSorcererError',
    'VideoLoadError',
    'AnalysisError', 
    'ConfigurationError',
    'ValidationError'
]