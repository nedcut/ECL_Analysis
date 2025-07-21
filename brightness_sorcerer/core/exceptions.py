"""
Custom exception classes for Brightness Sorcerer application.

Provides specific exception types for different error categories,
enabling more precise error handling and better user experience.
"""

from typing import Optional, Any


class BrightnessSorcererError(Exception):
    """Base exception class for Brightness Sorcerer application."""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details


class VideoLoadError(BrightnessSorcererError):
    """Raised when video file cannot be loaded or is invalid."""
    pass


class AnalysisError(BrightnessSorcererError):
    """Raised when analysis operations fail."""
    pass


class ConfigurationError(BrightnessSorcererError):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


class ValidationError(BrightnessSorcererError):
    """Raised when input validation fails."""
    pass


class ROIError(BrightnessSorcererError):
    """Raised when ROI operations fail."""
    pass


class CacheError(BrightnessSorcererError):
    """Raised when frame cache operations fail."""
    pass


class AudioError(BrightnessSorcererError):
    """Raised when audio operations fail."""
    pass