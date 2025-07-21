"""
Input validation functions for Brightness Sorcerer.

Provides comprehensive validation for video files, ROI coordinates,
frame ranges, and safe type conversions with bounds checking.
"""

import logging
from pathlib import Path
from typing import Optional

from ..core.exceptions import ValidationError
from .constants import SUPPORTED_VIDEO_FORMATS

logger = logging.getLogger(__name__)

# ROI validation constants
MIN_ROI_SIZE = (10, 10)  # minimum width, height in pixels
MAX_ROI_SIZE_RATIO = 0.9  # maximum fraction of frame size


def validate_video_file(file_path: str) -> bool:
    """
    Validate that the video file exists and is readable.
    
    Args:
        file_path: Path to the video file to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If file is invalid or inaccessible
    """
    if not file_path:
        raise ValidationError("Video file path cannot be empty")
    
    path = Path(file_path)
    if not path.exists():
        raise ValidationError(f"Video file does not exist: {file_path}")
    
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")
    
    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        supported_formats = ', '.join(SUPPORTED_VIDEO_FORMATS)
        raise ValidationError(f"Unsupported video format: {path.suffix}. Supported formats: {supported_formats}")
    
    if path.stat().st_size == 0:
        raise ValidationError(f"Video file is empty: {file_path}")
    
    return True


def validate_roi_coordinates(x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int) -> bool:
    """
    Validate ROI coordinates are within frame bounds and meet size requirements.
    
    Args:
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates  
        frame_width, frame_height: Frame dimensions
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If coordinates are invalid
    """
    # Ensure coordinates are in correct order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    # Check bounds
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        raise ValidationError(f"ROI coordinates ({x1}, {y1}, {x2}, {y2}) are outside frame bounds ({frame_width}x{frame_height})")
    
    # Check minimum size
    roi_width = x2 - x1
    roi_height = y2 - y1
    if roi_width < MIN_ROI_SIZE[0] or roi_height < MIN_ROI_SIZE[1]:
        raise ValidationError(f"ROI size ({roi_width}x{roi_height}) is below minimum size {MIN_ROI_SIZE}")
    
    # Check maximum size ratio
    max_width = int(frame_width * MAX_ROI_SIZE_RATIO)
    max_height = int(frame_height * MAX_ROI_SIZE_RATIO)
    if roi_width > max_width or roi_height > max_height:
        raise ValidationError(f"ROI size ({roi_width}x{roi_height}) exceeds maximum allowed size ({max_width}x{max_height})")
    
    return True


def validate_frame_range(start_frame: int, end_frame: int, total_frames: int) -> bool:
    """
    Validate frame range parameters.
    
    Args:
        start_frame: Starting frame index
        end_frame: Ending frame index
        total_frames: Total number of frames in video
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If frame range is invalid
    """
    if start_frame < 0:
        raise ValidationError(f"Start frame cannot be negative: {start_frame}")
    
    if end_frame < start_frame:
        raise ValidationError(f"End frame ({end_frame}) cannot be less than start frame ({start_frame})")
    
    if start_frame >= total_frames:
        raise ValidationError(f"Start frame ({start_frame}) must be less than total frames ({total_frames})")
    
    if end_frame >= total_frames:
        raise ValidationError(f"End frame ({end_frame}) must be less than total frames ({total_frames})")
    
    return True


def safe_float_conversion(value, default: float = 0.0, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """
    Safely convert value to float with bounds checking.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        
    Returns:
        Converted float value within bounds
    """
    try:
        result = float(value)
        if min_val is not None and result < min_val:
            logger.warning(f"Value {result} below minimum {min_val}, using minimum")
            return min_val
        if max_val is not None and result > max_val:
            logger.warning(f"Value {result} above maximum {max_val}, using maximum")
            return max_val
        return result
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert {value} to float: {e}, using default {default}")
        return default


def safe_int_conversion(value, default: int = 0, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """
    Safely convert value to int with bounds checking.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        
    Returns:
        Converted int value within bounds
    """
    try:
        result = int(value)
        if min_val is not None and result < min_val:
            logger.warning(f"Value {result} below minimum {min_val}, using minimum")
            return min_val
        if max_val is not None and result > max_val:
            logger.warning(f"Value {result} above maximum {max_val}, using maximum")
            return max_val
        return result
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert {value} to int: {e}, using default {default}")
        return default