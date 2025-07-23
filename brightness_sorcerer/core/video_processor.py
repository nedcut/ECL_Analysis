"""
Video processing module for Brightness Sorcerer.

This module handles all video-related operations including:
- Video file loading and validation
- Frame extraction and caching
- Video property management
- Frame navigation and seeking
"""

import os
import logging
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
from collections import OrderedDict

from ..utils.validation import validate_video_file, safe_int_conversion
from ..utils.constants import FRAME_CACHE_SIZE, SUPPORTED_VIDEO_FORMATS
from .exceptions import VideoLoadError, ValidationError

logger = logging.getLogger(__name__)


class FrameCache:
    """
    LRU cache for video frames to improve performance.
    
    Maintains a cache of recently accessed frames to avoid repeated
    video file reads. Uses OrderedDict for efficient LRU eviction.
    """
    
    def __init__(self, max_size: int = FRAME_CACHE_SIZE):
        """
        Initialize frame cache.
        
        Args:
            max_size: Maximum number of frames to cache
        """
        self.max_size = max_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        
    def get(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get frame from cache if available.
        
        Args:
            frame_index: Frame index to retrieve
            
        Returns:
            Frame array if cached, None otherwise
        """
        if frame_index in self._cache:
            # Move to end (most recently used)
            frame = self._cache.pop(frame_index)
            self._cache[frame_index] = frame
            return frame.copy()
        return None
        
    def put(self, frame_index: int, frame: np.ndarray) -> None:
        """
        Add frame to cache with LRU eviction.
        
        Args:
            frame_index: Frame index
            frame: Frame array to cache
        """
        if frame_index in self._cache:
            # Update existing entry
            self._cache.pop(frame_index)
        elif len(self._cache) >= self.max_size:
            # Remove least recently used
            self._cache.popitem(last=False)
            
        self._cache[frame_index] = frame.copy()
        
    def clear(self) -> None:
        """Clear all cached frames."""
        self._cache.clear()
        
    def get_size(self) -> int:
        """Get current number of cached frames."""
        return len(self._cache)
        
    def get_max_size(self) -> int:
        """Get maximum cache size."""
        return self.max_size


class VideoProcessor:
    """
    Core video processing functionality.
    
    Handles video file operations, frame extraction, caching,
    and video property management with proper error handling
    and resource cleanup.
    """
    
    def __init__(self, cache_size: int = FRAME_CACHE_SIZE):
        """
        Initialize video processor.
        
        Args:
            cache_size: Maximum number of frames to cache
        """
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: str = ""
        self.total_frames: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.fps: float = 0.0
        self.duration_seconds: float = 0.0
        
        # Frame caching
        self.frame_cache = FrameCache(cache_size)
        
        # Current state
        self.current_frame_index: int = 0
        self.current_frame: Optional[np.ndarray] = None
        
        logger.debug(f"VideoProcessor initialized with cache size {cache_size}")
        
    def __del__(self):
        """Cleanup resources on destruction."""
        try:
            self.close_video()
        except Exception as e:
            logger.debug(f"Error during VideoProcessor cleanup: {e}")
            
    def is_loaded(self) -> bool:
        """Check if a video is currently loaded."""
        return self.cap is not None and self.cap.isOpened()
        
    def load_video(self, video_path: str) -> Dict[str, Any]:
        """
        Load video file with comprehensive validation and error handling.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties and loading status
            
        Raises:
            VideoLoadError: If video loading fails
            ValidationError: If video file is invalid
        """
        logger.info(f"Loading video: {os.path.basename(video_path)}")
        
        # Validate file exists and has supported format
        if not os.path.exists(video_path):
            raise VideoLoadError(f"Video file not found: {video_path}")
            
        if not validate_video_file(video_path):
            raise ValidationError(f"Unsupported video format. Supported formats: {SUPPORTED_VIDEO_FORMATS}")
            
        # Close any existing video
        self.close_video()
        
        try:
            # Open video file
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise VideoLoadError(f"Failed to open video file: {video_path}")
                
            # Extract video properties
            self._extract_video_properties(video_path)
            
            # Load first frame
            self.seek_to_frame(0)
            
            logger.info(f"Video loaded successfully: {self.frame_width}x{self.frame_height}, "
                       f"{self.total_frames} frames, {self.fps:.2f} FPS, {self.duration_seconds:.2f}s")
            
            return {
                'path': self.video_path,
                'width': self.frame_width,
                'height': self.frame_height,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'duration': self.duration_seconds,
                'current_frame': 0
            }
            
        except cv2.error as e:
            self.close_video()
            raise VideoLoadError(f"OpenCV error loading video: {e}")
        except Exception as e:
            self.close_video()
            raise VideoLoadError(f"Unexpected error loading video: {e}")
            
    def _extract_video_properties(self, video_path: str) -> None:
        """Extract and validate video properties."""
        if not self.cap:
            raise VideoLoadError("No video capture object")
            
        self.video_path = video_path
        
        # Extract properties with validation
        total_frames_raw = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = safe_int_conversion(total_frames_raw, default=0, min_val=1)
        
        if self.total_frames <= 0:
            raise VideoLoadError("Invalid video: no frames detected")
            
        self.frame_width = safe_int_conversion(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), default=0, min_val=1
        )
        self.frame_height = safe_int_conversion(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), default=0, min_val=1
        )
        
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise VideoLoadError("Invalid video: invalid frame dimensions")
            
        self.fps = max(self.cap.get(cv2.CAP_PROP_FPS), 1.0)  # Avoid division by zero
        self.duration_seconds = self.total_frames / self.fps
        
    def close_video(self) -> None:
        """Close video file and cleanup resources."""
        if self.cap:
            try:
                self.cap.release()
                logger.debug("Video capture resources released")
            except Exception as e:
                logger.warning(f"Error releasing video capture: {e}")
            finally:
                self.cap = None
                
        # Clear cache and reset state
        self.frame_cache.clear()
        self._reset_state()
        
    def _reset_state(self) -> None:
        """Reset internal state after video closure."""
        self.video_path = ""
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
        self.duration_seconds = 0.0
        self.current_frame_index = 0
        self.current_frame = None
        
    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get frame at specified index with caching.
        
        Args:
            frame_index: Frame index to retrieve (0-based)
            
        Returns:
            Frame array if successful, None otherwise
        """
        if not self.is_loaded():
            logger.warning("No video loaded")
            return None
            
        # Validate frame index
        if not (0 <= frame_index < self.total_frames):
            logger.warning(f"Frame index {frame_index} out of range [0, {self.total_frames})")
            return None
            
        # Check cache first
        cached_frame = self.frame_cache.get(frame_index)
        if cached_frame is not None:
            logger.debug(f"Frame {frame_index} retrieved from cache")
            return cached_frame
            
        # Read from video file
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                # Cache the frame
                self.frame_cache.put(frame_index, frame)
                logger.debug(f"Frame {frame_index} loaded and cached")
                return frame.copy()
            else:
                logger.warning(f"Failed to read frame {frame_index}")
                return None
                
        except cv2.error as e:
            logger.error(f"OpenCV error reading frame {frame_index}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading frame {frame_index}: {e}")
            return None
            
    def seek_to_frame(self, frame_index: int) -> bool:
        """
        Seek to specific frame and update current frame.
        
        Args:
            frame_index: Target frame index
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_loaded():
            return False
            
        # Validate frame index
        if not (0 <= frame_index < self.total_frames):
            logger.warning(f"Cannot seek to frame {frame_index}: out of range")
            return False
            
        # Get the frame
        frame = self.get_frame(frame_index)
        if frame is not None:
            self.current_frame_index = frame_index
            self.current_frame = frame
            return True
            
        return False
        
    def step_frames(self, delta: int) -> bool:
        """
        Step forward or backward by delta frames.
        
        Args:
            delta: Number of frames to step (positive for forward, negative for backward)
            
        Returns:
            True if successful, False otherwise
        """
        new_index = self.current_frame_index + delta
        return self.seek_to_frame(new_index)
        
    def get_video_info(self) -> Dict[str, Any]:
        """
        Get comprehensive video information.
        
        Returns:
            Dictionary with video properties and current state
        """
        if not self.is_loaded():
            return {
                'loaded': False,
                'path': '',
                'error': 'No video loaded'
            }
            
        file_size_mb = 0.0
        try:
            file_size_mb = os.path.getsize(self.video_path) / (1024 * 1024)
        except (OSError, TypeError):
            pass
            
        return {
            'loaded': True,
            'path': self.video_path,
            'filename': os.path.basename(self.video_path),
            'width': self.frame_width,
            'height': self.frame_height,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'duration_seconds': self.duration_seconds,
            'file_size_mb': file_size_mb,
            'current_frame': self.current_frame_index,
            'cache_size': self.frame_cache.get_size(),
            'cache_max_size': self.frame_cache.get_max_size()
        }
        
    def get_frame_at_timestamp(self, timestamp_seconds: float) -> Optional[np.ndarray]:
        """
        Get frame at specific timestamp.
        
        Args:
            timestamp_seconds: Time in seconds
            
        Returns:
            Frame array if successful, None otherwise
        """
        if not self.is_loaded() or self.fps <= 0:
            return None
            
        frame_index = int(timestamp_seconds * self.fps)
        return self.get_frame(frame_index)
        
    def get_timestamp_for_frame(self, frame_index: int) -> float:
        """
        Get timestamp for frame index.
        
        Args:
            frame_index: Frame index
            
        Returns:
            Timestamp in seconds
        """
        if self.fps <= 0:
            return 0.0
        return frame_index / self.fps
        
    def validate_frame_range(self, start_frame: int, end_frame: int) -> bool:
        """
        Validate frame range for analysis.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index
            
        Returns:
            True if valid range, False otherwise
        """
        if not self.is_loaded():
            return False
            
        return (0 <= start_frame < self.total_frames and 
                0 <= end_frame < self.total_frames and 
                start_frame <= end_frame)