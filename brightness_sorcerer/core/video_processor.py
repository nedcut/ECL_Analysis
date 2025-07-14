"""Video processing and frame management for Brightness Sorcerer."""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from collections import OrderedDict

# Constants
FRAME_CACHE_SIZE = 100

class FrameCache:
    """Efficient frame caching system for better performance."""
    
    def __init__(self, max_size: int = FRAME_CACHE_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
    
    def get(self, frame_index: int) -> Optional[np.ndarray]:
        """Get frame from cache, moving it to end (most recently used)."""
        if frame_index in self._cache:
            # Move to end (most recently used)
            frame = self._cache.pop(frame_index)
            self._cache[frame_index] = frame
            return frame.copy()  # Return copy to prevent modifications
        return None
    
    def put(self, frame_index: int, frame: np.ndarray):
        """Add frame to cache, removing oldest if necessary."""
        if frame_index in self._cache:
            self._cache.pop(frame_index)
        
        self._cache[frame_index] = frame.copy()
        
        # Remove oldest items if cache is full
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
    
    def clear(self):
        """Clear all cached frames."""
        self._cache.clear()
    
    def get_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class VideoProcessor:
    """Handles video loading, frame caching, and navigation."""
    
    def __init__(self, cache_size: int = FRAME_CACHE_SIZE):
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 0.0
        self.frame_size: Tuple[int, int] = (0, 0)
        self.current_frame_index: int = 0
        
        # Frame caching
        self.frame_cache = FrameCache(cache_size)
        
        # Current frame
        self.current_frame: Optional[np.ndarray] = None
    
    def load_video(self, video_path: str) -> bool:
        """Load a video file and initialize the video capture."""
        try:
            # Release existing capture if any
            self.release()
            
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                logging.error(f"Failed to open video: {video_path}")
                return False
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_size = (width, height)
            
            # Clear cache and reset position
            self.frame_cache.clear()
            self.current_frame_index = 0
            
            # Load first frame
            success = self.seek_to_frame(0)
            if not success:
                logging.error("Failed to read first frame")
                return False
                
            logging.info(f"Video loaded: {video_path} ({self.total_frames} frames, {self.fps:.2f} fps)")
            return True
            
        except Exception as e:
            logging.error(f"Error loading video {video_path}: {e}")
            return False
    
    def seek_to_frame(self, frame_index: int) -> bool:
        """Navigate to a specific frame."""
        if not self.cap or frame_index < 0 or frame_index >= self.total_frames:
            return False
        
        # Check cache first
        cached_frame = self.frame_cache.get(frame_index)
        if cached_frame is not None:
            self.current_frame = cached_frame
            self.current_frame_index = frame_index
            return True
        
        # Seek to frame in video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            self.current_frame = frame
            self.current_frame_index = frame_index
            
            # Cache the frame
            self.frame_cache.put(frame_index, frame)
            return True
        
        return False
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame."""
        return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_frame_at_index(self, frame_index: int) -> Optional[np.ndarray]:
        """Get frame at specific index without changing current position."""
        if not self.cap or frame_index < 0 or frame_index >= self.total_frames:
            return None
        
        # Check cache first
        cached_frame = self.frame_cache.get(frame_index)
        if cached_frame is not None:
            return cached_frame.copy()
        
        # Store current position
        current_pos = self.current_frame_index
        
        # Seek to requested frame
        if self.seek_to_frame(frame_index):
            frame = self.current_frame.copy()
            # Restore original position
            self.seek_to_frame(current_pos)
            return frame
        
        return None
    
    def step_frames(self, step: int) -> bool:
        """Move forward or backward by specified number of frames."""
        new_index = self.current_frame_index + step
        new_index = max(0, min(new_index, self.total_frames - 1))
        return self.seek_to_frame(new_index)
    
    def get_video_info(self) -> dict:
        """Get comprehensive video information."""
        if not self.cap:
            return {}
        
        return {
            'path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0,
            'frame_size': self.frame_size,
            'current_frame': self.current_frame_index,
            'cache_size': self.frame_cache.get_size()
        }
    
    def is_loaded(self) -> bool:
        """Check if a video is currently loaded."""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release video capture and clear cache."""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.frame_cache.clear()
        self.current_frame = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = 0.0
        self.frame_size = (0, 0)
        self.video_path = None
    
    def cleanup(self):
        """Alias for release() method."""
        self.release()
    
    def __del__(self):
        """Ensure proper cleanup."""
        self.release()