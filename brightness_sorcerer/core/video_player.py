"""Video playback and frame management module."""

import logging
from typing import Optional

import cv2
import numpy as np

from .frame_cache import FrameCache


class VideoPlayer:
    """Manages video file loading, frame navigation, and playback."""

    def __init__(self, cache_size: int = 100):
        """
        Initialize video player.

        Args:
            cache_size: Number of frames to cache for performance
        """
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_cache = FrameCache(max_size=cache_size)

        self.total_frames: int = 0
        self.current_frame_index: int = 0
        self.fps: float = 30.0
        self.frame_width: int = 0
        self.frame_height: int = 0

        # Playback state
        self.is_playing: bool = False
        self.playback_speed: float = 1.0

    def load_video(self, video_path: str) -> bool:
        """
        Load video from file path.

        Args:
            video_path: Path to video file

        Returns:
            True if video loaded successfully, False otherwise
        """
        # Release any existing video
        self.release()

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            self.release()
            return False

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.fps <= 0:
            self.fps = 30.0  # Default fallback

        if self.total_frames <= 0:
            logging.error("Video file appears to have no frames")
            self.release()
            return False

        # Clear cache
        self.frame_cache.clear()

        # Reset to first frame
        self.current_frame_index = 0

        logging.info(
            f"Loaded video: {video_path} "
            f"({self.total_frames} frames, {self.fps:.2f} fps, "
            f"{self.frame_width}x{self.frame_height})"
        )

        return True

    def release(self) -> None:
        """Release video capture and clear cache."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.video_path = None
        self.total_frames = 0
        self.current_frame_index = 0
        self.frame_cache.clear()

    def is_loaded(self) -> bool:
        """Check if a video is currently loaded."""
        return self.cap is not None and self.cap.isOpened()

    def get_frame(self, frame_index: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get frame at specified index (or current frame if None).

        Args:
            frame_index: Frame index to retrieve, or None for current frame

        Returns:
            Frame as numpy array, or None if failed
        """
        if not self.is_loaded():
            return None

        if frame_index is None:
            frame_index = self.current_frame_index

        if frame_index < 0 or frame_index >= self.total_frames:
            logging.warning(f"Invalid frame index: {frame_index}")
            return None

        # Check cache first
        cached_frame = self.frame_cache.get(frame_index)
        if cached_frame is not None:
            return cached_frame

        # Read from video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()

        if ret and frame is not None:
            # Cache the frame
            self.frame_cache.put(frame_index, frame)
            return frame
        else:
            logging.warning(f"Failed to read frame at index {frame_index}")
            return None

    def seek_to_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Seek to specific frame and update current position.

        Args:
            frame_index: Frame index to seek to

        Returns:
            Frame at the specified index, or None if failed
        """
        if not self.is_loaded():
            return None

        if frame_index < 0 or frame_index >= self.total_frames:
            logging.warning(f"Attempted to seek to invalid frame index {frame_index}")
            return None

        frame = self.get_frame(frame_index)
        if frame is not None:
            self.current_frame_index = frame_index

        return frame

    def step_forward(self, num_frames: int = 1) -> Optional[np.ndarray]:
        """
        Step forward by specified number of frames.

        Args:
            num_frames: Number of frames to step forward

        Returns:
            Frame after stepping, or None if failed
        """
        new_index = min(self.current_frame_index + num_frames, self.total_frames - 1)
        return self.seek_to_frame(new_index)

    def step_backward(self, num_frames: int = 1) -> Optional[np.ndarray]:
        """
        Step backward by specified number of frames.

        Args:
            num_frames: Number of frames to step backward

        Returns:
            Frame after stepping, or None if failed
        """
        new_index = max(self.current_frame_index - num_frames, 0)
        return self.seek_to_frame(new_index)

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame.

        Returns:
            Current frame, or None if no video loaded
        """
        return self.get_frame(self.current_frame_index)

    def get_frame_shape(self) -> tuple:
        """
        Get frame shape (height, width).

        Returns:
            (height, width) tuple
        """
        return (self.frame_height, self.frame_width)

    def get_duration_seconds(self) -> float:
        """
        Get video duration in seconds.

        Returns:
            Duration in seconds
        """
        if self.fps > 0:
            return self.total_frames / self.fps
        return 0.0

    def get_current_time_seconds(self) -> float:
        """
        Get current playback time in seconds.

        Returns:
            Current time in seconds
        """
        if self.fps > 0:
            return self.current_frame_index / self.fps
        return 0.0

    def frame_to_time(self, frame_index: int) -> float:
        """
        Convert frame index to time in seconds.

        Args:
            frame_index: Frame index

        Returns:
            Time in seconds
        """
        if self.fps > 0:
            return frame_index / self.fps
        return 0.0

    def time_to_frame(self, time_seconds: float) -> int:
        """
        Convert time in seconds to frame index.

        Args:
            time_seconds: Time in seconds

        Returns:
            Frame index
        """
        return int(time_seconds * self.fps)

    def get_cache_info(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache information
        """
        return {
            "size": self.frame_cache.size(),
            "max_size": self.frame_cache.max_size,
            "hit_rate": self.frame_cache.get_hit_rate()
        }

    def set_playback_speed(self, speed: float) -> None:
        """
        Set playback speed multiplier.

        Args:
            speed: Speed multiplier (1.0 = normal speed)
        """
        self.playback_speed = max(0.1, min(speed, 10.0))

    def get_video_info(self) -> dict:
        """
        Get video information dictionary.

        Returns:
            Dictionary with video properties
        """
        return {
            "path": self.video_path,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame_index,
            "fps": self.fps,
            "width": self.frame_width,
            "height": self.frame_height,
            "duration_seconds": self.get_duration_seconds(),
            "is_playing": self.is_playing,
            "playback_speed": self.playback_speed
        }