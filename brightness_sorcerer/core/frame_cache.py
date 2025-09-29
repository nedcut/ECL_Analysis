"""Frame caching system for efficient video frame access."""

from collections import OrderedDict
from typing import Optional

import numpy as np

from ..constants import FRAME_CACHE_SIZE


class FrameCache:
    """Efficient frame caching system for better performance using LRU strategy."""

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
        Get frame from cache, moving it to end (most recently used).

        Args:
            frame_index: Frame index to retrieve

        Returns:
            Frame data if cached, None otherwise
        """
        if frame_index in self._cache:
            # Move to end (most recently used)
            frame = self._cache.pop(frame_index)
            self._cache[frame_index] = frame
            return frame.copy()  # Return copy to prevent modifications
        return None

    def put(self, frame_index: int, frame: np.ndarray) -> None:
        """
        Add frame to cache, removing oldest if necessary.

        Args:
            frame_index: Frame index to cache
            frame: Frame data to store
        """
        if frame_index in self._cache:
            self._cache.pop(frame_index)

        self._cache[frame_index] = frame.copy()

        # Remove oldest items if cache is full
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached frames."""
        self._cache.clear()

    def get_size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of frames currently cached
        """
        return len(self._cache)