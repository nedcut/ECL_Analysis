"""Frame caching utilities."""

from collections import OrderedDict
from typing import Optional

import numpy as np

from .constants import FRAME_CACHE_SIZE


class FrameCache:
    """Efficient frame caching system for better performance."""

    def __init__(self, max_size: int = FRAME_CACHE_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame_index: int, copy: bool = True) -> Optional[np.ndarray]:
        """Get frame from cache, moving it to end (most recently used).

        Pass ``copy=False`` to receive the cached array itself; the caller must
        then treat it as read-only so the cached frame is not corrupted.
        """
        if frame_index in self._cache:
            self._cache.move_to_end(frame_index)
            frame = self._cache[frame_index]
            return frame.copy() if copy else frame
        return None

    def put(self, frame_index: int, frame: np.ndarray, copy: bool = True):
        """Add frame to cache, removing oldest if necessary.

        Pass ``copy=False`` to store ``frame`` without a defensive copy; the
        caller must not mutate it afterwards.
        """
        if frame_index in self._cache:
            self._cache.pop(frame_index)

        self._cache[frame_index] = frame.copy() if copy else frame

        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self):
        """Clear all cached frames."""
        self._cache.clear()

    def get_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
