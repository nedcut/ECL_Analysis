import logging
import numpy as np
from collections import OrderedDict

# --- Constants ---
DEFAULT_FONT_FAMILY = "Segoe UI, Arial, sans-serif"
COLOR_BACKGROUND = "#2d2d2d"
COLOR_FOREGROUND = "#cccccc"
COLOR_ACCENT = "#5a9bd5"
COLOR_ACCENT_HOVER = "#7ab3e0"
COLOR_SECONDARY = "#404040"
COLOR_SECONDARY_LIGHT = "#555555"
COLOR_SUCCESS = "#70ad47"
COLOR_WARNING = "#ed7d31"
COLOR_ERROR = "#ff0000"
COLOR_INFO = "#ffc000"
COLOR_BRIGHTNESS_LABEL = "#ffeb3b"

ROI_COLORS = [
    (255, 50, 50), (50, 200, 50), (50, 150, 255), (255, 150, 50),
    (255, 50, 255), (50, 200, 200), (150, 50, 255), (255, 255, 50)
]
ROI_THICKNESS_DEFAULT = 2
ROI_THICKNESS_SELECTED = 4
ROI_LABEL_FONT_SCALE = 0.8
ROI_LABEL_THICKNESS = 2

AUTO_DETECT_BASELINE_PERCENTILE = 5
BRIGHTNESS_NOISE_FLOOR_PERCENTILE = 2
DEFAULT_MANUAL_THRESHOLD = 5.0
MORPHOLOGICAL_KERNEL_SIZE = 3

MOUSE_RESIZE_HANDLE_SENSITIVITY = 10

# New constants for improvements
DEFAULT_SETTINGS_FILE = "brightness_analyzer_settings.json"
MAX_RECENT_FILES = 10
FRAME_CACHE_SIZE = 100
JUMP_FRAMES = 10  # Number of frames to jump with Page Up/Down

class FrameCache:
    """Efficient frame caching system for better performance."""

    def __init__(self, max_size: int = FRAME_CACHE_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame_index: int):
        if frame_index in self._cache:
            frame = self._cache.pop(frame_index)
            self._cache[frame_index] = frame
            return frame.copy()
        return None

    def put(self, frame_index: int, frame: np.ndarray):
        if frame_index in self._cache:
            self._cache.pop(frame_index)
        self._cache[frame_index] = frame.copy()
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    def get_size(self) -> int:
        return len(self._cache)
