"""Application constants for Brightness Sorcerer."""

# --- UI Styling ---
DEFAULT_FONT_FAMILY = "Segoe UI, Arial, sans-serif"

# Modern color palette with improved contrast and visual appeal
COLOR_BACKGROUND = "#1a1a1a"          # Deeper, richer dark background
COLOR_FOREGROUND = "#e0e0e0"          # Softer white for better readability
COLOR_ACCENT = "#4f9cf9"              # Modern blue with better saturation
COLOR_ACCENT_HOVER = "#6bb3ff"        # Brighter blue for interactions
COLOR_SECONDARY = "#2d2d2d"           # Elevated surface color
COLOR_SECONDARY_LIGHT = "#404040"     # Lighter surface for borders/dividers
COLOR_SUCCESS = "#10b981"             # Modern emerald green
COLOR_WARNING = "#f59e0b"             # Warm amber
COLOR_ERROR = "#ef4444"               # Modern red with better contrast
COLOR_INFO = "#06b6d4"                # Cyan for info states
COLOR_BRIGHTNESS_LABEL = "#fbbf24"    # Golden yellow for brightness display

# --- ROI (Region of Interest) Settings ---
ROI_COLORS = [
    (255, 50, 50), (50, 200, 50), (50, 150, 255), (255, 150, 50),
    (255, 50, 255), (50, 200, 200), (150, 50, 255), (255, 255, 50)
]
ROI_THICKNESS_DEFAULT = 2
ROI_THICKNESS_SELECTED = 4
ROI_LABEL_FONT_SCALE = 0.8
ROI_LABEL_THICKNESS = 2

# --- Analysis Settings ---
AUTO_DETECT_BASELINE_PERCENTILE = 5
BRIGHTNESS_NOISE_FLOOR_PERCENTILE = 2
DEFAULT_MANUAL_THRESHOLD = 5.0
MORPHOLOGICAL_KERNEL_SIZE = 3

# --- Mouse Interaction ---
MOUSE_RESIZE_HANDLE_SENSITIVITY = 10

# --- Application Settings ---
DEFAULT_SETTINGS_FILE = "brightness_analyzer_settings.json"
MAX_RECENT_FILES = 10
FRAME_CACHE_SIZE = 100
JUMP_FRAMES = 10  # Number of frames to jump with Page Up/Down

# --- Video Formats ---
SUPPORTED_VIDEO_FORMATS = [
    "Video Files (*.mp4 *.mov *.avi *.mkv *.wmv *.m4v *.flv)"
]