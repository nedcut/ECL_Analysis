"""
Application constants and configuration values.

Contains all constants used throughout the Brightness Sorcerer application,
including UI colors, default values, and application metadata.
"""

# Application metadata
APP_NAME = "Brightness Sorcerer"
APP_VERSION = "2.0.0"

# UI styling constants
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

# ROI colors for visual distinction
ROI_COLORS = [
    (255, 50, 50), (50, 200, 50), (50, 150, 255), (255, 150, 50),
    (255, 50, 255), (50, 200, 200), (150, 50, 255), (255, 255, 50)
]

# ROI display settings
ROI_THICKNESS_DEFAULT = 2
ROI_THICKNESS_SELECTED = 4
ROI_HANDLE_SIZE = 8
ROI_MIN_SIZE = 10

# Video processing constants
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.m4v', '.flv')
FRAME_CACHE_SIZE = 100
JUMP_FRAMES = 10
MAX_RECENT_FILES = 10

# Analysis parameters
DEFAULT_MANUAL_THRESHOLD = 5.0
AUTO_DETECT_BASELINE_PERCENTILE = 5
BRIGHTNESS_NOISE_FLOOR_PERCENTILE = 2

# Low-light enhancement parameters
LOW_LIGHT_BILATERAL_D = 9
LOW_LIGHT_BILATERAL_SIGMA_COLOR = 75
LOW_LIGHT_BILATERAL_SIGMA_SPACE = 75
LOW_LIGHT_CHANNEL_BOOST_FACTOR = 1.2
LOW_LIGHT_SIGNAL_AMPLIFICATION_FACTOR = 1.5

# Audio processing constants
AUDIO_BEEP_FREQUENCY_RANGE = (800, 1200)  # Hz
AUDIO_BEEP_MIN_DURATION = 0.1  # seconds
AUDIO_BEEP_MIN_AMPLITUDE = 0.1  # normalized
AUDIO_SAMPLE_RATE = 44100  # Hz

# File and settings
DEFAULT_SETTINGS_FILE = "config/brightness_analyzer_settings.json"
DEFAULT_LOG_FILE = "brightness_analyzer.log"

# Performance settings
MAX_CACHE_MEMORY_MB = 512
UPDATE_INTERVAL_MS = 33  # ~30 FPS
PROGRESS_UPDATE_INTERVAL = 10  # frames

# Plot settings
PLOT_DPI = 300
PLOT_FIGURE_SIZE = (12, 8)
PLOT_LINE_WIDTH = 1.5

# ROI validation constants
MIN_ROI_SIZE = (10, 10)  # minimum width, height in pixels
MAX_ROI_SIZE_RATIO = 0.8  # maximum fraction of frame size

# Window dimensions and UI layout
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 800
WINDOW_DEFAULT_WIDTH = 1400  
WINDOW_DEFAULT_HEIGHT = 900
WINDOW_DEFAULT_X = 100
WINDOW_DEFAULT_Y = 100
VIDEO_DISPLAY_MIN_WIDTH = 640
VIDEO_DISPLAY_MIN_HEIGHT = 360
CONTROL_PANEL_MIN_WIDTH = 350
CONTROL_PANEL_MAX_WIDTH = 450

# Color space conversion constants  
LAB_L_CHANNEL_SCALE = 100.0  # L* channel scale factor
COLOR_CHANNEL_MAX = 255.0    # 8-bit color maximum
CLAHE_CLIP_LIMIT = 2.0       # CLAHE clip limit for contrast enhancement
CLAHE_TILE_SIZE = (8, 8)     # CLAHE tile grid size

# Image processing constants
BILATERAL_FILTER_D = 9       # Neighborhood diameter
BILATERAL_SIGMA_COLOR = 75   # Color sigma
BILATERAL_SIGMA_SPACE = 75   # Space sigma
NOISE_REDUCTION_DEFAULT = 0.5
GAUSSIAN_BLUR_SIGMA_DEFAULT = 0.5
SIGNAL_AMPLIFICATION_DEFAULT = 1.0
L_STAR_BOOST_DEFAULT = 1.0
BLUE_BOOST_DEFAULT = 1.0

# Statistical analysis constants
TRIMMED_MEAN_LOWER_PERCENTILE = 0.1  # Remove bottom 10%
TRIMMED_MEAN_UPPER_PERCENTILE = 0.9  # Remove top 10%
MIN_PIXELS_FOR_ROBUST_STATS = 10     # Minimum pixels for robust statistics

# Additional UI constants
ROI_LABEL_FONT_SCALE = 0.8
ROI_LABEL_THICKNESS = 2
MORPHOLOGICAL_KERNEL_SIZE = 3
MOUSE_RESIZE_HANDLE_SENSITIVITY = 10

# File and backup settings
BACKUP_SETTINGS_FILE = "config/brightness_analyzer_settings.backup.json"

# Video validation constants
MIN_VIDEO_DURATION = 0.1  # seconds
MAX_VIDEO_DURATION = 7200  # 2 hours
MIN_FRAME_SIZE = (32, 32)  # minimum video resolution
MAX_FRAME_SIZE = (7680, 4320)  # 8K maximum