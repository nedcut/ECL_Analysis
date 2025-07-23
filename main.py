import json
import logging
import os
import sys
import time
import traceback
from collections import OrderedDict
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

# Import from new modular structure
from brightness_sorcerer.core.exceptions import (
    VideoLoadError, ValidationError
)
from brightness_sorcerer.core.brightness_analyzer import BrightnessAnalyzer
from brightness_sorcerer.core.roi_manager import ROIManager
from brightness_sorcerer.utils.constants import *
from brightness_sorcerer.utils.validation import (
    validate_video_file, validate_frame_range,
    safe_float_conversion, safe_int_conversion
)

# Version information
__version__ = "2.0.0"
__author__ = "Brightness Sorcerer Development Team"
__description__ = "Professional Video Brightness Analysis Tool"

# Optional dependencies with graceful fallbacks

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory monitoring disabled")

# Enhanced logging configuration
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup enhanced logging with file output and better formatting."""
    logger = logging.getLogger('BrightnessSorcerer')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with enhanced formatting
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except (OSError, IOError, PermissionError) as e:
            logger.warning(f"Could not setup file logging: {e}")
    
    return logger

# Setup logging
logger = setup_logging(log_file="brightness_analyzer.log")

# Constants imported from brightness_sorcerer.utils.constants

# Low-light enhancement and signal processing
class LowLightEnhancer:
    """Advanced signal processing for low-light brightness analysis."""
    
    def __init__(self):
        self.enabled = True
        self.l_star_boost = 1.0
        self.blue_boost = 1.0
        self.noise_reduction = 0.5
        self.adaptive_gain = True
        self.histogram_equalization = False
        self.gaussian_blur_sigma = 0.5
        self.bilateral_filter = True
        self.morphological_cleanup = True
        self.signal_amplification = 1.0
        self.dynamic_range_compression = False
        
    def enhance_roi(self, roi_bgr: np.ndarray, background_brightness: Optional[float] = None) -> np.ndarray:
        """Apply comprehensive low-light enhancement to ROI."""
        if roi_bgr is None or roi_bgr.size == 0:
            return roi_bgr
            
        try:
            enhanced_roi = roi_bgr.copy().astype(np.float32)
            
            # Step 1: Noise reduction using bilateral filter
            if self.bilateral_filter and self.noise_reduction > 0:
                enhanced_roi = self._apply_bilateral_filter(enhanced_roi, self.noise_reduction)
            
            # Step 2: Adaptive histogram equalization for better contrast
            if self.histogram_equalization:
                enhanced_roi = self._apply_adaptive_histogram_equalization(enhanced_roi)
            
            # Step 3: Channel-specific boosting
            enhanced_roi = self._apply_channel_boost(enhanced_roi)
            
            # Step 4: Dynamic range compression for better SNR
            if self.dynamic_range_compression:
                enhanced_roi = self._apply_dynamic_range_compression(enhanced_roi)
            
            # Step 5: Signal amplification with noise suppression
            if self.signal_amplification > 1.0:
                enhanced_roi = self._apply_signal_amplification(enhanced_roi, background_brightness)
            
            return np.clip(enhanced_roi, 0, 255).astype(np.uint8)
            
        except (cv2.error, ValueError, TypeError, MemoryError) as e:
            logger.warning(f"Low-light enhancement failed: {e}")
            return roi_bgr
        except Exception as e:
            logger.error(f"Unexpected error in low-light enhancement: {e}")
            return roi_bgr
    
    def _apply_bilateral_filter(self, roi: np.ndarray, strength: float) -> np.ndarray:
        """Apply bilateral filter for edge-preserving noise reduction."""
        d = int(9 * strength)  # Neighborhood diameter
        sigma_color = 75 * strength  # Color sigma
        sigma_space = 75 * strength  # Space sigma
        
        # Apply bilateral filter to each channel
        for i in range(3):
            roi[:, :, i] = cv2.bilateralFilter(roi[:, :, i].astype(np.uint8), d, sigma_color, sigma_space)
        
        return roi
    
    def _apply_adaptive_histogram_equalization(self, roi: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        # Convert to LAB for better perceptual results
        lab = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced.astype(np.float32)
    
    def _apply_channel_boost(self, roi: np.ndarray) -> np.ndarray:
        """Apply channel-specific boost to L* and blue channels."""
        # Convert to LAB for L* boost
        lab = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2LAB)
        
        # Boost L* channel with gamma correction for better low-light performance
        if self.l_star_boost != 1.0:
            l_channel = lab[:, :, 0].astype(np.float32) / 255.0
            # Apply gamma correction: output = input^(1/gamma)
            gamma = 1.0 / self.l_star_boost
            l_channel = np.power(l_channel, gamma)
            lab[:, :, 0] = np.clip(l_channel * 255.0, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
        
        # Boost blue channel directly
        if self.blue_boost != 1.0:
            blue_channel = enhanced[:, :, 0]  # Blue is index 0 in BGR
            # Apply power law transformation
            blue_normalized = blue_channel / 255.0
            blue_enhanced = np.power(blue_normalized, 1.0 / self.blue_boost)
            enhanced[:, :, 0] = np.clip(blue_enhanced * 255.0, 0, 255)
        
        return enhanced
    
    def _apply_dynamic_range_compression(self, roi: np.ndarray) -> np.ndarray:
        """Apply logarithmic compression to expand low-light dynamic range."""
        # Convert to LAB for perceptual processing
        lab = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2LAB)
        
        # Apply logarithmic compression to L* channel
        l_channel = lab[:, :, 0].astype(np.float32)
        l_normalized = l_channel / 255.0
        
        # Logarithmic compression: log(1 + c * x) / log(1 + c)
        c = 10.0  # Compression factor
        l_compressed = np.log(1 + c * l_normalized) / np.log(1 + c)
        
        lab[:, :, 0] = np.clip(l_compressed * 255.0, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    
    def _apply_signal_amplification(self, roi: np.ndarray, background_brightness: Optional[float]) -> np.ndarray:
        """Apply intelligent signal amplification with noise suppression."""
        if background_brightness is None:
            return roi * self.signal_amplification
        
        # Convert to LAB for processing
        lab = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32) * 100.0 / 255.0
        
        # Create adaptive amplification mask
        signal_strength = np.maximum(l_channel - background_brightness, 0)
        max_signal = np.max(signal_strength)
        
        if max_signal > 0:
            # Normalize signal strength
            signal_norm = signal_strength / max_signal
            
            # Apply adaptive amplification (stronger for weaker signals)
            amplification_factor = 1.0 + (self.signal_amplification - 1.0) * (1.0 - signal_norm)
            
            # Apply amplification to all channels
            for i in range(3):
                channel = roi[:, :, i]
                roi[:, :, i] = channel * amplification_factor
        
        return roi
    
    def compute_enhanced_brightness_stats(self, roi_bgr: np.ndarray, background_brightness: Optional[float] = None) -> Tuple[float, float, float, float, float, float, float, float]:
        """Compute brightness statistics with low-light enhancement."""
        if not self.enabled:
            return self._compute_standard_brightness_stats(roi_bgr, background_brightness)
        
        # Apply enhancement
        enhanced_roi = self.enhance_roi(roi_bgr, background_brightness)
        
        # Compute statistics on enhanced ROI
        return self._compute_standard_brightness_stats(enhanced_roi, background_brightness)
    
    def _compute_standard_brightness_stats(self, roi_bgr: np.ndarray, background_brightness: Optional[float] = None) -> Tuple[float, float, float, float, float, float, float, float]:
        """Standard brightness calculation (matches original implementation)."""
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        try:
            # Convert BGR to LAB
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)
            l_star = l_chan * 100.0 / 255.0
            
            # Extract blue channel
            blue_chan = roi_bgr[:, :, 0].astype(np.float32)
            
            # Calculate raw statistics
            l_raw_mean = float(np.mean(l_star))
            l_raw_median = float(np.median(l_star))
            b_raw_mean = float(np.mean(blue_chan))
            b_raw_median = float(np.median(blue_chan))
            
            # Calculate background-subtracted statistics
            if background_brightness is not None:
                above_background_mask = l_star > background_brightness
                
                # Apply morphological cleanup if enabled
                if self.morphological_cleanup and np.any(above_background_mask):
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask_uint8 = above_background_mask.astype(np.uint8) * 255
                    cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
                    above_background_mask = cleaned_mask > 0
                
                if np.any(above_background_mask):
                    filtered_l_pixels = l_star[above_background_mask]
                    filtered_b_pixels = blue_chan[above_background_mask]
                    
                    # Apply additional noise filtering to extracted pixels
                    if self.gaussian_blur_sigma > 0 and len(filtered_l_pixels) > 10:
                        # Use robust statistics for noisy signals
                        l_bg_sub_mean = float(np.mean(filtered_l_pixels) - background_brightness)
                        l_bg_sub_median = float(np.median(filtered_l_pixels) - background_brightness)
                        
                        # Apply Gaussian smoothing to reduce noise in statistics
                        l_values = filtered_l_pixels - background_brightness
                        if len(l_values) > 5:
                            # Use trimmed mean for better noise resistance
                            l_trimmed = np.sort(l_values)[int(len(l_values)*0.1):int(len(l_values)*0.9)]
                            if len(l_trimmed) > 0:
                                l_bg_sub_mean = float(np.mean(l_trimmed))
                    else:
                        l_bg_sub_mean = float(np.mean(filtered_l_pixels) - background_brightness)
                        l_bg_sub_median = float(np.median(filtered_l_pixels) - background_brightness)
                    
                    b_bg_sub_mean = float(np.mean(filtered_b_pixels))
                    b_bg_sub_median = float(np.median(filtered_b_pixels))
                else:
                    l_bg_sub_mean = l_bg_sub_median = b_bg_sub_mean = b_bg_sub_median = 0.0
            else:
                l_bg_sub_mean = l_raw_mean
                l_bg_sub_median = l_raw_median
                b_bg_sub_mean = b_raw_mean
                b_bg_sub_median = b_raw_median
            
            return (l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median,
                   b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median)
            
        except (cv2.error, ValueError, TypeError, IndexError, MemoryError) as e:
            logger.error(f"Error computing brightness statistics: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        except Exception as e:
            logger.critical(f"Unexpected error in brightness statistics: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

# Enhanced progress dialogs and UI components

class ProgressDialog(QtWidgets.QProgressDialog):
    """Enhanced progress dialog with better user experience."""
    
    def __init__(self, title: str, label_text: str, minimum: int = 0, maximum: int = 100, parent=None):
        super().__init__(label_text, "Cancel", minimum, maximum, parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.setMinimumDuration(500)  # Show after 500ms
        self.setAutoClose(True)
        self.setAutoReset(True)
        
        # Enhanced styling
        self.setStyleSheet("""
            QProgressDialog {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #2d2d2d;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #5a9bd5;
                border-radius: 2px;
            }
            QPushButton {
                background-color: #555555;
                border: 1px solid #777777;
                border-radius: 3px;
                padding: 5px 15px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
        """)
        
        self._start_time = time.time()
        self._last_update = 0
        
    def update_progress(self, current: int, status_text: Optional[str] = None, eta_text: Optional[str] = None):
        """Update progress with optional status and ETA."""
        self.setValue(current)
        label_parts = []
        if status_text is not None:
            label_parts.append(str(status_text))
        if eta_text is not None:
            label_parts.append(str(eta_text))
        label = " | ".join(label_parts) if label_parts else ""
        self.setLabelText(label)
        # Force GUI update every 100ms to prevent freezing
        current_time = time.time()
        if current_time - self._last_update > 0.1:
            QtWidgets.QApplication.processEvents()
            self._last_update = current_time
    
    def calculate_eta(self, current: int, total: int) -> str:
        """Calculate and format estimated time remaining."""
        if current <= 0 or total <= 0:
            return "ETA: Calculating..."
            
        elapsed = time.time() - self._start_time
        if elapsed < 1.0:  # Don't calculate ETA for first second
            return "ETA: Calculating..."
            
        progress_ratio = current / total
        if progress_ratio <= 0:
            return "ETA: Calculating..."
            
        estimated_total_time = elapsed / progress_ratio
        remaining_time = estimated_total_time - elapsed
        
        if remaining_time < 60:
            return f"ETA: {remaining_time:.0f}s"
        elif remaining_time < 3600:
            minutes = remaining_time // 60
            seconds = remaining_time % 60
            return f"ETA: {minutes:.0f}m {seconds:.0f}s"
        else:
            hours = remaining_time // 3600
            minutes = (remaining_time % 3600) // 60
            return f"ETA: {hours:.0f}h {minutes:.0f}m"

class StatusBar(QtWidgets.QStatusBar):
    """Enhanced status bar with multiple information zones."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main status label
        self.status_label = QtWidgets.QLabel("Ready")
        self.addWidget(self.status_label)
        
        # Video info label
        self.video_info_label = QtWidgets.QLabel("No video loaded")
        self.addPermanentWidget(self.video_info_label)
        
        # Memory usage label
        self.memory_label = QtWidgets.QLabel("Memory: 0 MB")
        self.addPermanentWidget(self.memory_label)
        
        # Set up timer for periodic updates
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_memory_usage)
        self.update_timer.start(5000)  # Update every 5 seconds
        
    def set_status(self, message: str, timeout: int = 0):
        """Set main status message."""
        self.status_label.setText(message)
        if timeout > 0:
            QtCore.QTimer.singleShot(timeout, lambda: self.status_label.setText("Ready"))
            
    def set_video_info(self, info: str):
        """Set video information display."""
        self.video_info_label.setText(info)
        
    def update_memory_usage(self):
        """Update memory usage display."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"Memory: {memory_mb:.0f} MB")
        except ImportError:
            self.memory_label.setText("Memory: N/A")
        except (OSError, AttributeError, ValueError) as e:
            logger.debug(f"Could not update memory usage: {e}")
            self.memory_label.setText("Memory: N/A")
        except Exception as e:
            logger.warning(f"Unexpected error updating memory usage: {e}")
            self.memory_label.setText("Memory: N/A")

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


class VideoAnalyzer(QtWidgets.QMainWindow):  # Changed to QMainWindow for better menu support
    """Main application window for video brightness analysis."""
    
    def __init__(self):
        """Initializes the application window and UI elements."""
        super().__init__()
        self._init_vars()
        self._load_settings()
        self._init_ui()
        self._create_menus()

    def __del__(self):
        """Destructor to ensure video capture resources are released."""
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"Error during VideoAnalyzer destruction: {e}")

    def _init_vars(self):
        """Initialize instance variables."""
        # Video processing (refactored architecture) - Initialize first
        try:
            from brightness_sorcerer.core.video_processor import VideoProcessor
            self.video_processor = VideoProcessor(cache_size=FRAME_CACHE_SIZE)
            self._use_video_processor = True
            logger.debug("Using refactored VideoProcessor")
        except ImportError:
            # Fallback to old architecture during transition
            self._use_video_processor = False
            logger.debug("Using legacy video processing")
        
        self.video_path = None
        self.frame = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.cap = None
        self.out_paths = []
        
        # Frame caching (only for legacy mode)
        if not self._use_video_processor:
            self.frame_cache = FrameCache(FRAME_CACHE_SIZE)
        else:
            self.frame_cache = None  # VideoProcessor has its own caching
        
        
        # Low-light enhancement system
        self.low_light_enhancer = LowLightEnhancer()
        
        # Brightness analysis engine
        self.brightness_analyzer = BrightnessAnalyzer(
            analysis_method='enhanced',
            morphological_cleanup=True,
            gaussian_blur_sigma=0.0
        )
        
        # ROI management system
        self.roi_manager = ROIManager()
        
        # Recent files
        self.recent_files = []
        
        # Legacy ROI drawing state (for UI compatibility)
        self.drawing = False
        self.moving = False
        self.resizing = False
        self.start_point = None
        self.end_point = None
        self.move_offset = None
        self.resize_corner = None
        self._current_image_size = None
        
        # Frame range
        self.start_frame = 0
        self.end_frame = None
        
        # Analysis state
        self._analysis_in_progress = False
        
        # Settings
        self.settings = {}
        
        # Threshold / background
        self.manual_threshold = DEFAULT_MANUAL_THRESHOLD
        self.background_roi_idx = None       # index into self.rects
        
        # Enhanced manual threshold controls (Research-grade)
        self.use_global_threshold = True     # Toggle between global and per-ROI thresholds
        self.roi_thresholds = {}             # Dict[roi_idx, float] - per-ROI custom thresholds
        self.threshold_profiles = {          # Predefined threshold configurations
            'conservative': {'global': 8.0, 'description': 'High threshold for noise reduction'},
            'standard': {'global': 5.0, 'description': 'Balanced threshold for general use'},
            'sensitive': {'global': 2.0, 'description': 'Low threshold for subtle changes'},
            'custom': {'global': 5.0, 'description': 'User-defined custom settings'}
        }
        self.active_threshold_profile = 'standard'
        
        # Reference mask system (Professional-grade)
        self.reference_frame_idx = None      # Frame index used for reference masks
        self.reference_masks = {}            # Dict[roi_idx, np.ndarray] - binary masks per ROI
        self.use_reference_masks = False     # Toggle reference mask mode
        self.mask_generation_method = 'threshold'  # 'threshold', 'manual', 'adaptive'
        self.reference_mask_metadata = {}   # Dict[roi_idx, dict] - mask quality metrics
        
        # Emergency override system (Research-critical)
        self.emergency_manual_mode = False   # Bypass all automation
        self.force_manual_thresholds = False # Ignore auto-detection
        self.enable_expert_mode = False      # Show advanced controls
        
        # Analysis parameter overrides
        self.brightness_calc_method = 'standard'  # 'standard', 'enhanced', 'custom'
        self.statistical_method = 'mean'     # 'mean', 'median', 'trimmed_mean'
        self.background_subtraction_method = 'roi'  # 'roi', 'manual', 'none'
        
        # Pixel visualization
        self.show_pixel_mask = False
        self.show_reference_mask = False     # Show reference mask overlay
        
        # Video playback
        self.is_playing = False
        self.playback_timer = QtCore.QTimer()
        self.playback_fps = 30.0  # Default playback FPS
        self.playback_speed = 1.0  # Playback speed multiplier
        
        # Locked ROI for best-fit analysis (migrate to ROI manager)
        self.locked_roi = None

    # === ROI Manager Compatibility Properties ===
    @property
    def rects(self):
        """Compatibility property for ROI rectangles."""
        return self.roi_manager.rects
    
    @rects.setter
    def rects(self, value):
        """Compatibility setter for ROI rectangles."""
        self.roi_manager.rects = value
    
    @property
    def selected_rect_idx(self):
        """Compatibility property for selected ROI index."""
        return self.roi_manager.selected_rect_idx
    
    @selected_rect_idx.setter
    def selected_rect_idx(self, value):
        """Compatibility setter for selected ROI index."""
        if value is not None and self.roi_manager.is_valid_roi_index(value):
            self.roi_manager.select_roi(value)
        elif value is None:
            self.roi_manager.selected_rect_idx = None
    
    @property
    def background_roi_idx(self):
        """Compatibility property for background ROI index."""
        return self.roi_manager.background_roi_idx
    
    @background_roi_idx.setter
    def background_roi_idx(self, value):
        """Compatibility setter for background ROI index."""
        if value is not None and self.roi_manager.is_valid_roi_index(value):
            self.roi_manager.set_background_roi(value)
        elif value is None:
            self.roi_manager.background_roi_idx = None
    
    @property
    def reference_masks(self):
        """Compatibility property for reference masks."""
        return self.roi_manager.reference_masks
    
    @reference_masks.setter
    def reference_masks(self, value):
        """Compatibility setter for reference masks."""
        self.roi_manager.reference_masks = value
    
    @property
    def reference_frame_idx(self):
        """Compatibility property for reference frame index."""
        return self.roi_manager.reference_frame_idx
    
    @reference_frame_idx.setter
    def reference_frame_idx(self, value):
        """Compatibility setter for reference frame index."""
        self.roi_manager.reference_frame_idx = value

    def _load_settings(self):
        """Load application settings from file with robust error handling."""
        # Initialize default settings
        default_settings = {
            'recent_files': [],
            'window_geometry': None,
            'window_state': None,
            'last_directory': os.path.expanduser('~'),
            'auto_save_results': True,
            'default_analysis_name': 'analysis',
            'frame_cache_size': FRAME_CACHE_SIZE,
            'log_level': 'INFO'
        }
        
        self.settings = default_settings.copy()
        self.recent_files = []
        
        # Try to load existing settings
        try:
            if os.path.exists(DEFAULT_SETTINGS_FILE):
                logger.debug(f"Loading settings from {DEFAULT_SETTINGS_FILE}")
                with open(DEFAULT_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    
                # Validate and merge settings
                if isinstance(loaded_settings, dict):
                    # Validate recent files
                    recent_files = loaded_settings.get('recent_files', [])
                    if isinstance(recent_files, list):
                        # Filter out invalid file paths
                        valid_recent_files = []
                        for file_path in recent_files:
                            if isinstance(file_path, str) and os.path.exists(file_path):
                                valid_recent_files.append(file_path)
                        self.recent_files = valid_recent_files[:MAX_RECENT_FILES]
                        loaded_settings['recent_files'] = self.recent_files
                    
                    # Validate numeric settings
                    
                    cache_size = safe_int_conversion(
                        loaded_settings.get('frame_cache_size', FRAME_CACHE_SIZE),
                        default=FRAME_CACHE_SIZE, min_val=10, max_val=1000
                    )
                    loaded_settings['frame_cache_size'] = cache_size
                    
                    # Merge with defaults
                    self.settings.update(loaded_settings)
                    
                    logger.info(f"Loaded settings with {len(self.recent_files)} recent files")
                else:
                    logger.warning("Settings file contains invalid data, using defaults")
                    
        except FileNotFoundError:
            logger.info("No existing settings file found, using defaults")
        except json.JSONDecodeError as e:
            logger.warning(f"Settings file contains invalid JSON: {e}, using defaults")
            # Try to create backup before overwriting
            try:
                import shutil
                shutil.copy2(DEFAULT_SETTINGS_FILE, BACKUP_SETTINGS_FILE)
                logger.info(f"Backed up corrupted settings to {BACKUP_SETTINGS_FILE}")
            except Exception as backup_error:
                logger.warning(f"Could not backup corrupted settings: {backup_error}")
        except Exception as e:
            logger.error(f"Unexpected error loading settings: {e}", exc_info=True)
        
        
        # Update frame cache size if changed (only for legacy mode)
        if not self._use_video_processor:
            try:
                cache_size = self.settings.get('frame_cache_size', FRAME_CACHE_SIZE)
                self.frame_cache = FrameCache(cache_size)
            except Exception as e:
                logger.warning(f"Could not update frame cache size: {e}")

    def _save_settings(self):
        """Save application settings to file with error handling."""
        try:
            # Update current settings
            self.settings['recent_files'] = self.recent_files
            
            # Save window geometry and state
            try:
                geometry = self.saveGeometry()
                state = self.saveState()
                self.settings['window_geometry'] = geometry.toBase64().data().decode('utf-8')
                self.settings['window_state'] = state.toBase64().data().decode('utf-8')
            except Exception as e:
                logger.debug(f"Could not save window state: {e}")
            
            # Atomic write to prevent corruption
            temp_file = DEFAULT_SETTINGS_FILE + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            
            # Replace original file
            if os.path.exists(DEFAULT_SETTINGS_FILE):
                import shutil
                shutil.copy2(DEFAULT_SETTINGS_FILE, BACKUP_SETTINGS_FILE)
            
            os.replace(temp_file, DEFAULT_SETTINGS_FILE)
            logger.debug(f"Settings saved to {DEFAULT_SETTINGS_FILE}")
            
        except PermissionError:
            logger.error(f"Permission denied writing to {DEFAULT_SETTINGS_FILE}")
            QtWidgets.QMessageBox.warning(
                self, "Settings Error", 
                f"Could not save settings to {DEFAULT_SETTINGS_FILE}\n\nPermission denied."
            )
        except Exception as e:
            logger.error(f"Could not save settings: {e}", exc_info=True)
            # Clean up temp file if it exists
            temp_file = DEFAULT_SETTINGS_FILE + '.tmp'
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    def _add_recent_file(self, file_path: str):
        """Add file to recent files list."""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        self.recent_files = self.recent_files[:MAX_RECENT_FILES]
        self._update_recent_files_menu()

    def _init_ui(self):
        """Set up the main UI layout and widgets."""
        self.setWindowTitle('Brightness Sorcerer v2.0')
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)  # Minimum size to prevent UI overflow
        self.setGeometry(WINDOW_DEFAULT_X, WINDOW_DEFAULT_Y, WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)  # Default size
        self.setAcceptDrops(True)
        self._apply_stylesheet()

        # Create central widget with responsive splitter layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal splitter for responsive layout
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        self._create_layouts()
        self._create_widgets()
        self._connect_signals()
        self._setup_shortcuts()
        self._update_widget_states()
        self._update_ref_mask_status()  # Initialize reference mask status display

    def _create_menus(self):
        """Create application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        open_action = QtWidgets.QAction('&Open Video...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open a video file')
        open_action.triggered.connect(self.open_video_dialog)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Recent files submenu
        self.recent_files_menu = file_menu.addMenu('Recent Files')
        self._update_recent_files_menu()
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit the application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('&Analysis')
        
        analyze_action = QtWidgets.QAction('&Run Analysis', self)
        analyze_action.setShortcut('F5')
        analyze_action.setStatusTip('Run brightness analysis')
        analyze_action.triggered.connect(self.analyze_video)
        analysis_menu.addAction(analyze_action)
        
        
        # Settings menu
        settings_menu = menubar.addMenu('&Settings')
        
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        shortcuts_action = QtWidgets.QAction('&Keyboard Shortcuts', self)
        shortcuts_action.triggered.connect(self._show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)
        
        about_action = QtWidgets.QAction('&About', self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)
        
        # Status bar
        self.statusBar().showMessage('Ready - Load a video to begin')

    def _update_recent_files_menu(self):
        """Update the recent files menu."""
        self.recent_files_menu.clear()
        
        if not self.recent_files:
            no_recent_action = QtWidgets.QAction('No recent files', self)
            no_recent_action.setEnabled(False)
            self.recent_files_menu.addAction(no_recent_action)
            return
        
        for file_path in self.recent_files:
            if os.path.exists(file_path):
                action = QtWidgets.QAction(os.path.basename(file_path), self)
                action.setStatusTip(file_path)
                action.triggered.connect(lambda _checked, path=file_path: self._open_recent_file(path))
                self.recent_files_menu.addAction(action)

    def _open_recent_file(self, file_path: str):
        """Open a file from the recent files list."""
        if os.path.exists(file_path):
            self.video_path = file_path
            self.load_video()
        else:
            QtWidgets.QMessageBox.warning(self, 'File Not Found', 
                                        f'The file {file_path} no longer exists.')
            self.recent_files.remove(file_path)
            self._update_recent_files_menu()

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Playback shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, 
                          self.toggle_playback)
        
        # Frame navigation shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, 
                          lambda: self.step_frames(-1))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, 
                          lambda: self.step_frames(1))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self, 
                          lambda: self.step_frames(-1))
        
        # Jump navigation
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_PageDown), self, 
                          lambda: self.step_frames(JUMP_FRAMES))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_PageUp), self, 
                          lambda: self.step_frames(-JUMP_FRAMES))
        
        # Go to start/end
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Home), self, 
                          lambda: self.frame_slider.setValue(0))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_End), self, 
                          lambda: self.frame_slider.setValue(self.total_frames - 1))
        
        # ROI shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self, 
                          self.delete_selected_rectangle)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, 
                          self._cancel_current_action)

    def _cancel_current_action(self):
        """Cancel current drawing/moving/resizing action."""
        if self.drawing:
            self.add_rect_btn.setChecked(False)
            self.toggle_add_rectangle_mode(False)
        elif self.moving or self.resizing:
            self.moving = False
            self.resizing = False
            self.start_point = None
            self.end_point = None
            self.move_offset = None
            self.resize_corner = None
            self.image_label.unsetCursor()
            self.show_frame()

    def _show_shortcuts_dialog(self):
        """Show keyboard shortcuts dialog."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Keyboard Shortcuts')
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        
        shortcuts_text = """
<h3>Playback Shortcuts:</h3>
<b>Space:</b> Play/Pause video<br>

<h3>Navigation Shortcuts:</h3>
<b>Left/Right Arrow:</b> Previous/Next frame<br>
<b>Backspace:</b> Previous frame<br>
<b>Page Down/Up:</b> Jump 10 frames<br>
<b>Home/End:</b> Go to first/last frame<br>

<h3>Analysis Shortcuts:</h3>
<b>F5:</b> Run analysis<br>

<h3>ROI Shortcuts:</h3>
<b>Delete:</b> Delete selected ROI<br>
<b>Escape:</b> Cancel current action<br>

<h3>File Shortcuts:</h3>
<b>Ctrl+O:</b> Open video<br>
<b>Ctrl+Q:</b> Exit application<br>
        """
        
        label = QtWidgets.QLabel(shortcuts_text)
        label.setWordWrap(True)
        layout.addWidget(label)
        
        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()

    def _show_about_dialog(self):
        """Show about dialog."""
        QtWidgets.QMessageBox.about(self, 'About Brightness Sorcerer',
            """<h2>Brightness Sorcerer v2.0</h2>
            <p>Advanced video brightness analysis tool</p>
            <p>Analyze brightness changes in video regions of interest (ROIs) 
            with automatic detection and comprehensive plotting.</p>
            <p><b>Features:</b></p>
            <ul>
            <li>Interactive ROI selection and editing</li>
            <li>Automatic frame range detection</li>
            <li>Statistical analysis with mean and median</li>
            <li>High-quality plot generation</li>
            <li>Frame caching for smooth navigation</li>
            </ul>""")


    def _apply_stylesheet(self):
        """Apply a modern, clean stylesheet to the application."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLOR_BACKGROUND};
                color: {COLOR_FOREGROUND};
                font-family: {DEFAULT_FONT_FAMILY};
                font-size: 14px;
            }}
            QMenuBar {{
                background-color: {COLOR_SECONDARY};
                color: {COLOR_FOREGROUND};
                border-bottom: 1px solid {COLOR_SECONDARY_LIGHT};
            }}
            QMenuBar::item {{
                background: transparent;
                padding: 4px 8px;
            }}
            QMenuBar::item:selected {{
                background-color: {COLOR_ACCENT};
            }}
            QMenu {{
                background-color: {COLOR_SECONDARY};
                color: {COLOR_FOREGROUND};
                border: 1px solid {COLOR_SECONDARY_LIGHT};
            }}
            QMenu::item:selected {{
                background-color: {COLOR_ACCENT};
            }}
            QStatusBar {{
                background-color: {COLOR_SECONDARY};
                color: {COLOR_FOREGROUND};
                border-top: 1px solid {COLOR_SECONDARY_LIGHT};
            }}
            QWidget {{
                background-color: {COLOR_BACKGROUND};
                color: {COLOR_FOREGROUND};
                font-family: {DEFAULT_FONT_FAMILY};
                font-size: 14px;
            }}
            QLabel#titleLabel {{
                font-size: 24px;
                font-weight: bold;
                color: {COLOR_ACCENT};
                padding-bottom: 10px;
                qproperty-alignment: AlignCenter;
            }}
            QLabel#imageLabel {{
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                background: #1e1e1e;
                border-radius: 6px;
            }}
            QLabel#resultsLabel {{
                font-size: 13px;
                color: {COLOR_INFO};
                background: {COLOR_SECONDARY};
                border-radius: 4px;
                padding: 8px;
                border: 1px solid {COLOR_SECONDARY_LIGHT};
            }}
            QLabel#brightnessDisplayLabel {{
                font-size: 28px;
                font-weight: bold;
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                padding: 10px;
                color: {COLOR_BRIGHTNESS_LABEL};
                background: {COLOR_SECONDARY};
                border-radius: 6px;
                qproperty-alignment: AlignCenter;
            }}
            QLabel#statusLabel {{
                font-size: 12px;
                color: {COLOR_INFO};
                padding: 4px;
            }}
            QGroupBox {{
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 6px;
                margin-top: 10px;
                background: {COLOR_SECONDARY};
                font-weight: bold;
                font-size: 15px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 2px 5px;
                color: {COLOR_ACCENT};
                background-color: {COLOR_BACKGROUND};
                border-radius: 3px;
            }}
            QPushButton {{
                background-color: {COLOR_SECONDARY_LIGHT};
                color: {COLOR_FOREGROUND};
                border: 1px solid {COLOR_SECONDARY};
                border-radius: 4px;
                padding: 8px 15px;
                font-size: 14px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_ACCENT};
                color: {COLOR_BACKGROUND};
                border: 1px solid {COLOR_ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {COLOR_ACCENT_HOVER};
            }}
            QPushButton:disabled {{
                background-color: {COLOR_SECONDARY};
                color: #888888;
                border: 1px solid {COLOR_SECONDARY};
            }}
            QPushButton:checked {{
                background-color: {COLOR_ACCENT};
                color: {COLOR_BACKGROUND};
                border: 1px solid {COLOR_ACCENT_HOVER};
            }}
            QListWidget {{
                background: {COLOR_BACKGROUND};
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                color: {COLOR_FOREGROUND};
                font-size: 13px;
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background: {COLOR_ACCENT};
                color: {COLOR_BACKGROUND};
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {COLOR_SECONDARY};
                height: 6px;
                background: {COLOR_SECONDARY};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLOR_ACCENT};
                border: 1px solid {COLOR_ACCENT_HOVER};
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLOR_SUCCESS};
                border-radius: 3px;
            }}
            QSlider::add-page:horizontal {{
                background: {COLOR_SECONDARY};
                border-radius: 3px;
            }}
            QLineEdit, QSpinBox {{
                background-color: {COLOR_BACKGROUND};
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                padding: 4px;
                border-radius: 4px;
                min-height: 20px;
            }}
            QLineEdit:focus, QSpinBox:focus {{
                border: 1px solid {COLOR_ACCENT};
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                subcontrol-origin: border;
                width: 16px;
                border-left: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 2px;
            }}
            QSpinBox::up-button {{
                subcontrol-position: top right;
            }}
            QSpinBox::down-button {{
                subcontrol-position: bottom right;
            }}
            QSpinBox::up-arrow {{
                image: url(./icons/arrow_up.png); /* Requires icon files */
                width: 10px; height: 10px;
            }}
            QSpinBox::down-arrow {{
                image: url(./icons/arrow_down.png); /* Requires icon files */
                 width: 10px; height: 10px;
            }}
            QProgressDialog {{
                 font-size: 14px;
            }}
            QProgressDialog QLabel {{
                 color: {COLOR_FOREGROUND};
            }}
            QProgressBar {{
                border: 1px solid {COLOR_SECONDARY_LIGHT};
                border-radius: 4px;
                text-align: center;
                color: {COLOR_FOREGROUND};
            }}
            QProgressBar::chunk {{
                background-color: {COLOR_SUCCESS};
                border-radius: 3px;
            }}
        """
        )

    def _create_layouts(self):
        """Create the responsive video and control panels."""
        # Left panel: Video display and basic controls
        self.video_panel = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QVBoxLayout(self.video_panel)
        self.left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Right panel: Control tabs
        self.control_panel = QtWidgets.QWidget()
        self.control_panel.setMinimumWidth(CONTROL_PANEL_MIN_WIDTH)  # Prevent controls from becoming too narrow
        self.control_panel.setMaximumWidth(CONTROL_PANEL_MAX_WIDTH)  # Prevent controls from taking too much space
        self.right_layout = QtWidgets.QVBoxLayout(self.control_panel)
        self.right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add panels to splitter with appropriate sizing
        self.main_splitter.addWidget(self.video_panel)
        self.main_splitter.addWidget(self.control_panel)
        self.main_splitter.setSizes([1000, 350])  # Initial sizes
        self.main_splitter.setStretchFactor(0, 1)  # Video panel gets extra space

    def _create_advanced_controls_hidden(self):
        """Create hidden/minimal advanced controls for compatibility with existing code."""
        # Create minimal checkbox for enhanced preview (hidden/disabled state)
        self.show_enhanced_preview_cb = QtWidgets.QCheckBox("Enhanced Preview")
        self.show_enhanced_preview_cb.setChecked(False)
        self.show_enhanced_preview_cb.setVisible(False)  # Hidden by default in new UI
        
        # These can be added to advanced settings panel later if needed
        # For now, they maintain compatibility without cluttering the UI

    def _create_widgets(self):
        """Create all the widgets and add them to layouts."""
        # === VIDEO PANEL (Left Side) ===
        self._create_video_panel_widgets()
        
        # === CONTROL PANEL (Right Side) ===
        self._create_control_panel_widgets()

    def _create_video_panel_widgets(self):
        """Create widgets for the video display panel."""
        # Header with title and file info
        header_layout = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel("Brightness Sorcerer v2.0")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #5a9bd5;")
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # File info (cleaner, right-aligned)
        self.file_info_label = QtWidgets.QLabel("No video loaded")
        self.file_info_label.setObjectName("statusLabel")
        self.file_info_label.setStyleSheet("color: #888888; font-style: italic;")
        header_layout.addWidget(self.file_info_label)
        
        self.left_layout.addLayout(header_layout)

        # Open button (more prominent)
        self.open_btn = QtWidgets.QPushButton("📁 Open Video (Ctrl+O)")    
        self.open_btn.setToolTip("Choose a video file from disk")
        self.open_btn.setMinimumHeight(35)
        self.left_layout.addWidget(self.open_btn)

        # Video display area
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image_label.setText("Drag & Drop Video File Here")
        self.image_label.setMinimumSize(VIDEO_DISPLAY_MIN_WIDTH, VIDEO_DISPLAY_MIN_HEIGHT)  # Ensure minimum video display size
        self.left_layout.addWidget(self.image_label, stretch=1)

        # Compact video controls  
        self.video_controls_groupbox = QtWidgets.QGroupBox("Video Controls")
        controls_layout = QtWidgets.QVBoxLayout()
        
        # Timeline slider and frame info (compact)
        timeline_layout = QtWidgets.QHBoxLayout()
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setToolTip("Navigate video frames")
        timeline_layout.addWidget(self.frame_slider)
        
        # Compact frame info
        self.frame_label = QtWidgets.QLabel("0/0")
        self.frame_label.setMinimumWidth(60)
        self.frame_label.setAlignment(QtCore.Qt.AlignCenter)
        timeline_layout.addWidget(self.frame_label)
        
        self.frame_spinbox = QtWidgets.QSpinBox()
        self.frame_spinbox.setMaximumWidth(80)
        self.frame_spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        timeline_layout.addWidget(self.frame_spinbox)
        controls_layout.addLayout(timeline_layout)
        
        # Compact control buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.prev_frame_btn = QtWidgets.QPushButton("◀")
        self.prev_frame_btn.setFixedSize(30, 25)
        self.next_frame_btn = QtWidgets.QPushButton("▶")
        self.next_frame_btn.setFixedSize(30, 25)
        self.play_pause_btn = QtWidgets.QPushButton("▶")
        self.play_pause_btn.setFixedSize(40, 25)
        
        self.jump_back_btn = QtWidgets.QPushButton(f"◀◀")
        self.jump_back_btn.setFixedSize(35, 25)
        self.jump_forward_btn = QtWidgets.QPushButton(f"▶▶")
        self.jump_forward_btn.setFixedSize(35, 25)
        
        button_layout.addWidget(self.prev_frame_btn)
        button_layout.addWidget(self.play_pause_btn)
        button_layout.addWidget(self.next_frame_btn)
        button_layout.addWidget(QtWidgets.QLabel("|"))
        button_layout.addWidget(self.jump_back_btn)
        button_layout.addWidget(self.jump_forward_btn)
        
        # Speed control
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.setMaximumWidth(60)
        button_layout.addWidget(self.speed_combo)
        button_layout.addStretch()
        
        controls_layout.addLayout(button_layout)
        
        # Range controls (compact)
        range_layout = QtWidgets.QHBoxLayout()
        self.set_start_btn = QtWidgets.QPushButton("Set Start")
        self.set_end_btn = QtWidgets.QPushButton("Set End")
        self.auto_detect_btn = QtWidgets.QPushButton("Auto-Detect")
        
        range_layout.addWidget(self.set_start_btn)
        range_layout.addWidget(self.set_end_btn)
        range_layout.addWidget(self.auto_detect_btn)
        controls_layout.addLayout(range_layout)
        
        self.video_controls_groupbox.setLayout(controls_layout)
        self.left_layout.addWidget(self.video_controls_groupbox)
        
        # Analysis name and action button
        action_layout = QtWidgets.QVBoxLayout()
        
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Analysis Name:"))
        self.analysis_name_input = QtWidgets.QLineEdit()
        self.analysis_name_input.setPlaceholderText("DefaultAnalysis")
        name_layout.addWidget(self.analysis_name_input)
        action_layout.addLayout(name_layout)
        
        # Main analyze button (prominent)
        self.analyze_btn = QtWidgets.QPushButton('🔍 Analyze Brightness (F5)')
        self.analyze_btn.setToolTip("Run brightness analysis on the selected frame range and ROIs")
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.setStyleSheet("QPushButton { background-color: #5a9bd5; color: white; font-weight: bold; }")
        action_layout.addWidget(self.analyze_btn)
        
        self.left_layout.addLayout(action_layout)

    def _create_control_panel_widgets(self):
        """Create widgets for the control panel (simplified for now)."""
        # Create a scrollable widget for all controls
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        
        # === Video Information (cleaned up) ===
        self.video_info_groupbox = QtWidgets.QGroupBox("📹 Video Info")
        video_info_layout = QtWidgets.QVBoxLayout()
        self.video_info_label = QtWidgets.QLabel("No video loaded")
        self.video_info_label.setWordWrap(True)
        self.video_info_label.setStyleSheet("font-size: 11px; padding: 5px;")
        video_info_layout.addWidget(self.video_info_label)
        self.video_info_groupbox.setLayout(video_info_layout)
        scroll_layout.addWidget(self.video_info_groupbox)
        
        # === Current Frame Brightness (simplified) ===
        self.brightness_groupbox = QtWidgets.QGroupBox("📊 Current Frame")
        brightness_layout = QtWidgets.QVBoxLayout()
        self.brightness_display_label = QtWidgets.QLabel("N/A")
        self.brightness_display_label.setObjectName("brightnessDisplayLabel")
        self.brightness_display_label.setWordWrap(True)
        self.brightness_display_label.setStyleSheet("font-size: 11px; padding: 5px;")
        brightness_layout.addWidget(self.brightness_display_label)
        self.brightness_groupbox.setLayout(brightness_layout)
        scroll_layout.addWidget(self.brightness_groupbox)
        
        # === ROI Controls (simplified) ===
        self.rect_groupbox = QtWidgets.QGroupBox("🎯 ROI Controls")
        rect_layout = QtWidgets.QVBoxLayout()
        
        # ROI list (smaller)
        self.rect_list = QtWidgets.QListWidget()
        self.rect_list.setMaximumHeight(100)
        self.rect_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        rect_layout.addWidget(self.rect_list)
        
        # ROI buttons (compact)
        roi_btn_layout1 = QtWidgets.QHBoxLayout()
        self.add_rect_btn = QtWidgets.QPushButton("Add")
        self.add_rect_btn.setCheckable(True)
        self.del_rect_btn = QtWidgets.QPushButton("Delete")
        roi_btn_layout1.addWidget(self.add_rect_btn)
        roi_btn_layout1.addWidget(self.del_rect_btn)
        rect_layout.addLayout(roi_btn_layout1)
        
        roi_btn_layout2 = QtWidgets.QHBoxLayout()
        self.clear_rect_btn = QtWidgets.QPushButton("Clear All")
        self.find_lock_btn = QtWidgets.QPushButton("Find Best")
        roi_btn_layout2.addWidget(self.clear_rect_btn)
        roi_btn_layout2.addWidget(self.find_lock_btn)
        rect_layout.addLayout(roi_btn_layout2)
        
        self.rect_groupbox.setLayout(rect_layout)
        scroll_layout.addWidget(self.rect_groupbox)
        
        # === Basic Settings ===
        self.threshold_groupbox = QtWidgets.QGroupBox("⚙️ Settings")
        settings_layout = QtWidgets.QVBoxLayout()
        
        # Manual threshold (simplified)
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_layout.addWidget(QtWidgets.QLabel("Threshold:"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.setRange(0.0, 100.0)
        self.threshold_spin.setValue(self.manual_threshold)
        self.threshold_spin.setMaximumWidth(80)
        threshold_layout.addWidget(self.threshold_spin)
        settings_layout.addLayout(threshold_layout)
        
        # Background ROI button
        self.set_bg_btn = QtWidgets.QPushButton("Set Background ROI")
        settings_layout.addWidget(self.set_bg_btn)
        
        # Run duration (simplified)
        duration_layout = QtWidgets.QHBoxLayout()
        duration_layout.addWidget(QtWidgets.QLabel("Duration (s):"))
        self.run_duration_spin = QtWidgets.QDoubleSpinBox()
        self.run_duration_spin.setDecimals(1)
        self.run_duration_spin.setRange(0.0, 3600.0)
        self.run_duration_spin.setValue(0.0)
        self.run_duration_spin.setMaximumWidth(80)
        duration_layout.addWidget(self.run_duration_spin)
        settings_layout.addLayout(duration_layout)
        
        self.threshold_groupbox.setLayout(settings_layout)
        scroll_layout.addWidget(self.threshold_groupbox)
        
        # === Status and Results ===
        results_groupbox = QtWidgets.QGroupBox("📋 Status")
        results_layout = QtWidgets.QVBoxLayout()
        
        self.cache_status_label = QtWidgets.QLabel("Cache: 0 frames")
        self.cache_status_label.setStyleSheet("font-size: 10px; color: #888888;")
        results_layout.addWidget(self.cache_status_label)
        
        self.results_label = QtWidgets.QLabel("Load a video to begin analysis.")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("font-size: 11px; padding: 5px;")
        results_layout.addWidget(self.results_label)
        
        results_groupbox.setLayout(results_layout)
        scroll_layout.addWidget(results_groupbox)
        
        # Add stretch to push everything to top
        scroll_layout.addStretch()
        
        # Set scroll widget and add to layout
        scroll_area.setWidget(scroll_widget)
        self.right_layout.addWidget(scroll_area)
        
        # Create simplified hidden advanced controls (for compatibility)
        self._create_advanced_controls_hidden()


        # -- Threshold groupbox
        self.threshold_groupbox = QtWidgets.QGroupBox("Threshold Settings")
        th_layout = QtWidgets.QVBoxLayout()
        
        # Manual threshold controls
        manual_layout = QtWidgets.QHBoxLayout()
        manual_layout.addWidget(QtWidgets.QLabel("Manual ΔL*:"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.setRange(0.0, 100.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setValue(self.manual_threshold)
        manual_layout.addWidget(self.threshold_spin)
        th_layout.addLayout(manual_layout)
        
        # Background ROI controls
        bg_layout = QtWidgets.QHBoxLayout()
        self.set_bg_btn = QtWidgets.QPushButton("Set Selected ROI as Background")
        bg_layout.addWidget(self.set_bg_btn)
        th_layout.addLayout(bg_layout)
        
        # Current threshold display
        self.threshold_display_label = QtWidgets.QLabel("Active Threshold: Manual (5.0 L*)")
        self.threshold_display_label.setStyleSheet("color: #ffc000; font-weight: bold; padding: 4px;")
        th_layout.addWidget(self.threshold_display_label)
        
        self.threshold_groupbox.setLayout(th_layout)
        self.right_layout.addWidget(self.threshold_groupbox)

        # Professional Reference Mask Controls
        self.ref_mask_groupbox = QtWidgets.QGroupBox("Reference Mask System (Professional)")
        ref_mask_layout = QtWidgets.QVBoxLayout()
        
        # Reference frame controls
        ref_frame_layout = QtWidgets.QHBoxLayout()
        self.set_ref_frame_btn = QtWidgets.QPushButton("Set Reference Frame")
        self.set_ref_frame_btn.setToolTip("Use current frame to generate reference masks for consistent analysis")
        ref_frame_layout.addWidget(self.set_ref_frame_btn)
        
        self.clear_ref_masks_btn = QtWidgets.QPushButton("Clear")
        self.clear_ref_masks_btn.setToolTip("Clear all reference masks")
        ref_frame_layout.addWidget(self.clear_ref_masks_btn)
        ref_mask_layout.addLayout(ref_frame_layout)
        
        # Reference mask status display
        self.ref_mask_status_label = QtWidgets.QLabel("No reference frame set")
        self.ref_mask_status_label.setStyleSheet("color: #888888; font-style: italic; padding: 2px;")
        ref_mask_layout.addWidget(self.ref_mask_status_label)
        
        # Reference mask mode toggle
        self.use_ref_masks_checkbox = QtWidgets.QCheckBox("Use Reference Mask Analysis")
        self.use_ref_masks_checkbox.setToolTip("Apply reference masks for consistent pixel analysis across all frames")
        self.use_ref_masks_checkbox.setChecked(self.use_reference_masks)
        ref_mask_layout.addWidget(self.use_ref_masks_checkbox)
        
        # Mask generation method selection
        mask_method_layout = QtWidgets.QHBoxLayout()
        mask_method_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.mask_method_combo = QtWidgets.QComboBox()
        self.mask_method_combo.addItems(['threshold', 'adaptive'])
        self.mask_method_combo.setCurrentText(self.mask_generation_method)
        self.mask_method_combo.setToolTip("threshold: Simple threshold-based masking\nadaptive: Otsu-based adaptive masking")
        mask_method_layout.addWidget(self.mask_method_combo)
        ref_mask_layout.addLayout(mask_method_layout)
        
        self.ref_mask_groupbox.setLayout(ref_mask_layout)
        self.right_layout.addWidget(self.ref_mask_groupbox)

        # Advanced Manual Controls (Expert Mode)
        self.expert_groupbox = QtWidgets.QGroupBox("Expert Controls")
        expert_layout = QtWidgets.QVBoxLayout()
        
        # Expert mode toggle
        self.expert_mode_checkbox = QtWidgets.QCheckBox("Enable Expert Mode")
        self.expert_mode_checkbox.setToolTip("Show advanced manual override controls for research-grade precision")
        self.expert_mode_checkbox.setChecked(self.enable_expert_mode)
        expert_layout.addWidget(self.expert_mode_checkbox)
        
        # Emergency manual mode controls (initially hidden)
        self.emergency_controls_widget = QtWidgets.QWidget()
        emergency_layout = QtWidgets.QVBoxLayout(self.emergency_controls_widget)
        emergency_layout.setContentsMargins(0, 0, 0, 0)
        
        # Emergency override toggles
        self.emergency_manual_checkbox = QtWidgets.QCheckBox("Emergency Manual Mode")
        self.emergency_manual_checkbox.setToolTip("Bypass ALL automation - pure manual control")
        self.emergency_manual_checkbox.setChecked(self.emergency_manual_mode)
        self.emergency_manual_checkbox.setStyleSheet("QCheckBox { color: #ff6666; font-weight: bold; }")
        emergency_layout.addWidget(self.emergency_manual_checkbox)
        
        self.force_manual_thresholds_checkbox = QtWidgets.QCheckBox("Force Manual Thresholds")
        self.force_manual_thresholds_checkbox.setToolTip("Ignore all auto-detection and background ROI calculations")
        self.force_manual_thresholds_checkbox.setChecked(self.force_manual_thresholds)
        emergency_layout.addWidget(self.force_manual_thresholds_checkbox)
        
        # Per-ROI threshold controls
        per_roi_layout = QtWidgets.QHBoxLayout()
        self.use_global_threshold_checkbox = QtWidgets.QCheckBox("Use Global Threshold")
        self.use_global_threshold_checkbox.setToolTip("When unchecked, allows per-ROI custom thresholds")
        self.use_global_threshold_checkbox.setChecked(self.use_global_threshold)
        per_roi_layout.addWidget(self.use_global_threshold_checkbox)
        emergency_layout.addLayout(per_roi_layout)
        
        # Analysis method selection
        analysis_method_layout = QtWidgets.QHBoxLayout()
        analysis_method_layout.addWidget(QtWidgets.QLabel("Analysis Method:"))
        self.analysis_method_combo = QtWidgets.QComboBox()
        self.analysis_method_combo.addItems(['standard', 'enhanced'])
        self.analysis_method_combo.setCurrentText(self.brightness_calc_method)
        self.analysis_method_combo.setToolTip("standard: Basic LAB analysis\nenhanced: Advanced with low-light optimization")
        analysis_method_layout.addWidget(self.analysis_method_combo)
        emergency_layout.addLayout(analysis_method_layout)
        
        # Statistical method selection
        stats_method_layout = QtWidgets.QHBoxLayout()
        stats_method_layout.addWidget(QtWidgets.QLabel("Statistics:"))
        self.stats_method_combo = QtWidgets.QComboBox()
        self.stats_method_combo.addItems(['mean', 'median', 'both'])
        self.stats_method_combo.setCurrentText(self.statistical_method)
        self.stats_method_combo.setToolTip("Statistical method for brightness calculation")
        stats_method_layout.addWidget(self.stats_method_combo)
        emergency_layout.addLayout(stats_method_layout)
        
        # Initially hide expert controls
        self.emergency_controls_widget.setVisible(self.enable_expert_mode)
        expert_layout.addWidget(self.emergency_controls_widget)
        
        self.expert_groupbox.setLayout(expert_layout)
        self.right_layout.addWidget(self.expert_groupbox)

        # Visualization Controls
        self.viz_groupbox = QtWidgets.QGroupBox("Visualization")
        viz_layout = QtWidgets.QVBoxLayout()
        
        self.show_mask_checkbox = QtWidgets.QCheckBox("Show Pixel Mask")
        self.show_mask_checkbox.setToolTip("Highlight analyzed pixels in red overlay")
        self.show_mask_checkbox.setChecked(self.show_pixel_mask)
        viz_layout.addWidget(self.show_mask_checkbox)
        
        self.show_ref_mask_checkbox = QtWidgets.QCheckBox("Show Reference Mask")
        self.show_ref_mask_checkbox.setToolTip("Highlight reference mask pixels in blue overlay")
        self.show_ref_mask_checkbox.setChecked(self.show_reference_mask)
        viz_layout.addWidget(self.show_ref_mask_checkbox)
        
        self.viz_groupbox.setLayout(viz_layout)
        self.right_layout.addWidget(self.viz_groupbox)

        # Rectangle Controls
        self.rect_groupbox = QtWidgets.QGroupBox("Regions of Interest (ROI)")
        rect_groupbox_layout = QtWidgets.QVBoxLayout()
        self.rect_list = QtWidgets.QListWidget()
        self.rect_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        rect_groupbox_layout.addWidget(self.rect_list)

        rect_btn_layout = QtWidgets.QHBoxLayout()
        self.add_rect_btn = QtWidgets.QPushButton("Add ROI")
        self.add_rect_btn.setCheckable(True)
        self.add_rect_btn.setToolTip("Click then draw a rectangle on the video frame")
        rect_btn_layout.addWidget(self.add_rect_btn)

        self.find_lock_btn = QtWidgets.QPushButton("Find & Lock Best ROI")
        self.find_lock_btn.setToolTip("Find the brightest frame for the selected ROI and lock it for analysis")
        rect_btn_layout.addWidget(self.find_lock_btn)

        self.del_rect_btn = QtWidgets.QPushButton("Delete ROI")
        self.del_rect_btn.setToolTip("Delete the selected ROI from the list (Delete key)")
        rect_btn_layout.addWidget(self.del_rect_btn)

        self.clear_rect_btn = QtWidgets.QPushButton("Clear All")
        self.clear_rect_btn.setToolTip("Remove all ROIs")
        rect_btn_layout.addWidget(self.clear_rect_btn)
        rect_groupbox_layout.addLayout(rect_btn_layout)
        self.rect_groupbox.setLayout(rect_groupbox_layout)
        self.right_layout.addWidget(self.rect_groupbox)

        # Cache status
        self.cache_status_label = QtWidgets.QLabel("Cache: 0 frames")
        self.cache_status_label.setObjectName("statusLabel")
        self.right_layout.addWidget(self.cache_status_label)

        # Results/Status Label
        self.results_label = QtWidgets.QLabel("Load a video to begin analysis.")
        self.results_label.setObjectName("resultsLabel")
        self.results_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.results_label.setWordWrap(True)
        self.results_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.right_layout.addWidget(self.results_label, stretch=1)

    def _connect_signals(self):
        """Connect widget signals to their corresponding slots."""
        self.open_btn.clicked.connect(self.open_video_dialog)

        self.frame_slider.valueChanged.connect(self.slider_frame_changed)
        self.frame_spinbox.valueChanged.connect(self.spinbox_frame_changed)

        self.prev_frame_btn.clicked.connect(lambda: self.step_frames(-1))
        self.next_frame_btn.clicked.connect(lambda: self.step_frames(1))
        self.jump_back_btn.clicked.connect(lambda: self.step_frames(-JUMP_FRAMES))
        self.jump_forward_btn.clicked.connect(lambda: self.step_frames(JUMP_FRAMES))
        self.set_start_btn.clicked.connect(self.set_start_frame)
        self.set_end_btn.clicked.connect(self.set_end_frame)
        
        # Playback controls
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        self.playback_timer.timeout.connect(self.advance_frame)

        self.analyze_btn.clicked.connect(self.analyze_video)

        self.rect_list.currentRowChanged.connect(self.select_rectangle_from_list)
        self.add_rect_btn.clicked.connect(self.toggle_add_rectangle_mode)
        self.del_rect_btn.clicked.connect(self.delete_selected_rectangle)
        self.clear_rect_btn.clicked.connect(self.clear_all_rectangles)
        self.find_lock_btn.clicked.connect(self.find_and_lock_best_roi)

        # Connect mouse events directly with correct parameter names
        def mousePressEvent(ev):
            return self.image_mouse_press(ev)
        def mouseMoveEvent(ev):
            return self.image_mouse_move(ev)
        def mouseReleaseEvent(ev):
            return self.image_mouse_release(ev)
        self.image_label.mousePressEvent = mousePressEvent
        self.image_label.mouseMoveEvent = mouseMoveEvent
        self.image_label.mouseReleaseEvent = mouseReleaseEvent

        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self.set_bg_btn.clicked.connect(self._set_background_roi)
        self.show_mask_checkbox.toggled.connect(self._on_mask_checkbox_toggled)
        
        # Reference mask system signals
        self.set_ref_frame_btn.clicked.connect(self.set_reference_frame)
        self.clear_ref_masks_btn.clicked.connect(self.clear_reference_masks)
        self.use_ref_masks_checkbox.toggled.connect(self._on_use_ref_masks_toggled)
        self.mask_method_combo.currentTextChanged.connect(self._on_mask_method_changed)
        self.show_ref_mask_checkbox.toggled.connect(self._on_show_ref_mask_toggled)
        
        # Expert mode and manual override signals
        self.expert_mode_checkbox.toggled.connect(self._on_expert_mode_toggled)
        self.emergency_manual_checkbox.toggled.connect(self._on_emergency_manual_toggled)
        self.force_manual_thresholds_checkbox.toggled.connect(self._on_force_manual_thresholds_toggled)
        self.use_global_threshold_checkbox.toggled.connect(self._on_use_global_threshold_toggled)
        self.analysis_method_combo.currentTextChanged.connect(self._on_analysis_method_changed)
        self.stats_method_combo.currentTextChanged.connect(self._on_stats_method_changed)

    def _update_widget_states(self, video_loaded=False, rois_exist=False):
        """Enable/disable widgets based on application state."""
        self.frame_slider.setEnabled(video_loaded and not self._analysis_in_progress)
        self.frame_spinbox.setEnabled(video_loaded and not self._analysis_in_progress)
        self.prev_frame_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.next_frame_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.jump_back_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.jump_forward_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.set_start_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.set_end_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.auto_detect_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.analyze_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        
        # Playback controls
        self.play_pause_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.speed_combo.setEnabled(video_loaded and not self._analysis_in_progress)
        self.add_rect_btn.setEnabled(video_loaded and not self._analysis_in_progress)
        self.del_rect_btn.setEnabled(video_loaded and self.selected_rect_idx is not None and not self._analysis_in_progress)
        self.clear_rect_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        self.set_bg_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        self.threshold_spin.setEnabled(not self._analysis_in_progress)
        
        # Reference mask controls
        self.set_ref_frame_btn.setEnabled(video_loaded and rois_exist and not self._analysis_in_progress)
        self.clear_ref_masks_btn.setEnabled(bool(self.reference_masks) and not self._analysis_in_progress)
        self.use_ref_masks_checkbox.setEnabled(bool(self.reference_masks) and not self._analysis_in_progress)
        self.mask_method_combo.setEnabled(video_loaded and not self._analysis_in_progress)
        
        # Expert controls - always available when not analyzing
        self.expert_mode_checkbox.setEnabled(not self._analysis_in_progress)
        self.emergency_manual_checkbox.setEnabled(not self._analysis_in_progress)
        self.force_manual_thresholds_checkbox.setEnabled(not self._analysis_in_progress)
        self.use_global_threshold_checkbox.setEnabled(not self._analysis_in_progress)
        self.analysis_method_combo.setEnabled(not self._analysis_in_progress)
        self.stats_method_combo.setEnabled(not self._analysis_in_progress)

    def _update_cache_status(self):
        """Update cache status display."""
        if self.frame_cache is not None:
            cache_size = self.frame_cache.get_size()
            self.cache_status_label.setText(f"Cache: {cache_size}/{FRAME_CACHE_SIZE} frames")
        elif self._use_video_processor and hasattr(self, 'video_processor'):
            cache_size = self.video_processor.frame_cache.get_size()
            self.cache_status_label.setText(f"Cache: {cache_size}/{FRAME_CACHE_SIZE} frames")
        else:
            self.cache_status_label.setText("Cache: N/A")

    def _update_video_info(self):
        """Update video information display."""
        if not self.video_path or not self.cap:
            self.video_info_label.setText("No video loaded")
            self.file_info_label.setText("No video loaded")
            return
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = self.total_frames / fps if fps > 0 else 0
        
        file_name = os.path.basename(self.video_path)
        file_size = os.path.getsize(self.video_path) / (1024 * 1024)  # MB
        
        info_text = f"""
<b>📁 {file_name}</b><br>
<b>📐</b> {width} × {height} • {self.total_frames} frames<br>
<b>⏱️</b> {duration_sec:.1f}s @ {fps:.2f} FPS<br>
<b>💾</b> {file_size:.1f} MB<br>
<b>🎬</b> Analysis: {self.start_frame + 1}-{(self.end_frame or 0) + 1}
        """.strip()
        
        self.video_info_label.setText(info_text)
        self.file_info_label.setText(f"Loaded: {file_name}")

    # --- Event Handling ---

    # File-picker slot
    def open_video_dialog(self):
        """Present a file dialog to select a video file."""
        initial_dir = os.path.dirname(self.video_path) if self.video_path else ""

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            initial_dir,
            "Video Files (*.mp4 *.mov *.avi *.mkv *.wmv *.m4v *.flv);;All Files (*)"
        )
        if path:
            self.video_path = path
            self.load_video()

    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent):
        """Accept drag events if they contain URLs (files)."""
        if a0.mimeData().hasUrls():
            a0.acceptProposedAction() # Use acceptProposedAction for clarity
        else:
            a0.ignore()

    def dropEvent(self, a0: QtGui.QDropEvent):
        """Handle dropped files, attempting to load the first valid video."""
        urls = a0.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            # Basic check for video file extensions (can be improved)
            if os.path.splitext(path)[1].lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                self.video_path = path
                self.load_video() # Changed from load_first_frame
            else:
                QtWidgets.QMessageBox.warning(self, 'Invalid File',
                                              f'Unsupported file type: {os.path.basename(path)}')
            a0.acceptProposedAction()
        else:
            a0.ignore()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        """Release resources and save settings when the window closes."""
        if self.cap:
            self.cap.release()
        self._save_settings()
        super().closeEvent(a0)

    def resizeEvent(self, a0: QtGui.QResizeEvent):
        """
        Update cached label size *without* immediately redrawing the frame.
        Calling show_frame() synchronously inside resizeEvent can create
        a feedback loop: the freshly scaled pixmap changes the label's
        sizeHint, Qt recalculates the layout, and another resizeEvent fires.
        By deferring the redraw with QTimer.singleShot(0, …) we let the
        resize settle first and repaint exactly once.
        """
        if hasattr(self, "image_label") and self.image_label.size().isValid():
            self._current_image_size = self.image_label.size()

            # Schedule a one‑shot repaint after the event loop returns.
            if self.frame is not None:
                QtCore.QTimer.singleShot(0, self.show_frame)

        # Call base-class handler last (standard Qt practice)
        super().resizeEvent(a0)

    # --- Video Loading and Frame Display ---

    def load_video(self):
        """Loads the video specified by self.video_path with comprehensive validation."""
        # Clean up existing video capture
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Clear frame cache
        if self.frame_cache is not None:
            self.frame_cache.clear()
        
        # Validate video file path
        try:
            if not self.video_path:
                raise ValidationError("No video file selected")
                
            validate_video_file(self.video_path)
            logger.info(f"Loading video: {os.path.basename(self.video_path)}")
            
        except ValidationError as e:
            error_msg = f"Video validation failed: {e}"
            logger.error(error_msg)
            QtWidgets.QMessageBox.critical(self, 'Video Validation Error', str(e))
            self._reset_state()
            return
        except Exception as e:
            error_msg = f"Unexpected error during video validation: {e}"
            logger.error(error_msg, exc_info=True)
            QtWidgets.QMessageBox.critical(self, 'Error', error_msg)
            self._reset_state()
            return

        # Attempt to open video file
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise VideoLoadError(f"OpenCV could not open video file: {os.path.basename(self.video_path)}")
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.playback_fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Validate video properties
            if self.total_frames <= 0:
                raise VideoLoadError("Video has no frames or invalid frame count")
                
            if self.playback_fps <= 0:
                logger.warning("Invalid FPS detected, using fallback")
                self.playback_fps = 30.0  # Default fallback
                
            if frame_width < MIN_FRAME_SIZE[0] or frame_height < MIN_FRAME_SIZE[1]:
                raise VideoLoadError(f"Video resolution ({frame_width}x{frame_height}) is below minimum {MIN_FRAME_SIZE}")
                
            if frame_width > MAX_FRAME_SIZE[0] or frame_height > MAX_FRAME_SIZE[1]:
                raise VideoLoadError(f"Video resolution ({frame_width}x{frame_height}) exceeds maximum {MAX_FRAME_SIZE}")
            
            # Calculate duration and validate
            duration = self.total_frames / self.playback_fps
            if duration < MIN_VIDEO_DURATION:
                raise VideoLoadError(f"Video duration ({duration:.2f}s) is below minimum {MIN_VIDEO_DURATION}s")
                
            if duration > MAX_VIDEO_DURATION:
                logger.warning(f"Video duration ({duration:.2f}s) is very long, performance may be affected")
            
            # Test read first frame to ensure video is readable
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise VideoLoadError("Could not read first frame from video")
                
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            logger.info(f"Video loaded successfully: {frame_width}x{frame_height}, {self.total_frames} frames, {self.playback_fps:.2f} FPS, {duration:.2f}s")
            
        except VideoLoadError as e:
            logger.error(f"Video load error: {e}")
            QtWidgets.QMessageBox.critical(self, 'Video Load Error', str(e))
            self._reset_state()
            return
        except Exception as e:
            error_msg = f"Unexpected error loading video: {e}"
            logger.error(error_msg, exc_info=True)
            QtWidgets.QMessageBox.critical(self, 'Error', error_msg)
            self._reset_state()
            return
            
        if self.total_frames <= 0:
             QtWidgets.QMessageBox.warning(self, 'Warning', 'Video file appears to have no frames.')
             self._reset_state()
             return

        # Clear cache when loading new video
        if self.frame_cache is not None:
            self.frame_cache.clear()
        
        self.current_frame_index = 0
        self.start_frame = 0
        self.end_frame = self.total_frames - 1

        # Load the first frame
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            if self.frame_cache is not None:
                self.frame_cache.put(0, frame)  # Cache first frame

            # Ensure the pixmap scales to the label's current size the first time we draw it
            self._current_image_size = self.image_label.size()
            
            # Update UI elements for the loaded video
            self.frame_slider.setRange(0, self.total_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_spinbox.setRange(0, self.total_frames - 1)
            self.frame_spinbox.setValue(0)
            self.update_frame_label()
            self.show_frame()
            self._update_video_info()
            self._update_cache_status()
            self._update_threshold_display()
            
            # Add to recent files
            self._add_recent_file(self.video_path)
            
            self.results_label.setText(f"Loaded: {os.path.basename(self.video_path)}\n"
                                       f"Frames: {self.total_frames}\n"
                                       "Draw ROIs or use Auto-Detect.")
            self._update_widget_states(video_loaded=True, rois_exist=bool(self.rects))
            self.statusBar().showMessage(f"Loaded: {os.path.basename(self.video_path)}")

        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Could not read the first frame of the video.')
            self._reset_state()

    def _reset_state(self):
        """Resets the application state when a video fails to load or is closed."""
        logger.debug("Resetting application state")
        
        # Stop any ongoing operations
        if hasattr(self, 'is_playing') and self.is_playing:
            self.stop_playback()
            
        # Release video capture resources
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.debug("Video capture resources released")
        
        # Reset video-related variables
        self.video_path = None
        self.frame = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.cap = None
        self.rects = []
        self.selected_rect_idx = None
        self.background_roi_idx = None
        self.locked_roi = None # Clear locked ROI
        self.start_frame = 0
        self.end_frame = None
        self.out_paths = []
        self.playback_fps = 30.0
        
        # Clear frame cache and force memory cleanup
        if self.frame_cache is not None:
            self.frame_cache.clear()
        
        # Reset playback controls
        self.speed_combo.setCurrentText("1x")
        self.playback_speed = 1.0
        
        # Reset UI elements
        self.image_label.clear()
        self.image_label.setText("Drag & Drop Video File Here")
        self.image_label.setPixmap(QtGui.QPixmap())
        self.update_frame_label(reset=True)
        self.results_label.setText("")
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        logger.debug("Application state reset complete")
        
        self.image_label.setText("Drag & Drop Video File Here")
        self.image_label.setPixmap(QtGui.QPixmap())
        self.update_frame_label(reset=True)
        self.update_rect_list()
        self.brightness_display_label.setText("N/A")
        self.results_label.setText("Load a video to begin analysis.")
        self.file_info_label.setText("No video loaded")
        self.video_info_label.setText("No video loaded")
        self.frame_slider.setRange(0, 0)
        self.frame_spinbox.setRange(0, 0)
        self._update_cache_status()
        self._update_widget_states(video_loaded=False, rois_exist=False)
        self.statusBar().showMessage("Ready - Load a video to begin")

    def slider_frame_changed(self, value: int):
        """Handles frame changes initiated by the slider."""
        if self.cap and self.cap.isOpened() and value != self.current_frame_index:
            self._seek_to_frame(value)
            # Sync spinbox without triggering its signal
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(value)
            self.frame_spinbox.blockSignals(False)

    def spinbox_frame_changed(self, value: int):
        """Handles frame changes initiated by the spinbox."""
        if self.cap and self.cap.isOpened() and value != self.current_frame_index:
            # Sync slider, which will trigger slider_frame_changed -> _seek_to_frame
            self.frame_slider.setValue(value)

    def step_frames(self, delta: int):
        """Moves forward or backward by a specified number of frames."""
        if not self.cap or not self.cap.isOpened() or self.total_frames == 0:
            return
        new_idx = max(0, min(self.total_frames - 1, self.current_frame_index + delta))
        if new_idx != self.current_frame_index:
            self.frame_slider.setValue(new_idx) # Let slider signal handle the update

    def set_start_frame(self):
        """Set the current frame as the analysis start frame."""
        if not self.cap or not self.cap.isOpened() or self.total_frames == 0:
            return
        self.start_frame = self.current_frame_index
        logger.info(f"Start frame set to: {self.start_frame}")
        # Update any UI elements that display start frame
        self.update_frame_label()

    def set_end_frame(self):
        """Set the current frame as the analysis end frame."""
        if not self.cap or not self.cap.isOpened() or self.total_frames == 0:
            return
        self.end_frame = self.current_frame_index
        logger.info(f"End frame set to: {self.end_frame}")
        # Update any UI elements that display end frame
        self.update_frame_label()


    def toggle_playback(self):
        """Toggle video playback on/off."""
        if not self.cap or not self.cap.isOpened() or self.total_frames == 0:
            return
            
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start video playback."""
        if not self.cap or not self.cap.isOpened() or self.total_frames == 0:
            return
        
        self.is_playing = True
        self.play_pause_btn.setText("⏸ Pause")
        
        # Calculate timer interval based on FPS and speed
        if hasattr(self, 'playback_fps') and self.playback_fps > 0:
            interval = int(1000 / (self.playback_fps * self.playback_speed))
        else:
            interval = int(1000 / (30.0 * self.playback_speed))  # Default 30 FPS
        
        self.playback_timer.start(interval)
    
    def stop_playback(self):
        """Stop video playback."""
        self.is_playing = False
        self.play_pause_btn.setText("▶ Play")
        self.playback_timer.stop()
    
    def advance_frame(self):
        """Advance to next frame during playback."""
        if not self.is_playing or not self.cap or not self.cap.isOpened():
            return
        
        next_frame = self.current_frame_index + 1
        if next_frame >= self.total_frames:
            # Reached end of video, stop playback
            self.stop_playback()
            return
        
        self.frame_slider.setValue(next_frame)
    
    def on_speed_changed(self, speed_text: str):
        """Handle playback speed change."""
        try:
            speed_value = float(speed_text.replace('x', ''))
            self.playback_speed = speed_value
            
            # If currently playing, restart timer with new interval
            if self.is_playing:
                self.stop_playback()
                self.start_playback()
        except ValueError:
            logger.warning(f"Invalid speed value: {speed_text}")

    def _seek_to_frame(self, frame_index: int):
        """Reads and displays the specified frame index with caching."""
        if not self.cap or not self.cap.isOpened():
            return
        if frame_index < 0 or frame_index >= self.total_frames:
            logger.warning(f"Attempted to seek to invalid frame index {frame_index}")
            return

        # Check cache first
        cached_frame = self.frame_cache.get(frame_index) if self.frame_cache else None
        if cached_frame is not None:
            self.frame = cached_frame
            self.current_frame_index = frame_index
            self.show_frame()
            self.update_frame_label()
            self._update_current_brightness_display()
            self._update_threshold_display()
            return

        # Not in cache, read from video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self.current_frame_index = frame_index
            
            # Cache the frame
            if self.frame_cache is not None:
                self.frame_cache.put(frame_index, frame)
            self._update_cache_status()
            
            self.show_frame()
            self.update_frame_label()
            self._update_current_brightness_display()
            self._update_threshold_display()
        else:
            logger.warning(f"Failed to read frame at index {frame_index}")

    def update_frame_label(self, reset=False):
        """Updates the frame counter label (e.g., "Frame: 10 / 100")."""
        if reset or self.total_frames == 0:
            self.frame_label.setText("Frame: 0 / 0")
        else:
            # Display 1-based index for user-friendliness
            self.frame_label.setText(f"Frame: {self.current_frame_index + 1} / {self.total_frames}")

    def show_frame(self):
        """Displays the current self.frame in the image_label, drawing ROIs."""
        if self.frame is None:
            return

        frame_copy = self.frame.copy()
        
        # Apply low-light enhancement preview if enabled
        if (hasattr(self, 'show_enhanced_preview_cb') and 
            self.show_enhanced_preview_cb.isChecked() and 
            hasattr(self, 'low_light_enhancer') and 
            self.low_light_enhancer.enabled):
            try:
                frame_copy = self.low_light_enhancer.enhance_roi(frame_copy)
            except Exception as e:
                logger.warning(f"Failed to apply enhancement preview: {e}")
        
        # Apply pixel mask visualization if enabled
        if self.show_pixel_mask and len(self.rects) > 0:
            frame_copy = self._apply_pixel_mask_overlay(frame_copy)
            
        self._draw_rois(frame_copy)

        # Convert to QPixmap for display
        rgb_image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)

        # Scale pixmap to fit label while maintaining aspect ratio
        target_size = self._current_image_size if (self._current_image_size is not None and self._current_image_size.isValid()) else self.image_label.size()
        if target_size is not None and target_size.isValid() and not target_size.isEmpty():
            scaled_pixmap = pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            # Fallback if target size is invalid (e.g., during init)
            self.image_label.setPixmap(pixmap)


    def _draw_rois(self, frame_to_draw_on):
        """Draws all defined ROIs and the currently drawing ROI onto the frame."""
        # Draw existing rectangles
        for idx, (pt1, pt2) in enumerate(self.rects):
            color = ROI_COLORS[idx % len(ROI_COLORS)]
            thickness = ROI_THICKNESS_SELECTED if idx == self.selected_rect_idx else ROI_THICKNESS_DEFAULT
            cv2.rectangle(frame_to_draw_on, pt1, pt2, color, thickness)
            # Draw index label near the top-left corner
            label = f"{idx+1}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, ROI_LABEL_FONT_SCALE, ROI_LABEL_THICKNESS)
            label_pos = (pt1[0] + 5, pt1[1] + text_height + 5)
            # Simple background for label visibility
            cv2.rectangle(frame_to_draw_on, (pt1[0], pt1[1]), (pt1[0] + text_width + 10, pt1[1] + text_height + 10), (0,0,0), cv2.FILLED)
            cv2.putText(frame_to_draw_on, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, ROI_LABEL_FONT_SCALE, color, ROI_LABEL_THICKNESS, cv2.LINE_AA)

        # Draw rectangle currently being drawn
        if self.drawing and self.start_point is not None and self.end_point is not None:
            # Map points from label coordinates to frame coordinates
            mapped = self._map_label_to_frame_rect(self.start_point, self.end_point)
            if mapped is not None:
                pt1_frame, pt2_frame = mapped
                cv2.rectangle(frame_to_draw_on, pt1_frame, pt2_frame, (0, 255, 255), ROI_THICKNESS_DEFAULT) # Use a distinct color (cyan)

    def _update_current_brightness_display(self):
        """Calculates and displays comprehensive brightness information for the current frame's ROIs."""
        if self.frame is None or not self.rects:
            self.brightness_display_label.setText("N/A")
            return

        roi_data = []
        background_brightness = None
        fh, fw = self.frame.shape[:2]
        
        # Calculate background brightness if background ROI is defined
        if self.background_roi_idx is not None:
            background_brightness = self._compute_background_brightness(self.frame)
        
        for idx, (pt1, pt2) in enumerate(self.rects):
            # Handle background ROI separately
            if idx == self.background_roi_idx:
                continue
                
            # Ensure ROI coordinates are valid within the frame
            x1 = max(0, min(pt1[0], fw - 1))
            y1 = max(0, min(pt1[1], fh - 1))
            x2 = max(0, min(pt2[0], fw - 1))
            y2 = max(0, min(pt2[1], fh - 1))

            if x2 > x1 and y2 > y1: # Check for valid ROI area
                roi = self.frame[y1:y2, x1:x2]
                brightness_stats = self._compute_brightness_stats(roi, background_brightness)
                l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median = brightness_stats[:6]
                roi_data.append((idx, l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median))
            else:
                roi_data.append((idx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) # Append zeros if ROI is invalid/empty

        if roi_data:
            # Build comprehensive display
            display_lines = []
            
            # Clean header with frame info
            display_lines.append(f"<b>Frame {self.current_frame_index + 1}</b>")
            
            # ROI brightness data with better formatting
            for idx, l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median in roi_data:
                roi_prefix = "🎯" if idx == self.selected_rect_idx else "  "
                
                if self.background_roi_idx is not None:
                    # Clean format with background subtraction (show as raw → corrected)
                    display_lines.append(f"{roi_prefix} ROI {idx+1}: L* {l_raw_mean:.1f} → {l_bg_sub_mean:.1f} | B {b_raw_mean:.0f}")
                else:
                    # Clean format without background
                    display_lines.append(f"{roi_prefix} ROI {idx+1}: L* {l_raw_mean:.1f} | B {b_raw_mean:.0f}")
            
            # Background ROI info (if defined)
            if self.background_roi_idx is not None and background_brightness is not None:
                # Calculate blue channel for background ROI
                bg_pt1, bg_pt2 = self.rects[self.background_roi_idx]
                bg_x1 = max(0, min(bg_pt1[0], fw - 1))
                bg_y1 = max(0, min(bg_pt1[1], fh - 1))
                bg_x2 = max(0, min(bg_pt2[0], fw - 1))
                bg_y2 = max(0, min(bg_pt2[1], fh - 1))
                
                if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                    bg_roi = self.frame[bg_y1:bg_y2, bg_x1:bg_x2]
                    _, _, _, _, bg_b_mean, _, _, _ = self._compute_brightness_stats(bg_roi)
                    display_lines.append(f"   Background: L* {background_brightness:.1f} | B {bg_b_mean:.0f}")
            
            # Use HTML formatting for better display
            formatted_text = "<br>".join(display_lines)
            self.brightness_display_label.setText(formatted_text)
        else:
            self.brightness_display_label.setText("N/A")


    # --- ROI Management ---

    def update_rect_list(self):
        """Updates the QListWidget displaying the ROIs."""
        self.rect_list.blockSignals(True) # Prevent selection signals during update
        current_row = self.rect_list.currentRow() # Remember selection
        self.rect_list.clear()
        for idx, (pt1, pt2) in enumerate(self.rects):
            x1, y1 = pt1
            x2, y2 = pt2
            # Ensure coordinates are ordered correctly for display
            disp_x1, disp_y1 = min(x1, x2), min(y1, y2)
            disp_x2, disp_y2 = max(x1, x2), max(y1, y2)
            prefix = "* " if idx == self.background_roi_idx else ""
            locked_indicator = "(Locked) " if self.locked_roi and self.locked_roi['frame_index'] == self.current_frame_index and idx == self.selected_rect_idx else ""
            self.rect_list.addItem(f"{prefix}{locked_indicator}ROI {idx+1}: ({disp_x1},{disp_y1})-({disp_x2},{disp_y2})")

        # Restore selection if possible
        if 0 <= current_row < len(self.rects):
             self.rect_list.setCurrentRow(current_row)
        elif len(self.rects) > 0:
             # If previous selection invalid, select the last one if available
             self.rect_list.setCurrentRow(len(self.rects) - 1)
        else:
             self.selected_rect_idx = None # No items, no selection

        self.rect_list.blockSignals(False)
        self._update_widget_states(video_loaded=bool(self.cap), rois_exist=bool(self.rects))
        self._update_threshold_display()


    def toggle_add_rectangle_mode(self, checked: bool):
        """Enters or exits the mode for drawing a new ROI."""
        self.drawing = checked
        if checked:
            self.selected_rect_idx = None # Deselect any existing rectangle
            self.update_rect_list() # Update list to show no selection
            self.image_label.setCursor(QtCore.Qt.CrossCursor) # Change cursor
            self.results_label.setText("Click and drag on the frame to draw a new ROI.")
        else:
            self.image_label.unsetCursor()
        self.show_frame() # Redraw to potentially remove selection highlight

    def select_rectangle_from_list(self, row: int):
        """Handles selection changes in the ROI list widget."""
        if 0 <= row < len(self.rects):
            self.selected_rect_idx = row
            self.drawing = False # Exit drawing mode if active
            self.add_rect_btn.setChecked(False)
            self.image_label.unsetCursor()
        else:
            self.selected_rect_idx = None
        self._update_widget_states(video_loaded=bool(self.cap), rois_exist=bool(self.rects))
        self.show_frame() # Redraw to highlight selected rectangle

    def delete_selected_rectangle(self):
        """Deletes the currently selected ROI."""
        if self.selected_rect_idx is not None and 0 <= self.selected_rect_idx < len(self.rects):
            del self.rects[self.selected_rect_idx]
            
            # Handle background ROI index adjustment
            if self.background_roi_idx == self.selected_rect_idx:
                self.background_roi_idx = None
            elif self.background_roi_idx is not None and self.selected_rect_idx < self.background_roi_idx:
                self.background_roi_idx -= 1
            
            # Adjust selection if the deleted item wasn't the last one
            if self.selected_rect_idx >= len(self.rects) and len(self.rects) > 0:
                 self.selected_rect_idx = len(self.rects) - 1
            elif len(self.rects) == 0:
                 self.selected_rect_idx = None
            # No need to explicitly set selection index otherwise, update_rect_list handles it
            self.update_rect_list()
            self.show_frame()

    def clear_all_rectangles(self):
        """Removes all defined ROIs."""
        if not self.rects: return # Nothing to clear
        reply = QtWidgets.QMessageBox.question(self, 'Confirm Clear',
                                               'Are you sure you want to delete all ROIs?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.rects.clear()
            self.selected_rect_idx = None
            self.background_roi_idx = None
            self.locked_roi = None  # Clear locked ROI
            self.update_rect_list()
            self.show_frame()

    def find_and_lock_best_roi(self):
        """Finds the best frame for the selected ROI and locks it for analysis."""
        if self.selected_rect_idx is None:
            QtWidgets.QMessageBox.information(self, "Find & Lock ROI", "Select an ROI first.")
            return

        if self.start_frame is None or self.end_frame is None:
            QtWidgets.QMessageBox.information(self, "Find & Lock ROI", "Set the analysis range first.")
            return

        best_frame_index = -1
        max_brightness = -1.0
        locked_roi_rect = None

        progress = QtWidgets.QProgressDialog("Finding best frame...", "Cancel", self.start_frame, self.end_frame, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)

        for i in range(self.start_frame, self.end_frame + 1):
            progress.setValue(i)
            if progress.wasCanceled():
                break

            frame = self._get_frame(i)
            if frame is None:
                continue

            pt1, pt2 = self.rects[self.selected_rect_idx]
            roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            brightness, _, _, _, _, _, _, _ = self._compute_brightness_stats(roi)

            if brightness > max_brightness:
                max_brightness = brightness
                best_frame_index = i
                locked_roi_rect = (pt1, pt2)

        progress.setValue(self.end_frame)

        if best_frame_index != -1:
            self.locked_roi = {
                'rect': locked_roi_rect,
                'frame_index': best_frame_index
            }
            self.results_label.setText(f"ROI locked from frame {best_frame_index + 1}.")
            self.update_rect_list()
        else:
            QtWidgets.QMessageBox.warning(self, "Find & Lock ROI", "Could not find a suitable frame to lock the ROI.")

    def _set_background_roi(self):
        """Mark current ROI as background reference for auto-detect."""
        if self.selected_rect_idx is None:
            QtWidgets.QMessageBox.information(self, "Background ROI",
                                              "Select an ROI first.")
            return
        
        self.background_roi_idx = self.selected_rect_idx
        
        # Calculate background threshold for display
        if self.frame is not None:
            threshold_value = self._calculate_background_threshold()
            if threshold_value is not None:
                self.results_label.setText(f"Background ROI set to ROI {self.selected_rect_idx + 1}\n"
                                         f"Current background threshold: {threshold_value:.2f} L*")
            else:
                self.results_label.setText(f"Background ROI set to ROI {self.selected_rect_idx + 1}\n"
                                         f"(Threshold will be calculated during full video scan)")
        else:
            self.results_label.setText(f"Background ROI set to ROI {self.selected_rect_idx + 1}")
        
        self.update_rect_list()

    def _calculate_background_threshold(self) -> Optional[float]:
        """Calculate the current background threshold based on background ROI or manual setting."""
        if self.background_roi_idx is not None and self.frame is not None:
            # Calculate threshold from current frame's background ROI
            if 0 <= self.background_roi_idx < len(self.rects):
                pt1, pt2 = self.rects[self.background_roi_idx]
                fh, fw = self.frame.shape[:2]
                
                # Ensure ROI coordinates are valid within the frame
                x1 = max(0, min(pt1[0], fw - 1))
                y1 = max(0, min(pt1[1], fh - 1))
                x2 = max(0, min(pt2[0], fw - 1))
                y2 = max(0, min(pt2[1], fh - 1))
                
                if x2 > x1 and y2 > y1:
                    roi = self.frame[y1:y2, x1:x2]
                    l_raw_mean, _, _, _, _, _, _, _ = self._compute_brightness_stats(roi)
                    return l_raw_mean
        
        # If no background ROI or calculation failed, return manual threshold
        return None

    def _update_threshold_display(self):
        """Update the threshold display label with current active threshold."""
        if self.background_roi_idx is not None:
            # Background ROI mode
            if self.frame is not None:
                threshold_value = self._calculate_background_threshold()
                if threshold_value is not None:
                    self.threshold_display_label.setText(f"Active Threshold: Background ROI {self.background_roi_idx + 1} ({threshold_value:.2f} L*)")
                else:
                    self.threshold_display_label.setText(f"Active Threshold: Background ROI {self.background_roi_idx + 1} (calculating...)")
            else:
                self.threshold_display_label.setText(f"Active Threshold: Background ROI {self.background_roi_idx + 1} (no frame)")
        else:
            # Manual threshold mode
            self.threshold_display_label.setText(f"Active Threshold: Manual ({self.manual_threshold:.2f} L*)")

    def _on_mask_checkbox_toggled(self, checked: bool):
        """Handle pixel mask visualization checkbox toggle."""
        self.show_pixel_mask = checked
        if self.frame is not None:
            self.show_frame()

    def _apply_pixel_mask_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply colored overlays to show analyzed pixels and reference masks.
        
        Colors:
        - Red: Current frame dynamic threshold pixels (standard mode)
        - Blue: Reference mask pixels (reference mode)
        - Green: Per-ROI custom threshold pixels
        
        Args:
            frame: BGR frame to apply overlay to
            
        Returns:
            Frame with colored mask overlays applied
        """
        overlay = frame.copy()
        
        for roi_idx, (pt1, pt2) in enumerate(self.rects):
            # Skip background ROI
            if roi_idx == self.background_roi_idx:
                continue
                
            # Extract ROI bounds
            fh, fw = frame.shape[:2]
            x1 = max(0, min(pt1[0], fw - 1))
            y1 = max(0, min(pt1[1], fh - 1))
            x2 = max(0, min(pt2[0], fw - 1))
            y2 = max(0, min(pt2[1], fh - 1))
            
            if x2 > x1 and y2 > y1:
                roi_overlay = overlay[y1:y2, x1:x2].copy()
                
                try:
                    # Show reference mask if available and enabled
                    if (self.show_reference_mask and roi_idx in self.reference_masks):
                        reference_mask = self.reference_masks[roi_idx]
                        if reference_mask.shape == roi_overlay.shape[:2]:
                            # Blue overlay for reference mask pixels
                            roi_overlay[reference_mask] = roi_overlay[reference_mask] * 0.7 + np.array([255, 0, 0]) * 0.3  # Blue tint
                    
                    # Show current frame mask if pixel mask is enabled
                    elif self.show_pixel_mask:
                        roi = frame[y1:y2, x1:x2]
                        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                        l_chan = lab[:, :, 0].astype(np.float32)
                        l_star = l_chan * 100.0 / 255.0
                        
                        # Get active threshold for this ROI
                        active_threshold = self._get_active_threshold_for_roi(roi_idx)
                        
                        # Create mask for analyzed pixels
                        mask = l_star > active_threshold
                        
                        # Choose color based on threshold type
                        if not self.use_global_threshold and roi_idx in self.roi_thresholds:
                            # Green for per-ROI custom thresholds
                            color_overlay = np.array([0, 255, 0])  # Green
                        else:
                            # Red for global/background thresholds
                            color_overlay = np.array([0, 0, 255])  # Red
                        
                        # Apply colored overlay to analyzed pixels
                        roi_overlay[mask] = roi_overlay[mask] * 0.7 + color_overlay * 0.3
                    
                    # Apply the processed ROI back to the main overlay
                    overlay[y1:y2, x1:x2] = roi_overlay
                    
                except Exception as e:
                    logger.warning(f"Error creating pixel mask for ROI {roi_idx+1}: {e}")
                    continue
        
        return overlay

    def _on_threshold_changed(self, value: float):
        """Handle changes to the manual threshold spinbox."""
        self.manual_threshold = value
        self._update_threshold_display()
        if self.frame is not None:
            self.show_frame()

    # --- UI Signal Handlers for Professional Controls ---

    def _on_use_ref_masks_toggled(self, checked: bool):
        """Handle reference mask mode toggle."""
        if checked and not self.reference_masks:
            QtWidgets.QMessageBox.warning(self, 'Reference Mask Mode', 
                'No reference masks available. Please set a reference frame first.')
            self.use_ref_masks_checkbox.setChecked(False)
            return
        
        self.use_reference_masks = checked
        status = "enabled" if checked else "disabled"
        logger.info(f"Reference mask analysis {status}")
        
        # Update UI status
        self._update_ref_mask_status()
        
        # Update display
        if self.frame is not None:
            self.show_frame()

    def _on_mask_method_changed(self, method: str):
        """Handle mask generation method change."""
        self.mask_generation_method = method
        logger.info(f"Mask generation method changed to: {method}")
        
        # Regenerate masks if reference frame is set
        if self.reference_frame_idx is not None:
            self.generate_reference_masks()
            self._update_ref_mask_status()
            if self.frame is not None:
                self.show_frame()

    def _on_show_ref_mask_toggled(self, checked: bool):
        """Handle reference mask visualization toggle."""
        self.show_reference_mask = checked
        if self.frame is not None:
            self.show_frame()

    def _on_expert_mode_toggled(self, checked: bool):
        """Handle expert mode toggle."""
        self.enable_expert_mode = checked
        self.emergency_controls_widget.setVisible(checked)
        
        if checked:
            logger.info("Expert mode enabled - advanced manual controls available")
        else:
            logger.info("Expert mode disabled - simplified interface")
            # Reset emergency modes when exiting expert mode
            self.emergency_manual_mode = False
            self.force_manual_thresholds = False
            self.emergency_manual_checkbox.setChecked(False)
            self.force_manual_thresholds_checkbox.setChecked(False)

    def _on_emergency_manual_toggled(self, checked: bool):
        """Handle emergency manual mode toggle."""
        self.emergency_manual_mode = checked
        if checked:
            logger.warning("EMERGENCY MANUAL MODE ENABLED - All automation bypassed")
            QtWidgets.QMessageBox.warning(self, 'Emergency Manual Mode', 
                'Emergency Manual Mode enabled.\n\n'
                'ALL automation is now bypassed. You have complete manual control.\n'
                'Use this mode only when automatic systems fail.')
        else:
            logger.info("Emergency manual mode disabled")

    def _on_force_manual_thresholds_toggled(self, checked: bool):
        """Handle force manual thresholds toggle."""
        self.force_manual_thresholds = checked
        if checked:
            logger.info("Manual thresholds forced - ignoring auto-detection and background ROI")
        else:
            logger.info("Auto-detection and background ROI calculations re-enabled")
        
        # Update threshold display
        self._update_threshold_display()
        
        # Update display
        if self.frame is not None:
            self.show_frame()

    def _on_use_global_threshold_toggled(self, checked: bool):
        """Handle global threshold mode toggle."""
        self.use_global_threshold = checked
        if checked:
            logger.info("Using global threshold for all ROIs")
        else:
            logger.info("Per-ROI threshold mode enabled")
        
        # Update display
        if self.frame is not None:
            self.show_frame()

    def _on_analysis_method_changed(self, method: str):
        """Handle analysis method change."""
        self.brightness_calc_method = method
        logger.info(f"Brightness calculation method changed to: {method}")

    def _on_stats_method_changed(self, method: str):
        """Handle statistical method change."""
        self.statistical_method = method
        logger.info(f"Statistical method changed to: {method}")

    def _update_ref_mask_status(self):
        """Update the reference mask status display."""
        if self.reference_frame_idx is not None:
            mask_count = len(self.reference_masks)
            status_text = f"Reference frame: {self.reference_frame_idx + 1} | {mask_count} masks"
            if self.use_reference_masks:
                status_text += " | ACTIVE"
                self.ref_mask_status_label.setStyleSheet("color: #70ad47; font-weight: bold; padding: 2px;")
            else:
                status_text += " | inactive"
                self.ref_mask_status_label.setStyleSheet("color: #888888; font-style: italic; padding: 2px;")
        else:
            status_text = "No reference frame set"
            self.ref_mask_status_label.setStyleSheet("color: #888888; font-style: italic; padding: 2px;")
        
        self.ref_mask_status_label.setText(status_text)

    # --- Reference Mask System (Professional-Grade) ---

    def set_reference_frame(self):
        """Set the current frame as the reference for mask generation."""
        if not self.cap or not self.cap.isOpened() or self.frame is None:
            QtWidgets.QMessageBox.warning(self, 'Reference Frame Error', 'No video loaded or no current frame.')
            return
        
        if not self.rects:
            QtWidgets.QMessageBox.warning(self, 'Reference Frame Error', 'No ROIs defined. Please draw ROIs first.')
            return
        
        self.reference_frame_idx = self.current_frame_index
        logger.info(f"Reference frame set to: {self.reference_frame_idx}")
        
        # Generate reference masks for all ROIs
        self.generate_reference_masks()
        
        # Update UI to show reference frame is set
        self._update_ref_mask_status()
        QtWidgets.QMessageBox.information(self, 'Reference Frame Set', 
            f"Reference frame set to frame {self.reference_frame_idx + 1}.\n"
            f"Generated {len(self.reference_masks)} reference masks.")
        
        # Update display if reference mask visualization is enabled
        if self.show_reference_mask and self.frame is not None:
            self.show_frame()

    def generate_reference_masks(self):
        """Generate binary masks for all ROIs based on the reference frame."""
        if self.reference_frame_idx is None or self.frame is None:
            logger.warning("Cannot generate reference masks: no reference frame set")
            return
        
        # Get the reference frame
        reference_frame = self._get_frame(self.reference_frame_idx)
        if reference_frame is None:
            logger.error(f"Cannot load reference frame {self.reference_frame_idx}")
            return
        
        self.reference_masks.clear()
        self.reference_mask_metadata.clear()
        
        # Get active threshold for mask generation
        active_threshold = self._get_active_threshold_for_roi(None)  # Use global for reference
        
        for roi_idx, (pt1, pt2) in enumerate(self.rects):
            # Skip background ROI
            if roi_idx == self.background_roi_idx:
                continue
            
            try:
                # Extract ROI from reference frame
                mask, metadata = self._generate_roi_reference_mask(
                    reference_frame, pt1, pt2, roi_idx, active_threshold
                )
                
                if mask is not None:
                    self.reference_masks[roi_idx] = mask
                    self.reference_mask_metadata[roi_idx] = metadata
                    logger.info(f"Generated reference mask for ROI {roi_idx + 1}: "
                              f"{metadata['pixel_count']} pixels ({metadata['coverage_percentage']:.1f}%)")
                
            except Exception as e:
                logger.error(f"Failed to generate reference mask for ROI {roi_idx + 1}: {e}")
                continue
        
        logger.info(f"Reference mask generation complete: {len(self.reference_masks)} masks generated")

    def _generate_roi_reference_mask(self, frame: np.ndarray, pt1: tuple, pt2: tuple, 
                                   roi_idx: int, threshold: float) -> tuple:
        """Generate a reference mask for a single ROI."""
        fh, fw = frame.shape[:2]
        x1 = max(0, min(pt1[0], fw - 1))
        y1 = max(0, min(pt1[1], fh - 1))
        x2 = max(0, min(pt2[0], fw - 1))
        y2 = max(0, min(pt2[1], fh - 1))
        
        if x2 <= x1 or y2 <= y1:
            return None, {}
        
        # Extract ROI and convert to LAB
        roi = frame[y1:y2, x1:x2]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        l_star = l_chan * 100.0 / 255.0
        
        # Create binary mask based on threshold
        if self.mask_generation_method == 'threshold':
            mask = l_star > threshold
        elif self.mask_generation_method == 'adaptive':
            # Use adaptive thresholding for more sophisticated masking
            mask = self._generate_adaptive_mask(l_star, threshold)
        else:  # manual method would be implemented separately
            mask = l_star > threshold
        
        # Calculate mask quality metrics
        total_pixels = mask.size
        active_pixels = np.sum(mask)
        coverage_percentage = (active_pixels / total_pixels) * 100
        
        # Calculate brightness statistics for masked pixels
        masked_brightness = l_star[mask]
        if len(masked_brightness) > 0:
            brightness_mean = np.mean(masked_brightness)
            brightness_std = np.std(masked_brightness)
            brightness_median = np.median(masked_brightness)
        else:
            brightness_mean = brightness_std = brightness_median = 0.0
        
        metadata = {
            'roi_idx': roi_idx,
            'frame_idx': self.reference_frame_idx,
            'threshold_used': threshold,
            'generation_method': self.mask_generation_method,
            'pixel_count': int(active_pixels),
            'total_pixels': int(total_pixels),
            'coverage_percentage': coverage_percentage,
            'brightness_stats': {
                'mean': brightness_mean,
                'std': brightness_std,
                'median': brightness_median
            },
            'roi_bounds': (x1, y1, x2, y2),
            'mask_shape': mask.shape
        }
        
        return mask, metadata

    def _generate_adaptive_mask(self, l_star: np.ndarray, base_threshold: float) -> np.ndarray:
        """Generate an adaptive threshold mask for more sophisticated masking."""
        # Use Otsu's method as a starting point, then adjust based on base_threshold
        try:
            # Convert to uint8 for Otsu
            l_uint8 = (l_star * 255.0 / 100.0).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(l_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_thresh_l_star = (otsu_thresh / 255.0) * 100.0
            
            # Blend Otsu threshold with manual threshold
            adaptive_threshold = (otsu_thresh_l_star + base_threshold) / 2.0
            
            return l_star > adaptive_threshold
            
        except Exception as e:
            logger.warning(f"Adaptive masking failed, falling back to simple threshold: {e}")
            return l_star > base_threshold

    def _get_active_threshold_for_roi(self, roi_idx: Optional[int]) -> float:
        """Get the active threshold for a specific ROI, considering manual overrides."""
        # Emergency manual mode - always use manual threshold
        if self.emergency_manual_mode or self.force_manual_thresholds:
            return self.manual_threshold
        
        # Check for per-ROI threshold override
        if not self.use_global_threshold and roi_idx is not None and roi_idx in self.roi_thresholds:
            return self.roi_thresholds[roi_idx]
        
        # Use background ROI calculation if available and not forced manual
        if not self.force_manual_thresholds:
            bg_threshold = self._calculate_background_threshold()
            if bg_threshold is not None:
                return bg_threshold
        
        # Fall back to manual threshold
        return self.manual_threshold

    def apply_reference_mask_to_roi(self, frame: np.ndarray, roi_idx: int, pt1: tuple, pt2: tuple) -> np.ndarray:
        """Apply the reference mask to extract consistent pixels from an ROI."""
        if not self.use_reference_masks or roi_idx not in self.reference_masks:
            # Fall back to standard extraction
            fh, fw = frame.shape[:2]
            x1 = max(0, min(pt1[0], fw - 1))
            y1 = max(0, min(pt1[1], fh - 1))
            x2 = max(0, min(pt2[0], fw - 1))
            y2 = max(0, min(pt2[1], fh - 1))
            return frame[y1:y2, x1:x2]
        
        # Apply reference mask
        reference_mask = self.reference_masks[roi_idx]
        metadata = self.reference_mask_metadata[roi_idx]
        x1, y1, x2, y2 = metadata['roi_bounds']
        
        # Extract ROI and apply mask
        roi = frame[y1:y2, x1:x2]
        
        # Create masked ROI where non-mask pixels are set to a consistent value
        masked_roi = roi.copy()
        if reference_mask.shape == roi.shape[:2]:
            # Set non-masked pixels to a neutral value (mean of the ROI)
            roi_mean = np.mean(roi, axis=(0, 1)).astype(roi.dtype)
            masked_roi[~reference_mask] = roi_mean
        
        return masked_roi

    def toggle_reference_mask_mode(self):
        """Toggle reference mask analysis mode on/off."""
        if not self.reference_masks:
            QtWidgets.QMessageBox.warning(self, 'Reference Mask Mode', 
                'No reference masks available. Please set a reference frame first.')
            return
        
        self.use_reference_masks = not self.use_reference_masks
        status = "enabled" if self.use_reference_masks else "disabled"
        logger.info(f"Reference mask mode {status}")
        
        # Update display
        if self.frame is not None:
            self.show_frame()
        
        return self.use_reference_masks

    def clear_reference_masks(self):
        """Clear all reference masks and reset reference frame."""
        self.reference_masks.clear()
        self.reference_mask_metadata.clear()
        self.reference_frame_idx = None
        self.use_reference_masks = False
        
        # Update UI checkboxes
        self.use_ref_masks_checkbox.setChecked(False)
        self._update_ref_mask_status()
        
        logger.info("Reference masks cleared")
        
        # Update display
        if self.frame is not None:
            self.show_frame()

    # --- Mouse Interaction on Image Label ---

    def image_mouse_press(self, event: QtGui.QMouseEvent):
        """Handles mouse clicks on the video display area."""
        if self.frame is None or event.button() != QtCore.Qt.LeftButton:
            return # Ignore if no video or not left click

        pos_in_label = event.pos() # Position relative to the image_label widget

        # Check if the click is within the actual displayed pixmap area
        pixmap_rect = self._get_pixmap_rect_in_label()
        if pixmap_rect is None or not pixmap_rect.contains(pos_in_label):
             return # Click was outside the image area (in borders/empty space)

        # Convert click position to frame coordinates
        mapped_point = self._map_label_to_frame_point(pos_in_label)
        if mapped_point is None:
            return # Mapping failed
        frame_x, frame_y = mapped_point

        if self.drawing:
            # Start drawing a new rectangle
            self.start_point = pos_in_label # Store label coordinates for drawing feedback
            self.end_point = pos_in_label
            self.moving = False
            self.resizing = False
        elif self.selected_rect_idx is not None:
            # Check if clicking near a corner of the selected rectangle for resizing
            pt1, pt2 = self.rects[self.selected_rect_idx]
            corners = [
                (pt1[0], pt1[1]), (pt2[0], pt1[1]), # Top-left, Top-right
                (pt1[0], pt2[1]), (pt2[0], pt2[1])  # Bottom-left, Bottom-right
            ]
            resize_margin = self._scale_value_for_pixmap(MOUSE_RESIZE_HANDLE_SENSITIVITY) # Scale sensitivity

            for i, (cx, cy) in enumerate(corners):
                if abs(frame_x - cx) < resize_margin and abs(frame_y - cy) < resize_margin:
                    self.resizing = True
                    self.resize_corner = i # Store which corner (0=TL, 1=TR, 2=BL, 3=BR)
                    # Determine the opposite corner, which stays fixed during resize
                    fixed_corner_index = 3 - i
                    self.start_point = self._map_frame_to_label_point(corners[fixed_corner_index]) # Store fixed corner in label coords
                    self.end_point = pos_in_label # Moving point is the current mouse pos
                    self.moving = False
                    self.drawing = False
                    self.image_label.setCursor(self._get_resize_cursor(i)) # Set resize cursor
                    self.show_frame()
                    return # Resizing initiated

            # Check if clicking inside the selected rectangle for moving
            pt1, pt2 = self.rects[self.selected_rect_idx]
            rect_x1, rect_y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
            rect_x2, rect_y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
            if rect_x1 <= frame_x <= rect_x2 and rect_y1 <= frame_y <= rect_y2:
                self.moving = True
                # Calculate offset from top-left corner of the rect
                self.move_offset = (frame_x - rect_x1, frame_y - rect_y1)
                self.start_point = pos_in_label # Store initial position for smooth dragging
                self.resizing = False
                self.drawing = False
                self.image_label.setCursor(QtCore.Qt.SizeAllCursor) # Set move cursor
                self.show_frame()
                return # Moving initiated

        # If click wasn't for drawing, resizing, or moving a selected rect,
        # check if it hit any *other* rectangle to select it.
        clicked_on_rect_idx = -1
        for idx, (pt1, pt2) in enumerate(self.rects):
            rect_x1, rect_y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
            rect_x2, rect_y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
            if rect_x1 <= frame_x <= rect_x2 and rect_y1 <= frame_y <= rect_y2:
                clicked_on_rect_idx = idx
                break

        if clicked_on_rect_idx != -1:
            self.selected_rect_idx = clicked_on_rect_idx
            self.update_rect_list()
            self.show_frame()
        else:
            # If click was on empty space, deselect current ROI
            self.selected_rect_idx = None
            self.update_rect_list()
            self.show_frame()

    def image_mouse_move(self, event: QtGui.QMouseEvent):
        """Handles mouse movements on the video display area."""
        if self.frame is None:
            return

        pos_in_label = event.pos()
        mapped_point = self._map_label_to_frame_point(pos_in_label)
        if mapped_point is None:
            return
        frame_x, frame_y = mapped_point

        if self.drawing and self.start_point is not None:
            self.end_point = pos_in_label
            self.show_frame() # Redraw to show the rubber band rectangle
        elif self.moving and self.selected_rect_idx is not None:
            # Calculate new position based on mouse movement and initial offset
            pt1_orig, pt2_orig = self.rects[self.selected_rect_idx]
            rect_width = abs(pt2_orig[0] - pt1_orig[0])
            rect_height = abs(pt2_orig[1] - pt1_orig[1])

            if self.move_offset is not None:
                new_x1 = frame_x - self.move_offset[0]
                new_y1 = frame_y - self.move_offset[1]
                new_x2 = new_x1 + rect_width
                new_y2 = new_y1 + rect_height

                # Clamp to frame boundaries
                new_x1 = max(0, min(new_x1, self.frame.shape[1] - rect_width))
                new_y1 = max(0, min(new_y1, self.frame.shape[0] - rect_height))
                new_x2 = new_x1 + rect_width
                new_y2 = new_y1 + rect_height

                self.rects[self.selected_rect_idx] = ((new_x1, new_y1), (new_x2, new_y2))
                self.update_rect_list()
                self.show_frame()
        elif self.resizing and self.selected_rect_idx is not None and self.start_point is not None:
            pt1_fixed_label, pt2_moving_label = self.start_point, pos_in_label
            mapped = self._map_label_to_frame_rect(pt1_fixed_label, pt2_moving_label)
            if mapped is not None:
                pt1_frame, pt2_frame = mapped
                # Ensure minimum size during resize
                min_w, min_h = MIN_ROI_SIZE
                current_w = abs(pt2_frame[0] - pt1_frame[0])
                current_h = abs(pt2_frame[1] - pt1_frame[1])

                if current_w < min_w:
                    pt2_frame = (pt1_frame[0] + min_w if pt2_frame[0] > pt1_frame[0] else pt1_frame[0] - min_w, pt2_frame[1])
                if current_h < min_h:
                    pt2_frame = (pt2_frame[0], pt1_frame[1] + min_h if pt2_frame[1] > pt1_frame[1] else pt1_frame[1] - min_h)

                self.rects[self.selected_rect_idx] = (pt1_frame, pt2_frame)
                self.update_rect_list()
                self.show_frame()

        # Update cursor based on proximity to ROI handles
        if not self.drawing and not self.moving and not self.resizing and self.selected_rect_idx is not None:
            pt1, pt2 = self.rects[self.selected_rect_idx]
            corners = [
                (pt1[0], pt1[1]), (pt2[0], pt1[1]), # Top-left, Top-right
                (pt1[0], pt2[1]), (pt2[0], pt2[1])  # Bottom-left, Bottom-right
            ]
            resize_margin = self._scale_value_for_pixmap(MOUSE_RESIZE_HANDLE_SENSITIVITY)

            cursor_set = False
            for i, (cx, cy) in enumerate(corners):
                if abs(frame_x - cx) < resize_margin and abs(frame_y - cy) < resize_margin:
                    self.image_label.setCursor(self._get_resize_cursor(i))
                    cursor_set = True
                    break
            if not cursor_set:
                self.image_label.unsetCursor()

    def image_mouse_release(self, event: QtGui.QMouseEvent):
        """Handles mouse release events on the video display area."""
        if self.frame is None or event.button() != QtCore.Qt.LeftButton:
            return

        if self.drawing:
            self.drawing = False
            self.add_rect_btn.setChecked(False) # Uncheck the button
            self.image_label.unsetCursor() # Reset cursor
            
            # Map final points to frame coordinates
            if self.start_point is not None:
                rect_mapping = self._map_label_to_frame_rect(self.start_point, event.pos())
                if rect_mapping is not None:
                    pt1_frame, pt2_frame = rect_mapping
                    # Ensure valid rectangle (min size)
                    x1, y1 = pt1_frame
                    x2, y2 = pt2_frame
                    
                    # Ensure x1 < x2 and y1 < y2
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)

                    # Apply minimum size constraint
                    if (x2 - x1) < MIN_ROI_SIZE[0]:
                        x2 = x1 + MIN_ROI_SIZE[0]
                    if (y2 - y1) < MIN_ROI_SIZE[1]:
                        y2 = y1 + MIN_ROI_SIZE[1]

                    # Clamp to frame boundaries
                    x2 = min(x2, self.frame.shape[1])
                    y2 = min(y2, self.frame.shape[0])
                    x1 = max(0, x2 - (x2 - x1)) # Adjust x1 if x2 was clamped
                    y1 = max(0, y2 - (y2 - y1)) # Adjust y1 if y2 was clamped

                    if (x2 - x1) >= MIN_ROI_SIZE[0] and (y2 - y1) >= MIN_ROI_SIZE[1]:
                        self.rects.append(((x1, y1), (x2, y2)))
                        self.selected_rect_idx = len(self.rects) - 1 # Select the newly added ROI
                        self.update_rect_list()
                        self.show_frame()
                    else:
                        QtWidgets.QMessageBox.warning(self, "Invalid ROI", "The drawn ROI is too small.")
                else:
                    # Handle case where coordinate mapping failed
                    logger.warning("Failed to map ROI coordinates from label to frame")
                    QtWidgets.QMessageBox.warning(
                        self, "ROI Creation Failed", 
                        "Unable to create ROI at this location. Please ensure you're drawing within the video frame area."
                    )
            else:
                # Handle case where start_point is None
                logger.warning("ROI creation failed: start_point is None")
            self.start_point = None
            self.end_point = None
        elif self.moving or self.resizing:
            self.moving = False
            self.resizing = False
            self.start_point = None
            self.end_point = None
            self.move_offset = None
            self.resize_corner = None
            self.image_label.unsetCursor()
            self.show_frame() # Redraw to ensure final state is correct

    def _get_pixmap_rect_in_label(self) -> Optional[QtCore.QRect]:
        """Returns the QRect of the actual pixmap displayed within the QLabel, accounting for aspect ratio."""
        if not self.image_label.pixmap():
            return None

        pixmap_size = self.image_label.pixmap().size()
        label_size = self.image_label.size()

        if pixmap_size.isEmpty() or label_size.isEmpty():
            return None

        # Calculate scaled pixmap size maintaining aspect ratio
        scaled_size = pixmap_size.scaled(label_size, QtCore.Qt.KeepAspectRatio)

        # Calculate top-left corner to center the pixmap
        x_offset = (label_size.width() - scaled_size.width()) // 2
        y_offset = (label_size.height() - scaled_size.height()) // 2

        return QtCore.QRect(x_offset, y_offset, scaled_size.width(), scaled_size.height())

    def _map_label_to_frame_point(self, label_point: QtCore.QPoint) -> Optional[Tuple[int, int]]:
        """Maps a point from QLabel coordinates to original frame coordinates."""
        pixmap_rect = self._get_pixmap_rect_in_label()
        if pixmap_rect is None or self.frame is None:
            return None

        # Adjust point relative to the pixmap within the label
        x_in_pixmap = label_point.x() - pixmap_rect.x()
        y_in_pixmap = label_point.y() - pixmap_rect.y()

        # Scale to original frame dimensions
        frame_width = self.frame.shape[1]
        frame_height = self.frame.shape[0]

        scaled_x = int(x_in_pixmap * (frame_width / pixmap_rect.width()))
        scaled_y = int(y_in_pixmap * (frame_height / pixmap_rect.height()))

        # Clamp to frame boundaries
        scaled_x = max(0, min(scaled_x, frame_width - 1))
        scaled_y = max(0, min(scaled_y, frame_height - 1))

        return scaled_x, scaled_y

    def _map_frame_to_label_point(self, frame_point: Tuple[int, int]) -> Optional[QtCore.QPoint]:
        """Maps a point from original frame coordinates to QLabel coordinates."""
        pixmap_rect = self._get_pixmap_rect_in_label()
        if pixmap_rect is None or self.frame is None:
            return None

        frame_width = self.frame.shape[1]
        frame_height = self.frame.shape[0]

        if frame_width == 0 or frame_height == 0:
            return None

        # Scale from original frame dimensions to pixmap dimensions
        x_in_pixmap = int(frame_point[0] * (pixmap_rect.width() / frame_width))
        y_in_pixmap = int(frame_point[1] * (pixmap_rect.height() / frame_height))

        # Adjust point relative to the label
        label_x = x_in_pixmap + pixmap_rect.x()
        label_y = y_in_pixmap + pixmap_rect.y()

        return QtCore.QPoint(label_x, label_y)

    def _map_label_to_frame_rect(self, p1_label: QtCore.QPoint, p2_label: QtCore.QPoint) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Maps two points defining a rectangle from QLabel to frame coordinates."""
        p1_frame = self._map_label_to_frame_point(p1_label)
        p2_frame = self._map_label_to_frame_point(p2_label)
        if p1_frame and p2_frame:
            return p1_frame, p2_frame
        return None

    def _scale_value_for_pixmap(self, value: int) -> int:
        """Scales a value (e.g., sensitivity) from original frame scale to current pixmap scale."""
        if self.frame is None or self.image_label.pixmap() is None:
            return value
        
        original_width = self.frame.shape[1]
        current_width = self.image_label.pixmap().width()
        
        if original_width == 0:
            return value
            
        scale_factor = current_width / original_width
        return int(value * scale_factor)

    def _get_resize_cursor(self, corner_index: int) -> QtCore.Qt.CursorShape:
        """Returns the appropriate cursor for resizing based on the corner."""
        # 0: Top-Left, 1: Top-Right, 2: Bottom-Left, 3: Bottom-Right
        if corner_index == 0 or corner_index == 3:
            return QtCore.Qt.SizeFDiagCursor # NWSE
        else:
            return QtCore.Qt.SizeBDiagCursor # NESW

    # --- Analysis Logic ---

    def analyze_video(self):
        """Performs brightness analysis on the selected video range and ROIs."""
        if not self.cap or not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, 'Analysis Error', 'No video loaded.')
            return
        if not self.rects:
            QtWidgets.QMessageBox.warning(self, 'Analysis Error', 'No ROIs defined. Please draw at least one ROI.')
            return
        if self.selected_rect_idx is None and self.locked_roi is None:
            QtWidgets.QMessageBox.warning(self, 'Analysis Error', 'No ROI selected for analysis. Please select an ROI or lock one.')
            return

        self._analysis_in_progress = True
        self._update_widget_states(video_loaded=True, rois_exist=True)
        self.statusBar().showMessage("Analysis in progress...")

        start_frame = self.start_frame
        end_frame = self.end_frame if self.end_frame is not None else self.total_frames - 1

        # Validate frame range
        try:
            validate_frame_range(start_frame, end_frame, self.total_frames)
        except ValidationError as e:
            QtWidgets.QMessageBox.critical(self, 'Analysis Error', str(e))
            self._analysis_in_progress = False
            self._update_widget_states(video_loaded=True, rois_exist=True)
            self.statusBar().showMessage("Analysis failed.")
            return

        # Initialize results storage
        results: List[List[Tuple[int, float, float, float, float, float, float, float, float]]] = [[] for _ in self.rects]
        
        # Determine background brightness for the entire analysis if a background ROI is set
        overall_background_brightness = None
        if self.background_roi_idx is not None:
            # Calculate average brightness of the background ROI over the entire analysis range
            bg_roi_brightness_sum = 0.0
            bg_roi_frame_count = 0
            
            progress_bg = QtWidgets.QProgressDialog("Calculating background brightness...", "Cancel", start_frame, end_frame, self)
            progress_bg.setWindowModality(QtCore.Qt.WindowModal)
            
            for i in range(start_frame, end_frame + 1):
                progress_bg.setValue(i)
                if progress_bg.wasCanceled():
                    break
                
                frame = self._get_frame(i)
                if frame is None:
                    continue
                
                pt1, pt2 = self.rects[self.background_roi_idx]
                fh, fw = frame.shape[:2]
                x1 = max(0, min(pt1[0], fw - 1))
                y1 = max(0, min(pt1[1], fh - 1))
                x2 = max(0, min(pt2[0], fw - 1))
                y2 = max(0, min(pt2[1], fh - 1))
                
                if x2 > x1 and y2 > y1:
                    bg_roi = frame[y1:y2, x1:x2]
                    l_raw_mean, _, _, _, _, _, _, _ = self._compute_brightness_stats(bg_roi)
                    bg_roi_brightness_sum += l_raw_mean
                    bg_roi_frame_count += 1
            
            progress_bg.setValue(end_frame) # Ensure progress bar is full
            
            if bg_roi_frame_count > 0:
                overall_background_brightness = bg_roi_brightness_sum / bg_roi_frame_count
                logger.info(f"Overall background brightness calculated: {overall_background_brightness:.2f} L*")
            else:
                logger.warning("Could not calculate overall background brightness. Using manual threshold or no threshold.")
                QtWidgets.QMessageBox.warning(self, "Background Calculation Failed", "Could not calculate background brightness for the selected ROI over the analysis range. Using manual threshold or no threshold.")

        progress = QtWidgets.QProgressDialog("Analyzing frames...", "Cancel", start_frame, end_frame, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)

        for i in range(start_frame, end_frame + 1):
            progress.setValue(i)
            if progress.wasCanceled():
                break

            frame = self._get_frame(i)
            if frame is None:
                continue

            # Process each ROI
            for roi_idx, _ in enumerate(self.rects):
                if roi_idx == self.background_roi_idx:
                    continue

                # Get the ROI for the current frame
                if self.locked_roi and roi_idx == self.selected_rect_idx: # Only apply locked ROI to the selected ROI
                    pt1, pt2 = self.locked_roi['rect']
                else:
                    pt1, pt2 = self.rects[roi_idx]

                # Extract ROI with reference mask support
                if self.use_reference_masks and roi_idx in self.reference_masks:
                    # Use reference mask for consistent pixel analysis
                    roi = self.apply_reference_mask_to_roi(frame, roi_idx, pt1, pt2)
                    # Use specific threshold method for reference mask analysis
                    if roi_idx in self.reference_mask_metadata:
                        threshold_override = self.reference_mask_metadata[roi_idx]['threshold_used']
                        brightness_stats = self._compute_brightness_stats_with_reference_mask(
                            frame, roi_idx, pt1, pt2, threshold_override)
                    else:
                        # Fallback to standard computation
                        brightness_stats = self._compute_brightness_stats(roi, overall_background_brightness if overall_background_brightness is not None else self.manual_threshold)
                else:
                    # Standard ROI extraction and analysis
                    fh, fw = frame.shape[:2]
                    x1 = max(0, min(pt1[0], fw - 1))
                    y1 = max(0, min(pt1[1], fh - 1))
                    x2 = max(0, min(pt2[0], fw - 1))
                    y2 = max(0, min(pt2[1], fh - 1))
                    roi = frame[y1:y2, x1:x2]

                    # Get appropriate threshold for this ROI
                    active_threshold = self._get_active_threshold_for_roi(roi_idx)
                    if overall_background_brightness is not None and not self.force_manual_thresholds:
                        active_threshold = overall_background_brightness
                    
                    # Compute brightness stats
                    brightness_stats = self._compute_brightness_stats(roi, active_threshold)
                results[roi_idx].append((i,) + brightness_stats)

        progress.setValue(end_frame) # Ensure progress bar is full

        self._analysis_in_progress = False
        self._update_widget_states(video_loaded=True, rois_exist=True)
        self.statusBar().showMessage("Analysis complete.")

        # Post-analysis: Plotting and CSV export
        self._plot_results(results)
        self._export_results_to_csv(results)

    def _plot_results(self, results: List[List[Tuple[int, float, float, float, float, float, float, float, float]]]):
        """Plots the brightness analysis results."""
        if not results or all(not r for r in results):
            QtWidgets.QMessageBox.warning(self, 'Plotting Error', 'No analysis data to plot.')
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('Brightness Analysis Results', fontsize=16)

        # L* Channel Plot
        ax1 = axes[0]
        for roi_idx, roi_results in enumerate(results):
            if not roi_results:
                continue
            frames = [r[0] for r in roi_results]
            l_raw_means = [r[1] for r in roi_results]
            l_bg_sub_means = [r[3] for r in roi_results] # Background-subtracted mean

            color = plt.cm.get_cmap('tab10')(roi_idx) # Use a colormap for distinct colors
            ax1.plot(frames, l_raw_means, label=f'ROI {roi_idx+1} (Raw L*)', color=color, linestyle='-')
            if self.background_roi_idx is not None:
                ax1.plot(frames, l_bg_sub_means, label=f'ROI {roi_idx+1} (BG-Sub L*)', color=color, linestyle='--', alpha=0.7)
        
        ax1.set_ylabel('L* Brightness (0-100)')
        ax1.set_title('L* Brightness Over Time')
        ax1.legend()
        ax1.grid(True)

        # Blue Channel Plot
        ax2 = axes[1]
        for roi_idx, roi_results in enumerate(results):
            if not roi_results:
                continue
            frames = [r[0] for r in roi_results]
            b_raw_means = [r[5] for r in roi_results] # Raw blue mean
            b_bg_sub_means = [r[7] for r in roi_results] # Background-subtracted blue mean

            color = plt.cm.get_cmap('tab10')(roi_idx)
            ax2.plot(frames, b_raw_means, label=f'ROI {roi_idx+1} (Raw Blue)', color=color, linestyle='-')
            if self.background_roi_idx is not None:
                ax2.plot(frames, b_bg_sub_means, label=f'ROI {roi_idx+1} (BG-Sub Blue)', color=color, linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Blue Channel Value (0-255)')
        ax2.set_title('Blue Channel Value Over Time')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # Adjust layout to prevent title overlap
        
        # Save plot
        plot_filename = f"{self.analysis_name_input.text()}_brightness_analysis.png"
        plt.savefig(plot_filename)
        logger.info(f"Plot saved to {plot_filename}")
        self.out_paths.append(plot_filename)
        plt.close(fig) # Close the figure to free memory

    def _export_results_to_csv(self, results: List[List[Tuple[int, float, float, float, float, float, float, float, float]]]):
        """Exports the analysis results to a CSV file."""
        if not results or all(not r for r in results):
            QtWidgets.QMessageBox.warning(self, 'Export Error', 'No analysis data to export.')
            return

        all_data = []
        for roi_idx, roi_results in enumerate(results):
            for frame_data in roi_results:
                frame_num, l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median = frame_data
                all_data.append({
                    'ROI': roi_idx + 1,
                    'Frame': frame_num,
                    'L_Raw_Mean': l_raw_mean,
                    'L_Raw_Median': l_raw_median,
                    'L_BG_Sub_Mean': l_bg_sub_mean,
                    'L_BG_Sub_Median': l_bg_sub_median,
                    'Blue_Raw_Mean': b_raw_mean,
                    'Blue_Raw_Median': b_raw_median,
                    'Blue_BG_Sub_Mean': b_bg_sub_mean,
                    'Blue_BG_Sub_Median': b_bg_sub_median
                })
        
        if not all_data:
            QtWidgets.QMessageBox.warning(self, 'Export Error', 'No valid data collected for export.')
            return

        df = pd.DataFrame(all_data)
        csv_filename = f"{self.analysis_name_input.text()}_brightness_analysis.csv"
        df.to_csv(csv_filename, index=False)
        logger.info(f"Results exported to {csv_filename}")
        self.out_paths.append(csv_filename)

        self.results_label.setText(f"Analysis complete. Results saved to:\n"
                                   f"- {os.path.basename(csv_filename)}\n"
                                   f"- {os.path.basename(csv_filename.replace('.csv', '.png'))}")

    def _get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Retrieves a frame from the video, using cache if available."""
        cached_frame = self.frame_cache.get(frame_index) if self.frame_cache else None
        if cached_frame is not None:
            return cached_frame
        
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if ret:
                if self.frame_cache is not None:
                    self.frame_cache.put(frame_index, frame)
                return frame
        return None

    def _compute_background_brightness(self, frame: np.ndarray) -> Optional[float]:
        """Computes the background brightness from the designated background ROI."""
        if self.background_roi_idx is None or self.background_roi_idx >= len(self.rects):
            return None
        
        pt1, pt2 = self.rects[self.background_roi_idx]
        fh, fw = frame.shape[:2]
        x1 = max(0, min(pt1[0], fw - 1))
        y1 = max(0, min(pt1[1], fh - 1))
        x2 = max(0, min(pt2[0], fw - 1))
        y2 = max(0, min(pt2[1], fh - 1))
        
        if x2 > x1 and y2 > y1:
            bg_roi = frame[y1:y2, x1:x2]
            # Convert to LAB and get L* channel
            lab = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)
            l_star = l_chan * 100.0 / 255.0
            
            # Use 90th percentile for background brightness to be more robust to noise/small bright spots
            return float(np.percentile(l_star, 90))
        return None

    def _compute_brightness_stats_with_reference_mask(self, frame: np.ndarray, roi_idx: int, 
                                                    pt1: tuple, pt2: tuple, threshold: float) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Compute brightness statistics using reference mask for consistent pixel analysis.
        Only analyzes pixels that were identified in the reference mask.
        """
        if roi_idx not in self.reference_masks:
            # Fallback to standard analysis
            fh, fw = frame.shape[:2]
            x1 = max(0, min(pt1[0], fw - 1))
            y1 = max(0, min(pt1[1], fh - 1))
            x2 = max(0, min(pt2[0], fw - 1))
            y2 = max(0, min(pt2[1], fh - 1))
            roi = frame[y1:y2, x1:x2]
            return self._compute_brightness_stats(roi, threshold)
        
        # Get reference mask and metadata
        reference_mask = self.reference_masks[roi_idx]
        metadata = self.reference_mask_metadata[roi_idx]
        x1, y1, x2, y2 = metadata['roi_bounds']
        
        # Extract ROI from current frame
        roi = frame[y1:y2, x1:x2]
        
        # Convert to LAB color space
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        l_star = l_chan * 100.0 / 255.0  # Convert to 0-100 range
        blue_chan = roi[:, :, 0].astype(np.float32)  # Blue channel (0-255)
        
        # Apply reference mask to get only the consistent pixels
        if reference_mask.shape == l_star.shape:
            masked_l_star = l_star[reference_mask]
            masked_blue = blue_chan[reference_mask]
        else:
            logger.warning(f"Reference mask shape mismatch for ROI {roi_idx + 1}, falling back to threshold")
            mask = l_star > threshold
            masked_l_star = l_star[mask]
            masked_blue = blue_chan[mask]
        
        # Calculate statistics for masked pixels
        if len(masked_l_star) > 0:
            l_raw_mean = float(np.mean(masked_l_star))
            l_raw_median = float(np.median(masked_l_star))
            blue_mean = float(np.mean(masked_blue))
            blue_median = float(np.median(masked_blue))
        else:
            # No pixels pass the mask
            l_raw_mean = l_raw_median = blue_mean = blue_median = 0.0
        
        # Background subtraction (if threshold represents background)
        if threshold > 0:
            l_bg_sub_mean = max(0.0, l_raw_mean - threshold)
            l_bg_sub_median = max(0.0, l_raw_median - threshold)
        else:
            l_bg_sub_mean = l_raw_mean
            l_bg_sub_median = l_raw_median
        
        # Additional stats for compatibility
        l_enhanced_mean = l_raw_mean  # Could apply enhancement here if needed
        l_enhanced_median = l_raw_median
        
        logger.debug(f"Reference mask analysis ROI {roi_idx + 1}: {len(masked_l_star)} pixels, "
                    f"L* mean: {l_raw_mean:.2f}, blue mean: {blue_mean:.2f}")
        
        return (l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, 
                blue_mean, blue_median, l_enhanced_mean, l_enhanced_median)

    def _compute_brightness_stats(self, roi_bgr: np.ndarray, background_brightness: Optional[float] = None) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Calculates brightness statistics for an ROI with optional background subtraction.
        Uses low-light enhancement if enabled for better results in low-light conditions.

        Converts BGR to CIE LAB color space and uses the L* channel.
        Also extracts blue channel statistics for blue light analysis.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            # Use low-light enhancement if enabled
            if hasattr(self, 'low_light_enhancer') and self.low_light_enhancer.enabled:
                return self.low_light_enhancer.compute_enhanced_brightness_stats(roi_bgr, background_brightness)
            else:
                # Fall back to standard calculation
                return self._compute_standard_brightness_stats(roi_bgr, background_brightness)
            
        except Exception as e:
            logger.error(f"Error computing brightness statistics: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    def _compute_standard_brightness_stats(self, roi_bgr: np.ndarray, background_brightness: Optional[float] = None) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Standard brightness calculation without enhancement.
        
        Args:
            roi_bgr: The region of interest as a NumPy array (BGR format).
            background_brightness: Optional background L* value to subtract from all pixels.

        Returns:
            Tuple of brightness statistics.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)
            l_star = l_chan * 100.0 / 255.0
            
            # Extract blue channel
            blue_chan = roi_bgr[:, :, 0].astype(np.float32)
            
            # Calculate raw statistics
            l_raw_mean = float(np.mean(l_star))
            l_raw_median = float(np.median(l_star))
            b_raw_mean = float(np.mean(blue_chan))
            b_raw_median = float(np.median(blue_chan))
            
            # Calculate background-subtracted statistics
            if background_brightness is not None:
                above_background_mask = l_star > background_brightness
                
                # Apply morphological cleanup if enabled
                if self.low_light_enhancer.morphological_cleanup and np.any(above_background_mask):
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask_uint8 = above_background_mask.astype(np.uint8) * 255
                    cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
                    above_background_mask = cleaned_mask > 0
                
                if np.any(above_background_mask):
                    filtered_l_pixels = l_star[above_background_mask]
                    filtered_b_pixels = blue_chan[above_background_mask]
                    
                    # Apply additional noise filtering to extracted pixels
                    if self.low_light_enhancer.gaussian_blur_sigma > 0 and len(filtered_l_pixels) > 10:
                        # Use robust statistics for noisy signals
                        l_bg_sub_mean = float(np.mean(filtered_l_pixels) - background_brightness)
                        l_bg_sub_median = float(np.median(filtered_l_pixels) - background_brightness)
                        
                        # Apply Gaussian smoothing to reduce noise in statistics
                        l_values = filtered_l_pixels - background_brightness
                        if len(l_values) > 5:
                            # Use trimmed mean for better noise resistance
                            l_trimmed = np.sort(l_values)[int(len(l_values)*0.1):int(len(l_values)*0.9)]
                            if len(l_trimmed) > 0:
                                l_bg_sub_mean = float(np.mean(l_trimmed))
                    else:
                        l_bg_sub_mean = float(np.mean(filtered_l_pixels) - background_brightness)
                        l_bg_sub_median = float(np.median(filtered_l_pixels) - background_brightness)
                    
                    b_bg_sub_mean = float(np.mean(filtered_b_pixels))
                    b_bg_sub_median = float(np.median(filtered_b_pixels))
                else:
                    l_bg_sub_mean = l_bg_sub_median = b_bg_sub_mean = b_bg_sub_median = 0.0
            else:
                l_bg_sub_mean = l_raw_mean
                l_bg_sub_median = l_raw_median
                b_bg_sub_mean = b_raw_mean
                b_bg_sub_median = b_raw_median
            
            return (l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median,
                   b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median)
            
        except Exception as e:
            logger.error(f"Error computing brightness statistics: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def get_roi_brightness(self, roi_bgr: np.ndarray) -> float:
        """Helper to get L* mean brightness of an ROI."""
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0
        l_raw_mean, _, _, _, _, _, _, _ = self._compute_brightness_stats(roi_bgr)
        return l_raw_mean
    
def main():
    """Main application entry point with comprehensive error handling."""
    # Initialize application
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setApplicationDisplayName(f"{APP_NAME} v{APP_VERSION}")
    app.setOrganizationName("Brightness Sorcerer Development Team")
    app.setOrganizationDomain("brightnesssorcerer.dev")
    
    # Set application icon (if available)
    try:
        from pathlib import Path
        icon_path = Path("icon.ico")
        if icon_path.exists():
            app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    except Exception as e:
        logger.debug(f"Could not load application icon: {e}")
    
    # Global exception handler
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.critical(f"Uncaught exception: {error_msg}")
        
        # Show error dialog
        error_dialog = QtWidgets.QMessageBox()
        error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
        error_dialog.setWindowTitle("Critical Error")
        error_dialog.setText("An unexpected error occurred.")
        error_dialog.setDetailedText(error_msg)
        error_dialog.setInformativeText("The application will now close. Please check the log file for details.")
        error_dialog.exec_()
    
    # Install exception handler
    sys.excepthook = handle_exception
    
    try:
        # Create and show the main window
        logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
        win = VideoAnalyzer()
        
        # Center window on screen
        screen = app.desktop().screenGeometry()
        window = win.geometry()
        win.move((screen.width() - window.width()) // 2, 
                 (screen.height() - window.height()) // 2)
        
        win.show()
        logger.info("Application startup complete")
        
        # Run application
        exit_code = app.exec_()
        logger.info(f"Application exiting with code {exit_code}")
        return exit_code
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        
        # Show startup error dialog
        error_dialog = QtWidgets.QMessageBox()
        error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
        error_dialog.setWindowTitle("Startup Error")
        error_dialog.setText(f"Failed to start {APP_NAME}")
        error_dialog.setInformativeText(str(e))
        error_dialog.exec_()
        
        return 1
    
    finally:
        logger.info("Application cleanup complete")

if __name__ == '__main__':
    sys.exit(main())