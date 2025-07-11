"""Color space conversion and brightness calculation utilities."""

import cv2
import numpy as np
from typing import Tuple, Dict, Any
import logging

# Constants
BRIGHTNESS_NOISE_FLOOR = 10.0  # Filter out dark pixels below this L* value
BASELINE_PERCENTILE = 5
BRIGHTNESS_PERCENTILE = 2


def convert_bgr_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to CIE LAB color space."""
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    except Exception as e:
        logging.error(f"Error converting BGR to LAB: {e}")
        return image


def calculate_brightness_stats(roi_region: np.ndarray, 
                              noise_floor: float = BRIGHTNESS_NOISE_FLOOR) -> Dict[str, float]:
    """Calculate comprehensive brightness statistics for an ROI region."""
    if roi_region is None or roi_region.size == 0:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'valid_pixels': 0}
    
    try:
        # Convert to LAB color space
        lab_region = convert_bgr_to_lab(roi_region)
        
        # Extract L* channel (brightness)
        l_channel = lab_region[:, :, 0].astype(np.float32)
        
        # Filter out dark pixels (noise)
        valid_pixels = l_channel[l_channel >= noise_floor]
        
        if valid_pixels.size == 0:
            return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'valid_pixels': 0}
        
        # Calculate statistics
        stats = {
            'mean': float(np.mean(valid_pixels)),
            'median': float(np.median(valid_pixels)),
            'std': float(np.std(valid_pixels)),
            'valid_pixels': int(valid_pixels.size),
            'min': float(np.min(valid_pixels)),
            'max': float(np.max(valid_pixels)),
            'percentile_25': float(np.percentile(valid_pixels, 25)),
            'percentile_75': float(np.percentile(valid_pixels, 75))
        }
        
        return stats
        
    except Exception as e:
        logging.error(f"Error calculating brightness stats: {e}")
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'valid_pixels': 0}


def calculate_baseline_brightness(roi_region: np.ndarray, 
                                 percentile: float = BASELINE_PERCENTILE,
                                 noise_floor: float = BRIGHTNESS_NOISE_FLOOR) -> float:
    """Calculate baseline brightness using percentile method."""
    if roi_region is None or roi_region.size == 0:
        return 0.0
    
    try:
        # Convert to LAB and extract L* channel
        lab_region = convert_bgr_to_lab(roi_region)
        l_channel = lab_region[:, :, 0].astype(np.float32)
        
        # Filter out dark pixels
        valid_pixels = l_channel[l_channel >= noise_floor]
        
        if valid_pixels.size == 0:
            return 0.0
        
        return float(np.percentile(valid_pixels, percentile))
        
    except Exception as e:
        logging.error(f"Error calculating baseline brightness: {e}")
        return 0.0


def detect_brightness_threshold(background_roi_region: np.ndarray,
                               manual_threshold: float = 5.0,
                               noise_floor: float = BRIGHTNESS_NOISE_FLOOR) -> float:
    """Detect brightness threshold for analysis."""
    if background_roi_region is None:
        return manual_threshold
    
    try:
        baseline = calculate_baseline_brightness(background_roi_region, 
                                               BASELINE_PERCENTILE, 
                                               noise_floor)
        return baseline + manual_threshold
        
    except Exception as e:
        logging.error(f"Error detecting brightness threshold: {e}")
        return manual_threshold


def get_lab_l_channel(image: np.ndarray) -> np.ndarray:
    """Extract L* channel from BGR image."""
    lab_image = convert_bgr_to_lab(image)
    return lab_image[:, :, 0].astype(np.float32)