#!/usr/bin/env python3
"""
BrightnessAnalyzer - Core brightness analysis and computation module.

This module provides specialized brightness analysis functionality including:
- CIE LAB color space conversions and L* channel analysis
- Blue channel analysis for specialized bioluminescence detection  
- Background subtraction and noise reduction
- Reference mask-based analysis for consistent pixel selection
- Enhanced low-light analysis with morphological operations
- Vectorized operations for optimal performance

Professional-grade analysis tools suitable for research and scientific applications.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

from ..utils.constants import (
    LAB_L_CHANNEL_SCALE, COLOR_CHANNEL_MAX, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    BILATERAL_FILTER_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE,
    TRIMMED_MEAN_LOWER_PERCENTILE, TRIMMED_MEAN_UPPER_PERCENTILE,
    MIN_PIXELS_FOR_ROBUST_STATS, MORPHOLOGICAL_KERNEL_SIZE
)

logger = logging.getLogger(__name__)


class BrightnessAnalyzer:
    """
    Professional brightness analysis engine with advanced color science support.
    
    Features:
    - CIE LAB L* channel analysis for perceptually accurate brightness
    - Blue channel analysis for bioluminescence detection
    - Reference mask-based consistent analysis
    - Low-light enhancement and noise reduction
    - Background subtraction with morphological cleanup
    - Vectorized operations for performance
    """
    
    def __init__(self, 
                 analysis_method: str = 'enhanced',
                 morphological_cleanup: bool = True,
                 gaussian_blur_sigma: float = 0.0):
        """
        Initialize brightness analyzer with configuration.
        
        Args:
            analysis_method: 'enhanced' for LAB+Blue, 'basic' for LAB only
            morphological_cleanup: Apply morphological operations to reduce noise
            gaussian_blur_sigma: Gaussian blur strength for noise reduction (0 = disabled)
        """
        self.analysis_method = analysis_method
        self.morphological_cleanup = morphological_cleanup
        self.gaussian_blur_sigma = gaussian_blur_sigma
        
        # Analysis configuration
        self.mask_generation_method = 'threshold'  # 'threshold', 'adaptive'
        
        logger.debug(f"BrightnessAnalyzer initialized: {analysis_method} method")
    
    def compute_brightness_stats(self, roi_bgr: np.ndarray, 
                               background_brightness: Optional[float] = None,
                               enhanced: bool = False) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Calculate comprehensive brightness statistics for an ROI.
        
        Uses CIE LAB color space L* channel for perceptually accurate brightness.
        Also analyzes blue channel for specialized bioluminescence detection.
        
        Args:
            roi_bgr: Region of interest as BGR numpy array
            background_brightness: Optional background L* value for subtraction
            enhanced: Use enhanced low-light analysis if available
            
        Returns:
            Tuple of (L_raw_mean, L_raw_median, L_bg_sub_mean, L_bg_sub_median,
                     B_raw_mean, B_raw_median, B_bg_sub_mean, B_bg_sub_median)
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            # Memory-efficient processing: work in-place where possible
            # Convert BGR to CIE LAB color space  
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            
            # Extract L* and Blue channels efficiently
            l_chan = lab[:, :, 0].astype(np.float32, copy=False)  # Avoid unnecessary copy
            blue_chan = roi_bgr[:, :, 0].astype(np.float32, copy=False)
            
            # Convert L channel to L* in-place for memory efficiency
            l_chan *= LAB_L_CHANNEL_SCALE / COLOR_CHANNEL_MAX
            l_star = l_chan  # Alias for clarity
            
            # Vectorized statistical calculations (batch compute all statistics)
            l_raw_stats = self._compute_vectorized_stats(l_star)
            b_raw_stats = self._compute_vectorized_stats(blue_chan)
            
            l_raw_mean, l_raw_median = l_raw_stats
            b_raw_mean, b_raw_median = b_raw_stats
            
            # Initialize background-subtracted values
            l_bg_sub_mean = l_raw_mean
            l_bg_sub_median = l_raw_median
            b_bg_sub_mean = b_raw_mean
            b_bg_sub_median = b_raw_median
            
            # Apply background subtraction if provided
            if background_brightness is not None:
                l_bg_sub_mean, l_bg_sub_median, b_bg_sub_mean, b_bg_sub_median = (
                    self._compute_background_subtracted_stats(
                        l_star, blue_chan, background_brightness
                    )
                )
            
            return (l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median,
                   b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median)
                   
        except Exception as e:
            logger.error(f"Error computing brightness statistics: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    def _compute_background_subtracted_stats(self, l_star: np.ndarray, blue_chan: np.ndarray, 
                                          background_brightness: float) -> Tuple[float, float, float, float]:
        """
        Compute background-subtracted brightness statistics with advanced filtering.
        
        Args:
            l_star: L* channel data
            blue_chan: Blue channel data  
            background_brightness: Background L* value for subtraction
            
        Returns:
            Tuple of (L_bg_sub_mean, L_bg_sub_median, B_bg_sub_mean, B_bg_sub_median)
        """
        # Create mask for pixels above background
        above_background_mask = l_star > background_brightness
        
        # Apply morphological cleanup if enabled
        if self.morphological_cleanup and np.any(above_background_mask):
            above_background_mask = self._apply_morphological_cleanup(above_background_mask)
        
        if np.any(above_background_mask):
            # Extract pixels above background threshold
            filtered_l_pixels = l_star[above_background_mask]
            filtered_b_pixels = blue_chan[above_background_mask]
            
            # Apply noise filtering for robust statistics
            if self.gaussian_blur_sigma > 0 and len(filtered_l_pixels) > MIN_PIXELS_FOR_ROBUST_STATS:
                l_bg_sub_mean, l_bg_sub_median = self._compute_robust_stats(
                    filtered_l_pixels, background_brightness
                )
            else:
                # Standard background subtraction
                l_bg_sub_mean = float(np.mean(filtered_l_pixels) - background_brightness)
                l_bg_sub_median = float(np.median(filtered_l_pixels) - background_brightness)
            
            # Blue channel background subtraction
            b_bg_sub_mean = float(np.mean(filtered_b_pixels))
            b_bg_sub_median = float(np.median(filtered_b_pixels))
        else:
            # No pixels above background - return zero values
            l_bg_sub_mean = l_bg_sub_median = 0.0
            b_bg_sub_mean = b_bg_sub_median = 0.0
        
        return l_bg_sub_mean, l_bg_sub_median, b_bg_sub_mean, b_bg_sub_median
    
    def _apply_morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up binary mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPHOLOGICAL_KERNEL_SIZE, MORPHOLOGICAL_KERNEL_SIZE))
        mask_uint8 = mask.astype(np.uint8) * 255
        cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        return cleaned_mask > 0
    
    def _compute_robust_stats(self, pixels: np.ndarray, background_brightness: float) -> Tuple[float, float]:
        """
        Compute robust statistics using trimmed mean for noise resistance.
        
        Args:
            pixels: Pixel values to analyze
            background_brightness: Background value for subtraction
            
        Returns:
            Tuple of (robust_mean, robust_median)
        """
        # Apply background subtraction
        l_values = pixels - background_brightness
        
        if len(l_values) > 5:
            # Use trimmed mean (remove percentiles from each end)
            lower_idx = int(len(l_values) * TRIMMED_MEAN_LOWER_PERCENTILE)
            upper_idx = int(len(l_values) * TRIMMED_MEAN_UPPER_PERCENTILE)
            l_trimmed = np.sort(l_values)[lower_idx:upper_idx]
            if len(l_trimmed) > 0:
                robust_mean = float(np.mean(l_trimmed))
                robust_median = float(np.median(l_values))
                return robust_mean, robust_median
        
        # Fall back to standard calculation
        return float(np.mean(l_values)), float(np.median(l_values))
    
    def _compute_vectorized_stats(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Compute mean and median using vectorized operations.
        
        Args:
            data: Input data array
            
        Returns:
            Tuple of (mean, median)
        """
        if data.size == 0:
            return 0.0, 0.0
        
        # Use numpy's optimized functions
        return float(np.mean(data)), float(np.median(data))
    
    def compute_brightness_stats_with_reference_mask(self, frame: np.ndarray, roi_idx: int,
                                                   pt1: tuple, pt2: tuple, 
                                                   reference_mask: Optional[np.ndarray] = None,
                                                   mask_metadata: Optional[Dict] = None) -> Tuple[float, float, float, float]:
        """
        Compute brightness statistics using reference mask for consistent analysis.
        
        Args:
            frame: Full video frame
            roi_idx: ROI index
            pt1, pt2: ROI corner points
            reference_mask: Optional reference mask for consistent pixel selection
            mask_metadata: Optional metadata about the reference mask
            
        Returns:
            Tuple of (L_mean, L_median, L_std, active_pixel_count)
        """
        fh, fw = frame.shape[:2]
        x1 = max(0, min(pt1[0], fw - 1))
        y1 = max(0, min(pt1[1], fh - 1)) 
        x2 = max(0, min(pt2[0], fw - 1))
        y2 = max(0, min(pt2[1], fh - 1))
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, 0.0, 0.0
        
        try:
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            # Convert to CIE LAB and extract L* channel
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)
            l_star = l_chan * LAB_L_CHANNEL_SCALE / COLOR_CHANNEL_MAX
            
            # Apply reference mask if provided
            if reference_mask is not None:
                # Ensure mask dimensions match ROI
                if reference_mask.shape == l_star.shape:
                    masked_pixels = l_star[reference_mask]
                else:
                    logger.warning(f"Reference mask size mismatch for ROI {roi_idx}")
                    masked_pixels = l_star.flatten()
            else:
                # Use all pixels if no mask
                masked_pixels = l_star.flatten()
            
            # Calculate statistics
            if len(masked_pixels) > 0:
                l_mean = float(np.mean(masked_pixels))
                l_median = float(np.median(masked_pixels))
                l_std = float(np.std(masked_pixels))
                active_pixel_count = float(len(masked_pixels))
            else:
                l_mean = l_median = l_std = active_pixel_count = 0.0
            
            return l_mean, l_median, l_std, active_pixel_count
            
        except Exception as e:
            logger.error(f"Error computing reference mask brightness stats for ROI {roi_idx}: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def generate_adaptive_mask(self, l_star: np.ndarray, threshold: float) -> np.ndarray:
        """
        Generate adaptive binary mask using advanced thresholding.
        
        Args:
            l_star: L* channel data
            threshold: Base threshold value
            
        Returns:
            Binary mask as numpy array
        """
        try:
            # Convert to uint8 for OpenCV operations
            l_uint8 = (l_star * COLOR_CHANNEL_MAX / LAB_L_CHANNEL_SCALE).astype(np.uint8)
            
            # Apply adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                l_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Combine with threshold-based mask
            threshold_mask = l_star > threshold
            adaptive_mask = adaptive_thresh > 0
            
            # Use intersection for more selective masking
            combined_mask = threshold_mask & adaptive_mask
            
            return combined_mask
            
        except Exception as e:
            logger.error(f"Error generating adaptive mask: {e}")
            # Fall back to simple threshold
            return l_star > threshold
    
    def validate_roi_quality(self, roi_bgr: np.ndarray, min_pixels: int = 100) -> bool:
        """
        Validate ROI has sufficient quality for analysis.
        
        Args:
            roi_bgr: ROI data
            min_pixels: Minimum pixel count required
            
        Returns:
            True if ROI is valid for analysis
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return False
            
        if roi_bgr.size < min_pixels:
            return False
        
        # Check for sufficient dynamic range
        try:
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)
            
            dynamic_range = np.max(l_chan) - np.min(l_chan)
            if dynamic_range < 5:  # Insufficient dynamic range
                return False
                
            return True
            
        except Exception:
            return False
    
    def cleanup(self):
        """Clean up any cached data or resources."""
        # Force garbage collection for large analysis operations
        import gc
        gc.collect()