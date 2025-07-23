#!/usr/bin/env python3
"""
ROIManager - Advanced ROI and reference mask management.

This module provides comprehensive ROI (Region of Interest) management including:
- Dynamic ROI creation, modification, and deletion
- Background ROI selection and management  
- Reference mask generation and validation
- ROI quality assessment and optimization
- Coordinate validation and bounds checking
- Persistent ROI state management

Professional-grade tools for consistent and reliable spatial analysis.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class ROIManager:
    """
    Advanced ROI and reference mask management system.
    
    Features:
    - Dynamic ROI creation and management
    - Background ROI selection and validation
    - Reference mask generation with quality metrics
    - Coordinate bounds checking and validation
    - ROI quality assessment and optimization
    - State persistence and restoration
    """
    
    def __init__(self):
        """Initialize ROI manager with empty state."""
        # Core ROI storage
        self.rects: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.selected_rect_idx: Optional[int] = None
        self.background_roi_idx: Optional[int] = None
        
        # Reference mask system
        self.reference_masks: Dict[int, np.ndarray] = {}
        self.reference_frame_idx: Optional[int] = None
        self.mask_metadata: Dict[int, Dict[str, Any]] = {}
        
        # ROI state tracking
        self.locked_roi: Optional[Dict[str, Any]] = None
        
        logger.debug("ROIManager initialized")
    
    def add_roi(self, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                frame_shape: Optional[Tuple[int, int]] = None) -> int:
        """
        Add new ROI with coordinate validation.
        
        Args:
            pt1: First corner point (x, y)
            pt2: Second corner point (x, y) 
            frame_shape: Optional frame shape (height, width) for bounds checking
            
        Returns:
            Index of added ROI
        """
        # Validate and normalize coordinates
        if frame_shape is not None:
            pt1, pt2 = self._validate_roi_coordinates(pt1, pt2, frame_shape)
        
        # Add ROI to collection
        roi_idx = len(self.rects)
        self.rects.append((pt1, pt2))
        
        logger.debug(f"Added ROI {roi_idx}: ({pt1[0]},{pt1[1]})-({pt2[0]},{pt2[1]})")
        return roi_idx
    
    def delete_roi(self, roi_idx: int) -> bool:
        """
        Delete ROI by index with cleanup.
        
        Args:
            roi_idx: Index of ROI to delete
            
        Returns:
            True if ROI was deleted successfully
        """
        if not self.is_valid_roi_index(roi_idx):
            return False
        
        # Remove ROI
        del self.rects[roi_idx]
        
        # Clean up related state
        self._cleanup_roi_state_after_deletion(roi_idx)
        
        logger.debug(f"Deleted ROI {roi_idx}")
        return True
    
    def clear_all_rois(self):
        """Clear all ROIs and associated state."""
        self.rects.clear()
        self.selected_rect_idx = None
        self.background_roi_idx = None
        self.reference_masks.clear()
        self.mask_metadata.clear()
        self.locked_roi = None
        
        logger.debug("Cleared all ROIs")
    
    def select_roi(self, roi_idx: int) -> bool:
        """
        Select ROI by index.
        
        Args:
            roi_idx: Index of ROI to select
            
        Returns:
            True if ROI was selected successfully
        """
        if not self.is_valid_roi_index(roi_idx):
            return False
        
        self.selected_rect_idx = roi_idx
        logger.debug(f"Selected ROI {roi_idx}")
        return True
    
    def set_background_roi(self, roi_idx: int) -> bool:
        """
        Set ROI as background for subtraction.
        
        Args:
            roi_idx: Index of ROI to use as background
            
        Returns:
            True if background ROI was set successfully
        """
        if not self.is_valid_roi_index(roi_idx):
            return False
        
        self.background_roi_idx = roi_idx
        logger.debug(f"Set background ROI: {roi_idx}")
        return True
    
    def generate_reference_masks(self, frame: np.ndarray, threshold: float,
                               mask_generation_method: str = 'threshold') -> int:
        """
        Generate reference masks for all ROIs using current frame.
        
        Args:
            frame: Reference frame for mask generation
            threshold: Brightness threshold for mask creation
            mask_generation_method: Method for mask generation
            
        Returns:
            Number of masks generated successfully
        """
        if frame is None or len(self.rects) == 0:
            return 0
        
        self.reference_frame_idx = 0  # Would be set by caller
        masks_generated = 0
        
        for roi_idx, (pt1, pt2) in enumerate(self.rects):
            try:
                mask, metadata = self._generate_roi_reference_mask(
                    frame, pt1, pt2, roi_idx, threshold, mask_generation_method
                )
                
                if mask is not None:
                    self.reference_masks[roi_idx] = mask
                    self.mask_metadata[roi_idx] = metadata
                    masks_generated += 1
                    
            except Exception as e:
                logger.error(f"Failed to generate reference mask for ROI {roi_idx}: {e}")
                continue
        
        logger.info(f"Generated {masks_generated} reference masks")
        return masks_generated
    
    def _generate_roi_reference_mask(self, frame: np.ndarray, pt1: Tuple[int, int], 
                                   pt2: Tuple[int, int], roi_idx: int, threshold: float,
                                   mask_generation_method: str) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Generate reference mask for single ROI.
        
        Args:
            frame: Video frame
            pt1, pt2: ROI corner points
            roi_idx: ROI index
            threshold: Brightness threshold
            mask_generation_method: Mask generation method
            
        Returns:
            Tuple of (mask, metadata) or (None, {}) if failed
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = self._normalize_roi_bounds(pt1, pt2, fw, fh)
        
        if x2 <= x1 or y2 <= y1:
            return None, {}
        
        try:
            # Extract ROI and convert to LAB
            roi = frame[y1:y2, x1:x2]
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)
            l_star = l_chan * 100.0 / 255.0
            
            # Generate mask based on method
            if mask_generation_method == 'threshold':
                mask = l_star > threshold
            elif mask_generation_method == 'adaptive':
                mask = self._generate_adaptive_mask(l_star, threshold)
            else:
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
                'threshold': threshold,
                'method': mask_generation_method,
                'total_pixels': total_pixels,
                'active_pixels': active_pixels,
                'coverage_percentage': coverage_percentage,
                'brightness_mean': brightness_mean,
                'brightness_std': brightness_std,
                'brightness_median': brightness_median,
                'roi_bounds': (x1, y1, x2, y2)
            }
            
            return mask, metadata
            
        except Exception as e:
            logger.error(f"Error generating reference mask for ROI {roi_idx}: {e}")
            return None, {}
    
    def _generate_adaptive_mask(self, l_star: np.ndarray, threshold: float) -> np.ndarray:
        """Generate adaptive mask using multiple thresholding methods."""
        try:
            # Convert to uint8 for OpenCV
            l_uint8 = (l_star * 255.0 / 100.0).astype(np.uint8)
            
            # Apply adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                l_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Combine with simple threshold
            threshold_mask = l_star > threshold
            adaptive_mask = adaptive_thresh > 0
            
            # Use intersection for selective masking
            return threshold_mask & adaptive_mask
            
        except Exception as e:
            logger.error(f"Error in adaptive mask generation: {e}")
            return l_star > threshold
    
    def get_roi_bounds(self, roi_idx: int, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get validated ROI bounds.
        
        Args:
            roi_idx: ROI index
            frame_shape: Frame shape (height, width)
            
        Returns:
            Tuple of (x1, y1, x2, y2) or None if invalid
        """
        if not self.is_valid_roi_index(roi_idx):
            return None
        
        pt1, pt2 = self.rects[roi_idx]
        fh, fw = frame_shape
        return self._normalize_roi_bounds(pt1, pt2, fw, fh)
    
    def extract_roi_from_frame(self, frame: np.ndarray, roi_idx: int) -> Optional[np.ndarray]:
        """
        Extract ROI data from frame with bounds validation.
        
        Args:
            frame: Video frame
            roi_idx: ROI index
            
        Returns:
            ROI data as numpy array or None if invalid
        """
        if frame is None or not self.is_valid_roi_index(roi_idx):
            return None
        
        bounds = self.get_roi_bounds(roi_idx, frame.shape[:2])
        if bounds is None:
            return None
        
        x1, y1, x2, y2 = bounds
        if x2 <= x1 or y2 <= y1:
            return None
        
        return frame[y1:y2, x1:x2]
    
    def find_optimal_roi_location(self, frame: np.ndarray, roi_size: Tuple[int, int],
                                search_region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int]]:
        """
        Find optimal location for ROI based on brightness variance.
        
        Args:
            frame: Video frame
            roi_size: Desired ROI size (width, height)
            search_region: Optional search bounds (x1, y1, x2, y2)
            
        Returns:
            Optimal top-left corner (x, y) or None if not found
        """
        if frame is None:
            return None
        
        fh, fw = frame.shape[:2]
        roi_w, roi_h = roi_size
        
        # Set search bounds
        if search_region is not None:
            x1, y1, x2, y2 = search_region
            x1 = max(0, min(x1, fw - roi_w))
            y1 = max(0, min(y1, fh - roi_h))
            x2 = min(fw - roi_w, x2)
            y2 = min(fh - roi_h, y2)
        else:
            x1, y1 = 0, 0
            x2, y2 = fw - roi_w, fh - roi_h
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            best_variance = -1
            best_location = None
            
            # Sample search space (every 10 pixels for performance)
            for y in range(y1, y2, 10):
                for x in range(x1, x2, 10):
                    roi = gray[y:y+roi_h, x:x+roi_w]
                    variance = np.var(roi)
                    
                    if variance > best_variance:
                        best_variance = variance
                        best_location = (x, y)
            
            return best_location
            
        except Exception as e:
            logger.error(f"Error finding optimal ROI location: {e}")
            return None
    
    def validate_roi_quality(self, roi_data: np.ndarray, min_pixels: int = 100,
                           min_variance: float = 10.0) -> bool:
        """
        Validate ROI has sufficient quality for analysis.
        
        Args:
            roi_data: ROI pixel data
            min_pixels: Minimum pixel count
            min_variance: Minimum brightness variance
            
        Returns:
            True if ROI meets quality requirements
        """
        if roi_data is None or roi_data.size == 0:
            return False
        
        if roi_data.size < min_pixels:
            return False
        
        try:
            # Check brightness variance
            gray = cv2.cvtColor(roi_data, cv2.COLOR_BGR2GRAY) if len(roi_data.shape) == 3 else roi_data
            variance = np.var(gray)
            
            return variance >= min_variance
            
        except Exception:
            return False
    
    def get_roi_info_text(self, roi_idx: int) -> str:
        """
        Get formatted ROI information text.
        
        Args:
            roi_idx: ROI index
            
        Returns:
            Formatted ROI info string
        """
        if not self.is_valid_roi_index(roi_idx):
            return "Invalid ROI"
        
        pt1, pt2 = self.rects[roi_idx]
        x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
        x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        prefix = "* " if roi_idx == self.background_roi_idx else ""
        locked_indicator = "(Locked) " if self.is_roi_locked(roi_idx) else ""
        
        return f"{prefix}{locked_indicator}ROI {roi_idx+1}: ({x1},{y1})-({x2},{y2}) [{width}×{height}={area}px]"
    
    def is_valid_roi_index(self, roi_idx: int) -> bool:
        """Check if ROI index is valid."""
        return 0 <= roi_idx < len(self.rects)
    
    def is_roi_locked(self, roi_idx: int) -> bool:
        """Check if ROI is currently locked."""
        return (self.locked_roi is not None and 
                self.locked_roi.get('roi_idx') == roi_idx)
    
    def get_roi_count(self) -> int:
        """Get total number of ROIs."""
        return len(self.rects)
    
    def _validate_roi_coordinates(self, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                                 frame_shape: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Validate and clamp ROI coordinates to frame bounds."""
        fh, fw = frame_shape
        
        x1 = max(0, min(pt1[0], fw - 1))
        y1 = max(0, min(pt1[1], fh - 1))
        x2 = max(0, min(pt2[0], fw - 1))
        y2 = max(0, min(pt2[1], fh - 1))
        
        return (x1, y1), (x2, y2)
    
    def _normalize_roi_bounds(self, pt1: Tuple[int, int], pt2: Tuple[int, int],
                             frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Normalize ROI bounds to ensure proper ordering."""
        x1 = max(0, min(pt1[0], frame_width - 1))
        y1 = max(0, min(pt1[1], frame_height - 1))
        x2 = max(0, min(pt2[0], frame_width - 1))
        y2 = max(0, min(pt2[1], frame_height - 1))
        
        # Ensure proper ordering
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    
    def _cleanup_roi_state_after_deletion(self, deleted_idx: int):
        """Clean up state after ROI deletion."""
        # Adjust background ROI index
        if self.background_roi_idx is not None:
            if self.background_roi_idx == deleted_idx:
                self.background_roi_idx = None
            elif self.background_roi_idx > deleted_idx:
                self.background_roi_idx -= 1
        
        # Adjust selected ROI index
        if self.selected_rect_idx is not None:
            if self.selected_rect_idx == deleted_idx:
                self.selected_rect_idx = None
            elif self.selected_rect_idx > deleted_idx:
                self.selected_rect_idx -= 1
        
        # Clean up reference masks (shift indices)
        new_masks = {}
        new_metadata = {}
        for roi_idx, mask in self.reference_masks.items():
            if roi_idx < deleted_idx:
                new_masks[roi_idx] = mask
                new_metadata[roi_idx] = self.mask_metadata.get(roi_idx, {})
            elif roi_idx > deleted_idx:
                new_masks[roi_idx - 1] = mask
                new_metadata[roi_idx - 1] = self.mask_metadata.get(roi_idx, {})
        
        self.reference_masks = new_masks
        self.mask_metadata = new_metadata
        
        # Clean up locked ROI
        if self.locked_roi and self.locked_roi.get('roi_idx') == deleted_idx:
            self.locked_roi = None
    
    def cleanup(self):
        """Clean up resources and force garbage collection."""
        # Clear large data structures
        self.reference_masks.clear()
        self.mask_metadata.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        memory_info = {
            'roi_count': len(self.rects),
            'mask_count': len(self.reference_masks),
            'total_mask_pixels': 0
        }
        
        # Calculate total mask memory usage
        for mask in self.reference_masks.values():
            if mask is not None:
                memory_info['total_mask_pixels'] += mask.size
        
        return memory_info