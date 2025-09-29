"""Brightness analysis module for ROI brightness calculations."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from ..constants import MORPHOLOGICAL_KERNEL_SIZE


class BrightnessAnalyzer:
    """Analyzes brightness in ROIs using CIE LAB color space and blue channel extraction."""

    def __init__(
        self,
        morphological_kernel_size: int = MORPHOLOGICAL_KERNEL_SIZE,
        noise_floor_threshold: float = 0.0
    ):
        """
        Initialize brightness analyzer.

        Args:
            morphological_kernel_size: Size of morphological kernel for noise filtering
            noise_floor_threshold: Absolute L* threshold for noise floor filtering
        """
        self.morphological_kernel_size = morphological_kernel_size
        self.noise_floor_threshold = noise_floor_threshold

    def compute_brightness_stats(
        self,
        roi_bgr: np.ndarray,
        background_brightness: Optional[float] = None,
        roi_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Calculate brightness statistics for an ROI with optional background subtraction.

        Converts BGR to CIE LAB color space and uses the L* channel.
        Also extracts blue channel statistics for blue light analysis.

        Args:
            roi_bgr: The region of interest as a NumPy array (BGR format)
            background_brightness: Optional background L* value to subtract from all pixels
            roi_mask: Optional boolean mask selecting pixels to analyze within ROI

        Returns:
            Tuple of (l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median,
                     b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median)
            L* values in 0-100 range, Blue values in 0-255 range
            or (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if the ROI is invalid or calculation fails.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)

            # If a fixed ROI mask is provided, use it directly
            if roi_mask is not None:
                return self._compute_with_fixed_mask(
                    roi_bgr, l_chan, roi_mask, background_brightness
                )

            # Convert raw L to L* scale (0-100)
            l_star = l_chan * 100.0 / 255.0

            # Extract blue channel (BGR format, so blue is index 0)
            blue_chan = roi_bgr[:, :, 0].astype(np.float32)

            # Calculate raw L* statistics (unthresholded)
            l_raw_mean = float(np.mean(l_star))
            l_raw_median = float(np.median(l_star))

            # Calculate raw blue statistics (unthresholded)
            b_raw_mean = float(np.mean(blue_chan))
            b_raw_median = float(np.median(blue_chan))

            # Calculate background-subtracted statistics if background provided
            if background_brightness is not None:
                l_bg_sub_mean, l_bg_sub_median, b_bg_sub_mean, b_bg_sub_median = (
                    self._compute_background_subtracted(
                        l_star, blue_chan, background_brightness
                    )
                )
            else:
                l_bg_sub_mean = l_raw_mean
                l_bg_sub_median = l_raw_median
                b_bg_sub_mean = b_raw_mean
                b_bg_sub_median = b_raw_median

            return (
                l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median,
                b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median
            )

        except cv2.error as e:
            logging.error(f"OpenCV error during brightness computation: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        except Exception as e:
            logging.error(f"Error during brightness computation: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def _compute_with_fixed_mask(
        self,
        roi_bgr: np.ndarray,
        l_chan: np.ndarray,
        roi_mask: np.ndarray,
        background_brightness: Optional[float]
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Compute brightness statistics using a fixed pixel mask.

        Args:
            roi_bgr: ROI in BGR format
            l_chan: L channel from LAB conversion
            roi_mask: Boolean mask for pixel selection
            background_brightness: Optional background L* value

        Returns:
            Tuple of 8 brightness statistics
        """
        mask_bool = roi_mask.astype(bool)
        if mask_bool.shape[:2] != roi_bgr.shape[:2] or not np.any(mask_bool):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        l_star = l_chan * 100.0 / 255.0
        blue_chan = roi_bgr[:, :, 0].astype(np.float32)

        l_pixels = l_star[mask_bool]
        b_pixels = blue_chan[mask_bool]

        l_raw_mean = float(np.mean(l_pixels))
        l_raw_median = float(np.median(l_pixels))
        b_raw_mean = float(np.mean(b_pixels))
        b_raw_median = float(np.median(b_pixels))

        if background_brightness is not None:
            l_bg = l_pixels - background_brightness
            l_bg_sub_mean = float(np.mean(l_bg))
            l_bg_sub_median = float(np.median(l_bg))
            b_bg_sub_mean = b_raw_mean
            b_bg_sub_median = b_raw_median
        else:
            l_bg_sub_mean = l_raw_mean
            l_bg_sub_median = l_raw_median
            b_bg_sub_mean = b_raw_mean
            b_bg_sub_median = b_raw_median

        return (
            l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median,
            b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median
        )

    def _compute_background_subtracted(
        self,
        l_star: np.ndarray,
        blue_chan: np.ndarray,
        background_brightness: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute background-subtracted brightness statistics.

        Args:
            l_star: L* channel (0-100 scale)
            blue_chan: Blue channel (0-255 scale)
            background_brightness: Background L* threshold

        Returns:
            Tuple of (l_bg_sub_mean, l_bg_sub_median, b_bg_sub_mean, b_bg_sub_median)
        """
        # Filter pixels above background threshold
        above_background_mask = l_star > background_brightness

        # Apply morphological operations to clean up the mask (remove noise/stray pixels)
        if np.any(above_background_mask):
            above_background_mask = self._apply_morphological_cleaning(
                above_background_mask
            )

        if not np.any(above_background_mask):
            return 0.0, 0.0, 0.0, 0.0

        # Apply additional noise floor filtering if enabled
        if self.noise_floor_threshold > 0:
            noise_floor_mask = l_star > self.noise_floor_threshold
            combined_mask = above_background_mask & noise_floor_mask
        else:
            combined_mask = above_background_mask

        if not np.any(combined_mask):
            return 0.0, 0.0, 0.0, 0.0

        # Only analyze pixels above both background and noise floor thresholds
        filtered_l_pixels = l_star[combined_mask]
        filtered_b_pixels = blue_chan[combined_mask]

        # Background-subtracted L* statistics
        bg_subtracted_l_pixels = filtered_l_pixels - background_brightness
        l_bg_sub_mean = float(np.mean(bg_subtracted_l_pixels))
        l_bg_sub_median = float(np.median(bg_subtracted_l_pixels))

        # Blue channel statistics for masked pixels (no background subtraction for blue)
        b_bg_sub_mean = float(np.mean(filtered_b_pixels))
        b_bg_sub_median = float(np.median(filtered_b_pixels))

        return l_bg_sub_mean, l_bg_sub_median, b_bg_sub_mean, b_bg_sub_median

    def _apply_morphological_cleaning(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening to remove noise from binary mask.

        Args:
            mask: Boolean mask to clean

        Returns:
            Cleaned boolean mask
        """
        # Create structuring element for morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morphological_kernel_size, self.morphological_kernel_size)
        )

        # Convert boolean mask to uint8 for morphological operations
        mask_uint8 = mask.astype(np.uint8) * 255

        # Apply opening (erosion followed by dilation) to remove small noise
        cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        # Convert back to boolean mask
        return cleaned_mask > 0

    def compute_background_brightness(
        self,
        frame: np.ndarray,
        roi_coords: Tuple[Tuple[int, int], Tuple[int, int]],
        percentile: float = 90.0
    ) -> Optional[float]:
        """
        Calculate background ROI brightness for current frame.

        Args:
            frame: Current video frame in BGR format
            roi_coords: ROI coordinates as ((x1, y1), (x2, y2))
            percentile: Percentile for background calculation (default 90th)

        Returns:
            Percentile L* brightness of background ROI, or None if calculation fails
        """
        if frame is None:
            return None

        try:
            pt1, pt2 = roi_coords
            roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            if roi.size == 0:
                return None

            # Convert to LAB and get L* channel
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0].astype(np.float32)
            l_star = l_chan * 100.0 / 255.0

            return float(np.percentile(l_star, percentile))

        except Exception as e:
            logging.error(f"Error computing background brightness: {e}")
            return None

    def set_morphological_kernel_size(self, size: int) -> None:
        """
        Update morphological kernel size for noise filtering.

        Args:
            size: Kernel size (must be positive odd number)
        """
        self.morphological_kernel_size = max(1, size)

    def set_noise_floor_threshold(self, threshold: float) -> None:
        """
        Update noise floor threshold.

        Args:
            threshold: Absolute L* threshold (0-100 range)
        """
        self.noise_floor_threshold = max(0.0, min(100.0, threshold))