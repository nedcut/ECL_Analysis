"""Unified threshold management system for brightness analysis."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import logging

from ..models.roi import ROI
from ..utils.color_utils import calculate_baseline_brightness


@dataclass
class ThresholdConfig:
    """Configuration for threshold calculations."""
    
    # Manual threshold value (L* units above baseline)
    manual_threshold: float = 5.0
    
    # Noise floor for filtering out dark pixels
    noise_floor: float = 10.0
    
    # Percentile used for baseline calculation
    baseline_percentile: float = 5.0
    
    # Minimum consecutive frames for detection
    min_consecutive_frames: int = 3
    
    # Whether to use background ROI for threshold calculation
    use_background_roi: bool = True
    
    def validate(self) -> list[str]:
        """Validate threshold configuration."""
        errors = []
        
        if self.manual_threshold < 0:
            errors.append("Manual threshold must be non-negative")
        
        if self.noise_floor < 0:
            errors.append("Noise floor must be non-negative")
        
        if not 0 <= self.baseline_percentile <= 100:
            errors.append("Baseline percentile must be between 0 and 100")
        
        if self.min_consecutive_frames < 1:
            errors.append("Minimum consecutive frames must be at least 1")
        
        return errors


class ThresholdManager:
    """Manages threshold calculations for brightness analysis."""
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid threshold configuration: {', '.join(errors)}")
        
        logging.info(f"ThresholdManager initialized with config: {self.config}")
    
    def calculate_effective_threshold(self, 
                                    background_roi: Optional[ROI] = None,
                                    frame: Optional[np.ndarray] = None) -> float:
        """
        Calculate the effective threshold for brightness analysis.
        
        Args:
            background_roi: Optional background ROI for dynamic threshold calculation
            frame: Optional frame for background ROI analysis
            
        Returns:
            Effective threshold value in L* units
        """
        if not self.config.use_background_roi or background_roi is None or frame is None:
            # Use manual threshold
            return self.config.manual_threshold
        
        try:
            # Extract background region
            bg_region = background_roi.extract_region(frame)
            if bg_region is None:
                logging.warning("Could not extract background region, using manual threshold")
                return self.config.manual_threshold
            
            # Calculate baseline brightness from background
            baseline = calculate_baseline_brightness(
                bg_region, 
                self.config.baseline_percentile, 
                self.config.noise_floor
            )
            
            # Effective threshold is baseline + manual threshold
            effective_threshold = baseline + self.config.manual_threshold
            
            logging.debug(f"Background baseline: {baseline:.2f}, "
                         f"manual offset: {self.config.manual_threshold}, "
                         f"effective threshold: {effective_threshold:.2f}")
            
            return effective_threshold
            
        except Exception as e:
            logging.error(f"Error calculating background-based threshold: {e}")
            return self.config.manual_threshold
    
    def is_above_threshold(self, brightness_value: float, 
                          background_roi: Optional[ROI] = None,
                          frame: Optional[np.ndarray] = None) -> bool:
        """
        Check if a brightness value is above the effective threshold.
        
        Args:
            brightness_value: Brightness value to check
            background_roi: Optional background ROI for dynamic threshold
            frame: Optional frame for background ROI analysis
            
        Returns:
            True if brightness is above threshold
        """
        threshold = self.calculate_effective_threshold(background_roi, frame)
        return brightness_value >= threshold
    
    def filter_above_threshold(self, brightness_values: list[float],
                              background_roi: Optional[ROI] = None,
                              frame: Optional[np.ndarray] = None) -> list[bool]:
        """
        Filter brightness values based on threshold.
        
        Args:
            brightness_values: List of brightness values
            background_roi: Optional background ROI for dynamic threshold
            frame: Optional frame for background ROI analysis
            
        Returns:
            List of boolean values indicating which values are above threshold
        """
        threshold = self.calculate_effective_threshold(background_roi, frame)
        return [value >= threshold for value in brightness_values]
    
    def update_config(self, **kwargs) -> None:
        """Update threshold configuration."""
        # Create new config with updated values
        config_dict = {
            'manual_threshold': kwargs.get('manual_threshold', self.config.manual_threshold),
            'noise_floor': kwargs.get('noise_floor', self.config.noise_floor),
            'baseline_percentile': kwargs.get('baseline_percentile', self.config.baseline_percentile),
            'min_consecutive_frames': kwargs.get('min_consecutive_frames', self.config.min_consecutive_frames),
            'use_background_roi': kwargs.get('use_background_roi', self.config.use_background_roi)
        }
        
        new_config = ThresholdConfig(**config_dict)
        
        # Validate new configuration
        errors = new_config.validate()
        if errors:
            raise ValueError(f"Invalid threshold configuration: {', '.join(errors)}")
        
        self.config = new_config
        logging.info(f"ThresholdManager configuration updated: {self.config}")
    
    def get_config_dict(self) -> dict:
        """Get threshold configuration as dictionary."""
        return {
            'manual_threshold': self.config.manual_threshold,
            'noise_floor': self.config.noise_floor,
            'baseline_percentile': self.config.baseline_percentile,
            'min_consecutive_frames': self.config.min_consecutive_frames,
            'use_background_roi': self.config.use_background_roi
        }
    
    def load_config_dict(self, config_dict: dict) -> None:
        """Load threshold configuration from dictionary."""
        self.update_config(**config_dict)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = ThresholdConfig()
        logging.info("ThresholdManager reset to default configuration")
    
    def __repr__(self) -> str:
        return f"ThresholdManager(config={self.config})"