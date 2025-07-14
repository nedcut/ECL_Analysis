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
    
    threshold: float = 10.0
    
    def validate(self) -> list[str]:
        """Validate threshold configuration."""
        errors = []
        
        if self.threshold < 0:
            errors.append("Threshold must be non-negative")
        
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
    
    def get_threshold(self) -> float:
        """Get the current threshold value."""
        return self.config.threshold
    
    def set_threshold(self, threshold: float) -> None:
        """Set the threshold value."""
        self.config.threshold = threshold
        logging.info(f"Threshold set to: {threshold}")

    def is_above_threshold(self, brightness_value: float) -> bool:
        """
        Check if a brightness value is above the effective threshold.
        
        Args:
            brightness_value: Brightness value to check
            
        Returns:
            True if brightness is above threshold
        """
        return brightness_value >= self.config.threshold
    
    def filter_above_threshold(self, brightness_values: list[float]) -> list[bool]:
        """
        Filter brightness values based on threshold.
        
        Args:
            brightness_values: List of brightness values
            
        Returns:
            List of boolean values indicating which values are above threshold
        """
        return [value >= self.config.threshold for value in brightness_values]
    
    def update_config(self, **kwargs) -> None:
        """Update threshold configuration."""
        if 'threshold' in kwargs:
            self.set_threshold(kwargs['threshold'])
    
    def get_config_dict(self) -> dict:
        """Get threshold configuration as dictionary."""
        return {'threshold': self.config.threshold}
    
    def load_config_dict(self, config_dict: dict) -> None:
        """Load threshold configuration from dictionary."""
        if 'threshold' in config_dict:
            self.set_threshold(config_dict['threshold'])
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = ThresholdConfig()
        logging.info("ThresholdManager reset to default configuration")
    
    def __repr__(self) -> str:
        return f"ThresholdManager(config={self.config})"