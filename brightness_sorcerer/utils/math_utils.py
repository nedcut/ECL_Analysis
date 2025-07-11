"""Mathematical utilities for brightness analysis."""

import numpy as np
from typing import List, Tuple, Optional
import logging


def calculate_percentile(data: np.ndarray, percentile: float) -> float:
    """Calculate percentile of data array."""
    if data is None or data.size == 0:
        return 0.0
    
    try:
        return float(np.percentile(data, percentile))
    except Exception as e:
        logging.error(f"Error calculating percentile: {e}")
        return 0.0


def detect_brightness_peaks(brightness_values: List[float], 
                           threshold: float,
                           min_consecutive_frames: int = 3) -> List[Tuple[int, int]]:
    """Detect brightness peaks above threshold."""
    if not brightness_values:
        return []
    
    try:
        # Find frames above threshold
        above_threshold = [i for i, val in enumerate(brightness_values) if val >= threshold]
        
        if not above_threshold:
            return []
        
        # Group consecutive frames
        peaks = []
        start_frame = above_threshold[0]
        end_frame = start_frame
        
        for i in range(1, len(above_threshold)):
            if above_threshold[i] == above_threshold[i-1] + 1:
                # Consecutive frame
                end_frame = above_threshold[i]
            else:
                # Gap found, save current peak if it meets minimum length
                if end_frame - start_frame + 1 >= min_consecutive_frames:
                    peaks.append((start_frame, end_frame))
                start_frame = above_threshold[i]
                end_frame = start_frame
        
        # Don't forget the last peak
        if end_frame - start_frame + 1 >= min_consecutive_frames:
            peaks.append((start_frame, end_frame))
        
        return peaks
        
    except Exception as e:
        logging.error(f"Error detecting brightness peaks: {e}")
        return []


def find_analysis_range(brightness_values: List[float], 
                       threshold: float,
                       min_consecutive_frames: int = 3) -> Optional[Tuple[int, int]]:
    """Find the optimal start and end frames for analysis."""
    peaks = detect_brightness_peaks(brightness_values, threshold, min_consecutive_frames)
    
    if not peaks:
        return None
    
    # Return the range from first peak start to last peak end
    start_frame = peaks[0][0]
    end_frame = peaks[-1][1]
    
    return (start_frame, end_frame)


def calculate_moving_average(data: List[float], window_size: int) -> List[float]:
    """Calculate moving average with specified window size."""
    if not data or window_size <= 0:
        return data
    
    try:
        if window_size >= len(data):
            # Window larger than data, return mean for all points
            mean_val = np.mean(data)
            return [mean_val] * len(data)
        
        moving_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            window_data = data[start_idx:end_idx]
            moving_avg.append(np.mean(window_data))
        
        return moving_avg
        
    except Exception as e:
        logging.error(f"Error calculating moving average: {e}")
        return data


def calculate_confidence_intervals(data: List[float], 
                                 confidence_level: float = 0.95) -> Tuple[List[float], List[float]]:
    """Calculate confidence intervals for data."""
    if not data:
        return [], []
    
    try:
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        # Calculate confidence interval multiplier
        alpha = 1 - confidence_level
        z_score = 1.96  # For 95% confidence interval
        
        margin_of_error = z_score * std / np.sqrt(len(data))
        
        lower_bound = [mean - margin_of_error] * len(data)
        upper_bound = [mean + margin_of_error] * len(data)
        
        return lower_bound, upper_bound
        
    except Exception as e:
        logging.error(f"Error calculating confidence intervals: {e}")
        return data, data


def interpolate_missing_values(data: List[Optional[float]]) -> List[float]:
    """Interpolate missing values in data series."""
    if not data:
        return []
    
    try:
        # Convert to numpy array, replacing None with nan
        np_data = np.array([x if x is not None else np.nan for x in data])
        
        # Find valid (non-nan) indices
        valid_indices = ~np.isnan(np_data)
        
        if not np.any(valid_indices):
            # No valid data
            return [0.0] * len(data)
        
        if np.all(valid_indices):
            # No missing data
            return np_data.tolist()
        
        # Interpolate missing values
        indices = np.arange(len(np_data))
        np_data[~valid_indices] = np.interp(
            indices[~valid_indices], 
            indices[valid_indices], 
            np_data[valid_indices]
        )
        
        return np_data.tolist()
        
    except Exception as e:
        logging.error(f"Error interpolating missing values: {e}")
        return [x if x is not None else 0.0 for x in data]


def calculate_derivative(data: List[float]) -> List[float]:
    """Calculate first derivative (rate of change) of data."""
    if len(data) < 2:
        return [0.0] * len(data)
    
    try:
        derivative = []
        
        # First point (forward difference)
        derivative.append(data[1] - data[0])
        
        # Middle points (central difference)
        for i in range(1, len(data) - 1):
            derivative.append((data[i + 1] - data[i - 1]) / 2.0)
        
        # Last point (backward difference)
        derivative.append(data[-1] - data[-2])
        
        return derivative
        
    except Exception as e:
        logging.error(f"Error calculating derivative: {e}")
        return [0.0] * len(data)