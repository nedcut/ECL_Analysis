"""Analysis result data model."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime


@dataclass
class FrameAnalysis:
    """Analysis data for a single frame."""
    frame_index: int
    timestamp: float  # Time in seconds
    roi_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_roi_brightness(self, roi_label: str, stat_type: str = 'mean') -> Optional[float]:
        """Get brightness value for specific ROI and statistic type."""
        roi_data = self.roi_stats.get(roi_label)
        if roi_data:
            return roi_data.get(stat_type)
        return None


@dataclass 
class AnalysisResult:
    """Complete analysis result containing all frame data and metadata."""
    
    # Metadata
    video_path: str
    start_frame: int
    end_frame: int
    total_frames: int
    fps: float
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    # Analysis parameters
    roi_labels: List[str] = field(default_factory=list)
    brightness_threshold: Optional[float] = None
    background_roi_label: Optional[str] = None
    
    # Frame-by-frame data
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    
    # Summary statistics
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_frame_analysis(self, frame_analysis: FrameAnalysis):
        """Add analysis data for a single frame."""
        self.frame_analyses.append(frame_analysis)
    
    def get_roi_timeseries(self, roi_label: str, stat_type: str = 'mean') -> List[float]:
        """Get time series data for specific ROI and statistic."""
        timeseries = []
        for frame_data in self.frame_analyses:
            value = frame_data.get_roi_brightness(roi_label, stat_type)
            timeseries.append(value if value is not None else 0.0)
        return timeseries
    
    def get_timestamps(self) -> List[float]:
        """Get list of frame timestamps."""
        return [frame.timestamp for frame in self.frame_analyses]
    
    def get_frame_indices(self) -> List[int]:
        """Get list of frame indices."""
        return [frame.frame_index for frame in self.frame_analyses]
    
    def calculate_summary_statistics(self):
        """Calculate summary statistics for all ROIs."""
        self.summary_stats = {}
        
        for roi_label in self.roi_labels:
            roi_summary = {}
            
            # Get time series for mean and median
            mean_values = self.get_roi_timeseries(roi_label, 'mean')
            median_values = self.get_roi_timeseries(roi_label, 'median')
            
            if mean_values:
                roi_summary['mean_brightness'] = {
                    'overall_mean': np.mean(mean_values),
                    'overall_std': np.std(mean_values),
                    'min': np.min(mean_values),
                    'max': np.max(mean_values),
                    'range': np.max(mean_values) - np.min(mean_values)
                }
            
            if median_values:
                roi_summary['median_brightness'] = {
                    'overall_median': np.median(median_values),
                    'overall_std': np.std(median_values),
                    'min': np.min(median_values),
                    'max': np.max(median_values),
                    'range': np.max(median_values) - np.min(median_values)
                }
            
            # Calculate difference between mean and median
            if mean_values and median_values:
                differences = [m - med for m, med in zip(mean_values, median_values)]
                roi_summary['mean_median_difference'] = {
                    'overall_mean': np.mean(differences),
                    'overall_std': np.std(differences),
                    'min': np.min(differences),
                    'max': np.max(differences)
                }
            
            self.summary_stats[roi_label] = roi_summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert analysis results to pandas DataFrame."""
        data_rows = []
        
        for frame_data in self.frame_analyses:
            row = {
                'frame_index': frame_data.frame_index,
                'timestamp': frame_data.timestamp
            }
            
            # Add ROI data
            for roi_label in self.roi_labels:
                roi_stats = frame_data.roi_stats.get(roi_label, {})
                for stat_name, stat_value in roi_stats.items():
                    col_name = f"{roi_label}_{stat_name}"
                    row[col_name] = stat_value
            
            data_rows.append(row)
        
        return pd.DataFrame(data_rows)
    
    def save_csv(self, file_path: str) -> bool:
        """Save analysis results to CSV file."""
        try:
            df = self.to_dataframe()
            df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary for serialization."""
        return {
            'metadata': {
                'video_path': self.video_path,
                'start_frame': self.start_frame,
                'end_frame': self.end_frame,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'analysis_timestamp': self.analysis_timestamp.isoformat(),
                'roi_labels': self.roi_labels,
                'brightness_threshold': self.brightness_threshold,
                'background_roi_label': self.background_roi_label
            },
            'frame_data': [
                {
                    'frame_index': frame.frame_index,
                    'timestamp': frame.timestamp,
                    'roi_stats': frame.roi_stats
                }
                for frame in self.frame_analyses
            ],
            'summary_stats': self.summary_stats
        }
    
    def save_json(self, file_path: str) -> bool:
        """Save analysis results to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create AnalysisResult from dictionary."""
        metadata = data['metadata']
        
        result = cls(
            video_path=metadata['video_path'],
            start_frame=metadata['start_frame'],
            end_frame=metadata['end_frame'],
            total_frames=metadata['total_frames'],
            fps=metadata['fps'],
            analysis_timestamp=datetime.fromisoformat(metadata['analysis_timestamp']),
            roi_labels=metadata.get('roi_labels', []),
            brightness_threshold=metadata.get('brightness_threshold'),
            background_roi_label=metadata.get('background_roi_label'),
            summary_stats=data.get('summary_stats', {})
        )
        
        # Add frame data
        for frame_data in data.get('frame_data', []):
            frame_analysis = FrameAnalysis(
                frame_index=frame_data['frame_index'],
                timestamp=frame_data['timestamp'],
                roi_stats=frame_data['roi_stats']
            )
            result.add_frame_analysis(frame_analysis)
        
        return result
    
    @classmethod
    def load_json(cls, file_path: str) -> Optional['AnalysisResult']:
        """Load analysis results from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return None
    
    def get_analysis_duration(self) -> float:
        """Get duration of analyzed video segment in seconds."""
        if self.fps > 0:
            return (self.end_frame - self.start_frame + 1) / self.fps
        return 0.0
    
    def get_peak_frames(self, roi_label: str, stat_type: str = 'mean', 
                       threshold: Optional[float] = None) -> List[int]:
        """Get frame indices where ROI brightness exceeds threshold."""
        if threshold is None:
            threshold = self.brightness_threshold
        
        if threshold is None:
            return []
        
        peak_frames = []
        timeseries = self.get_roi_timeseries(roi_label, stat_type)
        
        for i, value in enumerate(timeseries):
            if value >= threshold:
                peak_frames.append(self.frame_analyses[i].frame_index)
        
        return peak_frames
    
    def __repr__(self) -> str:
        return (f"AnalysisResult(video='{self.video_path}', "
                f"frames={self.start_frame}-{self.end_frame}, "
                f"rois={len(self.roi_labels)}, "
                f"analyzed_frames={len(self.frame_analyses)})")