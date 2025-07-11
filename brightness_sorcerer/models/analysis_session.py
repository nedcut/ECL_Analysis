"""Domain models for analysis sessions and projects."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import logging

from .roi import ROI
from .video_data import VideoData
from .analysis_result import AnalysisResult
from ..core.threshold_manager import ThresholdConfig


@dataclass
class AnalysisConfig:
    """Configuration for brightness analysis."""
    
    # Frame range
    start_frame: int = 0
    end_frame: Optional[int] = None
    
    # Threshold configuration
    threshold_config: ThresholdConfig = field(default_factory=ThresholdConfig)
    
    # Output settings
    output_directory: str = ""
    save_csv: bool = True
    save_json: bool = True
    save_plots: bool = True
    plot_dpi: int = 300
    
    # Analysis options
    calculate_summary_stats: bool = True
    include_progress_tracking: bool = True
    
    def validate(self) -> List[str]:
        """Validate analysis configuration."""
        errors = []
        
        if self.start_frame < 0:
            errors.append("Start frame must be non-negative")
        
        if self.end_frame is not None and self.end_frame < self.start_frame:
            errors.append("End frame must be greater than or equal to start frame")
        
        if not self.output_directory:
            errors.append("Output directory must be specified")
        
        if self.plot_dpi < 72:
            errors.append("Plot DPI must be at least 72")
        
        # Validate threshold config
        threshold_errors = self.threshold_config.validate()
        errors.extend(threshold_errors)
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'threshold_config': {
                'manual_threshold': self.threshold_config.manual_threshold,
                'noise_floor': self.threshold_config.noise_floor,
                'baseline_percentile': self.threshold_config.baseline_percentile,
                'min_consecutive_frames': self.threshold_config.min_consecutive_frames,
                'use_background_roi': self.threshold_config.use_background_roi
            },
            'output_directory': self.output_directory,
            'save_csv': self.save_csv,
            'save_json': self.save_json,
            'save_plots': self.save_plots,
            'plot_dpi': self.plot_dpi,
            'calculate_summary_stats': self.calculate_summary_stats,
            'include_progress_tracking': self.include_progress_tracking
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisConfig':
        """Create from dictionary."""
        threshold_data = data.get('threshold_config', {})
        threshold_config = ThresholdConfig(
            manual_threshold=threshold_data.get('manual_threshold', 5.0),
            noise_floor=threshold_data.get('noise_floor', 10.0),
            baseline_percentile=threshold_data.get('baseline_percentile', 5.0),
            min_consecutive_frames=threshold_data.get('min_consecutive_frames', 3),
            use_background_roi=threshold_data.get('use_background_roi', True)
        )
        
        return cls(
            start_frame=data.get('start_frame', 0),
            end_frame=data.get('end_frame'),
            threshold_config=threshold_config,
            output_directory=data.get('output_directory', ''),
            save_csv=data.get('save_csv', True),
            save_json=data.get('save_json', True),
            save_plots=data.get('save_plots', True),
            plot_dpi=data.get('plot_dpi', 300),
            calculate_summary_stats=data.get('calculate_summary_stats', True),
            include_progress_tracking=data.get('include_progress_tracking', True)
        )


@dataclass
class AnalysisSession:
    """Represents a complete analysis session with video, ROIs, and configuration."""
    
    # Core components
    video_data: Optional[VideoData] = None
    rois: List[ROI] = field(default_factory=list)
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    # Session metadata
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    # Session state
    background_roi_index: Optional[int] = None
    selected_roi_index: Optional[int] = None
    
    # Results
    current_result: Optional[AnalysisResult] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.touch()  # Update modified_at
    
    def touch(self):
        """Update the modification timestamp."""
        self.modified_at = datetime.now()
    
    def add_roi(self, roi: ROI) -> int:
        """Add ROI to session and return its index."""
        self.rois.append(roi)
        self.touch()
        return len(self.rois) - 1
    
    def remove_roi(self, index: int) -> bool:
        """Remove ROI by index."""
        if 0 <= index < len(self.rois):
            self.rois.pop(index)
            
            # Update indices
            if self.background_roi_index == index:
                self.background_roi_index = None
            elif self.background_roi_index is not None and self.background_roi_index > index:
                self.background_roi_index -= 1
            
            if self.selected_roi_index == index:
                self.selected_roi_index = None
            elif self.selected_roi_index is not None and self.selected_roi_index > index:
                self.selected_roi_index -= 1
            
            self.touch()
            return True
        return False
    
    def set_background_roi(self, index: Optional[int]) -> bool:
        """Set background ROI by index."""
        if index is not None and (index < 0 or index >= len(self.rois)):
            return False
        
        # Reset previous background
        if self.background_roi_index is not None:
            self.rois[self.background_roi_index].is_background = False
        
        self.background_roi_index = index
        
        # Set new background
        if index is not None:
            self.rois[index].is_background = True
        
        self.touch()
        return True
    
    def get_background_roi(self) -> Optional[ROI]:
        """Get the background ROI if set."""
        if self.background_roi_index is not None and 0 <= self.background_roi_index < len(self.rois):
            return self.rois[self.background_roi_index]
        return None
    
    def validate(self) -> List[str]:
        """Validate the analysis session."""
        errors = []
        
        if self.video_data is None:
            errors.append("Video data is required")
        
        if not self.rois:
            errors.append("At least one ROI is required")
        
        # Validate config
        config_errors = self.config.validate()
        errors.extend(config_errors)
        
        # Validate frame range against video
        if self.video_data:
            if self.config.end_frame is None:
                self.config.end_frame = self.video_data.total_frames - 1
            
            if self.config.end_frame >= self.video_data.total_frames:
                errors.append(f"End frame ({self.config.end_frame}) exceeds video length ({self.video_data.total_frames})")
        
        # Validate ROI bounds
        if self.video_data:
            for i, roi in enumerate(self.rois):
                if not roi.is_valid_for_frame(self.video_data.width, self.video_data.height):
                    errors.append(f"ROI {i} ({roi.label}) is outside video bounds")
        
        return errors
    
    def is_ready_for_analysis(self) -> bool:
        """Check if session is ready for analysis."""
        errors = self.validate()
        return len(errors) == 0
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary information about the analysis session."""
        return {
            'session_id': self.session_id,
            'video_file': self.video_data.filename if self.video_data else None,
            'roi_count': len(self.rois),
            'background_roi': self.background_roi_index,
            'frame_range': f"{self.config.start_frame} - {self.config.end_frame}" if self.config.end_frame else f"{self.config.start_frame} - end",
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'has_results': self.current_result is not None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'video_data': self.video_data.to_dict() if self.video_data else None,
            'rois': [roi.to_dict() for roi in self.rois],
            'config': self.config.to_dict(),
            'background_roi_index': self.background_roi_index,
            'selected_roi_index': self.selected_roi_index,
            'current_result': self.current_result.to_dict() if self.current_result else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisSession':
        """Create session from dictionary."""
        # Parse video data
        video_data = None
        if data.get('video_data'):
            video_data = VideoData.from_dict(data['video_data'])
        
        # Parse ROIs
        rois = [ROI.from_dict(roi_data) for roi_data in data.get('rois', [])]
        
        # Parse config
        config = AnalysisConfig.from_dict(data.get('config', {}))
        
        # Parse result
        current_result = None
        if data.get('current_result'):
            current_result = AnalysisResult.from_dict(data['current_result'])
        
        session = cls(
            session_id=data.get('session_id', datetime.now().strftime("%Y%m%d_%H%M%S")),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            modified_at=datetime.fromisoformat(data.get('modified_at', datetime.now().isoformat())),
            video_data=video_data,
            rois=rois,
            config=config,
            background_roi_index=data.get('background_roi_index'),
            selected_roi_index=data.get('selected_roi_index'),
            current_result=current_result
        )
        
        return session
    
    def save_to_file(self, file_path: str) -> bool:
        """Save session to JSON file."""
        try:
            session_data = self.to_dict()
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            logging.info(f"Session saved to {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving session to {file_path}: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Optional['AnalysisSession']:
        """Load session from JSON file."""
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            session = cls.from_dict(session_data)
            logging.info(f"Session loaded from {file_path}")
            return session
        except Exception as e:
            logging.error(f"Error loading session from {file_path}: {e}")
            return None


@dataclass
class Project:
    """Represents a project containing multiple analysis sessions."""
    
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    # Sessions
    sessions: List[AnalysisSession] = field(default_factory=list)
    active_session_id: Optional[str] = None
    
    # Project settings
    default_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    def touch(self):
        """Update the modification timestamp."""
        self.modified_at = datetime.now()
    
    def add_session(self, session: AnalysisSession) -> None:
        """Add session to project."""
        self.sessions.append(session)
        self.active_session_id = session.session_id
        self.touch()
    
    def get_active_session(self) -> Optional[AnalysisSession]:
        """Get the currently active session."""
        if self.active_session_id:
            for session in self.sessions:
                if session.session_id == self.active_session_id:
                    return session
        return None
    
    def set_active_session(self, session_id: str) -> bool:
        """Set active session by ID."""
        for session in self.sessions:
            if session.session_id == session_id:
                self.active_session_id = session_id
                self.touch()
                return True
        return False
    
    def remove_session(self, session_id: str) -> bool:
        """Remove session by ID."""
        for i, session in enumerate(self.sessions):
            if session.session_id == session_id:
                self.sessions.pop(i)
                if self.active_session_id == session_id:
                    self.active_session_id = None
                    # Set new active session if available
                    if self.sessions:
                        self.active_session_id = self.sessions[-1].session_id
                self.touch()
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'sessions': [session.to_dict() for session in self.sessions],
            'active_session_id': self.active_session_id,
            'default_config': self.default_config.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create project from dictionary."""
        sessions = [AnalysisSession.from_dict(session_data) for session_data in data.get('sessions', [])]
        default_config = AnalysisConfig.from_dict(data.get('default_config', {}))
        
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            modified_at=datetime.fromisoformat(data.get('modified_at', datetime.now().isoformat())),
            sessions=sessions,
            active_session_id=data.get('active_session_id'),
            default_config=default_config
        )
    
    def save_to_file(self, file_path: str) -> bool:
        """Save project to JSON file."""
        try:
            project_data = self.to_dict()
            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            logging.info(f"Project saved to {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving project to {file_path}: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Optional['Project']:
        """Load project from JSON file."""
        try:
            with open(file_path, 'r') as f:
                project_data = json.load(f)
            project = cls.from_dict(project_data)
            logging.info(f"Project loaded from {file_path}")
            return project
        except Exception as e:
            logging.error(f"Error loading project from {file_path}: {e}")
            return None