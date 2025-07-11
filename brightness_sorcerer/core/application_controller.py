"""Application controller for coordinating business logic and UI interactions."""

from typing import Optional, Callable, Dict, Any
import logging
from pathlib import Path

from ..models.analysis_session import AnalysisSession, AnalysisConfig
from ..models.video_data import VideoData
from ..models.roi import ROI
from ..core.video_processor import VideoProcessor
from ..core.roi_manager import ROIManager
from ..core.brightness_analyzer import BrightnessAnalyzer
from ..core.settings_manager import SettingsManager
from ..core.threshold_manager import ThresholdManager, ThresholdConfig
from ..core.result import Result, success, error, safe_call
from ..core.exceptions import (
    BrightnessSorcererError, VideoLoadError, AnalysisError, 
    AnalysisConfigError, FileOperationError
)


class ApplicationController:
    """
    Coordinates business logic between UI components and core functionality.
    
    This controller handles:
    - Video loading and management
    - ROI operations and management
    - Analysis session management
    - Settings persistence
    - Analysis execution
    """
    
    def __init__(self):
        # Core components
        self.video_processor = VideoProcessor()
        self.roi_manager = ROIManager()
        self.brightness_analyzer = BrightnessAnalyzer()
        self.settings_manager = SettingsManager()
        
        # Threshold management
        self.threshold_manager = ThresholdManager()
        
        # Current session
        self.current_session: Optional[AnalysisSession] = None
        
        # UI callbacks
        self.ui_callbacks: Dict[str, Callable] = {}
        
        # Analysis state
        self._analysis_running = False
        
        logging.info("ApplicationController initialized")
    
    def register_ui_callback(self, event_name: str, callback: Callable):
        """Register a UI callback for specific events."""
        self.ui_callbacks[event_name] = callback
    
    def _notify_ui(self, event_name: str, *args, **kwargs):
        """Notify UI of events."""
        if event_name in self.ui_callbacks:
            try:
                self.ui_callbacks[event_name](*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in UI callback {event_name}: {e}")
    
    # Video Management
    
    def load_video(self, video_path: str) -> Result[VideoData, BrightnessSorcererError]:
        """Load a video file and create a new analysis session."""
        def _load_video():
            if not self.video_processor.load_video(video_path):
                raise VideoLoadError(video_path, "Failed to load video file")
            
            # Create video data model
            info = self.video_processor.get_video_info()
            video_data = VideoData(
                filename=Path(video_path).name,
                file_path=video_path,
                width=info['width'],
                height=info['height'],
                total_frames=info['total_frames'],
                fps=info['fps'],
                duration=info['duration']
            )
            
            # Create new session
            config = AnalysisConfig()
            config.end_frame = video_data.total_frames - 1
            
            self.current_session = AnalysisSession(
                video_data=video_data,
                config=config
            )
            
            # Reset ROI manager
            self.roi_manager.clear_rois()
            
            logging.info(f"Video loaded: {video_path}")
            return video_data
        
        result = safe_call(_load_video)
        
        if result.is_success():
            self._notify_ui('video_loaded', result.unwrap())
        else:
            self._notify_ui('video_load_failed', str(result.error))
        
        return result
    
    def get_current_frame(self) -> Optional[Any]:
        """Get the current video frame."""
        return self.video_processor.get_current_frame()
    
    def seek_to_frame(self, frame_index: int) -> bool:
        """Seek to a specific frame."""
        success = self.video_processor.seek_to_frame(frame_index)
        if success:
            self._notify_ui('frame_changed', frame_index)
        return success
    
    def step_frames(self, step: int) -> bool:
        """Step forward or backward by frames."""
        success = self.video_processor.step_frames(step)
        if success:
            self._notify_ui('frame_changed', self.video_processor.current_frame_index)
        return success
    
    # ROI Management
    
    def add_roi(self, x: int, y: int, width: int, height: int, label: str = "") -> Result[int, BrightnessSorcererError]:
        """Add a new ROI."""
        def _add_roi():
            roi_index = self.roi_manager.add_roi(x, y, width, height, label)
            
            if self.current_session and roi_index >= 0:
                roi = self.roi_manager.get_roi(roi_index)
                if roi:
                    session_index = self.current_session.add_roi(roi)
                    logging.info(f"ROI added: {roi.label} at index {roi_index}")
            
            return roi_index
        
        result = safe_call(_add_roi)
        
        if result.is_success():
            self._notify_ui('roi_added', result.unwrap())
        
        return result
    
    def delete_roi(self, roi_index: int) -> Result[bool, BrightnessSorcererError]:
        """Delete an ROI."""
        def _delete_roi():
            success = self.roi_manager.delete_roi(roi_index)
            
            if success and self.current_session:
                self.current_session.remove_roi(roi_index)
                logging.info(f"ROI deleted at index {roi_index}")
            
            return success
        
        result = safe_call(_delete_roi)
        
        if result.is_success() and result.unwrap():
            self._notify_ui('roi_deleted', roi_index)
        
        return result
    
    def select_roi(self, roi_index: Optional[int]) -> None:
        """Select an ROI."""
        self.roi_manager.select_roi(roi_index)
        
        if self.current_session:
            self.current_session.selected_roi_index = roi_index
        
        self._notify_ui('roi_selected', roi_index)
    
    def set_background_roi(self, roi_index: Optional[int]) -> Result[bool, BrightnessSorcererError]:
        """Set the background ROI."""
        def _set_background():
            success = self.roi_manager.set_background_roi(roi_index)
            
            if success and self.current_session:
                self.current_session.set_background_roi(roi_index)
                logging.info(f"Background ROI set to index: {roi_index}")
            
            return success
        
        result = safe_call(_set_background)
        
        if result.is_success():
            self._notify_ui('background_roi_changed', roi_index)
        
        return result
    
    def rename_roi(self, roi_index: int, new_label: str) -> Result[bool, BrightnessSorcererError]:
        """Rename an ROI."""
        def _rename_roi():
            roi = self.roi_manager.get_roi(roi_index)
            if roi is None:
                raise BrightnessSorcererError(f"ROI not found at index {roi_index}")
            
            roi.label = new_label
            
            if self.current_session and roi_index < len(self.current_session.rois):
                self.current_session.rois[roi_index].label = new_label
                self.current_session.touch()
            
            logging.info(f"ROI {roi_index} renamed to: {new_label}")
            return True
        
        result = safe_call(_rename_roi)
        
        if result.is_success():
            self._notify_ui('roi_renamed', roi_index, new_label)
        
        return result
    
    # Mouse Interaction Handling
    
    def handle_mouse_press(self, x: int, y: int, button: int) -> None:
        """Handle mouse press events on video display."""
        if not self.video_processor.is_loaded():
            return
        
        frame_width, frame_height = self.video_processor.frame_size
        
        if button == 1:  # Left button (Qt.LeftButton value)
            # Check for ROI interaction
            roi_index = self.roi_manager.get_roi_at_point(x, y)
            
            if roi_index is not None:
                # Select ROI
                self.select_roi(roi_index)
                
                # Check for resize handle
                resize_corner = self.roi_manager.get_resize_corner_at_point(roi_index, x, y)
                if resize_corner is not None:
                    # Start resizing
                    self.roi_manager.start_resizing(roi_index, resize_corner)
                else:
                    # Start moving
                    self.roi_manager.start_moving(roi_index, x, y)
            else:
                # Start drawing new ROI
                self.roi_manager.start_drawing(x, y)
            
            self._notify_ui('display_update_needed')
    
    def handle_mouse_move(self, x: int, y: int) -> None:
        """Handle mouse move events on video display."""
        if not self.video_processor.is_loaded():
            return
        
        frame_width, frame_height = self.video_processor.frame_size
        
        if self.roi_manager.drawing_mode:
            self.roi_manager.update_drawing(x, y)
            self._notify_ui('display_update_needed')
        elif self.roi_manager.moving_mode:
            self.roi_manager.update_moving(x, y, frame_width, frame_height)
            self._notify_ui('display_update_needed')
        elif self.roi_manager.resizing_mode:
            self.roi_manager.update_resizing(x, y, frame_width, frame_height)
            self._notify_ui('display_update_needed')
    
    def handle_mouse_release(self, x: int, y: int, button: int) -> None:
        """Handle mouse release events on video display."""
        if not self.video_processor.is_loaded():
            return
        
        frame_width, frame_height = self.video_processor.frame_size
        
        if button == 1:  # Left button
            if self.roi_manager.drawing_mode:
                # Finish drawing
                roi_index = self.roi_manager.finish_drawing(x, y, frame_width, frame_height)
                if roi_index >= 0:
                    # Add ROI to session
                    if self.current_session:
                        roi = self.roi_manager.get_roi(roi_index)
                        if roi:
                            self.current_session.add_roi(roi)
                    self._notify_ui('roi_created', roi_index)
            elif self.roi_manager.moving_mode:
                # Finish moving
                self.roi_manager.finish_moving()
            elif self.roi_manager.resizing_mode:
                # Finish resizing
                self.roi_manager.finish_resizing()
            
            self._notify_ui('display_update_needed')
    
    # Analysis Management
    
    def validate_analysis_config(self, start_frame: int, end_frame: int, 
                                output_dir: str) -> Result[AnalysisConfig, AnalysisConfigError]:
        """Validate analysis configuration."""
        def _validate():
            if not self.current_session:
                raise AnalysisConfigError(["No video session loaded"])
            
            errors = []
            
            if start_frame < 0:
                errors.append("Start frame must be non-negative")
            
            if end_frame < start_frame:
                errors.append("End frame must be greater than or equal to start frame")
            
            if not output_dir or not Path(output_dir).exists():
                errors.append("Output directory must be valid and exist")
            
            if not self.current_session.rois:
                errors.append("At least one ROI must be defined")
            
            if errors:
                raise AnalysisConfigError(errors)
            
            # Update session config
            config = self.current_session.config
            config.start_frame = start_frame
            config.end_frame = end_frame
            config.output_directory = output_dir
            
            # Validate the complete config
            config_errors = config.validate()
            if config_errors:
                raise AnalysisConfigError(config_errors)
            
            return config
        
        return safe_call(_validate)
    
    def start_analysis(self, start_frame: int, end_frame: int, output_dir: str) -> Result[bool, AnalysisError]:
        """Start brightness analysis."""
        if self._analysis_running:
            return error(AnalysisError("Analysis already running"))
        
        # Validate configuration
        config_result = self.validate_analysis_config(start_frame, end_frame, output_dir)
        if config_result.is_error():
            return error(AnalysisError("Invalid configuration", str(config_result.error)))
        
        def _start_analysis():
            self._analysis_running = True
            
            try:
                # Set up progress callback
                def progress_callback(current: int, total: int, message: str = ""):
                    self._notify_ui('analysis_progress', current, total, message)
                
                # Run analysis
                result = self.brightness_analyzer.analyze_video(
                    self.video_processor,
                    self.current_session.rois,
                    start_frame,
                    end_frame,
                    output_dir,
                    progress_callback=progress_callback,
                    threshold_manager=self.threshold_manager,
                    background_roi=self.current_session.get_background_roi()
                )
                
                if result:
                    self.current_session.current_result = result
                    logging.info(f"Analysis completed successfully: {output_dir}")
                    return True
                else:
                    raise AnalysisError("Analysis failed - no result returned")
                
            finally:
                self._analysis_running = False
        
        analysis_result = safe_call(_start_analysis)
        
        if analysis_result.is_success():
            self._notify_ui('analysis_completed', analysis_result.unwrap())
        else:
            self._notify_ui('analysis_failed', str(analysis_result.error))
        
        return analysis_result
    
    def auto_detect_frame_range(self) -> Result[tuple[int, int], AnalysisError]:
        """Auto-detect optimal frame range for analysis."""
        def _auto_detect():
            if not self.current_session or not self.current_session.video_data:
                raise AnalysisError("No video loaded")
            
            if not self.current_session.rois:
                raise AnalysisError("No ROIs defined for auto-detection")
            
            # Use brightness analyzer to detect range
            start, end = self.brightness_analyzer.auto_detect_range(
                self.video_processor,
                self.current_session.rois[0]  # Use first ROI for detection
            )
            
            logging.info(f"Auto-detected frame range: {start} - {end}")
            return (start, end)
        
        result = safe_call(_auto_detect)
        
        if result.is_success():
            start, end = result.unwrap()
            self._notify_ui('frame_range_detected', start, end)
        
        return result
    
    # Settings Management
    
    def update_threshold_settings(self, **kwargs) -> Result[bool, BrightnessSorcererError]:
        """Update threshold settings."""
        def _update_threshold():
            self.threshold_manager.update_config(**kwargs)
            
            # Update session config if available
            if self.current_session:
                for key, value in kwargs.items():
                    if hasattr(self.current_session.config.threshold_config, key):
                        setattr(self.current_session.config.threshold_config, key, value)
                self.current_session.touch()
            
            # Save to settings
            self.settings_manager.save_threshold_config(self.threshold_manager.get_config_dict())
            
            logging.info(f"Threshold settings updated: {kwargs}")
            return True
        
        result = safe_call(_update_threshold)
        
        if result.is_success():
            self._notify_ui('threshold_settings_updated', kwargs)
        
        return result
    
    def get_analysis_defaults(self) -> Dict[str, Any]:
        """Get default analysis settings."""
        return self.settings_manager.get_analysis_defaults()
    
    # Session Management
    
    def save_session(self, file_path: str) -> Result[bool, FileOperationError]:
        """Save the current analysis session."""
        def _save_session():
            if not self.current_session:
                raise FileOperationError(file_path, "save", "No session to save")
            
            success = self.current_session.save_to_file(file_path)
            if not success:
                raise FileOperationError(file_path, "save", "Failed to save session")
            
            return True
        
        result = safe_call(_save_session)
        
        if result.is_success():
            self._notify_ui('session_saved', file_path)
        
        return result
    
    def load_session(self, file_path: str) -> Result[AnalysisSession, FileOperationError]:
        """Load an analysis session."""
        def _load_session():
            session = AnalysisSession.load_from_file(file_path)
            if session is None:
                raise FileOperationError(file_path, "load", "Failed to load session")
            
            self.current_session = session
            
            # Load video if available
            if session.video_data and session.video_data.file_path:
                video_result = self.load_video(session.video_data.file_path)
                if video_result.is_error():
                    logging.warning(f"Could not reload video: {session.video_data.file_path}")
            
            # Restore ROIs
            self.roi_manager.clear_rois()
            for roi in session.rois:
                self.roi_manager.add_roi_object(roi)
            
            # Set background ROI
            if session.background_roi_index is not None:
                self.roi_manager.set_background_roi(session.background_roi_index)
            
            logging.info(f"Session loaded: {file_path}")
            return session
        
        result = safe_call(_load_session)
        
        if result.is_success():
            self._notify_ui('session_loaded', result.unwrap())
        
        return result
    
    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current session."""
        if self.current_session:
            return self.current_session.get_analysis_summary()
        return None
    
    # Cleanup
    
    def cleanup(self):
        """Clean up resources."""
        self.video_processor.cleanup()
        self.roi_manager.clear_rois()
        self.current_session = None
        logging.info("ApplicationController cleaned up")