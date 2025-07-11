"""Custom exception hierarchy for Brightness Sorcerer."""

from typing import Optional, Any


class BrightnessSorcererError(Exception):
    """Base exception for all Brightness Sorcerer errors."""
    
    def __init__(self, message: str, details: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.details = details
        self.cause = cause
    
    def __str__(self) -> str:
        result = self.message
        if self.details:
            result += f"\nDetails: {self.details}"
        if self.cause:
            result += f"\nCaused by: {self.cause}"
        return result


# Video processing errors
class VideoError(BrightnessSorcererError):
    """Base class for video-related errors."""
    pass


class VideoLoadError(VideoError):
    """Error loading video file."""
    
    def __init__(self, file_path: str, reason: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to load video: {file_path}",
            f"Reason: {reason}",
            cause
        )
        self.file_path = file_path
        self.reason = reason


class VideoSeekError(VideoError):
    """Error seeking to specific frame."""
    
    def __init__(self, frame_index: int, total_frames: int, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to seek to frame {frame_index}",
            f"Valid range: 0 to {total_frames - 1}",
            cause
        )
        self.frame_index = frame_index
        self.total_frames = total_frames


class VideoNotLoadedError(VideoError):
    """Video operation attempted when no video is loaded."""
    
    def __init__(self, operation: str):
        super().__init__(
            f"Cannot perform '{operation}': no video loaded",
            "Load a video file before attempting this operation"
        )
        self.operation = operation


# ROI-related errors
class ROIError(BrightnessSorcererError):
    """Base class for ROI-related errors."""
    pass


class InvalidROIError(ROIError):
    """ROI has invalid dimensions or position."""
    
    def __init__(self, roi_data: dict, reason: str):
        super().__init__(
            f"Invalid ROI: {reason}",
            f"ROI data: {roi_data}"
        )
        self.roi_data = roi_data
        self.reason = reason


class ROIOutOfBoundsError(ROIError):
    """ROI extends outside frame boundaries."""
    
    def __init__(self, roi_bounds: tuple, frame_size: tuple):
        super().__init__(
            f"ROI extends outside frame boundaries",
            f"ROI bounds: {roi_bounds}, Frame size: {frame_size}"
        )
        self.roi_bounds = roi_bounds
        self.frame_size = frame_size


class ROINotFoundError(ROIError):
    """ROI with specified index or ID not found."""
    
    def __init__(self, roi_identifier: Any):
        super().__init__(
            f"ROI not found: {roi_identifier}",
            "Check that the ROI index or ID is valid"
        )
        self.roi_identifier = roi_identifier


# Analysis errors
class AnalysisError(BrightnessSorcererError):
    """Base class for analysis-related errors."""
    pass


class AnalysisConfigError(AnalysisError):
    """Invalid analysis configuration."""
    
    def __init__(self, config_errors: list[str]):
        super().__init__(
            "Invalid analysis configuration",
            f"Errors: {'; '.join(config_errors)}"
        )
        self.config_errors = config_errors


class AnalysisExecutionError(AnalysisError):
    """Error during analysis execution."""
    
    def __init__(self, stage: str, frame_index: Optional[int] = None, cause: Optional[Exception] = None):
        message = f"Analysis failed at stage: {stage}"
        details = f"Frame index: {frame_index}" if frame_index is not None else None
        
        super().__init__(message, details, cause)
        self.stage = stage
        self.frame_index = frame_index


class ThresholdError(AnalysisError):
    """Error in threshold calculation or configuration."""
    
    def __init__(self, threshold_type: str, value: Any, reason: str):
        super().__init__(
            f"Invalid {threshold_type} threshold: {value}",
            f"Reason: {reason}"
        )
        self.threshold_type = threshold_type
        self.value = value
        self.reason = reason


# Settings and configuration errors
class SettingsError(BrightnessSorcererError):
    """Base class for settings-related errors."""
    pass


class SettingsLoadError(SettingsError):
    """Error loading settings from file."""
    
    def __init__(self, file_path: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to load settings from: {file_path}",
            "Settings will be reset to defaults",
            cause
        )
        self.file_path = file_path


class SettingsSaveError(SettingsError):
    """Error saving settings to file."""
    
    def __init__(self, file_path: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Failed to save settings to: {file_path}",
            "Settings changes may be lost",
            cause
        )
        self.file_path = file_path


class InvalidSettingError(SettingsError):
    """Invalid setting name or value."""
    
    def __init__(self, setting_name: str, value: Any, valid_values: Optional[list] = None):
        details = f"Value: {value}"
        if valid_values:
            details += f", Valid values: {valid_values}"
        
        super().__init__(
            f"Invalid setting: {setting_name}",
            details
        )
        self.setting_name = setting_name
        self.value = value
        self.valid_values = valid_values


# File and I/O errors
class FileOperationError(BrightnessSorcererError):
    """Base class for file operation errors."""
    pass


class FileNotFoundError(FileOperationError):
    """File not found or not accessible."""
    
    def __init__(self, file_path: str, operation: str):
        super().__init__(
            f"File not found: {file_path}",
            f"Operation: {operation}"
        )
        self.file_path = file_path
        self.operation = operation


class FilePermissionError(FileOperationError):
    """Insufficient permissions for file operation."""
    
    def __init__(self, file_path: str, operation: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Permission denied: {file_path}",
            f"Operation: {operation}",
            cause
        )
        self.file_path = file_path
        self.operation = operation


class InvalidFileFormatError(FileOperationError):
    """File format is not supported."""
    
    def __init__(self, file_path: str, expected_formats: list[str]):
        super().__init__(
            f"Unsupported file format: {file_path}",
            f"Supported formats: {', '.join(expected_formats)}"
        )
        self.file_path = file_path
        self.expected_formats = expected_formats


# UI and interaction errors
class UIError(BrightnessSorcererError):
    """Base class for UI-related errors."""
    pass


class InvalidInteractionError(UIError):
    """Invalid user interaction or operation."""
    
    def __init__(self, interaction: str, current_state: str):
        super().__init__(
            f"Invalid interaction: {interaction}",
            f"Current state: {current_state}"
        )
        self.interaction = interaction
        self.current_state = current_state


class ComponentNotInitializedError(UIError):
    """UI component not properly initialized."""
    
    def __init__(self, component_name: str):
        super().__init__(
            f"Component not initialized: {component_name}",
            "Initialize the component before use"
        )
        self.component_name = component_name