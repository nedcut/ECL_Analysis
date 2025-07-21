"""
Unit tests for custom exception classes.

Tests the custom exception hierarchy and error handling functionality.
"""

import pytest
from brightness_sorcerer.core.exceptions import (
    BrightnessSorcererError,
    VideoLoadError, 
    AnalysisError,
    ConfigurationError,
    ValidationError,
    ROIError,
    CacheError,
    AudioError
)


class TestBrightnessSorcererError:
    """Test the base exception class."""
    
    def test_basic_exception(self):
        """Test basic exception creation and message."""
        error = BrightnessSorcererError("Test error message")
        assert str(error) == "Test error message"
        assert error.error_code is None
        assert error.details is None
    
    def test_exception_with_error_code(self):
        """Test exception with error code."""
        error = BrightnessSorcererError("Test error", error_code="E001")
        assert str(error) == "Test error"
        assert error.error_code == "E001"
        assert error.details is None
    
    def test_exception_with_details(self):
        """Test exception with additional details."""
        details = {"file_path": "/test/path", "line": 42}
        error = BrightnessSorcererError("Test error", details=details)
        assert str(error) == "Test error"
        assert error.error_code is None
        assert error.details == details
    
    def test_exception_with_all_params(self):
        """Test exception with all parameters."""
        details = {"context": "testing"}
        error = BrightnessSorcererError(
            "Full test error",
            error_code="E999", 
            details=details
        )
        assert str(error) == "Full test error"
        assert error.error_code == "E999"
        assert error.details == details


class TestVideoLoadError:
    """Test video loading specific errors."""
    
    def test_inheritance(self):
        """Test that VideoLoadError inherits from BrightnessSorcererError."""
        error = VideoLoadError("Video load failed")
        assert isinstance(error, BrightnessSorcererError)
        assert isinstance(error, VideoLoadError)
    
    def test_video_load_error_message(self):
        """Test video load error with descriptive message."""
        file_path = "/path/to/video.mp4"
        error = VideoLoadError(f"Could not load video: {file_path}")
        assert "Could not load video: /path/to/video.mp4" in str(error)


class TestAnalysisError:
    """Test analysis specific errors."""
    
    def test_inheritance(self):
        """Test that AnalysisError inherits from BrightnessSorcererError."""
        error = AnalysisError("Analysis failed")
        assert isinstance(error, BrightnessSorcererError)
        assert isinstance(error, AnalysisError)
    
    def test_analysis_error_with_context(self):
        """Test analysis error with ROI context."""
        error = AnalysisError(
            "ROI analysis failed",
            details={"roi_index": 1, "frame": 150}
        )
        assert str(error) == "ROI analysis failed"
        assert error.details["roi_index"] == 1
        assert error.details["frame"] == 150


class TestConfigurationError:
    """Test configuration specific errors."""
    
    def test_inheritance(self):
        """Test that ConfigurationError inherits from BrightnessSorcererError."""
        error = ConfigurationError("Config invalid")
        assert isinstance(error, BrightnessSorcererError)
        assert isinstance(error, ConfigurationError)
    
    def test_config_error_with_file_info(self):
        """Test configuration error with file information."""
        error = ConfigurationError(
            "Invalid configuration file",
            error_code="CONFIG001",
            details={"file": "config/settings.json", "field": "frame_cache_size"}
        )
        assert "Invalid configuration file" in str(error)
        assert error.error_code == "CONFIG001"
        assert error.details["file"] == "config/settings.json"


class TestValidationError:
    """Test validation specific errors."""
    
    def test_inheritance(self):
        """Test that ValidationError inherits from BrightnessSorcererError."""
        error = ValidationError("Validation failed")
        assert isinstance(error, BrightnessSorcererError)
        assert isinstance(error, ValidationError)
    
    def test_validation_error_types(self):
        """Test different types of validation errors."""
        # File validation
        file_error = ValidationError("File does not exist: test.mp4")
        assert "File does not exist" in str(file_error)
        
        # ROI validation  
        roi_error = ValidationError("ROI coordinates out of bounds")
        assert "ROI coordinates" in str(roi_error)
        
        # Frame validation
        frame_error = ValidationError("Frame index out of range: 1000")
        assert "Frame index" in str(frame_error)


class TestROIError:
    """Test ROI specific errors."""
    
    def test_inheritance(self):
        """Test that ROIError inherits from BrightnessSorcererError."""
        error = ROIError("ROI operation failed")
        assert isinstance(error, BrightnessSorcererError)
        assert isinstance(error, ROIError)
    
    def test_roi_error_scenarios(self):
        """Test various ROI error scenarios."""
        # Invalid coordinates
        coord_error = ROIError("Invalid ROI coordinates: negative values")
        assert "Invalid ROI coordinates" in str(coord_error)
        
        # Size too small
        size_error = ROIError("ROI too small: minimum size is 10x10 pixels")
        assert "ROI too small" in str(size_error)


class TestCacheError:
    """Test cache specific errors."""
    
    def test_inheritance(self):
        """Test that CacheError inherits from BrightnessSorcererError."""
        error = CacheError("Cache operation failed")
        assert isinstance(error, BrightnessSorcererError)
        assert isinstance(error, CacheError)
    
    def test_cache_error_scenarios(self):
        """Test various cache error scenarios."""
        # Memory error
        memory_error = CacheError(
            "Cache memory limit exceeded",
            details={"current_size": 1024, "max_size": 512}
        )
        assert "Cache memory limit exceeded" in str(memory_error)
        assert memory_error.details["current_size"] == 1024


class TestAudioError:
    """Test audio specific errors."""
    
    def test_inheritance(self):
        """Test that AudioError inherits from BrightnessSorcererError."""
        error = AudioError("Audio system failed")
        assert isinstance(error, BrightnessSorcererError)
        assert isinstance(error, AudioError)
    
    def test_audio_error_scenarios(self):
        """Test various audio error scenarios."""
        # Initialization error
        init_error = AudioError("Failed to initialize audio system")
        assert "Failed to initialize" in str(init_error)
        
        # Playback error
        playback_error = AudioError(
            "Audio playback failed",
            error_code="AUDIO001",
            details={"device": "default", "format": "wav"}
        )
        assert "Audio playback failed" in str(playback_error)
        assert playback_error.error_code == "AUDIO001"


class TestExceptionHierarchy:
    """Test the overall exception hierarchy."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from BrightnessSorcererError."""
        exception_classes = [
            VideoLoadError,
            AnalysisError, 
            ConfigurationError,
            ValidationError,
            ROIError,
            CacheError,
            AudioError
        ]
        
        for exc_class in exception_classes:
            error = exc_class("Test message")
            assert isinstance(error, BrightnessSorcererError)
            assert isinstance(error, Exception)
    
    def test_exception_raising_and_catching(self):
        """Test that exceptions can be raised and caught properly."""
        # Test catching specific exception
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test validation error")
        
        assert "Test validation error" in str(exc_info.value)
        
        # Test catching base exception
        with pytest.raises(BrightnessSorcererError) as exc_info:
            raise VideoLoadError("Test video error")
        
        assert isinstance(exc_info.value, VideoLoadError)
        assert isinstance(exc_info.value, BrightnessSorcererError)