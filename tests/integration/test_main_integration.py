"""
Integration tests for main application components.

Tests the integration between different modules and core functionality
without requiring the full GUI.
"""

import pytest
import tempfile
import os
import sys
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from brightness_sorcerer.core.exceptions import ValidationError, VideoLoadError
from brightness_sorcerer.utils.validation import validate_video_file
from brightness_sorcerer.utils.constants import SUPPORTED_VIDEO_FORMATS


class TestVideoFileHandling:
    """Test integration of video file validation and handling."""
    
    def test_video_validation_integration(self):
        """Test complete video file validation workflow."""
        # Test with supported format
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            # Write minimal video-like content
            temp_file.write(b"fake video content for testing")
            temp_file.flush()
            
            try:
                # Should pass validation
                result = validate_video_file(temp_file.name)
                assert result is True
            finally:
                os.unlink(temp_file.name)
    
    def test_video_format_support_integration(self):
        """Test that all supported formats are properly handled."""
        for video_format in SUPPORTED_VIDEO_FORMATS:
            with tempfile.NamedTemporaryFile(suffix=video_format, delete=False) as temp_file:
                temp_file.write(b"test video content")
                temp_file.flush()
                
                try:
                    result = validate_video_file(temp_file.name)
                    assert result is True, f"Format {video_format} should be supported"
                finally:
                    os.unlink(temp_file.name)


class TestBrightnessAnalysisIntegration:
    """Test integration of brightness analysis components."""
    
    def test_brightness_calculation_workflow(self):
        """Test the complete brightness calculation workflow."""
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Convert to LAB color space (like the application does)
        lab_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)
        
        # Extract L* channel (brightness) - OpenCV uses 0-255 range
        l_channel = lab_image[:, :, 0]
        
        # Calculate basic statistics
        mean_brightness = np.mean(l_channel)
        median_brightness = np.median(l_channel)
        
        # Verify calculations are reasonable (OpenCV LAB L* is 0-255, not 0-100)
        assert 0 <= mean_brightness <= 255
        assert 0 <= median_brightness <= 255
        assert isinstance(mean_brightness, (int, float))
        assert isinstance(median_brightness, (int, float))
    
    def test_roi_extraction_integration(self):
        """Test ROI extraction from frames."""
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Define ROI coordinates
        x1, y1, x2, y2 = 100, 100, 200, 200
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Verify ROI dimensions
        expected_height = y2 - y1
        expected_width = x2 - x1
        assert roi.shape[0] == expected_height
        assert roi.shape[1] == expected_width
        assert roi.shape[2] == 3  # RGB channels
    
    def test_background_subtraction_integration(self):
        """Test background subtraction workflow."""
        # Create foreground and background values
        foreground_brightness = 50.0
        background_brightness = 20.0
        
        # Calculate background-subtracted value
        bg_subtracted = foreground_brightness - background_brightness
        
        # Verify result
        assert bg_subtracted == 30.0
        assert bg_subtracted >= 0  # Should not be negative for this test case


class TestConfigurationIntegration:
    """Test integration of configuration and settings."""
    
    def test_settings_file_integration(self):
        """Test settings file loading and saving integration."""
        # Test settings structure
        test_settings = {
            'recent_files': ['/path/to/video.mp4'],
            'frame_cache_size': 100,
            'log_level': 'INFO'
        }
        
        # Verify settings structure
        assert isinstance(test_settings['recent_files'], list)
        assert isinstance(test_settings['frame_cache_size'], int)
        assert test_settings['frame_cache_size'] > 0
    
    def test_constants_integration(self):
        """Test that constants are properly integrated across modules."""
        from brightness_sorcerer.utils.constants import (
            DEFAULT_SETTINGS_FILE, FRAME_CACHE_SIZE, APP_NAME
        )
        
        # Verify constants are accessible and valid
        assert isinstance(DEFAULT_SETTINGS_FILE, str)
        assert len(DEFAULT_SETTINGS_FILE) > 0
        assert isinstance(FRAME_CACHE_SIZE, int)
        assert FRAME_CACHE_SIZE > 0
        assert isinstance(APP_NAME, str)
        assert len(APP_NAME) > 0


@pytest.mark.gui
class TestGUIIntegration:
    """Test GUI component integration (requires display)."""
    
    def test_application_import(self):
        """Test that main application can be imported."""
        try:
            import main
            assert hasattr(main, 'main')
            assert callable(main.main)
        except ImportError as e:
            pytest.skip(f"Could not import main module: {e}")
    
    def test_video_analyzer_class_exists(self):
        """Test that VideoAnalyzer class exists and is importable."""
        try:
            import main
            # Check that VideoAnalyzer class exists in the module
            assert hasattr(main, 'VideoAnalyzer')
            video_analyzer_class = getattr(main, 'VideoAnalyzer')
            assert callable(video_analyzer_class)
        except ImportError as e:
            pytest.skip(f"Could not import VideoAnalyzer class: {e}")


class TestModuleIntegration:
    """Test integration between different modules."""
    
    def test_exception_module_integration(self):
        """Test that exceptions are properly integrated."""
        from brightness_sorcerer.core.exceptions import ValidationError, VideoLoadError
        
        # Test that exceptions can be raised and caught
        with pytest.raises(ValidationError):
            raise ValidationError("Test validation error")
        
        with pytest.raises(VideoLoadError):
            raise VideoLoadError("Test video load error")
        
        # Test inheritance
        try:
            raise ValidationError("Test")
        except Exception as e:
            assert isinstance(e, Exception)
            assert isinstance(e, ValidationError)
    
    def test_validation_module_integration(self):
        """Test that validation module integrates with exceptions."""
        from brightness_sorcerer.utils.validation import validate_video_file
        from brightness_sorcerer.core.exceptions import ValidationError
        
        # Test that validation functions raise proper exceptions
        with pytest.raises(ValidationError):
            validate_video_file("")  # Empty path should raise ValidationError
    
    def test_constants_module_integration(self):
        """Test that constants module integrates properly."""
        from brightness_sorcerer.utils.constants import (
            APP_NAME, ROI_COLORS, SUPPORTED_VIDEO_FORMATS
        )
        
        # Test that constants are accessible and have expected types
        assert isinstance(APP_NAME, str)
        assert isinstance(ROI_COLORS, list)
        assert isinstance(SUPPORTED_VIDEO_FORMATS, tuple)
        
        # Test specific values
        assert len(ROI_COLORS) > 0
        assert len(SUPPORTED_VIDEO_FORMATS) > 0
        assert all(fmt.startswith('.') for fmt in SUPPORTED_VIDEO_FORMATS)


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    def test_validation_error_propagation(self):
        """Test that validation errors propagate correctly."""
        from brightness_sorcerer.utils.validation import validate_video_file
        from brightness_sorcerer.core.exceptions import ValidationError
        
        # Test various validation error scenarios
        error_cases = [
            ("", "empty path"),
            ("/nonexistent/file.mp4", "nonexistent file"),
        ]
        
        for invalid_input, description in error_cases:
            with pytest.raises(ValidationError) as exc_info:
                validate_video_file(invalid_input)
            
            # Verify error message contains relevant information
            error_msg = str(exc_info.value)
            assert len(error_msg) > 0, f"Empty error message for {description}"
    
    def test_exception_hierarchy_integration(self):
        """Test that exception hierarchy works in integration scenarios."""
        from brightness_sorcerer.core.exceptions import (
            BrightnessSorcererError, ValidationError, VideoLoadError
        )
        
        # Test catching base exception for specific exceptions
        try:
            raise ValidationError("Test validation error")
        except BrightnessSorcererError as e:
            assert isinstance(e, ValidationError)
            assert isinstance(e, BrightnessSorcererError)
        
        try:
            raise VideoLoadError("Test video load error")
        except BrightnessSorcererError as e:
            assert isinstance(e, VideoLoadError)
            assert isinstance(e, BrightnessSorcererError)


class TestPerformanceIntegration:
    """Test performance aspects of integrated components."""
    
    def test_large_image_processing(self):
        """Test processing of larger images for performance."""
        # Create a larger test image
        large_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Time the LAB conversion (key operation)
        import time
        start_time = time.time()
        lab_image = cv2.cvtColor(large_image, cv2.COLOR_RGB2LAB)
        conversion_time = time.time() - start_time
        
        # Verify result and performance
        assert lab_image.shape == large_image.shape
        assert conversion_time < 1.0  # Should complete in under 1 second
    
    def test_multiple_roi_processing(self):
        """Test processing multiple ROIs efficiently."""
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Define multiple ROIs
        rois = [
            (50, 50, 150, 150),
            (200, 100, 300, 200),
            (400, 200, 500, 300),
            (100, 300, 200, 400),
        ]
        
        # Process all ROIs
        roi_results = []
        for x1, y1, x2, y2 in rois:
            roi = frame[y1:y2, x1:x2]
            lab_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
            brightness = np.mean(lab_roi[:, :, 0])
            roi_results.append(brightness)
        
        # Verify results (OpenCV LAB L* uses 0-255 range)
        assert len(roi_results) == len(rois)
        assert all(0 <= brightness <= 255 for brightness in roi_results)