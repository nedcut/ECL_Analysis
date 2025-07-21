"""
Unit tests for validation functions.

Tests input validation, file validation, and safe type conversion functions.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from brightness_sorcerer.core.exceptions import ValidationError
from brightness_sorcerer.utils.validation import (
    validate_video_file,
    validate_frame_range, 
    safe_float_conversion,
    safe_int_conversion
)


class TestValidateVideoFile:
    """Test video file validation."""
    
    def test_validate_video_file_empty_path(self):
        """Test validation fails for empty path."""
        with pytest.raises(ValidationError, match="Video file path cannot be empty"):
            validate_video_file("")
        
        with pytest.raises(ValidationError, match="Video file path cannot be empty"):
            validate_video_file(None)
    
    def test_validate_video_file_nonexistent(self):
        """Test validation fails for non-existent file."""
        with pytest.raises(ValidationError, match="Video file does not exist"):
            validate_video_file("/nonexistent/path/video.mp4")
    
    def test_validate_video_file_not_a_file(self):
        """Test validation fails for directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError, match="Path is not a file"):
                validate_video_file(temp_dir)
    
    def test_validate_video_file_unsupported_format(self):
        """Test validation fails for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_file.flush()
            
            try:
                with pytest.raises(ValidationError, match="Unsupported video format"):
                    validate_video_file(temp_file.name)
            finally:
                os.unlink(temp_file.name)
    
    def test_validate_video_file_empty_file(self):
        """Test validation fails for empty video file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            # File is created but empty
            try:
                with pytest.raises(ValidationError, match="Video file is empty"):
                    validate_video_file(temp_file.name)
            finally:
                os.unlink(temp_file.name)
    
    def test_validate_video_file_valid_formats(self):
        """Test validation passes for supported formats."""
        supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.m4v', '.flv']
        
        for ext in supported_formats:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                # Write some dummy content
                temp_file.write(b"dummy video content")
                temp_file.flush()
                
                try:
                    # Should not raise exception
                    result = validate_video_file(temp_file.name)
                    assert result is True
                finally:
                    os.unlink(temp_file.name)
    
    def test_validate_video_file_case_insensitive(self):
        """Test validation is case insensitive for file extensions."""
        with tempfile.NamedTemporaryFile(suffix='.MP4', delete=False) as temp_file:
            temp_file.write(b"dummy video content")
            temp_file.flush()
            
            try:
                result = validate_video_file(temp_file.name)
                assert result is True
            finally:
                os.unlink(temp_file.name)


class TestValidateFrameRange:
    """Test frame range validation."""
    
    def test_validate_frame_range_negative_start(self):
        """Test validation fails for negative start frame."""
        with pytest.raises(ValidationError, match="Start frame cannot be negative"):
            validate_frame_range(-1, 100, 200)
    
    def test_validate_frame_range_end_before_start(self):
        """Test validation fails when end frame is before start frame."""
        with pytest.raises(ValidationError, match="End frame .* cannot be less than start frame"):
            validate_frame_range(100, 50, 200)
    
    def test_validate_frame_range_start_exceeds_total(self):
        """Test validation fails when start frame exceeds total frames."""
        with pytest.raises(ValidationError, match="Start frame .* must be less than total frames"):
            validate_frame_range(200, 250, 150)
    
    def test_validate_frame_range_end_exceeds_total(self):
        """Test validation fails when end frame exceeds total frames."""
        with pytest.raises(ValidationError, match="End frame .* must be less than total frames"):
            validate_frame_range(50, 200, 150)
    
    def test_validate_frame_range_valid_ranges(self):
        """Test validation passes for valid frame ranges."""
        test_cases = [
            (0, 50, 100),      # Start from beginning
            (25, 75, 100),     # Middle range
            (0, 99, 100),      # Full range
            (50, 50, 100),     # Single frame
        ]
        
        for start, end, total in test_cases:
            result = validate_frame_range(start, end, total)
            assert result is True
    
    def test_validate_frame_range_edge_cases(self):
        """Test validation for edge cases."""
        # Last possible frame
        result = validate_frame_range(99, 99, 100)
        assert result is True
        
        # First frame only
        result = validate_frame_range(0, 0, 100)
        assert result is True


class TestSafeFloatConversion:
    """Test safe float conversion function."""
    
    def test_safe_float_conversion_valid_numbers(self):
        """Test conversion of valid numeric values."""
        test_cases = [
            ("3.14", 3.14),
            ("42", 42.0),
            ("-1.5", -1.5),
            ("0", 0.0),
            ("1e-3", 0.001),
        ]
        
        for input_val, expected in test_cases:
            result = safe_float_conversion(input_val)
            assert result == expected
    
    def test_safe_float_conversion_invalid_values(self):
        """Test conversion falls back to default for invalid values."""
        invalid_values = ["abc", "", None, [], {}]
        default_value = 5.0
        
        for invalid_val in invalid_values:
            result = safe_float_conversion(invalid_val, default=default_value)
            assert result == default_value
    
    def test_safe_float_conversion_with_bounds(self):
        """Test conversion respects minimum and maximum bounds."""
        # Test minimum bound
        result = safe_float_conversion("-10", min_val=0.0)
        assert result == 0.0
        
        # Test maximum bound
        result = safe_float_conversion("100", max_val=50.0)
        assert result == 50.0
        
        # Test both bounds
        result = safe_float_conversion("75", min_val=10.0, max_val=50.0)
        assert result == 50.0
        
        result = safe_float_conversion("5", min_val=10.0, max_val=50.0)
        assert result == 10.0
        
        # Test within bounds
        result = safe_float_conversion("25", min_val=10.0, max_val=50.0)
        assert result == 25.0
    
    def test_safe_float_conversion_default_values(self):
        """Test various default values."""
        defaults = [0.0, 1.0, -1.0, 3.14159]
        
        for default in defaults:
            result = safe_float_conversion("invalid", default=default)
            assert result == default
    
    @patch('brightness_sorcerer.utils.validation.logger')
    def test_safe_float_conversion_logging(self, mock_logger):
        """Test that warnings are logged for invalid conversions."""
        safe_float_conversion("invalid_value", default=0.0)
        mock_logger.warning.assert_called()


class TestSafeIntConversion:
    """Test safe integer conversion function."""
    
    def test_safe_int_conversion_valid_numbers(self):
        """Test conversion of valid numeric values."""
        test_cases = [
            ("42", 42),
            ("0", 0),
            ("-15", -15),
            ("100", 100),  # Valid integer string
        ]
        
        for input_val, expected in test_cases:
            result = safe_int_conversion(input_val)
            assert result == expected
    
    def test_safe_int_conversion_float_strings(self):
        """Test that float strings fall back to default since int() can't parse them."""
        test_cases = ["3.14", "1e2"]
        default_value = 0
        
        for input_val in test_cases:
            result = safe_int_conversion(input_val, default=default_value)
            assert result == default_value
    
    def test_safe_int_conversion_invalid_values(self):
        """Test conversion falls back to default for invalid values."""
        invalid_values = ["abc", "", None, [], {}]
        default_value = 10
        
        for invalid_val in invalid_values:
            result = safe_int_conversion(invalid_val, default=default_value)
            assert result == default_value
    
    def test_safe_int_conversion_with_bounds(self):
        """Test conversion respects minimum and maximum bounds."""
        # Test minimum bound
        result = safe_int_conversion("-10", min_val=0)
        assert result == 0
        
        # Test maximum bound
        result = safe_int_conversion("100", max_val=50)
        assert result == 50
        
        # Test both bounds
        result = safe_int_conversion("75", min_val=10, max_val=50)
        assert result == 50
        
        result = safe_int_conversion("5", min_val=10, max_val=50)
        assert result == 10
        
        # Test within bounds
        result = safe_int_conversion("25", min_val=10, max_val=50)
        assert result == 25
    
    def test_safe_int_conversion_default_values(self):
        """Test various default values."""
        defaults = [0, 1, -1, 42]
        
        for default in defaults:
            result = safe_int_conversion("invalid", default=default)
            assert result == default
    
    @patch('brightness_sorcerer.utils.validation.logger')
    def test_safe_int_conversion_logging(self, mock_logger):
        """Test that warnings are logged for invalid conversions."""
        safe_int_conversion("invalid_value", default=0)
        mock_logger.warning.assert_called()


class TestValidationIntegration:
    """Test integration between validation functions."""
    
    def test_validation_error_consistency(self):
        """Test that all validation functions use consistent error messages."""
        # All should raise ValidationError for invalid inputs
        with pytest.raises(ValidationError):
            validate_video_file("")
        
        with pytest.raises(ValidationError):
            validate_frame_range(-1, 10, 100)
    
    def test_validation_success_consistency(self):
        """Test that all validation functions return True on success."""
        # Create a temporary valid video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(b"dummy video content")
            temp_file.flush()
            
            try:
                # Video file validation
                result = validate_video_file(temp_file.name)
                assert result is True
                
                # Frame range validation
                result = validate_frame_range(0, 50, 100)
                assert result is True
                
            finally:
                os.unlink(temp_file.name)