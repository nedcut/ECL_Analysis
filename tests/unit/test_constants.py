"""
Unit tests for application constants.

Tests that all constants are properly defined and have expected values.
"""

import pytest
from brightness_sorcerer.utils import constants


class TestApplicationConstants:
    """Test application metadata constants."""
    
    def test_app_metadata_defined(self):
        """Test that application metadata constants are defined."""
        assert hasattr(constants, 'APP_NAME')
        assert hasattr(constants, 'APP_VERSION')
        assert constants.APP_NAME == "Brightness Sorcerer"
        assert constants.APP_VERSION == "2.0.0"
    
    def test_app_metadata_types(self):
        """Test that application metadata constants have correct types."""
        assert isinstance(constants.APP_NAME, str)
        assert isinstance(constants.APP_VERSION, str)
        assert len(constants.APP_NAME) > 0
        assert len(constants.APP_VERSION) > 0


class TestUIConstants:
    """Test UI styling and color constants."""
    
    def test_color_constants_defined(self):
        """Test that all color constants are defined."""
        color_constants = [
            'COLOR_BACKGROUND', 'COLOR_FOREGROUND', 'COLOR_ACCENT',
            'COLOR_ACCENT_HOVER', 'COLOR_SECONDARY', 'COLOR_SECONDARY_LIGHT',
            'COLOR_SUCCESS', 'COLOR_WARNING', 'COLOR_ERROR', 'COLOR_INFO',
            'COLOR_BRIGHTNESS_LABEL'
        ]
        
        for const_name in color_constants:
            assert hasattr(constants, const_name), f"Missing constant: {const_name}"
    
    def test_color_format(self):
        """Test that color constants are in valid hex format."""
        color_constants = [
            constants.COLOR_BACKGROUND, constants.COLOR_FOREGROUND,
            constants.COLOR_ACCENT, constants.COLOR_SUCCESS
        ]
        
        for color in color_constants:
            assert isinstance(color, str)
            assert color.startswith('#')
            assert len(color) == 7  # #RRGGBB format
            # Test that it's valid hex
            int(color[1:], 16)  # Should not raise ValueError
    
    def test_roi_colors_defined(self):
        """Test that ROI colors are properly defined."""
        assert hasattr(constants, 'ROI_COLORS')
        assert isinstance(constants.ROI_COLORS, list)
        assert len(constants.ROI_COLORS) == 8  # Should have 8 colors
        
        for color in constants.ROI_COLORS:
            assert isinstance(color, tuple)
            assert len(color) == 3  # RGB tuple
            for component in color:
                assert isinstance(component, int)
                assert 0 <= component <= 255
    
    def test_roi_styling_constants(self):
        """Test ROI styling constants."""
        assert hasattr(constants, 'ROI_THICKNESS_DEFAULT')
        assert hasattr(constants, 'ROI_THICKNESS_SELECTED')
        assert hasattr(constants, 'ROI_HANDLE_SIZE')
        assert hasattr(constants, 'ROI_MIN_SIZE')
        
        assert constants.ROI_THICKNESS_DEFAULT > 0
        assert constants.ROI_THICKNESS_SELECTED > constants.ROI_THICKNESS_DEFAULT
        assert constants.ROI_HANDLE_SIZE > 0
        assert constants.ROI_MIN_SIZE > 0


class TestVideoConstants:
    """Test video processing constants."""
    
    def test_supported_video_formats(self):
        """Test supported video formats are defined."""
        assert hasattr(constants, 'SUPPORTED_VIDEO_FORMATS')
        assert isinstance(constants.SUPPORTED_VIDEO_FORMATS, tuple)
        assert len(constants.SUPPORTED_VIDEO_FORMATS) > 0
        
        # Check that all formats start with a dot
        for fmt in constants.SUPPORTED_VIDEO_FORMATS:
            assert isinstance(fmt, str)
            assert fmt.startswith('.')
            assert len(fmt) > 1
    
    def test_video_formats_include_common_types(self):
        """Test that common video formats are included."""
        common_formats = ['.mp4', '.mov', '.avi']
        for fmt in common_formats:
            assert fmt in constants.SUPPORTED_VIDEO_FORMATS
    
    def test_cache_size_constants(self):
        """Test frame cache size constants."""
        assert hasattr(constants, 'FRAME_CACHE_SIZE')
        assert isinstance(constants.FRAME_CACHE_SIZE, int)
        assert constants.FRAME_CACHE_SIZE > 0
        assert constants.FRAME_CACHE_SIZE <= 1000  # Reasonable upper bound
    
    def test_navigation_constants(self):
        """Test video navigation constants."""
        assert hasattr(constants, 'JUMP_FRAMES')
        assert hasattr(constants, 'MAX_RECENT_FILES')
        
        assert isinstance(constants.JUMP_FRAMES, int)
        assert isinstance(constants.MAX_RECENT_FILES, int)
        assert constants.JUMP_FRAMES > 0
        assert constants.MAX_RECENT_FILES > 0


class TestAnalysisConstants:
    """Test analysis parameter constants."""
    
    def test_brightness_analysis_constants(self):
        """Test brightness analysis constants."""
        analysis_constants = [
            'DEFAULT_MANUAL_THRESHOLD',
            'AUTO_DETECT_BASELINE_PERCENTILE', 
            'BRIGHTNESS_NOISE_FLOOR_PERCENTILE'
        ]
        
        for const_name in analysis_constants:
            assert hasattr(constants, const_name), f"Missing constant: {const_name}"
            const_value = getattr(constants, const_name)
            assert isinstance(const_value, (int, float))
            assert const_value >= 0
    
    def test_threshold_values_reasonable(self):
        """Test that threshold values are in reasonable ranges."""
        assert 0 <= constants.AUTO_DETECT_BASELINE_PERCENTILE <= 100
        assert 0 <= constants.BRIGHTNESS_NOISE_FLOOR_PERCENTILE <= 100
        assert constants.DEFAULT_MANUAL_THRESHOLD > 0
    
    def test_low_light_enhancement_constants(self):
        """Test low-light enhancement constants."""
        enhancement_constants = [
            'LOW_LIGHT_BILATERAL_D',
            'LOW_LIGHT_BILATERAL_SIGMA_COLOR',
            'LOW_LIGHT_BILATERAL_SIGMA_SPACE',
            'LOW_LIGHT_CHANNEL_BOOST_FACTOR',
            'LOW_LIGHT_SIGNAL_AMPLIFICATION_FACTOR'
        ]
        
        for const_name in enhancement_constants:
            assert hasattr(constants, const_name), f"Missing constant: {const_name}"
            const_value = getattr(constants, const_name)
            assert isinstance(const_value, (int, float))
            assert const_value > 0


class TestAudioConstants:
    """Test audio processing constants."""
    
    def test_audio_processing_constants(self):
        """Test audio processing constants."""
        audio_constants = [
            'AUDIO_BEEP_FREQUENCY_RANGE',
            'AUDIO_BEEP_MIN_DURATION',
            'AUDIO_BEEP_MIN_AMPLITUDE',
            'AUDIO_SAMPLE_RATE'
        ]
        
        for const_name in audio_constants:
            assert hasattr(constants, const_name), f"Missing constant: {const_name}"
    
    def test_audio_frequency_range(self):
        """Test audio frequency range is valid."""
        freq_range = constants.AUDIO_BEEP_FREQUENCY_RANGE
        assert isinstance(freq_range, tuple)
        assert len(freq_range) == 2
        assert freq_range[0] < freq_range[1]  # Min < Max
        assert freq_range[0] > 0  # Positive frequencies
    
    def test_audio_timing_constants(self):
        """Test audio timing constants."""
        assert constants.AUDIO_BEEP_MIN_DURATION > 0
        assert constants.AUDIO_BEEP_MIN_AMPLITUDE > 0
        assert constants.AUDIO_SAMPLE_RATE > 0
        assert constants.AUDIO_SAMPLE_RATE >= 8000  # Reasonable minimum


class TestFileConstants:
    """Test file and path constants."""
    
    def test_settings_file_constants(self):
        """Test settings file path constants."""
        assert hasattr(constants, 'DEFAULT_SETTINGS_FILE')
        assert hasattr(constants, 'DEFAULT_LOG_FILE')
        assert hasattr(constants, 'BACKUP_SETTINGS_FILE')
        
        # Should be relative paths
        assert not constants.DEFAULT_SETTINGS_FILE.startswith('/')
        assert not constants.DEFAULT_LOG_FILE.startswith('/')
        assert not constants.BACKUP_SETTINGS_FILE.startswith('/')
        
        # Should have reasonable file extensions
        assert constants.DEFAULT_SETTINGS_FILE.endswith('.json')
        assert constants.DEFAULT_LOG_FILE.endswith('.log')
        assert constants.BACKUP_SETTINGS_FILE.endswith('.json')
    
    def test_settings_paths_in_config_folder(self):
        """Test that settings files are in config folder."""
        assert constants.DEFAULT_SETTINGS_FILE.startswith('config/')
        assert constants.BACKUP_SETTINGS_FILE.startswith('config/')


class TestPerformanceConstants:
    """Test performance-related constants."""
    
    def test_performance_constants_defined(self):
        """Test that performance constants are defined."""
        perf_constants = [
            'MAX_CACHE_MEMORY_MB',
            'UPDATE_INTERVAL_MS', 
            'PROGRESS_UPDATE_INTERVAL'
        ]
        
        for const_name in perf_constants:
            assert hasattr(constants, const_name), f"Missing constant: {const_name}"
            const_value = getattr(constants, const_name)
            assert isinstance(const_value, int)
            assert const_value > 0
    
    def test_performance_values_reasonable(self):
        """Test that performance values are reasonable."""
        assert constants.MAX_CACHE_MEMORY_MB <= 2048  # Reasonable upper bound
        assert constants.UPDATE_INTERVAL_MS >= 16  # At least 60 FPS
        assert constants.PROGRESS_UPDATE_INTERVAL >= 1


class TestPlotConstants:
    """Test plotting and visualization constants."""
    
    def test_plot_constants_defined(self):
        """Test that plot constants are defined."""
        plot_constants = [
            'PLOT_DPI',
            'PLOT_FIGURE_SIZE', 
            'PLOT_LINE_WIDTH'
        ]
        
        for const_name in plot_constants:
            assert hasattr(constants, const_name), f"Missing constant: {const_name}"
    
    def test_plot_values_valid(self):
        """Test that plot values are valid."""
        assert isinstance(constants.PLOT_DPI, int)
        assert constants.PLOT_DPI > 0
        assert constants.PLOT_DPI <= 600  # Reasonable upper bound
        
        assert isinstance(constants.PLOT_FIGURE_SIZE, tuple)
        assert len(constants.PLOT_FIGURE_SIZE) == 2
        assert all(isinstance(x, (int, float)) and x > 0 for x in constants.PLOT_FIGURE_SIZE)
        
        assert isinstance(constants.PLOT_LINE_WIDTH, (int, float))
        assert constants.PLOT_LINE_WIDTH > 0


class TestConstantConsistency:
    """Test consistency between related constants."""
    
    def test_thickness_consistency(self):
        """Test that thickness constants are consistent."""
        assert constants.ROI_THICKNESS_SELECTED > constants.ROI_THICKNESS_DEFAULT
    
    def test_percentile_constants_valid(self):
        """Test that percentile constants are valid percentages."""
        percentile_constants = [
            constants.AUTO_DETECT_BASELINE_PERCENTILE,
            constants.BRIGHTNESS_NOISE_FLOOR_PERCENTILE
        ]
        
        for percentile in percentile_constants:
            assert 0 <= percentile <= 100
    
    def test_audio_range_consistency(self):
        """Test that audio frequency range is consistent."""
        freq_min, freq_max = constants.AUDIO_BEEP_FREQUENCY_RANGE
        assert freq_min < freq_max
        assert freq_min > 0
        assert freq_max <= constants.AUDIO_SAMPLE_RATE / 2  # Nyquist limit