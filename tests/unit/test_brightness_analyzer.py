"""
Unit tests for BrightnessAnalyzer class.

Tests brightness analysis functionality including:
- CIE LAB color space conversions
- Blue channel analysis for bioluminescence detection
- Background subtraction and noise reduction
- Statistical computations (mean, median, robust stats)
- Reference mask-based analysis
- Edge case handling
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from brightness_sorcerer.core.brightness_analyzer import BrightnessAnalyzer
from brightness_sorcerer.utils.constants import (
    LAB_L_CHANNEL_SCALE, COLOR_CHANNEL_MAX, 
    TRIMMED_MEAN_LOWER_PERCENTILE, TRIMMED_MEAN_UPPER_PERCENTILE,
    MIN_PIXELS_FOR_ROBUST_STATS, MORPHOLOGICAL_KERNEL_SIZE
)


class TestBrightnessAnalyzerInitialization:
    """Test BrightnessAnalyzer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test analyzer initializes with default parameters."""
        analyzer = BrightnessAnalyzer()
        
        assert analyzer.analysis_method == 'enhanced'
        assert analyzer.morphological_cleanup is True
        assert analyzer.gaussian_blur_sigma == 0.0
        assert analyzer.mask_generation_method == 'threshold'
    
    def test_custom_initialization(self):
        """Test analyzer initializes with custom parameters."""
        analyzer = BrightnessAnalyzer(
            analysis_method='basic',
            morphological_cleanup=False,
            gaussian_blur_sigma=1.5
        )
        
        assert analyzer.analysis_method == 'basic'
        assert analyzer.morphological_cleanup is False
        assert analyzer.gaussian_blur_sigma == 1.5
        assert analyzer.mask_generation_method == 'threshold'
    
    def test_initialization_logging(self):
        """Test initialization creates appropriate log messages."""
        with patch('brightness_sorcerer.core.brightness_analyzer.logger') as mock_logger:
            analyzer = BrightnessAnalyzer(analysis_method='enhanced')
            mock_logger.debug.assert_called_once_with("BrightnessAnalyzer initialized: enhanced method")


class TestComputeBrightnessStats:
    """Test main brightness statistics computation method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BrightnessAnalyzer()
        
        # Create test BGR image (solid color)
        self.test_bgr = np.ones((10, 10, 3), dtype=np.uint8) * 128  # Mid-gray
        
        # Create gradient test image
        self.gradient_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        for i in range(10):
            self.gradient_bgr[i, :, :] = i * 25  # Gradient from 0 to 225
    
    def test_compute_brightness_stats_valid_input(self):
        """Test brightness computation with valid BGR input."""
        result = self.analyzer.compute_brightness_stats(self.test_bgr)
        
        assert len(result) == 8
        l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, \
        b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median = result
        
        # All values should be reasonable numbers
        assert isinstance(l_raw_mean, float)
        assert isinstance(l_raw_median, float)
        assert l_raw_mean > 0
        assert l_raw_median > 0
        
        # Without background subtraction, raw and bg_sub should be equal
        assert l_raw_mean == l_bg_sub_mean
        assert l_raw_median == l_bg_sub_median
        assert b_raw_mean == b_bg_sub_mean
        assert b_raw_median == b_bg_sub_median
    
    def test_compute_brightness_stats_with_background(self):
        """Test brightness computation with background subtraction."""
        background_brightness = 20.0
        
        result = self.analyzer.compute_brightness_stats(
            self.test_bgr, 
            background_brightness=background_brightness
        )
        
        l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, \
        b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median = result
        
        # Raw values should be unaffected
        assert l_raw_mean > 0
        assert l_raw_median > 0
        
        # Background-subtracted values should be different
        assert l_bg_sub_mean != l_raw_mean
        assert l_bg_sub_median != l_raw_median
    
    def test_compute_brightness_stats_empty_array(self):
        """Test brightness computation with empty array."""
        empty_array = np.array([])
        
        result = self.analyzer.compute_brightness_stats(empty_array)
        expected = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        assert result == expected
    
    def test_compute_brightness_stats_none_input(self):
        """Test brightness computation with None input."""
        result = self.analyzer.compute_brightness_stats(None)
        expected = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        assert result == expected
    
    def test_compute_brightness_stats_single_pixel(self):
        """Test brightness computation with single pixel."""
        single_pixel = np.array([[[100, 150, 200]]], dtype=np.uint8)
        
        result = self.analyzer.compute_brightness_stats(single_pixel)
        
        l_raw_mean, l_raw_median, _, _, b_raw_mean, b_raw_median, _, _ = result
        
        # Mean and median should be equal for single pixel
        assert l_raw_mean == l_raw_median
        assert b_raw_mean == b_raw_median
        assert b_raw_mean == 100.0  # Blue channel value
    
    def test_compute_brightness_stats_error_handling(self):
        """Test error handling in brightness computation."""
        with patch('cv2.cvtColor', side_effect=Exception("CV2 Error")):
            with patch('brightness_sorcerer.core.brightness_analyzer.logger') as mock_logger:
                result = self.analyzer.compute_brightness_stats(self.test_bgr)
                
                expected = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                assert result == expected
                mock_logger.error.assert_called_once()


class TestBackgroundSubtractedStats:
    """Test background subtraction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BrightnessAnalyzer(morphological_cleanup=True)
        
        # Create test data: L* channel and blue channel
        self.l_star = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        self.blue_chan = np.array([[100, 110, 120], [130, 140, 150]], dtype=np.float32)
    
    def test_background_subtracted_stats_basic(self):
        """Test basic background subtraction."""
        background_brightness = 25.0
        
        result = self.analyzer._compute_background_subtracted_stats(
            self.l_star, self.blue_chan, background_brightness
        )
        
        l_bg_sub_mean, l_bg_sub_median, b_bg_sub_mean, b_bg_sub_median = result
        
        # Only pixels above background (30, 40, 50, 60) should be considered
        # Background subtracted: (5, 15, 25, 35)
        assert l_bg_sub_mean > 0  # Should be positive
        assert l_bg_sub_median > 0
        assert b_bg_sub_mean > 0  # Blue channel mean of pixels above threshold
        assert b_bg_sub_median > 0
    
    def test_background_subtracted_stats_no_pixels_above_threshold(self):
        """Test background subtraction when no pixels are above threshold."""
        background_brightness = 100.0  # Higher than all values
        
        result = self.analyzer._compute_background_subtracted_stats(
            self.l_star, self.blue_chan, background_brightness
        )
        
        l_bg_sub_mean, l_bg_sub_median, b_bg_sub_mean, b_bg_sub_median = result
        
        # All should be zero when no pixels above threshold
        assert l_bg_sub_mean == 0.0
        assert l_bg_sub_median == 0.0
        assert b_bg_sub_mean == 0.0
        assert b_bg_sub_median == 0.0
    
    def test_background_subtracted_stats_with_morphological_cleanup(self):
        """Test background subtraction with morphological cleanup enabled."""
        background_brightness = 25.0
        
        with patch.object(self.analyzer, '_apply_morphological_cleanup') as mock_cleanup:
            # Mock returns the same mask (no change)
            mock_cleanup.return_value = self.l_star > background_brightness
            
            result = self.analyzer._compute_background_subtracted_stats(
                self.l_star, self.blue_chan, background_brightness
            )
            
            # Verify morphological cleanup was called
            mock_cleanup.assert_called_once()
            assert len(result) == 4
    
    def test_background_subtracted_stats_without_morphological_cleanup(self):
        """Test background subtraction with morphological cleanup disabled."""
        analyzer = BrightnessAnalyzer(morphological_cleanup=False)
        background_brightness = 25.0
        
        with patch.object(analyzer, '_apply_morphological_cleanup') as mock_cleanup:
            result = analyzer._compute_background_subtracted_stats(
                self.l_star, self.blue_chan, background_brightness
            )
            
            # Verify morphological cleanup was NOT called
            mock_cleanup.assert_not_called()
            assert len(result) == 4


class TestMorphologicalCleanup:
    """Test morphological cleanup operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BrightnessAnalyzer()
        
        # Create test binary mask
        self.test_mask = np.array([[True, False, True], 
                                  [False, True, False], 
                                  [True, False, True]], dtype=bool)
    
    def test_apply_morphological_cleanup(self):
        """Test morphological cleanup operation."""
        with patch('cv2.getStructuringElement') as mock_kernel, \
             patch('cv2.morphologyEx') as mock_morph:
            
            # Mock kernel and morphological operation
            mock_kernel.return_value = np.ones((3, 3), dtype=np.uint8)
            mock_morph.return_value = np.array([[255, 0, 255], 
                                              [0, 255, 0], 
                                              [255, 0, 255]], dtype=np.uint8)
            
            result = self.analyzer._apply_morphological_cleanup(self.test_mask)
            
            # Verify OpenCV functions were called correctly
            mock_kernel.assert_called_once_with(cv2.MORPH_ELLIPSE, (MORPHOLOGICAL_KERNEL_SIZE, MORPHOLOGICAL_KERNEL_SIZE))
            mock_morph.assert_called_once()
            
            # Result should be boolean array
            assert result.dtype == bool
            assert result.shape == self.test_mask.shape
    
    def test_morphological_cleanup_mask_conversion(self):
        """Test mask conversion in morphological cleanup."""
        with patch('cv2.getStructuringElement') as mock_kernel, \
             patch('cv2.morphologyEx') as mock_morph:
            
            mock_kernel.return_value = np.ones((3, 3), dtype=np.uint8)
            mock_morph.return_value = np.zeros((3, 3), dtype=np.uint8)  # All False result
            
            result = self.analyzer._apply_morphological_cleanup(self.test_mask)
            
            # Check that input mask was properly converted to uint8
            args, kwargs = mock_morph.call_args
            input_mask = args[0]
            assert input_mask.dtype == np.uint8
            assert np.array_equal(input_mask, self.test_mask.astype(np.uint8) * 255)


class TestRobustStats:
    """Test robust statistical computations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BrightnessAnalyzer()
        
        # Create test pixel data with outliers
        self.test_pixels = np.array([10, 12, 13, 14, 15, 16, 17, 18, 100], dtype=np.float32)  # 100 is outlier
    
    def test_compute_robust_stats_with_trimming(self):
        """Test robust statistics with trimmed mean."""
        background_brightness = 5.0
        
        result = self.analyzer._compute_robust_stats(self.test_pixels, background_brightness)
        robust_mean, robust_median = result
        
        # Robust mean should be less affected by outliers than regular mean
        assert isinstance(robust_mean, float)
        assert isinstance(robust_median, float)
        assert robust_mean > 0
        assert robust_median > 0
    
    def test_compute_robust_stats_small_sample(self):
        """Test robust statistics with small sample (≤5 pixels)."""
        small_pixels = np.array([10, 12, 13], dtype=np.float32)
        background_brightness = 5.0
        
        result = self.analyzer._compute_robust_stats(small_pixels, background_brightness)
        robust_mean, robust_median = result
        
        # Should fallback to regular mean/median for small samples
        expected_mean = float(np.mean(small_pixels - background_brightness))
        expected_median = float(np.median(small_pixels - background_brightness))
        
        assert robust_mean == expected_mean
        assert robust_median == expected_median
    
    def test_compute_robust_stats_empty_after_trimming(self):
        """Test robust statistics when trimming results in empty array."""
        # Create minimal dataset that becomes empty after trimming
        minimal_pixels = np.array([10], dtype=np.float32)
        background_brightness = 5.0
        
        result = self.analyzer._compute_robust_stats(minimal_pixels, background_brightness)
        robust_mean, robust_median = result
        
        # Should fallback to regular stats
        assert robust_mean == 5.0  # 10 - 5
        assert robust_median == 5.0


class TestVectorizedStats:
    """Test vectorized statistical computations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BrightnessAnalyzer()
    
    def test_compute_vectorized_stats_2d_array(self):
        """Test vectorized statistics on 2D array."""
        test_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        result = self.analyzer._compute_vectorized_stats(test_data)
        mean_val, median_val = result
        
        # Should flatten array and compute stats
        expected_mean = float(np.mean(test_data))
        expected_median = float(np.median(test_data))
        
        assert mean_val == expected_mean
        assert median_val == expected_median
    
    def test_compute_vectorized_stats_1d_array(self):
        """Test vectorized statistics on 1D array."""
        test_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        result = self.analyzer._compute_vectorized_stats(test_data)
        mean_val, median_val = result
        
        assert mean_val == 3.0
        assert median_val == 3.0
    
    def test_compute_vectorized_stats_empty_array(self):
        """Test vectorized statistics on empty array."""
        empty_data = np.array([], dtype=np.float32)
        
        result = self.analyzer._compute_vectorized_stats(empty_data)
        mean_val, median_val = result
        
        assert mean_val == 0.0
        assert median_val == 0.0


class TestROIQualityValidation:
    """Test ROI quality validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BrightnessAnalyzer()
        
        # Create test ROI data
        self.valid_roi = np.ones((20, 20, 3), dtype=np.uint8) * 100  # Large enough, non-zero
        self.small_roi = np.ones((5, 5, 3), dtype=np.uint8) * 100    # Too small
        self.zero_roi = np.zeros((20, 20, 3), dtype=np.uint8)        # All zero pixels
    
    def test_validate_roi_quality_valid_roi(self):
        """Test ROI quality validation with valid ROI."""
        # Create ROI with sufficient dynamic range (difference > 5)
        valid_roi_with_range = np.zeros((20, 20, 3), dtype=np.uint8)
        valid_roi_with_range[:10, :, :] = 50  # Dark region
        valid_roi_with_range[10:, :, :] = 200  # Bright region
        
        result = self.analyzer.validate_roi_quality(valid_roi_with_range, min_pixels=100)
        assert result is True
    
    def test_validate_roi_quality_too_small(self):
        """Test ROI quality validation with too small ROI."""
        result = self.analyzer.validate_roi_quality(self.small_roi, min_pixels=100)
        assert result is False
    
    def test_validate_roi_quality_zero_pixels(self):
        """Test ROI quality validation with all-zero ROI."""
        result = self.analyzer.validate_roi_quality(self.zero_roi, min_pixels=100)
        assert result is False
    
    def test_validate_roi_quality_custom_threshold(self):
        """Test ROI quality validation with custom pixel threshold."""
        # Create small ROI with dynamic range
        small_roi_with_range = np.zeros((5, 5, 3), dtype=np.uint8)
        small_roi_with_range[:2, :, :] = 50
        small_roi_with_range[2:, :, :] = 200
        
        # Should pass with lower threshold
        result = self.analyzer.validate_roi_quality(small_roi_with_range, min_pixels=10)
        assert result is True
        
        # Should fail with higher threshold
        result = self.analyzer.validate_roi_quality(small_roi_with_range, min_pixels=100)
        assert result is False


class TestAdaptiveMaskGeneration:
    """Test adaptive mask generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BrightnessAnalyzer()
        
        # Create L* channel data with varying brightness
        self.l_star_data = np.array([[10, 20, 30], 
                                   [40, 50, 60], 
                                   [70, 80, 90]], dtype=np.float32)
    
    def test_generate_adaptive_mask_basic(self):
        """Test basic adaptive mask generation."""
        threshold = 50.0
        
        result = self.analyzer.generate_adaptive_mask(self.l_star_data, threshold)
        
        # Result should be boolean array
        assert result.dtype == bool
        assert result.shape == self.l_star_data.shape
        
        # Pixels above threshold should be True
        expected_mask = self.l_star_data > threshold
        assert np.array_equal(result, expected_mask)
    
    def test_generate_adaptive_mask_all_below_threshold(self):
        """Test adaptive mask when all pixels below threshold."""
        threshold = 100.0  # Higher than all values
        
        result = self.analyzer.generate_adaptive_mask(self.l_star_data, threshold)
        
        # All should be False
        assert not np.any(result)
    
    def test_generate_adaptive_mask_all_above_threshold(self):
        """Test adaptive mask when all pixels above threshold."""
        threshold = 5.0  # Lower than all values
        
        result = self.analyzer.generate_adaptive_mask(self.l_star_data, threshold)
        
        # Result should be boolean array with some True values (adaptive threshold affects result)
        assert result.dtype == bool
        assert result.shape == self.l_star_data.shape
        # At least some pixels should be above threshold
        assert np.any(result)


class TestCleanup:
    """Test analyzer cleanup functionality."""
    
    def test_cleanup_method_exists(self):
        """Test that cleanup method exists and is callable."""
        analyzer = BrightnessAnalyzer()
        
        # Should not raise any exception
        analyzer.cleanup()
        
        # Method should exist
        assert hasattr(analyzer, 'cleanup')
        assert callable(analyzer.cleanup)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple methods."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.analyzer = BrightnessAnalyzer(
            analysis_method='enhanced',
            morphological_cleanup=True,
            gaussian_blur_sigma=0.5
        )
        
        # Create realistic test image (simulated bioluminescence)
        self.biolum_image = np.zeros((50, 50, 3), dtype=np.uint8)
        # Add bright spot in blue channel
        self.biolum_image[20:30, 20:30, 0] = 200  # Blue channel
        self.biolum_image[20:30, 20:30, 1] = 50   # Green channel  
        self.biolum_image[20:30, 20:30, 2] = 50   # Red channel
        # Add background noise
        self.biolum_image += np.random.randint(0, 20, self.biolum_image.shape, dtype=np.uint8)
    
    def test_complete_brightness_analysis_workflow(self):
        """Test complete brightness analysis workflow."""
        # Step 1: Validate ROI quality
        is_valid = self.analyzer.validate_roi_quality(self.biolum_image)
        assert is_valid is True
        
        # Step 2: Compute brightness statistics
        result = self.analyzer.compute_brightness_stats(
            self.biolum_image, 
            background_brightness=10.0
        )
        
        l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, \
        b_raw_mean, b_raw_median, b_bg_sub_mean, b_bg_sub_median = result
        
        # Verify all statistics are computed
        assert all(isinstance(val, float) for val in result)
        assert all(val >= 0 for val in result)
        
        # Blue channel should show higher values due to bright spot
        assert b_raw_mean > 10  # Should be above background noise (adjusted for realistic values)
        assert b_bg_sub_mean >= 0
    
    def test_edge_case_handling_integration(self):
        """Test integration handling of multiple edge cases."""
        edge_cases = [
            np.array([]),  # Empty array
            None,          # None input
            np.ones((1, 1, 3), dtype=np.uint8),  # Single pixel
            np.zeros((100, 100, 3), dtype=np.uint8)  # All zeros
        ]
        
        for test_case in edge_cases:
            result = self.analyzer.compute_brightness_stats(test_case)
            assert len(result) == 8
            assert all(isinstance(val, float) for val in result)
            # Should not raise exceptions
    
    def test_performance_with_large_roi(self):
        """Test performance characteristics with large ROI."""
        # Create large test ROI (simulating real-world usage)
        large_roi = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        
        # Should complete without errors
        result = self.analyzer.compute_brightness_stats(
            large_roi,
            background_brightness=20.0
        )
        
        assert len(result) == 8
        assert all(isinstance(val, float) for val in result)
        
        # Performance check: should complete reasonably quickly
        # (This is more of a smoke test than a strict performance test)
        import time
        start_time = time.time()
        for _ in range(10):  # Run multiple times
            self.analyzer.compute_brightness_stats(large_roi)
        elapsed = time.time() - start_time
        
        # Should complete 10 iterations in reasonable time (adjust as needed)
        assert elapsed < 5.0  # 5 seconds for 10 iterations


if __name__ == '__main__':
    pytest.main([__file__])