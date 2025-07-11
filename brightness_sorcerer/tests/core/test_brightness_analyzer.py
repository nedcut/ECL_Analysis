"""Tests for BrightnessAnalyzer class."""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from ...core.brightness_analyzer import BrightnessAnalyzer
from ...models.roi import ROI
from ...models.analysis_result import AnalysisResult


class TestBrightnessAnalyzer:
    """Tests for BrightnessAnalyzer class."""
    
    @pytest.mark.unit
    def test_init(self, mock_video_processor):
        """Test BrightnessAnalyzer initialization."""
        analyzer = BrightnessAnalyzer(mock_video_processor)
        
        assert analyzer.video_processor is mock_video_processor
        assert analyzer.manual_threshold == 5.0
        assert analyzer.noise_floor == 10.0
        assert analyzer.progress_callback is None
        assert not analyzer.cancel_requested
        assert analyzer.current_result is None
    
    @pytest.mark.unit
    def test_set_progress_callback(self, brightness_analyzer, mock_progress_callback):
        """Test setting progress callback."""
        brightness_analyzer.set_progress_callback(mock_progress_callback)
        assert brightness_analyzer.progress_callback is mock_progress_callback
    
    @pytest.mark.unit
    def test_cancel_analysis(self, brightness_analyzer):
        """Test canceling analysis."""
        brightness_analyzer.cancel_analysis()
        assert brightness_analyzer.cancel_requested
    
    @pytest.mark.unit
    def test_auto_detect_frame_range_no_video(self, brightness_analyzer, sample_rois):
        """Test auto-detection with no video loaded."""
        brightness_analyzer.video_processor.is_loaded.return_value = False
        
        result = brightness_analyzer.auto_detect_frame_range(sample_rois)
        assert result is None
    
    @pytest.mark.unit
    def test_auto_detect_frame_range_no_rois(self, brightness_analyzer):
        """Test auto-detection with no ROIs."""
        result = brightness_analyzer.auto_detect_frame_range([])
        assert result is None
    
    @pytest.mark.unit
    @patch('brightness_sorcerer.core.brightness_analyzer.detect_brightness_threshold')
    @patch('brightness_sorcerer.core.brightness_analyzer.find_analysis_range')
    def test_auto_detect_frame_range_success(self, mock_find_range, mock_detect_threshold, 
                                           brightness_analyzer, sample_rois, sample_video_frame):
        """Test successful auto-detection."""
        # Setup mocks
        brightness_analyzer.video_processor.is_loaded.return_value = True
        brightness_analyzer.video_processor.total_frames = 100
        brightness_analyzer.video_processor.get_frame_at_index.return_value = sample_video_frame
        brightness_analyzer.video_processor.get_current_frame.return_value = sample_video_frame
        
        mock_detect_threshold.return_value = 15.0
        mock_find_range.return_value = (10, 80)
        
        # Set progress callback to track calls
        progress_calls = []
        def progress_callback(current, total, message):
            progress_calls.append((current, total, message))
        brightness_analyzer.set_progress_callback(progress_callback)
        
        # Run auto-detection
        result = brightness_analyzer.auto_detect_frame_range(sample_rois[:2])  # Non-background ROIs
        
        assert result == (10, 80)
        assert len(progress_calls) > 0  # Progress should be reported
        mock_find_range.assert_called_once()
    
    @pytest.mark.unit
    @patch('brightness_sorcerer.core.brightness_analyzer.detect_brightness_threshold')
    @patch('brightness_sorcerer.core.brightness_analyzer.find_analysis_range')
    def test_auto_detect_with_background_roi(self, mock_find_range, mock_detect_threshold,
                                           brightness_analyzer, sample_rois, sample_video_frame):
        """Test auto-detection with background ROI."""
        # Setup mocks
        brightness_analyzer.video_processor.is_loaded.return_value = True
        brightness_analyzer.video_processor.total_frames = 100
        brightness_analyzer.video_processor.get_frame_at_index.return_value = sample_video_frame
        brightness_analyzer.video_processor.get_current_frame.return_value = sample_video_frame
        
        mock_detect_threshold.return_value = 20.0
        mock_find_range.return_value = (15, 85)
        
        # Use background ROI
        background_roi = sample_rois[2]  # Third ROI is marked as background
        
        result = brightness_analyzer.auto_detect_frame_range(sample_rois[:2], background_roi)
        
        assert result == (15, 85)
        mock_detect_threshold.assert_called_once()
    
    @pytest.mark.unit
    def test_auto_detect_cancellation(self, brightness_analyzer, sample_rois):
        """Test canceling auto-detection."""
        brightness_analyzer.video_processor.is_loaded.return_value = True
        brightness_analyzer.video_processor.total_frames = 100
        
        # Cancel immediately
        brightness_analyzer.cancel_requested = True
        
        result = brightness_analyzer.auto_detect_frame_range(sample_rois)
        assert result is None
    
    @pytest.mark.unit
    def test_analyze_brightness_no_video(self, brightness_analyzer, sample_rois, temp_dir):
        """Test analysis with no video loaded."""
        brightness_analyzer.video_processor.is_loaded.return_value = False
        
        result = brightness_analyzer.analyze_brightness(sample_rois, 0, 10, temp_dir)
        assert result is None
    
    @pytest.mark.unit
    def test_analyze_brightness_no_rois(self, brightness_analyzer, temp_dir):
        """Test analysis with no ROIs."""
        result = brightness_analyzer.analyze_brightness([], 0, 10, temp_dir)
        assert result is None
    
    @pytest.mark.unit
    @patch('brightness_sorcerer.core.brightness_analyzer.calculate_brightness_stats')
    @patch('brightness_sorcerer.core.brightness_analyzer.detect_brightness_threshold')
    def test_analyze_brightness_success(self, mock_detect_threshold, mock_calc_stats,
                                      brightness_analyzer, sample_rois, sample_video_frame, temp_dir):
        """Test successful brightness analysis."""
        # Setup mocks
        brightness_analyzer.video_processor.is_loaded.return_value = True
        brightness_analyzer.video_processor.get_video_info.return_value = {
            'path': '/test/video.mp4',
            'total_frames': 100,
            'fps': 30.0,
            'duration_seconds': 100/30.0,
            'frame_size': (640, 480),
            'current_frame': 0,
            'cache_size': 10
        }
        brightness_analyzer.video_processor.get_frame_at_index.return_value = sample_video_frame
        
        mock_detect_threshold.return_value = 15.0
        mock_calc_stats.return_value = {
            'mean': 50.0,
            'median': 48.0,
            'std': 12.0,
            'valid_pixels': 1000,
            'min': 10.0,
            'max': 90.0,
            'percentile_25': 40.0,
            'percentile_75': 60.0
        }
        
        # Set progress callback
        progress_calls = []
        def progress_callback(current, total, message):
            progress_calls.append((current, total, message))
        brightness_analyzer.set_progress_callback(progress_callback)
        
        # Run analysis
        result = brightness_analyzer.analyze_brightness(sample_rois[:2], 0, 5, temp_dir)
        
        assert result is not None
        assert isinstance(result, AnalysisResult)
        assert result.video_path == '/test/video.mp4'
        assert result.start_frame == 0
        assert result.end_frame == 5
        assert result.total_frames == 6  # 5 - 0 + 1
        assert result.fps == 30.0
        assert len(result.roi_labels) == 2
        assert result.noise_floor == brightness_analyzer.noise_floor
        assert result.brightness_threshold == 15.0
        
        # Check that frames were analyzed
        assert len(result.frame_analyses) == 6
        
        # Check progress was reported
        assert len(progress_calls) > 0
        
        # Verify current result is set
        assert brightness_analyzer.current_result is result
    
    @pytest.mark.unit
    def test_analyze_brightness_cancellation(self, brightness_analyzer, sample_rois, temp_dir):
        """Test canceling brightness analysis."""
        brightness_analyzer.video_processor.is_loaded.return_value = True
        brightness_analyzer.video_processor.get_video_info.return_value = {
            'path': '/test/video.mp4',
            'total_frames': 100,
            'fps': 30.0
        }
        
        # Cancel immediately
        brightness_analyzer.cancel_requested = True
        
        result = brightness_analyzer.analyze_brightness(sample_rois, 0, 10, temp_dir)
        assert result is None
    
    @pytest.mark.unit
    @patch('brightness_sorcerer.core.brightness_analyzer.plt')
    def test_generate_plots(self, mock_plt, brightness_analyzer, temp_dir):
        """Test plot generation."""
        # Create a mock analysis result
        result = Mock(spec=AnalysisResult)
        result.video_path = '/test/video.mp4'
        result.analysis_timestamp.strftime.return_value = '20240101_120000'
        result.roi_labels = ['ROI 1', 'ROI 2']
        result.get_timestamps.return_value = [0.0, 1.0, 2.0, 3.0, 4.0]
        result.get_roi_timeseries.side_effect = lambda label, stat: [50.0, 52.0, 48.0, 55.0, 51.0]
        result.brightness_threshold = 15.0
        result.start_frame = 0
        result.end_frame = 4
        result.get_analysis_duration.return_value = 4.0
        result.noise_floor = 10.0
        
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.cm.get_cmap.return_value = Mock()
        
        # Call the method
        brightness_analyzer._generate_plots(result, temp_dir)
        
        # Verify plot creation calls
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
    
    @pytest.mark.unit
    def test_save_analysis_results(self, brightness_analyzer, temp_dir):
        """Test saving analysis results."""
        # Create a mock analysis result
        result = Mock(spec=AnalysisResult)
        result.video_path = '/test/video.mp4'
        result.analysis_timestamp.strftime.return_value = '20240101_120000'
        result.save_csv.return_value = True
        result.save_json.return_value = True
        
        # Call the method
        brightness_analyzer._save_analysis_results(result, temp_dir)
        
        # Verify save calls
        result.save_csv.assert_called_once()
        result.save_json.assert_called_once()
        
        # Check that file paths were constructed correctly
        csv_call_args = result.save_csv.call_args[0][0]
        json_call_args = result.save_json.call_args[0][0]
        
        assert 'video_analysis_20240101_120000.csv' in csv_call_args
        assert 'video_analysis_20240101_120000_metadata.json' in json_call_args
    
    @pytest.mark.unit
    def test_get_analysis_summary_none_result(self, brightness_analyzer):
        """Test getting analysis summary with None result."""
        summary = brightness_analyzer.get_analysis_summary(None)
        assert summary == {}
    
    @pytest.mark.unit
    def test_get_analysis_summary_valid_result(self, brightness_analyzer):
        """Test getting analysis summary with valid result."""
        # Create a mock analysis result
        result = Mock(spec=AnalysisResult)
        result.video_path = '/test/video.mp4'
        result.get_analysis_duration.return_value = 10.5
        result.start_frame = 100
        result.end_frame = 200
        result.total_frames = 101
        result.noise_floor = 10.0
        result.brightness_threshold = 15.0
        result.roi_labels = ['ROI 1', 'ROI 2']
        result.summary_stats = {
            'ROI 1': {
                'mean_brightness': {
                    'overall_mean': 50.0,
                    'range': 20.0
                }
            },
            'ROI 2': {
                'mean_brightness': {
                    'overall_mean': 45.0,
                    'range': 15.0
                }
            }
        }
        result.get_peak_frames.return_value = [110, 120, 130, 140, 150]
        
        summary = brightness_analyzer.get_analysis_summary(result)
        
        assert 'video_info' in summary
        assert 'analysis_parameters' in summary
        assert 'roi_summaries' in summary
        
        # Check video info
        video_info = summary['video_info']
        assert video_info['filename'] == 'video.mp4'
        assert video_info['duration_analyzed'] == '10.5s'
        assert video_info['frame_range'] == '100 - 200'
        assert video_info['total_frames'] == 101
        
        # Check analysis parameters
        params = summary['analysis_parameters']
        assert params['noise_floor'] == 10.0
        assert params['brightness_threshold'] == 15.0
        assert params['roi_count'] == 2
        
        # Check ROI summaries
        roi_summaries = summary['roi_summaries']
        assert 'ROI 1' in roi_summaries
        assert 'ROI 2' in roi_summaries
        
        roi1_summary = roi_summaries['ROI 1']
        assert roi1_summary['average_brightness'] == '50.0'
        assert roi1_summary['brightness_range'] == '20.0'
        assert roi1_summary['peak_frames_count'] == 5
        assert roi1_summary['peak_percentage'] == '5.0%'  # 5/101 * 100


@pytest.mark.integration
class TestBrightnessAnalyzerIntegration:
    """Integration tests for BrightnessAnalyzer."""
    
    @pytest.mark.requires_video
    @pytest.mark.slow
    def test_full_analysis_workflow(self, video_processor, sample_video_file, 
                                  sample_rois, temp_dir):
        """Test complete analysis workflow with real video."""
        # Load video
        assert video_processor.load_video(sample_video_file)
        
        # Create analyzer
        analyzer = BrightnessAnalyzer(video_processor)
        
        # Set up progress tracking
        progress_reports = []
        def track_progress(current, total, message):
            progress_reports.append((current, total, message))
        analyzer.set_progress_callback(track_progress)
        
        # Run auto-detection first
        detection_result = analyzer.auto_detect_frame_range(sample_rois[:2])
        
        if detection_result:
            start_frame, end_frame = detection_result
            # Limit range for faster testing
            end_frame = min(start_frame + 10, end_frame)
        else:
            start_frame, end_frame = 0, 10
        
        # Run analysis
        result = analyzer.analyze_brightness(sample_rois[:2], start_frame, end_frame, temp_dir)
        
        # Verify result
        assert result is not None
        assert isinstance(result, AnalysisResult)
        assert result.total_frames == end_frame - start_frame + 1
        assert len(result.frame_analyses) == result.total_frames
        assert len(result.roi_labels) == 2
        
        # Verify progress was reported
        assert len(progress_reports) > 0
        
        # Verify files were created
        files_created = os.listdir(temp_dir)
        csv_files = [f for f in files_created if f.endswith('.csv')]
        json_files = [f for f in files_created if f.endswith('.json')]
        png_files = [f for f in files_created if f.endswith('.png')]
        
        assert len(csv_files) >= 1
        assert len(json_files) >= 1
        assert len(png_files) >= 1
        
        # Verify analysis summary
        summary = analyzer.get_analysis_summary(result)
        assert summary
        assert 'video_info' in summary
        assert 'roi_summaries' in summary
    
    @pytest.mark.unit
    def test_analysis_with_mock_data(self, mock_video_processor, sample_rois, temp_dir):
        """Test analysis with controlled mock data."""
        # Setup mock video processor
        mock_video_processor.get_video_info.return_value = {
            'path': '/mock/video.mp4',
            'total_frames': 50,
            'fps': 25.0,
            'duration_seconds': 2.0,
            'frame_size': (320, 240),
            'current_frame': 0,
            'cache_size': 5
        }
        
        # Create test frame with known brightness pattern
        test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        test_frame[50:100, 50:100] = [100, 100, 100]  # Gray square for ROI 1
        test_frame[150:200, 150:200] = [150, 150, 150]  # Brighter square for ROI 2
        
        mock_video_processor.get_frame_at_index.return_value = test_frame
        
        # Create analyzer
        analyzer = BrightnessAnalyzer(mock_video_processor)
        analyzer.manual_threshold = 10.0
        analyzer.noise_floor = 5.0
        
        # Run analysis on small range
        result = analyzer.analyze_brightness(sample_rois[:2], 0, 4, temp_dir)
        
        # Verify result structure
        assert result is not None
        assert result.total_frames == 5
        assert len(result.frame_analyses) == 5
        assert len(result.roi_labels) == 2
        assert result.brightness_threshold == analyzer.manual_threshold
        
        # Verify that each frame has data for both ROIs
        for frame_analysis in result.frame_analyses:
            assert len(frame_analysis.roi_stats) == 2
            assert sample_rois[0].label in frame_analysis.roi_stats
            assert sample_rois[1].label in frame_analysis.roi_stats
    
    @pytest.mark.unit
    def test_error_handling(self, brightness_analyzer, sample_rois, temp_dir):
        """Test error handling in analysis."""
        # Setup to cause errors
        brightness_analyzer.video_processor.is_loaded.return_value = True
        brightness_analyzer.video_processor.get_video_info.return_value = {
            'path': '/nonexistent/video.mp4',
            'total_frames': 10,
            'fps': 30.0
        }
        brightness_analyzer.video_processor.get_frame_at_index.return_value = None  # Simulate error
        
        # Analysis should handle errors gracefully
        result = brightness_analyzer.analyze_brightness(sample_rois, 0, 5, temp_dir)
        
        # Should return None or handle gracefully without crashing
        # The exact behavior depends on error handling implementation
        # This test ensures no unhandled exceptions are raised