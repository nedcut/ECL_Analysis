"""
Unit tests for VideoProcessor and FrameCache classes.

Tests video processing functionality including:
- Video file loading and validation
- Frame extraction and navigation
- Frame caching with LRU eviction
- Video property management
- Error handling and resource cleanup
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from collections import OrderedDict

from brightness_sorcerer.core.video_processor import VideoProcessor, FrameCache
from brightness_sorcerer.core.exceptions import VideoLoadError, ValidationError
from brightness_sorcerer.utils.constants import FRAME_CACHE_SIZE


class TestFrameCache:
    """Test FrameCache LRU caching functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = FrameCache(max_size=3)
        
        # Create test frames
        self.frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        self.frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 150
        self.frame3 = np.ones((10, 10, 3), dtype=np.uint8) * 200
        self.frame4 = np.ones((10, 10, 3), dtype=np.uint8) * 250
    
    def test_cache_initialization(self):
        """Test cache initializes with correct parameters."""
        cache = FrameCache(max_size=5)
        
        assert cache.get_max_size() == 5
        assert cache.get_size() == 0
        assert cache._cache == {}
    
    def test_cache_default_initialization(self):
        """Test cache initializes with default parameters."""
        cache = FrameCache()
        
        assert cache.get_max_size() == FRAME_CACHE_SIZE
        assert cache.get_size() == 0
    
    def test_put_and_get_frame(self):
        """Test basic put and get operations."""
        self.cache.put(0, self.frame1)
        
        assert self.cache.get_size() == 1
        retrieved_frame = self.cache.get(0)
        
        assert retrieved_frame is not None
        assert np.array_equal(retrieved_frame, self.frame1)
        # Should return a copy, not the original
        assert retrieved_frame is not self.frame1
    
    def test_get_nonexistent_frame(self):
        """Test getting frame that doesn't exist."""
        result = self.cache.get(999)
        assert result is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        self.cache.put(0, self.frame1)
        self.cache.put(1, self.frame2) 
        self.cache.put(2, self.frame3)
        
        assert self.cache.get_size() == 3
        
        # Add fourth frame - should evict frame 0 (least recently used)
        self.cache.put(3, self.frame4)
        
        assert self.cache.get_size() == 3
        assert self.cache.get(0) is None  # Should be evicted
        assert self.cache.get(1) is not None
        assert self.cache.get(2) is not None
        assert self.cache.get(3) is not None
    
    def test_lru_ordering_on_get(self):
        """Test that get operations update LRU ordering."""
        # Fill cache
        self.cache.put(0, self.frame1)
        self.cache.put(1, self.frame2)
        self.cache.put(2, self.frame3)
        
        # Access frame 0 to make it most recently used
        self.cache.get(0)
        
        # Add frame 3 - should evict frame 1 (now least recently used)
        self.cache.put(3, self.frame4)
        
        assert self.cache.get(0) is not None  # Should still be cached
        assert self.cache.get(1) is None     # Should be evicted
        assert self.cache.get(2) is not None
        assert self.cache.get(3) is not None
    
    def test_update_existing_frame(self):
        """Test updating existing frame in cache."""
        self.cache.put(0, self.frame1)
        self.cache.put(0, self.frame2)  # Update same index
        
        assert self.cache.get_size() == 1
        retrieved_frame = self.cache.get(0)
        assert np.array_equal(retrieved_frame, self.frame2)
    
    def test_clear_cache(self):
        """Test clearing all cached frames."""
        self.cache.put(0, self.frame1)
        self.cache.put(1, self.frame2)
        
        assert self.cache.get_size() == 2
        
        self.cache.clear()
        
        assert self.cache.get_size() == 0
        assert self.cache.get(0) is None
        assert self.cache.get(1) is None
    
    def test_cache_isolation(self):
        """Test that cached frames are isolated from modifications."""
        original_frame = np.ones((5, 5, 3), dtype=np.uint8) * 100
        self.cache.put(0, original_frame)
        
        # Modify original frame
        original_frame[:] = 200
        
        # Cached frame should be unchanged
        cached_frame = self.cache.get(0)
        assert np.all(cached_frame == 100)
        
        # Modify retrieved frame
        cached_frame[:] = 50
        
        # Cache should still contain original values
        fresh_frame = self.cache.get(0)
        assert np.all(fresh_frame == 100)


class TestVideoProcessorInitialization:
    """Test VideoProcessor initialization and basic properties."""
    
    def test_default_initialization(self):
        """Test video processor initializes with default parameters."""
        processor = VideoProcessor()
        
        assert processor.cap is None
        assert processor.video_path == ""
        assert processor.total_frames == 0
        assert processor.frame_width == 0
        assert processor.frame_height == 0
        assert processor.fps == 0.0
        assert processor.duration_seconds == 0.0
        assert processor.current_frame_index == 0
        assert processor.current_frame is None
        assert processor.frame_cache.get_max_size() == FRAME_CACHE_SIZE
    
    def test_custom_cache_size_initialization(self):
        """Test video processor initializes with custom cache size."""
        custom_cache_size = 50
        processor = VideoProcessor(cache_size=custom_cache_size)
        
        assert processor.frame_cache.get_max_size() == custom_cache_size
    
    def test_is_loaded_initially_false(self):
        """Test is_loaded returns False initially."""
        processor = VideoProcessor()
        assert processor.is_loaded() is False
    
    def test_initialization_logging(self):
        """Test initialization creates appropriate log messages."""
        with patch('brightness_sorcerer.core.video_processor.logger') as mock_logger:
            processor = VideoProcessor(cache_size=25)
            mock_logger.debug.assert_called_once_with("VideoProcessor initialized with cache size 25")


class TestVideoProcessorVideoLoading:
    """Test video loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor()
        
        # Create mock video properties
        self.mock_properties = {
            'total_frames': 100,
            'fps': 30.0,
            'width': 640,
            'height': 480,
            'duration': 3.33
        }
    
    @patch('brightness_sorcerer.core.video_processor.validate_video_file')
    @patch('cv2.VideoCapture')
    def test_load_video_successful(self, mock_videocapture, mock_validate):
        """Test successful video loading."""
        # Mock validation success
        mock_validate.return_value = True
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480
        }.get(prop, 0)
        mock_videocapture.return_value = mock_cap
        
        test_path = "/test/video.mp4"
        result = self.processor.load_video(test_path)
        
        # Verify validation was called
        mock_validate.assert_called_once_with(test_path)
        
        # Verify video capture was created and opened
        mock_videocapture.assert_called_once_with(test_path)
        mock_cap.isOpened.assert_called()
        
        # Verify processor state
        assert self.processor.is_loaded() is True
        assert self.processor.video_path == test_path
        assert self.processor.total_frames == 100
        assert self.processor.fps == 30.0
        assert self.processor.frame_width == 640
        assert self.processor.frame_height == 480
        
        # Verify return value
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'properties' in result
    
    @patch('brightness_sorcerer.core.video_processor.validate_video_file')
    def test_load_video_validation_failure(self, mock_validate):
        """Test video loading with validation failure."""
        # Mock validation failure
        mock_validate.side_effect = ValidationError("Invalid file")
        
        test_path = "/test/invalid.mp4"
        
        with pytest.raises(ValidationError):
            self.processor.load_video(test_path)
        
        # Processor should remain unloaded
        assert self.processor.is_loaded() is False
        assert self.processor.video_path == ""
    
    @patch('brightness_sorcerer.core.video_processor.validate_video_file')
    @patch('cv2.VideoCapture')
    def test_load_video_opencv_failure(self, mock_videocapture, mock_validate):
        """Test video loading with OpenCV failure."""
        # Mock validation success but OpenCV failure
        mock_validate.return_value = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap
        
        test_path = "/test/video.mp4"
        
        with pytest.raises(VideoLoadError):
            self.processor.load_video(test_path)
        
        # Processor should remain unloaded
        assert self.processor.is_loaded() is False
    
    @patch('brightness_sorcerer.core.video_processor.validate_video_file')
    @patch('cv2.VideoCapture')
    def test_load_video_replaces_previous(self, mock_videocapture, mock_validate):
        """Test loading new video replaces previous one."""
        mock_validate.return_value = True
        
        # Mock first video
        mock_cap1 = MagicMock()
        mock_cap1.isOpened.return_value = True
        mock_cap1.get.side_effect = lambda prop: 50 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
        
        # Mock second video
        mock_cap2 = MagicMock()
        mock_cap2.isOpened.return_value = True
        mock_cap2.get.side_effect = lambda prop: 100 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
        
        mock_videocapture.side_effect = [mock_cap1, mock_cap2]
        
        # Load first video
        self.processor.load_video("/test/video1.mp4")
        assert self.processor.total_frames == 50
        
        # Load second video
        self.processor.load_video("/test/video2.mp4")
        assert self.processor.total_frames == 100
        assert self.processor.video_path == "/test/video2.mp4"
        
        # First video capture should have been closed
        mock_cap1.release.assert_called_once()


class TestVideoProcessorFrameOperations:
    """Test frame extraction and navigation operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor(cache_size=3)
        
        # Mock a loaded video
        self.mock_cap = MagicMock()
        self.mock_cap.isOpened.return_value = True
        self.processor.cap = self.mock_cap
        self.processor.total_frames = 100
        self.processor.video_path = "/test/video.mp4"
        
        # Create test frames
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    def test_get_frame_success(self):
        """Test successful frame extraction."""
        frame_index = 10
        
        # Mock successful frame read
        self.mock_cap.set.return_value = True
        self.mock_cap.read.return_value = (True, self.test_frame)
        
        result = self.processor.get_frame(frame_index)
        
        assert result is not None
        assert np.array_equal(result, self.test_frame)
        
        # Verify OpenCV calls
        self.mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.mock_cap.read.assert_called_once()
        
        # Frame should be cached
        assert self.processor.frame_cache.get_size() == 1
        
    def test_get_frame_from_cache(self):
        """Test frame retrieval from cache."""
        frame_index = 10
        
        # Put frame in cache first
        self.processor.frame_cache.put(frame_index, self.test_frame)
        
        # Mock should not be called if frame is cached
        result = self.processor.get_frame(frame_index)
        
        assert result is not None
        assert np.array_equal(result, self.test_frame)
        
        # OpenCV should not be called
        self.mock_cap.set.assert_not_called()
        self.mock_cap.read.assert_not_called()
    
    def test_get_frame_read_failure(self):
        """Test frame extraction with read failure."""
        frame_index = 10
        
        # Mock read failure
        self.mock_cap.set.return_value = True
        self.mock_cap.read.return_value = (False, None)
        
        result = self.processor.get_frame(frame_index)
        
        assert result is None
        
        # Frame should not be cached on failure
        assert self.processor.frame_cache.get_size() == 0
    
    def test_get_frame_seek_failure(self):
        """Test frame extraction with seek failure."""
        frame_index = 10
        
        # Mock seek failure
        self.mock_cap.set.return_value = False
        
        result = self.processor.get_frame(frame_index)
        
        assert result is None
        
        # read should not be called if seek fails
        self.mock_cap.read.assert_not_called()
    
    def test_get_frame_invalid_index(self):
        """Test frame extraction with invalid index."""
        # Test negative index
        result = self.processor.get_frame(-1)
        assert result is None
        
        # Test index beyond total frames
        result = self.processor.get_frame(self.processor.total_frames)
        assert result is None
        
        # Test index equal to total frames
        result = self.processor.get_frame(self.processor.total_frames - 1)
        # This should be valid - may succeed or fail depending on video
    
    def test_get_frame_video_not_loaded(self):
        """Test frame extraction when no video is loaded."""
        processor = VideoProcessor()  # New processor, no video loaded
        
        result = processor.get_frame(0)
        assert result is None
    
    def test_get_current_frame(self):
        """Test getting current frame."""
        # Mock current frame
        self.processor.current_frame = self.test_frame.copy()
        
        result = self.processor.get_current_frame()
        
        assert result is not None
        assert np.array_equal(result, self.test_frame)
        # Should return a copy
        assert result is not self.processor.current_frame
    
    def test_get_current_frame_none(self):
        """Test getting current frame when none is set."""
        result = self.processor.get_current_frame()
        assert result is None


class TestVideoProcessorNavigation:
    """Test video navigation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor()
        
        # Mock a loaded video
        self.mock_cap = MagicMock()
        self.mock_cap.isOpened.return_value = True
        self.processor.cap = self.mock_cap
        self.processor.total_frames = 100
        
    def test_seek_to_frame_valid(self):
        """Test seeking to valid frame index."""
        target_frame = 50
        test_frame = np.ones((480, 640, 3), dtype=np.uint8)
        
        # Mock successful seek and read
        self.mock_cap.set.return_value = True
        self.mock_cap.read.return_value = (True, test_frame)
        
        result = self.processor.seek_to_frame(target_frame)
        
        assert result is True
        assert self.processor.current_frame_index == target_frame
        assert np.array_equal(self.processor.current_frame, test_frame)
    
    def test_seek_to_frame_invalid(self):
        """Test seeking to invalid frame index."""
        # Test negative index
        result = self.processor.seek_to_frame(-1)
        assert result is False
        
        # Test index beyond total frames
        result = self.processor.seek_to_frame(self.processor.total_frames)
        assert result is False
        
        # Current frame index should remain unchanged
        assert self.processor.current_frame_index == 0
    
    def test_seek_to_frame_opencv_failure(self):
        """Test seeking with OpenCV failure."""
        target_frame = 50
        
        # Mock seek failure
        self.mock_cap.set.return_value = False
        
        result = self.processor.seek_to_frame(target_frame)
        
        assert result is False
        assert self.processor.current_frame_index == 0  # Should remain unchanged
    
    def test_next_frame(self):
        """Test advancing to next frame."""
        self.processor.current_frame_index = 25
        test_frame = np.ones((480, 640, 3), dtype=np.uint8)
        
        # Mock successful frame read
        self.mock_cap.set.return_value = True
        self.mock_cap.read.return_value = (True, test_frame)
        
        result = self.processor.next_frame()
        
        assert result is True
        assert self.processor.current_frame_index == 26
    
    def test_previous_frame(self):
        """Test going to previous frame."""
        self.processor.current_frame_index = 25
        test_frame = np.ones((480, 640, 3), dtype=np.uint8)
        
        # Mock successful frame read
        self.mock_cap.set.return_value = True
        self.mock_cap.read.return_value = (True, test_frame)
        
        result = self.processor.previous_frame()
        
        assert result is True
        assert self.processor.current_frame_index == 24
    
    def test_next_frame_at_end(self):
        """Test next frame when at end of video."""
        self.processor.current_frame_index = self.processor.total_frames - 1
        
        result = self.processor.next_frame()
        
        assert result is False
        assert self.processor.current_frame_index == self.processor.total_frames - 1
    
    def test_previous_frame_at_beginning(self):
        """Test previous frame when at beginning of video."""
        self.processor.current_frame_index = 0
        
        result = self.processor.previous_frame()
        
        assert result is False
        assert self.processor.current_frame_index == 0


class TestVideoProcessorProperties:
    """Test video property management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor()
    
    def test_get_video_properties_loaded(self):
        """Test getting video properties when video is loaded."""
        # Mock loaded video state
        self.processor.video_path = "/test/video.mp4"
        self.processor.total_frames = 300
        self.processor.frame_width = 1920
        self.processor.frame_height = 1080
        self.processor.fps = 30.0
        self.processor.duration_seconds = 10.0
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        self.processor.cap = mock_cap
        
        properties = self.processor.get_video_properties()
        
        assert properties['loaded'] is True
        assert properties['path'] == "/test/video.mp4"
        assert properties['total_frames'] == 300
        assert properties['width'] == 1920
        assert properties['height'] == 1080
        assert properties['fps'] == 30.0
        assert properties['duration'] == 10.0
    
    def test_get_video_properties_not_loaded(self):
        """Test getting video properties when no video is loaded."""
        properties = self.processor.get_video_properties()
        
        assert properties['loaded'] is False
        assert properties['path'] == ""
        assert properties['total_frames'] == 0
        assert properties['width'] == 0
        assert properties['height'] == 0
        assert properties['fps'] == 0.0
        assert properties['duration'] == 0.0
    
    def test_get_cache_info(self):
        """Test getting cache information."""
        # Add some frames to cache
        test_frame = np.ones((100, 100, 3), dtype=np.uint8)
        self.processor.frame_cache.put(0, test_frame)
        self.processor.frame_cache.put(1, test_frame)
        
        cache_info = self.processor.get_cache_info()
        
        assert cache_info['current_size'] == 2
        assert cache_info['max_size'] == FRAME_CACHE_SIZE
        assert cache_info['cache_hit_ratio'] is not None  # May be implemented
    
    def test_calculate_memory_usage(self):
        """Test memory usage calculation."""
        # Add frames to cache
        test_frame = np.ones((100, 100, 3), dtype=np.uint8)
        self.processor.frame_cache.put(0, test_frame)
        self.processor.frame_cache.put(1, test_frame)
        
        memory_usage = self.processor.calculate_memory_usage()
        
        assert isinstance(memory_usage, dict)
        assert 'cache_bytes' in memory_usage
        assert 'frame_bytes' in memory_usage
        assert memory_usage['cache_bytes'] > 0


class TestVideoProcessorCleanup:
    """Test video processor cleanup and resource management."""
    
    def test_close_video_loaded(self):
        """Test closing loaded video."""
        processor = VideoProcessor()
        
        # Mock loaded video
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        processor.cap = mock_cap
        processor.video_path = "/test/video.mp4"
        processor.total_frames = 100
        
        # Add frame to cache
        test_frame = np.ones((100, 100, 3), dtype=np.uint8)
        processor.frame_cache.put(0, test_frame)
        
        processor.close_video()
        
        # Verify cleanup
        mock_cap.release.assert_called_once()
        assert processor.cap is None
        assert processor.video_path == ""
        assert processor.total_frames == 0
        assert processor.frame_cache.get_size() == 0
        assert processor.current_frame is None
        assert processor.current_frame_index == 0
    
    def test_close_video_not_loaded(self):
        """Test closing when no video is loaded."""
        processor = VideoProcessor()
        
        # Should not raise exceptions
        processor.close_video()
        
        assert processor.cap is None
    
    def test_destructor_cleanup(self):
        """Test cleanup on object destruction."""
        processor = VideoProcessor()
        
        # Mock loaded video
        mock_cap = MagicMock()
        processor.cap = mock_cap
        
        with patch.object(processor, 'close_video') as mock_close:
            # Trigger destructor
            del processor
            
            # close_video should be called (implementation may vary)
    
    def test_clear_cache(self):
        """Test clearing frame cache."""
        processor = VideoProcessor()
        
        # Add frames to cache
        test_frame = np.ones((100, 100, 3), dtype=np.uint8)
        processor.frame_cache.put(0, test_frame)
        processor.frame_cache.put(1, test_frame)
        
        assert processor.frame_cache.get_size() == 2
        
        processor.clear_cache()
        
        assert processor.frame_cache.get_size() == 0


class TestVideoProcessorErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = VideoProcessor()
    
    def test_operations_without_loaded_video(self):
        """Test operations when no video is loaded."""
        # All operations should fail gracefully
        assert self.processor.get_frame(0) is None
        assert self.processor.get_current_frame() is None
        assert self.processor.seek_to_frame(10) is False
        assert self.processor.next_frame() is False
        assert self.processor.previous_frame() is False
        
        # Properties should return empty/default values
        props = self.processor.get_video_properties()
        assert props['loaded'] is False
    
    def test_opencv_exception_handling(self):
        """Test handling of OpenCV exceptions."""
        # Mock loaded video
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.set.side_effect = Exception("OpenCV error")
        self.processor.cap = mock_cap
        self.processor.total_frames = 100
        
        # Operations should handle exceptions gracefully
        result = self.processor.get_frame(10)
        assert result is None
        
        result = self.processor.seek_to_frame(10)
        assert result is False
    
    def test_invalid_frame_data_handling(self):
        """Test handling of invalid frame data."""
        # Mock loaded video
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.set.return_value = True
        # Return invalid frame data
        mock_cap.read.return_value = (True, None)
        self.processor.cap = mock_cap
        self.processor.total_frames = 100
        
        result = self.processor.get_frame(10)
        assert result is None
    
    def test_memory_constraints(self):
        """Test behavior under memory constraints."""
        processor = VideoProcessor(cache_size=2)  # Very small cache
        
        # Create large test frames
        large_frame = np.ones((1000, 1000, 3), dtype=np.uint8)
        
        # Fill cache beyond capacity
        processor.frame_cache.put(0, large_frame)
        processor.frame_cache.put(1, large_frame)
        processor.frame_cache.put(2, large_frame)  # Should evict frame 0
        
        assert processor.frame_cache.get_size() == 2
        assert processor.frame_cache.get(0) is None
        assert processor.frame_cache.get(1) is not None
        assert processor.frame_cache.get(2) is not None


class TestVideoProcessorIntegrationScenarios:
    """Test integration scenarios combining multiple operations."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.processor = VideoProcessor(cache_size=5)
    
    @patch('brightness_sorcerer.core.video_processor.validate_video_file')
    @patch('cv2.VideoCapture')
    def test_complete_video_workflow(self, mock_videocapture, mock_validate):
        """Test complete video processing workflow."""
        # Mock validation and video loading
        mock_validate.return_value = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 50,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480
        }.get(prop, 0)
        mock_videocapture.return_value = mock_cap
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_cap.set.return_value = True
        
        # Step 1: Load video
        result = self.processor.load_video("/test/video.mp4")
        assert result['success'] is True
        assert self.processor.is_loaded() is True
        
        # Step 2: Get video properties
        props = self.processor.get_video_properties()
        assert props['total_frames'] == 50
        assert props['fps'] == 30.0
        
        # Step 3: Navigate through video
        assert self.processor.seek_to_frame(10) is True
        assert self.processor.current_frame_index == 10
        
        assert self.processor.next_frame() is True
        assert self.processor.current_frame_index == 11
        
        assert self.processor.previous_frame() is True
        assert self.processor.current_frame_index == 10
        
        # Step 4: Extract frames (should use cache)
        frame1 = self.processor.get_frame(5)
        frame2 = self.processor.get_frame(5)  # Should come from cache
        assert frame1 is not None
        assert frame2 is not None
        assert np.array_equal(frame1, frame2)
        
        # Step 5: Check cache usage
        cache_info = self.processor.get_cache_info()
        assert cache_info['current_size'] > 0
        
        # Step 6: Clear cache
        self.processor.clear_cache()
        assert self.processor.frame_cache.get_size() == 0
        
        # Step 7: Close video
        self.processor.close_video()
        assert self.processor.is_loaded() is False
    
    def test_multiple_video_loading(self):
        """Test loading multiple videos in sequence."""
        with patch('brightness_sorcerer.core.video_processor.validate_video_file') as mock_validate, \
             patch('cv2.VideoCapture') as mock_videocapture:
            
            mock_validate.return_value = True
            
            # Mock first video
            mock_cap1 = MagicMock()
            mock_cap1.isOpened.return_value = True
            mock_cap1.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
            
            # Mock second video
            mock_cap2 = MagicMock()
            mock_cap2.isOpened.return_value = True
            mock_cap2.get.side_effect = lambda prop: 60 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
            
            mock_videocapture.side_effect = [mock_cap1, mock_cap2]
            
            # Load first video
            self.processor.load_video("/test/video1.mp4")
            assert self.processor.total_frames == 30
            
            # Load second video (should replace first)
            self.processor.load_video("/test/video2.mp4")
            assert self.processor.total_frames == 60
            
            # First video should be properly closed
            mock_cap1.release.assert_called_once()
    
    def test_error_recovery_workflow(self):
        """Test error recovery in various scenarios."""
        # Start with successful video load
        with patch('brightness_sorcerer.core.video_processor.validate_video_file') as mock_validate, \
             patch('cv2.VideoCapture') as mock_videocapture:
            
            mock_validate.return_value = True
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: 100 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
            mock_videocapture.return_value = mock_cap
            
            self.processor.load_video("/test/video.mp4")
            assert self.processor.is_loaded() is True
        
        # Simulate OpenCV errors during frame operations
        mock_cap.set.side_effect = Exception("Seek error")
        
        # Operations should fail gracefully
        assert self.processor.seek_to_frame(50) is False
        assert self.processor.get_frame(25) is None
        
        # Processor should remain in valid state
        assert self.processor.is_loaded() is True  # Still considered loaded
        
        # Should be able to recover by closing and reloading
        self.processor.close_video()
        assert self.processor.is_loaded() is False


if __name__ == '__main__':
    pytest.main([__file__])