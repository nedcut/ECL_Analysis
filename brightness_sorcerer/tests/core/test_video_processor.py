"""Tests for VideoProcessor class."""

import pytest
import numpy as np
import os
from unittest.mock import Mock, patch

from ...core.video_processor import VideoProcessor, FrameCache


class TestFrameCache:
    """Tests for FrameCache class."""
    
    @pytest.mark.unit
    def test_init(self):
        """Test FrameCache initialization."""
        cache = FrameCache(max_size=5)
        assert cache.max_size == 5
        assert cache.get_size() == 0
    
    @pytest.mark.unit
    def test_put_and_get(self, sample_video_frame):
        """Test putting and getting frames from cache."""
        cache = FrameCache(max_size=3)
        
        # Put a frame
        cache.put(0, sample_video_frame)
        assert cache.get_size() == 1
        
        # Get the frame
        retrieved_frame = cache.get(0)
        assert retrieved_frame is not None
        np.testing.assert_array_equal(retrieved_frame, sample_video_frame)
        
        # Frame should still be in cache
        assert cache.get_size() == 1
    
    @pytest.mark.unit
    def test_get_nonexistent_frame(self):
        """Test getting a frame that doesn't exist in cache."""
        cache = FrameCache(max_size=3)
        assert cache.get(999) is None
    
    @pytest.mark.unit
    def test_lru_eviction(self, sample_video_frame):
        """Test LRU eviction when cache is full."""
        cache = FrameCache(max_size=2)
        
        # Create different frames
        frame1 = sample_video_frame.copy()
        frame2 = sample_video_frame.copy() + 10
        frame3 = sample_video_frame.copy() + 20
        
        # Fill cache to capacity
        cache.put(0, frame1)
        cache.put(1, frame2)
        assert cache.get_size() == 2
        
        # Add another frame (should evict oldest)
        cache.put(2, frame3)
        assert cache.get_size() == 2
        
        # Frame 0 should be evicted (it was oldest)
        assert cache.get(0) is None
        assert cache.get(1) is not None
        assert cache.get(2) is not None
    
    @pytest.mark.unit
    def test_lru_access_order(self, sample_video_frame):
        """Test that accessing frames updates LRU order."""
        cache = FrameCache(max_size=2)
        
        frame1 = sample_video_frame.copy()
        frame2 = sample_video_frame.copy() + 10
        frame3 = sample_video_frame.copy() + 20
        
        # Fill cache
        cache.put(0, frame1)
        cache.put(1, frame2)
        
        # Access frame 0 (should make it most recently used)
        cache.get(0)
        
        # Add frame 2 (should evict frame 1, not frame 0)
        cache.put(2, frame3)
        
        assert cache.get(0) is not None  # Should still exist
        assert cache.get(1) is None      # Should be evicted
        assert cache.get(2) is not None  # Should exist
    
    @pytest.mark.unit
    def test_clear(self, sample_video_frame):
        """Test clearing the cache."""
        cache = FrameCache(max_size=3)
        
        cache.put(0, sample_video_frame)
        cache.put(1, sample_video_frame)
        assert cache.get_size() == 2
        
        cache.clear()
        assert cache.get_size() == 0
        assert cache.get(0) is None
        assert cache.get(1) is None


class TestVideoProcessor:
    """Tests for VideoProcessor class."""
    
    @pytest.mark.unit
    def test_init(self):
        """Test VideoProcessor initialization."""
        processor = VideoProcessor(cache_size=50)
        
        assert processor.video_path is None
        assert processor.cap is None
        assert processor.total_frames == 0
        assert processor.fps == 0.0
        assert processor.frame_size == (0, 0)
        assert processor.current_frame_index == 0
        assert processor.current_frame is None
        assert processor.frame_cache.max_size == 50
        assert not processor.is_loaded()
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_load_video_success(self, video_processor, sample_video_file):
        """Test successful video loading."""
        assert video_processor.load_video(sample_video_file)
        
        assert video_processor.is_loaded()
        assert video_processor.video_path == sample_video_file
        assert video_processor.total_frames > 0
        assert video_processor.fps > 0
        assert video_processor.frame_size == (640, 480)
        assert video_processor.current_frame is not None
        assert video_processor.current_frame_index == 0
    
    @pytest.mark.unit
    def test_load_video_nonexistent_file(self, video_processor):
        """Test loading a non-existent video file."""
        assert not video_processor.load_video("/nonexistent/video.mp4")
        assert not video_processor.is_loaded()
    
    @pytest.mark.unit
    def test_load_video_invalid_file(self, video_processor, temp_dir):
        """Test loading an invalid video file."""
        # Create a text file with video extension
        invalid_file = os.path.join(temp_dir, "invalid.mp4")
        with open(invalid_file, 'w') as f:
            f.write("This is not a video file")
        
        assert not video_processor.load_video(invalid_file)
        assert not video_processor.is_loaded()
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_seek_to_frame(self, video_processor, sample_video_file):
        """Test seeking to specific frames."""
        video_processor.load_video(sample_video_file)
        
        # Seek to middle frame
        mid_frame = video_processor.total_frames // 2
        assert video_processor.seek_to_frame(mid_frame)
        assert video_processor.current_frame_index == mid_frame
        assert video_processor.current_frame is not None
        
        # Seek to last frame
        last_frame = video_processor.total_frames - 1
        assert video_processor.seek_to_frame(last_frame)
        assert video_processor.current_frame_index == last_frame
        
        # Seek back to first frame
        assert video_processor.seek_to_frame(0)
        assert video_processor.current_frame_index == 0
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_seek_to_invalid_frame(self, video_processor, sample_video_file):
        """Test seeking to invalid frame indices."""
        video_processor.load_video(sample_video_file)
        
        # Negative frame
        assert not video_processor.seek_to_frame(-1)
        
        # Frame beyond video length
        assert not video_processor.seek_to_frame(video_processor.total_frames)
        assert not video_processor.seek_to_frame(9999)
    
    @pytest.mark.unit
    def test_seek_without_loaded_video(self, video_processor):
        """Test seeking when no video is loaded."""
        assert not video_processor.seek_to_frame(0)
        assert not video_processor.seek_to_frame(10)
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_step_frames(self, video_processor, sample_video_file):
        """Test stepping through frames."""
        video_processor.load_video(sample_video_file)
        initial_frame = video_processor.current_frame_index
        
        # Step forward
        assert video_processor.step_frames(1)
        assert video_processor.current_frame_index == initial_frame + 1
        
        # Step forward multiple frames
        assert video_processor.step_frames(5)
        assert video_processor.current_frame_index == initial_frame + 6
        
        # Step backward
        assert video_processor.step_frames(-2)
        assert video_processor.current_frame_index == initial_frame + 4
        
        # Step to boundary (should clamp)
        large_step = video_processor.total_frames + 100
        video_processor.step_frames(large_step)
        assert video_processor.current_frame_index == video_processor.total_frames - 1
        
        # Step backward past beginning (should clamp)
        video_processor.step_frames(-9999)
        assert video_processor.current_frame_index == 0
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_get_current_frame(self, video_processor, sample_video_file):
        """Test getting current frame."""
        video_processor.load_video(sample_video_file)
        
        frame = video_processor.get_current_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)  # Height, Width, Channels
        
        # Returned frame should be a copy
        original_frame = video_processor.current_frame
        returned_frame = video_processor.get_current_frame()
        assert returned_frame is not original_frame
        np.testing.assert_array_equal(returned_frame, original_frame)
    
    @pytest.mark.unit
    def test_get_current_frame_no_video(self, video_processor):
        """Test getting current frame when no video is loaded."""
        frame = video_processor.get_current_frame()
        assert frame is None
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_get_frame_at_index(self, video_processor, sample_video_file):
        """Test getting frame at specific index without changing position."""
        video_processor.load_video(sample_video_file)
        
        original_index = video_processor.current_frame_index
        target_index = min(10, video_processor.total_frames - 1)
        
        # Get frame at different index
        frame = video_processor.get_frame_at_index(target_index)
        
        # Current position should not change
        assert video_processor.current_frame_index == original_index
        assert frame is not None
        assert isinstance(frame, np.ndarray)
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_get_frame_at_invalid_index(self, video_processor, sample_video_file):
        """Test getting frame at invalid index."""
        video_processor.load_video(sample_video_file)
        
        # Invalid indices
        assert video_processor.get_frame_at_index(-1) is None
        assert video_processor.get_frame_at_index(video_processor.total_frames) is None
        assert video_processor.get_frame_at_index(9999) is None
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_frame_caching(self, video_processor, sample_video_file):
        """Test that frames are properly cached."""
        video_processor.load_video(sample_video_file)
        
        # Access frame 5
        video_processor.seek_to_frame(5)
        frame5_first = video_processor.get_current_frame()
        
        # Move to different frame
        video_processor.seek_to_frame(10)
        
        # Access frame 5 again (should come from cache)
        video_processor.seek_to_frame(5)
        frame5_second = video_processor.get_current_frame()
        
        # Should be the same data
        np.testing.assert_array_equal(frame5_first, frame5_second)
        
        # Cache should contain frames
        assert video_processor.frame_cache.get_size() > 0
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_get_video_info(self, video_processor, sample_video_file):
        """Test getting video information."""
        # No video loaded
        info = video_processor.get_video_info()
        assert info == {}
        
        # Load video
        video_processor.load_video(sample_video_file)
        info = video_processor.get_video_info()
        
        assert 'path' in info
        assert 'total_frames' in info
        assert 'fps' in info
        assert 'duration_seconds' in info
        assert 'frame_size' in info
        assert 'current_frame' in info
        assert 'cache_size' in info
        
        assert info['path'] == sample_video_file
        assert info['total_frames'] > 0
        assert info['fps'] > 0
        assert info['frame_size'] == (640, 480)
        assert info['current_frame'] >= 0
        assert info['cache_size'] >= 0
        
        # Duration should be calculated correctly
        expected_duration = info['total_frames'] / info['fps']
        assert abs(info['duration_seconds'] - expected_duration) < 0.1
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_release(self, video_processor, sample_video_file):
        """Test releasing video resources."""
        video_processor.load_video(sample_video_file)
        assert video_processor.is_loaded()
        
        video_processor.release()
        
        assert not video_processor.is_loaded()
        assert video_processor.cap is None
        assert video_processor.current_frame is None
        assert video_processor.current_frame_index == 0
        assert video_processor.total_frames == 0
        assert video_processor.fps == 0.0
        assert video_processor.frame_size == (0, 0)
        assert video_processor.video_path is None
        assert video_processor.frame_cache.get_size() == 0
    
    @pytest.mark.unit
    def test_release_without_video(self, video_processor):
        """Test releasing when no video is loaded."""
        # Should not raise an exception
        video_processor.release()
        assert not video_processor.is_loaded()
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_multiple_video_loads(self, video_processor, sample_video_file):
        """Test loading multiple videos sequentially."""
        # Load first video
        assert video_processor.load_video(sample_video_file)
        first_total_frames = video_processor.total_frames
        
        # Load same video again (should replace)
        assert video_processor.load_video(sample_video_file)
        assert video_processor.total_frames == first_total_frames
        assert video_processor.current_frame_index == 0
        
        # Cache should be cleared
        assert video_processor.frame_cache.get_size() <= 1  # Only current frame
    
    @pytest.mark.unit
    @pytest.mark.requires_video
    def test_destructor_cleanup(self, sample_video_file):
        """Test that destructor properly cleans up resources."""
        processor = VideoProcessor()
        processor.load_video(sample_video_file)
        assert processor.is_loaded()
        
        # Delete processor (should trigger __del__)
        del processor
        
        # Can't easily test __del__ directly, but it should not raise exceptions


@pytest.mark.integration
class TestVideoProcessorIntegration:
    """Integration tests for VideoProcessor with real video files."""
    
    @pytest.mark.requires_video
    def test_full_workflow(self, video_processor, sample_video_file):
        """Test complete video processing workflow."""
        # Load video
        assert video_processor.load_video(sample_video_file)
        
        # Navigate through video
        total_frames = video_processor.total_frames
        middle_frame = total_frames // 2
        
        # Seek to middle
        assert video_processor.seek_to_frame(middle_frame)
        middle_frame_data = video_processor.get_current_frame()
        
        # Step around
        video_processor.step_frames(5)
        video_processor.step_frames(-3)
        
        # Get frame without changing position
        frame_at_0 = video_processor.get_frame_at_index(0)
        assert video_processor.current_frame_index != 0
        
        # Verify frames are different
        current_frame = video_processor.get_current_frame()
        assert not np.array_equal(current_frame, frame_at_0)
        
        # Check cache usage
        assert video_processor.frame_cache.get_size() > 0
        
        # Release and verify cleanup
        video_processor.release()
        assert not video_processor.is_loaded()
    
    @pytest.mark.requires_video
    def test_performance_with_large_cache(self, sample_video_file):
        """Test performance with large cache."""
        processor = VideoProcessor(cache_size=50)
        processor.load_video(sample_video_file)
        
        # Access many frames to fill cache
        total_frames = min(processor.total_frames, 30)
        for i in range(0, total_frames, 2):
            processor.seek_to_frame(i)
            frame = processor.get_current_frame()
            assert frame is not None
        
        # Cache should have multiple frames
        assert processor.frame_cache.get_size() > 1
        
        # Re-access cached frames (should be fast)
        for i in range(0, total_frames, 4):
            processor.seek_to_frame(i)
            frame = processor.get_current_frame()
            assert frame is not None
        
        processor.release()