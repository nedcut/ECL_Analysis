"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
import os
import numpy as np
import cv2
from unittest.mock import Mock
from PyQt5 import QtWidgets, QtCore

from ..core.video_processor import VideoProcessor
from ..core.roi_manager import ROIManager
from ..core.brightness_analyzer import BrightnessAnalyzer
from ..core.settings_manager import SettingsManager
from ..models.roi import ROI


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for GUI tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app
    # Don't quit the app as it might be used by other tests


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def settings_file(temp_dir):
    """Create a temporary settings file."""
    settings_path = os.path.join(temp_dir, "test_settings.json")
    yield settings_path
    # Cleanup is handled by temp_dir fixture


@pytest.fixture
def sample_video_frame():
    """Create a sample video frame for testing."""
    # Create a 640x480 color frame with some pattern
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some patterns for testing
    # Blue rectangle in top-left
    frame[50:150, 50:150] = [255, 0, 0]  # BGR format
    
    # Green rectangle in top-right
    frame[50:150, 490:590] = [0, 255, 0]
    
    # Red rectangle in bottom-left
    frame[330:430, 50:150] = [0, 0, 255]
    
    # White rectangle in bottom-right
    frame[330:430, 490:590] = [255, 255, 255]
    
    # Add some noise
    noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame


@pytest.fixture
def sample_video_file(temp_dir, sample_video_frame):
    """Create a sample video file for testing."""
    video_path = os.path.join(temp_dir, "test_video.mp4")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    # Write 100 frames with slight variations
    for i in range(100):
        frame = sample_video_frame.copy()
        # Add frame-specific variation
        frame = cv2.addWeighted(frame, 0.9, 
                               np.full_like(frame, i * 2), 0.1, 0)
        out.write(frame)
    
    out.release()
    
    # Verify the file was created
    if os.path.exists(video_path):
        yield video_path
    else:
        pytest.skip("Could not create test video file")


@pytest.fixture
def mock_video_processor():
    """Create a mock VideoProcessor for testing."""
    processor = Mock(spec=VideoProcessor)
    processor.is_loaded.return_value = True
    processor.total_frames = 100
    processor.current_frame_index = 0
    processor.frame_size = (640, 480)
    processor.fps = 30.0
    
    # Mock video info
    processor.get_video_info.return_value = {
        'path': '/fake/video.mp4',
        'total_frames': 100,
        'fps': 30.0,
        'duration_seconds': 100 / 30.0,
        'frame_size': (640, 480),
        'current_frame': 0,
        'cache_size': 10
    }
    
    return processor


@pytest.fixture
def video_processor():
    """Create a real VideoProcessor instance for testing."""
    return VideoProcessor(cache_size=10)


@pytest.fixture
def roi_manager():
    """Create a ROIManager instance for testing."""
    return ROIManager()


@pytest.fixture
def brightness_analyzer(video_processor):
    """Create a BrightnessAnalyzer instance for testing."""
    return BrightnessAnalyzer(video_processor)


@pytest.fixture
def settings_manager(settings_file):
    """Create a SettingsManager instance for testing."""
    return SettingsManager(settings_file)


@pytest.fixture
def sample_roi():
    """Create a sample ROI for testing."""
    return ROI(x=100, y=100, width=200, height=150, 
              color=(255, 0, 0), label="Test ROI")


@pytest.fixture
def sample_rois():
    """Create multiple sample ROIs for testing."""
    return [
        ROI(x=50, y=50, width=100, height=100, 
            color=(255, 0, 0), label="ROI 1"),
        ROI(x=200, y=200, width=150, height=100, 
            color=(0, 255, 0), label="ROI 2"),
        ROI(x=400, y=300, width=120, height=80, 
            color=(0, 0, 255), label="ROI 3", is_background=True)
    ]


@pytest.fixture
def lab_test_image():
    """Create a test image in LAB color space."""
    # Create a gradient image for testing brightness calculations
    height, width = 100, 100
    lab_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create L* channel gradient (brightness)
    for i in range(height):
        for j in range(width):
            # L* value from 0 to 100
            l_value = int((i * 100) / height)
            lab_image[i, j] = [l_value, 128, 128]  # Neutral a* and b*
    
    return lab_image


@pytest.fixture
def mock_progress_callback():
    """Create a mock progress callback for testing."""
    return Mock()


# Pytest markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.ui = pytest.mark.ui
pytest.mark.slow = pytest.mark.slow
pytest.mark.requires_video = pytest.mark.requires_video
pytest.mark.requires_display = pytest.mark.requires_display


# Helper functions for tests
def assert_roi_equal(roi1: ROI, roi2: ROI):
    """Assert that two ROIs are equal."""
    assert roi1.x == roi2.x
    assert roi1.y == roi2.y
    assert roi1.width == roi2.width
    assert roi1.height == roi2.height
    assert roi1.color == roi2.color
    assert roi1.label == roi2.label
    assert roi1.is_background == roi2.is_background


def create_test_frame(width: int = 640, height: int = 480, 
                     brightness: int = 128) -> np.ndarray:
    """Create a test frame with specified dimensions and brightness."""
    frame = np.full((height, width, 3), brightness, dtype=np.uint8)
    return frame


def skip_if_no_display():
    """Skip test if no display is available."""
    if not os.environ.get('DISPLAY') and os.name != 'nt':
        pytest.skip("No display available for GUI tests")