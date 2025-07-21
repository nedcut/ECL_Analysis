"""
Global test configuration and fixtures for Brightness Sorcerer tests.

Provides shared fixtures, mock data, and test utilities used across
the entire test suite.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock
from PyQt5.QtWidgets import QApplication

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brightness_sorcerer.core.exceptions import ValidationError
from brightness_sorcerer.utils.constants import *


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for GUI tests."""
    if not QApplication.instance():
        app = QApplication([])
        app.setAttribute(QtCore.Qt.AA_X11InitThreads, True)
        yield app
        app.quit()
    else:
        yield QApplication.instance()


@pytest.fixture
def sample_image():
    """Create a sample test image (RGB)."""
    # Create a 100x100 RGB image with gradient
    height, width = 100, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient pattern
    for y in range(height):
        for x in range(width):
            image[y, x] = [x * 255 // width, y * 255 // height, 128]
    
    return image


@pytest.fixture
def sample_lab_image():
    """Create a sample LAB color space image."""
    # Create a sample image and convert to LAB
    rgb_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    return lab_image


@pytest.fixture
def sample_roi():
    """Create a sample ROI (Region of Interest)."""
    return {
        'x1': 10,
        'y1': 10, 
        'x2': 50,
        'y2': 50,
        'width': 40,
        'height': 40
    }


@pytest.fixture
def mock_video_file():
    """Create a mock video file path."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        # Create a simple test video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(f.name, fourcc, 30.0, (640, 480))
        
        # Write 10 frames
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        
        writer.release()
        yield f.name
        
    # Cleanup
    try:
        os.unlink(f.name)
    except OSError:
        pass


@pytest.fixture
def sample_video_capture():
    """Create a mock video capture object."""
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480
    }.get(prop, 0)
    
    # Mock frame reading
    def mock_read():
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return True, frame
    
    mock_cap.read = mock_read
    mock_cap.set.return_value = True
    mock_cap.release.return_value = None
    
    return mock_cap


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = os.path.join(temp_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)
        yield config_dir


@pytest.fixture
def sample_settings():
    """Create sample application settings."""
    return {
        'recent_files': [],
        'audio_enabled': True,
        'audio_volume': 0.7,
        'window_geometry': None,
        'window_state': None,
        'last_directory': '/tmp',
        'auto_save_results': True,
        'default_analysis_name': 'test_analysis',
        'frame_cache_size': 50,
        'log_level': 'INFO'
    }


@pytest.fixture
def mock_brightness_stats():
    """Create mock brightness statistics."""
    return {
        'l_raw_mean': 45.5,
        'l_raw_median': 44.0,
        'l_bg_sub_mean': 30.2,
        'l_bg_sub_median': 29.5,
        'blue_mean': 125.3,
        'blue_median': 120.0,
        'enhanced_mean': 52.1,
        'enhanced_median': 50.8
    }


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Configure logging for tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    yield
    # Reset logging after test
    logging.getLogger().handlers.clear()


@pytest.fixture
def mock_audio_manager():
    """Create a mock audio manager."""
    mock_manager = Mock()
    mock_manager.initialize.return_value = True
    mock_manager.is_initialized = True
    mock_manager.play_sound.return_value = None
    mock_manager.cleanup.return_value = None
    return mock_manager


# Test utilities
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_test_video_frames(num_frames=10, width=640, height=480):
        """Generate test video frames."""
        frames = []
        for i in range(num_frames):
            # Create frames with varying brightness
            brightness = int(255 * (i / num_frames))
            frame = np.full((height, width, 3), brightness, dtype=np.uint8)
            frames.append(frame)
        return frames
    
    @staticmethod
    def create_brightness_data(num_frames=100):
        """Generate test brightness analysis data."""
        data = []
        for i in range(num_frames):
            frame_data = {
                'frame': i,
                'l_raw_mean': 30 + 20 * np.sin(i * 0.1),
                'l_raw_median': 28 + 18 * np.sin(i * 0.1),
                'l_bg_sub_mean': 15 + 10 * np.sin(i * 0.1),
                'l_bg_sub_median': 13 + 8 * np.sin(i * 0.1),
                'blue_mean': 100 + 50 * np.cos(i * 0.05),
                'blue_median': 95 + 45 * np.cos(i * 0.05),
                'timestamp': i / 30.0  # Assuming 30 FPS
            }
            data.append(frame_data)
        return data


# Make test generator available globally
@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()