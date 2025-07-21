"""
Create test data and fixtures for Brightness Sorcerer tests.

This module generates sample video files, images, and data for testing.
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path


def create_test_video(output_path: str, frames: int = 30, width: int = 640, height: int = 480, fps: float = 30.0):
    """
    Create a test video file with varying brightness patterns.
    
    Args:
        output_path: Path where to save the test video
        frames: Number of frames to generate
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        for i in range(frames):
            # Create frame with varying brightness
            # Sinusoidal brightness pattern
            brightness = int(128 + 100 * np.sin(i * 2 * np.pi / frames))
            
            # Create frame with gradient and brightness variation
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create gradient pattern
            for y in range(height):
                for x in range(width):
                    # Base color with brightness variation
                    r = min(255, brightness + (x * 50 // width))
                    g = min(255, brightness + (y * 50 // height))
                    b = brightness
                    frame[y, x] = [b, g, r]  # BGR format for OpenCV
            
            # Add some ROI-like bright spots
            if i % 10 == 0:  # Every 10th frame
                # Add bright rectangle (simulates ROI with activity)
                cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
            
            writer.write(frame)
    
    finally:
        writer.release()


def create_test_images():
    """Create test images for various scenarios."""
    images = {}
    
    # Gradient image
    gradient = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(100):
        for x in range(100):
            gradient[y, x] = [x * 255 // 100, y * 255 // 100, 128]
    images['gradient'] = gradient
    
    # Uniform brightness image
    uniform = np.full((100, 100, 3), 128, dtype=np.uint8)
    images['uniform'] = uniform
    
    # High contrast image
    contrast = np.zeros((100, 100, 3), dtype=np.uint8)
    contrast[:50, :] = 255  # Top half white
    contrast[50:, :] = 0    # Bottom half black
    images['high_contrast'] = contrast
    
    # Low light image
    low_light = np.full((100, 100, 3), 20, dtype=np.uint8)
    images['low_light'] = low_light
    
    # Bright image
    bright = np.full((100, 100, 3), 200, dtype=np.uint8)
    images['bright'] = bright
    
    return images


def create_sample_settings():
    """Create sample application settings for testing."""
    return {
        "recent_files": [
            "/path/to/test_video1.mp4",
            "/path/to/test_video2.avi"
        ],
        "audio_enabled": True,
        "audio_volume": 0.7,
        "window_geometry": None,
        "window_state": None,
        "last_directory": "/tmp",
        "auto_save_results": True,
        "default_analysis_name": "test_analysis",
        "frame_cache_size": 50,
        "log_level": "INFO"
    }


def create_sample_analysis_data():
    """Create sample brightness analysis data."""
    num_frames = 100
    data = []
    
    for i in range(num_frames):
        # Generate realistic brightness data with some variation
        base_brightness = 40
        variation = 15 * np.sin(i * 0.1) + 5 * np.random.normal()
        
        frame_data = {
            "frame": i,
            "l_raw_mean": max(0, min(100, base_brightness + variation)),
            "l_raw_median": max(0, min(100, base_brightness + variation - 2)),
            "l_bg_sub_mean": max(0, min(100, base_brightness + variation - 10)),
            "l_bg_sub_median": max(0, min(100, base_brightness + variation - 12)),
            "blue_mean": max(0, min(255, 120 + 30 * np.cos(i * 0.05))),
            "blue_median": max(0, min(255, 115 + 25 * np.cos(i * 0.05))),
            "timestamp": i / 30.0  # 30 FPS
        }
        data.append(frame_data)
    
    return data


def create_roi_test_data():
    """Create sample ROI data for testing."""
    return [
        {"x1": 50, "y1": 50, "x2": 150, "y2": 150, "name": "ROI_1"},
        {"x1": 200, "y1": 100, "x2": 300, "y2": 200, "name": "ROI_2"},
        {"x1": 400, "y1": 200, "x2": 500, "y2": 300, "name": "ROI_3"},
    ]


def setup_test_fixtures(fixtures_dir: str):
    """
    Set up all test fixtures in the specified directory.
    
    Args:
        fixtures_dir: Directory where to create test fixtures
    """
    fixtures_path = Path(fixtures_dir)
    fixtures_path.mkdir(exist_ok=True)
    
    # Create test video
    video_path = fixtures_path / "test_video.mp4"
    print(f"Creating test video: {video_path}")
    create_test_video(str(video_path))
    
    # Create small test video
    small_video_path = fixtures_path / "small_test_video.mp4"
    print(f"Creating small test video: {small_video_path}")
    create_test_video(str(small_video_path), frames=10, width=320, height=240)
    
    # Save test images
    images = create_test_images()
    for name, image in images.items():
        image_path = fixtures_path / f"test_{name}.png"
        cv2.imwrite(str(image_path), image)
        print(f"Created test image: {image_path}")
    
    # Save sample settings
    settings_path = fixtures_path / "sample_settings.json"
    with open(settings_path, 'w') as f:
        json.dump(create_sample_settings(), f, indent=2)
    print(f"Created sample settings: {settings_path}")
    
    # Save sample analysis data
    analysis_path = fixtures_path / "sample_analysis_data.json"
    with open(analysis_path, 'w') as f:
        json.dump(create_sample_analysis_data(), f, indent=2)
    print(f"Created sample analysis data: {analysis_path}")
    
    # Save ROI test data
    roi_path = fixtures_path / "sample_roi_data.json"
    with open(roi_path, 'w') as f:
        json.dump(create_roi_test_data(), f, indent=2)
    print(f"Created sample ROI data: {roi_path}")
    
    print(f"Test fixtures created in: {fixtures_path}")


if __name__ == "__main__":
    # Create fixtures when run directly
    current_dir = Path(__file__).parent
    setup_test_fixtures(str(current_dir))