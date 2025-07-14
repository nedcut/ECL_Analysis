# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brightness Sorcerer v2.0 is a PyQt5-based desktop application for advanced video brightness analysis. The application allows users to analyze brightness changes in user-defined regions of interest (ROIs) across video frames using CIE LAB color space for perceptually uniform measurements.

## Core Architecture

### Single-File Application Structure
- **main.py**: Complete application implementation (~2200+ lines)
  - `FrameCache` class: LRU cache system for smooth video navigation (100 frame default)
  - `VideoAnalyzer` class: Main PyQt5 application window with comprehensive video analysis functionality

### Key Components

#### Video Processing Pipeline
- OpenCV-based video loading and frame extraction
- CIE LAB color space conversion for perceptually uniform brightness (L* channel, 0-100 scale)
- Frame caching system for performance optimization
- Noise filtering (removes pixels below 5 L* units by default)

#### ROI Management System
- Interactive ROI drawing, selection, moving, and resizing
- Up to 8 color-coded ROIs with visual handles
- Background ROI support for threshold-based analysis
- Real-time brightness display for current frame

#### Analysis Engine
- Dual statistics: mean and median brightness calculation
- Auto-detection of frame ranges based on brightness thresholds
- Manual threshold controls with background ROI support
- Progress tracking with ETA estimation

#### Data Export
- CSV export with frame-by-frame brightness data
- Enhanced dual-panel matplotlib plots (300 DPI)
- Statistical overlays, confidence bands, and peak detection

## Development Commands

### Running the Application
```bash
python main.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- PyQt5 (>=5.15.0) - GUI framework
- opencv-python (>=4.5.0) - Video processing
- pandas (>=1.3.0) - Data manipulation
- numpy (>=1.21.0) - Numerical operations
- matplotlib (>=3.4.0) - Plotting

## Important Constants and Configuration

### Brightness Analysis
- `AUTO_DETECT_BASELINE_PERCENTILE = 5` - Baseline brightness calculation
- `DEFAULT_MANUAL_THRESHOLD = 5.0` - Default threshold above baseline
- `BRIGHTNESS_NOISE_FLOOR_PERCENTILE = 2` - Noise filtering level

### Performance Settings
- `FRAME_CACHE_SIZE = 100` - Number of frames to cache
- `JUMP_FRAMES = 10` - Frame jump amount for Page Up/Down
- `MAX_RECENT_FILES = 10` - Recent files list size

### UI Configuration
- Settings auto-saved to `brightness_analyzer_settings.json`
- Color scheme uses dark theme with predefined color constants
- ROI colors and styling defined in `ROI_COLORS` array

## Key Methods and Functionality

### Core Analysis Methods
- `analyze_video()` (main.py:1825) - Primary analysis execution
- `_compute_brightness_stats()` (main.py:2137) - Brightness calculation for ROI
- `_save_analysis_results()` (main.py:1956) - Data export and plotting
- `_generate_enhanced_plot()` (main.py:2033) - Statistical plot generation

### ROI Management
- `_draw_rois()` (main.py:1083) - ROI rendering on video frame
- `_set_background_roi()` (main.py:1232) - Background ROI designation
- Interactive mouse handling for ROI manipulation

### Video Navigation
- Frame caching system with `FrameCache` class
- Keyboard shortcuts for efficient navigation
- Real-time brightness display updates

## File Structure and Output

### Generated Files
- CSV files: `{analysis_name}_{video_name}_ROI{N}_frames{start}-{end}_brightness.csv`
- Plot images: `{analysis_name}_{video_name}_ROI{N}_frames{start}-{end}_plot.png`
- Settings: `brightness_analyzer_settings.json`

### Supported Video Formats
MP4, MOV, AVI, MKV, WMV, M4V, FLV

## Development Notes

### Code Style
- Uses type hints throughout for better code clarity
- Comprehensive error handling with logging
- Modern PyQt5 patterns with proper signal/slot connections
- Efficient memory management with frame caching

### Testing Approach
No formal test framework is configured. Testing should focus on:
- Video loading with various formats
- ROI interaction and manipulation
- Brightness calculation accuracy
- Export functionality verification
- Performance with large video files

### Performance Considerations
- Frame caching significantly improves navigation performance
- ROI processing is optimized for real-time feedback
- Large video files may require memory management consideration
- Analysis progress tracking prevents UI freezing during long operations