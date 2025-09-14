# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brightness Sorcerer v2.0 is a PyQt5-based desktop application for advanced video brightness analysis. The application allows users to analyze brightness changes in user-defined regions of interest (ROIs) across video frames using CIE LAB color space for perceptually uniform measurements.

## Core Architecture

### Single-File Application Structure
- **main.py**: Complete application implementation (~2900+ lines)
  - `FrameCache` class: LRU cache system for smooth video navigation (100 frame default)
  - `VideoAnalyzer` class: Main PyQt5 application window with comprehensive video analysis functionality
  - Modern dark theme with responsive UI design and professional gradients

### Key Components

#### Video Processing Pipeline
- OpenCV-based video loading and frame extraction
- CIE LAB color space conversion for perceptually uniform brightness (L* channel, 0-100 scale)
- Blue channel extraction for blue light analysis (0-255 scale)
- Frame caching system for performance optimization
- **Advanced Noise Filtering System**: Real-time adjustable morphological kernel size, background percentile thresholds, and noise floor filtering

#### ROI Management System
- Interactive ROI drawing, selection, moving, and resizing
- Up to 8 color-coded ROIs with visual handles
- Background ROI support for threshold-based analysis (accessible from both Threshold and ROI tabs)
- **Fixed Pixel Mask System**: Capture and reuse pixel masks from specific frames across entire analysis
- **Auto-Brightest Frame Detection**: Automatically finds and captures masks from optimal frames
- Real-time brightness display for current frame

#### Analysis Engine
- Dual statistics: mean and median brightness calculation for both L* and blue channels
- Background ROI subtraction for noise reduction and baseline correction
- Auto-detection of frame ranges based on brightness thresholds
- Manual threshold controls with background ROI support
- Progress tracking with ETA estimation
- Real-time brightness display with comprehensive statistics

#### Data Export
- CSV export with frame-by-frame brightness data (L* and blue channels, raw and background-subtracted)
- Enhanced dual-panel matplotlib plots (300 DPI) with separate L* and blue channel subplots
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

## Noise Filtering System

### Real-Time Adjustable Parameters
- **Morphological Kernel Size**: 1-15 pixel elliptical kernel for noise removal (default: 3×3)
- **Background Percentile**: 50-99% threshold for background ROI calculation (default: 90%)
- **Noise Floor Threshold**: 0-10 L* absolute brightness threshold for extreme noise filtering (default: 0.0)

### Noise Filtering Features
- **Live Preview**: All parameters update brightness display and pixel mask visualization in real-time
- **Smart Mask Invalidation**: Parameter changes automatically invalidate cached pixel masks
- **Combined Filtering**: Morphological cleaning + background subtraction + noise floor thresholding
- **Auto-Brightest Frame**: Scans analysis range to find optimal frame for mask capture
- **Fixed Mask Persistence**: Captured masks remain consistent across all frames during analysis

### UI Controls Location
- **Visualization Tab**: Contains all noise filtering sliders and mask capture controls
- **Real-time Feedback**: Parameter changes immediately visible in current frame display
- **Progress Tracking**: Auto-brightest frame search shows progress with cancellation option

## Important Constants and Configuration

### Brightness Analysis (Legacy - now adjustable via UI)
- `AUTO_DETECT_BASELINE_PERCENTILE = 5` - Baseline brightness calculation
- `DEFAULT_MANUAL_THRESHOLD = 5.0` - Default threshold above baseline
- `MORPHOLOGICAL_KERNEL_SIZE = 3` - Default morphological kernel size (now adjustable)
- `BRIGHTNESS_NOISE_FLOOR_PERCENTILE = 2` - Noise filtering level (replaced by adjustable threshold)

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
- `analyze_video()` - Primary analysis execution
- `_compute_brightness_stats()` - Enhanced brightness and blue channel calculation with adjustable noise filtering
- `_compute_background_brightness()` - Background ROI brightness calculation with adjustable percentile
- `_save_analysis_results()` - Data export and plotting
- `_generate_enhanced_plot()` - Dual-panel statistical plot generation
- `_update_current_brightness_display()` - Real-time comprehensive brightness display

### Noise Filtering Methods
- `_capture_fixed_masks()` - Capture pixel masks from current frame with current noise parameters
- `_auto_capture_brightest_frame_masks()` - Automatically find brightest frame and capture masks
- `_on_kernel_size_changed()` - Handle morphological kernel size adjustments
- `_on_bg_percentile_changed()` - Handle background percentile threshold changes
- `_on_noise_floor_changed()` - Handle noise floor threshold adjustments
- `_invalidate_fixed_masks()` - Clear cached masks when parameters change

### ROI Management
- `_draw_rois()` (main.py:1083) - ROI rendering on video frame
- `_set_background_roi()` (main.py:1232) - Background ROI designation
- Interactive mouse handling for ROI manipulation

### Video Navigation
- Frame caching system with `FrameCache` class
- Keyboard shortcuts for efficient navigation
- Real-time brightness display updates

### Blue Channel Analysis Feature
- Extracts blue channel values (0-255 range) alongside L* brightness measurements
- Useful for blue light analysis and screen brightness correlation studies
- Displays in real-time brightness panel: "ROI 1: L* 45.2 (BG-Sub: 29.9) | Blue: 125"
- Exported in CSV with `blue_mean` and `blue_median` columns
- Visualized in dedicated subplot in enhanced dual-panel plots
- Background ROI blue values also calculated and displayed

## File Structure and Output

### Generated Files
- CSV files: `{analysis_name}_{video_name}_ROI{N}_frames{start}-{end}_brightness.csv`
  - Columns: frame, l_raw_mean, l_raw_median, l_bg_sub_mean, l_bg_sub_median, blue_mean, blue_median, timestamp
- Plot images: `{analysis_name}_{video_name}_ROI{N}_frames{start}-{end}_plot.png`
  - Dual-panel plots with L* brightness (top) and blue channel (bottom) subplots
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