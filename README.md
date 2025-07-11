# Brightness Sorcerer v2.0

Advanced video brightness analysis tool for analyzing brightness changes in video regions of interest (ROIs) with automatic detection and comprehensive plotting.

## Features

### Core Analysis
- **Interactive ROI Management**: Draw, move, resize, and delete regions of interest directly on video frames
- **Dual Statistics**: Calculate both mean and median brightness values for comprehensive analysis
- **Automatic Frame Range Detection**: Intelligently detect start/end frames based on brightness thresholds
- **CIE LAB Color Space**: Uses perceptually uniform L* brightness measurements (0-100 scale)
- **Noise Filtering**: Automatically filters out dark pixels to focus on meaningful brightness data

### User Experience
- **Frame Caching**: Smooth video navigation with intelligent frame caching (100 frames)
- **Keyboard Shortcuts**: Comprehensive shortcuts for efficient workflow
- **Recent Files**: Quick access to recently opened videos
- **Drag & Drop**: Simple file loading via drag and drop
- **Real-time Feedback**: Live brightness display for current frame and selected ROIs

### Analysis & Visualization
- **Enhanced Plotting**: Dual-panel plots with brightness trends and difference analysis
- **Statistical Overlays**: Confidence bands, peak detection, and comprehensive statistics
- **High-Quality Output**: 300 DPI plots with professional styling
- **CSV Export**: Detailed frame-by-frame data export for further analysis
- **Progress Tracking**: Real-time analysis progress with ETA estimation

## Installation

### Requirements
- Python 3.7+
- PyQt5
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib

### Setup
```bash
# Clone the repository
git clone [repository-url]
cd ECL_Analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Usage

### Getting Started
1. **Load Video**: Click "Open Video" or drag & drop a video file
2. **Define ROIs**: Click "Add ROI" and draw rectangles on areas of interest
3. **Set Analysis Range**: Use "Set Start/End" buttons or "Auto-Detect" for automatic range detection
4. **Run Analysis**: Click "Analyze Brightness" and choose output directory

### Keyboard Shortcuts

#### Navigation
- **Arrow Keys**: Previous/Next frame
- **Space**: Next frame
- **Backspace**: Previous frame
- **Page Up/Down**: Jump 10 frames
- **Home/End**: Go to first/last frame

#### Analysis
- **F5**: Run brightness analysis
- **Ctrl+D**: Auto-detect frame range

#### ROI Management
- **Delete**: Remove selected ROI
- **Escape**: Cancel current drawing/editing action

#### File Operations
- **Ctrl+O**: Open video file
- **Ctrl+Q**: Exit application

### ROI Interaction
- **Drawing**: Click "Add ROI" button, then click and drag on video
- **Selection**: Click on any existing ROI to select it
- **Moving**: Click and drag inside a selected ROI
- **Resizing**: Click and drag corner handles of selected ROI
- **Deletion**: Select ROI and press Delete key or use "Delete ROI" button

### Auto-Detection
The auto-detection feature scans the entire video to find frames where ROI brightness significantly exceeds baseline levels:
- Uses 5th percentile as baseline brightness
- Threshold set 5 L* units above baseline
- Automatically sets start/end frames based on brightness peaks

### Analysis Output
Each analysis generates:
- **CSV Files**: Frame-by-frame brightness data (mean and median)
- **Plot Images**: Dual-panel visualization with:
  - Main brightness trends over time
  - Difference plot (mean - median)
  - Statistical annotations and peak markers
  - Confidence bands showing data variability

## Technical Details

### Brightness Calculation
- Converts BGR video frames to CIE LAB color space
- Uses L* channel for perceptually uniform brightness
- Filters pixels below threshold, default 5 L* units to remove noise
- Calculates both mean and median for robust statistics

### Performance Optimizations
- **Frame Caching**: LRU cache system for smooth navigation
- **Efficient Seeking**: Smart frame reading with minimal video file access
- **Progress Tracking**: Real-time feedback during long analyses
- **Memory Management**: Automatic cleanup and resource management

### File Support
Supported video formats:
- MP4, MOV, AVI, MKV, WMV, M4V, FLV

## Configuration

Settings are automatically saved to `brightness_analyzer_settings.json`:
- Recent files list (up to 10 files)
- Window geometry and preferences
- Analysis parameters

## Troubleshooting

### Common Issues
- **Video won't load**: Ensure the video format is supported and file isn't corrupted
- **Slow performance**: Reduce cache size or close other applications
- **Analysis fails**: Check that ROIs are within frame boundaries and frame range is valid

### Performance Tips
- Use smaller ROIs for faster analysis
- Close unnecessary applications during long analyses
- Consider reducing video resolution for very large files
