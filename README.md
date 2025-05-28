# ECL_Analysis
## Ned Cutler and Jasper Pearcy

A tool to analyze brightness over time of a section of a video.

## Features

- **Interactive Video Analysis**: Load video files via drag-and-drop or file dialog
- **ROI Management**: Draw, resize, and move multiple regions of interest on video frames
- **Automated Analysis**: Automatic brightness analysis with integrated plot generation
- **Comprehensive Brightness Metrics**: Calculate both mean and median brightness values for each ROI
- **Real-time Brightness Display**: See current frame brightness (mean±median) for all ROIs
- **Auto-Detection**: Automatically detect frame ranges based on brightness thresholds
- **Export Results**: Save analysis data as CSV files with both mean and median columns, plus plots as PNG images
- **Modern UI**: Dark theme with professional styling

## Usage

1. **Load Video**: Drag and drop a video file or click "Open Video..."
2. **Define ROIs**: Click "Add ROI" and draw rectangles on the video frame
3. **Set Frame Range**: Use "Set Start", "Set End", or "Auto-Detect" buttons
4. **Analyze**: Click "Analyze Brightness" to process the video and automatically generate plots
5. **Results**: CSV data and PNG plots are saved to your chosen directory

## Requirements

- Python 3.6+
- PyQt5
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib

## Recent Improvements

- **Streamlined Workflow**: Analysis now automatically generates and displays plots
- **Single Directory Output**: CSV and PNG files are saved to the same location
- **Enhanced Brightness Analysis**: Now calculates both mean and median brightness values
- **Dual-Metric CSV Output**: CSV files include separate columns for brightness_mean and brightness_median
- **Comparative Plotting**: Plots display both mean and median brightness lines with distinct peak markers
- **Improved Real-time Display**: Current frame brightness shows both mean±median for each ROI
- **Enhanced Progress Tracking**: Better progress dialogs for analysis and plot generation
- **Improved Error Handling**: More robust error handling throughout the application
- **Modern Plot Styling**: Enhanced plot appearance with grid lines and better formatting