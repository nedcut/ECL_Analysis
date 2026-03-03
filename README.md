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

The core dependencies above are enough to launch the UI and run brightness analysis. Optional extras unlock quality-of-life features:

- `pygame` — enables audio cues for analysis start/finish events
- `librosa` + `soundfile` — unlock automated audio-based run detection

The application gracefully disables these extras when the packages are absent.

### Setup
```bash
# Clone the repository
git clone [repository-url]
cd ecl

# (Recommended) create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: preflight import compilation
python -m compileall ecl_analysis

# Run the application
python main.py
# or launch with a video preloaded
python main.py /path/to/video.mp4
```

### Development Checks
```bash
# Install runtime + developer tooling
pip install -r requirements-dev.txt

# Fast quality gate
python -m compileall ecl_analysis
pytest -q -m "not performance"
pytest -q -m "not performance" --cov=ecl_analysis --cov-report=term-missing
python -m ruff check ecl_analysis tests

# Optional performance checks
pytest -q -m performance
```

If you are using the project venv:
```bash
.venv/bin/python -m compileall ecl_analysis
.venv/bin/python -m pytest -q -m "not performance"
.venv/bin/python -m pytest -q -m "not performance" --cov=ecl_analysis --cov-report=term-missing
.venv/bin/python -m ruff check ecl_analysis tests

# Optional performance checks
.venv/bin/python -m pytest -q -m performance
```

Manual smoke workflow for UI-affecting changes:
1. Launch `python main.py`
2. Open a short sample MP4
3. Draw at least one ROI
4. Run auto-detect
5. Run analysis
6. Confirm CSV and plot outputs are produced

## Usage

### Getting Started
1. **Load Video**: Click "Open Video" or drag & drop a video file
2. **Define ROIs**: Click "Add ROI" and draw rectangles on areas of interest
3. **Set Analysis Range**: Use "Set Start/End" buttons or "Auto-Detect" for automatic range detection
4. **Run Analysis**: Click "Analyze Brightness" and choose output directory

### Keyboard Shortcuts

#### Navigation
- **Left/Right Arrow**: Previous/Next frame (or nudge selected ROI)
- **Up/Down Arrow**: Nudge selected ROI vertically
- **Shift + Arrows**: Large 10px ROI nudge
- **Space**: Play/Pause video playback
- **Backspace**: Previous frame
- **Page Up/Down**: Jump 10 frames
- **Home/End**: Go to first/last frame

#### Analysis
- **F5**: Run brightness analysis
- **Ctrl+D**: Auto-detect frame range

#### ROI Management
- **Delete**: Remove selected ROI
- **Ctrl+Shift+D**: Duplicate selected ROI
- **Ctrl+Alt+D**: Duplicate selected ROI multiple times
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

### Architecture Map
- `main.py` launches the app.
- `ecl_analysis/video_analyzer.py` is the main Qt window/orchestrator.
- `ecl_analysis/analysis/` contains pure analysis logic and typed request/result models.
- `ecl_analysis/workers.py` runs long tasks (`analysis`, `audio detect`, `mask scans`) off the UI thread.
- `ecl_analysis/export/` handles CSV/plot export orchestration from `AnalysisResult`.
- `ecl_analysis/roi_geometry.py` owns frame/label coordinate mapping helpers.
- `ecl_analysis/cache.py` provides LRU frame caching.
- `ecl_analysis/audio.py` provides playback cues and optional beep detection.

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
- **Background Workers**: Analysis/audio/mask scans run in `QThread` workers

### Benchmark Harness
Run lightweight local benchmarks:
```bash
python tests/performance/benchmark_analysis.py
```

### File Support
Supported video formats:
- MP4, MOV, AVI, MKV, WMV, M4V, FLV

## Configuration

Settings are automatically saved to `brightness_analyzer_settings.json`:
- Recent files list (up to 10 entries)
- Audio feedback enabled/disabled flag
- Audio volume level

Window layout and analysis parameters are currently session-based and reset when you relaunch the app.
## Measurement Considerations

Brightness Sorcerer reports **relative** L\* brightness values derived from smartphone video. The following considerations apply when interpreting results for scientific publication.

### Camera Requirements

- **Manual exposure mode is required.** Auto-exposure adjusts sensor gain between frames, meaning brightness changes in the video may reflect camera behavior rather than electrode behavior. Lock exposure, ISO, and white balance before recording.
- **sRGB assumption.** The BGR → CIE LAB conversion assumes the input conforms to the sRGB color space (D65 illuminant). Most smartphone cameras produce approximately sRGB output, but manufacturer-specific tone mapping and HDR processing can introduce non-linearities. Disabling HDR, "night mode," and similar post-processing features is strongly recommended.
- **No camera response function (CRF) calibration.** Results are internally consistent within a single recording session (same device, same settings) but are not directly comparable across different devices or recording sessions unless an external calibration target is used.

### Spatial and Optical Effects

- **Lens vignetting** (brightness falloff toward frame edges) is not corrected. Background subtraction partially compensates when the background ROI is near the analysis ROI, but systematic error increases with spatial separation.
- **No lens distortion correction.** Minor barrel/pincushion distortion from smartphone lenses may slightly affect ROI pixel counts at frame edges.
- **ROI placement matters.** For best results, keep analysis and background ROIs in the same region of the frame to minimize vignetting and illumination gradient effects.

### Analysis Pipeline Notes

- **Background subtraction method.** A user-designated background ROI provides a per-frame reference brightness (configurable percentile, default 90th). This adapts to gradual lighting drift but assumes the background ROI contains no glow signal.
- **Morphological filtering.** The opening operation (erode → dilate) removes isolated bright pixels but may erode the edges of small glow regions. For ROIs smaller than ~50 pixels, smaller kernel sizes (1–3) are recommended.
- **No temporal smoothing.** Each frame is analyzed independently. Frame-to-frame noise is preserved in the exported data, which allows post-hoc filtering (e.g., moving average, Savitzky-Golay) but means raw traces may appear noisier than time-averaged instruments.
- **Blue channel values** are reported on the raw 0–255 sensor scale without perceptual correction. They are useful for qualitative spectral trends but should not be treated as calibrated spectral measurements.

### Reporting Recommendations

When citing results from this tool in publications, consider noting:
1. Brightness values are reported as CIE L\* (0–100, perceptually uniform) relative to a background reference region.
2. Camera model, recording settings (resolution, frame rate, exposure, ISO, white balance), and any disabled post-processing features.
3. Background subtraction percentile and morphological kernel size used.
4. Whether fixed masks or per-frame adaptive thresholding was applied.

## Troubleshooting

### Common Issues
- **Video won't load**: Ensure the video format is supported and file isn't corrupted
- **Slow performance**: Reduce cache size or close other applications
- **Analysis fails**: Check that ROIs are within frame boundaries and frame range is valid

### Performance Tips
- Use smaller ROIs for faster analysis
- Close unnecessary applications during long analyses
- Consider reducing video resolution for very large files
