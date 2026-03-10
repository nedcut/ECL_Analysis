# Brightness Sorcerer

Video brightness analysis tool for electrochemiluminescence (ECL) experiments. Measures CIE L\* brightness in user-defined regions of interest (ROIs) across video frames, exports frame-by-frame CSV data and annotated plots.

## Quick Start

### First-Time Setup

```bash
# 1. Clone the repo
git clone https://github.com/nedcut/ECL_Analysis.git
cd ECL_Analysis

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python main.py
```

### Updating When New Code Is Pushed

Whenever updates are pushed, run these commands to get the latest version:

```bash
cd ECL_Analysis
source .venv/bin/activate
git pull
pip install -r requirements.txt   # only needed if dependencies changed
python main.py
```

If `git pull` shows a conflict or error, reach out before trying to fix it.

## Usage

1. **Open a video** — click "Open Video" or drag & drop an MP4/MOV/AVI file
2. **Draw ROIs** — click "Add ROI" and draw rectangles over areas of interest (electrodes, background reference, etc.)
3. **Set frame range** — use "Set Start/End" buttons to find the active region automatically
4. **Run analysis** — click "Analyze Brightness" (or press F5), choose an output folder

### Output

Each analysis produces:
- **CSV files** — one per ROI with columns: `frame, brightness_mean, brightness_median, blue_mean, blue_median`
- **Plot images** — dual-panel PNG (brightness trends + difference plot) with statistical annotations

### Useful Shortcuts

| Action | Key |
|---|---|
| Previous/Next frame | Left/Right Arrow |
| Jump 10 frames | Page Up/Down |
| Play/Pause | Space |
| Run analysis | F5 |
| Auto-detect range | Ctrl+D |
| Delete selected ROI | Delete |
| Duplicate ROI | Ctrl+Shift+D |
| Open video | Ctrl+O |

Arrow keys nudge a selected ROI instead of navigating frames. Shift+Arrow for 10px nudge.

## How It Works

### Analysis Pipeline

1. User draws ROIs on the video frame (one can be designated as a background reference).
2. For each frame in the selected range, the tool converts BGR pixels to **CIE LAB** color space and extracts the **L\* channel** (perceptually uniform brightness, 0–100 scale).
3. Pixels below a noise threshold (default 5 L\*) are filtered out. An optional morphological opening (erode then dilate) removes isolated bright pixels.
4. If a background ROI is set, its brightness (configurable percentile, default 90th) is subtracted per-frame to compensate for lighting drift.
5. Both mean and median brightness are computed per ROI per frame.
6. Results are exported to CSV and plotted.

### Architecture

```
main.py                        → entry point
ecl_analysis/
  app.py                       → Qt bootstrap
  video_analyzer.py            → main window / UI orchestrator
  workers.py                   → QThread workers (analysis, audio detect, mask scans)
  cache.py                     → LRU frame cache
  roi_geometry.py              → coordinate mapping helpers
  audio.py                     → optional audio cues & beep detection
  constants.py                 → app-wide defaults and thresholds
  analysis/
    models.py                  → AnalysisRequest / AnalysisResult dataclasses
    brightness.py              → core brightness computation (BGR → L*)
    background.py              → background ROI subtraction
    duration.py                → frame range helpers
  export/
    csv_exporter.py            → CSV + plot output
```

### Dependencies

**Required:** PyQt5, OpenCV, NumPy, Pandas, Matplotlib

**Optional** (the app works without these):
- `pygame` — audio cues when analysis starts/finishes
- `plotly` — interactive HTML plots alongside static PNGs
- `librosa` + `soundfile` — audio-based automatic run detection

## Measurement Considerations

Brightness Sorcerer reports **relative** L\* brightness values derived from smartphone video. Keep the following in mind when interpreting results.

### Camera Setup

- **Manual exposure mode is required.** Auto-exposure adjusts sensor gain between frames, so brightness changes may reflect camera behavior rather than electrode behavior. Lock exposure, ISO, and white balance before recording.
- **sRGB assumption.** The BGR → CIE LAB conversion assumes sRGB input (D65 illuminant). Disable HDR, "night mode," and similar post-processing features.
- **No cross-device calibration.** Results are internally consistent within a single recording session (same device, same settings) but not directly comparable across devices unless an external calibration target is used.

### Spatial Effects

- **Lens vignetting** (brightness falloff toward frame edges) is not corrected. Background subtraction partially compensates when the background ROI is near the analysis ROI.
- **ROI placement matters.** Keep analysis and background ROIs in the same region of the frame to minimize vignetting and illumination gradient effects.

### Pipeline Notes

- **Background subtraction** uses a configurable percentile (default 90th) from the background ROI. This adapts to gradual lighting drift but assumes the background ROI contains no glow signal.
- **Morphological filtering** removes isolated bright pixels but may erode edges of very small glow regions. For ROIs smaller than ~50 px, use smaller kernel sizes (1–3).
- **No temporal smoothing.** Each frame is analyzed independently. Raw traces may appear noisier than time-averaged instruments; post-hoc filtering (moving average, Savitzky-Golay) can be applied to the exported CSV data.
- **Blue channel values** are on the raw 0–255 sensor scale without perceptual correction — useful for qualitative spectral trends, not calibrated spectral measurements.

### Reporting Recommendations

When citing results in publications, note:
1. Brightness values are CIE L\* (0–100, perceptually uniform) relative to a background reference region.
2. Camera model, recording settings (resolution, frame rate, exposure, ISO, white balance), and any disabled post-processing.
3. Background subtraction percentile and morphological kernel size used.
4. Whether fixed masks or per-frame adaptive thresholding was applied.

## Development

```bash
# Install dev tools
pip install -r requirements-dev.txt

# Run tests
pytest -q -m "not performance"

# Lint
python -m ruff check ecl_analysis tests

# Coverage
pytest -q -m "not performance" --cov=ecl_analysis --cov-report=term-missing
```
