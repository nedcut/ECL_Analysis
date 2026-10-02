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

# 3. Install dependencies (core + optional audio/interactive-plot extras)
pip install -e ".[audio,interactive-plots]"

# 4. Run the app
python main.py
```

Only need the core features? `pip install -e .` skips the optional `pygame`/`librosa`/`soundfile`/`plotly` extras.

### Updating When New Code Is Pushed

Whenever updates are pushed, run these commands to get the latest version:

```bash
cd ECL_Analysis
source .venv/bin/activate
git pull
pip install -e ".[audio,interactive-plots]"   # only needed if dependencies changed
python main.py
```

If `git pull` shows a conflict or error, reach out before trying to fix it.

## Usage

1. **Open a video** — click "Open Video" or drag & drop an MP4/MOV/AVI file
2. **Draw ROIs** — click "Add ROI" and draw rectangles over areas of interest (electrodes, background reference, etc.)
3. **Set frame range** — use "Set Start/End" buttons to find the active region automatically
4. **Run analysis** — click "Analyze Brightness" (or press F5), choose an output folder

### Output

Each analysis run produces, in the chosen folder:
- **CSV files** (and optional JSON) — one per ROI, named `<analysis>_<video>_ROI<n>_frames<start>-<end>_brightness.csv`
- **Plots** — dual-panel PNG (L\* and blue channel) and an optional interactive HTML plot per ROI
- **Run metadata** — one `<analysis>_<video>_frames<start>-<end>_metadata.json` sidecar per run

#### CSV columns

| Column | Meaning |
|---|---|
| `frame` | **1-based** video frame number, the same numbering as the UI, the filename and the JSON `frame_range` |
| `brightness_mean`, `brightness_median` | CIE L\* (0–100) for the ROI; how it is computed depends on the ROI's measurement method (see below) |
| `blue_mean`, `blue_median` | Raw blue channel (0–255) over the same pixels as the L\* values. Blue is never background-subtracted |
| `pixel_count` | Number of pixels that contributed to the L\* values in that frame |
| `background_l` | Background/threshold L\* applied to that frame (background-ROI percentile, or the manual threshold). Empty when none was applied |
| `time_s` | Video timestamp of the frame in seconds, `(frame - 1) / fps`. Empty if the video's fps is unknown |

> **Breaking change:** older exports wrote `frame` as a 0-based index, so it was 1 less than the frame
> number shown in the UI and in the filename. Scripts that align by `frame` need updating: subtract 1
> from new files to get the old values. `pixel_count`, `background_l` and `time_s` are new columns,
> added after the original five.

#### Measurement methods

The run metadata records which method was used for each ROI:

- **`threshold`** (a background ROI is set, or the manual threshold is > 0): only pixels with L\* above
  the background/threshold value (after morphological opening and the noise floor) are averaged, after
  subtracting that value. Values are always ≥ 0 and describe only the lit pixels, so they read higher
  than a whole-ROI average. `pixel_count × brightness_mean` gives the integrated L\* above background.
- **`fixed_mask`** (fixed masks enabled and a mask was captured for the ROI): the background/threshold
  value is subtracted from every mask pixel, with no gating, morphology or noise floor. Values can be negative.
- **`whole_roi`** (no background ROI and manual threshold of 0): plain L\* over every ROI pixel.

When no background ROI is set, the manual threshold (default **5.0 L\***) is used as the threshold. The
metadata shows whether it was applied and whether it was the default value.

#### Run metadata sidecar

The sidecar records the app version, a timestamp, the video path/name and fps, the 1-based frame range
(requested and processed, and whether the run was truncated), the threshold mode, manual threshold,
background ROI and percentile, morphological kernel size, noise floor, and, for each ROI, its rectangle,
whether a fixed mask was applied, missing, or dropped because of a size mismatch, its measurement
method and its output files. It also includes column definitions, any export failures, and a warning
if the decoder did not land on the requested start frame.

#### Re-running

Existing files are never overwritten. If any output of a new run would replace an existing file,
every file of that run gets the same `_run2`, `_run3`, … suffix
(for example `…_frames6-8_run2_brightness.csv`).

#### Plots

The shaded band in the plots is the series average ± 1 sample standard deviation across frames. It
is not a per-frame error bar. The background/threshold level is raw L\*, so it is drawn against a
separate right-hand axis.

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
3. Without a background ROI, the manual threshold defaults to 5 L\*. Threshold-selected pixels undergo a morphological opening (erode then dilate), followed by the separate absolute L\* noise floor. Fixed masks bypass these filters; with the manual threshold at 0 and no background ROI, the whole ROI is measured.
4. If a background ROI is set, its brightness (configurable percentile, default 90th) is subtracted per-frame to compensate for lighting drift. If the background ROI is set but unusable (e.g. it lies outside the frame), the analysis stops with an error. It does not fall back to raw values.
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
# Install dev tools (core + all optional extras + dev deps)
pip install -e ".[audio,interactive-plots,dev]"

# Run tests
pytest -q -m "not performance"

# Lint
python -m ruff check ecl_analysis tests

# Coverage
pytest -q -m "not performance" --cov=ecl_analysis --cov-report=term-missing
```
