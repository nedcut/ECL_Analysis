# iPhone Capture Pipeline Feasibility Review

## Context
The current app analyzes pre-recorded videos and assumes camera settings are stable enough for relative brightness trends.

## What the project already does well
- Computes brightness using CIE L* from each frame and supports background subtraction and noise/morphological filtering.
- Exports reproducible frame-level CSV files and plots.
- Explicitly documents that manual exposure/ISO/white balance lock is required for valid results.

## Current gap vs. requested workflow
Your proposed workflow is:
1. Record on iPhone with exposure lock and stable imaging pipeline.
2. Persist capture settings in metadata.
3. Automatically deliver video into ECL_Analysis for processing.

The repository currently starts analysis from a local file picker / drag-drop and does not include:
- iPhone capture controls.
- In-app metadata ingestion/validation for camera settings.
- An automated watch/import service for incoming files.

## Feasibility assessment
This is feasible and likely worth it if consistency is your top priority.

### Why it is worth doing
- This codebase already depends on consistency of acquisition conditions for scientific validity.
- Most of your measurement error risk is upstream (capture variability), not downstream (analysis code).
- A capture-controlled iPhone flow should reduce false trends caused by auto-exposure, tone mapping, HDR, or AWB drift.

### Practical constraints to account for
- iPhone camera APIs are iOS-native (AVFoundation). A robust capture app is best built as a separate iOS app, not inside this PyQt desktop app.
- iOS may not allow writing arbitrary custom metadata into the container exactly how you want for every codec/profile; often you should also create a sidecar JSON record.
- HEVC/HDR/Dolby Vision defaults can distort analysis unless explicitly disabled.

## Recommended architecture (incremental)

### Phase 1 (highest ROI, low risk): metadata-aware import in this repo
Add import-time validation in ECL_Analysis:
- Parse container metadata via ffprobe/exiftool (codec, fps, dimensions, capture date, color transfer/profile when available).
- Use a lightweight versioned sidecar JSON contract (`schema_version: "1.0"`) from `ecl_analysis/ingest/metadata.py` with fields like:
  - device_model
  - exposure_mode_locked
  - exposure_duration
  - iso
  - white_balance_mode_locked
  - fps
  - resolution
  - hdr_disabled
- Warn, rather than block, when required fields are missing or invalid so legacy videos remain analyzable during the transition.
- Normalize recognized sidecar fields before export so downstream analysis artifacts stay reproducible even when inputs vary in representation.

### Phase 2: automatic ingest
- Add a watched inbox folder (`incoming/`).
- New files with valid sidecar metadata are queued for analysis automatically.
- Save outputs to deterministic folder names tied to capture IDs.

### Phase 3: iPhone acquisition app
- Build a lightweight iOS capture app (Swift + AVFoundation):
  - lock exposure/ISO/white balance/focus
  - disable HDR/night mode/deep tone mapping where possible
  - force fixed FPS and resolution
  - export MOV + sidecar JSON
  - upload directly to shared storage / API endpoint consumed by the desktop pipeline

## Suggested acceptance criteria
- Repeated static-scene captures produce <= X% frame-level brightness variance across runs.
- Pipeline surfaces capture-provenance warnings for any run lacking lock-confirmed metadata.
- Analysis output includes capture settings provenance in summary artifacts, including schema version, validation status, and normalized sidecar fields.

## Bottom line
Yes, this is feasible. It is also strategically aligned with the project’s own measurement assumptions.

Best path: keep this Python analyzer as the analysis engine, and add (1) metadata-gated ingest now, then (2) iPhone capture app integration. That gives you immediate quality gains without a risky full rewrite.
