# Metadata Ingest Execution Plan

## Goal
Improve acquisition consistency and provenance in the desktop analyzer while keeping legacy videos analyzable during the transition to a dedicated iPhone capture app.

## Decisions
- Capture metadata validation is warning-first, not hard-blocking.
- The schema authority is a lightweight versioned sidecar contract with `schema_version: "1.0"`.
- The Python desktop app remains the analysis engine.
- The iPhone capture app should live in a separate repository and can start as a minimal AVFoundation MVP.

## Phase Status

### Phase 1: metadata-aware import in this repo
Status: in progress

Implemented:
- Sidecar schema contract and validator in `ecl_analysis/ingest/metadata.py`
- UI load-time metadata status in `ecl_analysis/video_analyzer.py`
- Provenance export fields in analysis metadata outputs
- Tests covering validation behavior and metadata export wiring

Remaining:
- Optional container-level metadata parsing (`ffprobe` / `exiftool`) to cross-check sidecar claims
- More explicit UI surfacing of validation warnings/details beyond the status line

### Phase 2: automatic ingest
Status: in progress

Implemented:
- Inbox ingest script in `tools/ingest_capture_inbox.py`
- Deterministic capture output folders using `capture_id` when present
- Per-capture ingest summaries and optional archive behavior
- Manifest-driven optional auto-analysis flow

Remaining:
- Decide where the watched inbox should live in real deployments
- Add any daemon/service wrapper if continuous unattended ingest is needed

### Phase 3: iPhone capture app
Status: not started in this repository

Planned:
- Minimal Swift / AVFoundation capture app in a separate repository
- Fixed capture settings, sidecar JSON export, and transfer into the desktop ingest path

## Near-Term Next Steps
1. Commit the Phase 1 and Phase 2 desktop-side ingest work.
2. Decide whether container metadata cross-checking is required before starting the iPhone app.
3. Create a separate repository for the iPhone capture MVP.
