# Capture Metadata Sidecar Schema

The analyzer accepts an optional sidecar JSON file next to each video:

- Video: `experiment_01.mov`
- Sidecar: `experiment_01.capture.json`

The current lightweight schema authority is:

- `schema_version: "1.0"`
- Schema contract source of truth: `ecl_analysis/ingest/metadata.py`

This is intentionally versioned but non-blocking during the transition from legacy videos to the dedicated iPhone capture app. Missing or invalid metadata should warn, not block analysis.

## Required fields for schema `1.0`

```json
{
  "schema_version": "1.0",
  "device_model": "iPhone 15 Pro",
  "capture_id": "8A0F0A5A-2A79-4D8C-9C2A-0CCF9F9368EA",
  "recorded_at": "2026-04-01T10:15:30Z",
  "app_version": "0.1.0",
  "ios_version": "iOS 26.0",
  "video_codec": "h264",
  "color_space": "sdr",
  "exposure_mode_locked": true,
  "exposure_duration": 0.0333333333,
  "iso": 80,
  "white_balance_mode_locked": true,
  "fps": 30,
  "resolution": "1920x1080",
  "hdr_disabled": true
}
```

## Validation behavior

- Missing sidecar: warning-only in the UI; analysis still proceeds.
- Missing `schema_version`: warning; validator assumes compatibility with schema `1.0` and marks `schema_version_assumed: true` in exported provenance.
- Unknown `schema_version`: warning; validator performs best-effort validation against current fields.
- Missing required acquisition fields: warning-only at load time, but surfaced as validation errors in exported provenance.
- Unknown fields are retained in provenance as `unrecognized_fields` so schema drift is visible without blocking ingest.

## Export behavior

Analysis metadata exports now include:

- `capture_metadata_validation`: whether the sidecar passed validation plus any warnings/errors
- `capture_metadata`: normalized capture provenance when a sidecar is present
- `capture_provenance`: grouped export view that carries both the normalized metadata and the validation record used for the run

That contract is the boundary the iPhone capture app should target.
