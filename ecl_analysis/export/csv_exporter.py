"""CSV/plot export orchestration for completed analysis results."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ecl_analysis.analysis.models import (
    MASK_STATUS_APPLIED,
    MASK_STATUS_SHAPE_MISMATCH,
    THRESHOLD_MODE_BACKGROUND_ROI,
    THRESHOLD_MODE_MANUAL,
    THRESHOLD_MODE_NONE,
    AnalysisResult,
)
from ecl_analysis.constants import DEFAULT_MANUAL_THRESHOLD
from ecl_analysis.export.stats import format_std, sample_std

PlotBuilder = Callable[
    [pd.DataFrame, str, str, int, str, str, Sequence[float], bool, bool],
    Tuple[Optional[str], Optional[str]],
]
ProgressCallback = Callable[[int, int], bool]

METADATA_SCHEMA_VERSION = 1

# Every per-ROI file the exporter (and the default plot builder) can write,
# keyed by suffix appended to the ROI base filename. Used to pick a run
# suffix that does not collide with any earlier output.
ROI_OUTPUT_SUFFIXES = (
    "_brightness.csv",
    "_brightness.json",
    "_plot.png",
    "_interactive.html",
)
METADATA_SUFFIX = "_metadata.json"
MAX_RUN_SUFFIX = 10000

MEASUREMENT_METHOD_FIXED_MASK = "fixed_mask"
MEASUREMENT_METHOD_THRESHOLD = "threshold"
MEASUREMENT_METHOD_WHOLE_ROI = "whole_roi"

MEASUREMENT_METHOD_DESCRIPTIONS = {
    MEASUREMENT_METHOD_FIXED_MASK: (
        "Fixed mask: brightness_* = mean/median of (L* - background_l) over every pixel of the "
        "captured mask (plain L* when background_l is empty). No threshold gating, morphology or "
        "noise floor; values can be negative. pixel_count = number of mask pixels."
    ),
    MEASUREMENT_METHOD_THRESHOLD: (
        "Threshold: pixels with L* > background_l are cleaned with a morphological opening, then "
        "restricted to L* > noise_floor_threshold when that is > 0. brightness_* = mean/median of "
        "(L* - background_l) over only those pixels, so values are >= 0 and describe lit pixels "
        "only (biased upward relative to a whole-ROI mean). 0.0 with pixel_count 0 when no pixel "
        "passes. pixel_count * brightness_mean = integrated L* above background."
    ),
    MEASUREMENT_METHOD_WHOLE_ROI: (
        "Whole ROI: no background/threshold active. brightness_* = mean/median of L* over every "
        "ROI pixel. The noise floor is not applied in this mode. pixel_count = ROI area."
    ),
}

COLUMN_DEFINITIONS = {
    "frame": "1-based video frame number (same numbering as the UI, filenames and frame_range).",
    "brightness_mean": "Mean CIE L* (0-100) for this ROI and frame; see the ROI's measurement_method.",
    "brightness_median": "Median CIE L* (0-100) over the same pixels as brightness_mean.",
    "blue_mean": (
        "Mean raw blue channel (0-255) over the same pixels as brightness_mean. "
        "Blue is never background-subtracted."
    ),
    "blue_median": "Median raw blue channel (0-255) over the same pixels as brightness_mean.",
    "pixel_count": "Number of pixels contributing to brightness_mean/median for this frame.",
    "background_l": (
        "Background/threshold L* applied to this frame: background-ROI percentile in "
        "background_roi mode, the manual threshold in manual_threshold mode, empty in none mode."
    ),
    "time_s": "Video timestamp of the frame in seconds, (frame - 1) / fps. Empty when fps is unknown.",
}


@dataclass(frozen=True)
class ExportOptions:
    """Output formats to write after analysis."""

    csv: bool = True
    json: bool = False
    plot: bool = True
    interactive_plot: bool = True

    def has_outputs(self) -> bool:
        return self.csv or self.json or self.plot or self.interactive_plot


@dataclass
class ExportResult:
    """Structured summary returned by CSV/plot export."""

    summary_lines: List[str]
    avg_brightness_summary: List[str]
    out_paths: List[str]
    # True when any CSV, JSON, plot or metadata write raised for any ROI.
    export_failed: bool
    cancelled: bool
    # 1-based ROI numbers whose exports failed.
    failed_rois: List[int] = field(default_factory=list)
    metadata_path: Optional[str] = None

    @property
    def no_outputs_produced(self) -> bool:
        """True when the export ran to completion (not cancelled) but wrote no files."""
        return not self.cancelled and not self.out_paths


def _app_version() -> str:
    try:
        return importlib_metadata.version("ecl-analysis")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _json_safe(value: Any) -> Any:
    """Convert NumPy/pandas scalars and non-finite floats into JSON-safe values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def _aligned(values: Sequence[Any], length: int) -> List[Any]:
    """Return values as a list of exactly `length` items (None-padded/truncated)."""
    values = list(values or [])
    if len(values) >= length:
        return values[:length]
    return values + [None] * (length - len(values))


def _measurement_method(threshold_mode: Optional[str], mask_status: Optional[str]) -> str:
    if mask_status == MASK_STATUS_APPLIED:
        return MEASUREMENT_METHOD_FIXED_MASK
    if threshold_mode in (THRESHOLD_MODE_BACKGROUND_ROI, THRESHOLD_MODE_MANUAL):
        return MEASUREMENT_METHOD_THRESHOLD
    return MEASUREMENT_METHOD_WHOLE_ROI


def _choose_run_suffix(save_dir: str, roi_bases: Sequence[str], metadata_base: str) -> str:
    """Return '' or '_runN' so that no file of this run overwrites an existing file."""
    for run_number in range(1, MAX_RUN_SUFFIX):
        suffix = "" if run_number == 1 else f"_run{run_number}"
        candidates = [
            os.path.join(save_dir, f"{base}{suffix}{ext}") for base in roi_bases for ext in ROI_OUTPUT_SUFFIXES
        ]
        candidates.append(os.path.join(save_dir, f"{metadata_base}{suffix}{METADATA_SUFFIX}"))
        if not any(os.path.exists(path) for path in candidates):
            return suffix
    raise RuntimeError(f"Could not find a free output filename in {save_dir}")


def _build_frame_dataframe(
    analysis_result: AnalysisResult,
    data_idx: int,
    threshold_mode: Optional[str],
) -> pd.DataFrame:
    mean_data = analysis_result.brightness_mean_data[data_idx]
    n_frames = len(mean_data)
    first_frame = analysis_result.start_frame + 1  # exported frames are 1-based
    frame_numbers = list(range(first_frame, first_frame + n_frames))

    if data_idx < len(analysis_result.pixel_count_data):
        pixel_counts = _aligned(analysis_result.pixel_count_data[data_idx], n_frames)
    else:
        pixel_counts = [None] * n_frames

    if threshold_mode == THRESHOLD_MODE_NONE:
        background_l: List[Any] = [None] * n_frames
    else:
        background_l = _aligned(analysis_result.background_values_per_frame, n_frames)

    fps = analysis_result.fps
    if fps is not None and fps > 0:
        time_s: List[Any] = [(frame - 1) / fps for frame in frame_numbers]
    else:
        time_s = [None] * n_frames

    return pd.DataFrame(
        {
            "frame": frame_numbers,
            "brightness_mean": mean_data,
            "brightness_median": analysis_result.brightness_median_data[data_idx],
            "blue_mean": analysis_result.blue_mean_data[data_idx],
            "blue_median": analysis_result.blue_median_data[data_idx],
            "pixel_count": pixel_counts,
            "background_l": background_l,
            "time_s": time_s,
        }
    )


def _build_metadata(
    analysis_result: AnalysisResult,
    video_path: str,
    analysis_name: str,
    frame_start: int,
    frame_end: int,
    roi_entries: List[Dict[str, Any]],
    failures: List[str],
    cancelled: bool,
) -> Dict[str, Any]:
    req = analysis_result.request
    threshold_mode = req.threshold_mode if req is not None else None
    manual_threshold = float(req.manual_threshold) if req is not None else None
    background_roi_idx = req.background_roi_idx if req is not None else None

    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "app": {"name": "ecl-analysis", "version": _app_version()},
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "analysis_name": analysis_name,
        "video": {
            "path": os.path.abspath(video_path),
            "name": os.path.basename(video_path),
            "fps": analysis_result.fps,
        },
        "frame_range": {
            "numbering": "1-based",
            "start": frame_start,
            "end": frame_end,
            "requested_start": analysis_result.start_frame + 1,
            "requested_end": analysis_result.end_frame + 1,
            "frames_processed": analysis_result.frames_processed,
            "frames_requested": analysis_result.total_frames,
            "truncated": analysis_result.truncated,
        },
        "seek_warning": analysis_result.seek_warning,
        "settings": {
            "threshold_mode": threshold_mode,
            "manual_threshold": manual_threshold,
            "manual_threshold_applied": threshold_mode == THRESHOLD_MODE_MANUAL,
            "manual_threshold_is_default": (
                manual_threshold == DEFAULT_MANUAL_THRESHOLD if manual_threshold is not None else None
            ),
            "background_roi": background_roi_idx + 1 if background_roi_idx is not None else None,
            "background_percentile": float(req.background_percentile) if req is not None else None,
            "morphological_kernel_size": int(req.morphological_kernel_size) if req is not None else None,
            "noise_floor_threshold": float(req.noise_floor_threshold) if req is not None else None,
            "use_fixed_mask": bool(req.use_fixed_mask) if req is not None else None,
        },
        "rois": roi_entries,
        "measurement_methods": MEASUREMENT_METHOD_DESCRIPTIONS,
        "columns": COLUMN_DEFINITIONS,
        "export_failures": failures,
        "cancelled": cancelled,
    }


def save_analysis_outputs(
    analysis_result: AnalysisResult,
    save_dir: str,
    video_path: str,
    analysis_name: str,
    plot_builder: PlotBuilder,
    export_options: Optional[ExportOptions] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> ExportResult:
    """Persist CSV/JSON/plots for each ROI plus a run metadata sidecar.

    Never overwrites existing files: when any output of this run would collide
    with an existing file, every file of the run gets the same ``_runN`` suffix.
    """
    options = export_options or ExportOptions()
    base_video_name = os.path.splitext(os.path.basename(video_path))[0]
    clean_analysis_name = "".join(
        c for c in (analysis_name.strip() or "DefaultAnalysis") if c.isalnum() or c in ("_", "-")
    ).rstrip()
    req = analysis_result.request
    threshold_mode = req.threshold_mode if req is not None else None

    summary_lines = [f"Analysis Complete ({analysis_result.frames_processed} frames analyzed):"]
    if analysis_result.truncated:
        summary_lines.append(
            " - WARNING: Analysis truncated by early end-of-file "
            f"({analysis_result.frames_processed} of {analysis_result.total_frames} frames processed)."
        )
    if analysis_result.seek_warning:
        summary_lines.append(f" - WARNING: {analysis_result.seek_warning}")
    for roi_idx, status in sorted(analysis_result.mask_status.items()):
        if status == MASK_STATUS_SHAPE_MISMATCH:
            summary_lines.append(
                f" - WARNING: Fixed mask for ROI {roi_idx + 1} did not match the ROI size and was "
                "not applied (threshold method used instead)."
            )
    avg_brightness_summary: List[str] = []
    out_paths: List[str] = []
    failures: List[str] = []
    failed_rois: List[int] = []
    roi_entries: List[Dict[str, Any]] = []
    cancelled = False

    n_frames = max((len(data) for data in analysis_result.brightness_mean_data), default=0)
    frame_start = analysis_result.start_frame + 1
    frame_end = analysis_result.start_frame + n_frames
    frames_tag = f"frames{frame_start}-{frame_end}"
    roi_bases = {
        data_idx: f"{clean_analysis_name}_{base_video_name}_ROI{actual_roi_idx + 1}_{frames_tag}"
        for data_idx, actual_roi_idx in enumerate(analysis_result.non_background_rois)
        if data_idx < len(analysis_result.brightness_mean_data)
        and analysis_result.brightness_mean_data[data_idx]
    }
    metadata_base = f"{clean_analysis_name}_{base_video_name}_{frames_tag}"
    run_suffix = _choose_run_suffix(save_dir, list(roi_bases.values()), metadata_base)
    if run_suffix:
        summary_lines.append(
            f" - Existing outputs found; this run's files use the suffix '{run_suffix}'."
        )

    total_rois = len(analysis_result.brightness_mean_data)
    for data_idx in range(total_rois):
        if progress_callback is not None and not progress_callback(data_idx + 1, total_rois):
            cancelled = True
            break

        actual_roi_idx = analysis_result.non_background_rois[data_idx]
        mean_data = analysis_result.brightness_mean_data[data_idx]
        median_data = analysis_result.brightness_median_data[data_idx]
        blue_mean = analysis_result.blue_mean_data[data_idx]
        blue_median = analysis_result.blue_median_data[data_idx]

        if not mean_data:
            continue

        df = _build_frame_dataframe(analysis_result, data_idx, threshold_mode)

        avg_mean = np.mean(mean_data)
        std_mean = sample_std(mean_data)
        avg_median = np.mean(median_data)
        avg_blue_mean = np.mean(blue_mean)
        std_blue_mean = sample_std(blue_mean)
        avg_blue_median = np.mean(blue_median)
        avg_brightness_summary.append(
            f"ROI {actual_roi_idx + 1} L*: {avg_mean:.2f}±{format_std(std_mean, 2)} (mean±SD), "
            f"median {avg_median:.2f}; "
            f"Blue: {avg_blue_mean:.1f}±{format_std(std_blue_mean, 1)} (mean±SD)"
        )

        mask_status = analysis_result.mask_status.get(actual_roi_idx)
        method = _measurement_method(threshold_mode, mask_status)
        roi_rect = None
        if req is not None and actual_roi_idx < len(req.rects):
            (x1, y1), (x2, y2) = req.rects[actual_roi_idx]
            roi_rect = [[int(x1), int(y1)], [int(x2), int(y2)]]
        roi_entry: Dict[str, Any] = {
            "roi": actual_roi_idx + 1,
            "rect": roi_rect,
            "fixed_mask_requested": bool(req.use_fixed_mask) if req is not None else None,
            "fixed_mask_status": mask_status,
            "measurement_method": method,
            "files": [],
        }
        roi_entries.append(roi_entry)

        base_filename = f"{roi_bases[data_idx]}{run_suffix}"
        csv_file = f"{base_filename}_brightness.csv"
        csv_path = os.path.join(save_dir, csv_file)

        stage = "CSV"
        try:
            if options.csv:
                df.to_csv(csv_path, index=False)
                out_paths.append(csv_path)
                roi_entry["files"].append(csv_file)
                summary_lines.append(f" - Saved CSV: {csv_file}")

            if options.json:
                stage = "JSON"
                json_file = f"{base_filename}_brightness.json"
                json_path = os.path.join(save_dir, json_file)
                payload = {
                    "analysis_name": clean_analysis_name,
                    "video_name": base_video_name,
                    "roi": actual_roi_idx + 1,
                    "measurement_method": method,
                    "frame_range": {
                        "start": frame_start,
                        "end": analysis_result.start_frame + len(mean_data),
                    },
                    "fps": analysis_result.fps,
                    "summary": {
                        "brightness_mean": float(avg_mean),
                        "brightness_median": float(avg_median),
                        "blue_mean": float(avg_blue_mean),
                        "blue_median": float(avg_blue_median),
                    },
                    "data": [
                        {key: _json_safe(value) for key, value in record.items()}
                        for record in df.to_dict(orient="records")
                    ],
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                out_paths.append(json_path)
                roi_entry["files"].append(json_file)
                summary_lines.append(f" - Saved JSON: {json_file}")

            if options.plot or options.interactive_plot:
                stage = "plot"
                png_path, interactive_path = plot_builder(
                    df,
                    base_filename,
                    save_dir,
                    actual_roi_idx,
                    clean_analysis_name,
                    base_video_name,
                    analysis_result.background_values_per_frame,
                    options.plot,
                    options.interactive_plot,
                )
                if png_path:
                    summary_lines.append(f" - Saved Plot: {os.path.basename(png_path)}")
                    out_paths.append(png_path)
                    roi_entry["files"].append(os.path.basename(png_path))
                if interactive_path:
                    summary_lines.append(f" - Saved Interactive Plot: {os.path.basename(interactive_path)}")
                    out_paths.append(interactive_path)
                    roi_entry["files"].append(os.path.basename(interactive_path))
        except Exception as exc:
            logging.exception("Failed to export ROI %s (%s) to %s: %s", actual_roi_idx + 1, stage, save_dir, exc)
            failed_rois.append(actual_roi_idx + 1)
            failures.append(f"ROI {actual_roi_idx + 1} {stage}: {exc}")
            summary_lines.append(f" - FAILED: ROI {actual_roi_idx + 1} ({stage} export: {exc})")

    metadata_path: Optional[str] = None
    if out_paths:
        metadata_file = f"{metadata_base}{run_suffix}{METADATA_SUFFIX}"
        candidate_path = os.path.join(save_dir, metadata_file)
        try:
            metadata = _build_metadata(
                analysis_result,
                video_path,
                clean_analysis_name,
                frame_start,
                frame_end,
                roi_entries,
                failures,
                cancelled,
            )
            with open(candidate_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=_json_safe)
            metadata_path = candidate_path
            out_paths.append(candidate_path)
            summary_lines.append(f" - Saved run metadata: {metadata_file}")
        except Exception as exc:
            logging.exception("Failed to write run metadata to %s: %s", candidate_path, exc)
            failures.append(f"metadata: {exc}")
            summary_lines.append(f" - FAILED: run metadata ({exc})")

    return ExportResult(
        summary_lines=summary_lines,
        avg_brightness_summary=avg_brightness_summary,
        out_paths=out_paths,
        export_failed=bool(failures),
        cancelled=cancelled,
        failed_rois=failed_rois,
        metadata_path=metadata_path,
    )
