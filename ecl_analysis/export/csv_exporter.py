"""CSV/plot export orchestration for completed analysis results."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ecl_analysis.analysis.models import AnalysisResult

PlotBuilder = Callable[
    [pd.DataFrame, str, str, int, str, str, Sequence[float], bool, bool],
    Tuple[Optional[str], Optional[str]],
]
ProgressCallback = Callable[[int, int], bool]


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
    plot_failed: bool
    cancelled: bool

    @property
    def no_outputs_produced(self) -> bool:
        """True when the export ran to completion (not cancelled) but wrote no files."""
        return not self.cancelled and not self.out_paths


def save_analysis_outputs(
    analysis_result: AnalysisResult,
    save_dir: str,
    video_path: str,
    analysis_name: str,
    plot_builder: PlotBuilder,
    export_options: Optional[ExportOptions] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> ExportResult:
    """Persist CSV + plots for each ROI and return a UI-ready summary payload."""
    options = export_options or ExportOptions()
    base_video_name = os.path.splitext(os.path.basename(video_path))[0]
    clean_analysis_name = "".join(
        c for c in (analysis_name.strip() or "DefaultAnalysis") if c.isalnum() or c in ("_", "-")
    ).rstrip()

    summary_lines = [f"Analysis Complete ({analysis_result.frames_processed} frames analyzed):"]
    avg_brightness_summary: List[str] = []
    out_paths: List[str] = []
    plot_failed = False
    cancelled = False

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

        frame_numbers = range(
            analysis_result.start_frame,
            analysis_result.start_frame + len(mean_data),
        )
        df = pd.DataFrame(
            {
                "frame": frame_numbers,
                "brightness_mean": mean_data,
                "brightness_median": median_data,
                "blue_mean": blue_mean,
                "blue_median": blue_median,
            }
        )

        avg_mean = np.mean(mean_data)
        avg_median = np.mean(median_data)
        avg_blue_mean = np.mean(blue_mean)
        avg_blue_median = np.mean(blue_median)
        avg_brightness_summary.append(
            f"ROI {actual_roi_idx + 1} L*: {avg_mean:.2f}±{avg_median:.2f}, "
            f"Blue: {avg_blue_mean:.1f}±{avg_blue_median:.1f}"
        )

        base_filename = (
            f"{clean_analysis_name}_{base_video_name}_ROI{actual_roi_idx + 1}_"
            f"frames{analysis_result.start_frame + 1}-{analysis_result.start_frame + len(mean_data)}"
        )
        csv_file = f"{base_filename}_brightness.csv"
        csv_path = os.path.join(save_dir, csv_file)

        try:
            if options.csv:
                df.to_csv(csv_path, index=False)
                out_paths.append(csv_path)
                summary_lines.append(f" - Saved CSV: {csv_file}")

            if options.json:
                json_file = f"{base_filename}_brightness.json"
                json_path = os.path.join(save_dir, json_file)
                payload = {
                    "analysis_name": clean_analysis_name,
                    "video_name": base_video_name,
                    "roi": actual_roi_idx + 1,
                    "frame_range": {
                        "start": analysis_result.start_frame + 1,
                        "end": analysis_result.start_frame + len(mean_data),
                    },
                    "summary": {
                        "brightness_mean": float(avg_mean),
                        "brightness_median": float(avg_median),
                        "blue_mean": float(avg_blue_mean),
                        "blue_median": float(avg_blue_median),
                    },
                    "data": df.to_dict(orient="records"),
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                out_paths.append(json_path)
                summary_lines.append(f" - Saved JSON: {json_file}")

            if options.plot or options.interactive_plot:
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
                if interactive_path:
                    summary_lines.append(f" - Saved Interactive Plot: {os.path.basename(interactive_path)}")
                    out_paths.append(interactive_path)
        except Exception as exc:
            logging.exception("Failed to export ROI %s to %s: %s", actual_roi_idx + 1, save_dir, exc)
            plot_failed = True
            summary_lines.append(f" - FAILED: ROI {actual_roi_idx + 1}")

    return ExportResult(
        summary_lines=summary_lines,
        avg_brightness_summary=avg_brightness_summary,
        out_paths=out_paths,
        plot_failed=plot_failed,
        cancelled=cancelled,
    )
