from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from ecl_analysis.analysis.models import AnalysisResult
from ecl_analysis.export.csv_exporter import ExportOptions, save_analysis_outputs


def _noop_plot_builder(
    _df: pd.DataFrame,
    _base_filename: str,
    _save_dir: str,
    _roi_idx: int,
    _analysis_name: str,
    _base_video_name: str,
    _background_values: Sequence[float],
    _generate_static: bool,
    _generate_interactive: bool,
) -> Tuple[Optional[str], Optional[str]]:
    return None, None


def test_save_analysis_outputs_writes_csv_and_summary(tmp_path: Path):
    result = AnalysisResult(
        brightness_mean_data=[[1.0, 2.0, 3.0]],
        brightness_median_data=[[0.5, 1.5, 2.5]],
        blue_mean_data=[[10.0, 11.0, 12.0]],
        blue_median_data=[[9.0, 10.0, 11.0]],
        background_values_per_frame=[0.0, 0.0, 0.0],
        frames_processed=3,
        total_frames=3,
        non_background_rois=[0],
        elapsed_seconds=0.1,
        start_frame=5,
        end_frame=7,
    )

    export = save_analysis_outputs(
        analysis_result=result,
        save_dir=str(tmp_path),
        video_path="/tmp/input.mp4",
        analysis_name="Demo",
        plot_builder=_noop_plot_builder,
    )

    assert export.cancelled is False
    assert export.plot_failed is False
    assert any("Saved CSV:" in line for line in export.summary_lines)
    assert len(export.out_paths) == 1
    assert export.out_paths[0].endswith("_brightness.csv")

    csv_path = Path(export.out_paths[0])
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["frame", "brightness_mean", "brightness_median", "blue_mean", "blue_median"]
    assert len(df) == 3


def test_save_analysis_outputs_writes_json_when_selected(tmp_path: Path):
    result = AnalysisResult(
        brightness_mean_data=[[1.0, 2.0, 3.0]],
        brightness_median_data=[[0.5, 1.5, 2.5]],
        blue_mean_data=[[10.0, 11.0, 12.0]],
        blue_median_data=[[9.0, 10.0, 11.0]],
        background_values_per_frame=[0.0, 0.0, 0.0],
        frames_processed=3,
        total_frames=3,
        non_background_rois=[0],
        elapsed_seconds=0.1,
        start_frame=5,
        end_frame=7,
    )

    export = save_analysis_outputs(
        analysis_result=result,
        save_dir=str(tmp_path),
        video_path="/tmp/input.mp4",
        analysis_name="Demo",
        plot_builder=_noop_plot_builder,
        export_options=ExportOptions(csv=False, json=True, plot=False, interactive_plot=False),
    )

    assert export.plot_failed is False
    assert any("Saved JSON:" in line for line in export.summary_lines)
    assert len(export.out_paths) == 1
    json_path = Path(export.out_paths[0])
    assert json_path.exists()
    assert json_path.suffix == ".json"
    payload = json_path.read_text()
    assert '"analysis_name": "Demo"' in payload
    assert '"frame": 5' in payload


def test_save_analysis_outputs_supports_cancellation(tmp_path: Path):
    result = AnalysisResult(
        brightness_mean_data=[[1.0], [2.0]],
        brightness_median_data=[[1.0], [2.0]],
        blue_mean_data=[[1.0], [2.0]],
        blue_median_data=[[1.0], [2.0]],
        background_values_per_frame=[0.0],
        frames_processed=2,
        total_frames=2,
        non_background_rois=[0, 1],
        elapsed_seconds=0.1,
        start_frame=0,
        end_frame=1,
    )

    calls = {"count": 0}

    def _cancel_on_first(_current: int, _total: int) -> bool:
        calls["count"] += 1
        return calls["count"] < 2

    export = save_analysis_outputs(
        analysis_result=result,
        save_dir=str(tmp_path),
        video_path="/tmp/input.mp4",
        analysis_name="Demo",
        plot_builder=_noop_plot_builder,
        progress_callback=_cancel_on_first,
    )

    assert export.cancelled is True


def test_export_result_no_outputs_produced_when_plot_builder_returns_nothing(tmp_path: Path):
    """Selecting only the interactive plot with an unavailable plot builder should not look like success."""
    result = AnalysisResult(
        brightness_mean_data=[[1.0, 2.0, 3.0]],
        brightness_median_data=[[0.5, 1.5, 2.5]],
        blue_mean_data=[[10.0, 11.0, 12.0]],
        blue_median_data=[[9.0, 10.0, 11.0]],
        background_values_per_frame=[0.0, 0.0, 0.0],
        frames_processed=3,
        total_frames=3,
        non_background_rois=[0],
        elapsed_seconds=0.1,
        start_frame=5,
        end_frame=7,
    )

    export = save_analysis_outputs(
        analysis_result=result,
        save_dir=str(tmp_path),
        video_path="/tmp/input.mp4",
        analysis_name="Demo",
        plot_builder=_noop_plot_builder,
        export_options=ExportOptions(csv=False, json=False, plot=False, interactive_plot=True),
    )

    assert export.out_paths == []
    assert export.cancelled is False
    assert export.no_outputs_produced is True


def test_export_result_no_outputs_produced_false_when_files_written(tmp_path: Path):
    result = AnalysisResult(
        brightness_mean_data=[[1.0, 2.0, 3.0]],
        brightness_median_data=[[0.5, 1.5, 2.5]],
        blue_mean_data=[[10.0, 11.0, 12.0]],
        blue_median_data=[[9.0, 10.0, 11.0]],
        background_values_per_frame=[0.0, 0.0, 0.0],
        frames_processed=3,
        total_frames=3,
        non_background_rois=[0],
        elapsed_seconds=0.1,
        start_frame=5,
        end_frame=7,
    )

    export = save_analysis_outputs(
        analysis_result=result,
        save_dir=str(tmp_path),
        video_path="/tmp/input.mp4",
        analysis_name="Demo",
        plot_builder=_noop_plot_builder,
    )

    assert export.no_outputs_produced is False
