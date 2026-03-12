from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from ecl_analysis.analysis.models import AnalysisResult
from ecl_analysis.export.csv_exporter import save_analysis_outputs


def _noop_plot_builder(
    _df: pd.DataFrame,
    _base_filename: str,
    _save_dir: str,
    _roi_idx: int,
    _analysis_name: str,
    _base_video_name: str,
    _background_values: Sequence[float],
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
    assert len(export.out_paths) == 2
    csv_path = next(Path(path) for path in export.out_paths if path.endswith("_brightness.csv"))
    metadata_path = next(Path(path) for path in export.out_paths if path.endswith("_analysis_metadata.json"))
    assert metadata_path.exists()

    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["frame", "brightness_mean", "brightness_median", "blue_mean", "blue_median"]
    assert len(df) == 3


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
