import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from ecl_analysis.analysis.models import AnalysisRequest, AnalysisResult
from ecl_analysis.export.csv_exporter import ExportOptions, save_analysis_outputs

EXPECTED_COLUMNS = [
    "frame",
    "brightness_mean",
    "brightness_median",
    "blue_mean",
    "blue_median",
    "pixel_count",
    "background_l",
    "time_s",
]


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


def _request(**overrides) -> AnalysisRequest:
    kwargs = dict(
        video_path="/tmp/input.mp4",
        rects=[((0, 0), (10, 10)), ((10, 0), (20, 10))],
        background_roi_idx=None,
        start_frame=5,
        end_frame=7,
        use_fixed_mask=False,
        fixed_roi_masks=[],
        background_percentile=90.0,
        morphological_kernel_size=3,
        noise_floor_threshold=0.0,
        manual_threshold=0.0,
    )
    kwargs.update(overrides)
    return AnalysisRequest(**kwargs)


def _result(**overrides) -> AnalysisResult:
    kwargs = dict(
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
    kwargs.update(overrides)
    return AnalysisResult(**kwargs)


def _export(tmp_path: Path, result: AnalysisResult, **kwargs):
    kwargs.setdefault("plot_builder", _noop_plot_builder)
    return save_analysis_outputs(
        analysis_result=result,
        save_dir=str(tmp_path),
        video_path="/tmp/input.mp4",
        analysis_name="Demo",
        **kwargs,
    )


def test_save_analysis_outputs_writes_csv_and_summary(tmp_path: Path):
    export = _export(tmp_path, _result())

    assert export.cancelled is False
    assert export.export_failed is False
    assert any("Saved CSV:" in line for line in export.summary_lines)
    csv_paths = [p for p in export.out_paths if p.endswith("_brightness.csv")]
    assert len(csv_paths) == 1
    assert export.metadata_path is not None
    assert export.metadata_path in export.out_paths

    csv_path = Path(csv_paths[0])
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 3


def test_csv_frame_column_is_one_based_and_matches_filename(tmp_path: Path):
    """start_frame is a 0-based index internally; exported frames are 1-based like the UI."""
    export = _export(tmp_path, _result(start_frame=5, end_frame=7))

    csv_path = next(p for p in export.out_paths if p.endswith(".csv"))
    assert "frames6-8" in Path(csv_path).name
    df = pd.read_csv(csv_path)
    assert df["frame"].tolist() == [6, 7, 8]


def test_csv_time_column_uses_fps(tmp_path: Path):
    export = _export(tmp_path, _result(start_frame=5, end_frame=7, fps=10.0))

    df = pd.read_csv(next(p for p in export.out_paths if p.endswith(".csv")))
    # time_s is the video timestamp of the frame: (frame - 1) / fps
    assert df["time_s"].tolist() == pytest.approx([0.5, 0.6, 0.7])


def test_csv_time_column_blank_without_fps(tmp_path: Path):
    export = _export(tmp_path, _result(fps=None))

    df = pd.read_csv(next(p for p in export.out_paths if p.endswith(".csv")))
    assert df["time_s"].isna().all()


def test_csv_pixel_count_and_background_columns(tmp_path: Path):
    result = _result(
        pixel_count_data=[[100, 40, 0]],
        background_values_per_frame=[5.0, 5.0, 5.0],
        request=_request(manual_threshold=5.0),
    )
    export = _export(tmp_path, result)

    df = pd.read_csv(next(p for p in export.out_paths if p.endswith(".csv")))
    assert df["pixel_count"].tolist() == [100, 40, 0]
    assert df["background_l"].tolist() == [5.0, 5.0, 5.0]


def test_csv_background_column_blank_when_no_threshold(tmp_path: Path):
    result = _result(
        pixel_count_data=[[100, 100, 100]],
        background_values_per_frame=[0.0, 0.0, 0.0],
        request=_request(manual_threshold=0.0),
    )
    export = _export(tmp_path, result)

    df = pd.read_csv(next(p for p in export.out_paths if p.endswith(".csv")))
    assert df["background_l"].isna().all()


def test_avg_brightness_summary_uses_sample_std_of_mean_series(tmp_path: Path):
    mean_data = [1.0, 2.0, 3.0, 10.0]
    median_data = [0.5, 1.5, 2.5, 3.5]
    result = _result(
        brightness_mean_data=[mean_data],
        brightness_median_data=[median_data],
        blue_mean_data=[[10.0, 11.0, 12.0, 13.0]],
        blue_median_data=[[9.0, 10.0, 11.0, 12.0]],
        background_values_per_frame=[0.0, 0.0, 0.0, 0.0],
        frames_processed=4,
        total_frames=4,
        start_frame=0,
        end_frame=3,
    )

    export = _export(tmp_path, result)

    assert len(export.avg_brightness_summary) == 1
    summary = export.avg_brightness_summary[0]

    expected_mean = np.mean(mean_data)
    expected_std = np.std(mean_data, ddof=1)
    expected_median = np.mean(median_data)

    # The ± value must be the sample std (ddof=1) of the mean series.
    assert f"{expected_mean:.2f}±{expected_std:.2f}" in summary
    assert f"{expected_mean:.2f}±{np.std(mean_data):.2f}" not in summary
    assert f"{expected_median:.2f}" in summary
    assert f"{expected_mean:.2f}±{expected_median:.2f}" not in summary


def test_avg_brightness_summary_single_frame_std_is_na(tmp_path: Path):
    result = _result(
        brightness_mean_data=[[4.0]],
        brightness_median_data=[[4.0]],
        blue_mean_data=[[1.0]],
        blue_median_data=[[1.0]],
        background_values_per_frame=[0.0],
        frames_processed=1,
        total_frames=1,
        start_frame=0,
        end_frame=0,
    )
    export = _export(tmp_path, result)
    assert "4.00±n/a" in export.avg_brightness_summary[0]


def test_save_analysis_outputs_writes_json_when_selected(tmp_path: Path):
    export = _export(
        tmp_path,
        _result(fps=None),
        export_options=ExportOptions(csv=False, json=True, plot=False, interactive_plot=False),
    )

    assert export.export_failed is False
    assert any("Saved JSON:" in line for line in export.summary_lines)
    json_paths = [p for p in export.out_paths if p.endswith("_brightness.json")]
    assert len(json_paths) == 1
    json_path = Path(json_paths[0])
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["analysis_name"] == "Demo"
    assert payload["frame_range"] == {"start": 6, "end": 8}
    assert payload["data"][0]["frame"] == 6
    # Missing values must be valid JSON nulls, not NaN.
    assert payload["data"][0]["time_s"] is None
    assert "NaN" not in json_path.read_text()


def test_metadata_sidecar_records_provenance(tmp_path: Path):
    request = _request(
        rects=[((0, 0), (10, 10)), ((10, 0), (20, 10)), ((20, 0), (30, 10))],
        use_fixed_mask=True,
        manual_threshold=5.0,
        noise_floor_threshold=2.0,
        morphological_kernel_size=5,
        background_percentile=75.0,
    )
    result = _result(
        brightness_mean_data=[[1.0, 2.0, 3.0]] * 3,
        brightness_median_data=[[1.0, 2.0, 3.0]] * 3,
        blue_mean_data=[[1.0, 2.0, 3.0]] * 3,
        blue_median_data=[[1.0, 2.0, 3.0]] * 3,
        pixel_count_data=[[1, 1, 1]] * 3,
        non_background_rois=[0, 1, 2],
        background_values_per_frame=[5.0, 5.0, 5.0],
        fps=30.0,
        request=request,
        mask_status={0: "applied", 1: "dropped_shape_mismatch", 2: "missing"},
        seek_warning="seek drift",
    )

    export = _export(tmp_path, result)

    assert export.metadata_path is not None
    meta = json.loads(Path(export.metadata_path).read_text())
    assert Path(export.metadata_path).name == "Demo_input_frames6-8_metadata.json"
    assert meta["app"]["version"]
    assert meta["created_at"]
    assert meta["video"] == {"path": "/tmp/input.mp4", "name": "input.mp4", "fps": 30.0}
    assert meta["frame_range"]["numbering"] == "1-based"
    assert (meta["frame_range"]["start"], meta["frame_range"]["end"]) == (6, 8)
    assert meta["seek_warning"] == "seek drift"

    settings = meta["settings"]
    assert settings["threshold_mode"] == "manual_threshold"
    assert settings["manual_threshold"] == 5.0
    assert settings["manual_threshold_applied"] is True
    assert settings["manual_threshold_is_default"] is True
    assert settings["background_roi"] is None
    assert settings["background_percentile"] == 75.0
    assert settings["morphological_kernel_size"] == 5
    assert settings["noise_floor_threshold"] == 2.0
    assert settings["use_fixed_mask"] is True

    rois = {entry["roi"]: entry for entry in meta["rois"]}
    assert rois[1]["fixed_mask_status"] == "applied"
    assert rois[1]["measurement_method"] == "fixed_mask"
    assert rois[2]["fixed_mask_status"] == "dropped_shape_mismatch"
    assert rois[2]["measurement_method"] == "threshold"
    assert rois[3]["fixed_mask_status"] == "missing"
    assert rois[1]["rect"] == [[0, 0], [10, 10]]
    assert rois[1]["files"] == ["Demo_input_ROI1_frames6-8_brightness.csv"]

    for column in ("frame", "pixel_count", "background_l", "time_s", "blue_mean"):
        assert column in meta["columns"]
    assert set(meta["measurement_methods"]) == {"fixed_mask", "threshold", "whole_roi"}
    assert any("did not match the ROI size" in line for line in export.summary_lines)
    assert any("seek drift" in line for line in export.summary_lines)


def test_metadata_background_roi_mode(tmp_path: Path):
    result = _result(
        non_background_rois=[0],
        background_values_per_frame=[12.0, 13.0, 14.0],
        request=_request(background_roi_idx=1, manual_threshold=5.0),
    )
    export = _export(tmp_path, result)

    meta = json.loads(Path(export.metadata_path).read_text())
    assert meta["settings"]["threshold_mode"] == "background_roi"
    assert meta["settings"]["background_roi"] == 2
    assert meta["settings"]["manual_threshold_applied"] is False
    df = pd.read_csv(next(p for p in export.out_paths if p.endswith(".csv")))
    assert df["background_l"].tolist() == [12.0, 13.0, 14.0]


def _written_plot_builder(df, base_filename, save_dir, *_args):
    path = Path(save_dir) / f"{base_filename}_plot.png"
    path.write_bytes(b"png")
    return str(path), None


def test_rerun_does_not_overwrite_and_suffixes_all_files(tmp_path: Path):
    result = _result(
        brightness_mean_data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        brightness_median_data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        blue_mean_data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        blue_median_data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        non_background_rois=[0, 1],
    )
    first = _export(tmp_path, result, plot_builder=_written_plot_builder)
    first_contents = {p: Path(p).read_bytes() for p in first.out_paths}

    second = _export(tmp_path, result, plot_builder=_written_plot_builder)
    third = _export(tmp_path, result, plot_builder=_written_plot_builder)

    # Earlier outputs are untouched.
    for path, content in first_contents.items():
        assert Path(path).read_bytes() == content

    assert not any("_run" in Path(p).name for p in first.out_paths)
    assert len(second.out_paths) == len(first.out_paths) == 5  # 2 CSV + 2 PNG + metadata
    assert all("_run2" in Path(p).name for p in second.out_paths)
    assert all("_run3" in Path(p).name for p in third.out_paths)
    assert set(first.out_paths).isdisjoint(second.out_paths)
    assert any("_run2" in line for line in second.summary_lines)


def test_any_existing_run_file_triggers_suffix(tmp_path: Path):
    """A leftover plot from an earlier run must push the whole new run to _run2."""
    (tmp_path / "Demo_input_ROI1_frames6-8_interactive.html").write_text("old")

    export = _export(tmp_path, _result())

    names = sorted(Path(p).name for p in export.out_paths)
    assert names == [
        "Demo_input_ROI1_frames6-8_run2_brightness.csv",
        "Demo_input_frames6-8_run2_metadata.json",
    ]
    assert (tmp_path / "Demo_input_ROI1_frames6-8_interactive.html").read_text() == "old"


def test_plot_failure_reported_as_export_failure(tmp_path: Path):
    def _failing_plot_builder(*_args):
        raise RuntimeError("plot boom")

    export = _export(tmp_path, _result(), plot_builder=_failing_plot_builder)

    assert export.export_failed is True
    assert export.failed_rois == [1]
    assert any("FAILED: ROI 1 (plot export" in line for line in export.summary_lines)
    meta = json.loads(Path(export.metadata_path).read_text())
    assert meta["export_failures"] == ["ROI 1 plot: plot boom"]


def test_csv_write_failure_reported_as_export_failure(tmp_path: Path, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _boom)

    export = _export(
        tmp_path,
        _result(),
        export_options=ExportOptions(csv=True, json=False, plot=False, interactive_plot=False),
    )

    assert export.export_failed is True
    assert any("FAILED: ROI 1 (CSV export: disk full)" in line for line in export.summary_lines)
    # Nothing written, so no metadata sidecar either.
    assert export.out_paths == []
    assert export.metadata_path is None


def test_save_analysis_outputs_supports_cancellation(tmp_path: Path):
    result = _result(
        brightness_mean_data=[[1.0], [2.0]],
        brightness_median_data=[[1.0], [2.0]],
        blue_mean_data=[[1.0], [2.0]],
        blue_median_data=[[1.0], [2.0]],
        background_values_per_frame=[0.0],
        frames_processed=2,
        total_frames=2,
        non_background_rois=[0, 1],
        start_frame=0,
        end_frame=1,
    )

    calls = {"count": 0}

    def _cancel_on_first(_current: int, _total: int) -> bool:
        calls["count"] += 1
        return calls["count"] < 2

    export = _export(tmp_path, result, progress_callback=_cancel_on_first)

    assert export.cancelled is True
    meta = json.loads(Path(export.metadata_path).read_text())
    assert meta["cancelled"] is True


def test_export_result_no_outputs_produced_when_plot_builder_returns_nothing(tmp_path: Path):
    """Selecting only the interactive plot with an unavailable plot builder should not look like success."""
    export = _export(
        tmp_path,
        _result(),
        export_options=ExportOptions(csv=False, json=False, plot=False, interactive_plot=True),
    )

    assert export.out_paths == []
    assert export.cancelled is False
    assert export.no_outputs_produced is True
    assert not any(tmp_path.iterdir())


def test_export_result_no_outputs_produced_false_when_files_written(tmp_path: Path):
    export = _export(tmp_path, _result())

    assert export.no_outputs_produced is False
