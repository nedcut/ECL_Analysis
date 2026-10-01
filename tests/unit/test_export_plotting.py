import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecl_analysis.export.plotting import (
    BACKGROUND_AXIS_LABEL,
    build_selection_post_script,
    generate_enhanced_plot,
)
from ecl_analysis.export.stats import format_std, sample_std


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4],
            "brightness_mean": [10.0, 12.0, 15.0, 11.0, 9.0],
            "brightness_median": [9.0, 11.0, 14.0, 10.0, 8.0],
            "blue_mean": [5.0, 6.0, 8.0, 6.5, 5.5],
            "blue_median": [4.5, 5.5, 7.5, 6.0, 5.0],
        }
    )


def test_generate_enhanced_plot_writes_static_png(tmp_path: Path):
    df = _make_df()

    png_path, interactive_path = generate_enhanced_plot(
        df,
        "roi0",
        str(tmp_path),
        0,
        "TestAnalysis",
        "test_video",
        background_values_per_frame=None,
        generate_static=True,
        generate_interactive=False,
    )

    assert png_path is not None
    assert os.path.exists(png_path)
    assert interactive_path is None


def test_generate_enhanced_plot_empty_dataframe_returns_none(tmp_path: Path):
    empty_df = pd.DataFrame(
        {
            "frame": [],
            "brightness_mean": [],
            "brightness_median": [],
            "blue_mean": [],
            "blue_median": [],
        }
    )

    png_path, interactive_path = generate_enhanced_plot(
        empty_df,
        "roi0",
        str(tmp_path),
        0,
        "TestAnalysis",
        "test_video",
        background_values_per_frame=None,
        generate_static=True,
        generate_interactive=False,
    )

    assert png_path is None
    assert interactive_path is None


def test_build_selection_post_script_embeds_data():
    script = build_selection_post_script(
        div_id="roi-interactive-1",
        frames=[0, 1, 2],
        brightness_values=[1.0, 2.0, 3.0],
        blue_values=[0.5, 1.5, 2.5],
        accent_color="#5a9bd5",
        selection_fill="rgba(90,155,213,0.18)",
    )

    assert "roi-interactive-1" in script
    assert "[0, 1, 2]" in script


def _capture_static_figure(monkeypatch):
    """Keep a reference to the matplotlib figure that generate_enhanced_plot closes."""
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close

    def _close(fig=None):
        captured["fig"] = fig
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _close)
    return captured


def test_static_plot_band_is_horizontal_series_mean_pm_sample_sd(tmp_path: Path, monkeypatch):
    captured = _capture_static_figure(monkeypatch)
    df = _make_df()

    generate_enhanced_plot(df, "roi0", str(tmp_path), 0, "T", "v", None, True, False)

    ax1 = captured["fig"].axes[0]
    mean = df["brightness_mean"].mean()
    sd = float(np.std(df["brightness_mean"], ddof=1))
    band_labels = [label for label in ax1.get_legend_handles_labels()[1] if "SD across frames" in label]
    assert band_labels == [
        f"Mean avg ±1 SD across frames ({sd:.1f})",
        f"Median avg ±1 SD across frames ({float(np.std(df['brightness_median'], ddof=1)):.1f})",
    ]
    # The mean band spans exactly mean ± SD on the y axis (horizontal, constant width).
    band = next(p for p in ax1.patches if p.get_label().startswith("Mean avg"))
    ys = band.get_path().transformed(band.get_patch_transform()).vertices[:, 1]
    assert min(ys) == pytest.approx(mean - sd)
    assert max(ys) == pytest.approx(mean + sd)
    # No legacy per-point ±σ envelopes.
    assert not any("±1σ" in label for label in ax1.get_legend_handles_labels()[1])


def test_static_plot_background_on_secondary_axis(tmp_path: Path, monkeypatch):
    captured = _capture_static_figure(monkeypatch)
    df = _make_df()

    generate_enhanced_plot(df, "roi0", str(tmp_path), 0, "T", "v", [20.0] * 5, True, False)

    fig = captured["fig"]
    assert len(fig.axes) == 3  # L* panel, blue panel, background twin axis
    twin = fig.axes[2]
    assert twin.get_ylabel() == BACKGROUND_AXIS_LABEL
    assert twin.lines[0].get_ydata().tolist() == [20.0] * 5
    # Background is not drawn on the background-subtracted L* axis.
    assert all(line.get_label() != "Background level (right axis)" for line in fig.axes[0].lines)
    legend_labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Background level (right axis)" in legend_labels


def test_static_plot_single_frame_has_no_band(tmp_path: Path, monkeypatch):
    captured = _capture_static_figure(monkeypatch)
    df = _make_df().iloc[:1]

    png_path, _ = generate_enhanced_plot(df, "roi0", str(tmp_path), 0, "T", "v", None, True, False)

    assert png_path is not None
    ax1 = captured["fig"].axes[0]
    assert not any("SD across frames" in label for label in ax1.get_legend_handles_labels()[1])
    stats_texts = [t.get_text() for t in ax1.texts]
    assert any("± n/a" in text for text in stats_texts)


def test_interactive_plot_uses_horizontal_band_and_background_axis(tmp_path: Path, monkeypatch):
    pytest.importorskip("plotly")
    from PyQt5 import QtGui

    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", lambda _url: True)
    df = _make_df()

    _, interactive_path = generate_enhanced_plot(df, "roi0", str(tmp_path), 0, "T", "v", [20.0] * 5, False, True)

    assert interactive_path is not None
    html = Path(interactive_path).read_text(encoding="utf-8")
    assert BACKGROUND_AXIS_LABEL in html
    assert "SD across frames" in html
    assert "\\u00b11\\u03c3" not in html and "±1σ" not in html


def test_sample_std_uses_ddof_1_and_guards_small_n():
    assert sample_std([1.0, 2.0, 3.0, 10.0]) == pytest.approx(float(np.std([1.0, 2.0, 3.0, 10.0], ddof=1)))
    assert sample_std([5.0]) is None
    assert sample_std([]) is None
    assert format_std(None, 2) == "n/a"
    assert format_std(1.234, 1) == "1.2"
