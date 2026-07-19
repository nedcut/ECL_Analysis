import os
from pathlib import Path

import pandas as pd

from ecl_analysis.export.plotting import build_selection_post_script, generate_enhanced_plot


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
