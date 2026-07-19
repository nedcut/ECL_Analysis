"""Plot generation for completed analysis results (static PNG + interactive HTML)."""

from __future__ import annotations

import json
import logging
import os
from string import Template
from typing import List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui

from ecl_analysis.constants import (
    COLOR_ACCENT,
    COLOR_INFO,
    COLOR_WARNING,
    DEFAULT_FONT_FAMILY,
)
from ecl_analysis.dependencies import get_plotly


def generate_enhanced_plot(
    df,
    base_filename,
    save_dir,
    r_idx,
    analysis_name,
    base_video_name,
    background_values_per_frame,
    generate_static=True,
    generate_interactive=True,
):
    """Generate enhanced plots and an interactive visualization for the ROI."""
    from ecl_analysis.video_analyzer import _hex_to_rgba

    png_path: Optional[str] = None
    interactive_path: Optional[str] = None
    try:
        frames = df['frame']
        brightness_mean = df['brightness_mean']
        brightness_median = df['brightness_median']
        blue_mean = df['blue_mean']
        blue_median = df['blue_median']

        if brightness_mean.empty:
            return png_path, interactive_path

        # Statistics
        idx_peak_mean = brightness_mean.idxmax()
        frame_peak_mean, val_peak_mean = frames.iloc[idx_peak_mean], brightness_mean.iloc[idx_peak_mean]
        mean_of_means = brightness_mean.mean()
        std_of_means = brightness_mean.std()

        idx_peak_median = brightness_median.idxmax()
        frame_peak_median, val_peak_median = frames.iloc[idx_peak_median], brightness_median.iloc[idx_peak_median]
        mean_of_medians = brightness_median.mean()
        std_of_medians = brightness_median.std()

        # Blue channel statistics
        idx_peak_blue_mean = blue_mean.idxmax()
        frame_peak_blue_mean, val_peak_blue_mean = frames.iloc[idx_peak_blue_mean], blue_mean.iloc[idx_peak_blue_mean]
        mean_of_blue_means = blue_mean.mean()
        std_of_blue_means = blue_mean.std()

        idx_peak_blue_median = blue_median.idxmax()
        frame_peak_blue_median, val_peak_blue_median = frames.iloc[idx_peak_blue_median], blue_median.iloc[idx_peak_blue_median]
        mean_of_blue_medians = blue_median.mean()
        std_of_blue_medians = blue_median.std()

        frame_list = frames.tolist()
        brightness_mean_values = brightness_mean.tolist()
        brightness_median_values = brightness_median.tolist()
        blue_mean_values = blue_mean.tolist()
        blue_median_values = blue_median.tolist()

        background_array: Optional[np.ndarray] = None
        if background_values_per_frame and len(background_values_per_frame) == len(frames):
            candidate_background = np.array(background_values_per_frame)
            valid_background_mask = candidate_background > 0
            if np.any(valid_background_mask):
                background_array = candidate_background

        if generate_static:
            # Create enhanced plot with dual subplots
            import matplotlib.pyplot as plt

            plt.style.use('seaborn-v0_8-darkgrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Main brightness plot
            ax1.plot(frames, brightness_mean, label='Mean Brightness', color='#5a9bd5', linewidth=2, alpha=0.8)
            ax1.plot(frames, brightness_median, label='Median Brightness', color='#70ad47', linewidth=2, alpha=0.8)

            # Add background line if background values are available
            if background_array is not None:
                ax1.plot(frames, background_array, label='Background Level', color='#808080',
                         linewidth=1.5, linestyle=':', alpha=0.9)

            # Add confidence bands (mean ± std)
            ax1.fill_between(frames, brightness_mean - std_of_means, brightness_mean + std_of_means,
                             alpha=0.2, color='#5a9bd5', label=f'Mean ±1σ ({std_of_means:.1f})')
            ax1.fill_between(frames, brightness_median - std_of_medians, brightness_median + std_of_medians,
                             alpha=0.2, color='#70ad47', label=f'Median ±1σ ({std_of_medians:.1f})')

            # Add horizontal lines for averages
            ax1.axhline(mean_of_means, color='#5a9bd5', linestyle='--', alpha=0.7,
                        label=f'Avg Mean ({mean_of_means:.1f})')
            ax1.axhline(mean_of_medians, color='#70ad47', linestyle='--', alpha=0.7,
                        label=f'Avg Median ({mean_of_medians:.1f})')

            # Mark peak points
            ax1.scatter([frame_peak_mean], [val_peak_mean], color='#ff0000', zorder=5, s=100,
                        marker='^', label=f'Peak Mean ({val_peak_mean:.1f})')
            ax1.scatter([frame_peak_median], [val_peak_median], color='#ed7d31', zorder=5, s=100,
                        marker='v', label=f'Peak Median ({val_peak_median:.1f})')

            ax1.set_title(f"{analysis_name} - {base_video_name} - ROI {r_idx+1}", fontsize=16, fontweight='bold')
            ax1.set_ylabel('L* Brightness', fontsize=12)
            ax1.legend(fontsize=10, loc='best')
            ax1.grid(True, alpha=0.3)

            # Adjust y-axis limits to provide more space at the top for statistics panel
            y_min, y_max = ax1.get_ylim()
            y_range = y_max - y_min
            ax1.set_ylim(y_min, y_max + 0.15 * y_range)

            # Add statistics text box
            stats_text = f"""Statistics:
Mean: {mean_of_means:.2f} ± {std_of_means:.2f}
Median: {mean_of_medians:.2f} ± {std_of_medians:.2f}
Peak Mean: {val_peak_mean:.2f} @ Frame {frame_peak_mean}
Peak Median: {val_peak_median:.2f} @ Frame {frame_peak_median}
Frames Analyzed: {len(frames)}"""

            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Blue channel plot
            ax2.plot(frames, blue_mean, label='Blue Mean', color='#0066cc', linewidth=2, alpha=0.8)
            ax2.plot(frames, blue_median, label='Blue Median', color='#3399ff', linewidth=2, alpha=0.8)

            # Add confidence bands for blue channel
            ax2.fill_between(frames, blue_mean - std_of_blue_means, blue_mean + std_of_blue_means,
                             alpha=0.2, color='#0066cc', label=f'Blue Mean ±1σ ({std_of_blue_means:.1f})')
            ax2.fill_between(frames, blue_median - std_of_blue_medians, blue_median + std_of_blue_medians,
                             alpha=0.2, color='#3399ff', label=f'Blue Median ±1σ ({std_of_blue_medians:.1f})')

            # Add horizontal lines for blue averages
            ax2.axhline(mean_of_blue_means, color='#0066cc', linestyle='--', alpha=0.7,
                        label=f'Avg Blue Mean ({mean_of_blue_means:.1f})')
            ax2.axhline(mean_of_blue_medians, color='#3399ff', linestyle='--', alpha=0.7,
                        label=f'Avg Blue Median ({mean_of_blue_medians:.1f})')

            # Mark blue peak points
            ax2.scatter([frame_peak_blue_mean], [val_peak_blue_mean], color='#ff0000', zorder=5, s=100,
                        marker='^', label=f'Peak Blue Mean ({val_peak_blue_mean:.1f})')
            ax2.scatter([frame_peak_blue_median], [val_peak_blue_median], color='#ed7d31', zorder=5, s=100,
                        marker='v', label=f'Peak Blue Median ({val_peak_blue_median:.1f})')

            ax2.set_xlabel('Frame Number', fontsize=12)
            ax2.set_ylabel('Blue Channel Value', fontsize=12)
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, alpha=0.3)

            # Add blue channel statistics text box
            blue_stats_text = f"""Blue Channel Statistics:
Mean: {mean_of_blue_means:.1f} ± {std_of_blue_means:.1f}
Median: {mean_of_blue_medians:.1f} ± {std_of_blue_medians:.1f}
Peak Mean: {val_peak_blue_mean:.1f} @ Frame {frame_peak_blue_mean}
Peak Median: {val_peak_blue_median:.1f} @ Frame {frame_peak_blue_median}"""

            ax2.text(0.98, 0.98, blue_stats_text, transform=ax2.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            plt.tight_layout()

            # Save plot
            plot_filename = f"{base_filename}_plot.png"
            plot_save_path = os.path.join(save_dir, plot_filename)
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            png_path = plot_save_path

        if generate_interactive:
            plotly_go, plotly_make_subplots = get_plotly()
            if plotly_go is not None and plotly_make_subplots is not None:
                try:
                    interactive_filename = f"{base_filename}_interactive.html"
                    interactive_save_path = os.path.join(save_dir, interactive_filename)

                    fig_interactive = plotly_make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("L* Brightness", "Blue Channel")
                    )
                    selection_fill = _hex_to_rgba(COLOR_ACCENT, 0.18)

                    # L* mean confidence band
                    upper_mean_band = (brightness_mean + std_of_means).tolist()
                    lower_mean_band = (brightness_mean - std_of_means).tolist()
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=upper_mean_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=lower_mean_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=True,
                            name=f"Mean ±1σ ({std_of_means:.1f})",
                            fill='tonexty',
                            fillcolor='rgba(90,155,213,0.25)',
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )

                    # Median confidence band
                    upper_median_band = (brightness_median + std_of_medians).tolist()
                    lower_median_band = (brightness_median - std_of_medians).tolist()
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=upper_median_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=lower_median_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=True,
                            name=f"Median ±1σ ({std_of_medians:.1f})",
                            fill='tonexty',
                            fillcolor='rgba(112,173,71,0.25)',
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )

                    # Brightness lines
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=brightness_mean_values,
                            mode='lines',
                            name='Mean Brightness',
                            line=dict(color='#5a9bd5', width=2),
                            hovertemplate="Frame %{x}<br>Mean L*: %{y:.2f}<extra></extra>"
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=brightness_median_values,
                            mode='lines',
                            name='Median Brightness',
                            line=dict(color='#70ad47', width=2),
                            hovertemplate="Frame %{x}<br>Median L*: %{y:.2f}<extra></extra>"
                        ),
                        row=1,
                        col=1
                    )

                    # Background level
                    if background_array is not None:
                        fig_interactive.add_trace(
                            plotly_go.Scatter(
                                x=frame_list,
                                y=background_array.tolist(),
                                mode='lines',
                                name='Background Level',
                                line=dict(color='#808080', width=1.5, dash='dot'),
                                hovertemplate="Frame %{x}<br>Background L*: %{y:.2f}<extra></extra>"
                            ),
                            row=1,
                            col=1
                        )

                    # Peak annotations
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=[frame_peak_mean],
                            y=[val_peak_mean],
                            mode='markers',
                            name=f'Peak Mean ({val_peak_mean:.1f})',
                            marker=dict(color='#ff0000', size=10, symbol='triangle-up'),
                            hovertemplate="Frame %{x}<br>Peak Mean L*: %{y:.2f}<extra></extra>"
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=[frame_peak_median],
                            y=[val_peak_median],
                            mode='markers',
                            name=f'Peak Median ({val_peak_median:.1f})',
                            marker=dict(color='#ed7d31', size=10, symbol='triangle-down'),
                            hovertemplate="Frame %{x}<br>Peak Median L*: %{y:.2f}<extra></extra>"
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=[frame_peak_mean],
                            y=[val_peak_mean],
                            mode='markers',
                            name='Selected Range Peak (L*)',
                            marker=dict(
                                color=COLOR_WARNING,
                                size=14,
                                symbol='star',
                                line=dict(color='#92400e', width=1.2)
                            ),
                            hovertemplate="Frame %{x}<br>Selected L* Peak: %{y:.2f}<extra></extra>"
                        ),
                        row=1,
                        col=1
                    )

                    # Horizontal averages
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=[mean_of_means] * len(frame_list),
                            mode='lines',
                            name=f'Avg Mean ({mean_of_means:.1f})',
                            line=dict(color='#5a9bd5', dash='dash'),
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=[mean_of_medians] * len(frame_list),
                            mode='lines',
                            name=f'Avg Median ({mean_of_medians:.1f})',
                            line=dict(color='#70ad47', dash='dash'),
                            hoverinfo='skip'
                        ),
                        row=1,
                        col=1
                    )

                    # Blue channel confidence bands
                    upper_blue_mean_band = (blue_mean + std_of_blue_means).tolist()
                    lower_blue_mean_band = (blue_mean - std_of_blue_means).tolist()
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=upper_blue_mean_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=lower_blue_mean_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=True,
                            name=f'Blue Mean ±1σ ({std_of_blue_means:.1f})',
                            fill='tonexty',
                            fillcolor='rgba(0,102,204,0.25)',
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )

                    upper_blue_median_band = (blue_median + std_of_blue_medians).tolist()
                    lower_blue_median_band = (blue_median - std_of_blue_medians).tolist()
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=upper_blue_median_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=lower_blue_median_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=True,
                            name=f'Blue Median ±1σ ({std_of_blue_medians:.1f})',
                            fill='tonexty',
                            fillcolor='rgba(51,153,255,0.25)',
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )

                    # Blue channel lines
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=blue_mean_values,
                            mode='lines',
                            name='Blue Mean',
                            line=dict(color='#0066cc', width=2),
                            hovertemplate="Frame %{x}<br>Blue Mean: %{y:.2f}<extra></extra>"
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=blue_median_values,
                            mode='lines',
                            name='Blue Median',
                            line=dict(color='#3399ff', width=2),
                            hovertemplate="Frame %{x}<br>Blue Median: %{y:.2f}<extra></extra>"
                        ),
                        row=2,
                        col=1
                    )

                    # Blue channel peaks
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=[frame_peak_blue_mean],
                            y=[val_peak_blue_mean],
                            mode='markers',
                            name=f'Peak Blue Mean ({val_peak_blue_mean:.1f})',
                            marker=dict(color='#ff0000', size=10, symbol='triangle-up'),
                            hovertemplate="Frame %{x}<br>Peak Blue Mean: %{y:.2f}<extra></extra>"
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=[frame_peak_blue_median],
                            y=[val_peak_blue_median],
                            mode='markers',
                            name=f'Peak Blue Median ({val_peak_blue_median:.1f})',
                            marker=dict(color='#ed7d31', size=10, symbol='triangle-down'),
                            hovertemplate="Frame %{x}<br>Peak Blue Median: %{y:.2f}<extra></extra>"
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=[frame_peak_blue_mean],
                            y=[val_peak_blue_mean],
                            mode='markers',
                            name='Selected Range Peak (Blue)',
                            marker=dict(
                                color=COLOR_INFO,
                                size=14,
                                symbol='star',
                                line=dict(color='#0e7490', width=1.2)
                            ),
                            hovertemplate="Frame %{x}<br>Selected Blue Peak: %{y:.2f}<extra></extra>"
                        ),
                        row=2,
                        col=1
                    )

                    # Blue channel averages
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=[mean_of_blue_means] * len(frame_list),
                            mode='lines',
                            name=f'Avg Blue Mean ({mean_of_blue_means:.1f})',
                            line=dict(color='#0066cc', dash='dash'),
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )
                    fig_interactive.add_trace(
                        plotly_go.Scatter(
                            x=frame_list,
                            y=[mean_of_blue_medians] * len(frame_list),
                            mode='lines',
                            name=f'Avg Blue Median ({mean_of_blue_medians:.1f})',
                            line=dict(color='#3399ff', dash='dash'),
                            hoverinfo='skip'
                        ),
                        row=2,
                        col=1
                    )

                    # Layout
                    fig_interactive.update_layout(
                        title=f"{analysis_name} - {base_video_name} - ROI {r_idx+1}",
                        height=820,
                        dragmode='select',
                        selectdirection='h',
                        hovermode='x unified',
                        template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
                        margin=dict(t=80, b=60, l=60, r=30),
                        newselection=dict(
                            line=dict(color=COLOR_ACCENT, width=2)
                        )
                    )
                    fig_interactive.update_xaxes(title_text="Frame Number", row=2, col=1)
                    fig_interactive.update_yaxes(title_text="L* Brightness", row=1, col=1)
                    fig_interactive.update_yaxes(title_text="Blue Channel Value", row=2, col=1)

                    # Summary annotation
                    fig_interactive.add_annotation(
                        text=(
                            f"Mean: {mean_of_means:.2f} ± {std_of_means:.2f} | "
                            f"Median: {mean_of_medians:.2f} ± {std_of_medians:.2f}<br>"
                            f"Blue Mean: {mean_of_blue_means:.1f} ± {std_of_blue_means:.1f} | "
                            f"Blue Median: {mean_of_blue_medians:.1f} ± {std_of_blue_medians:.1f}"
                        ),
                        xref="paper",
                        yref="paper",
                        x=0.0,
                        y=1.12,
                        showarrow=False,
                        align='left',
                        font=dict(size=12),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#cccccc',
                        borderwidth=1,
                        borderpad=6
                    )

                    div_id = f"roi-interactive-{r_idx+1}"
                    selection_script = build_selection_post_script(
                        div_id=div_id,
                        frames=frame_list,
                        brightness_values=brightness_mean_values,
                        blue_values=blue_mean_values,
                        accent_color=COLOR_ACCENT,
                        selection_fill=selection_fill,
                    )
                    fig_interactive.write_html(
                        interactive_save_path,
                        include_plotlyjs='cdn',
                        div_id=div_id,
                        post_script=selection_script,
                    )
                    interactive_path = interactive_save_path

                    try:
                        opened = QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(interactive_save_path))
                        if not opened:
                            logging.warning("Could not automatically open interactive plot %s", interactive_save_path)
                    except Exception as exc:
                        logging.warning("Could not automatically open interactive plot %s: %s", interactive_save_path, exc)
                except Exception as plotly_error:
                    logging.warning(f"Failed to generate interactive plot for ROI {r_idx+1}: {plotly_error}")
            else:
                logging.info("Plotly not available - skipping interactive plot generation.")

        return png_path, interactive_path

    except Exception as e:
        logging.error(f"Failed to generate plot for ROI {r_idx+1}: {e}")
        raise


def build_selection_post_script(
    div_id: str,
    frames: List[float],
    brightness_values: List[float],
    blue_values: List[float],
    accent_color: str,
    selection_fill: str,
) -> str:
    """Generate a JS snippet that highlights peaks inside the active brush selection."""
    frames_json = json.dumps(frames)
    brightness_json = json.dumps(brightness_values)
    blue_json = json.dumps(blue_values)
    font_family = DEFAULT_FONT_FAMILY.replace("\\", "\\\\").replace("'", "\\'")
    accent_color = accent_color.replace("\\", "\\\\").replace("'", "\\'")
    selection_fill = selection_fill.replace("\\", "\\\\").replace("'", "\\'")

    script = Template(
        "(function() {\n"
        "  const divId = '$div_id';\n"
        "  const frames = $frames_json;\n"
        "  const brightnessValues = $brightness_json;\n"
        "  const blueValues = $blue_json;\n"
        "  const accentColor = '$accent_color';\n"
        "  const selectionFill = '$selection_fill';\n"
        "  const fontFamily = '$font_family';\n"
        "\n"
        "  const createSelectionEnhancements = () => {\n"
        "    const gd = document.getElementById(divId);\n"
        "    if (!gd || gd.__selectionInitialized) {\n"
        "      return;\n"
        "    }\n"
        "\n"
        "    const initialiseWhenReady = () => {\n"
        "      if (typeof Plotly === 'undefined') {\n"
        "        return false;\n"
        "      }\n"
        "      if (!gd.data || !gd.data.length) {\n"
        "        return false;\n"
        "      }\n"
        "\n"
        "      const selectedMeanIndex = gd.data.findIndex(trace => trace.name === 'Selected Range Peak (L*)');\n"
        "      const selectedBlueIndex = gd.data.findIndex(trace => trace.name === 'Selected Range Peak (Blue)');\n"
        "      if (selectedMeanIndex === -1 || selectedBlueIndex === -1) {\n"
        "        return false;\n"
        "      }\n"
        "\n"
        "      gd.__selectionInitialized = true;\n"
        "\n"
        "      const infoPanel = document.createElement('div');\n"
        "      infoPanel.className = 'selection-info';\n"
        "      infoPanel.style.marginTop = '12px';\n"
        "      infoPanel.style.fontFamily = fontFamily;\n"
        "      infoPanel.style.fontSize = '14px';\n"
        "      infoPanel.style.color = '#1f2937';\n"
        "      infoPanel.style.background = 'rgba(255,255,255,0.92)';\n"
        "      infoPanel.style.border = '1px solid #e5e7eb';\n"
        "      infoPanel.style.borderRadius = '8px';\n"
        "      infoPanel.style.padding = '8px 12px';\n"
        "      infoPanel.style.boxShadow = '0 1px 3px rgba(15,23,42,0.12)';\n"
        "      infoPanel.style.display = 'inline-block';\n"
        "      if (gd.parentNode) {\n"
        "        gd.parentNode.insertBefore(infoPanel, gd.nextSibling);\n"
        "      }\n"
        "\n"
        "      const domainStart = frames[0];\n"
        "      const domainEnd = frames[frames.length - 1];\n"
        "      const nearlyEqual = (a, b) => Math.abs(a - b) <= Math.max(1, Math.abs(domainEnd - domainStart)) * 1e-6;\n"
        "\n"
        "      const findPeakIndex = (range, values) => {\n"
        "        let candidate = -1;\n"
        "        let maxValue = -Infinity;\n"
        "        for (let i = 0; i < frames.length; i += 1) {\n"
        "          const frame = frames[i];\n"
        "          if (frame >= range[0] && frame <= range[1]) {\n"
        "            const value = values[i];\n"
        "            if (value > maxValue) {\n"
        "              maxValue = value;\n"
        "              candidate = i;\n"
        "            }\n"
        "          }\n"
        "        }\n"
        "        if (candidate !== -1) {\n"
        "          return candidate;\n"
        "        }\n"
        "        let bestDistance = Infinity;\n"
        "        for (let i = 0; i < frames.length; i += 1) {\n"
        "          const frame = frames[i];\n"
        "          const distance = frame < range[0] ? range[0] - frame : frame > range[1] ? frame - range[1] : 0;\n"
        "          if (distance < bestDistance) {\n"
        "            bestDistance = distance;\n"
        "            candidate = i;\n"
        "          }\n"
        "        }\n"
        "        return candidate;\n"
        "      };\n"
        "\n"
        "      const extractRange = (rangeLike) => {\n"
        "        if (Array.isArray(rangeLike) && rangeLike.length >= 2) {\n"
        "          return [Number(rangeLike[0]), Number(rangeLike[1])];\n"
        "        }\n"
        "        if (rangeLike && Array.isArray(rangeLike.x) && rangeLike.x.length >= 2) {\n"
        "          return [Number(rangeLike.x[0]), Number(rangeLike.x[1])];\n"
        "        }\n"
        "        return [domainStart, domainEnd];\n"
        "      };\n"
        "\n"
        "      const restyleMarker = (traceIndex, frameIndex, values) => {\n"
        "        if (traceIndex === -1 || frameIndex < 0 || frameIndex >= frames.length) {\n"
        "          return;\n"
        "        }\n"
        "        try {\n"
        "          Plotly.restyle(gd, {\n"
        "            x: [[frames[frameIndex]]],\n"
        "            y: [[values[frameIndex]]]\n"
        "          }, [traceIndex]);\n"
        "        } catch (err) {\n"
        "          /* ignore restyle issues */\n"
        "        }\n"
        "      };\n"
        "\n"
        "      const applySelectionShape = (start, end) => {\n"
        "        const coversAll = nearlyEqual(start, domainStart) && nearlyEqual(end, domainEnd);\n"
        "        const shapes = coversAll ? [] : [{\n"
        "          type: 'rect',\n"
        "          xref: 'x',\n"
        "          x0: start,\n"
        "          x1: end,\n"
        "          yref: 'paper',\n"
        "          y0: 0,\n"
        "          y1: 1,\n"
        "          fillcolor: selectionFill,\n"
        "          line: { color: accentColor, width: 1, dash: 'dot' },\n"
        "          layer: 'below'\n"
        "        }];\n"
        "        try {\n"
        "          Plotly.relayout(gd, { shapes });\n"
        "        } catch (err) {\n"
        "          /* ignore relayout issues */\n"
        "        }\n"
        "      };\n"
        "\n"
        "      const updateInfoPanel = (start, end, brightnessIndex, blueIndex) => {\n"
        "        const formatValue = (value, digits = 2) => {\n"
        "          const numeric = Number.parseFloat(value);\n"
        "          return Number.isFinite(numeric) ? numeric.toFixed(digits) : 'n/a';\n"
        "        };\n"
        "        const parts = [];\n"
        "        const framesLabel = 'Frames ' + Math.round(start) + '–' + Math.round(end);\n"
        "        parts.push('<div style=\"font-weight:600;color:' + accentColor + '\">' + framesLabel + '</div>');\n"
        "        if (brightnessIndex >= 0) {\n"
        "          const frame = frames[brightnessIndex];\n"
        "          const value = formatValue(brightnessValues[brightnessIndex]);\n"
        "          parts.push('<div><strong>L*</strong> frame ' + frame + ' (' + value + ')</div>');\n"
        "        } else {\n"
        "          parts.push('<div><strong>L*</strong> peak n/a</div>');\n"
        "        }\n"
        "        if (blueIndex >= 0) {\n"
        "          const frame = frames[blueIndex];\n"
        "          const value = formatValue(blueValues[blueIndex]);\n"
        "          parts.push('<div><strong>Blue</strong> frame ' + frame + ' (' + value + ')</div>');\n"
        "        } else {\n"
        "          parts.push('<div><strong>Blue</strong> peak n/a</div>');\n"
        "        }\n"
        "        infoPanel.innerHTML = parts.join('');\n"
        "      };\n"
        "\n"
        "      const updateSelection = (rangeLike) => {\n"
        "        const [rawStart, rawEnd] = extractRange(rangeLike);\n"
        "        const clamp = (value) => Math.min(Math.max(value, domainStart), domainEnd);\n"
        "        const start = clamp(Math.min(rawStart, rawEnd));\n"
        "        const end = clamp(Math.max(rawStart, rawEnd));\n"
        "        const brightnessIndex = findPeakIndex([start, end], brightnessValues);\n"
        "        const blueIndex = findPeakIndex([start, end], blueValues);\n"
        "        restyleMarker(selectedMeanIndex, brightnessIndex, brightnessValues);\n"
        "        restyleMarker(selectedBlueIndex, blueIndex, blueValues);\n"
        "        applySelectionShape(start, end);\n"
        "        updateInfoPanel(start, end, brightnessIndex, blueIndex);\n"
        "      };\n"
        "\n"
        "      const reset = () => {\n"
        "        updateSelection([domainStart, domainEnd]);\n"
        "      };\n"
        "\n"
        "      gd.on('plotly_selected', (eventData) => {\n"
        "        if (eventData && eventData.range && eventData.range.x) {\n"
        "          updateSelection(eventData.range);\n"
        "        }\n"
        "      });\n"
        "      gd.on('plotly_selecting', (eventData) => {\n"
        "        if (eventData && eventData.range && eventData.range.x) {\n"
        "          updateSelection(eventData.range);\n"
        "        }\n"
        "      });\n"
        "      gd.on('plotly_doubleclick', reset);\n"
        "      gd.on('plotly_deselect', reset);\n"
        "\n"
        "      reset();\n"
        "      return true;\n"
        "    };\n"
        "\n"
        "    if (!initialiseWhenReady()) {\n"
        "      const handler = () => {\n"
        "        if (initialiseWhenReady() && gd.removeListener) {\n"
        "          gd.removeListener('plotly_afterplot', handler);\n"
        "        }\n"
        "      };\n"
        "      if (gd.on) {\n"
        "        gd.on('plotly_afterplot', handler);\n"
        "      } else {\n"
        "        setTimeout(handler, 60);\n"
        "      }\n"
        "    }\n"
        "  };\n"
        "\n"
        "  if (document.readyState === 'loading') {\n"
        "    document.addEventListener('DOMContentLoaded', createSelectionEnhancements, { once: true });\n"
        "  } else {\n"
        "    createSelectionEnhancements();\n"
        "  }\n"
        "})();"
    )

    return script.substitute(
        div_id=div_id,
        frames_json=frames_json,
        brightness_json=brightness_json,
        blue_json=blue_json,
        accent_color=accent_color,
        selection_fill=selection_fill,
        font_family=font_family,
    )
