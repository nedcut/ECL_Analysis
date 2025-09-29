"""Analysis results export and visualization module."""

import logging
import os
import subprocess
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AnalysisExporter:
    """Handles exporting analysis results to CSV and generating plots."""

    def __init__(self):
        """Initialize analysis exporter."""
        self.output_paths: List[str] = []

    def save_results(
        self,
        brightness_mean_data: List[List[float]],
        brightness_median_data: List[List[float]],
        blue_mean_data: List[List[float]],
        blue_median_data: List[List[float]],
        save_dir: str,
        start_frame: int,
        non_background_roi_indices: List[int],
        analysis_name: str,
        video_name: str,
        background_values_per_frame: Optional[List[float]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Save analysis results to CSV and generate plots.

        Args:
            brightness_mean_data: List of mean brightness values per ROI
            brightness_median_data: List of median brightness values per ROI
            blue_mean_data: List of mean blue channel values per ROI
            blue_median_data: List of median blue channel values per ROI
            save_dir: Directory to save results
            start_frame: Starting frame index
            non_background_roi_indices: List of ROI indices (excluding background)
            analysis_name: Name of the analysis
            video_name: Name of the video file
            background_values_per_frame: Optional background brightness values

        Returns:
            Tuple of (output_paths, summary_lines)
        """
        self.output_paths = []
        summary_lines = [f"Analysis Complete ({len(brightness_mean_data[0]) if brightness_mean_data else 0} frames analyzed):"]
        avg_brightness_summary = []

        # Sanitize analysis name
        analysis_name = "".join(c for c in analysis_name if c.isalnum() or c in ('_', '-')).rstrip() or "DefaultAnalysis"
        base_video_name = os.path.splitext(video_name)[0]

        for data_idx in range(len(brightness_mean_data)):
            actual_roi_idx = non_background_roi_indices[data_idx]
            mean_data = brightness_mean_data[data_idx]
            median_data = brightness_median_data[data_idx]
            blue_mean = blue_mean_data[data_idx]
            blue_median = blue_median_data[data_idx]

            if not mean_data:
                continue

            # Create DataFrame with L* and blue channel data
            frame_numbers = range(start_frame, start_frame + len(mean_data))
            df = pd.DataFrame({
                "frame": frame_numbers,
                "l_raw_mean": mean_data,
                "l_raw_median": median_data,
                "blue_mean": blue_mean,
                "blue_median": blue_median
            })

            # Calculate averages for summary
            avg_mean = np.mean(mean_data)
            avg_median = np.mean(median_data)
            avg_blue_mean = np.mean(blue_mean)
            avg_blue_median = np.mean(blue_median)
            avg_brightness_summary.append(
                f"ROI {actual_roi_idx + 1} L*: {avg_mean:.2f}±{avg_median:.2f}, "
                f"Blue: {avg_blue_mean:.1f}±{avg_blue_median:.1f}"
            )

            # Construct filename and save CSV
            base_filename = (
                f"{analysis_name}_{base_video_name}_ROI{actual_roi_idx + 1}_"
                f"frames{start_frame + 1}-{start_frame + len(mean_data)}"
            )
            csv_file = f"{base_filename}_brightness.csv"
            csv_path = os.path.join(save_dir, csv_file)

            try:
                df.to_csv(csv_path, index=False)
                self.output_paths.append(csv_path)
                summary_lines.append(f" - Saved CSV: {csv_file}")

                # Generate enhanced plot for this ROI
                plot_path = self.generate_plot(
                    df, base_filename, save_dir, actual_roi_idx,
                    analysis_name, base_video_name, background_values_per_frame
                )

                if plot_path:
                    self.output_paths.append(plot_path)
                    summary_lines.append(f" - Saved Plot: {os.path.basename(plot_path)}")

            except Exception as e:
                logging.error(f"Failed to save/plot ROI {actual_roi_idx + 1}: {e}")
                summary_lines.append(f" - FAILED: ROI {actual_roi_idx + 1}")

        return self.output_paths, summary_lines

    def generate_plot(
        self,
        df: pd.DataFrame,
        base_filename: str,
        save_dir: str,
        roi_index: int,
        analysis_name: str,
        video_name: str,
        background_values_per_frame: Optional[List[float]] = None
    ) -> Optional[str]:
        """
        Generate enhanced dual-panel plot for analysis results.

        Args:
            df: DataFrame with analysis results
            base_filename: Base filename for saving
            save_dir: Directory to save plot
            roi_index: ROI index
            analysis_name: Name of the analysis
            video_name: Name of the video
            background_values_per_frame: Optional background brightness values

        Returns:
            Path to saved plot, or None if failed
        """
        try:
            frames = df['frame']
            brightness_mean = df['l_raw_mean']
            brightness_median = df['l_raw_median']
            blue_mean = df['blue_mean']
            blue_median = df['blue_median']

            if brightness_mean.empty:
                return None

            # Calculate L* statistics
            idx_peak_mean = brightness_mean.idxmax()
            frame_peak_mean = frames.iloc[idx_peak_mean]
            val_peak_mean = brightness_mean.iloc[idx_peak_mean]
            mean_of_means = brightness_mean.mean()
            std_of_means = brightness_mean.std()

            idx_peak_median = brightness_median.idxmax()
            frame_peak_median = frames.iloc[idx_peak_median]
            val_peak_median = brightness_median.iloc[idx_peak_median]
            mean_of_medians = brightness_median.mean()
            std_of_medians = brightness_median.std()

            # Calculate blue channel statistics
            idx_peak_blue_mean = blue_mean.idxmax()
            frame_peak_blue_mean = frames.iloc[idx_peak_blue_mean]
            val_peak_blue_mean = blue_mean.iloc[idx_peak_blue_mean]
            mean_of_blue_means = blue_mean.mean()
            std_of_blue_means = blue_mean.std()

            idx_peak_blue_median = blue_median.idxmax()
            frame_peak_blue_median = frames.iloc[idx_peak_blue_median]
            val_peak_blue_median = blue_median.iloc[idx_peak_blue_median]
            mean_of_blue_medians = blue_median.mean()
            std_of_blue_medians = blue_median.std()

            # Create enhanced plot with dual subplots
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # === L* Brightness Plot (Top) ===
            ax1.plot(frames, brightness_mean, label='Mean Brightness',
                    color='#5a9bd5', linewidth=2, alpha=0.8)
            ax1.plot(frames, brightness_median, label='Median Brightness',
                    color='#70ad47', linewidth=2, alpha=0.8)

            # Add background line if available
            if background_values_per_frame and len(background_values_per_frame) == len(frames):
                background_array = np.array(background_values_per_frame)
                valid_background_mask = background_array > 0
                if np.any(valid_background_mask):
                    ax1.plot(frames, background_array, label='Background Level',
                           color='#808080', linewidth=1.5, linestyle=':', alpha=0.9)

            # Add confidence bands
            ax1.fill_between(
                frames, brightness_mean - std_of_means, brightness_mean + std_of_means,
                alpha=0.2, color='#5a9bd5', label=f'Mean ±1σ ({std_of_means:.1f})'
            )
            ax1.fill_between(
                frames, brightness_median - std_of_medians, brightness_median + std_of_medians,
                alpha=0.2, color='#70ad47', label=f'Median ±1σ ({std_of_medians:.1f})'
            )

            # Add horizontal lines for averages
            ax1.axhline(mean_of_means, color='#5a9bd5', linestyle='--', alpha=0.7,
                       label=f'Avg Mean ({mean_of_means:.1f})')
            ax1.axhline(mean_of_medians, color='#70ad47', linestyle='--', alpha=0.7,
                       label=f'Avg Median ({mean_of_medians:.1f})')

            # Mark peak points
            ax1.scatter([frame_peak_mean], [val_peak_mean], color='#ff0000', zorder=5,
                       s=100, marker='^', label=f'Peak Mean ({val_peak_mean:.1f})')
            ax1.scatter([frame_peak_median], [val_peak_median], color='#ed7d31', zorder=5,
                       s=100, marker='v', label=f'Peak Median ({val_peak_median:.1f})')

            ax1.set_title(f"{analysis_name} - {video_name} - ROI {roi_index + 1}",
                         fontsize=16, fontweight='bold')
            ax1.set_ylabel('L* Brightness', fontsize=12)
            ax1.legend(fontsize=10, loc='best')
            ax1.grid(True, alpha=0.3)

            # Adjust y-axis limits for statistics panel
            y_min, y_max = ax1.get_ylim()
            y_range = y_max - y_min
            ax1.set_ylim(y_min, y_max + 0.15 * y_range)

            # Add statistics text box
            stats_text = (
                f"Statistics:\n"
                f"Mean: {mean_of_means:.2f} ± {std_of_means:.2f}\n"
                f"Median: {mean_of_medians:.2f} ± {std_of_medians:.2f}\n"
                f"Peak Mean: {val_peak_mean:.2f} @ Frame {frame_peak_mean}\n"
                f"Peak Median: {val_peak_median:.2f} @ Frame {frame_peak_median}\n"
                f"Frames Analyzed: {len(frames)}"
            )
            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # === Blue Channel Plot (Bottom) ===
            ax2.plot(frames, blue_mean, label='Blue Mean',
                    color='#0066cc', linewidth=2, alpha=0.8)
            ax2.plot(frames, blue_median, label='Blue Median',
                    color='#3399ff', linewidth=2, alpha=0.8)

            # Add confidence bands for blue channel
            ax2.fill_between(
                frames, blue_mean - std_of_blue_means, blue_mean + std_of_blue_means,
                alpha=0.2, color='#0066cc', label=f'Blue Mean ±1σ ({std_of_blue_means:.1f})'
            )
            ax2.fill_between(
                frames, blue_median - std_of_blue_medians, blue_median + std_of_blue_medians,
                alpha=0.2, color='#3399ff', label=f'Blue Median ±1σ ({std_of_blue_medians:.1f})'
            )

            # Add horizontal lines for blue averages
            ax2.axhline(mean_of_blue_means, color='#0066cc', linestyle='--', alpha=0.7,
                       label=f'Avg Blue Mean ({mean_of_blue_means:.1f})')
            ax2.axhline(mean_of_blue_medians, color='#3399ff', linestyle='--', alpha=0.7,
                       label=f'Avg Blue Median ({mean_of_blue_medians:.1f})')

            # Mark blue peak points
            ax2.scatter([frame_peak_blue_mean], [val_peak_blue_mean], color='#ff0000',
                       zorder=5, s=100, marker='^', label=f'Peak Blue Mean ({val_peak_blue_mean:.1f})')
            ax2.scatter([frame_peak_blue_median], [val_peak_blue_median], color='#ed7d31',
                       zorder=5, s=100, marker='v', label=f'Peak Blue Median ({val_peak_blue_median:.1f})')

            ax2.set_xlabel('Frame Number', fontsize=12)
            ax2.set_ylabel('Blue Channel Value', fontsize=12)
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, alpha=0.3)

            # Add blue channel statistics text box
            blue_stats_text = (
                f"Blue Channel Statistics:\n"
                f"Mean: {mean_of_blue_means:.1f} ± {std_of_blue_means:.1f}\n"
                f"Median: {mean_of_blue_medians:.1f} ± {std_of_blue_medians:.1f}\n"
                f"Peak Mean: {val_peak_blue_mean:.1f} @ Frame {frame_peak_blue_mean}\n"
                f"Peak Median: {val_peak_blue_median:.1f} @ Frame {frame_peak_blue_median}"
            )
            ax2.text(0.98, 0.98, blue_stats_text, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            plt.tight_layout()

            # Save plot
            plot_filename = f"{base_filename}_plot.png"
            plot_save_path = os.path.join(save_dir, plot_filename)
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)

            # Automatically open the generated PNG file (macOS)
            try:
                subprocess.run(['open', plot_save_path], check=True)
            except Exception as e:
                logging.warning(f"Could not automatically open plot file {plot_save_path}: {e}")

            return plot_save_path

        except Exception as e:
            logging.error(f"Failed to generate plot for ROI {roi_index + 1}: {e}")
            return None

    def get_output_paths(self) -> List[str]:
        """
        Get list of all output file paths.

        Returns:
            List of output file paths
        """
        return self.output_paths.copy()