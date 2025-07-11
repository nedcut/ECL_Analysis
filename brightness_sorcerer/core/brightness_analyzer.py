"""Brightness analysis engine for Brightness Sorcerer."""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import List, Optional, Callable, Dict, Tuple, Any
import logging

from ..models.roi import ROI
from ..models.analysis_result import AnalysisResult, FrameAnalysis
from ..utils.color_utils import calculate_brightness_stats, detect_brightness_threshold
from ..utils.math_utils import find_analysis_range
from .video_processor import VideoProcessor

# Constants
AUTO_DETECT_BASELINE_PERCENTILE = 5
DEFAULT_MANUAL_THRESHOLD = 5.0
DEFAULT_BRIGHTNESS_NOISE_FLOOR = 10.0
MIN_CONSECUTIVE_FRAMES = 3


class BrightnessAnalyzer:
    """Core brightness analysis engine."""
    
    def __init__(self, video_processor: VideoProcessor):
        self.video_processor = video_processor
        self.manual_threshold = DEFAULT_MANUAL_THRESHOLD
        self.noise_floor = DEFAULT_BRIGHTNESS_NOISE_FLOOR
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None
        self.cancel_requested = False
        
        # Analysis state
        self.current_result: Optional[AnalysisResult] = None
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback
    
    def cancel_analysis(self):
        """Request cancellation of current analysis."""
        self.cancel_requested = True
    
    def auto_detect_frame_range(self, rois: List[ROI], 
                               background_roi: Optional[ROI] = None) -> Optional[Tuple[int, int]]:
        """Auto-detect optimal frame range for analysis."""
        if not rois or not self.video_processor.is_loaded():
            return None
        
        try:
            self.cancel_requested = False
            total_frames = self.video_processor.total_frames
            
            # Sample frames for detection (every 10th frame for speed)
            sample_frames = list(range(0, total_frames, 10))
            brightness_values = []
            
            # Determine threshold
            threshold = self.manual_threshold
            if background_roi:
                frame = self.video_processor.get_current_frame()
                if frame is not None:
                    bg_region = background_roi.extract_region(frame)
                    if bg_region is not None:
                        threshold = detect_brightness_threshold(bg_region, self.manual_threshold, self.noise_floor)
            
            # Analyze sample frames
            for i, frame_idx in enumerate(sample_frames):
                if self.cancel_requested:
                    return None
                
                if self.progress_callback:
                    self.progress_callback(i, len(sample_frames), f"Scanning frame {frame_idx}")
                
                frame = self.video_processor.get_frame_at_index(frame_idx)
                if frame is None:
                    continue
                
                # Calculate average brightness across all ROIs
                total_brightness = 0.0
                valid_rois = 0
                
                for roi in rois:
                    roi_region = roi.extract_region(frame)
                    if roi_region is not None:
                        stats = calculate_brightness_stats(roi_region, self.noise_floor)
                        total_brightness += stats['mean']
                        valid_rois += 1
                
                avg_brightness = total_brightness / valid_rois if valid_rois > 0 else 0.0
                brightness_values.append(avg_brightness)
            
            # Find analysis range
            analysis_range = find_analysis_range(brightness_values, threshold, MIN_CONSECUTIVE_FRAMES)
            
            if analysis_range:
                # Convert sample indices back to actual frame indices
                start_sample, end_sample = analysis_range
                start_frame = sample_frames[start_sample]
                end_frame = min(sample_frames[end_sample], total_frames - 1)
                
                logging.info(f"Auto-detected frame range: {start_frame} - {end_frame}")
                return (start_frame, end_frame)
            
            return None
            
        except Exception as e:
            logging.error(f"Error in auto-detection: {e}")
            return None
    
    def analyze_brightness(self, rois: List[ROI], start_frame: int, end_frame: int,
                          output_directory: str, background_roi: Optional[ROI] = None) -> Optional[AnalysisResult]:
        """Perform complete brightness analysis."""
        if not rois or not self.video_processor.is_loaded():
            return None
        
        try:
            self.cancel_requested = False
            
            # Initialize result
            video_info = self.video_processor.get_video_info()
            roi_labels = [roi.label for roi in rois]
            
            result = AnalysisResult(
                video_path=video_info['path'],
                start_frame=start_frame,
                end_frame=end_frame,
                total_frames=end_frame - start_frame + 1,
                fps=video_info['fps'],
                roi_labels=roi_labels,
                noise_floor=self.noise_floor,
                background_roi_label=background_roi.label if background_roi else None
            )
            
            # Calculate threshold
            threshold = self.manual_threshold
            if background_roi:
                frame = self.video_processor.get_frame_at_index(start_frame)
                if frame:
                    bg_region = background_roi.extract_region(frame)
                    if bg_region is not None:
                        threshold = detect_brightness_threshold(bg_region, self.manual_threshold, self.noise_floor)
            
            result.brightness_threshold = threshold
            
            # Analyze each frame
            total_frames = end_frame - start_frame + 1
            start_time = time.time()
            
            for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                if self.cancel_requested:
                    return None
                
                # Progress reporting
                if self.progress_callback and i % 10 == 0:
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        frames_per_sec = i / elapsed
                        remaining_frames = total_frames - i
                        eta_seconds = remaining_frames / frames_per_sec if frames_per_sec > 0 else 0
                        eta_str = f"ETA: {eta_seconds:.0f}s" if eta_seconds > 0 else ""
                    else:
                        eta_str = ""
                    
                    self.progress_callback(i, total_frames, f"Analyzing frame {frame_idx} {eta_str}")
                
                # Get frame
                frame = self.video_processor.get_frame_at_index(frame_idx)
                if frame is None:
                    continue
                
                # Calculate timestamp
                timestamp = frame_idx / video_info['fps'] if video_info['fps'] > 0 else 0
                
                # Analyze each ROI
                frame_analysis = FrameAnalysis(frame_idx, timestamp)
                
                for roi in rois:
                    roi_region = roi.extract_region(frame)
                    if roi_region is not None:
                        stats = calculate_brightness_stats(roi_region, self.noise_floor)
                        frame_analysis.roi_stats[roi.label] = stats
                
                result.add_frame_analysis(frame_analysis)
            
            # Calculate summary statistics
            result.calculate_summary_statistics()
            
            # Save results
            self._save_analysis_results(result, output_directory)
            
            # Generate plots
            self._generate_plots(result, output_directory)
            
            self.current_result = result
            return result
            
        except Exception as e:
            logging.error(f"Error during brightness analysis: {e}")
            return None
    
    def _save_analysis_results(self, result: AnalysisResult, output_directory: str):
        """Save analysis results to files."""
        try:
            # Create base filename
            video_name = os.path.splitext(os.path.basename(result.video_path))[0]
            timestamp = result.analysis_timestamp.strftime("%Y%m%d_%H%M%S")
            base_filename = f"{video_name}_analysis_{timestamp}"
            
            # Save CSV
            csv_path = os.path.join(output_directory, f"{base_filename}.csv")
            result.save_csv(csv_path)
            logging.info(f"Saved CSV: {csv_path}")
            
            # Save JSON metadata
            json_path = os.path.join(output_directory, f"{base_filename}_metadata.json")
            result.save_json(json_path)
            logging.info(f"Saved metadata: {json_path}")
            
        except Exception as e:
            logging.error(f"Error saving analysis results: {e}")
    
    def _generate_plots(self, result: AnalysisResult, output_directory: str):
        """Generate analysis plots."""
        try:
            # Create base filename
            video_name = os.path.splitext(os.path.basename(result.video_path))[0]
            timestamp = result.analysis_timestamp.strftime("%Y%m%d_%H%M%S")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'Brightness Analysis: {video_name}', fontsize=16, fontweight='bold')
            
            # Get timestamps for x-axis
            timestamps = result.get_timestamps()
            
            # Plot 1: Brightness trends
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(result.roi_labels)))
            
            for i, roi_label in enumerate(result.roi_labels):
                mean_values = result.get_roi_timeseries(roi_label, 'mean')
                median_values = result.get_roi_timeseries(roi_label, 'median')
                
                color = colors[i]
                ax1.plot(timestamps, mean_values, label=f'{roi_label} (Mean)', 
                        color=color, linewidth=2)
                ax1.plot(timestamps, median_values, label=f'{roi_label} (Median)', 
                        color=color, linewidth=1, linestyle='--', alpha=0.7)
            
            # Add threshold line
            if result.brightness_threshold:
                ax1.axhline(y=result.brightness_threshold, color='red', linestyle=':',
                           label=f'Threshold ({result.brightness_threshold:.1f})')
            
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Brightness (L* units)')
            ax1.set_title('Brightness Trends Over Time')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Mean-Median differences
            for i, roi_label in enumerate(result.roi_labels):
                mean_values = result.get_roi_timeseries(roi_label, 'mean')
                median_values = result.get_roi_timeseries(roi_label, 'median')
                differences = [m - med for m, med in zip(mean_values, median_values)]
                
                color = colors[i]
                ax2.plot(timestamps, differences, label=f'{roi_label}', 
                        color=color, linewidth=1.5)
            
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Mean - Median Difference')
            ax2.set_title('Brightness Distribution Analysis')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Add analysis info text
            info_text = []
            info_text.append(f"Video: {os.path.basename(result.video_path)}")
            info_text.append(f"Frames: {result.start_frame} - {result.end_frame}")
            info_text.append(f"Duration: {result.get_analysis_duration():.1f}s")
            info_text.append(f"Noise Floor: {result.noise_floor:.1f}")
            if result.brightness_threshold:
                info_text.append(f"Threshold: {result.brightness_threshold:.1f}")
            
            fig.text(0.02, 0.02, '\\n'.join(info_text), fontsize=9, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(right=0.85, bottom=0.15)
            
            # Save plot
            plot_path = os.path.join(output_directory, f"{video_name}_analysis_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved plot: {plot_path}")
            
        except Exception as e:
            logging.error(f"Error generating plots: {e}")
    
    def get_analysis_summary(self, result: Optional[AnalysisResult]) -> Dict[str, Any]:
        """Get human-readable analysis summary."""
        if result is None:
            return {}
        
        summary = {
            'video_info': {
                'filename': os.path.basename(result.video_path),
                'duration_analyzed': f"{result.get_analysis_duration():.1f}s",
                'frame_range': f"{result.start_frame} - {result.end_frame}",
                'total_frames': result.total_frames
            },
            'analysis_parameters': {
                'noise_floor': result.noise_floor,
                'brightness_threshold': result.brightness_threshold,
                'roi_count': len(result.roi_labels)
            },
            'roi_summaries': {}
        }
        
        # Add ROI summaries
        for roi_label in result.roi_labels:
            roi_stats = result.summary_stats.get(roi_label, {})
            mean_stats = roi_stats.get('mean_brightness', {})
            
            if mean_stats:
                peak_frames = result.get_peak_frames(roi_label)
                summary['roi_summaries'][roi_label] = {
                    'average_brightness': f"{mean_stats.get('overall_mean', 0):.1f}",
                    'brightness_range': f"{mean_stats.get('range', 0):.1f}",
                    'peak_frames_count': len(peak_frames),
                    'peak_percentage': f"{(len(peak_frames) / result.total_frames * 100):.1f}%"
                }
        
        return summary