export interface VideoMeta {
  video_id: string
  path: string
  name: string
  frame_count: number
  fps: number
  width: number
  height: number
  duration_seconds: number
}

export interface Roi {
  id: number
  name: string
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface AnalysisSettings {
  background_percentile: number
  morphological_kernel_size: number
  noise_floor_threshold: number
  manual_threshold: number
}

export const DEFAULT_SETTINGS: AnalysisSettings = {
  background_percentile: 90,
  morphological_kernel_size: 3,
  noise_floor_threshold: 0,
  manual_threshold: 0,
}

export interface RoiSeries {
  roi_index: number
  brightness_mean: number[]
  brightness_median: number[]
  blue_mean: number[]
  blue_median: number[]
}

export interface AnalysisResultPayload {
  start_frame: number
  end_frame: number
  frames_processed: number
  total_frames: number
  truncated: boolean
  elapsed_seconds: number
  background_values_per_frame: number[]
  rois: RoiSeries[]
}

export interface JobStatus {
  job_id: string
  video_id: string
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  progress: { done: number; total: number }
  message: string
  error: string | null
  result?: AnalysisResultPayload
}

export interface FsListing {
  path: string
  parent: string | null
  dirs: string[]
  videos: string[]
}

export interface ExportResponse {
  save_dir: string
  out_paths: string[]
  summary_lines: string[]
  avg_brightness_summary: string[]
  plot_failed: boolean
}

export type MetricKey = 'brightness_mean' | 'brightness_median' | 'blue_mean' | 'blue_median'

export const METRIC_LABELS: Record<MetricKey, string> = {
  brightness_mean: 'L* mean',
  brightness_median: 'L* median',
  blue_mean: 'Blue mean',
  blue_median: 'Blue median',
}
