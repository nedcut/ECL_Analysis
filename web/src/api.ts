import type {
  AnalysisSettings,
  DetectedBeep,
  ExportResponse,
  FsListing,
  JobStatus,
  Roi,
  VideoMeta,
} from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (typeof body?.detail === 'string') detail = body.detail
    } catch {
      // Non-JSON error body; keep the status text.
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export function openVideo(path: string): Promise<VideoMeta> {
  return request('/api/videos', { method: 'POST', body: JSON.stringify({ path }) })
}

export function frameUrl(videoId: string, index: number, threshold?: number): string {
  const params = threshold && threshold > 0 ? `?threshold=${threshold}` : ''
  return `/api/videos/${videoId}/frame/${index}${params}`
}

export function listDirectory(path?: string): Promise<FsListing> {
  const params = path ? `?path=${encodeURIComponent(path)}` : ''
  return request(`/api/fs${params}`)
}

function serializeRois(rois: Roi[]) {
  return rois.map((roi) => ({ x1: roi.x1, y1: roi.y1, x2: roi.x2, y2: roi.y2, name: roi.name }))
}

export function startAnalysis(
  videoId: string,
  rois: Roi[],
  backgroundRoiIdx: number | null,
  startFrame: number,
  endFrame: number,
  settings: AnalysisSettings,
  maskJobId?: string | null,
): Promise<{ job_id: string }> {
  return request(`/api/videos/${videoId}/analyze`, {
    method: 'POST',
    body: JSON.stringify({
      rois: serializeRois(rois),
      background_roi_idx: backgroundRoiIdx,
      start_frame: startFrame,
      end_frame: endFrame,
      mask_job_id: maskJobId ?? null,
      ...settings,
    }),
  })
}

export function startMaskScan(
  videoId: string,
  mode: 'global' | 'per_roi',
  rois: Roi[],
  backgroundRoiIdx: number | null,
  startFrame: number,
  endFrame: number,
  step: number,
  settings: AnalysisSettings,
): Promise<{ job_id: string }> {
  return request(`/api/videos/${videoId}/mask-scan`, {
    method: 'POST',
    body: JSON.stringify({
      mode,
      rois: serializeRois(rois),
      background_roi_idx: backgroundRoiIdx,
      start_frame: startFrame,
      end_frame: endFrame,
      step,
      background_percentile: settings.background_percentile,
      morphological_kernel_size: settings.morphological_kernel_size,
    }),
  })
}

export function detectRange(videoId: string, expectedDuration: number): Promise<{ beeps: DetectedBeep[] }> {
  return request(`/api/videos/${videoId}/detect-range`, {
    method: 'POST',
    body: JSON.stringify({ expected_duration: expectedDuration }),
  })
}

export function jobStatus(jobId: string): Promise<JobStatus> {
  return request(`/api/jobs/${jobId}`)
}

export function cancelJob(jobId: string): Promise<{ cancelled: boolean }> {
  return request(`/api/jobs/${jobId}/cancel`, { method: 'POST' })
}

export interface ExportRequestBody {
  analysis_name: string
  save_dir?: string
  csv: boolean
  json_export: boolean
  plot: boolean
  interactive_plot: boolean
}

export function exportAnalysis(jobId: string, body: ExportRequestBody): Promise<ExportResponse> {
  return request(`/api/jobs/${jobId}/export`, { method: 'POST', body: JSON.stringify(body) })
}

export function exportedFileUrl(jobId: string, path: string): string {
  return `/api/jobs/${jobId}/files?path=${encodeURIComponent(path)}`
}
