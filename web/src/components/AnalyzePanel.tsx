import { useState } from 'react'
import * as api from '../api'
import type { FrameRange } from '../App'
import type { AnalysisSettings, JobStatus, Roi, VideoMeta } from '../types'

interface AnalyzePanelProps {
  video: VideoMeta | null
  rois: Roi[]
  backgroundRoiId: number | null
  range: FrameRange
  settings: AnalysisSettings
  job: JobStatus | null
  onJobChange: (job: JobStatus | null) => void
}

export function AnalyzePanel({
  video,
  rois,
  backgroundRoiId,
  range,
  settings,
  job,
  onJobChange,
}: AnalyzePanelProps) {
  const [startError, setStartError] = useState<string | null>(null)

  const backgroundIdx = backgroundRoiId === null
    ? null
    : rois.findIndex((roi) => roi.id === backgroundRoiId)
  const analyzableCount = rois.filter((roi) => roi.id !== backgroundRoiId).length
  const running = job?.status === 'queued' || job?.status === 'running'
  const canAnalyze = Boolean(video) && analyzableCount > 0 && !running

  const start = async () => {
    if (!video) return
    setStartError(null)
    try {
      const { job_id } = await api.startAnalysis(
        video.video_id,
        rois,
        backgroundIdx === -1 ? null : backgroundIdx,
        range.start,
        range.end,
        settings,
      )
      onJobChange(await api.jobStatus(job_id))
    } catch (error) {
      setStartError(error instanceof Error ? error.message : String(error))
    }
  }

  const progress = job && job.progress.total > 0 ? job.progress.done / job.progress.total : 0

  return (
    <section className="panel analyze-panel">
      <div className="panel-title">
        <span className="eyebrow">Run</span>
        <span className="mono" style={{ fontSize: 11, color: 'var(--faint)' }}>
          frames {range.start}–{range.end}
        </span>
      </div>

      {running ? (
        <button className="btn btn-danger" style={{ width: '100%', justifyContent: 'center' }}
          onClick={() => job && api.cancelJob(job.job_id)}>
          Cancel analysis
        </button>
      ) : (
        <button className="btn btn-primary" disabled={!canAnalyze} onClick={start}>
          Analyze brightness
        </button>
      )}

      {!running && analyzableCount === 0 && (
        <div className="field-hint" style={{ marginTop: 8 }}>
          Draw at least one non-background region to analyze.
        </div>
      )}

      {running && (
        <div className="progress">
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
          </div>
          <div className="progress-message">
            {job?.message || `Frame ${job?.progress.done ?? 0} / ${job?.progress.total ?? 0}`}
          </div>
        </div>
      )}

      {job?.status === 'cancelled' && <div className="field-hint" style={{ marginTop: 8 }}>Analysis cancelled.</div>}
      {job?.status === 'error' && <div className="error-text">{job.error}</div>}
      {job?.status === 'done' && job.result && (
        <div className="ok-text mono">
          {job.result.frames_processed} frames in {job.result.elapsed_seconds.toFixed(1)}s
          {job.result.truncated ? ' · truncated at end of file' : ''}
        </div>
      )}
      {startError && <div className="error-text">{startError}</div>}
    </section>
  )
}
