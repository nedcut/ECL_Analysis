import { useState } from 'react'
import * as api from '../api'
import type { FrameRange } from '../App'
import type { AnalysisSettings, DetectedBeep, JobStatus, Roi, VideoMeta } from '../types'

interface AnalyzePanelProps {
  video: VideoMeta | null
  rois: Roi[]
  backgroundRoiId: number | null
  range: FrameRange
  settings: AnalysisSettings
  job: JobStatus | null
  maskJobId: string | null
  onJobChange: (job: JobStatus | null) => void
  onRangeDetected: (range: FrameRange) => void
}

export function AnalyzePanel({
  video,
  rois,
  backgroundRoiId,
  range,
  settings,
  job,
  maskJobId,
  onJobChange,
  onRangeDetected,
}: AnalyzePanelProps) {
  const [startError, setStartError] = useState<string | null>(null)
  const [expectedDuration, setExpectedDuration] = useState(10)
  const [detecting, setDetecting] = useState(false)
  const [beeps, setBeeps] = useState<DetectedBeep[] | null>(null)
  const [detectError, setDetectError] = useState<string | null>(null)

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
        maskJobId,
      )
      onJobChange(await api.jobStatus(job_id))
    } catch (error) {
      setStartError(error instanceof Error ? error.message : String(error))
    }
  }

  const detect = async () => {
    if (!video) return
    setDetecting(true)
    setDetectError(null)
    setBeeps(null)
    try {
      const { beeps: found } = await api.detectRange(video.video_id, expectedDuration)
      setBeeps(found)
      if (found.length === 1) {
        onRangeDetected({ start: found[0].start_frame, end: found[0].end_frame })
      }
    } catch (error) {
      setDetectError(error instanceof Error ? error.message : String(error))
    } finally {
      setDetecting(false)
    }
  }

  const progress = job && job.progress.total > 0 ? job.progress.done / job.progress.total : 0

  return (
    <section className="panel analyze-panel">
      <div className="panel-title">
        <span className="eyebrow">Run</span>
        <span className="mono" style={{ fontSize: 11, color: 'var(--faint)' }}>
          frames {range.start}–{range.end}
          {maskJobId ? ' · masked' : ''}
        </span>
      </div>

      <div className="field">
        <label htmlFor="expected-duration">Expected run duration (s)</label>
        <input
          id="expected-duration"
          type="number"
          min={0.1}
          step={0.5}
          value={expectedDuration}
          onChange={(event) => setExpectedDuration(Number(event.target.value))}
        />
      </div>
      <button
        className="btn"
        style={{ width: '100%', justifyContent: 'center', marginBottom: 10 }}
        disabled={!video || detecting || expectedDuration <= 0}
        onClick={() => void detect()}
      >
        {detecting ? 'Listening for completion beeps…' : 'Detect range from audio'}
      </button>
      {beeps !== null && beeps.length === 0 && (
        <div className="field-hint">No completion beeps found in this video's audio.</div>
      )}
      {beeps !== null && beeps.length === 1 && (
        <div className="ok-text mono">
          Range set from beep at {beeps[0].beep_time.toFixed(1)}s
        </div>
      )}
      {beeps !== null && beeps.length > 1 && (
        <div style={{ marginBottom: 8 }}>
          <div className="field-hint">Several beeps found — pick the run's end:</div>
          {beeps.map((beep) => (
            <button
              key={beep.beep_time}
              className="btn btn-ghost mono"
              style={{ display: 'block', width: '100%', textAlign: 'left' }}
              onClick={() => onRangeDetected({ start: beep.start_frame, end: beep.end_frame })}
            >
              {beep.beep_time.toFixed(1)}s → frames {beep.start_frame}–{beep.end_frame}
            </button>
          ))}
        </div>
      )}
      {detectError && <div className="error-text">{detectError}</div>}

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
      {job?.status === 'done' && job.result && 'frames_processed' in job.result && (
        <div className="ok-text mono">
          {job.result.frames_processed} frames in {job.result.elapsed_seconds.toFixed(1)}s
          {job.result.truncated ? ' · truncated at end of file' : ''}
        </div>
      )}
      {startError && <div className="error-text">{startError}</div>}
    </section>
  )
}
