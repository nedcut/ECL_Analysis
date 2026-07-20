import { useEffect, useState } from 'react'
import * as api from '../api'
import type { FrameRange } from '../App'
import { useJob } from '../hooks'
import type {
  AnalysisSettings,
  BrightestFrameResultPayload,
  PerRoiMaskResultPayload,
  Roi,
  VideoMeta,
} from '../types'

interface MasksPanelProps {
  video: VideoMeta | null
  rois: Roi[]
  backgroundRoiId: number | null
  range: FrameRange
  settings: AnalysisSettings
  maskJobId: string | null
  useMasks: boolean
  onMaskJobChange: (jobId: string | null) => void
  onUseMasksChange: (useMasks: boolean) => void
  onSeek: (frame: number) => void
}

export function MasksPanel({
  video,
  rois,
  backgroundRoiId,
  range,
  settings,
  maskJobId,
  useMasks,
  onMaskJobChange,
  onUseMasksChange,
  onSeek,
}: MasksPanelProps) {
  const [step, setStep] = useState(5)
  const [globalJobId, setGlobalJobId] = useState<string | null>(null)
  const [captureJobId, setCaptureJobId] = useState<string | null>(null)
  const [scanError, setScanError] = useState<string | null>(null)

  const globalJob = useJob(globalJobId)
  const captureJob = useJob(captureJobId)

  const backgroundIdx = backgroundRoiId === null
    ? null
    : rois.findIndex((roi) => roi.id === backgroundRoiId)
  const analyzableCount = rois.filter((roi) => roi.id !== backgroundRoiId).length
  const scanning =
    globalJob?.status === 'queued' ||
    globalJob?.status === 'running' ||
    captureJob?.status === 'queued' ||
    captureJob?.status === 'running'
  const canScan = Boolean(video) && analyzableCount > 0 && !scanning

  // Jump to the brightest frame as soon as the global scan lands.
  useEffect(() => {
    if (globalJob?.status === 'done' && globalJob.result) {
      onSeek((globalJob.result as BrightestFrameResultPayload).brightest_frame_idx)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [globalJob?.status])

  // Register a finished capture with the app so analysis can use it.
  useEffect(() => {
    if (captureJob?.status === 'done' && captureJobId) {
      onMaskJobChange(captureJobId)
      onUseMasksChange(true)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captureJob?.status])

  const startScan = async (mode: 'global' | 'per_roi') => {
    if (!video) return
    setScanError(null)
    try {
      const { job_id } = await api.startMaskScan(
        video.video_id,
        mode,
        rois,
        backgroundIdx === -1 ? null : backgroundIdx,
        range.start,
        range.end,
        step,
        settings,
      )
      if (mode === 'global') setGlobalJobId(job_id)
      else {
        onMaskJobChange(null)
        setCaptureJobId(job_id)
      }
    } catch (error) {
      setScanError(error instanceof Error ? error.message : String(error))
    }
  }

  const activeScan = [globalJob, captureJob].find(
    (job) => job?.status === 'queued' || job?.status === 'running',
  )
  const captureResult =
    maskJobId && captureJob?.status === 'done'
      ? (captureJob.result as PerRoiMaskResultPayload)
      : null

  return (
    <section className="panel">
      <div className="panel-title">
        <span className="eyebrow">Masks</span>
        <span className="mono" style={{ fontSize: 11, color: 'var(--faint)' }}>
          <label htmlFor="scan-step" style={{ marginRight: 4 }}>
            step
          </label>
          <input
            id="scan-step"
            type="number"
            min={1}
            value={step}
            style={{ width: 52 }}
            onChange={(event) => setStep(Math.max(1, Number(event.target.value)))}
          />
        </span>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button className="btn" disabled={!canScan} onClick={() => void startScan('global')}>
          Jump to brightest frame
        </button>
        <button className="btn" disabled={!canScan} onClick={() => void startScan('per_roi')}>
          Capture region masks
        </button>
      </div>
      <div className="field-hint" style={{ marginTop: 6 }}>
        Masks lock analysis to the pixels lit at each region's brightest frame.
      </div>

      {activeScan && (
        <div className="progress">
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{
                width: `${
                  activeScan.progress.total > 0
                    ? (activeScan.progress.done / activeScan.progress.total) * 100
                    : 0
                }%`,
              }}
            />
          </div>
          <div className="progress-message">{activeScan.message || 'Scanning…'}</div>
        </div>
      )}

      {globalJob?.status === 'done' && globalJob.result && (
        <div className="ok-text mono">
          Brightest: frame {(globalJob.result as BrightestFrameResultPayload).brightest_frame_idx} (L*{' '}
          {(globalJob.result as BrightestFrameResultPayload).max_brightness.toFixed(1)})
        </div>
      )}

      {captureResult && (
        <>
          <label className="switch-row" style={{ marginTop: 8 }}>
            <input
              type="checkbox"
              checked={useMasks}
              onChange={(event) => onUseMasksChange(event.target.checked)}
            />
            Analyze inside captured masks
          </label>
          <ul className="export-paths">
            {rois.map((roi, index) =>
              captureResult.sources[index] !== null ? (
                <li key={roi.id}>
                  {roi.name}: frame {captureResult.sources[index]}
                  {captureResult.mask_coverage[index] !== null
                    ? ` · ${(captureResult.mask_coverage[index]! * 100).toFixed(0)}% of region`
                    : ''}
                </li>
              ) : null,
            )}
          </ul>
        </>
      )}
      {maskJobId === null && captureJob?.status === 'done' && (
        <div className="field-hint" style={{ marginTop: 6 }}>
          Regions changed since capture — masks were discarded. Recapture to use them.
        </div>
      )}

      {(globalJob?.status === 'error' || captureJob?.status === 'error') && (
        <div className="error-text">{globalJob?.error || captureJob?.error}</div>
      )}
      {scanError && <div className="error-text">{scanError}</div>}
    </section>
  )
}
