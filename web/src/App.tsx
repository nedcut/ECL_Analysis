import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'
import { AnalyzePanel } from './components/AnalyzePanel'
import { FilePicker } from './components/FilePicker'
import { MasksPanel } from './components/MasksPanel'
import { ResultsDrawer } from './components/ResultsDrawer'
import { RoiPanel } from './components/RoiPanel'
import { SettingsPanel } from './components/SettingsPanel'
import { Transport } from './components/Transport'
import { VideoWell } from './components/VideoWell'
import { nextRoiName } from './roi'
import type { AnalysisSettings, JobStatus, Roi, VideoMeta } from './types'
import { DEFAULT_SETTINGS } from './types'

export interface FrameRange {
  start: number
  end: number
}

export function App() {
  const [video, setVideo] = useState<VideoMeta | null>(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [rois, setRois] = useState<Roi[]>([])
  const [selectedRoiId, setSelectedRoiId] = useState<number | null>(null)
  const [backgroundRoiId, setBackgroundRoiId] = useState<number | null>(null)
  const [range, setRange] = useState<FrameRange>({ start: 0, end: 0 })
  const [settings, setSettings] = useState<AnalysisSettings>(DEFAULT_SETTINGS)
  const [previewThreshold, setPreviewThreshold] = useState(false)
  const [drawMode, setDrawMode] = useState(false)
  const [job, setJob] = useState<JobStatus | null>(null)
  const [pickerOpen, setPickerOpen] = useState(false)
  const [openError, setOpenError] = useState<string | null>(null)
  const [speed, setSpeed] = useState(1)
  const [maskJobId, setMaskJobId] = useState<string | null>(null)
  const [useMasks, setUseMasks] = useState(false)
  const nextRoiId = useRef(1)

  const lastFrame = video ? Math.max(0, video.frame_count - 1) : 0

  const openVideo = useCallback(async (path: string) => {
    setOpenError(null)
    try {
      const meta = await api.openVideo(path)
      setVideo(meta)
      setFrame(0)
      setPlaying(false)
      setRois([])
      setSelectedRoiId(null)
      setBackgroundRoiId(null)
      setRange({ start: 0, end: Math.max(0, meta.frame_count - 1) })
      setJob(null)
      setPickerOpen(false)
    } catch (error) {
      setOpenError(error instanceof Error ? error.message : String(error))
    }
  }, [])

  // Playback. At >=1× we tick at native fps and skip frames (server decode
  // can't exceed native rate); below 1× we slow the tick instead.
  useEffect(() => {
    if (!playing || !video) return
    const fps = video.fps > 0 ? video.fps : 30
    const stride = speed >= 1 ? Math.round(speed) : 1
    const intervalMs = speed >= 1 ? 1000 / fps : 1000 / (fps * speed)
    const interval = window.setInterval(() => {
      setFrame((current) => {
        if (current >= lastFrame) {
          setPlaying(false)
          return current
        }
        return Math.min(current + stride, lastFrame)
      })
    }, intervalMs)
    return () => window.clearInterval(interval)
  }, [playing, video, lastFrame, speed])

  // Captured masks are pinned to region geometry; any move/resize/add/delete
  // (or background change) invalidates them so analysis can't silently use
  // masks that no longer match.
  const geometrySignature = JSON.stringify([
    rois.map((roi) => [roi.id, roi.x1, roi.y1, roi.x2, roi.y2]),
    backgroundRoiId,
  ])
  const lastGeometry = useRef(geometrySignature)
  useEffect(() => {
    if (lastGeometry.current !== geometrySignature) {
      lastGeometry.current = geometrySignature
      setMaskJobId(null)
      setUseMasks(false)
    }
  }, [geometrySignature])

  // Poll a queued/running job until it settles.
  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'running')) return
    const interval = window.setInterval(async () => {
      try {
        setJob(await api.jobStatus(job.job_id))
      } catch {
        // Poll again on transient failures; the next tick will retry.
      }
    }, 300)
    return () => window.clearInterval(interval)
  }, [job])

  const seek = useCallback(
    (target: number) => {
      setFrame(Math.max(0, Math.min(Math.round(target), lastFrame)))
    },
    [lastFrame],
  )

  // Keyboard transport: arrows step, PageUp/Down jump 10, space plays — the
  // same bindings the desktop app trained users on.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA') {
        return
      }
      if (!video) return
      switch (event.key) {
        case 'ArrowLeft':
          seek(frame - 1)
          break
        case 'ArrowRight':
          seek(frame + 1)
          break
        case 'PageUp':
          seek(frame - 10)
          break
        case 'PageDown':
          seek(frame + 10)
          break
        case ' ':
          setPlaying((value) => !value)
          break
        case 'Escape':
          setDrawMode(false)
          return
        default:
          return
      }
      event.preventDefault()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [video, frame, seek])

  const addRoi = useCallback(
    (rect: { x1: number; y1: number; x2: number; y2: number }) => {
      const roi: Roi = { id: nextRoiId.current, name: '', ...rect }
      roi.name = nextRoiName(rois)
      nextRoiId.current += 1
      setRois((current) => [...current, roi])
      setSelectedRoiId(roi.id)
      setDrawMode(false)
    },
    [rois],
  )

  const updateRoi = useCallback((id: number, patch: Partial<Roi>) => {
    setRois((current) => current.map((roi) => (roi.id === id ? { ...roi, ...patch } : roi)))
  }, [])

  const deleteRoi = useCallback(
    (id: number) => {
      setRois((current) => current.filter((roi) => roi.id !== id))
      if (selectedRoiId === id) setSelectedRoiId(null)
      if (backgroundRoiId === id) setBackgroundRoiId(null)
    },
    [selectedRoiId, backgroundRoiId],
  )

  // Manual threshold only gates analysis when no background ROI exists, so the
  // live preview mirrors that rule instead of showing a mask that won't apply.
  const effectivePreview =
    previewThreshold && backgroundRoiId === null && settings.manual_threshold > 0
      ? settings.manual_threshold
      : undefined

  return (
    <div className="app">
      <header className="topbar">
        <div className="wordmark">
          <span className="lumen">◈</span> Brightness Sorcerer
        </div>
        {video ? (
          <div className="topbar-video">
            <span className="name">{video.name}</span>
            <span className="meta mono">
              {video.width}×{video.height} · {video.fps.toFixed(2)} fps · {video.frame_count} frames
            </span>
          </div>
        ) : (
          <div className="topbar-video" />
        )}
        <button className="btn" onClick={() => setPickerOpen(true)}>
          Open video
        </button>
      </header>

      <div className="workspace">
        <main className="stage">
          <VideoWell
            video={video}
            frame={frame}
            previewThreshold={effectivePreview}
            rois={rois}
            selectedRoiId={selectedRoiId}
            backgroundRoiId={backgroundRoiId}
            drawMode={drawMode}
            onSelect={setSelectedRoiId}
            onRoiChange={updateRoi}
            onRoiDrawn={addRoi}
            onOpenClick={() => setPickerOpen(true)}
          />
          {video && (
            <Transport
              frame={frame}
              frameCount={video.frame_count}
              fps={video.fps}
              playing={playing}
              speed={speed}
              range={range}
              onSeek={seek}
              onTogglePlay={() => setPlaying((value) => !value)}
              onSpeedChange={setSpeed}
              onRangeChange={setRange}
            />
          )}
          {job?.status === 'done' && job.result && (
            <ResultsDrawer job={job} rois={rois} backgroundRoiId={backgroundRoiId} />
          )}
        </main>

        <aside className="sidebar">
          <RoiPanel
            rois={rois}
            selectedRoiId={selectedRoiId}
            backgroundRoiId={backgroundRoiId}
            drawMode={drawMode}
            disabled={!video}
            onArmDraw={() => setDrawMode(true)}
            onSelect={setSelectedRoiId}
            onRename={(id, name) => updateRoi(id, { name })}
            onSetBackground={(id) => setBackgroundRoiId((current) => (current === id ? null : id))}
            onDelete={deleteRoi}
          />
          <SettingsPanel
            settings={settings}
            onChange={setSettings}
            backgroundRoiSet={backgroundRoiId !== null}
            previewThreshold={previewThreshold}
            onPreviewThresholdChange={setPreviewThreshold}
          />
          <MasksPanel
            video={video}
            rois={rois}
            backgroundRoiId={backgroundRoiId}
            range={range}
            settings={settings}
            maskJobId={maskJobId}
            useMasks={useMasks}
            onMaskJobChange={setMaskJobId}
            onUseMasksChange={setUseMasks}
            onSeek={seek}
          />
          <AnalyzePanel
            video={video}
            rois={rois}
            backgroundRoiId={backgroundRoiId}
            range={range}
            settings={settings}
            job={job}
            maskJobId={useMasks ? maskJobId : null}
            onJobChange={setJob}
            onRangeDetected={(detected) => {
              setRange(detected)
              seek(detected.start)
            }}
          />
        </aside>
      </div>

      {pickerOpen && (
        <FilePicker
          error={openError}
          onPick={openVideo}
          onClose={() => {
            setPickerOpen(false)
            setOpenError(null)
          }}
        />
      )}
    </div>
  )
}
