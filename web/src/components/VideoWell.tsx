import { useCallback, useEffect, useRef, useState } from 'react'
import { frameUrl } from '../api'
import { BACKGROUND_ROI_COLOR, dragRoi, hitTestRoi, normalizeRoi, roiArea, roiColor } from '../roi'
import type { Handle } from '../roi'
import type { Roi, VideoMeta } from '../types'

const MIN_ROI_AREA = 16

interface DragState {
  roiId: number | 'new'
  handle: Handle
  lastX: number
  lastY: number
  draft?: { x1: number; y1: number; x2: number; y2: number }
}

interface VideoWellProps {
  video: VideoMeta | null
  frame: number
  previewThreshold?: number
  rois: Roi[]
  selectedRoiId: number | null
  backgroundRoiId: number | null
  drawMode: boolean
  onSelect: (id: number | null) => void
  onRoiChange: (id: number, patch: Partial<Roi>) => void
  onRoiDrawn: (rect: { x1: number; y1: number; x2: number; y2: number }) => void
  onOpenClick: () => void
}

export function VideoWell({
  video,
  frame,
  previewThreshold,
  rois,
  selectedRoiId,
  backgroundRoiId,
  drawMode,
  onSelect,
  onRoiChange,
  onRoiDrawn,
  onOpenClick,
}: VideoWellProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const [imageVersion, setImageVersion] = useState(0)
  const [drag, setDrag] = useState<DragState | null>(null)

  // Load the current frame; keep the previous image on screen until the new
  // one decodes so scrubbing never flashes black.
  useEffect(() => {
    if (!video) {
      imageRef.current = null
      setImageVersion((version) => version + 1)
      return
    }
    let stale = false
    const image = new Image()
    image.onload = () => {
      if (stale) return
      imageRef.current = image
      setImageVersion((version) => version + 1)
    }
    image.src = frameUrl(video.video_id, frame, previewThreshold)
    return () => {
      stale = true
    }
  }, [video, frame, previewThreshold])

  // Repaint the canvas whenever the frame image or any overlay input changes.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !video) return
    const context = canvas.getContext('2d')
    if (!context) return

    canvas.width = video.width
    canvas.height = video.height
    context.fillStyle = '#0a0d10'
    context.fillRect(0, 0, canvas.width, canvas.height)
    if (imageRef.current) {
      context.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height)
    }

    const drawRect = (
      rect: { x1: number; y1: number; x2: number; y2: number },
      color: string,
      options: { selected?: boolean; dashed?: boolean; label?: string },
    ) => {
      const width = rect.x2 - rect.x1
      const height = rect.y2 - rect.y1
      context.save()
      context.lineWidth = options.selected ? 2.5 : 1.5
      context.strokeStyle = color
      if (options.dashed) context.setLineDash([6, 4])
      context.strokeRect(rect.x1 + 0.5, rect.y1 + 0.5, width, height)
      context.setLineDash([])

      if (options.label) {
        context.font = '600 12px "IBM Plex Mono", monospace'
        const metrics = context.measureText(options.label)
        const labelY = rect.y1 >= 18 ? rect.y1 - 5 : rect.y2 + 14
        context.fillStyle = 'rgba(10, 13, 16, 0.8)'
        context.fillRect(rect.x1, labelY - 11, metrics.width + 8, 15)
        context.fillStyle = color
        context.fillText(options.label, rect.x1 + 4, labelY)
      }

      if (options.selected) {
        context.fillStyle = color
        for (const [cx, cy] of [
          [rect.x1, rect.y1],
          [rect.x2, rect.y1],
          [rect.x1, rect.y2],
          [rect.x2, rect.y2],
        ]) {
          context.fillRect(cx - 3, cy - 3, 6, 6)
        }
      }
      context.restore()
    }

    for (const roi of rois) {
      const isBackground = roi.id === backgroundRoiId
      drawRect(roi, isBackground ? BACKGROUND_ROI_COLOR : roiColor(roi.id), {
        selected: roi.id === selectedRoiId,
        dashed: isBackground,
        label: isBackground ? `${roi.name} · BG` : roi.name,
      })
    }

    if (drag?.draft) {
      drawRect(normalizeRoi(drag.draft, video.width, video.height), '#53d8ff', {
        dashed: true,
      })
    }
  }, [video, rois, selectedRoiId, backgroundRoiId, drag, imageVersion])

  const toFrameCoords = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
      scale: scaleX,
    }
  }, [])

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      if (!video) return
      event.currentTarget.setPointerCapture(event.pointerId)
      const { x, y, scale } = toFrameCoords(event)

      if (drawMode) {
        setDrag({ roiId: 'new', handle: 'se', lastX: x, lastY: y, draft: { x1: x, y1: y, x2: x, y2: y } })
        return
      }

      // Topmost ROI wins: iterate back-to-front.
      for (let i = rois.length - 1; i >= 0; i -= 1) {
        const roi = rois[i]
        const handle = hitTestRoi(roi, x, y, scale)
        if (handle !== null) {
          onSelect(roi.id)
          setDrag({ roiId: roi.id, handle, lastX: x, lastY: y })
          return
        }
      }
      onSelect(null)
    },
    [video, drawMode, rois, onSelect, toFrameCoords],
  )

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      if (!drag || !video) return
      const { x, y } = toFrameCoords(event)

      if (drag.roiId === 'new' && drag.draft) {
        setDrag({ ...drag, draft: { ...drag.draft, x2: x, y2: y } })
        return
      }

      const roi = rois.find((candidate) => candidate.id === drag.roiId)
      if (!roi) return
      const moved = dragRoi(roi, drag.handle, x - drag.lastX, y - drag.lastY, video.width, video.height)
      onRoiChange(roi.id, moved)
      setDrag({ ...drag, lastX: x, lastY: y })
    },
    [drag, video, rois, onRoiChange, toFrameCoords],
  )

  const onPointerUp = useCallback(() => {
    if (drag?.roiId === 'new' && drag.draft && video) {
      const rect = normalizeRoi(drag.draft, video.width, video.height)
      if (roiArea(rect) >= MIN_ROI_AREA) {
        onRoiDrawn(rect)
      }
    }
    setDrag(null)
  }, [drag, video, onRoiDrawn])

  if (!video) {
    return (
      <div className="well">
        <div className="well-empty">
          <div className="glyph">◈</div>
          <div>
            <div style={{ fontWeight: 600, color: 'var(--text)' }}>No video loaded</div>
            <div>Open an experiment recording to begin.</div>
          </div>
          <button className="btn btn-primary" onClick={onOpenClick}>
            Open video
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="well">
      <canvas
        ref={canvasRef}
        style={{ cursor: drawMode ? 'crosshair' : 'default' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
      {drawMode && <div className="well-hint">Drag to draw a region · Esc to cancel</div>}
      {previewThreshold !== undefined && (
        <div className="well-hint" style={{ top: 'auto', bottom: 10 }}>
          Threshold preview · pixels above L* {previewThreshold.toFixed(1)} highlighted
        </div>
      )}
    </div>
  )
}
