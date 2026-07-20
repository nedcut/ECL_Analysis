import { useCallback, useRef } from 'react'
import type { FrameRange } from '../App'

export const PLAYBACK_SPEEDS = [0.25, 0.5, 1, 2, 4] as const

interface TransportProps {
  frame: number
  frameCount: number
  fps: number
  playing: boolean
  speed: number
  range: FrameRange
  onSeek: (frame: number) => void
  onTogglePlay: () => void
  onSpeedChange: (speed: number) => void
  onRangeChange: (range: FrameRange) => void
}

function pad(value: number, width: number): string {
  return String(value).padStart(width, '0')
}

export function Transport({
  frame,
  frameCount,
  fps,
  playing,
  speed,
  range,
  onSeek,
  onTogglePlay,
  onSpeedChange,
  onRangeChange,
}: TransportProps) {
  const scrubberRef = useRef<HTMLDivElement>(null)
  const lastFrame = Math.max(0, frameCount - 1)
  const digits = String(lastFrame).length
  const seconds = fps > 0 ? frame / fps : 0

  const frameFromEvent = useCallback(
    (clientX: number) => {
      const element = scrubberRef.current
      if (!element || lastFrame === 0) return 0
      const rect = element.getBoundingClientRect()
      const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
      return Math.round(ratio * lastFrame)
    },
    [lastFrame],
  )

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.currentTarget.setPointerCapture(event.pointerId)
      onSeek(frameFromEvent(event.clientX))
    },
    [frameFromEvent, onSeek],
  )

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.buttons !== 1) return
      onSeek(frameFromEvent(event.clientX))
    },
    [frameFromEvent, onSeek],
  )

  const toPercent = (value: number) => (lastFrame > 0 ? (value / lastFrame) * 100 : 0)

  return (
    <div className="transport">
      <button className="btn" onClick={onTogglePlay} aria-label={playing ? 'Pause' : 'Play'}>
        {playing ? '❚❚' : '▶'}
      </button>
      <select
        aria-label="Playback speed"
        className="mono"
        value={speed}
        onChange={(event) => onSpeedChange(Number(event.target.value))}
      >
        {PLAYBACK_SPEEDS.map((value) => (
          <option key={value} value={value}>
            {value}×
          </option>
        ))}
      </select>
      <span className="counter">
        {pad(frame, digits)}
        <span className="total"> / {lastFrame} · {seconds.toFixed(2)}s</span>
      </span>
      <div
        ref={scrubberRef}
        className="scrubber"
        role="slider"
        aria-label="Frame"
        aria-valuemin={0}
        aria-valuemax={lastFrame}
        aria-valuenow={frame}
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
      >
        <div className="track" />
        <div
          className="range"
          style={{
            left: `${toPercent(range.start)}%`,
            width: `${Math.max(0, toPercent(range.end) - toPercent(range.start))}%`,
          }}
        />
        <div className="playhead" style={{ left: `${toPercent(frame)}%` }} />
      </div>
      <span className="counter total mono">
        {pad(range.start, digits)}–{pad(range.end, digits)}
      </span>
      <div className="range-buttons">
        <button
          className="btn btn-ghost"
          title="Set analysis start to the current frame"
          onClick={() => onRangeChange({ start: frame, end: Math.max(frame, range.end) })}
        >
          Set start
        </button>
        <button
          className="btn btn-ghost"
          title="Set analysis end to the current frame"
          onClick={() => onRangeChange({ start: Math.min(frame, range.start), end: frame })}
        >
          Set end
        </button>
        <button
          className="btn btn-ghost"
          title="Reset the analysis range to the whole video"
          onClick={() => onRangeChange({ start: 0, end: lastFrame })}
        >
          Reset
        </button>
      </div>
    </div>
  )
}
