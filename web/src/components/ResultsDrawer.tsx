import { useMemo, useRef, useState } from 'react'
import * as api from '../api'
import { roiColor } from '../roi'
import type { AnalysisResultPayload, ExportResponse, JobStatus, MetricKey, Roi } from '../types'
import { METRIC_LABELS } from '../types'

interface ResultsDrawerProps {
  job: JobStatus
  rois: Roi[]
  backgroundRoiId: number | null
}

interface SeriesView {
  name: string
  color: string
  values: number[]
}

const CHART_WIDTH = 800
const CHART_HEIGHT = 260
const MARGIN = { top: 12, right: 16, bottom: 28, left: 48 }

export function ResultsDrawer({ job, rois, backgroundRoiId }: ResultsDrawerProps) {
  const [metric, setMetric] = useState<MetricKey>('brightness_mean')
  const result = job.result as AnalysisResultPayload

  const series: SeriesView[] = useMemo(() => {
    const nonBackground = rois.filter((roi) => roi.id !== backgroundRoiId)
    return result.rois.map((roiSeries, index) => {
      const roi = rois[roiSeries.roi_index] ?? nonBackground[index]
      return {
        name: roi?.name ?? `ROI ${roiSeries.roi_index + 1}`,
        color: roi ? roiColor(roi.id) : roiColor(index),
        values: roiSeries[metric],
      }
    })
  }, [result, rois, backgroundRoiId, metric])

  return (
    <section className="results">
      <div className="results-header">
        <span className="eyebrow">Results · {METRIC_LABELS[metric]}</span>
        <div className="controls">
          <label className="eyebrow" htmlFor="metric-select">
            Metric
          </label>
          <select
            id="metric-select"
            value={metric}
            onChange={(event) => setMetric(event.target.value as MetricKey)}
          >
            {Object.entries(METRIC_LABELS).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </div>
      </div>
      <BrightnessChart series={series} startFrame={result.start_frame} metricLabel={METRIC_LABELS[metric]} />
      <ExportPanel job={job} />
    </section>
  )
}

function BrightnessChart({
  series,
  startFrame,
  metricLabel,
}: {
  series: SeriesView[]
  startFrame: number
  metricLabel: string
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)
  const [hoverPx, setHoverPx] = useState<{ x: number; y: number } | null>(null)

  const frameCount = Math.max(...series.map((entry) => entry.values.length), 0)
  const plotWidth = CHART_WIDTH - MARGIN.left - MARGIN.right
  const plotHeight = CHART_HEIGHT - MARGIN.top - MARGIN.bottom

  const [yMin, yMax] = useMemo(() => {
    let min = Infinity
    let max = -Infinity
    for (const entry of series) {
      for (const value of entry.values) {
        if (value < min) min = value
        if (value > max) max = value
      }
    }
    if (!Number.isFinite(min)) return [0, 1]
    if (min === max) return [min - 1, max + 1]
    const pad = (max - min) * 0.06
    return [min - pad, max + pad]
  }, [series])

  const xAt = (index: number) => MARGIN.left + (frameCount > 1 ? (index / (frameCount - 1)) * plotWidth : 0)
  const yAt = (value: number) => MARGIN.top + plotHeight - ((value - yMin) / (yMax - yMin)) * plotHeight

  const paths = useMemo(
    () =>
      series.map((entry) =>
        entry.values
          .map((value, index) => `${index === 0 ? 'M' : 'L'}${xAt(index).toFixed(1)},${yAt(value).toFixed(1)}`)
          .join(''),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [series, yMin, yMax, frameCount],
  )

  const yTicks = useMemo(() => {
    const ticks: number[] = []
    for (let i = 0; i <= 4; i += 1) ticks.push(yMin + ((yMax - yMin) * i) / 4)
    return ticks
  }, [yMin, yMax])

  const onMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const svg = event.currentTarget
    const rect = svg.getBoundingClientRect()
    const px = ((event.clientX - rect.left) / rect.width) * CHART_WIDTH
    if (frameCount < 1) return
    const ratio = (px - MARGIN.left) / plotWidth
    const index = Math.max(0, Math.min(frameCount - 1, Math.round(ratio * (frameCount - 1))))
    setHoverIndex(index)
    const wrapRect = wrapRef.current?.getBoundingClientRect()
    if (wrapRect) {
      setHoverPx({ x: event.clientX - wrapRect.left, y: event.clientY - wrapRect.top })
    }
  }

  if (frameCount === 0) {
    return <div className="chart-wrap">No data points to plot.</div>
  }

  return (
    <div className="chart-wrap" ref={wrapRef} style={{ position: 'relative' }}>
      <svg
        className="chart-svg"
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        role="img"
        aria-label={`${metricLabel} per frame for ${series.length} region${series.length === 1 ? '' : 's'}`}
        onMouseMove={onMove}
        onMouseLeave={() => {
          setHoverIndex(null)
          setHoverPx(null)
        }}
      >
        {yTicks.map((tick) => (
          <g key={tick}>
            <line
              x1={MARGIN.left}
              x2={CHART_WIDTH - MARGIN.right}
              y1={yAt(tick)}
              y2={yAt(tick)}
              stroke="var(--line)"
              strokeWidth={1}
            />
            <text
              x={MARGIN.left - 8}
              y={yAt(tick) + 3}
              textAnchor="end"
              fontSize={10}
              fill="var(--faint)"
              fontFamily="var(--font-data)"
            >
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        {[0, 0.5, 1].map((ratio) => {
          const index = Math.round(ratio * (frameCount - 1))
          return (
            <text
              key={ratio}
              x={xAt(index)}
              y={CHART_HEIGHT - 8}
              textAnchor="middle"
              fontSize={10}
              fill="var(--faint)"
              fontFamily="var(--font-data)"
            >
              {startFrame + index}
            </text>
          )
        })}
        {series.map((entry, index) => (
          <path key={entry.name} d={paths[index]} fill="none" stroke={entry.color} strokeWidth={2} />
        ))}
        {hoverIndex !== null && (
          <g>
            <line
              x1={xAt(hoverIndex)}
              x2={xAt(hoverIndex)}
              y1={MARGIN.top}
              y2={CHART_HEIGHT - MARGIN.bottom}
              stroke="var(--dim)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            {series.map((entry) =>
              hoverIndex < entry.values.length ? (
                <circle
                  key={entry.name}
                  cx={xAt(hoverIndex)}
                  cy={yAt(entry.values[hoverIndex])}
                  r={4}
                  fill={entry.color}
                  stroke="var(--ink-1)"
                  strokeWidth={2}
                />
              ) : null,
            )}
          </g>
        )}
      </svg>
      {hoverIndex !== null && hoverPx && (
        <div
          className="tooltip"
          style={{
            left: Math.min(hoverPx.x + 14, (wrapRef.current?.clientWidth ?? 400) - 160),
            top: hoverPx.y + 10,
          }}
        >
          <div style={{ color: 'var(--faint)' }}>frame {startFrame + hoverIndex}</div>
          {series.map((entry) => (
            <div className="row" key={entry.name}>
              <span className="roi-chip" style={{ background: entry.color }} />
              {entry.name}: {hoverIndex < entry.values.length ? entry.values[hoverIndex].toFixed(2) : '—'}
            </div>
          ))}
        </div>
      )}
      {series.length >= 2 && (
        <div className="legend">
          {series.map((entry) => (
            <span className="item" key={entry.name}>
              <span className="roi-chip" style={{ background: entry.color }} />
              {entry.name}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function ExportPanel({ job }: { job: JobStatus }) {
  const [analysisName, setAnalysisName] = useState('analysis')
  const [saveDir, setSaveDir] = useState('')
  const [csv, setCsv] = useState(true)
  const [plot, setPlot] = useState(true)
  const [busy, setBusy] = useState(false)
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null)
  const [exportError, setExportError] = useState<string | null>(null)

  const runExport = async () => {
    setBusy(true)
    setExportError(null)
    try {
      const response = await api.exportAnalysis(job.job_id, {
        analysis_name: analysisName.trim() || 'analysis',
        save_dir: saveDir.trim() || undefined,
        csv,
        json_export: false,
        plot,
        interactive_plot: false,
      })
      setExportResult(response)
    } catch (error) {
      setExportError(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  const plotPaths = exportResult?.out_paths.filter((path) => path.endsWith('.png')) ?? []

  return (
    <div style={{ marginTop: 14 }}>
      <div className="panel-title">
        <span className="eyebrow">Export</span>
      </div>
      <div className="export-grid">
        <div className="field">
          <label htmlFor="analysis-name">Analysis name</label>
          <input
            id="analysis-name"
            value={analysisName}
            onChange={(event) => setAnalysisName(event.target.value)}
            style={{ width: 180 }}
          />
        </div>
        <div className="field">
          <label htmlFor="save-dir">Save folder</label>
          <input
            id="save-dir"
            value={saveDir}
            placeholder="next to the video"
            onChange={(event) => setSaveDir(event.target.value)}
            style={{ width: 180, fontFamily: 'var(--font-data)', fontSize: 12 }}
          />
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <label className="switch-row" style={{ margin: 0 }}>
            <input type="checkbox" checked={csv} onChange={(event) => setCsv(event.target.checked)} />
            CSV per region
          </label>
          <label className="switch-row" style={{ margin: 0 }}>
            <input type="checkbox" checked={plot} onChange={(event) => setPlot(event.target.checked)} />
            Plot images
          </label>
          <button className="btn btn-primary" disabled={busy || (!csv && !plot)} onClick={runExport}>
            {busy ? 'Exporting…' : 'Export'}
          </button>
        </div>
      </div>
      {exportError && <div className="error-text">{exportError}</div>}
      {exportResult && (
        <>
          <div className="ok-text">
            Wrote {exportResult.out_paths.length} file{exportResult.out_paths.length === 1 ? '' : 's'} to{' '}
            <span className="mono">{exportResult.save_dir}</span>
            {exportResult.plot_failed ? ' — plot generation failed, see server log.' : ''}
          </div>
          <ul className="export-paths">
            {exportResult.out_paths.map((path) => (
              <li key={path}>{path}</li>
            ))}
          </ul>
          {plotPaths.map((path) => (
            <img key={path} className="export-plot" src={api.exportedFileUrl(job.job_id, path)} alt={`Plot: ${path}`} />
          ))}
        </>
      )}
    </div>
  )
}
