import { BACKGROUND_ROI_COLOR, roiColor } from '../roi'
import type { Roi } from '../types'

interface RoiPanelProps {
  rois: Roi[]
  selectedRoiId: number | null
  backgroundRoiId: number | null
  drawMode: boolean
  disabled: boolean
  onArmDraw: () => void
  onSelect: (id: number) => void
  onRename: (id: number, name: string) => void
  onSetBackground: (id: number) => void
  onDelete: (id: number) => void
}

export function RoiPanel({
  rois,
  selectedRoiId,
  backgroundRoiId,
  drawMode,
  disabled,
  onArmDraw,
  onSelect,
  onRename,
  onSetBackground,
  onDelete,
}: RoiPanelProps) {
  return (
    <section className="panel">
      <div className="panel-title">
        <span className="eyebrow">Regions</span>
        <button className="btn" onClick={onArmDraw} disabled={disabled || drawMode}>
          {drawMode ? 'Drag on the frame…' : '+ Add region'}
        </button>
      </div>
      {rois.length === 0 && (
        <div style={{ color: 'var(--faint)', fontSize: 13 }}>
          Draw regions over each electrode. Mark one as background to enable
          background subtraction.
        </div>
      )}
      {rois.map((roi) => {
        const isBackground = roi.id === backgroundRoiId
        return (
          <div
            key={roi.id}
            className={`roi-row${roi.id === selectedRoiId ? ' selected' : ''}`}
            onClick={() => onSelect(roi.id)}
          >
            <span
              className={`roi-chip${isBackground ? ' background' : ''}`}
              style={{ background: isBackground ? BACKGROUND_ROI_COLOR : roiColor(roi.id) }}
            />
            <input
              className="roi-name-input"
              value={roi.name}
              aria-label="Region name"
              onChange={(event) => onRename(roi.id, event.target.value)}
              onClick={(event) => event.stopPropagation()}
            />
            {isBackground && <span className="roi-tag">BG</span>}
            <span className="roi-coords mono">
              {roi.x2 - roi.x1}×{roi.y2 - roi.y1}
            </span>
            <button
              className="btn-ghost btn"
              title={isBackground ? 'Unset background region' : 'Use as background region'}
              onClick={(event) => {
                event.stopPropagation()
                onSetBackground(roi.id)
              }}
            >
              {isBackground ? '◎' : '○'}
            </button>
            <button
              className="btn-ghost btn"
              title="Delete region"
              onClick={(event) => {
                event.stopPropagation()
                onDelete(roi.id)
              }}
            >
              ✕
            </button>
          </div>
        )
      })}
    </section>
  )
}
