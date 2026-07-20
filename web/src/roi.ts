import type { Roi } from './types'

/** Series colors: validated categorical order (dark mode), assigned by ROI id,
 * never re-cycled when ROIs are deleted — color follows the entity. */
export const ROI_COLORS = [
  '#3987e5',
  '#008300',
  '#d55181',
  '#c98500',
  '#199e70',
  '#d95926',
  '#9085e9',
  '#e66767',
] as const

export const BACKGROUND_ROI_COLOR = '#ffb454'

export function roiColor(roiId: number): string {
  // Ids are 1-based; slot 1 of the palette goes to the first ROI ever drawn.
  const index = Math.max(0, roiId - 1) % ROI_COLORS.length
  return ROI_COLORS[index]
}

export interface NormalizedRoi {
  x1: number
  y1: number
  x2: number
  y2: number
}

/** Return the ROI with x1<x2, y1<y2, clamped to the frame. */
export function normalizeRoi(roi: NormalizedRoi, frameW: number, frameH: number): NormalizedRoi {
  const [x1, x2] = [roi.x1, roi.x2].sort((a, b) => a - b)
  const [y1, y2] = [roi.y1, roi.y2].sort((a, b) => a - b)
  return {
    x1: Math.max(0, Math.min(Math.round(x1), frameW)),
    y1: Math.max(0, Math.min(Math.round(y1), frameH)),
    x2: Math.max(0, Math.min(Math.round(x2), frameW)),
    y2: Math.max(0, Math.min(Math.round(y2), frameH)),
  }
}

export function roiArea(roi: NormalizedRoi): number {
  return Math.abs(roi.x2 - roi.x1) * Math.abs(roi.y2 - roi.y1)
}

export type Handle = 'nw' | 'ne' | 'sw' | 'se' | 'move' | null

const HANDLE_HIT_RADIUS = 8

/** Hit-test a frame-space point against a normalized ROI. Corners win over the
 * interior so small ROIs stay resizable. `scale` converts the on-screen grab
 * radius into frame pixels (frame px per CSS px). */
export function hitTestRoi(roi: NormalizedRoi, x: number, y: number, scale: number): Handle {
  const radius = HANDLE_HIT_RADIUS * scale
  const corners: Array<[Handle, number, number]> = [
    ['nw', roi.x1, roi.y1],
    ['ne', roi.x2, roi.y1],
    ['sw', roi.x1, roi.y2],
    ['se', roi.x2, roi.y2],
  ]
  for (const [handle, cx, cy] of corners) {
    if (Math.abs(x - cx) <= radius && Math.abs(y - cy) <= radius) return handle
  }
  if (x >= roi.x1 && x <= roi.x2 && y >= roi.y1 && y <= roi.y2) return 'move'
  return null
}

/** Apply a drag to a ROI given the active handle. */
export function dragRoi(
  roi: NormalizedRoi,
  handle: Handle,
  dx: number,
  dy: number,
  frameW: number,
  frameH: number,
): NormalizedRoi {
  if (handle === null) return roi
  if (handle === 'move') {
    const width = roi.x2 - roi.x1
    const height = roi.y2 - roi.y1
    const x1 = Math.max(0, Math.min(roi.x1 + dx, frameW - width))
    const y1 = Math.max(0, Math.min(roi.y1 + dy, frameH - height))
    return { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x1 + width), y2: Math.round(y1 + height) }
  }
  const next = { ...roi }
  if (handle === 'nw' || handle === 'sw') next.x1 += dx
  if (handle === 'ne' || handle === 'se') next.x2 += dx
  if (handle === 'nw' || handle === 'ne') next.y1 += dy
  if (handle === 'sw' || handle === 'se') next.y2 += dy
  return normalizeRoi(next, frameW, frameH)
}

export function nextRoiName(rois: Roi[]): string {
  const used = new Set(rois.map((roi) => roi.name))
  for (let n = 1; ; n += 1) {
    const candidate = `ROI ${n}`
    if (!used.has(candidate)) return candidate
  }
}
