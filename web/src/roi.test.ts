import { describe, expect, it } from 'vitest'
import { dragRoi, hitTestRoi, nextRoiName, normalizeRoi, roiColor } from './roi'

describe('normalizeRoi', () => {
  it('orders coordinates and clamps to the frame', () => {
    expect(normalizeRoi({ x1: 50, y1: 40, x2: 10, y2: -5 }, 100, 100)).toEqual({
      x1: 10,
      y1: 0,
      x2: 50,
      y2: 40,
    })
  })

  it('clamps beyond-frame coordinates', () => {
    expect(normalizeRoi({ x1: -10, y1: 0, x2: 500, y2: 60 }, 100, 50)).toEqual({
      x1: 0,
      y1: 0,
      x2: 100,
      y2: 50,
    })
  })
})

describe('hitTestRoi', () => {
  const roi = { x1: 20, y1: 20, x2: 80, y2: 60 }

  it('prefers a corner handle over the interior', () => {
    expect(hitTestRoi(roi, 21, 21, 1)).toBe('nw')
    expect(hitTestRoi(roi, 79, 59, 1)).toBe('se')
  })

  it('returns move inside and null outside', () => {
    expect(hitTestRoi(roi, 50, 40, 1)).toBe('move')
    expect(hitTestRoi(roi, 5, 5, 1)).toBeNull()
  })

  it('scales the grab radius with zoom', () => {
    // At scale 4 (zoomed out), a point 20 frame-px from the corner still grabs it.
    expect(hitTestRoi(roi, 50, 40, 4)).not.toBeNull()
    expect(hitTestRoi(roi, 44, 44, 4)).toBe('nw')
  })
})

describe('dragRoi', () => {
  const roi = { x1: 20, y1: 20, x2: 80, y2: 60 }

  it('moves without resizing and clamps at frame edges', () => {
    const moved = dragRoi(roi, 'move', -100, 5, 100, 100)
    expect(moved).toEqual({ x1: 0, y1: 25, x2: 60, y2: 65 })
  })

  it('resizes from a corner', () => {
    const resized = dragRoi(roi, 'se', 10, -10, 100, 100)
    expect(resized).toEqual({ x1: 20, y1: 20, x2: 90, y2: 50 })
  })

  it('normalizes when a corner crosses its opposite edge', () => {
    const crossed = dragRoi(roi, 'se', -70, 0, 100, 100)
    expect(crossed.x1).toBeLessThanOrEqual(crossed.x2)
  })
})

describe('roiColor', () => {
  it('is stable per id so colors follow the entity', () => {
    expect(roiColor(3)).toBe(roiColor(3))
    expect(roiColor(1)).not.toBe(roiColor(2))
  })

  it('assigns palette slot 1 to the first ROI drawn (ids are 1-based)', () => {
    expect(roiColor(1)).toBe('#3987e5')
  })
})

describe('nextRoiName', () => {
  it('fills the first free slot', () => {
    const rois = [
      { id: 1, name: 'ROI 1', x1: 0, y1: 0, x2: 1, y2: 1 },
      { id: 3, name: 'ROI 3', x1: 0, y1: 0, x2: 1, y2: 1 },
    ]
    expect(nextRoiName(rois)).toBe('ROI 2')
  })
})
