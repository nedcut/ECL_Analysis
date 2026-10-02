import { render, screen } from '@testing-library/react'
import { expect, it } from 'vitest'
import type { AnalysisResultPayload, JobStatus } from '../types'
import { ResultsDrawer } from './ResultsDrawer'

it('keeps captured region labels and displays decoder warnings after edits', () => {
  const result: AnalysisResultPayload = {
    start_frame: 0, end_frame: 1, frames_processed: 2, total_frames: 2,
    truncated: false, elapsed_seconds: 0.1, fps: 30,
    threshold_mode: 'whole_roi', seek_warning: 'Decoder position differs; frame numbers may be offset.',
    background_values_per_frame: [0, 0],
    rois: [{
      roi_index: 0, name: 'original electrode', brightness_mean: [1, 2],
      brightness_median: [1, 2], blue_mean: [1, 2], blue_median: [1, 2],
      pixel_count: [4, 4], fixed_mask_status: 'not_requested',
    }],
  }
  result.rois.push({ ...result.rois[0], roi_index: 1, name: 'reference electrode' })
  const job: JobStatus = {
    job_id: 'completed', kind: 'analysis', video_id: 'video', status: 'done',
    progress: { done: 2, total: 2 }, message: '', error: null, result,
  }
  render(<ResultsDrawer job={job} backgroundRoiId={null} rois={[
    { id: 1, name: 'different region', x1: 5, y1: 5, x2: 7, y2: 7 },
  ]} />)
  expect(screen.getByText('original electrode')).toBeInTheDocument()
  expect(screen.queryByText('different region')).not.toBeInTheDocument()
  expect(screen.getByText(result.seek_warning!)).toBeInTheDocument()
})
