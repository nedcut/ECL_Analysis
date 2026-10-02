import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import * as api from '../api'
import { DEFAULT_SETTINGS } from '../types'
import { AnalyzePanel } from './AnalyzePanel'

afterEach(() => vi.restoreAllMocks())

it('shows a failed cancellation request instead of dropping the rejection', async () => {
  vi.spyOn(api, 'cancelJob').mockRejectedValue(new Error('Cancellation request failed'))
  render(<AnalyzePanel video={null} rois={[]} backgroundRoiId={null}
    range={{ start: 0, end: 1 }} settings={DEFAULT_SETTINGS} maskJobId={null}
    onJobChange={vi.fn()} onRangeDetected={vi.fn()} job={{
      job_id: 'running', kind: 'analysis', video_id: 'video', status: 'running',
      progress: { done: 0, total: 2 }, message: '', error: null,
    }} />)
  fireEvent.click(screen.getByRole('button', { name: 'Cancel analysis' }))
  expect(await screen.findByText('Cancellation request failed')).toBeInTheDocument()
})
