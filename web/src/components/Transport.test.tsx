import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Transport } from './Transport'

describe('frame numbering', () => {
  it('shows 1-based frames while range changes keep decoder indices', () => {
    const onRangeChange = vi.fn()
    render(<Transport frame={4} frameCount={30} fps={30} playing={false} speed={1}
      range={{ start: 0, end: 29 }} onSeek={vi.fn()} onTogglePlay={vi.fn()}
      onSpeedChange={vi.fn()} onRangeChange={onRangeChange} />)
    const slider = screen.getByRole('slider', { name: 'Frame' })
    expect(slider).toHaveAttribute('aria-valuemin', '1')
    expect(slider).toHaveAttribute('aria-valuemax', '30')
    expect(slider).toHaveAttribute('aria-valuenow', '5')
    expect(screen.getByText('01–30')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Set start' }))
    expect(onRangeChange).toHaveBeenCalledWith({ start: 4, end: 29 })
    fireEvent.click(screen.getByRole('button', { name: 'Set end' }))
    expect(onRangeChange).toHaveBeenCalledWith({ start: 0, end: 4 })
  })
})
