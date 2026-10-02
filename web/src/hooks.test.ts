import { act, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import * as api from './api'
import { useJob } from './hooks'
import type { JobStatus } from './types'

afterEach(() => vi.restoreAllMocks())

const done = (id: string): JobStatus => ({
  job_id: id, kind: 'mask_scan_global', video_id: 'video', status: 'done',
  progress: { done: 1, total: 1 }, message: '', error: null,
})

it('hides a previous completed scan while the next scan is pending', async () => {
  let finish!: (job: JobStatus) => void
  vi.spyOn(api, 'jobStatus').mockResolvedValueOnce(done('old'))
    .mockImplementationOnce(() => new Promise((resolve) => { finish = resolve }))
  const hook = renderHook(({ id }) => useJob(id), { initialProps: { id: 'old' } })
  await act(async () => {})
  expect(hook.result.current?.job_id).toBe('old')
  hook.rerender({ id: 'new' })
  expect(hook.result.current).toBeNull()
  await act(async () => finish(done('new')))
  expect(hook.result.current?.job_id).toBe('new')
})

it('discards a late response for a superseded scan', async () => {
  let finishOld!: (job: JobStatus) => void
  vi.spyOn(api, 'jobStatus')
    .mockImplementationOnce(() => new Promise((resolve) => { finishOld = resolve }))
    .mockResolvedValueOnce(done('new'))
  const hook = renderHook(({ id }) => useJob(id), { initialProps: { id: 'old' } })
  hook.rerender({ id: 'new' })
  await act(async () => {})
  await act(async () => finishOld(done('old')))
  expect(hook.result.current?.job_id).toBe('new')
})
