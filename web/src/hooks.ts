import { useEffect, useState } from 'react'
import * as api from './api'
import type { JobStatus } from './types'

/** Poll a job until it settles. Returns null until the first status arrives. */
export function useJob(jobId: string | null): JobStatus | null {
  const [job, setJob] = useState<JobStatus | null>(null)

  useEffect(() => {
    if (jobId === null) {
      setJob(null)
      return
    }
    let cancelled = false
    let timer: number | undefined

    const poll = async () => {
      try {
        const status = await api.jobStatus(jobId)
        if (cancelled) return
        setJob(status)
        if (status.status === 'queued' || status.status === 'running') {
          timer = window.setTimeout(poll, 300)
        }
      } catch {
        if (!cancelled) timer = window.setTimeout(poll, 600)
      }
    }
    void poll()
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [jobId])

  return job
}
