"""Threaded background jobs with cooperative cancellation and polled progress.

A job wraps any UI-free runner function (analysis, mask scans) that accepts
progress/message callbacks and a cancel check — the same contract the Qt
workers use, so both frontends drive identical code.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ecl_analysis.analysis.runner import AnalysisCancelled

JobRunner = Callable[
    [Callable[[int, int], None], Callable[[str], None], Callable[[], bool]],
    Any,
]


@dataclass
class Job:
    """State of one background run, mutated only under the manager lock."""

    job_id: str
    kind: str  # analysis | mask_scan_global | mask_scan_per_roi
    video_id: str
    status: str = "queued"  # queued | running | done | error | cancelled
    progress_done: int = 0
    progress_total: int = 0
    message: str = ""
    error: Optional[str] = None
    result: Optional[Any] = None
    exported_paths: List[str] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobManager:
    """Run job functions on daemon threads and expose polled status."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, video_id: str, runner: JobRunner) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, video_id=video_id)
        with self._lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(target=self._run, args=(job, runner), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.cancel_event.set()
        return True

    def _run(self, job: Job, runner: JobRunner) -> None:
        with self._lock:
            job.status = "running"

        def on_progress(done: int, total: int) -> None:
            with self._lock:
                job.progress_done = done
                job.progress_total = total

        def on_message(message: str) -> None:
            with self._lock:
                job.message = message

        try:
            result = runner(on_progress, on_message, job.cancel_event.is_set)
        except AnalysisCancelled:
            with self._lock:
                job.status = "cancelled"
        except Exception as exc:  # noqa: BLE001 — surfaced to the client as job.error
            with self._lock:
                job.status = "error"
                job.error = str(exc)
        else:
            with self._lock:
                job.status = "done"
                job.result = result
