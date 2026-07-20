"""Threaded analysis jobs with cooperative cancellation and polled progress."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ecl_analysis.analysis.models import AnalysisRequest, AnalysisResult
from ecl_analysis.analysis.runner import AnalysisCancelled, run_analysis


@dataclass
class AnalysisJob:
    """State of one analysis run, mutated only under the manager lock."""

    job_id: str
    video_id: str
    request: AnalysisRequest
    status: str = "queued"  # queued | running | done | error | cancelled
    progress_done: int = 0
    progress_total: int = 0
    message: str = ""
    error: Optional[str] = None
    result: Optional[AnalysisResult] = None
    exported_paths: List[str] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobManager:
    """Run analysis requests on daemon threads and expose polled status."""

    def __init__(self) -> None:
        self._jobs: Dict[str, AnalysisJob] = {}
        self._lock = threading.Lock()

    def start(self, video_id: str, request: AnalysisRequest) -> AnalysisJob:
        job = AnalysisJob(job_id=uuid.uuid4().hex[:12], video_id=video_id, request=request)
        with self._lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[AnalysisJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.cancel_event.set()
        return True

    def _run(self, job: AnalysisJob) -> None:
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
            result = run_analysis(
                job.request,
                progress_callback=on_progress,
                message_callback=on_message,
                cancel_check=job.cancel_event.is_set,
            )
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
                job.progress_done = result.frames_processed
                job.progress_total = result.total_frames
