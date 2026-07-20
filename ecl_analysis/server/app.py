"""FastAPI application exposing the analysis pipeline to the browser UI."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Plots render on worker threads with no GUI event loop.

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ecl_analysis.analysis.brightness import compute_l_star_frame
from ecl_analysis.analysis.models import AnalysisRequest, AnalysisResult, has_analyzable_rois
from ecl_analysis.analysis.runner import run_analysis
from ecl_analysis.analysis.scans import (
    BrightestFrameResult,
    MaskScanRequest,
    PerRoiMaskCaptureResult,
    capture_per_roi_masks,
    find_brightest_frame,
)
from ecl_analysis.export.csv_exporter import ExportOptions, save_analysis_outputs

from .jobs import Job, JobManager
from .videos import VIDEO_EXTENSIONS, VideoOpenError, VideoRegistry

JPEG_DEFAULT_QUALITY = 85
THRESHOLD_TINT_BGR = (255, 0, 200)  # magenta highlight for above-threshold pixels


class VideoOpenRequest(BaseModel):
    path: str = Field(min_length=1)


class RoiModel(BaseModel):
    """Axis-aligned ROI rectangle in frame coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    name: str = ""

    def as_rect(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return ((self.x1, self.y1), (self.x2, self.y2))


class AnalyzeRequest(BaseModel):
    rois: List[RoiModel] = Field(min_length=1)
    background_roi_idx: Optional[int] = None
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    background_percentile: float = Field(default=90.0, ge=0.0, le=100.0)
    morphological_kernel_size: int = Field(default=3, ge=1)
    noise_floor_threshold: float = Field(default=0.0, ge=0.0)
    manual_threshold: float = Field(default=0.0, ge=0.0)
    # Analyze inside masks captured by an earlier per-ROI mask scan job.
    mask_job_id: Optional[str] = None


class MaskScanRequestModel(BaseModel):
    mode: str = Field(pattern="^(global|per_roi)$")
    rois: List[RoiModel] = Field(min_length=1)
    background_roi_idx: Optional[int] = None
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    step: int = Field(default=5, ge=1)
    background_percentile: float = Field(default=90.0, ge=0.0, le=100.0)
    morphological_kernel_size: int = Field(default=3, ge=1)


class DetectRangeRequest(BaseModel):
    expected_duration: float = Field(gt=0.0)


class ExportRequest(BaseModel):
    analysis_name: str = Field(min_length=1)
    save_dir: Optional[str] = None
    csv: bool = True
    json_export: bool = False
    plot: bool = True
    interactive_plot: bool = False


def create_app(web_dist: Optional[str] = None) -> FastAPI:
    videos = VideoRegistry()
    jobs = JobManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        videos.close_all()

    app = FastAPI(title="Brightness Sorcerer API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.videos = videos
    app.state.jobs = jobs

    # ---------- filesystem browsing (for the open-video picker) ----------

    @app.get("/api/fs")
    def list_directory(path: Optional[str] = None) -> dict:
        base = Path(path).expanduser() if path else Path.home()
        try:
            base = base.resolve()
            if not base.is_dir():
                raise HTTPException(status_code=400, detail=f"Not a directory: {base}")
            entries = sorted(base.iterdir(), key=lambda p: p.name.lower())
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        dirs = [e.name for e in entries if e.is_dir() and not e.name.startswith(".")]
        videos_found = [
            e.name
            for e in entries
            if e.is_file() and e.suffix.lower() in VIDEO_EXTENSIONS
        ]
        return {
            "path": str(base),
            "parent": str(base.parent) if base.parent != base else None,
            "dirs": dirs,
            "videos": videos_found,
        }

    # ---------- video sessions ----------

    @app.post("/api/videos")
    def open_video(payload: VideoOpenRequest) -> dict:
        try:
            session = videos.open(payload.path)
        except VideoOpenError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return session.metadata()

    @app.get("/api/videos/{video_id}")
    def video_metadata(video_id: str) -> dict:
        session = videos.get(video_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown video id")
        return session.metadata()

    @app.get("/api/videos/{video_id}/frame/{index}")
    def video_frame(
        video_id: str,
        index: int,
        threshold: Optional[float] = Query(default=None, ge=0.0, le=100.0),
        quality: int = Query(default=JPEG_DEFAULT_QUALITY, ge=10, le=100),
    ) -> Response:
        session = videos.get(video_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown video id")
        frame = session.read_frame(index)
        if frame is None:
            raise HTTPException(status_code=404, detail=f"Frame {index} could not be read")

        if threshold is not None and threshold > 0:
            l_star = compute_l_star_frame(frame)
            mask = l_star > threshold
            if np.any(mask):
                tint = np.empty_like(frame)
                tint[:] = THRESHOLD_TINT_BGR
                blended = cv2.addWeighted(frame, 0.45, tint, 0.55, 0.0)
                frame = frame.copy()
                frame[mask] = blended[mask]

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise HTTPException(status_code=500, detail="Frame could not be encoded")
        return Response(
            content=encoded.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"},
        )

    # ---------- analysis jobs ----------

    @app.post("/api/videos/{video_id}/analyze")
    def start_analysis(video_id: str, payload: AnalyzeRequest) -> dict:
        session = videos.get(video_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown video id")

        if payload.end_frame < payload.start_frame:
            raise HTTPException(status_code=400, detail="end_frame must be >= start_frame")
        if session.frame_count > 0 and payload.end_frame >= session.frame_count:
            raise HTTPException(
                status_code=400,
                detail=f"end_frame {payload.end_frame} exceeds last frame {session.frame_count - 1}",
            )
        if payload.background_roi_idx is not None and not (
            0 <= payload.background_roi_idx < len(payload.rois)
        ):
            raise HTTPException(status_code=400, detail="background_roi_idx out of range")

        rects = [roi.as_rect() for roi in payload.rois]
        if not has_analyzable_rois(rects, payload.background_roi_idx):
            raise HTTPException(
                status_code=400,
                detail="At least one non-background ROI is required",
            )

        fixed_roi_masks: list = []
        use_fixed_mask = False
        if payload.mask_job_id is not None:
            mask_job = jobs.get(payload.mask_job_id)
            if mask_job is None or mask_job.kind != "mask_scan_per_roi":
                raise HTTPException(status_code=400, detail="Unknown mask scan job id")
            if mask_job.status != "done" or not isinstance(mask_job.result, PerRoiMaskCaptureResult):
                raise HTTPException(status_code=409, detail="Mask scan has not completed")
            if mask_job.video_id != session.video_id:
                raise HTTPException(status_code=400, detail="Masks belong to a different video")
            masks = mask_job.result.masks
            if len(masks) != len(rects):
                raise HTTPException(
                    status_code=409,
                    detail="Captured masks no longer match the region list; recapture masks.",
                )
            for idx, (mask, roi) in enumerate(zip(masks, payload.rois)):
                if mask is None:
                    continue
                expected_shape = (abs(roi.y2 - roi.y1), abs(roi.x2 - roi.x1))
                if mask.shape[:2] != expected_shape:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Region {idx + 1} was resized or moved since masks were captured; "
                            "recapture masks."
                        ),
                    )
            fixed_roi_masks = masks
            use_fixed_mask = True

        request = AnalysisRequest(
            video_path=session.path,
            rects=rects,
            background_roi_idx=payload.background_roi_idx,
            start_frame=payload.start_frame,
            end_frame=payload.end_frame,
            use_fixed_mask=use_fixed_mask,
            fixed_roi_masks=fixed_roi_masks,
            background_percentile=payload.background_percentile,
            morphological_kernel_size=payload.morphological_kernel_size,
            noise_floor_threshold=payload.noise_floor_threshold,
            manual_threshold=payload.manual_threshold,
        )
        job = jobs.start(
            "analysis",
            session.video_id,
            lambda progress, message, cancelled: run_analysis(
                request,
                progress_callback=progress,
                message_callback=message,
                cancel_check=cancelled,
            ),
        )
        return {"job_id": job.job_id}

    @app.post("/api/videos/{video_id}/mask-scan")
    def start_mask_scan(video_id: str, payload: MaskScanRequestModel) -> dict:
        session = videos.get(video_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown video id")
        if payload.end_frame < payload.start_frame:
            raise HTTPException(status_code=400, detail="end_frame must be >= start_frame")
        if payload.background_roi_idx is not None and not (
            0 <= payload.background_roi_idx < len(payload.rois)
        ):
            raise HTTPException(status_code=400, detail="background_roi_idx out of range")

        request = MaskScanRequest(
            video_path=session.path,
            rects=[roi.as_rect() for roi in payload.rois],
            background_roi_idx=payload.background_roi_idx,
            start_frame=payload.start_frame,
            end_frame=payload.end_frame,
            step=payload.step,
            background_percentile=payload.background_percentile,
            morphological_kernel_size=payload.morphological_kernel_size,
        )
        scan = find_brightest_frame if payload.mode == "global" else capture_per_roi_masks
        job = jobs.start(
            f"mask_scan_{payload.mode}",
            session.video_id,
            lambda progress, message, cancelled: scan(
                request,
                progress_callback=progress,
                message_callback=message,
                cancel_check=cancelled,
            ),
        )
        return {"job_id": job.job_id}

    @app.post("/api/videos/{video_id}/detect-range")
    def detect_range(video_id: str, payload: DetectRangeRequest) -> dict:
        session = videos.get(video_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown video id")

        # Imported lazily so the server works without the optional audio extras.
        from ecl_analysis.audio import AudioAnalyzer

        analyzer = AudioAnalyzer()
        if not analyzer.is_available():
            raise HTTPException(
                status_code=501,
                detail="Audio analysis is not installed. Run: pip install -e '.[audio]'",
            )

        beeps = analyzer.find_completion_beeps(session.path, payload.expected_duration)
        fps = session.fps if session.fps > 0 else 30.0
        last_frame = max(0, session.frame_count - 1)
        results = []
        for beep_time, end_frame in beeps:
            # The beep marks the run's end; count back the expected duration.
            start_frame = max(0, int((beep_time - payload.expected_duration) * fps))
            clamped_end = min(end_frame, last_frame)
            results.append(
                {
                    "beep_time": beep_time,
                    "start_frame": min(start_frame, clamped_end),
                    "end_frame": clamped_end,
                }
            )
        return {"beeps": results}

    def _serialize_result(job: Job) -> Optional[dict]:
        result = job.result
        if isinstance(result, AnalysisResult):
            return {
                "start_frame": result.start_frame,
                "end_frame": result.end_frame,
                "frames_processed": result.frames_processed,
                "total_frames": result.total_frames,
                "truncated": result.truncated,
                "elapsed_seconds": result.elapsed_seconds,
                "background_values_per_frame": result.background_values_per_frame,
                "rois": [
                    {
                        "roi_index": roi_idx,
                        "brightness_mean": result.brightness_mean_data[data_idx],
                        "brightness_median": result.brightness_median_data[data_idx],
                        "blue_mean": result.blue_mean_data[data_idx],
                        "blue_median": result.blue_median_data[data_idx],
                    }
                    for data_idx, roi_idx in enumerate(result.non_background_rois)
                ],
            }
        if isinstance(result, BrightestFrameResult):
            return {
                "brightest_frame_idx": result.brightest_frame_idx,
                "max_brightness": result.max_brightness,
            }
        if isinstance(result, PerRoiMaskCaptureResult):
            return {
                "sources": result.sources,
                "max_brightness": {str(k): v for k, v in result.max_brightness.items()},
                "mask_coverage": [
                    float(np.count_nonzero(mask)) / mask.size if mask is not None and mask.size else None
                    for mask in result.masks
                ],
            }
        return None

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")

        payload: dict = {
            "job_id": job.job_id,
            "kind": job.kind,
            "video_id": job.video_id,
            "status": job.status,
            "progress": {"done": job.progress_done, "total": job.progress_total},
            "message": job.message,
            "error": job.error,
        }
        serialized = _serialize_result(job)
        if serialized is not None:
            payload["result"] = serialized
        return payload

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        if not jobs.cancel(job_id):
            raise HTTPException(status_code=404, detail="Unknown job id")
        return {"cancelled": True}

    # ---------- export ----------

    @app.post("/api/jobs/{job_id}/export")
    def export_analysis(job_id: str, payload: ExportRequest) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        if job.status != "done" or not isinstance(job.result, AnalysisResult):
            raise HTTPException(status_code=409, detail="Job has no completed analysis result to export")

        session = videos.get(job.video_id)
        video_path = session.path if session is not None else job.request.video_path

        if payload.save_dir:
            save_dir = Path(payload.save_dir).expanduser()
        else:
            save_dir = Path(video_path).parent / f"{Path(video_path).stem}_analysis"
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Cannot create {save_dir}: {exc}") from exc

        # Imported lazily: pulls in matplotlib figure machinery (and the Qt
        # main-window module for a color helper) only when an export happens.
        from ecl_analysis.export.plotting import generate_enhanced_plot

        export_result = save_analysis_outputs(
            analysis_result=job.result,
            save_dir=str(save_dir),
            video_path=video_path,
            analysis_name=payload.analysis_name,
            plot_builder=generate_enhanced_plot,
            export_options=ExportOptions(
                csv=payload.csv,
                json=payload.json_export,
                plot=payload.plot,
                interactive_plot=payload.interactive_plot,
            ),
        )
        job.exported_paths.extend(export_result.out_paths)
        return {
            "save_dir": str(save_dir),
            "out_paths": export_result.out_paths,
            "summary_lines": export_result.summary_lines,
            "avg_brightness_summary": export_result.avg_brightness_summary,
            "plot_failed": export_result.plot_failed,
        }

    @app.get("/api/jobs/{job_id}/files")
    def exported_file(job_id: str, path: str) -> FileResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        resolved = str(Path(path).expanduser().resolve())
        if resolved not in {str(Path(p).resolve()) for p in job.exported_paths}:
            raise HTTPException(status_code=403, detail="Path was not produced by this job")
        if not os.path.isfile(resolved):
            raise HTTPException(status_code=404, detail="File no longer exists")
        return FileResponse(resolved)

    # ---------- built frontend (production mode) ----------

    if web_dist and Path(web_dist).is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app
