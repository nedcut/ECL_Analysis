#!/usr/bin/env python3
"""Scan an inbox for capture videos + sidecars and optionally run analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecl_analysis.ingest.metadata import CaptureMetadataValidation, validate_capture_metadata
from tools import run_mask_review

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4"}
SUMMARY_FILENAME = "capture_ingest_summary.json"


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_videos(inbox_dir: Path, patterns: Sequence[str]) -> Iterable[Path]:
    if patterns:
        seen: set[Path] = set()
        for pattern in patterns:
            for candidate in sorted(inbox_dir.glob(pattern)):
                if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS and candidate not in seen:
                    seen.add(candidate)
                    yield candidate
        return

    for candidate in sorted(inbox_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            yield candidate


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_")
    return safe or "capture"


def _capture_label(video_path: Path, validation: CaptureMetadataValidation) -> str:
    normalized = validation.normalized_metadata or {}
    capture_id = normalized.get("capture_id")
    if isinstance(capture_id, str) and capture_id.strip():
        return _safe_name(capture_id.strip())
    return _safe_name(video_path.stem)


def _source_signature(video_path: Path, sidecar_path: Path) -> Dict[str, Any]:
    return {
        "video_path": str(video_path),
        "video_size": video_path.stat().st_size,
        "video_mtime_ns": video_path.stat().st_mtime_ns,
        "sidecar_path": str(sidecar_path),
        "sidecar_exists": sidecar_path.exists(),
        "sidecar_size": sidecar_path.stat().st_size if sidecar_path.exists() else None,
        "sidecar_mtime_ns": sidecar_path.stat().st_mtime_ns if sidecar_path.exists() else None,
    }


def _is_already_processed(capture_dir: Path, signature: Dict[str, Any]) -> bool:
    summary_path = capture_dir / SUMMARY_FILENAME
    if not summary_path.exists():
        return False

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    previous_signature = summary.get("source_signature")
    return isinstance(previous_signature, dict) and previous_signature == signature


def _resolve_end_frame(video_path: Path, configured_end_frame: Optional[int]) -> int:
    if configured_end_frame is not None:
        return int(configured_end_frame)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file while resolving frame count: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if frame_count <= 0:
        raise RuntimeError(f"Could not resolve a positive frame count for {video_path}")
    return frame_count - 1


def _run_analysis_case(
    template_case: Dict[str, Any],
    video_path: Path,
    capture_label: str,
    root_output_dir: Path,
) -> Dict[str, Any]:
    case = dict(template_case)
    case.setdefault("name", capture_label)
    case["name"] = f"{case['name']}_{capture_label}" if case["name"] != capture_label else capture_label
    case["video_path"] = str(video_path)
    case.setdefault("start_frame", 0)
    case["end_frame"] = _resolve_end_frame(video_path, case.get("end_frame"))
    case.setdefault("analysis_name", capture_label)
    return run_mask_review._run_case(case, root_output_dir)


def _write_summary(
    capture_dir: Path,
    *,
    capture_label: str,
    video_path: Path,
    validation: CaptureMetadataValidation,
    signature: Dict[str, Any],
    analysis_result: Optional[Dict[str, Any]],
    archived_paths: Optional[Dict[str, str]] = None,
) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    summary_path = capture_dir / SUMMARY_FILENAME
    payload = {
        "capture_label": capture_label,
        "video_path": str(video_path),
        "created_at_epoch": time.time(),
        "source_signature": signature,
        "validation": validation.to_dict(),
        "analysis_result": analysis_result,
        "archived_paths": archived_paths or {},
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return summary_path


def _archive_sources(video_path: Path, sidecar_path: Path, archive_root: Path, capture_label: str) -> Dict[str, str]:
    archive_dir = archive_root / capture_label
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived_video = archive_dir / video_path.name
    shutil.move(str(video_path), archived_video)

    archived_sidecar = None
    if sidecar_path.exists():
        archived_sidecar = archive_dir / sidecar_path.name
        shutil.move(str(sidecar_path), archived_sidecar)

    return {
        "video_path": str(archived_video),
        "sidecar_path": str(archived_sidecar) if archived_sidecar is not None else "",
    }


def process_inbox_once(manifest: Dict[str, Any]) -> Dict[str, Any]:
    inbox_dir = Path(manifest.get("inbox_dir", "incoming")).expanduser().resolve()
    output_dir = Path(manifest.get("output_dir", "incoming_outputs")).expanduser().resolve()
    archive_dir_value = manifest.get("archive_dir")
    archive_dir = Path(archive_dir_value).expanduser().resolve() if archive_dir_value else None
    patterns = list(manifest.get("include_patterns", []))
    analysis_case = manifest.get("analysis_case")
    force_reprocess = bool(manifest.get("force_reprocess", False))

    inbox_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)

    processed: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for video_path in _iter_videos(inbox_dir, patterns):
        validation = validate_capture_metadata(str(video_path))
        sidecar_path = Path(validation.sidecar_path)
        capture_label = _capture_label(video_path, validation)
        capture_dir = output_dir / capture_label
        summary_path = capture_dir / SUMMARY_FILENAME
        signature = _source_signature(video_path, sidecar_path)

        if not force_reprocess and _is_already_processed(capture_dir, signature):
            skipped.append(
                {
                    "capture_label": capture_label,
                    "video_path": str(video_path),
                    "reason": "already_processed",
                    "existing_summary_path": str(summary_path),
                }
            )
            continue

        analysis_result = None
        if isinstance(analysis_case, dict):
            analysis_result = _run_analysis_case(
                template_case=analysis_case,
                video_path=video_path,
                capture_label=capture_label,
                root_output_dir=output_dir,
            )

        archived_paths = None
        if archive_dir is not None:
            archived_paths = _archive_sources(video_path, sidecar_path, archive_dir, capture_label)

        summary_path = _write_summary(
            capture_dir=capture_dir,
            capture_label=capture_label,
            video_path=video_path,
            validation=validation,
            signature=signature,
            analysis_result=analysis_result,
            archived_paths=archived_paths,
        )
        processed.append(
            {
                "capture_label": capture_label,
                "video_path": str(video_path),
                "summary_path": str(summary_path),
                "validation_status": validation.status,
                "analysis_ran": analysis_result is not None,
            }
        )

    result = {
        "inbox_dir": str(inbox_dir),
        "output_dir": str(output_dir),
        "archive_dir": str(archive_dir) if archive_dir is not None else None,
        "processed_count": len(processed),
        "skipped_count": len(skipped),
        "processed": processed,
        "skipped": skipped,
    }
    (output_dir / "inbox_run_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to a capture inbox manifest JSON file.")
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess captures even when an identical source signature already exists in the output directory.",
    )
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=None,
        help="If provided, rescan the inbox on this interval until interrupted.",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    if args.force_reprocess:
        manifest["force_reprocess"] = True

    if args.watch_seconds is None:
        result = process_inbox_once(manifest)
        print(json.dumps(result, indent=2))
        return 0

    while True:
        result = process_inbox_once(manifest)
        print(json.dumps(result, indent=2))
        time.sleep(max(0.5, args.watch_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
