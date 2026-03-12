#!/usr/bin/env python3
"""Run repeatable mask-review cases from a local manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecl_analysis.analysis.background import compute_background_brightness
from ecl_analysis.analysis.brightness import compute_l_star_frame
from ecl_analysis.analysis.masking import build_signal_mask
from ecl_analysis.analysis.models import AnalysisRequest, AnalysisResult, MaskCaptureMetadata
from ecl_analysis.export.csv_exporter import save_analysis_outputs
from ecl_analysis.workers import (
    AnalysisWorker,
    BrightestFrameResult,
    BrightestFrameWorker,
    MaskScanRequest,
    PerRoiMaskCaptureResult,
    PerRoiMaskCaptureWorker,
)


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_worker(worker: Any) -> Any:
    captured: Dict[str, Any] = {}
    worker.finished.connect(lambda payload: captured.setdefault("result", payload))
    worker.error.connect(lambda message: captured.setdefault("error", message))
    worker.cancelled.connect(lambda: captured.setdefault("cancelled", True))
    worker.run()
    if captured.get("cancelled"):
        raise RuntimeError("Worker cancelled unexpectedly.")
    if "error" in captured:
        raise RuntimeError(str(captured["error"]))
    if "result" not in captured:
        raise RuntimeError("Worker did not produce a result.")
    return captured["result"]


def _noop_plot_builder(*_args: Any, **_kwargs: Any) -> Tuple[None, None]:
    return None, None


def _case_output_dir(root: Path, case_name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in case_name).strip("_")
    return root / (safe_name or "review_case")


def _overlay_masks(
    frame: np.ndarray,
    rects: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    background_roi_idx: Optional[int],
    fixed_masks: Sequence[Optional[np.ndarray]],
    noise_floor_threshold: float,
    morphological_kernel_size: int,
) -> np.ndarray:
    overlay = frame.copy()
    l_star_frame = compute_l_star_frame(frame)
    background_value = compute_background_brightness(
        frame=frame,
        rects=rects,
        background_roi_idx=background_roi_idx,
        background_percentile=90.0,
        frame_l_star=l_star_frame,
    )
    fh, fw = frame.shape[:2]
    for roi_idx, (pt1, pt2) in enumerate(rects):
        if roi_idx == background_roi_idx:
            continue
        x1 = max(0, min(int(min(pt1[0], pt2[0])), fw))
        x2 = max(0, min(int(max(pt1[0], pt2[0])), fw))
        y1 = max(0, min(int(min(pt1[1], pt2[1])), fh))
        y2 = max(0, min(int(max(pt1[1], pt2[1])), fh))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = overlay[y1:y2, x1:x2]
        roi_l_star = l_star_frame[y1:y2, x1:x2]
        adaptive_mask, _threshold, _min_area = build_signal_mask(
            roi_l_star=roi_l_star,
            background_brightness=background_value,
            noise_floor_threshold=noise_floor_threshold,
            morphological_kernel_size=morphological_kernel_size,
        )
        fixed_mask = None
        if roi_idx < len(fixed_masks):
            candidate = fixed_masks[roi_idx]
            if isinstance(candidate, np.ndarray) and candidate.shape[:2] == roi.shape[:2]:
                fixed_mask = candidate.astype(bool)

        if fixed_mask is None:
            roi[adaptive_mask] = roi[adaptive_mask] * 0.7 + np.array([0, 0, 255]) * 0.3
            continue

        overlap_mask = fixed_mask & adaptive_mask
        fixed_only_mask = fixed_mask & ~adaptive_mask
        adaptive_only_mask = adaptive_mask & ~fixed_mask
        roi[fixed_only_mask] = roi[fixed_only_mask] * 0.55 + np.array([0, 0, 255]) * 0.45
        roi[adaptive_only_mask] = roi[adaptive_only_mask] * 0.55 + np.array([255, 0, 0]) * 0.45
        roi[overlap_mask] = roi[overlap_mask] * 0.45 + np.array([255, 0, 255]) * 0.55

    return overlay


def _save_overlay_images(
    video_path: str,
    case_dir: Path,
    rects: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    background_roi_idx: Optional[int],
    fixed_masks: Sequence[Optional[np.ndarray]],
    sources: Sequence[Optional[int]],
    noise_floor_threshold: float,
    morphological_kernel_size: int,
) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file for overlays: {video_path}")

    overlay_paths: List[str] = []
    try:
        for frame_idx in sorted({value for value in sources if value is not None}):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            overlay = _overlay_masks(
                frame=frame,
                rects=rects,
                background_roi_idx=background_roi_idx,
                fixed_masks=fixed_masks,
                noise_floor_threshold=noise_floor_threshold,
                morphological_kernel_size=morphological_kernel_size,
            )
            filename = f"mask_overlay_frame{int(frame_idx) + 1:05d}.png"
            out_path = case_dir / filename
            cv2.imwrite(str(out_path), overlay)
            overlay_paths.append(str(out_path))
    finally:
        cap.release()
    return overlay_paths


def _mask_review_summary(
    metadata: Sequence[Optional[MaskCaptureMetadata]],
    expected_nonempty_rois: Sequence[int],
    expected_empty_rois: Sequence[int],
) -> Tuple[bool, List[str]]:
    lines: List[str] = []
    passed = True

    for roi_idx in expected_nonempty_rois:
        metadata_entry = metadata[roi_idx] if roi_idx < len(metadata) else None
        pixel_count = metadata_entry.pixel_count if isinstance(metadata_entry, MaskCaptureMetadata) else 0
        roi_pass = pixel_count > 0
        passed &= roi_pass
        lines.append(
            f"- ROI {roi_idx + 1} expected non-empty: {'PASS' if roi_pass else 'FAIL'} "
            f"(pixels={pixel_count})"
        )

    for roi_idx in expected_empty_rois:
        metadata_entry = metadata[roi_idx] if roi_idx < len(metadata) else None
        pixel_count = metadata_entry.pixel_count if isinstance(metadata_entry, MaskCaptureMetadata) else 0
        roi_pass = pixel_count == 0
        passed &= roi_pass
        lines.append(
            f"- ROI {roi_idx + 1} expected empty: {'PASS' if roi_pass else 'FAIL'} "
            f"(pixels={pixel_count})"
        )

    for roi_idx, metadata_entry in enumerate(metadata):
        if not isinstance(metadata_entry, MaskCaptureMetadata):
            continue
        warning_text = ", ".join(metadata_entry.warnings) if metadata_entry.warnings else "none"
        lines.append(
            f"- ROI {roi_idx + 1} quality: {metadata_entry.confidence_label}, "
            f"consensus={metadata_entry.consensus_ratio:.2f}, warnings={warning_text}"
        )

    return passed, lines


def _run_case(case: Dict[str, Any], root_output_dir: Path) -> Dict[str, Any]:
    case_name = str(case["name"])
    case_dir = _case_output_dir(root_output_dir, case_name)
    case_dir.mkdir(parents=True, exist_ok=True)

    rects = [
        ((int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])))
        for pt1, pt2 in case["rects"]
    ]
    start_frame = int(case.get("start_frame", 0))
    end_frame = int(case["end_frame"])
    background_roi_idx = case.get("background_roi_idx")
    capture_mode = str(case.get("capture_mode", "per_roi_auto"))
    background_percentile = float(case.get("background_percentile", 90.0))
    morphological_kernel_size = int(case.get("morphological_kernel_size", 3))
    noise_floor_threshold = float(case.get("noise_floor_threshold", 5.0))

    scan_request = MaskScanRequest(
        video_path=str(case["video_path"]),
        rects=rects,
        background_roi_idx=background_roi_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        step=1,
        background_percentile=background_percentile,
        morphological_kernel_size=morphological_kernel_size,
        noise_floor_threshold=noise_floor_threshold,
    )

    if capture_mode == "global_auto":
        scan_result = _run_worker(BrightestFrameWorker(scan_request))
        if not isinstance(scan_result, BrightestFrameResult):
            raise RuntimeError("Unexpected global auto-capture result type.")
        fixed_masks = scan_result.masks
        mask_sources = scan_result.sources
        mask_metadata = scan_result.metadata
    else:
        scan_result = _run_worker(PerRoiMaskCaptureWorker(scan_request))
        if not isinstance(scan_result, PerRoiMaskCaptureResult):
            raise RuntimeError("Unexpected per-ROI auto-capture result type.")
        fixed_masks = scan_result.masks
        mask_sources = scan_result.sources
        mask_metadata = scan_result.metadata

    analysis_request = AnalysisRequest(
        video_path=str(case["video_path"]),
        rects=rects,
        background_roi_idx=background_roi_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        use_fixed_mask=True,
        fixed_roi_masks=fixed_masks,
        background_percentile=background_percentile,
        morphological_kernel_size=morphological_kernel_size,
        noise_floor_threshold=noise_floor_threshold,
        mask_metadata=[metadata.clone() if metadata is not None else None for metadata in mask_metadata],
        analysis_metadata={
            "review_case": case_name,
            "capture_mode": capture_mode,
            "rects": case["rects"],
            "background_roi_idx": background_roi_idx,
            "background_percentile": background_percentile,
            "morphological_kernel_size": morphological_kernel_size,
            "noise_floor_threshold": noise_floor_threshold,
        },
    )
    analysis_result = _run_worker(AnalysisWorker(analysis_request))
    if not isinstance(analysis_result, AnalysisResult):
        raise RuntimeError("Unexpected analysis result type.")

    export_result = save_analysis_outputs(
        analysis_result=analysis_result,
        save_dir=str(case_dir),
        video_path=str(case["video_path"]),
        analysis_name=str(case.get("analysis_name", case_name)),
        plot_builder=_noop_plot_builder,
    )
    overlay_paths = _save_overlay_images(
        video_path=str(case["video_path"]),
        case_dir=case_dir,
        rects=rects,
        background_roi_idx=background_roi_idx,
        fixed_masks=fixed_masks,
        sources=mask_sources,
        noise_floor_threshold=noise_floor_threshold,
        morphological_kernel_size=morphological_kernel_size,
    )
    passed, summary_lines = _mask_review_summary(
        metadata=mask_metadata,
        expected_nonempty_rois=[int(value) for value in case.get("expected_nonempty_rois", [])],
        expected_empty_rois=[int(value) for value in case.get("expected_empty_rois", [])],
    )

    review_report_path = case_dir / "review_summary.md"
    review_report_path.write_text(
        "\n".join(
            [
                f"# {case_name}",
                "",
                f"- Capture mode: `{capture_mode}`",
                f"- Video: `{case['video_path']}`",
                f"- Result: {'PASS' if passed else 'FAIL'}",
                "",
                "## Checks",
                *summary_lines,
                "",
                "## Exported artifacts",
                *[f"- `{Path(path).name}`" for path in export_result.out_paths + overlay_paths],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "name": case_name,
        "passed": passed,
        "case_dir": str(case_dir),
        "overlay_paths": overlay_paths,
        "export_paths": export_result.out_paths,
        "review_report": str(review_report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to a local JSON review manifest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated review artifacts. Defaults to manifest `output_dir` or `mask_review_outputs`.",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    output_dir = args.output_dir or Path(manifest.get("output_dir", "mask_review_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    case_results = [_run_case(case, output_dir) for case in manifest.get("cases", [])]
    passed_count = sum(1 for result in case_results if result["passed"])
    summary_path = output_dir / "manifest_review_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "passed_cases": passed_count,
                "total_cases": len(case_results),
                "results": case_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved review summary to {summary_path}")
    return 0 if passed_count == len(case_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
