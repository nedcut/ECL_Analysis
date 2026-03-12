#!/usr/bin/env python3
"""Assemble a repeatable review bundle for exported real-video analyses."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIDENCE_RANK = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


@dataclass
class RunEvaluation:
    """Materialized review result for one manifest entry."""

    label: str
    status: str
    video_path: str
    analysis_dir: str
    metadata_path: Optional[str]
    copied_artifacts: List[str]
    notes: List[str]
    roi_summaries: List[Dict[str, Any]]


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Manifest must contain a non-empty 'runs' array.")
    return manifest


def _find_metadata_file(run_entry: Dict[str, Any]) -> Path:
    explicit_path = run_entry.get("analysis_metadata_path")
    if explicit_path:
        metadata_path = Path(explicit_path).expanduser()
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        return metadata_path

    analysis_dir = Path(run_entry["analysis_dir"]).expanduser()
    matches = sorted(analysis_dir.glob("*_analysis_metadata.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_analysis_metadata.json in {analysis_dir}, found {len(matches)}."
        )
    return matches[0]


def _copy_artifacts(analysis_dir: Path, output_dir: Path) -> List[str]:
    copied: List[str] = []
    patterns = [
        "*_analysis_metadata.json",
        "*_brightness.csv",
        "*.png",
        "*.html",
    ]
    for pattern in patterns:
        for src in sorted(analysis_dir.glob(pattern)):
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst.name)
    return copied


def _summarize_mask_metadata(mask_metadata: List[Any]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for idx, metadata in enumerate(mask_metadata):
        if not isinstance(metadata, dict):
            continue
        summaries.append(
            {
                "roi_index": idx,
                "confidence_label": metadata.get("confidence_label", "none"),
                "pixel_count": int(metadata.get("pixel_count", 0)),
                "warnings": list(metadata.get("warnings", [])),
                "source_frames": list(metadata.get("source_frames", [])),
            }
        )
    return summaries


def _evaluate_run(run_entry: Dict[str, Any], bundle_dir: Path) -> RunEvaluation:
    label = str(run_entry.get("label") or Path(run_entry.get("video_path", "run")).stem)
    video_path = str(Path(run_entry["video_path"]).expanduser())
    analysis_dir = Path(run_entry["analysis_dir"]).expanduser()
    metadata_path = _find_metadata_file(run_entry)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    roi_summaries = _summarize_mask_metadata(metadata.get("mask_metadata", []))

    require_fixed_mask = bool(run_entry.get("require_fixed_mask", True))
    min_confidence = str(run_entry.get("min_confidence", "medium")).lower()
    max_warning_count = run_entry.get("max_warning_count")
    expected_rois = run_entry.get("expected_non_background_rois")

    notes: List[str] = []
    status = "PASS"

    if require_fixed_mask and not bool(metadata.get("use_fixed_mask")):
        status = "FAIL"
        notes.append("Fixed mask was not enabled during export.")

    if expected_rois is not None and int(expected_rois) != len(roi_summaries):
        status = "FAIL"
        notes.append(f"Expected {expected_rois} ROI summaries, found {len(roi_summaries)}.")

    total_warning_count = 0
    for summary in roi_summaries:
        confidence_label = str(summary["confidence_label"]).lower()
        if CONFIDENCE_RANK.get(confidence_label, 0) < CONFIDENCE_RANK.get(min_confidence, 0):
            status = "FAIL"
            notes.append(
                f"ROI {summary['roi_index'] + 1} confidence {confidence_label!r} is below {min_confidence!r}."
            )
        warning_count = len(summary["warnings"])
        total_warning_count += warning_count
        if warning_count:
            notes.append(
                f"ROI {summary['roi_index'] + 1} warnings: {', '.join(summary['warnings'])}."
            )

    if max_warning_count is not None and total_warning_count > int(max_warning_count):
        status = "FAIL"
        notes.append(
            f"Total warning count {total_warning_count} exceeded max_warning_count={int(max_warning_count)}."
        )

    run_output_dir = bundle_dir / label
    run_output_dir.mkdir(parents=True, exist_ok=True)
    copied_artifacts = _copy_artifacts(analysis_dir, run_output_dir)

    run_summary = {
        "label": label,
        "status": status,
        "video_path": video_path,
        "analysis_dir": str(analysis_dir),
        "metadata_path": str(metadata_path),
        "notes": notes,
        "roi_summaries": roi_summaries,
        "copied_artifacts": copied_artifacts,
    }
    (run_output_dir / "review_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )

    return RunEvaluation(
        label=label,
        status=status,
        video_path=video_path,
        analysis_dir=str(analysis_dir),
        metadata_path=str(metadata_path),
        copied_artifacts=copied_artifacts,
        notes=notes,
        roi_summaries=roi_summaries,
    )


def _render_report(evaluations: List[RunEvaluation], bundle_dir: Path) -> str:
    lines = [
        "# Real-Video Pixel Mask Review",
        "",
        "| Run | Status | ROI Count | Notes |",
        "|---|---|---:|---|",
    ]
    for evaluation in evaluations:
        note_text = "; ".join(evaluation.notes) if evaluation.notes else "No blocking issues."
        lines.append(
            f"| {evaluation.label} | {evaluation.status} | {len(evaluation.roi_summaries)} | {note_text} |"
        )

    lines.extend(["", "## Run Details", ""])
    for evaluation in evaluations:
        lines.append(f"### {evaluation.label}")
        lines.append("")
        lines.append(f"- Status: `{evaluation.status}`")
        lines.append(f"- Video: `{evaluation.video_path}`")
        lines.append(f"- Analysis dir: `{evaluation.analysis_dir}`")
        lines.append(f"- Metadata: `{evaluation.metadata_path or 'n/a'}`")
        lines.append(f"- Review bundle: `{bundle_dir / evaluation.label}`")
        if evaluation.notes:
            lines.append(f"- Notes: {'; '.join(evaluation.notes)}")
        else:
            lines.append("- Notes: No blocking issues.")
        for summary in evaluation.roi_summaries:
            lines.append(
                f"- ROI {summary['roi_index'] + 1}: confidence `{summary['confidence_label']}`, "
                f"pixels `{summary['pixel_count']}`, warnings `{', '.join(summary['warnings']) or 'none'}`, "
                f"source frames `{summary['source_frames']}`"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest describing exported real-video analyses")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("review_output"),
        help="Directory where the review bundle should be written",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations = [_evaluate_run(run_entry, output_dir) for run_entry in manifest["runs"]]
    report_path = output_dir / "review_report.md"
    report_path.write_text(_render_report(evaluations, output_dir), encoding="utf-8")

    index_path = output_dir / "review_index.json"
    index_payload = {
        "runs": [
            {
                "label": evaluation.label,
                "status": evaluation.status,
                "metadata_path": evaluation.metadata_path,
                "copied_artifacts": evaluation.copied_artifacts,
            }
            for evaluation in evaluations
        ]
    }
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    print(f"Wrote review bundle to {output_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
