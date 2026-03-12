from __future__ import annotations

import json
from pathlib import Path

from tools.run_real_video_review import main


def test_real_video_review_bundle_generation(tmp_path: Path, monkeypatch):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    metadata_path = analysis_dir / "Demo_input_frames1-3_analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "use_fixed_mask": True,
                "mask_metadata": [
                    {
                        "confidence_label": "high",
                        "pixel_count": 24,
                        "warnings": [],
                        "source_frames": [3, 4, 5],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (analysis_dir / "Demo_input_ROI1_frames1-3_brightness.csv").write_text(
        "frame,brightness_mean\n0,1\n",
        encoding="utf-8",
    )
    (analysis_dir / "Demo_input_plot.png").write_bytes(b"png")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "label": "demo-run",
                        "video_path": str(tmp_path / "raw.mp4"),
                        "analysis_dir": str(analysis_dir),
                        "require_fixed_mask": True,
                        "min_confidence": "medium",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "review"
    monkeypatch.setattr(
        "sys.argv",
        ["run_real_video_review.py", str(manifest_path), "--output-dir", str(output_dir)],
    )

    exit_code = main()

    assert exit_code == 0
    assert (output_dir / "review_report.md").exists()
    assert (output_dir / "review_index.json").exists()
    assert (output_dir / "demo-run" / metadata_path.name).exists()
    assert (output_dir / "demo-run" / "review_summary.json").exists()
