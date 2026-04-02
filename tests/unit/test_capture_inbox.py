from __future__ import annotations

import json
from pathlib import Path

from tools.ingest_capture_inbox import process_inbox_once


def _write_valid_sidecar(path: Path, capture_id: str = "capture-001") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "capture_id": capture_id,
                "device_model": "iPhone 15 Pro",
                "exposure_mode_locked": True,
                "exposure_duration": 0.0333,
                "iso": 80,
                "white_balance_mode_locked": True,
                "fps": 30,
                "resolution": "1920x1080",
                "hdr_disabled": True,
            }
        ),
        encoding="utf-8",
    )


def test_process_inbox_once_writes_summary_and_analysis_result(tmp_path: Path, monkeypatch) -> None:
    inbox_dir = tmp_path / "incoming"
    output_dir = tmp_path / "outputs"
    inbox_dir.mkdir()

    video_path = inbox_dir / "capture.mov"
    video_path.write_bytes(b"video-stub")
    _write_valid_sidecar(inbox_dir / "capture.capture.json")

    def _fake_run_analysis_case(template_case, video_path, capture_label, root_output_dir):
        case_dir = root_output_dir / capture_label
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "analysis_done.txt").write_text("ok", encoding="utf-8")
        return {
            "name": capture_label,
            "passed": True,
            "case_dir": str(case_dir),
        }

    monkeypatch.setattr("tools.ingest_capture_inbox._run_analysis_case", _fake_run_analysis_case)

    result = process_inbox_once(
        {
            "inbox_dir": str(inbox_dir),
            "output_dir": str(output_dir),
            "analysis_case": {"rects": [[[0, 0], [1, 1]]]},
        }
    )

    assert result["processed_count"] == 1
    summary_path = output_dir / "capture-001" / "capture_ingest_summary.json"
    assert summary_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["capture_label"] == "capture-001"
    assert payload["validation"]["is_valid"] is True
    assert payload["analysis_result"]["passed"] is True


def test_process_inbox_once_skips_already_processed_signature(tmp_path: Path, monkeypatch) -> None:
    inbox_dir = tmp_path / "incoming"
    output_dir = tmp_path / "outputs"
    inbox_dir.mkdir()

    video_path = inbox_dir / "capture.mov"
    video_path.write_bytes(b"video-stub")
    _write_valid_sidecar(inbox_dir / "capture.capture.json")

    calls = {"count": 0}

    def _fake_run_analysis_case(template_case, video_path, capture_label, root_output_dir):
        calls["count"] += 1
        return {"name": capture_label, "passed": True, "case_dir": str(root_output_dir / capture_label)}

    monkeypatch.setattr("tools.ingest_capture_inbox._run_analysis_case", _fake_run_analysis_case)

    manifest = {
        "inbox_dir": str(inbox_dir),
        "output_dir": str(output_dir),
        "analysis_case": {"rects": [[[0, 0], [1, 1]]]},
    }
    process_inbox_once(manifest)
    second = process_inbox_once(manifest)

    assert calls["count"] == 1
    assert second["processed_count"] == 0
    assert second["skipped_count"] == 1
    assert second["skipped"][0]["existing_summary_path"] == str(
        output_dir / "capture-001" / "capture_ingest_summary.json"
    )


def test_process_inbox_once_force_reprocess_overrides_signature_skip(tmp_path: Path, monkeypatch) -> None:
    inbox_dir = tmp_path / "incoming"
    output_dir = tmp_path / "outputs"
    inbox_dir.mkdir()

    video_path = inbox_dir / "capture.mov"
    video_path.write_bytes(b"video-stub")
    _write_valid_sidecar(inbox_dir / "capture.capture.json")

    calls = {"count": 0}

    def _fake_run_analysis_case(template_case, video_path, capture_label, root_output_dir):
        calls["count"] += 1
        return {"name": capture_label, "passed": True, "case_dir": str(root_output_dir / capture_label)}

    monkeypatch.setattr("tools.ingest_capture_inbox._run_analysis_case", _fake_run_analysis_case)

    manifest = {
        "inbox_dir": str(inbox_dir),
        "output_dir": str(output_dir),
        "analysis_case": {"rects": [[[0, 0], [1, 1]]]},
        "force_reprocess": True,
    }
    first = process_inbox_once(manifest)
    second = process_inbox_once(manifest)

    assert calls["count"] == 2
    assert first["processed_count"] == 1
    assert second["processed_count"] == 1
    assert second["skipped_count"] == 0


def test_process_inbox_once_archives_sources_after_processing(tmp_path: Path, monkeypatch) -> None:
    inbox_dir = tmp_path / "incoming"
    output_dir = tmp_path / "outputs"
    archive_dir = tmp_path / "archive"
    inbox_dir.mkdir()

    video_path = inbox_dir / "capture.mov"
    sidecar_path = inbox_dir / "capture.capture.json"
    video_path.write_bytes(b"video-stub")
    _write_valid_sidecar(sidecar_path)

    monkeypatch.setattr(
        "tools.ingest_capture_inbox._run_analysis_case",
        lambda template_case, video_path, capture_label, root_output_dir: {"name": capture_label, "passed": True},
    )

    process_inbox_once(
        {
            "inbox_dir": str(inbox_dir),
            "output_dir": str(output_dir),
            "archive_dir": str(archive_dir),
            "analysis_case": {"rects": [[[0, 0], [1, 1]]]},
        }
    )

    assert not video_path.exists()
    assert not sidecar_path.exists()
    assert (archive_dir / "capture-001" / "capture.mov").exists()
    assert (archive_dir / "capture-001" / "capture.capture.json").exists()
