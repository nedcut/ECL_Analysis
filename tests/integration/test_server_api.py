"""API-level tests for the local web server against a synthetic video."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ecl_analysis.server.app import create_app


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Write a small MP4 whose left half brightens over 30 frames."""
    path = tmp_path_factory.mktemp("videos") / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
    )
    assert writer.isOpened()
    for i in range(30):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :32] = min(255, 8 * i)
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _open_video(client, path):
    response = client.post("/api/videos", json={"path": path})
    assert response.status_code == 200, response.text
    return response.json()


def test_open_video_returns_metadata(client, synthetic_video):
    meta = _open_video(client, synthetic_video)
    assert meta["frame_count"] == 30
    assert meta["width"] == 64
    assert meta["height"] == 48
    assert meta["fps"] == pytest.approx(30.0)


def test_open_missing_video_is_a_client_error(client):
    response = client.post("/api/videos", json={"path": "/nonexistent/video.mp4"})
    assert response.status_code == 400


def test_frame_endpoint_serves_jpeg(client, synthetic_video):
    meta = _open_video(client, synthetic_video)
    response = client.get(f"/api/videos/{meta['video_id']}/frame/5")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    decoded = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape == (48, 64, 3)


def test_frame_out_of_range_is_404(client, synthetic_video):
    meta = _open_video(client, synthetic_video)
    response = client.get(f"/api/videos/{meta['video_id']}/frame/999")
    assert response.status_code == 404


def test_threshold_overlay_changes_bright_frame(client, synthetic_video):
    meta = _open_video(client, synthetic_video)
    plain = client.get(f"/api/videos/{meta['video_id']}/frame/29")
    tinted = client.get(f"/api/videos/{meta['video_id']}/frame/29?threshold=50")
    assert plain.status_code == tinted.status_code == 200
    assert plain.content != tinted.content


def _wait_for_job(client, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/analysis/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"done", "error", "cancelled"}:
            return payload
        time.sleep(0.05)
    pytest.fail(f"Job {job_id} did not finish within {timeout}s")


def test_analysis_job_produces_series(client, synthetic_video):
    meta = _open_video(client, synthetic_video)
    response = client.post(
        f"/api/videos/{meta['video_id']}/analyze",
        json={
            "rois": [
                {"x1": 0, "y1": 0, "x2": 32, "y2": 48, "name": "electrode"},
                {"x1": 40, "y1": 0, "x2": 64, "y2": 48, "name": "background"},
            ],
            "background_roi_idx": 1,
            "start_frame": 0,
            "end_frame": 29,
        },
    )
    assert response.status_code == 200, response.text
    payload = _wait_for_job(client, response.json()["job_id"])

    assert payload["status"] == "done", payload
    result = payload["result"]
    assert result["frames_processed"] == 30
    assert len(result["rois"]) == 1
    series = result["rois"][0]["brightness_mean"]
    assert len(series) == 30
    # The left half brightens monotonically, so late frames must beat early ones.
    assert series[-1] > series[0]


def test_analysis_rejects_background_only(client, synthetic_video):
    meta = _open_video(client, synthetic_video)
    response = client.post(
        f"/api/videos/{meta['video_id']}/analyze",
        json={
            "rois": [{"x1": 0, "y1": 0, "x2": 32, "y2": 48}],
            "background_roi_idx": 0,
            "start_frame": 0,
            "end_frame": 29,
        },
    )
    assert response.status_code == 400


def test_export_writes_csv_and_serves_it(client, synthetic_video, tmp_path):
    meta = _open_video(client, synthetic_video)
    response = client.post(
        f"/api/videos/{meta['video_id']}/analyze",
        json={
            "rois": [{"x1": 0, "y1": 0, "x2": 32, "y2": 48, "name": "electrode"}],
            "start_frame": 0,
            "end_frame": 29,
        },
    )
    job_id = response.json()["job_id"]
    payload = _wait_for_job(client, job_id)
    assert payload["status"] == "done", payload

    export_dir = tmp_path / "exports"
    response = client.post(
        f"/api/analysis/{job_id}/export",
        json={
            "analysis_name": "synthetic_run",
            "save_dir": str(export_dir),
            "csv": True,
            "plot": False,
            "interactive_plot": False,
        },
    )
    assert response.status_code == 200, response.text
    out_paths = response.json()["out_paths"]
    csv_paths = [p for p in out_paths if p.endswith(".csv")]
    assert csv_paths, out_paths

    served = client.get(f"/api/analysis/{job_id}/files", params={"path": csv_paths[0]})
    assert served.status_code == 200
    assert "frame" in served.text.splitlines()[0]

    denied = client.get(
        f"/api/analysis/{job_id}/files", params={"path": synthetic_video}
    )
    assert denied.status_code == 403


def test_fs_listing(client, tmp_path):
    (tmp_path / "clip.mp4").touch()
    (tmp_path / "subdir").mkdir()
    response = client.get("/api/fs", params={"path": str(tmp_path)})
    assert response.status_code == 200
    payload = response.json()
    assert "clip.mp4" in payload["videos"]
    assert "subdir" in payload["dirs"]
