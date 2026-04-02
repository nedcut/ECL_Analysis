"""Unit tests for capture metadata sidecar validation."""

from ecl_analysis.ingest.metadata import (
    CURRENT_CAPTURE_SCHEMA_VERSION,
    get_capture_metadata_schema_contract,
    validate_capture_metadata,
)


def test_validate_capture_metadata_valid_sidecar(tmp_path):
    video = tmp_path / "sample.mov"
    video.write_bytes(b"stub")

    sidecar = tmp_path / "sample.capture.json"
    sidecar.write_text(
        """
        {
          "schema_version": "1.0",
          "device_model": "iPhone 15 Pro",
          "exposure_mode_locked": true,
          "exposure_duration": 0.0333,
          "iso": 80,
          "white_balance_mode_locked": true,
          "fps": 30,
          "resolution": "1920x1080",
          "hdr_disabled": true
        }
        """,
        encoding="utf-8",
    )

    result = validate_capture_metadata(str(video))

    assert result.is_valid is True
    assert result.errors == []
    assert result.normalized_metadata is not None
    assert result.normalized_metadata["schema_version"] == CURRENT_CAPTURE_SCHEMA_VERSION
    assert result.normalized_metadata["resolution"] == "1920x1080"
    assert result.schema_version_assumed is False
    assert result.to_dict()["status"] == "valid"


def test_validate_capture_metadata_missing_sidecar(tmp_path):
    video = tmp_path / "sample.mov"
    video.write_bytes(b"stub")

    result = validate_capture_metadata(str(video))

    assert result.is_valid is False
    assert any("Missing capture metadata sidecar" in err for err in result.errors)
    assert result.to_dict()["status"] == "invalid"


def test_validate_capture_metadata_missing_schema_version_warns_only(tmp_path):
    video = tmp_path / "sample.mov"
    video.write_bytes(b"stub")

    sidecar = tmp_path / "sample.capture.json"
    sidecar.write_text(
        """
        {
          "device_model": "iPhone 15 Pro",
          "exposure_mode_locked": true,
          "exposure_duration": 0.0167,
          "iso": 55,
          "white_balance_mode_locked": true,
          "fps": 60,
          "resolution": {"width": 1920, "height": 1080},
          "hdr_disabled": true
        }
        """,
        encoding="utf-8",
    )

    result = validate_capture_metadata(str(video))

    assert result.is_valid is True
    assert any("schema_version missing" in warning for warning in result.warnings)
    assert result.normalized_metadata is not None
    assert result.normalized_metadata["schema_version"] == CURRENT_CAPTURE_SCHEMA_VERSION
    assert result.normalized_metadata["resolution"] == "1920x1080"
    assert result.schema_version_assumed is True
    assert result.to_dict()["schema_version_assumed"] is True


def test_capture_metadata_schema_contract_is_versioned():
    contract = get_capture_metadata_schema_contract()

    assert contract["schema_name"] == "ecl_capture_metadata"
    assert contract["schema_version"] == CURRENT_CAPTURE_SCHEMA_VERSION
    assert "device_model" in contract["required_fields"]
    assert "capture_id" in contract["optional_fields"]


def test_validate_capture_metadata_tracks_unrecognized_fields(tmp_path):
    video = tmp_path / "sample.mov"
    video.write_bytes(b"stub")

    sidecar = tmp_path / "sample.capture.json"
    sidecar.write_text(
        """
        {
          "schema_version": "1.0",
          "device_model": "iPhone 15 Pro",
          "exposure_mode_locked": true,
          "exposure_duration": 0.0333,
          "iso": 80,
          "white_balance_mode_locked": true,
          "fps": 30,
          "resolution": "1920x1080",
          "hdr_disabled": true,
          "unexpected_field": "present"
        }
        """,
        encoding="utf-8",
    )

    result = validate_capture_metadata(str(video))

    assert result.is_valid is True
    assert result.unrecognized_fields == ["unexpected_field"]
    assert result.to_dict()["unrecognized_fields"] == ["unexpected_field"]


def test_validate_capture_metadata_unknown_schema_warns_but_normalizes(tmp_path):
    video = tmp_path / "sample.mov"
    video.write_bytes(b"stub")

    sidecar = tmp_path / "sample.capture.json"
    sidecar.write_text(
        """
        {
          "schema_version": "2.5",
          "device_model": "iPhone 15 Pro",
          "exposure_mode_locked": true,
          "exposure_duration": 0.0167,
          "iso": 55,
          "white_balance_mode_locked": true,
          "fps": 60,
          "resolution": "1920x1080",
          "hdr_disabled": true
        }
        """,
        encoding="utf-8",
    )

    result = validate_capture_metadata(str(video))

    assert result.is_valid is True
    assert any("Unrecognized schema_version" in warning for warning in result.warnings)
    assert result.normalized_metadata is not None
    assert result.normalized_metadata["schema_version"] == "2.5"
