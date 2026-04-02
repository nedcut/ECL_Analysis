"""Validation helpers for camera-capture sidecar metadata."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

CAPTURE_METADATA_SCHEMA_NAME = "ecl_capture_metadata"
CURRENT_CAPTURE_SCHEMA_VERSION = "1.0"
CAPTURE_METADATA_SCHEMA_VERSION = CURRENT_CAPTURE_SCHEMA_VERSION
SUPPORTED_CAPTURE_SCHEMA_VERSIONS = {CURRENT_CAPTURE_SCHEMA_VERSION}
REQUIRED_CAPTURE_FIELDS = (
    "device_model",
    "exposure_mode_locked",
    "exposure_duration",
    "iso",
    "white_balance_mode_locked",
    "fps",
    "resolution",
    "hdr_disabled",
)
OPTIONAL_CAPTURE_FIELDS = (
    "capture_id",
    "recorded_at",
    "app_version",
    "ios_version",
    "color_space",
    "video_codec",
)
KNOWN_CAPTURE_FIELDS = ("schema_version",) + REQUIRED_CAPTURE_FIELDS + OPTIONAL_CAPTURE_FIELDS

CAPTURE_METADATA_FIELD_SPECS: Dict[str, Dict[str, str]] = {
    "schema_version": {
        "type": "string",
        "required": "false during transition; assumed when omitted",
        "description": "Version of the sidecar contract emitted by capture.",
    },
    "device_model": {
        "type": "string",
        "required": "true",
        "description": "Capture device model.",
    },
    "exposure_mode_locked": {
        "type": "boolean",
        "required": "true",
        "description": "Exposure lock state during capture.",
    },
    "exposure_duration": {
        "type": "number",
        "required": "true",
        "description": "Exposure duration in seconds.",
    },
    "iso": {
        "type": "number",
        "required": "true",
        "description": "Sensor ISO at capture time.",
    },
    "white_balance_mode_locked": {
        "type": "boolean",
        "required": "true",
        "description": "White-balance lock state during capture.",
    },
    "fps": {
        "type": "number",
        "required": "true",
        "description": "Configured frames per second.",
    },
    "resolution": {
        "type": "string|object",
        "required": "true",
        "description": "Capture resolution as WIDTHxHEIGHT or {width,height}.",
    },
    "hdr_disabled": {
        "type": "boolean",
        "required": "true",
        "description": "Whether HDR/tone mapping was disabled.",
    },
    "capture_id": {
        "type": "string",
        "required": "false",
        "description": "Stable capture identifier for downstream provenance.",
    },
    "recorded_at": {
        "type": "string",
        "required": "false",
        "description": "Capture timestamp in ISO-8601 format.",
    },
    "app_version": {
        "type": "string",
        "required": "false",
        "description": "Version of the capture app.",
    },
    "ios_version": {
        "type": "string",
        "required": "false",
        "description": "iOS version on the capture device.",
    },
    "color_space": {
        "type": "string",
        "required": "false",
        "description": "Recorded color space/profile label.",
    },
    "video_codec": {
        "type": "string",
        "required": "false",
        "description": "Recorded video codec label.",
    },
}


def get_capture_metadata_schema_contract() -> Dict[str, object]:
    """Return the current lightweight sidecar schema contract."""
    return {
        "schema_name": CAPTURE_METADATA_SCHEMA_NAME,
        "schema_version": CAPTURE_METADATA_SCHEMA_VERSION,
        "supported_schema_versions": sorted(SUPPORTED_CAPTURE_SCHEMA_VERSIONS),
        "required_fields": {
            field: dict(CAPTURE_METADATA_FIELD_SPECS[field]) for field in REQUIRED_CAPTURE_FIELDS
        },
        "optional_fields": {
            field: dict(CAPTURE_METADATA_FIELD_SPECS[field]) for field in OPTIONAL_CAPTURE_FIELDS
        },
        "schema_version_field": dict(CAPTURE_METADATA_FIELD_SPECS["schema_version"]),
        "resolution_formats": ["1920x1080", {"width": 1920, "height": 1080}],
    }


@dataclass(frozen=True)
class CaptureMetadataValidation:
    """Structured validation result for capture metadata sidecars."""

    is_valid: bool
    sidecar_path: str
    errors: List[str]
    warnings: List[str]
    metadata: Optional[Dict[str, object]] = None
    normalized_metadata: Optional[Dict[str, object]] = None
    detected_schema_version: Optional[str] = None
    schema_version_assumed: bool = False
    unrecognized_fields: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.is_valid:
            return "invalid"
        if self.warnings:
            return "valid_with_warnings"
        return "valid"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the validation result for export/reporting."""
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "sidecar_path": self.sidecar_path,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "schema_name": CAPTURE_METADATA_SCHEMA_NAME,
            "expected_schema_version": CAPTURE_METADATA_SCHEMA_VERSION,
            "detected_schema_version": self.detected_schema_version,
            "schema_version": (
                None if self.normalized_metadata is None else self.normalized_metadata.get("schema_version")
            ),
            "schema_version_assumed": self.schema_version_assumed,
            "unrecognized_fields": list(self.unrecognized_fields),
            "schema_contract": get_capture_metadata_schema_contract(),
            "metadata": dict(self.metadata) if isinstance(self.metadata, dict) else None,
            "normalized_metadata": (
                dict(self.normalized_metadata) if isinstance(self.normalized_metadata, dict) else None
            ),
        }


def _sidecar_path_for_video(video_path: str) -> str:
    base, _ = os.path.splitext(video_path)
    return f"{base}.capture.json"


def _coerce_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _coerce_positive_float(value: object, field_name: str, errors: List[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field_name} must be numeric.")
        return None
    if numeric <= 0:
        errors.append(f"{field_name} must be greater than 0.")
        return None
    return numeric


def _normalize_resolution(value: object, errors: List[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "")
        if "x" in normalized:
            width, height = normalized.split("x", 1)
            if width.isdigit() and height.isdigit():
                return f"{int(width)}x{int(height)}"
    if isinstance(value, dict):
        width = value.get("width")
        height = value.get("height")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
            return f"{int(width)}x{int(height)}"
    errors.append("resolution must be a string like '1920x1080' or an object with width/height.")
    return None


def validate_capture_metadata(video_path: str) -> CaptureMetadataValidation:
    """Load and validate `<video>.capture.json` sidecar metadata.

    The validator is authoritative about the current schema contract while remaining
    non-blocking at the UI layer. Missing or invalid sidecars still return a
    structured validation object so legacy videos can be analyzed during the
    transition.
    """
    sidecar_path = _sidecar_path_for_video(video_path)
    if not os.path.exists(sidecar_path):
        return CaptureMetadataValidation(
            is_valid=False,
            sidecar_path=sidecar_path,
            errors=["Missing capture metadata sidecar file."],
            warnings=["Expected sidecar format: <video_filename>.capture.json"],
            metadata=None,
            normalized_metadata=None,
        )

    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        return CaptureMetadataValidation(
            is_valid=False,
            sidecar_path=sidecar_path,
            errors=[f"Invalid JSON in sidecar: {exc.msg} (line {exc.lineno}, col {exc.colno})."],
            warnings=[],
            metadata=None,
            normalized_metadata=None,
        )
    except OSError as exc:
        return CaptureMetadataValidation(
            is_valid=False,
            sidecar_path=sidecar_path,
            errors=[f"Could not read sidecar file: {exc}"],
            warnings=[],
            metadata=None,
            normalized_metadata=None,
        )

    if not isinstance(payload, dict):
        return CaptureMetadataValidation(
            is_valid=False,
            sidecar_path=sidecar_path,
            errors=["Sidecar JSON must be an object at the top level."],
            warnings=[],
            metadata=None,
            normalized_metadata=None,
        )

    errors: List[str] = []
    warnings: List[str] = []
    normalized_metadata: Dict[str, object] = {}

    schema_version_raw = payload.get("schema_version")
    schema_version_assumed = False
    detected_schema_version: Optional[str] = None
    if schema_version_raw is None:
        warnings.append(
            f"schema_version missing; assuming legacy metadata compatible with schema {CURRENT_CAPTURE_SCHEMA_VERSION}."
        )
        normalized_metadata["schema_version"] = CURRENT_CAPTURE_SCHEMA_VERSION
        schema_version_assumed = True
    else:
        schema_version = str(schema_version_raw).strip()
        detected_schema_version = schema_version or None
        if not schema_version:
            warnings.append(
                f"schema_version empty; assuming legacy metadata compatible with schema {CURRENT_CAPTURE_SCHEMA_VERSION}."
            )
            normalized_metadata["schema_version"] = CURRENT_CAPTURE_SCHEMA_VERSION
            schema_version_assumed = True
        else:
            normalized_metadata["schema_version"] = schema_version
        if schema_version and schema_version not in SUPPORTED_CAPTURE_SCHEMA_VERSIONS:
            warnings.append(f"Unrecognized schema_version '{schema_version}'; attempting best-effort validation.")

    missing_fields = [field for field in REQUIRED_CAPTURE_FIELDS if field not in payload]
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")

    device_model = payload.get("device_model")
    if isinstance(device_model, str) and device_model.strip():
        normalized_metadata["device_model"] = device_model.strip()
    else:
        errors.append("device_model must be a non-empty string.")

    exposure_locked = _coerce_bool(payload.get("exposure_mode_locked"))
    if exposure_locked is not None:
        normalized_metadata["exposure_mode_locked"] = exposure_locked
    if exposure_locked is not True:
        errors.append("exposure_mode_locked must be true.")

    wb_locked = _coerce_bool(payload.get("white_balance_mode_locked"))
    if wb_locked is not None:
        normalized_metadata["white_balance_mode_locked"] = wb_locked
    if wb_locked is not True:
        errors.append("white_balance_mode_locked must be true.")

    hdr_disabled = _coerce_bool(payload.get("hdr_disabled"))
    if hdr_disabled is not None:
        normalized_metadata["hdr_disabled"] = hdr_disabled
    if hdr_disabled is not True:
        warnings.append("hdr_disabled is not explicitly true; HDR may distort brightness traces.")

    exposure_duration = _coerce_positive_float(payload.get("exposure_duration"), "exposure_duration", errors)
    if exposure_duration is not None:
        normalized_metadata["exposure_duration"] = exposure_duration

    fps = _coerce_positive_float(payload.get("fps"), "fps", errors)
    if fps is not None:
        normalized_metadata["fps"] = fps

    iso_numeric = _coerce_positive_float(payload.get("iso"), "iso", errors)
    if iso_numeric is not None:
        normalized_metadata["iso"] = iso_numeric

    resolution = _normalize_resolution(payload.get("resolution"), errors)
    if resolution is not None:
        normalized_metadata["resolution"] = resolution

    for optional_field in OPTIONAL_CAPTURE_FIELDS:
        value = payload.get(optional_field)
        if isinstance(value, str) and value.strip():
            normalized_metadata[optional_field] = value.strip()

    unrecognized_fields = sorted(field for field in payload.keys() if field not in KNOWN_CAPTURE_FIELDS)

    return CaptureMetadataValidation(
        is_valid=not errors,
        sidecar_path=sidecar_path,
        errors=errors,
        warnings=warnings,
        metadata=payload,
        normalized_metadata=normalized_metadata,
        detected_schema_version=detected_schema_version,
        schema_version_assumed=schema_version_assumed,
        unrecognized_fields=unrecognized_fields,
    )
