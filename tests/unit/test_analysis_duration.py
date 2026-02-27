import pytest

from ecl_analysis.analysis.duration import validate_run_duration


def test_validate_run_duration_perfect_match():
    confidence = validate_run_duration(start_frame=0, end_frame=59, expected_duration=2.0, fps=30.0)
    assert confidence == pytest.approx(1.0)


def test_validate_run_duration_returns_one_when_fps_invalid():
    confidence = validate_run_duration(start_frame=0, end_frame=10, expected_duration=1.0, fps=0.0)
    assert confidence == pytest.approx(1.0)


def test_validate_run_duration_drops_for_large_mismatch():
    confidence = validate_run_duration(start_frame=0, end_frame=299, expected_duration=2.0, fps=30.0)
    assert confidence < 0.5
