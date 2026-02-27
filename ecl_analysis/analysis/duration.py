"""Run duration validation helpers."""


def validate_run_duration(
    start_frame: int,
    end_frame: int,
    expected_duration: float,
    fps: float,
) -> float:
    """
    Return confidence (0.0-1.0) for detected duration versus expected duration.
    """
    if expected_duration <= 0.0 or fps <= 0.0:
        return 1.0

    actual_duration = (end_frame - start_frame + 1) / fps
    duration_difference = abs(actual_duration - expected_duration)

    tolerance = expected_duration * 0.2
    if duration_difference <= tolerance:
        return 1.0

    max_difference = expected_duration * 0.8
    return max(0.0, 1.0 - (duration_difference - tolerance) / max_difference)
