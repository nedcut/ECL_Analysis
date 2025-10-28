import pytest

from ecl_analysis.video_analyzer import _hex_to_rgba


def test_hex_to_rgba_valid_color():
    assert _hex_to_rgba("#112233", 0.5) == "rgba(17,34,51,0.5)"
    assert _hex_to_rgba("445566", 1.2) == "rgba(68,85,102,1.0)"
    assert _hex_to_rgba("#ABCDEF", -0.1) == "rgba(171,205,239,0.0)"


def test_hex_to_rgba_invalid_input():
    with pytest.raises(ValueError):
        _hex_to_rgba("#12345", 0.5)

    with pytest.raises(ValueError):
        _hex_to_rgba("xyzxyz", 0.5)
