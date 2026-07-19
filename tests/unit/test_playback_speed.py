import pytest

from ecl_analysis.video_analyzer import _parse_speed_text


@pytest.mark.parametrize(
    "speed_text,expected",
    [
        ("0.25×", 0.25),
        ("0.5×", 0.5),
        ("1×", 1.0),
        ("2×", 2.0),
        ("4×", 4.0),
    ],
)
def test_parse_speed_text_unicode_multiplication_sign(speed_text, expected):
    assert _parse_speed_text(speed_text) == expected


def test_parse_speed_text_ascii_x():
    assert _parse_speed_text("1x") == 1.0
    assert _parse_speed_text("2X") == 2.0
