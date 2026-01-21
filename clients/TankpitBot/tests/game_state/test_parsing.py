"""Tests for game state parsing functions."""

from __future__ import annotations

from tankpit_bot.game_state import (
    NearbyEntity,
    parse_location,
    parse_radar_detection,
)

# =============================================================================
# parse_location Tests
# =============================================================================


def test_parse_location_valid() -> None:
    """Test parsing valid location string."""
    result = parse_location("123,456")
    assert result["x"] == 123
    assert result["y"] == 456
    assert result["raw"] == "123,456"


def test_parse_location_with_spaces() -> None:
    """Test parsing location with spaces."""
    result = parse_location(" 100 , 200 ")
    assert result["x"] == 100
    assert result["y"] == 200


def test_parse_location_empty_string() -> None:
    """Test parsing empty location string."""
    result = parse_location("")
    assert result["x"] == 0
    assert result["y"] == 0
    assert result["raw"] == ""


def test_parse_location_no_comma() -> None:
    """Test parsing location without comma."""
    result = parse_location("12345")
    assert result["x"] == 0
    assert result["y"] == 0
    assert result["raw"] == "12345"


def test_parse_location_too_many_parts() -> None:
    """Test parsing location with too many parts."""
    result = parse_location("1,2,3")
    assert result["x"] == 0
    assert result["y"] == 0
    assert result["raw"] == "1,2,3"


def test_parse_location_non_numeric() -> None:
    """Test parsing location with non-numeric values."""
    result = parse_location("abc,def")
    assert result["x"] == 0
    assert result["y"] == 0


def test_parse_location_mixed_numeric() -> None:
    """Test parsing location with mixed values."""
    result = parse_location("123,abc")
    assert result["x"] == 0
    assert result["y"] == 0


# =============================================================================
# parse_radar_detection Tests
# =============================================================================


def test_parse_radar_detection_basic() -> None:
    """Test parsing basic radar detection."""
    result = parse_radar_detection("blue-7 detected to W [57,135]")
    # Verify all fields are correctly parsed
    assert result == NearbyEntity(
        name="blue-7",
        direction="W",
        coordinates="57,135",
        is_private=False,
    )


def test_parse_radar_detection_with_private() -> None:
    """Test parsing radar detection with private flag."""
    result = parse_radar_detection("blue-7 (private) detected to W [57,135]")
    # Verify private flag is correctly parsed
    assert result == NearbyEntity(
        name="blue-7",
        direction="W",
        coordinates="57,135",
        is_private=True,
    )


def test_parse_radar_detection_cardinal_directions() -> None:
    """Test parsing all cardinal directions."""
    for direction in ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]:
        result = parse_radar_detection(f"enemy detected to {direction} [0,0]")
        # Each direction should parse to correct NearbyEntity
        assert result == NearbyEntity(
            name="enemy",
            direction=direction,
            coordinates="0,0",
            is_private=False,
        )


def test_parse_radar_detection_with_spaces() -> None:
    """Test parsing radar detection with extra spaces."""
    result = parse_radar_detection("  red-1 detected to NE [100,200]  ")
    # Verify spaces are trimmed correctly
    assert result == NearbyEntity(
        name="red-1",
        direction="NE",
        coordinates="100,200",
        is_private=False,
    )


def test_parse_radar_detection_invalid_format() -> None:
    """Test parsing invalid radar detection format."""
    result = parse_radar_detection("some random text")
    assert result is None


def test_parse_radar_detection_missing_brackets() -> None:
    """Test parsing radar detection without brackets."""
    result = parse_radar_detection("blue-7 detected to W 57,135")
    assert result is None


def test_parse_radar_detection_empty_string() -> None:
    """Test parsing empty string."""
    result = parse_radar_detection("")
    assert result is None
