"""Tests for coordinate key functions."""

import pytest

from tankpit_bot.state import coord_key, parse_coord_key


class TestCoordKey:
    """Tests for coord_key function."""

    def test_creates_key(self) -> None:
        """Creates comma-separated key."""
        assert coord_key(100, 200) == "100,200"

    def test_zero_coords(self) -> None:
        """Handles zero coordinates."""
        assert coord_key(0, 0) == "0,0"

    def test_max_coords(self) -> None:
        """Handles maximum coordinates."""
        assert coord_key(255, 255) == "255,255"


class TestParseCoordKey:
    """Tests for parse_coord_key function."""

    def test_parses_key(self) -> None:
        """Parses comma-separated key."""
        x, y = parse_coord_key("100,200")
        assert x == 100
        assert y == 200

    def test_zero_coords(self) -> None:
        """Parses zero coordinates."""
        x, y = parse_coord_key("0,0")
        assert x == 0
        assert y == 0

    def test_invalid_format_raises(self) -> None:
        """Raises ValueError for invalid format."""
        with pytest.raises(ValueError, match="Invalid coord key format"):
            parse_coord_key("invalid")

    def test_too_many_parts_raises(self) -> None:
        """Raises ValueError for too many parts."""
        with pytest.raises(ValueError):
            parse_coord_key("1,2,3")
