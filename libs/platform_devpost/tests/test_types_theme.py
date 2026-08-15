"""Tests for types: Theme."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_devpost.types import (
    DisplayedLocation,
    Theme,
    decode_displayed_location,
    decode_theme,
    encode_displayed_location,
    encode_theme,
)


class TestTheme:
    """Tests for Theme type and encode/decode."""

    def test_theme_creation(self) -> None:
        """Test creating a Theme instance."""
        theme = Theme(id=1, name="AI/ML")
        assert theme.id == 1
        assert theme.name == "AI/ML"

    def test_encode_theme(self) -> None:
        """Test encoding Theme to dict."""
        theme = Theme(id=42, name="Blockchain")
        result = encode_theme(theme)
        assert result == {"id": 42, "name": "Blockchain"}

    def test_decode_theme(self) -> None:
        """Test decoding Theme from dict."""
        data: JSONObject = {"id": 10, "name": "Web3"}
        theme = decode_theme(data)
        assert theme.id == 10
        assert theme.name == "Web3"

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip preserves data."""
        original = Theme(id=99, name="Healthcare")
        decoded = decode_theme(encode_theme(original))
        assert decoded.id == original.id
        assert decoded.name == original.name

    def test_decode_theme_missing_id(self) -> None:
        """Test decode_theme raises on missing id."""
        data: JSONObject = {"name": "Test"}
        with pytest.raises(JSONTypeError, match="Missing required field 'id'"):
            decode_theme(data)

    def test_decode_theme_missing_name(self) -> None:
        """Test decode_theme raises on missing name."""
        data: JSONObject = {"id": 1}
        with pytest.raises(JSONTypeError, match="Missing required field 'name'"):
            decode_theme(data)


class TestDisplayedLocation:
    """Tests for DisplayedLocation type and encode/decode."""

    def test_location_creation(self) -> None:
        """Test creating a DisplayedLocation instance."""
        loc = DisplayedLocation(icon="globe", location="Online")
        assert loc.icon == "globe"
        assert loc.location == "Online"

    def test_encode_displayed_location(self) -> None:
        """Test encoding DisplayedLocation to dict."""
        loc = DisplayedLocation(icon="map", location="San Francisco, CA")
        result = encode_displayed_location(loc)
        assert result == {"icon": "map", "location": "San Francisco, CA"}

    def test_decode_displayed_location(self) -> None:
        """Test decoding DisplayedLocation from dict."""
        data: JSONObject = {"icon": "pin", "location": "Remote"}
        loc = decode_displayed_location(data)
        assert loc.icon == "pin"
        assert loc.location == "Remote"

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip preserves data."""
        original = DisplayedLocation(icon="building", location="NYC")
        decoded = decode_displayed_location(encode_displayed_location(original))
        assert decoded.icon == original.icon
        assert decoded.location == original.location
