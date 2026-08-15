"""Tests for types: Hackathon."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_devpost.types import (
    DisplayedLocation,
    Hackathon,
    Theme,
    decode_hackathon,
    encode_hackathon,
)


class TestHackathon:
    """Tests for Hackathon type and encode/decode."""

    def _make_hackathon(self) -> Hackathon:
        """Create a sample hackathon for testing."""
        return Hackathon(
            id=12345,
            title="AI Innovation Challenge",
            url="https://devpost.com/hackathons/ai-challenge",
            thumbnail_url="https://example.com/thumb.jpg",
            organization_name="TechCorp",
            displayed_location=DisplayedLocation(icon="globe", location="Online"),
            open_state="open",
            time_left_to_submission="3 days left",
            submission_period_dates="Jan 1 - Jan 31, 2025",
            themes=(Theme(id=1, name="AI/ML"), Theme(id=2, name="Data Science")),
            prize_amount="$10,000",
            registrations_count=500,
            featured=True,
            winners_announced=False,
            invite_only=False,
        )

    def test_hackathon_creation(self) -> None:
        """Test creating a Hackathon instance."""
        h = self._make_hackathon()
        assert h.id == 12345
        assert h.title == "AI Innovation Challenge"
        assert h.featured is True

    def test_encode_hackathon(self) -> None:
        """Test encoding Hackathon to dict."""
        h = self._make_hackathon()
        result = encode_hackathon(h)
        assert result["id"] == 12345
        assert result["title"] == "AI Innovation Challenge"
        assert result["featured"] is True
        themes_list = result["themes"]
        assert themes_list == [
            {"id": 1, "name": "AI/ML"},
            {"id": 2, "name": "Data Science"},
        ]

    def test_decode_hackathon(self) -> None:
        """Test decoding Hackathon from dict."""
        data: JSONObject = {
            "id": 999,
            "title": "Test Hackathon",
            "url": "https://devpost.com/test",
            "thumbnail_url": "https://example.com/img.jpg",
            "organization_name": "TestOrg",
            "displayed_location": {"icon": "pin", "location": "Virtual"},
            "open_state": "upcoming",
            "time_left_to_submission": "10 days",
            "submission_period_dates": "Feb 1-28",
            "themes": [{"id": 5, "name": "Gaming"}],
            "prize_amount": "$5,000",
            "registrations_count": 100,
            "featured": False,
            "winners_announced": True,
            "invite_only": True,
        }
        h = decode_hackathon(data)
        assert h.id == 999
        assert h.open_state == "upcoming"
        assert h.invite_only is True
        assert len(h.themes) == 1

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip preserves data."""
        original = self._make_hackathon()
        decoded = decode_hackathon(encode_hackathon(original))
        assert decoded.id == original.id
        assert decoded.title == original.title
        assert decoded.featured == original.featured

    def test_decode_hackathon_invalid_state(self) -> None:
        """Test decode_hackathon raises on invalid state."""
        data: JSONObject = {
            "id": 1,
            "title": "Test",
            "url": "https://example.com",
            "thumbnail_url": "https://example.com/img.jpg",
            "organization_name": "Org",
            "displayed_location": {"icon": "x", "location": "x"},
            "open_state": "invalid_state",
            "time_left_to_submission": "1 day",
            "submission_period_dates": "Jan 1-2",
            "themes": [],
            "prize_amount": "$0",
            "registrations_count": 0,
            "featured": False,
            "winners_announced": False,
            "invite_only": False,
        }
        with pytest.raises(JSONTypeError, match="must be a valid state"):
            decode_hackathon(data)

    def test_decode_hackathon_invalid_location_type(self) -> None:
        """Test decode_hackathon raises when location is not a dict."""
        data: JSONObject = {
            "id": 1,
            "title": "Test",
            "url": "https://example.com",
            "thumbnail_url": "https://example.com/img.jpg",
            "organization_name": "Org",
            "displayed_location": "not a dict",
            "open_state": "open",
            "time_left_to_submission": "1 day",
            "submission_period_dates": "Jan 1-2",
            "themes": [],
            "prize_amount": "$0",
            "registrations_count": 0,
            "featured": False,
            "winners_announced": False,
            "invite_only": False,
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_hackathon(data)

    def test_decode_hackathon_invalid_theme_type(self) -> None:
        """Test decode_hackathon raises when theme is not a dict."""
        data: JSONObject = {
            "id": 1,
            "title": "Test",
            "url": "https://example.com",
            "thumbnail_url": "https://example.com/img.jpg",
            "organization_name": "Org",
            "displayed_location": {"icon": "x", "location": "x"},
            "open_state": "open",
            "time_left_to_submission": "1 day",
            "submission_period_dates": "Jan 1-2",
            "themes": ["not a dict"],
            "prize_amount": "$0",
            "registrations_count": 0,
            "featured": False,
            "winners_announced": False,
            "invite_only": False,
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_hackathon(data)

    def test_all_hackathon_states(self) -> None:
        """Test all valid hackathon states."""
        states = ["open", "upcoming", "ended", "submissions"]
        for state in states:
            data: JSONObject = {
                "id": 1,
                "title": "Test",
                "url": "https://example.com",
                "thumbnail_url": "https://example.com/img.jpg",
                "organization_name": "Org",
                "displayed_location": {"icon": "x", "location": "x"},
                "open_state": state,
                "time_left_to_submission": "1 day",
                "submission_period_dates": "Jan 1-2",
                "themes": [],
                "prize_amount": "$0",
                "registrations_count": 0,
                "featured": False,
                "winners_announced": False,
                "invite_only": False,
            }
            h = decode_hackathon(data)
            assert h.open_state == state
