"""Tests for types: HackathonMatch."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_devpost.types import (
    DisplayedLocation,
    Hackathon,
    HackathonMatch,
    InterestFilter,
    decode_filter,
    decode_match,
    encode_filter,
    encode_match,
)


class TestHackathonMatch:
    """Tests for HackathonMatch type and encode/decode."""

    def _make_hackathon(self) -> Hackathon:
        """Create a sample hackathon for testing."""
        return Hackathon(
            id=1,
            title="Test",
            url="https://example.com",
            thumbnail_url="https://example.com/img.jpg",
            organization_name="Org",
            displayed_location=DisplayedLocation(icon="x", location="x"),
            open_state="open",
            time_left_to_submission="1 day",
            submission_period_dates="Jan 1-2",
            themes=(),
            prize_amount="$0",
            registrations_count=0,
            featured=False,
            winners_announced=False,
            invite_only=False,
        )

    def test_match_creation(self) -> None:
        """Test creating a HackathonMatch instance."""
        match = HackathonMatch(
            hackathon=self._make_hackathon(),
            match_score=0.85,
            matched_capabilities=("web_development", "python_development"),
            missing_capabilities=("machine_learning",),
            recommendation="good_fit",
        )
        assert match.match_score == 0.85
        assert len(match.matched_capabilities) == 2
        assert match.recommendation == "good_fit"

    def test_encode_match(self) -> None:
        """Test encoding HackathonMatch to dict."""
        match = HackathonMatch(
            hackathon=self._make_hackathon(),
            match_score=0.5,
            matched_capabilities=("ai",),
            missing_capabilities=(),
            recommendation="stretch",
        )
        result = encode_match(match)
        assert result["match_score"] == 0.5
        assert result["recommendation"] == "stretch"

    def test_decode_match(self) -> None:
        """Test decoding HackathonMatch from dict."""
        data: JSONObject = {
            "hackathon": {
                "id": 1,
                "title": "Test",
                "url": "https://example.com",
                "thumbnail_url": "https://example.com/img.jpg",
                "organization_name": "Org",
                "displayed_location": {"icon": "x", "location": "x"},
                "open_state": "open",
                "time_left_to_submission": "1 day",
                "submission_period_dates": "Jan 1-2",
                "themes": [],
                "prize_amount": "$0",
                "registrations_count": 0,
                "featured": False,
                "winners_announced": False,
                "invite_only": False,
            },
            "match_score": 0.75,
            "matched_capabilities": ["cap1"],
            "missing_capabilities": ["cap2", "cap3"],
            "recommendation": "strong_fit",
        }
        match = decode_match(data)
        assert match.match_score == 0.75
        assert match.recommendation == "strong_fit"

    def test_decode_match_all_recommendations(self) -> None:
        """Test all valid recommendation values."""
        recs = ["strong_fit", "good_fit", "stretch", "new_territory"]
        for rec in recs:
            data: JSONObject = {
                "hackathon": {
                    "id": 1,
                    "title": "Test",
                    "url": "https://example.com",
                    "thumbnail_url": "https://example.com/img.jpg",
                    "organization_name": "Org",
                    "displayed_location": {"icon": "x", "location": "x"},
                    "open_state": "open",
                    "time_left_to_submission": "1 day",
                    "submission_period_dates": "Jan 1-2",
                    "themes": [],
                    "prize_amount": "$0",
                    "registrations_count": 0,
                    "featured": False,
                    "winners_announced": False,
                    "invite_only": False,
                },
                "match_score": 0.5,
                "matched_capabilities": [],
                "missing_capabilities": [],
                "recommendation": rec,
            }
            match = decode_match(data)
            assert match.recommendation == rec

    def test_decode_match_invalid_recommendation(self) -> None:
        """Test decode_match raises on invalid recommendation."""
        data: JSONObject = {
            "hackathon": {
                "id": 1,
                "title": "Test",
                "url": "https://example.com",
                "thumbnail_url": "https://example.com/img.jpg",
                "organization_name": "Org",
                "displayed_location": {"icon": "x", "location": "x"},
                "open_state": "open",
                "time_left_to_submission": "1 day",
                "submission_period_dates": "Jan 1-2",
                "themes": [],
                "prize_amount": "$0",
                "registrations_count": 0,
                "featured": False,
                "winners_announced": False,
                "invite_only": False,
            },
            "match_score": 0.5,
            "matched_capabilities": [],
            "missing_capabilities": [],
            "recommendation": "invalid",
        }
        with pytest.raises(JSONTypeError, match="must be a valid recommendation"):
            decode_match(data)

    def test_decode_match_invalid_hackathon_type(self) -> None:
        """Test decode_match raises when hackathon is not a dict."""
        data: JSONObject = {
            "hackathon": "not a dict",
            "match_score": 0.5,
            "matched_capabilities": [],
            "missing_capabilities": [],
            "recommendation": "good_fit",
        }
        with pytest.raises(JSONTypeError, match="hackathon must be an object"):
            decode_match(data)

    def test_decode_match_invalid_capabilities_type(self) -> None:
        """Test decode_match raises when capabilities contain non-strings."""
        data: JSONObject = {
            "hackathon": {
                "id": 1,
                "title": "Test",
                "url": "https://example.com",
                "thumbnail_url": "https://example.com/img.jpg",
                "organization_name": "Org",
                "displayed_location": {"icon": "x", "location": "x"},
                "open_state": "open",
                "time_left_to_submission": "1 day",
                "submission_period_dates": "Jan 1-2",
                "themes": [],
                "prize_amount": "$0",
                "registrations_count": 0,
                "featured": False,
                "winners_announced": False,
                "invite_only": False,
            },
            "match_score": 0.5,
            "matched_capabilities": [123],
            "missing_capabilities": [],
            "recommendation": "good_fit",
        }
        with pytest.raises(JSONTypeError, match="must be a string"):
            decode_match(data)


class TestInterestFilter:
    """Tests for InterestFilter type and encode/decode."""

    def test_filter_creation(self) -> None:
        """Test creating an InterestFilter instance."""
        f = InterestFilter(
            include_themes=("AI", "ML"),
            exclude_themes=("Blockchain",),
            states=("open", "upcoming"),
            featured_only=True,
        )
        assert f.include_themes == ("AI", "ML")
        assert f.featured_only is True

    def test_encode_filter(self) -> None:
        """Test encoding InterestFilter to dict."""
        f = InterestFilter(
            include_themes=("Web",),
            exclude_themes=(),
            states=None,
            featured_only=False,
        )
        result = encode_filter(f)
        assert result["include_themes"] == ["Web"]
        assert result["states"] is None

    def test_decode_filter(self) -> None:
        """Test decoding InterestFilter from dict."""
        data: JSONObject = {
            "include_themes": ["Gaming", "VR"],
            "exclude_themes": ["Finance"],
            "states": ["open"],
            "featured_only": True,
        }
        f = decode_filter(data)
        assert f.include_themes == ("Gaming", "VR")
        assert f.states == ("open",)
        assert f.featured_only is True

    def test_decode_filter_null_states(self) -> None:
        """Test decoding InterestFilter with null states."""
        data: JSONObject = {
            "include_themes": [],
            "exclude_themes": [],
            "states": None,
            "featured_only": False,
        }
        f = decode_filter(data)
        assert f.states is None

    def test_decode_filter_all_states(self) -> None:
        """Test decoding InterestFilter with all valid states."""
        data: JSONObject = {
            "include_themes": [],
            "exclude_themes": [],
            "states": ["open", "upcoming", "ended", "submissions"],
            "featured_only": False,
        }
        f = decode_filter(data)
        assert f.states == ("open", "upcoming", "ended", "submissions")

    def test_decode_filter_invalid_state(self) -> None:
        """Test decode_filter raises on invalid state."""
        data: JSONObject = {
            "include_themes": [],
            "exclude_themes": [],
            "states": ["invalid_state"],
            "featured_only": False,
        }
        with pytest.raises(JSONTypeError, match="must be a valid state"):
            decode_filter(data)

    def test_decode_filter_invalid_states_type(self) -> None:
        """Test decode_filter raises when states is not an array."""
        data: JSONObject = {
            "include_themes": [],
            "exclude_themes": [],
            "states": "not an array",
            "featured_only": False,
        }
        with pytest.raises(JSONTypeError, match="must be an array"):
            decode_filter(data)

    def test_decode_filter_invalid_state_value_type(self) -> None:
        """Test decode_filter raises when state value is not a string."""
        data: JSONObject = {
            "include_themes": [],
            "exclude_themes": [],
            "states": [123],
            "featured_only": False,
        }
        with pytest.raises(JSONTypeError, match="must be a string"):
            decode_filter(data)
