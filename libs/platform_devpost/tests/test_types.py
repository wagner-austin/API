"""Tests for platform_devpost.types module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_devpost.types import (
    DisplayedLocation,
    Hackathon,
    HackathonListMeta,
    HackathonListResponse,
    HackathonMatch,
    InterestFilter,
    Theme,
    decode_displayed_location,
    decode_filter,
    decode_hackathon,
    decode_list_meta,
    decode_list_response,
    decode_match,
    decode_theme,
    encode_displayed_location,
    encode_filter,
    encode_hackathon,
    encode_list_meta,
    encode_list_response,
    encode_match,
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


class TestHackathonListMeta:
    """Tests for HackathonListMeta type and encode/decode."""

    def test_meta_creation(self) -> None:
        """Test creating a HackathonListMeta instance."""
        meta = HackathonListMeta(total_count=100, per_page=10)
        assert meta.total_count == 100
        assert meta.per_page == 10

    def test_encode_list_meta(self) -> None:
        """Test encoding HackathonListMeta to dict."""
        meta = HackathonListMeta(total_count=50, per_page=20)
        result = encode_list_meta(meta)
        assert result == {"total_count": 50, "per_page": 20}

    def test_decode_list_meta(self) -> None:
        """Test decoding HackathonListMeta from dict."""
        data: JSONObject = {"total_count": 200, "per_page": 25}
        meta = decode_list_meta(data)
        assert meta.total_count == 200
        assert meta.per_page == 25


class TestHackathonListResponse:
    """Tests for HackathonListResponse type and encode/decode."""

    def test_response_creation(self) -> None:
        """Test creating a HackathonListResponse instance."""
        resp = HackathonListResponse(
            hackathons=(),
            meta=HackathonListMeta(total_count=0, per_page=10),
        )
        assert len(resp.hackathons) == 0
        assert resp.meta.total_count == 0

    def test_encode_list_response(self) -> None:
        """Test encoding HackathonListResponse to dict."""
        resp = HackathonListResponse(
            hackathons=(),
            meta=HackathonListMeta(total_count=0, per_page=10),
        )
        result = encode_list_response(resp)
        assert result["hackathons"] == []
        assert result["meta"] == {"total_count": 0, "per_page": 10}

    def test_decode_list_response(self) -> None:
        """Test decoding HackathonListResponse from dict."""
        data: JSONObject = {
            "hackathons": [
                {
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
                }
            ],
            "meta": {"total_count": 1, "per_page": 10},
        }
        resp = decode_list_response(data)
        assert len(resp.hackathons) == 1
        assert resp.meta.total_count == 1

    def test_decode_list_response_invalid_hackathon_type(self) -> None:
        """Test decode_list_response raises when hackathon is not a dict."""
        data: JSONObject = {
            "hackathons": ["not a dict"],
            "meta": {"total_count": 1, "per_page": 10},
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_list_response(data)

    def test_decode_list_response_invalid_meta_type(self) -> None:
        """Test decode_list_response raises when meta is not a dict."""
        data: JSONObject = {
            "hackathons": [],
            "meta": "not a dict",
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_list_response(data)
