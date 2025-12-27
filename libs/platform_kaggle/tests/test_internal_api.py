"""Tests for platform_kaggle.internal_api module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from platform_kaggle.internal_api import (
    KagglePageFetcher,
    KaggleSession,
    _parse_page_from_response,
    _parse_pages_response,
    _require_dict_value,
    create_page_fetcher,
)
from platform_kaggle.testing import (
    FakeKagglePageFetcher,
    make_fake_competition_page,
    make_fake_competition_pages,
)
from platform_kaggle.types import CompetitionPage


class TestRequireDictValue:
    """Tests for _require_dict_value helper."""

    def test_valid_dict(self) -> None:
        """Test with a valid dict."""
        data: JSONValue = {"key": "value"}
        result = _require_dict_value(data, "test")
        assert result == {"key": "value"}

    def test_invalid_string(self) -> None:
        """Test raises on string input."""
        value: JSONValue = "not a dict"
        with pytest.raises(JSONTypeError, match="test must be an object, got str"):
            _require_dict_value(value, "test")

    def test_invalid_list(self) -> None:
        """Test raises on list input."""
        value: JSONValue = []
        with pytest.raises(JSONTypeError, match="context must be an object, got list"):
            _require_dict_value(value, "context")

    def test_invalid_int(self) -> None:
        """Test raises on int input."""
        value: JSONValue = 123
        with pytest.raises(JSONTypeError, match="field must be an object, got int"):
            _require_dict_value(value, "field")


class TestParsePageFromResponse:
    """Tests for _parse_page_from_response helper."""

    def test_valid_page(self) -> None:
        """Test parsing a valid page."""
        data: JSONObject = {
            "id": 42,
            "name": "Description",
            "content": "# Test Content",
        }
        page = _parse_page_from_response(data, 0)
        assert page.id == 42
        assert page.name == "Description"
        assert page.content == "# Test Content"

    def test_missing_id(self) -> None:
        """Test raises on missing id."""
        data: JSONObject = {
            "name": "Description",
            "content": "Content",
        }
        with pytest.raises(JSONTypeError):
            _parse_page_from_response(data, 0)

    def test_missing_name(self) -> None:
        """Test raises on missing name."""
        data: JSONObject = {
            "id": 1,
            "content": "Content",
        }
        with pytest.raises(JSONTypeError):
            _parse_page_from_response(data, 0)

    def test_missing_content(self) -> None:
        """Test raises on missing content."""
        data: JSONObject = {
            "id": 1,
            "name": "Description",
        }
        with pytest.raises(JSONTypeError):
            _parse_page_from_response(data, 0)


class TestParsePagesResponse:
    """Tests for _parse_pages_response helper."""

    def test_valid_response(self) -> None:
        """Test parsing a valid response with multiple pages."""
        data: JSONObject = {
            "pages": [
                {"id": 1, "name": "Description", "content": "Overview"},
                {"id": 2, "name": "Evaluation", "content": "Metrics"},
                {"id": 3, "name": "Timeline", "content": "Dates"},
                {"id": 4, "name": "Rules", "content": "Guidelines"},
            ],
        }
        result = _parse_pages_response(data, 12345)
        assert result.competition_id == 12345
        assert len(result.pages) == 4
        assert result.description == "Overview"
        assert result.evaluation == "Metrics"
        assert result.timeline == "Dates"
        assert result.rules == "Guidelines"

    def test_overview_page_name(self) -> None:
        """Test parsing with 'Overview' page name as description."""
        data: JSONObject = {
            "pages": [
                {"id": 1, "name": "Overview", "content": "Comp overview"},
            ],
        }
        result = _parse_pages_response(data, 123)
        assert result.description == "Comp overview"

    def test_empty_pages(self) -> None:
        """Test parsing response with no pages."""
        data: JSONObject = {"pages": []}
        result = _parse_pages_response(data, 999)
        assert result.competition_id == 999
        assert result.pages == ()
        assert result.description == ""
        assert result.evaluation == ""
        assert result.timeline == ""
        assert result.rules == ""

    def test_missing_pages_field(self) -> None:
        """Test raises on missing pages field."""
        data: JSONObject = {}
        with pytest.raises(JSONTypeError):
            _parse_pages_response(data, 123)

    def test_invalid_page_type(self) -> None:
        """Test raises on invalid page type in array."""
        data: JSONObject = {"pages": ["not a dict"]}
        with pytest.raises(JSONTypeError, match="must be an object"):
            _parse_pages_response(data, 123)


class TestFakeKagglePageFetcher:
    """Tests for FakeKagglePageFetcher."""

    def test_fetch_pages_configured(self) -> None:
        """Test fetching pages for a configured competition."""
        pages = make_fake_competition_pages(
            competition_id=12345,
            description="Test desc",
            evaluation="Test eval",
        )
        fetcher = FakeKagglePageFetcher(pages={12345: pages})

        result = fetcher.fetch_pages(12345)

        assert result.competition_id == 12345
        assert result.description == "Test desc"
        assert result.evaluation == "Test eval"
        assert 12345 in fetcher._fetch_calls

    def test_fetch_pages_not_configured(self) -> None:
        """Test fetching pages for unconfigured competition raises."""
        fetcher = FakeKagglePageFetcher()

        with pytest.raises(RuntimeError, match="Competition 99999 not found"):
            fetcher.fetch_pages(99999)

        assert 99999 in fetcher._fetch_calls

    def test_get_competition_id_configured(self) -> None:
        """Test getting competition ID for configured slug."""
        fetcher = FakeKagglePageFetcher(competition_ids={"test-comp": 12345})

        result = fetcher.get_competition_id("test-comp")

        assert result == 12345
        assert "test-comp" in fetcher._id_calls

    def test_get_competition_id_not_configured(self) -> None:
        """Test getting competition ID for unconfigured slug raises RuntimeError."""
        fetcher = FakeKagglePageFetcher()

        with pytest.raises(RuntimeError, match="Competition 'unknown-comp' not found"):
            fetcher.get_competition_id("unknown-comp")

        assert "unknown-comp" in fetcher._id_calls

    def test_default_empty_state(self) -> None:
        """Test default initialization has empty state."""
        fetcher = FakeKagglePageFetcher()

        assert fetcher._pages == {}
        assert fetcher._competition_ids == {}
        assert fetcher._fetch_calls == []
        assert fetcher._id_calls == []


class TestMakeFakeCompetitionPage:
    """Tests for make_fake_competition_page factory."""

    def test_default_values(self) -> None:
        """Test factory creates page with default values."""
        page = make_fake_competition_page()
        assert page.id == 1
        assert page.name == "Description"
        assert page.content == "Test content"

    def test_custom_values(self) -> None:
        """Test factory creates page with custom values."""
        page = make_fake_competition_page(
            id=42,
            name="Evaluation",
            content="Custom content",
        )
        assert page.id == 42
        assert page.name == "Evaluation"
        assert page.content == "Custom content"


class TestMakeFakeCompetitionPages:
    """Tests for make_fake_competition_pages factory."""

    def test_default_values(self) -> None:
        """Test factory creates pages with default values."""
        pages = make_fake_competition_pages()
        assert pages.competition_id == 12345
        assert len(pages.pages) == 4
        assert pages.description == "Test description"
        assert pages.evaluation == "Test evaluation"
        assert pages.timeline == "Test timeline"
        assert pages.rules == "Test rules"

    def test_custom_values(self) -> None:
        """Test factory creates pages with custom values."""
        pages = make_fake_competition_pages(
            competition_id=99999,
            description="Custom desc",
            evaluation="Custom eval",
            timeline="Custom time",
            rules="Custom rules",
        )
        assert pages.competition_id == 99999
        assert pages.description == "Custom desc"
        assert pages.evaluation == "Custom eval"
        assert pages.timeline == "Custom time"
        assert pages.rules == "Custom rules"

    def test_custom_pages_tuple(self) -> None:
        """Test factory uses custom pages tuple when provided."""
        custom_page = CompetitionPage(id=100, name="Custom", content="Custom")
        pages = make_fake_competition_pages(
            pages=(custom_page,),
            description="Desc",
            evaluation="Eval",
            timeline="Time",
            rules="Rules",
        )
        assert len(pages.pages) == 1
        assert pages.pages[0].id == 100
        assert pages.pages[0].name == "Custom"


# -----------------------------------------------------------------------------
# Integration Tests (Real HTTP)
# -----------------------------------------------------------------------------


class TestKaggleSession:
    """Integration tests for KaggleSession."""

    def test_initialize_gets_xsrf_token(self) -> None:
        """Test initialize obtains XSRF token from Kaggle."""
        session = KaggleSession()
        session.initialize()
        # XSRF token should be set after initialization
        token = session.xsrf_token
        # Token should be a non-empty base64-like string (typically starts with CfDJ)
        assert token.startswith("CfDJ")

    def test_xsrf_token_before_initialize_raises(self) -> None:
        """Test accessing xsrf_token before initialize raises RuntimeError."""
        session = KaggleSession()
        with pytest.raises(RuntimeError, match="Session not initialized"):
            _ = session.xsrf_token

    def test_request_before_initialize_raises(self) -> None:
        """Test making request before initialize raises RuntimeError."""
        session = KaggleSession()
        with pytest.raises(RuntimeError, match="Session not initialized"):
            session.request("https://www.kaggle.com")

    def test_request_returns_bytes(self) -> None:
        """Test request returns response as bytes."""
        session = KaggleSession()
        session.initialize()
        # Make a simple GET request
        response = session.request("https://www.kaggle.com")
        # Response should contain HTML content
        assert b"<!doctype html>" in response.lower() or b"<html" in response.lower()

    def test_extract_xsrf_token_returns_none_for_empty_jar(self) -> None:
        """Test _extract_xsrf_token returns None when no cookies."""
        session = KaggleSession()
        # Cookie jar is empty initially
        result = session._extract_xsrf_token()
        assert result is None

    def test_initialize_raises_when_xsrf_token_missing(self) -> None:
        """Test initialize raises RuntimeError when XSRF token extraction fails."""

        class SessionWithNoToken(KaggleSession):
            """Test subclass that simulates missing XSRF token."""

            def _extract_xsrf_token(self) -> str | None:
                # Always return None to simulate missing token
                return None

        session = SessionWithNoToken()
        with pytest.raises(RuntimeError, match="Failed to obtain XSRF token"):
            session.initialize()


class TestKagglePageFetcher:
    """Integration tests for KagglePageFetcher."""

    def test_fetch_pages_returns_competition_pages(self) -> None:
        """Test fetch_pages returns CompetitionPages for valid ID."""
        session = KaggleSession()
        session.initialize()
        fetcher = KagglePageFetcher(session)

        # Use a known competition ID (Titanic is 3136)
        pages = fetcher.fetch_pages(3136)

        assert pages.competition_id == 3136
        # Titanic should have description with relevant keywords
        desc_lower = pages.description.lower()
        assert "titanic" in desc_lower or "passenger" in desc_lower or "survival" in desc_lower

    def test_get_competition_id_returns_id(self) -> None:
        """Test get_competition_id returns numeric ID for known slug."""
        session = KaggleSession()
        session.initialize()
        fetcher = KagglePageFetcher(session)

        # Use a well-known competition slug
        comp_id = fetcher.get_competition_id("titanic")

        # Titanic competition ID is 3136
        assert comp_id == 3136

    def test_get_competition_id_handles_404(self) -> None:
        """Test get_competition_id handles 404 for invalid slug."""
        from urllib.error import HTTPError

        session = KaggleSession()
        session.initialize()
        fetcher = KagglePageFetcher(session)

        # Invalid slug should raise 404
        with pytest.raises(HTTPError) as exc_info:
            fetcher.get_competition_id("definitely-not-a-real-competition-xyz123")

        assert exc_info.value.code == 404


class TestCreatePageFetcher:
    """Integration tests for create_page_fetcher factory."""

    def test_create_page_fetcher_returns_initialized_fetcher(self) -> None:
        """Test create_page_fetcher returns ready-to-use fetcher."""
        fetcher = create_page_fetcher()

        # Should be able to fetch pages immediately
        pages = fetcher.fetch_pages(3136)  # Titanic
        assert pages.competition_id == 3136
        # Verify we got actual content
        desc_lower = pages.description.lower()
        assert "titanic" in desc_lower or "passenger" in desc_lower or "survival" in desc_lower
