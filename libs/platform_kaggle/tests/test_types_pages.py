"""Tests for types: CompetitionPage."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_kaggle.types import (
    CompetitionPage,
    CompetitionPages,
    decode_competition_page,
    decode_competition_pages,
    encode_competition_page,
    encode_competition_pages,
)
from tests._types_equality import _competition_pages_equal, _pages_equal


class TestCompetitionPage:
    """Tests for CompetitionPage type and encode/decode."""

    def test_page_creation(self) -> None:
        """Test creating a CompetitionPage instance."""
        page = CompetitionPage(
            id=1,
            name="Description",
            content="# Competition Overview\n\nThis is a test.",
        )
        assert page.id == 1
        assert page.name == "Description"
        assert page.content == "# Competition Overview\n\nThis is a test."

    def test_page_equality(self) -> None:
        """Test CompetitionPage equality comparison."""
        page1 = CompetitionPage(id=1, name="Description", content="Test")
        page2 = CompetitionPage(id=1, name="Description", content="Test")
        page3 = CompetitionPage(id=2, name="Evaluation", content="Test")
        assert _pages_equal(page1, page2)
        assert not _pages_equal(page1, page3)

    def test_encode_decode_page_roundtrip(self) -> None:
        """Test CompetitionPage encode/decode roundtrip."""
        original = CompetitionPage(
            id=42,
            name="Evaluation",
            content="# Evaluation Criteria\n\n- Accuracy\n- F1 Score",
        )
        encoded = encode_competition_page(original)
        decoded = decode_competition_page(encoded)
        assert _pages_equal(decoded, original)

    def test_decode_page_missing_field(self) -> None:
        """Test decode_competition_page raises on missing field."""
        data: JSONObject = {
            "id": 1,
            "name": "Description",
        }
        with pytest.raises(JSONTypeError):
            decode_competition_page(data)

    def test_decode_page_invalid_id(self) -> None:
        """Test decode_competition_page raises on invalid id type."""
        data: JSONObject = {
            "id": "not-an-int",
            "name": "Description",
            "content": "Test",
        }
        with pytest.raises(JSONTypeError, match="must be an integer"):
            decode_competition_page(data)


class TestCompetitionPages:
    """Tests for CompetitionPages type and encode/decode."""

    def test_pages_creation(self) -> None:
        """Test creating a CompetitionPages instance."""
        page1 = CompetitionPage(id=1, name="Description", content="Desc content")
        page2 = CompetitionPage(id=2, name="Evaluation", content="Eval content")
        pages = CompetitionPages(
            competition_id=12345,
            pages=(page1, page2),
            description="Desc content",
            evaluation="Eval content",
            timeline="",
            rules="",
        )
        assert pages.competition_id == 12345
        assert len(pages.pages) == 2
        assert pages.description == "Desc content"
        assert pages.evaluation == "Eval content"
        assert pages.timeline == ""
        assert pages.rules == ""

    def test_pages_equality(self) -> None:
        """Test CompetitionPages equality comparison."""
        page = CompetitionPage(id=1, name="Description", content="Test")
        pages1 = CompetitionPages(
            competition_id=123,
            pages=(page,),
            description="Test",
            evaluation="",
            timeline="",
            rules="",
        )
        pages2 = CompetitionPages(
            competition_id=123,
            pages=(page,),
            description="Test",
            evaluation="",
            timeline="",
            rules="",
        )
        pages3 = CompetitionPages(
            competition_id=456,
            pages=(page,),
            description="Test",
            evaluation="",
            timeline="",
            rules="",
        )
        assert _competition_pages_equal(pages1, pages2)
        assert not _competition_pages_equal(pages1, pages3)

    def test_pages_equality_different_page_count(self) -> None:
        """Test CompetitionPages equality with different page counts."""
        page1 = CompetitionPage(id=1, name="Description", content="Test")
        page2 = CompetitionPage(id=2, name="Evaluation", content="Eval")
        pages1 = CompetitionPages(
            competition_id=123,
            pages=(page1,),
            description="Test",
            evaluation="",
            timeline="",
            rules="",
        )
        pages2 = CompetitionPages(
            competition_id=123,
            pages=(page1, page2),
            description="Test",
            evaluation="Eval",
            timeline="",
            rules="",
        )
        assert not _competition_pages_equal(pages1, pages2)

    def test_encode_decode_pages_roundtrip(self) -> None:
        """Test CompetitionPages encode/decode roundtrip."""
        page1 = CompetitionPage(id=1, name="Description", content="Overview")
        page2 = CompetitionPage(id=2, name="Evaluation", content="Metrics")
        page3 = CompetitionPage(id=3, name="Timeline", content="Dates")
        page4 = CompetitionPage(id=4, name="Rules", content="Guidelines")
        original = CompetitionPages(
            competition_id=98765,
            pages=(page1, page2, page3, page4),
            description="Overview",
            evaluation="Metrics",
            timeline="Dates",
            rules="Guidelines",
        )
        encoded = encode_competition_pages(original)
        decoded = decode_competition_pages(encoded)
        assert _competition_pages_equal(decoded, original)

    def test_decode_pages_missing_field(self) -> None:
        """Test decode_competition_pages raises on missing field."""
        data: JSONObject = {
            "competition_id": 123,
            "pages": [],
        }
        with pytest.raises(JSONTypeError):
            decode_competition_pages(data)

    def test_decode_pages_invalid_page_type(self) -> None:
        """Test decode_competition_pages raises on invalid page type."""
        data: JSONObject = {
            "competition_id": 123,
            "pages": ["not a dict"],
            "description": "",
            "evaluation": "",
            "timeline": "",
            "rules": "",
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_competition_pages(data)
