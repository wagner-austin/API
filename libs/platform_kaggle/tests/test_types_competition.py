"""Tests for types: Competition."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_kaggle.types import (
    Competition,
    decode_competition,
    encode_competition,
)
from tests._types_equality import _competitions_equal


class TestCompetition:
    """Tests for Competition type and encode/decode."""

    def test_competition_creation(self) -> None:
        """Test creating a Competition instance."""
        comp = Competition(
            ref="test-comp",
            title="Test Competition",
            category="Playground",
            reward="Knowledge",
            deadline="2025-12-31",
            team_count=100,
            tags=("tabular", "classification"),
            description="Test description",
            url="https://www.kaggle.com/competitions/test-comp",
        )
        assert comp.ref == "test-comp"
        assert comp.title == "Test Competition"
        assert comp.category == "Playground"
        assert comp.reward == "Knowledge"
        assert comp.deadline == "2025-12-31"
        assert comp.team_count == 100
        assert comp.tags == ("tabular", "classification")
        assert comp.description == "Test description"
        assert comp.url == "https://www.kaggle.com/competitions/test-comp"

    def test_competition_equality(self) -> None:
        """Test Competition equality comparison."""
        comp1 = Competition(
            ref="test",
            title="Test",
            category="Playground",
            reward="Knowledge",
            deadline="2025-12-31",
            team_count=100,
            tags=("tabular",),
            description="Test",
            url="https://example.com",
        )
        comp2 = Competition(
            ref="test",
            title="Test",
            category="Playground",
            reward="Knowledge",
            deadline="2025-12-31",
            team_count=100,
            tags=("tabular",),
            description="Test",
            url="https://example.com",
        )
        comp3 = Competition(
            ref="other",
            title="Test",
            category="Playground",
            reward="Knowledge",
            deadline="2025-12-31",
            team_count=100,
            tags=("tabular",),
            description="Test",
            url="https://example.com",
        )
        assert _competitions_equal(comp1, comp2)
        assert not _competitions_equal(comp1, comp3)

    def test_encode_decode_competition_roundtrip(self) -> None:
        """Test Competition encode/decode roundtrip."""
        original = Competition(
            ref="test-comp",
            title="Test Competition",
            category="Featured",
            reward="$100,000",
            deadline="2025-08-15",
            team_count=5000,
            tags=("tabular", "classification", "finance"),
            description="Predict credit default",
            url="https://www.kaggle.com/competitions/test-comp",
        )
        encoded = encode_competition(original)
        decoded = decode_competition(encoded)
        assert _competitions_equal(decoded, original)

    def test_decode_competition_all_categories(self) -> None:
        """Test decode_competition handles all valid categories."""
        categories = ["Featured", "Research", "Playground", "Getting Started", "Masters", "Kudos"]
        for category in categories:
            data: JSONObject = {
                "ref": "test",
                "title": "Test",
                "category": category,
                "reward": "Knowledge",
                "deadline": "2025-12-31",
                "team_count": 100,
                "tags": ["tabular"],
                "description": "Test",
                "url": "https://example.com",
            }
            decoded = decode_competition(data)
            assert decoded.category == category

    def test_decode_competition_invalid_category(self) -> None:
        """Test decode_competition raises on invalid category."""
        data: JSONObject = {
            "ref": "test",
            "title": "Test",
            "category": "Invalid",
            "reward": "Knowledge",
            "deadline": "2025-12-31",
            "team_count": 100,
            "tags": ["tabular"],
            "description": "Test",
            "url": "https://example.com",
        }
        with pytest.raises(JSONTypeError, match="must be a valid category"):
            decode_competition(data)

    def test_decode_competition_missing_field(self) -> None:
        """Test decode_competition raises on missing field."""
        data: JSONObject = {
            "ref": "test",
            "title": "Test",
        }
        with pytest.raises(JSONTypeError):
            decode_competition(data)

    def test_decode_competition_invalid_tags(self) -> None:
        """Test decode_competition raises on invalid tags."""
        data: JSONObject = {
            "ref": "test",
            "title": "Test",
            "category": "Playground",
            "reward": "Knowledge",
            "deadline": "2025-12-31",
            "team_count": 100,
            "tags": [123],
            "description": "Test",
            "url": "https://example.com",
        }
        with pytest.raises(JSONTypeError, match="must be a string"):
            decode_competition(data)
