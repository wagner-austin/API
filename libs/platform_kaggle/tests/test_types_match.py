"""Tests for types: CompetitionMatch."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_kaggle.types import (
    Competition,
    CompetitionMatch,
    InterestFilter,
    decode_filter,
    decode_match,
    encode_filter,
    encode_match,
)
from tests._types_equality import _filters_equal, _matches_equal


class TestCompetitionMatch:
    """Tests for CompetitionMatch type and encode/decode."""

    def test_match_creation(self) -> None:
        """Test creating a CompetitionMatch instance."""
        comp = Competition(
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
        match = CompetitionMatch(
            competition=comp,
            match_score=0.75,
            matched_capabilities=("xgboost_tabular",),
            missing_capabilities=("pytorch_deep_learning",),
            recommendation="good_fit",
        )
        assert match.competition == comp
        assert match.match_score == 0.75
        assert match.matched_capabilities == ("xgboost_tabular",)
        assert match.missing_capabilities == ("pytorch_deep_learning",)
        assert match.recommendation == "good_fit"

    def test_match_equality(self) -> None:
        """Test CompetitionMatch equality comparison."""
        comp = Competition(
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
        match1 = CompetitionMatch(
            competition=comp,
            match_score=0.75,
            matched_capabilities=(),
            missing_capabilities=(),
            recommendation="good_fit",
        )
        match2 = CompetitionMatch(
            competition=comp,
            match_score=0.75,
            matched_capabilities=(),
            missing_capabilities=(),
            recommendation="good_fit",
        )
        match3 = CompetitionMatch(
            competition=comp,
            match_score=0.5,
            matched_capabilities=(),
            missing_capabilities=(),
            recommendation="stretch",
        )
        assert _matches_equal(match1, match2)
        assert not _matches_equal(match1, match3)

    def test_encode_decode_match_roundtrip(self) -> None:
        """Test CompetitionMatch encode/decode roundtrip."""
        comp = Competition(
            ref="test",
            title="Test",
            category="Featured",
            reward="$100,000",
            deadline="2025-08-15",
            team_count=5000,
            tags=("tabular", "finance"),
            description="Test",
            url="https://example.com",
        )
        original = CompetitionMatch(
            competition=comp,
            match_score=0.85,
            matched_capabilities=("xgboost_tabular", "lightgbm_tabular"),
            missing_capabilities=("pytorch_deep_learning",),
            recommendation="strong_fit",
        )
        encoded = encode_match(original)
        decoded = decode_match(encoded)
        assert _matches_equal(decoded, original)

    def test_decode_match_all_recommendations(self) -> None:
        """Test decode_match handles all valid recommendations."""
        recommendations = ["strong_fit", "good_fit", "stretch", "new_territory"]
        for rec in recommendations:
            data: JSONObject = {
                "competition": {
                    "ref": "test",
                    "title": "Test",
                    "category": "Playground",
                    "reward": "Knowledge",
                    "deadline": "2025-12-31",
                    "team_count": 100,
                    "tags": ["tabular"],
                    "description": "Test",
                    "url": "https://example.com",
                },
                "match_score": 0.5,
                "matched_capabilities": [],
                "missing_capabilities": [],
                "recommendation": rec,
            }
            decoded = decode_match(data)
            assert decoded.recommendation == rec

    def test_decode_match_invalid_recommendation(self) -> None:
        """Test decode_match raises on invalid recommendation."""
        data: JSONObject = {
            "competition": {
                "ref": "test",
                "title": "Test",
                "category": "Playground",
                "reward": "Knowledge",
                "deadline": "2025-12-31",
                "team_count": 100,
                "tags": ["tabular"],
                "description": "Test",
                "url": "https://example.com",
            },
            "match_score": 0.5,
            "matched_capabilities": [],
            "missing_capabilities": [],
            "recommendation": "invalid",
        }
        with pytest.raises(JSONTypeError, match="must be a valid recommendation"):
            decode_match(data)

    def test_decode_match_invalid_competition_type(self) -> None:
        """Test decode_match raises when competition is not a dict."""
        data: JSONObject = {
            "competition": "not a dict",
            "match_score": 0.5,
            "matched_capabilities": [],
            "missing_capabilities": [],
            "recommendation": "good_fit",
        }
        with pytest.raises(JSONTypeError, match="competition must be an object"):
            decode_match(data)


class TestInterestFilter:
    """Tests for InterestFilter type and encode/decode."""

    def test_filter_creation(self) -> None:
        """Test creating an InterestFilter instance."""
        filter_ = InterestFilter(
            include_tags=("tabular", "nlp"),
            exclude_tags=("computer-vision",),
            min_reward=1000,
            categories=("Featured", "Research"),
        )
        assert filter_.include_tags == ("tabular", "nlp")
        assert filter_.exclude_tags == ("computer-vision",)
        assert filter_.min_reward == 1000
        assert filter_.categories == ("Featured", "Research")

    def test_filter_equality(self) -> None:
        """Test InterestFilter equality comparison."""
        filter1 = InterestFilter(
            include_tags=("tabular",),
            exclude_tags=(),
            min_reward=None,
            categories=None,
        )
        filter2 = InterestFilter(
            include_tags=("tabular",),
            exclude_tags=(),
            min_reward=None,
            categories=None,
        )
        filter3 = InterestFilter(
            include_tags=("nlp",),
            exclude_tags=(),
            min_reward=None,
            categories=None,
        )
        assert _filters_equal(filter1, filter2)
        assert not _filters_equal(filter1, filter3)

    def test_encode_decode_filter_roundtrip(self) -> None:
        """Test InterestFilter encode/decode roundtrip."""
        original = InterestFilter(
            include_tags=("tabular", "classification"),
            exclude_tags=("computer-vision", "image"),
            min_reward=50000,
            categories=("Featured", "Research", "Playground"),
        )
        encoded = encode_filter(original)
        decoded = decode_filter(encoded)
        assert _filters_equal(decoded, original)

    def test_encode_decode_filter_none_categories(self) -> None:
        """Test InterestFilter encode/decode with None categories."""
        original = InterestFilter(
            include_tags=("tabular",),
            exclude_tags=(),
            min_reward=None,
            categories=None,
        )
        encoded = encode_filter(original)
        decoded = decode_filter(encoded)
        assert _filters_equal(decoded, original)
        assert decoded.categories is None

    def test_decode_filter_all_categories(self) -> None:
        """Test decode_filter handles all valid categories."""
        data: JSONObject = {
            "include_tags": ["tabular"],
            "exclude_tags": [],
            "min_reward": None,
            "categories": [
                "Featured",
                "Research",
                "Playground",
                "Getting Started",
                "Masters",
                "Kudos",
            ],
        }
        decoded = decode_filter(data)
        if decoded.categories is None:
            raise AssertionError("Expected categories to be set")
        assert decoded.categories == (
            "Featured",
            "Research",
            "Playground",
            "Getting Started",
            "Masters",
            "Kudos",
        )

    def test_decode_filter_invalid_category(self) -> None:
        """Test decode_filter raises on invalid category."""
        data: JSONObject = {
            "include_tags": ["tabular"],
            "exclude_tags": [],
            "min_reward": None,
            "categories": ["Invalid"],
        }
        with pytest.raises(JSONTypeError, match="must be a valid category"):
            decode_filter(data)

    def test_decode_filter_categories_not_list(self) -> None:
        """Test decode_filter raises when categories is not a list."""
        data: JSONObject = {
            "include_tags": ["tabular"],
            "exclude_tags": [],
            "min_reward": None,
            "categories": "Featured",
        }
        with pytest.raises(JSONTypeError, match="must be an array"):
            decode_filter(data)

    def test_decode_filter_category_item_not_string(self) -> None:
        """Test decode_filter raises when category item is not a string."""
        data: JSONObject = {
            "include_tags": ["tabular"],
            "exclude_tags": [],
            "min_reward": None,
            "categories": [123],
        }
        with pytest.raises(JSONTypeError, match="must be a string"):
            decode_filter(data)
