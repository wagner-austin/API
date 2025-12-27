"""Tests for platform_kaggle.types module."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_kaggle.types import (
    CodebaseCapability,
    CodebaseProfile,
    Competition,
    CompetitionMatch,
    InterestFilter,
    LibInfo,
    ServiceInfo,
    decode_capability,
    decode_competition,
    decode_filter,
    decode_match,
    decode_profile,
    encode_capability,
    encode_competition,
    encode_filter,
    encode_match,
    encode_profile,
)

# -----------------------------------------------------------------------------
# Comparison Helpers (classes use __slots__, so no built-in equality)
# -----------------------------------------------------------------------------


def _competitions_equal(a: Competition, b: Competition) -> bool:
    """Check if two Competition instances are equal."""
    return (
        a.ref == b.ref
        and a.title == b.title
        and a.category == b.category
        and a.reward == b.reward
        and a.deadline == b.deadline
        and a.team_count == b.team_count
        and a.tags == b.tags
        and a.description == b.description
        and a.url == b.url
    )


def _capabilities_equal(a: CodebaseCapability, b: CodebaseCapability) -> bool:
    """Check if two CodebaseCapability instances are equal."""
    return (
        a.name == b.name
        and a.strength == b.strength
        and a.tags == b.tags
        and a.description == b.description
    )


def _profiles_equal(a: CodebaseProfile, b: CodebaseProfile) -> bool:
    """Check if two CodebaseProfile instances are equal."""
    if len(a.capabilities) != len(b.capabilities):
        return False
    for cap_a, cap_b in zip(a.capabilities, b.capabilities, strict=True):
        if not _capabilities_equal(cap_a, cap_b):
            return False
    return (
        a.ml_backends == b.ml_backends
        and a.data_formats == b.data_formats
        and a.task_types == b.task_types
    )


def _matches_equal(a: CompetitionMatch, b: CompetitionMatch) -> bool:
    """Check if two CompetitionMatch instances are equal."""
    return (
        _competitions_equal(a.competition, b.competition)
        and a.match_score == b.match_score
        and a.matched_capabilities == b.matched_capabilities
        and a.missing_capabilities == b.missing_capabilities
        and a.recommendation == b.recommendation
    )


def _filters_equal(a: InterestFilter, b: InterestFilter) -> bool:
    """Check if two InterestFilter instances are equal."""
    return (
        a.include_tags == b.include_tags
        and a.exclude_tags == b.exclude_tags
        and a.min_reward == b.min_reward
        and a.categories == b.categories
    )


def _libinfos_equal(a: LibInfo, b: LibInfo) -> bool:
    """Check if two LibInfo instances are equal."""
    return a.name == b.name and a.path == b.path and a.dependencies == b.dependencies


def _serviceinfos_equal(a: ServiceInfo, b: ServiceInfo) -> bool:
    """Check if two ServiceInfo instances are equal."""
    return (
        a.name == b.name
        and a.path == b.path
        and a.dependencies == b.dependencies
        and a.has_rules_files == b.has_rules_files
    )


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


class TestCodebaseCapability:
    """Tests for CodebaseCapability type and encode/decode."""

    def test_capability_creation(self) -> None:
        """Test creating a CodebaseCapability instance."""
        cap = CodebaseCapability(
            name="xgboost_tabular",
            strength="strong",
            tags=("tabular", "classification"),
            description="XGBoost for tabular data",
        )
        assert cap.name == "xgboost_tabular"
        assert cap.strength == "strong"
        assert cap.tags == ("tabular", "classification")
        assert cap.description == "XGBoost for tabular data"

    def test_capability_equality(self) -> None:
        """Test CodebaseCapability equality comparison."""
        cap1 = CodebaseCapability(
            name="test",
            strength="moderate",
            tags=("test",),
            description="Test",
        )
        cap2 = CodebaseCapability(
            name="test",
            strength="moderate",
            tags=("test",),
            description="Test",
        )
        cap3 = CodebaseCapability(
            name="other",
            strength="moderate",
            tags=("test",),
            description="Test",
        )
        assert _capabilities_equal(cap1, cap2)
        assert not _capabilities_equal(cap1, cap3)

    def test_encode_decode_capability_roundtrip(self) -> None:
        """Test CodebaseCapability encode/decode roundtrip."""
        original = CodebaseCapability(
            name="xgboost_tabular",
            strength="strong",
            tags=("tabular", "classification", "regression"),
            description="XGBoost gradient boosting",
        )
        encoded = encode_capability(original)
        decoded = decode_capability(encoded)
        assert _capabilities_equal(decoded, original)

    def test_decode_capability_all_strengths(self) -> None:
        """Test decode_capability handles all valid strengths."""
        strengths = ["strong", "moderate", "basic"]
        for strength in strengths:
            data: JSONObject = {
                "name": "test",
                "strength": strength,
                "tags": ["test"],
                "description": "Test",
            }
            decoded = decode_capability(data)
            assert decoded.strength == strength

    def test_decode_capability_invalid_strength(self) -> None:
        """Test decode_capability raises on invalid strength."""
        data: JSONObject = {
            "name": "test",
            "strength": "super",
            "tags": ["test"],
            "description": "Test",
        }
        with pytest.raises(JSONTypeError, match="must be strong/moderate/basic"):
            decode_capability(data)


class TestCodebaseProfile:
    """Tests for CodebaseProfile type and encode/decode."""

    def test_profile_creation(self) -> None:
        """Test creating a CodebaseProfile instance."""
        cap = CodebaseCapability(
            name="test",
            strength="moderate",
            tags=("test",),
            description="Test",
        )
        profile = CodebaseProfile(
            capabilities=(cap,),
            ml_backends=("xgboost", "lightgbm"),
            data_formats=("csv", "parquet"),
            task_types=("binary_classification",),
        )
        assert len(profile.capabilities) == 1
        assert profile.ml_backends == ("xgboost", "lightgbm")
        assert profile.data_formats == ("csv", "parquet")
        assert profile.task_types == ("binary_classification",)

    def test_profile_equality(self) -> None:
        """Test CodebaseProfile equality comparison."""
        profile1 = CodebaseProfile(
            capabilities=(),
            ml_backends=("xgboost",),
            data_formats=("csv",),
            task_types=("classification",),
        )
        profile2 = CodebaseProfile(
            capabilities=(),
            ml_backends=("xgboost",),
            data_formats=("csv",),
            task_types=("classification",),
        )
        profile3 = CodebaseProfile(
            capabilities=(),
            ml_backends=("lightgbm",),
            data_formats=("csv",),
            task_types=("classification",),
        )
        assert _profiles_equal(profile1, profile2)
        assert not _profiles_equal(profile1, profile3)

    def test_encode_decode_profile_roundtrip(self) -> None:
        """Test CodebaseProfile encode/decode roundtrip."""
        cap = CodebaseCapability(
            name="xgboost_tabular",
            strength="strong",
            tags=("tabular",),
            description="XGBoost",
        )
        original = CodebaseProfile(
            capabilities=(cap,),
            ml_backends=("xgboost", "pytorch"),
            data_formats=("csv", "parquet"),
            task_types=("binary_classification", "regression"),
        )
        encoded = encode_profile(original)
        decoded = decode_profile(encoded)
        assert _profiles_equal(decoded, original)

    def test_decode_profile_invalid_capability(self) -> None:
        """Test decode_profile raises on invalid capability."""
        data: JSONObject = {
            "capabilities": ["not a dict"],
            "ml_backends": ["xgboost"],
            "data_formats": ["csv"],
            "task_types": ["classification"],
        }
        with pytest.raises(JSONTypeError, match="must be an object"):
            decode_profile(data)


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


class TestLibInfo:
    """Tests for LibInfo type."""

    def test_libinfo_creation(self) -> None:
        """Test creating a LibInfo instance."""
        info = LibInfo(
            name="sample-lib",
            path=Path("/libs/sample_lib"),
            dependencies=("xgboost", "pandas"),
        )
        assert info.name == "sample-lib"
        assert info.path == Path("/libs/sample_lib")
        assert info.dependencies == ("xgboost", "pandas")

    def test_libinfo_equality(self) -> None:
        """Test LibInfo equality comparison."""
        info1 = LibInfo(
            name="test",
            path=Path("/test"),
            dependencies=("dep1",),
        )
        info2 = LibInfo(
            name="test",
            path=Path("/test"),
            dependencies=("dep1",),
        )
        info3 = LibInfo(
            name="other",
            path=Path("/test"),
            dependencies=("dep1",),
        )
        assert _libinfos_equal(info1, info2)
        assert not _libinfos_equal(info1, info3)


class TestServiceInfo:
    """Tests for ServiceInfo type."""

    def test_serviceinfo_creation(self) -> None:
        """Test creating a ServiceInfo instance."""
        info = ServiceInfo(
            name="sample-service",
            path=Path("/services/sample_service"),
            dependencies=("openai", "fastapi"),
            has_rules_files=True,
        )
        assert info.name == "sample-service"
        assert info.path == Path("/services/sample_service")
        assert info.dependencies == ("openai", "fastapi")
        assert info.has_rules_files is True

    def test_serviceinfo_equality(self) -> None:
        """Test ServiceInfo equality comparison."""
        info1 = ServiceInfo(
            name="test",
            path=Path("/test"),
            dependencies=("dep1",),
            has_rules_files=False,
        )
        info2 = ServiceInfo(
            name="test",
            path=Path("/test"),
            dependencies=("dep1",),
            has_rules_files=False,
        )
        info3 = ServiceInfo(
            name="other",
            path=Path("/test"),
            dependencies=("dep1",),
            has_rules_files=False,
        )
        assert _serviceinfos_equal(info1, info2)
        assert not _serviceinfos_equal(info1, info3)
