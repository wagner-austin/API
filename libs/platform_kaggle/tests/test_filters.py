"""Tests for platform_kaggle.filters module."""

from __future__ import annotations

from platform_kaggle.filters import (
    _has_any_tag,
    _has_excluded_tag,
    _normalize_tag,
    _parse_reward_amount,
    _passes_filter,
    filter_competitions,
    make_interest_filter,
)
from platform_kaggle.testing import make_fake_competition
from platform_kaggle.types import InterestFilter


class TestParseRewardAmount:
    """Tests for _parse_reward_amount function."""

    def test_dollar_amount(self) -> None:
        """Test parsing dollar amount."""
        assert _parse_reward_amount("$100,000") == 100000

    def test_dollar_amount_no_comma(self) -> None:
        """Test parsing dollar amount without comma."""
        assert _parse_reward_amount("$50000") == 50000

    def test_euro_amount(self) -> None:
        """Test parsing euro amount."""
        assert _parse_reward_amount("€25,000") == 25000

    def test_knowledge_reward(self) -> None:
        """Test knowledge reward returns None."""
        assert _parse_reward_amount("Knowledge") is None

    def test_kudos_reward(self) -> None:
        """Test kudos reward returns None."""
        assert _parse_reward_amount("Kudos") is None

    def test_swag_reward(self) -> None:
        """Test swag reward returns None."""
        assert _parse_reward_amount("Swag") is None

    def test_medals_reward(self) -> None:
        """Test medals reward returns None."""
        assert _parse_reward_amount("Medals") is None

    def test_no_number(self) -> None:
        """Test reward with no number returns None."""
        assert _parse_reward_amount("No Prize") is None


class TestNormalizeTag:
    """Tests for _normalize_tag function."""

    def test_lowercase(self) -> None:
        """Test tag is lowercased."""
        assert _normalize_tag("TABULAR") == "tabular"

    def test_underscore_to_hyphen(self) -> None:
        """Test underscores are converted to hyphens."""
        assert _normalize_tag("binary_classification") == "binary-classification"

    def test_strip_whitespace(self) -> None:
        """Test whitespace is stripped."""
        assert _normalize_tag("  tabular  ") == "tabular"


class TestHasAnyTag:
    """Tests for _has_any_tag function."""

    def test_has_matching_tag(self) -> None:
        """Test competition has matching tag."""
        comp = make_fake_competition(tags=("tabular", "classification"))
        assert _has_any_tag(comp, ("tabular",)) is True

    def test_no_matching_tag(self) -> None:
        """Test competition has no matching tag."""
        comp = make_fake_competition(tags=("nlp", "text"))
        assert _has_any_tag(comp, ("tabular",)) is False

    def test_empty_filter_tags(self) -> None:
        """Test empty filter tags returns True."""
        comp = make_fake_competition(tags=("tabular",))
        assert _has_any_tag(comp, ()) is True

    def test_normalized_matching(self) -> None:
        """Test matching with normalization."""
        comp = make_fake_competition(tags=("binary_classification",))
        assert _has_any_tag(comp, ("binary-classification",)) is True


class TestHasExcludedTag:
    """Tests for _has_excluded_tag function."""

    def test_has_excluded_tag(self) -> None:
        """Test competition has excluded tag."""
        comp = make_fake_competition(tags=("computer-vision", "image"))
        assert _has_excluded_tag(comp, ("computer-vision",)) is True

    def test_no_excluded_tag(self) -> None:
        """Test competition has no excluded tag."""
        comp = make_fake_competition(tags=("tabular",))
        assert _has_excluded_tag(comp, ("computer-vision",)) is False

    def test_empty_exclude_tags(self) -> None:
        """Test empty exclude tags returns False."""
        comp = make_fake_competition(tags=("computer-vision",))
        assert _has_excluded_tag(comp, ()) is False


class TestPassesFilter:
    """Tests for _passes_filter function."""

    def test_passes_include_tags(self) -> None:
        """Test competition passes include tags filter."""
        comp = make_fake_competition(tags=("tabular",))
        filter_ = InterestFilter(
            include_tags=("tabular",),
            exclude_tags=(),
            min_reward=None,
            categories=None,
        )
        assert _passes_filter(comp, filter_) is True

    def test_fails_include_tags(self) -> None:
        """Test competition fails include tags filter."""
        comp = make_fake_competition(tags=("nlp",))
        filter_ = InterestFilter(
            include_tags=("tabular",),
            exclude_tags=(),
            min_reward=None,
            categories=None,
        )
        assert _passes_filter(comp, filter_) is False

    def test_fails_exclude_tags(self) -> None:
        """Test competition fails exclude tags filter."""
        comp = make_fake_competition(tags=("computer-vision",))
        filter_ = InterestFilter(
            include_tags=(),
            exclude_tags=("computer-vision",),
            min_reward=None,
            categories=None,
        )
        assert _passes_filter(comp, filter_) is False

    def test_passes_min_reward(self) -> None:
        """Test competition passes min reward filter."""
        comp = make_fake_competition(reward="$100,000")
        filter_ = InterestFilter(
            include_tags=(),
            exclude_tags=(),
            min_reward=50000,
            categories=None,
        )
        assert _passes_filter(comp, filter_) is True

    def test_fails_min_reward_too_low(self) -> None:
        """Test competition fails min reward filter (too low)."""
        comp = make_fake_competition(reward="$10,000")
        filter_ = InterestFilter(
            include_tags=(),
            exclude_tags=(),
            min_reward=50000,
            categories=None,
        )
        assert _passes_filter(comp, filter_) is False

    def test_fails_min_reward_knowledge(self) -> None:
        """Test competition fails min reward filter (knowledge)."""
        comp = make_fake_competition(reward="Knowledge")
        filter_ = InterestFilter(
            include_tags=(),
            exclude_tags=(),
            min_reward=1,
            categories=None,
        )
        assert _passes_filter(comp, filter_) is False

    def test_passes_categories(self) -> None:
        """Test competition passes categories filter."""
        comp = make_fake_competition(category="Featured")
        filter_ = InterestFilter(
            include_tags=(),
            exclude_tags=(),
            min_reward=None,
            categories=("Featured", "Research"),
        )
        assert _passes_filter(comp, filter_) is True

    def test_fails_categories(self) -> None:
        """Test competition fails categories filter."""
        comp = make_fake_competition(category="Playground")
        filter_ = InterestFilter(
            include_tags=(),
            exclude_tags=(),
            min_reward=None,
            categories=("Featured", "Research"),
        )
        assert _passes_filter(comp, filter_) is False


class TestFilterCompetitions:
    """Tests for filter_competitions function."""

    def test_filter_by_tags(self) -> None:
        """Test filtering competitions by tags."""
        comps = (
            make_fake_competition(ref="tabular-comp", tags=("tabular",)),
            make_fake_competition(ref="nlp-comp", tags=("nlp",)),
            make_fake_competition(ref="vision-comp", tags=("computer-vision",)),
        )
        filter_ = InterestFilter(
            include_tags=("tabular", "nlp"),
            exclude_tags=(),
            min_reward=None,
            categories=None,
        )

        filtered = filter_competitions(comps, filter_)

        assert len(filtered) == 2
        refs = [c.ref for c in filtered]
        assert "tabular-comp" in refs
        assert "nlp-comp" in refs

    def test_filter_by_exclude_tags(self) -> None:
        """Test filtering competitions by exclude tags."""
        comps = (
            make_fake_competition(ref="tabular-comp", tags=("tabular",)),
            make_fake_competition(ref="vision-comp", tags=("computer-vision",)),
        )
        filter_ = InterestFilter(
            include_tags=(),
            exclude_tags=("computer-vision",),
            min_reward=None,
            categories=None,
        )

        filtered = filter_competitions(comps, filter_)

        assert len(filtered) == 1
        assert filtered[0].ref == "tabular-comp"

    def test_filter_empty_result(self) -> None:
        """Test filtering with no matches."""
        comps = (make_fake_competition(ref="vision-comp", tags=("computer-vision",)),)
        filter_ = InterestFilter(
            include_tags=("tabular",),
            exclude_tags=(),
            min_reward=None,
            categories=None,
        )

        filtered = filter_competitions(comps, filter_)

        assert filtered == ()


class TestMakeInterestFilter:
    """Tests for make_interest_filter function."""

    def test_basic_filter(self) -> None:
        """Test creating a basic filter."""
        filter_ = make_interest_filter(
            include_tags=("tabular",),
            exclude_tags=("vision",),
        )
        assert filter_.include_tags == ("tabular",)
        assert filter_.exclude_tags == ("vision",)
        assert filter_.min_reward is None
        assert filter_.categories is None

    def test_filter_with_categories(self) -> None:
        """Test creating filter with categories."""
        filter_ = make_interest_filter(
            categories=("Featured", "Research"),
        )
        assert filter_.categories == ("Featured", "Research")

    def test_filter_with_all_categories(self) -> None:
        """Test creating filter with all valid categories."""
        filter_ = make_interest_filter(
            categories=(
                "Featured",
                "Research",
                "Playground",
                "Getting Started",
                "Masters",
                "Kudos",
            ),
        )
        assert filter_.categories == (
            "Featured",
            "Research",
            "Playground",
            "Getting Started",
            "Masters",
            "Kudos",
        )

    def test_filter_with_invalid_category(self) -> None:
        """Test creating filter with invalid category ignores it."""
        filter_ = make_interest_filter(
            categories=("Featured", "Invalid"),
        )
        # Invalid category should be skipped
        assert filter_.categories == ("Featured",)

    def test_filter_with_min_reward(self) -> None:
        """Test creating filter with min reward."""
        filter_ = make_interest_filter(min_reward=50000)
        assert filter_.min_reward == 50000
