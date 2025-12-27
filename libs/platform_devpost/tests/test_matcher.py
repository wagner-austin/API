"""Tests for platform_devpost.matcher module."""

from __future__ import annotations

import pytest

from platform_devpost.matcher import (
    _calculate_match_score,
    _determine_recommendation,
    _get_hackathon_tags,
    _get_matched_capabilities,
    _get_missing_capabilities,
    _get_profile_tags,
    match_hackathon,
    match_hackathons,
)
from platform_devpost.testing import (
    make_fake_capability,
    make_fake_hackathon,
    make_fake_profile,
    make_fake_theme,
)


class TestGetHackathonTags:
    """Tests for _get_hackathon_tags function."""

    def test_empty_themes(self) -> None:
        """Test hackathon with no themes returns empty set."""
        h = make_fake_hackathon(themes=())
        tags = _get_hackathon_tags(h)
        assert tags == set()

    def test_theme_name_added(self) -> None:
        """Test theme names are added as tags."""
        theme = make_fake_theme(name="Gaming")
        h = make_fake_hackathon(themes=(theme,))
        tags = _get_hackathon_tags(h)
        assert "gaming" in tags

    def test_mapped_tags_added(self) -> None:
        """Test mapped tags are added from theme name."""
        theme = make_fake_theme(name="Machine Learning")
        h = make_fake_hackathon(themes=(theme,))
        tags = _get_hackathon_tags(h)
        assert "ml" in tags
        assert "ai" in tags
        assert "data-science" in tags

    def test_multiple_themes(self) -> None:
        """Test multiple themes combine tags."""
        t1 = make_fake_theme(name="Web Development")
        t2 = make_fake_theme(name="API")
        h = make_fake_hackathon(themes=(t1, t2))
        tags = _get_hackathon_tags(h)
        assert "web" in tags
        assert "api" in tags
        assert "backend" in tags


class TestGetProfileTags:
    """Tests for _get_profile_tags function."""

    def test_empty_profile(self) -> None:
        """Test empty profile returns empty set."""
        profile = make_fake_profile(
            capabilities=(),
            technologies=(),
            frameworks=(),
        )
        tags = _get_profile_tags(profile)
        assert tags == set()

    def test_capability_tags_added(self) -> None:
        """Test capability tags are added."""
        cap = make_fake_capability(tags=("ml", "ai", "deep-learning"))
        profile = make_fake_profile(capabilities=(cap,), technologies=(), frameworks=())
        tags = _get_profile_tags(profile)
        assert "ml" in tags
        assert "ai" in tags
        assert "deep-learning" in tags

    def test_technologies_added(self) -> None:
        """Test technologies are added as tags."""
        profile = make_fake_profile(
            capabilities=(),
            technologies=("python", "javascript"),
            frameworks=(),
        )
        tags = _get_profile_tags(profile)
        assert "python" in tags
        assert "javascript" in tags

    def test_frameworks_added(self) -> None:
        """Test frameworks are added as tags."""
        profile = make_fake_profile(
            capabilities=(),
            technologies=(),
            frameworks=("flask", "react"),
        )
        tags = _get_profile_tags(profile)
        assert "flask" in tags
        assert "react" in tags


class TestCalculateMatchScore:
    """Tests for _calculate_match_score function."""

    def test_empty_hackathon_tags(self) -> None:
        """Test empty hackathon tags returns 0.0."""
        score = _calculate_match_score(set(), {"python", "ml"})
        assert score == 0.0

    def test_no_overlap(self) -> None:
        """Test no overlap returns 0.0."""
        score = _calculate_match_score({"web", "frontend"}, {"ml", "backend"})
        assert score == 0.0

    def test_full_overlap(self) -> None:
        """Test full overlap returns 1.0."""
        score = _calculate_match_score({"python", "ml"}, {"python", "ml", "ai"})
        assert score == 1.0

    def test_partial_overlap(self) -> None:
        """Test partial overlap returns fractional score."""
        score = _calculate_match_score({"python", "ml", "data"}, {"python", "web"})
        assert 0.0 < score < 1.0
        assert score == pytest.approx(1.0 / 3.0)


class TestDetermineRecommendation:
    """Tests for _determine_recommendation function."""

    def test_strong_fit(self) -> None:
        """Test high score returns strong_fit."""
        assert _determine_recommendation(0.8) == "strong_fit"
        assert _determine_recommendation(0.7) == "strong_fit"

    def test_good_fit(self) -> None:
        """Test medium score returns good_fit."""
        assert _determine_recommendation(0.5) == "good_fit"
        assert _determine_recommendation(0.4) == "good_fit"

    def test_stretch(self) -> None:
        """Test low score returns stretch."""
        assert _determine_recommendation(0.3) == "stretch"
        assert _determine_recommendation(0.2) == "stretch"

    def test_new_territory(self) -> None:
        """Test very low score returns new_territory."""
        assert _determine_recommendation(0.1) == "new_territory"
        assert _determine_recommendation(0.0) == "new_territory"


class TestGetMatchedCapabilities:
    """Tests for _get_matched_capabilities function."""

    def test_no_capabilities(self) -> None:
        """Test empty capabilities returns empty tuple."""
        profile = make_fake_profile(capabilities=())
        result = _get_matched_capabilities({"ml", "ai"}, profile)
        assert result == ()

    def test_no_matching_caps(self) -> None:
        """Test no matching capabilities returns empty tuple."""
        cap = make_fake_capability(name="web_dev", tags=("web", "frontend"))
        profile = make_fake_profile(capabilities=(cap,), technologies=(), frameworks=())
        result = _get_matched_capabilities({"ml", "ai"}, profile)
        assert result == ()

    def test_matching_caps(self) -> None:
        """Test matching capabilities are returned."""
        cap1 = make_fake_capability(name="ml_cap", tags=("ml", "ai"))
        cap2 = make_fake_capability(name="web_cap", tags=("web", "frontend"))
        profile = make_fake_profile(capabilities=(cap1, cap2), technologies=(), frameworks=())
        result = _get_matched_capabilities({"ml", "data"}, profile)
        assert "ml_cap" in result
        assert "web_cap" not in result

    def test_sorted_and_deduplicated(self) -> None:
        """Test results are sorted and deduplicated."""
        cap1 = make_fake_capability(name="b_cap", tags=("ml",))
        cap2 = make_fake_capability(name="a_cap", tags=("ml",))
        profile = make_fake_profile(capabilities=(cap1, cap2), technologies=(), frameworks=())
        result = _get_matched_capabilities({"ml"}, profile)
        assert result == ("a_cap", "b_cap")


class TestGetMissingCapabilities:
    """Tests for _get_missing_capabilities function."""

    def test_no_missing(self) -> None:
        """Test no missing returns empty tuple."""
        result = _get_missing_capabilities({"ml", "ai"}, {"ml", "ai", "data"})
        assert result == ()

    def test_all_missing(self) -> None:
        """Test all missing returns all hackathon tags."""
        result = _get_missing_capabilities({"ml", "ai"}, {"web", "frontend"})
        assert set(result) == {"ml", "ai"}

    def test_partial_missing(self) -> None:
        """Test partial missing returns only missing tags."""
        result = _get_missing_capabilities({"ml", "ai", "web"}, {"ml", "frontend"})
        assert set(result) == {"ai", "web"}

    def test_sorted(self) -> None:
        """Test results are sorted."""
        result = _get_missing_capabilities({"c", "a", "b"}, set())
        assert result == ("a", "b", "c")


class TestMatchHackathon:
    """Tests for match_hackathon function."""

    def test_match_returns_hackathon_match(self) -> None:
        """Test match_hackathon returns HackathonMatch."""
        theme = make_fake_theme(name="Machine Learning")
        h = make_fake_hackathon(themes=(theme,))
        cap = make_fake_capability(name="ml_cap", tags=("ml", "ai"))
        profile = make_fake_profile(capabilities=(cap,), technologies=(), frameworks=())

        result = match_hackathon(h, profile)

        assert result.hackathon == h
        assert 0.0 <= result.match_score <= 1.0
        assert "ml_cap" in result.matched_capabilities

    def test_match_no_overlap(self) -> None:
        """Test match with no overlap."""
        theme = make_fake_theme(name="Blockchain")
        h = make_fake_hackathon(themes=(theme,))
        cap = make_fake_capability(name="ml_cap", tags=("ml",))
        profile = make_fake_profile(capabilities=(cap,), technologies=(), frameworks=())

        result = match_hackathon(h, profile)

        assert result.match_score == 0.0
        assert result.recommendation == "new_territory"

    def test_match_empty_hackathon_themes(self) -> None:
        """Test match with no hackathon themes."""
        h = make_fake_hackathon(themes=())
        profile = make_fake_profile()

        result = match_hackathon(h, profile)

        assert result.match_score == 0.0
        assert result.missing_capabilities == ()


class TestMatchHackathons:
    """Tests for match_hackathons function."""

    def test_empty_hackathons(self) -> None:
        """Test empty hackathons returns empty tuple."""
        profile = make_fake_profile()
        result = match_hackathons((), profile)
        assert result == ()

    def test_sorted_by_score(self) -> None:
        """Test results are sorted by score descending."""
        t_ml = make_fake_theme(name="Machine Learning")
        t_web = make_fake_theme(name="Web Development")
        h1 = make_fake_hackathon(id=1, themes=(t_web,))
        h2 = make_fake_hackathon(id=2, themes=(t_ml,))

        cap = make_fake_capability(name="ml_cap", tags=("ml", "ai", "data-science"))
        profile = make_fake_profile(capabilities=(cap,), technologies=(), frameworks=())

        result = match_hackathons((h1, h2), profile)

        assert len(result) == 2
        assert result[0].hackathon.id == 2  # ML hackathon should be first
        assert result[0].match_score >= result[1].match_score

    def test_multiple_hackathons(self) -> None:
        """Test matching multiple hackathons."""
        h1 = make_fake_hackathon(id=1)
        h2 = make_fake_hackathon(id=2)
        h3 = make_fake_hackathon(id=3)
        profile = make_fake_profile()

        result = match_hackathons((h1, h2, h3), profile)

        assert len(result) == 3
