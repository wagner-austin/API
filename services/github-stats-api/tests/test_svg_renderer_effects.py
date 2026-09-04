"""Tests for svg renderer validation: RenderStatsCardWithEffects."""

from __future__ import annotations

from github_stats_api.api.schemas.stats import (
    LanguageStats,
    UserStats,
)
from github_stats_api.renderers import (
    render_langs_card,
    render_stats_card,
)


class TestRenderStatsCardWithEffects:
    """Tests for render_stats_card with visual effects themes."""

    def test_render_stats_card_default_no_sparkles(self) -> None:
        """Test that default theme has no sparkle decorations."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "A",
            "rank_percentile": 30.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="default",
            hide_border=False,
            show_icons=True,
            hide=(),
            disable_animations=False,
        )

        # Default theme should NOT have sparkles
        assert 'class="sparkles"' not in svg

    def test_render_stats_card_cyberpunk_has_gradient(self) -> None:
        """Test that cyberpunk theme includes gradient."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "A",
            "rank_percentile": 30.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="cyberpunk",
            hide_border=False,
            show_icons=True,
            hide=(),
            disable_animations=False,
        )

        assert "<defs>" in svg
        assert "linearGradient" in svg
        assert 'id="stats-grad"' in svg

    def test_render_stats_card_cyberpunk_has_sparkles(self) -> None:
        """Test that cyberpunk theme includes sparkle decorations."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "A",
            "rank_percentile": 30.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="cyberpunk",
            hide_border=False,
            show_icons=True,
            hide=(),
            disable_animations=False,
        )

        assert 'class="sparkles"' in svg
        assert "<path" in svg  # Sparkle paths

    def test_render_stats_card_cyberpunk_has_glow(self) -> None:
        """Test that cyberpunk theme includes glow CSS."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "A",
            "rank_percentile": 30.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="cyberpunk",
            hide_border=False,
            show_icons=True,
            hide=(),
            disable_animations=False,
        )

        assert ".glow-text" in svg
        assert "drop-shadow" in svg
        assert 'class="header glow-text"' in svg


class TestRenderLangsCardWithEffects:
    """Tests for render_langs_card with visual effects themes."""

    def test_render_langs_card_neon_has_gradient(self) -> None:
        """Test that neon theme includes gradient."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 50000, "percentage": 50.0, "color": "#3572A5"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=50000,
            theme_name="neon",
            hide_border=False,
            layout="default",
            langs_count=8,
            disable_animations=False,
        )

        assert "<defs>" in svg
        assert "linearGradient" in svg
        assert 'id="langs-grad"' in svg

    def test_render_langs_card_aurora_has_sparkles(self) -> None:
        """Test that aurora theme includes sparkle decorations."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 50000, "percentage": 50.0, "color": "#3572A5"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=50000,
            theme_name="aurora",
            hide_border=False,
            layout="compact",
            langs_count=8,
            disable_animations=False,
        )

        assert 'class="sparkles"' in svg
