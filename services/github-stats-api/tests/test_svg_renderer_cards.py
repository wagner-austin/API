"""Tests for svg renderer: RenderStatsCard."""

from __future__ import annotations

from github_stats_api.api.schemas.stats import (
    LanguageStats,
    UserStats,
)
from github_stats_api.renderers import (
    render_langs_card,
    render_stats_card,
)


class TestRenderStatsCard:
    """Tests for render_stats_card function."""

    def test_render_stats_card_basic(self) -> None:
        """Test rendering stats card."""
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

        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "Test User" in svg
        assert "Total Stars" in svg
        assert "Total Commits" in svg

    def test_render_stats_card_hides_stats(self) -> None:
        """Test that hide parameter hides stats."""
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
            hide=("stars", "commits"),
            disable_animations=False,
        )

        assert "Total Stars" not in svg
        assert "Total Commits" not in svg
        assert "Total PRs" in svg

    def test_render_stats_card_respects_theme(self) -> None:
        """Test that theme colors are applied."""
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
            theme_name="dracula",
            hide_border=False,
            show_icons=True,
            hide=(),
            disable_animations=False,
        )

        assert "#282a36" in svg  # dracula bg color
        assert "#ff79c6" in svg  # dracula title color


class TestRenderLangsCard:
    """Tests for render_langs_card function."""

    def test_render_langs_card_default_layout(self) -> None:
        """Test rendering langs card with default layout."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 50000, "percentage": 50.0, "color": "#3572A5"},
            {"name": "TypeScript", "size": 30000, "percentage": 30.0, "color": "#3178c6"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=100000,
            theme_name="default",
            hide_border=False,
            layout="default",
            langs_count=8,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "Python" in svg
        assert "TypeScript" in svg

    def test_render_langs_card_compact_layout(self) -> None:
        """Test rendering langs card with compact layout."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 50000, "percentage": 50.0, "color": "#3572A5"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=50000,
            theme_name="dracula",
            hide_border=True,
            layout="compact",
            langs_count=8,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "Python" in svg

    def test_render_langs_card_donut_layout(self) -> None:
        """Test rendering langs card with donut layout."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 50000, "percentage": 50.0, "color": "#3572A5"},
            {"name": "JavaScript", "size": 50000, "percentage": 50.0, "color": "#f1e05a"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=100000,
            theme_name="default",
            hide_border=False,
            layout="donut",
            langs_count=8,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert 'height="250"' in svg  # donut layout height

    def test_render_langs_card_pie_layout(self) -> None:
        """Test rendering langs card with pie layout."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 50000, "percentage": 50.0, "color": "#3572A5"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=50000,
            theme_name="default",
            hide_border=False,
            layout="pie",
            langs_count=8,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert 'height="250"' in svg  # pie layout height

    def test_render_langs_card_zero_percentage_segment(self) -> None:
        """Test rendering langs card with zero percentage language."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 100, "percentage": 100.0, "color": "#3572A5"},
            {"name": "Other", "size": 0, "percentage": 0.0, "color": "#858585"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=100,
            theme_name="default",
            hide_border=False,
            layout="compact",
            langs_count=8,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "Python" in svg


class TestRenderStatsCardVariations:
    """Additional tests for render_stats_card edge cases."""

    def test_render_stats_card_without_icons(self) -> None:
        """Test rendering stats card without icons."""
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
            show_icons=False,
            hide=(),
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "Total Stars" in svg
        # Icons should not appear in stat lines when show_icons=False
        assert "⭐" not in svg

    def test_render_stats_card_hides_prs_and_issues(self) -> None:
        """Test hiding prs and issues stats."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "B+",
            "rank_percentile": 45.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="default",
            hide_border=True,
            show_icons=True,
            hide=("prs", "issues"),
            disable_animations=False,
        )

        assert "Total PRs" not in svg
        assert "Total Issues" not in svg
        assert "Total Stars" in svg

    def test_render_stats_card_hides_contribs(self) -> None:
        """Test hiding contribs stat."""
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
            hide=("contribs",),
            disable_animations=False,
        )

        assert "Contributed to" not in svg
        assert "Total Stars" in svg
