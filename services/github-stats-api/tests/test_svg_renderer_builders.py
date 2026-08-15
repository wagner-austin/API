"""Tests for svg renderer: BuildUserStats."""

from __future__ import annotations

from github_stats_api.svg_renderer import (
    build_language_stats,
    build_user_stats,
)


class TestBuildUserStats:
    """Tests for build_user_stats function."""

    def test_build_user_stats_basic(self) -> None:
        """Test building user stats from data."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
        }

        stats = build_user_stats(data)

        assert stats["username"] == "testuser"
        assert stats["name"] == "Test User"
        assert stats["total_commits"] == 100
        assert stats["total_prs"] == 20
        assert stats["total_issues"] == 10
        assert stats["total_stars"] == 50
        assert stats["total_contributions"] == 150
        assert stats["rank"] in ("S+", "S", "A+", "A", "B+", "B", "C")
        assert 0 <= stats["rank_percentile"] <= 100

    def test_build_user_stats_high_activity_gets_high_rank(self) -> None:
        """Test that high activity results in high rank."""
        data: dict[str, int | str] = {
            "login": "superstar",
            "name": "Super Star",
            "total_commits": 50000,
            "total_prs": 5000,
            "total_issues": 2000,
            "total_stars": 100000,
            "total_contributions": 60000,
        }

        stats = build_user_stats(data)

        assert stats["rank"] in ("S+", "S", "A+")

    def test_build_user_stats_low_activity_gets_low_rank(self) -> None:
        """Test that low activity results in low rank."""
        data: dict[str, int | str] = {
            "login": "newbie",
            "name": "New User",
            "total_commits": 1,
            "total_prs": 0,
            "total_issues": 0,
            "total_stars": 0,
            "total_contributions": 1,
        }

        stats = build_user_stats(data)

        assert stats["rank"] in ("B", "C")

    def test_build_user_stats_handles_missing_name(self) -> None:
        """Test that missing name falls back to login."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "",
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": 20,
        }

        stats = build_user_stats(data)

        assert stats["name"] == "testuser"


class TestBuildLanguageStats:
    """Tests for build_language_stats function."""

    def test_build_language_stats_basic(self) -> None:
        """Test building language stats from data."""
        languages: list[dict[str, int | str]] = [
            {"name": "Python", "size": 50000, "color": "#3572A5"},
            {"name": "JavaScript", "size": 30000, "color": "#f1e05a"},
            {"name": "TypeScript", "size": 20000, "color": "#3178c6"},
        ]

        stats, total = build_language_stats(languages)

        assert total == 100000
        assert len(stats) == 3
        assert stats[0]["name"] == "Python"
        assert stats[0]["percentage"] == 50.0
        assert stats[1]["name"] == "JavaScript"
        assert stats[1]["percentage"] == 30.0
        assert stats[2]["name"] == "TypeScript"
        assert stats[2]["percentage"] == 20.0

    def test_build_language_stats_empty(self) -> None:
        """Test building language stats from empty data."""
        stats, total = build_language_stats([])

        assert total == 0
        assert stats == []

    def test_build_language_stats_filters_invalid(self) -> None:
        """Test that invalid entries are filtered out."""
        languages: list[dict[str, int | str]] = [
            {"name": "Python", "size": 100, "color": "#3572A5"},
            {"name": "", "size": 50, "color": "#000"},  # Empty name
            {"name": "JS", "size": 0, "color": "#f1e05a"},  # Zero size
        ]

        stats, total = build_language_stats(languages)

        assert total == 100
        assert len(stats) == 1
        assert stats[0]["name"] == "Python"
