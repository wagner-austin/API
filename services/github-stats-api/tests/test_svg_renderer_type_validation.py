"""Tests for svg renderer validation: BuildUserStatsTypeValidation."""

from __future__ import annotations

from github_stats_api.api.schemas.stats import (
    Capability,
)
from github_stats_api.svg_renderer import (
    build_capabilities_response,
    build_language_stats,
    build_user_stats,
)


class TestBuildUserStatsTypeValidation:
    """Tests for build_user_stats type validation."""

    def test_build_user_stats_non_int_commits(self) -> None:
        """Test handling non-int commits."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test",
            "total_commits": "not_an_int",
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        assert stats["total_commits"] == 0

    def test_build_user_stats_non_int_prs(self) -> None:
        """Test handling non-int prs."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test",
            "total_commits": 10,
            "total_prs": "not_an_int",
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        assert stats["total_prs"] == 0

    def test_build_user_stats_non_int_issues(self) -> None:
        """Test handling non-int issues."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test",
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": "not_an_int",
            "total_stars": 1,
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        assert stats["total_issues"] == 0

    def test_build_user_stats_non_int_stars(self) -> None:
        """Test handling non-int stars."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test",
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": "not_an_int",
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        assert stats["total_stars"] == 0

    def test_build_user_stats_non_str_name(self) -> None:
        """Test handling non-str name."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": 123,
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        # Should fall back to login
        assert stats["name"] == "testuser"

    def test_build_user_stats_non_str_login(self) -> None:
        """Test handling non-str login."""
        data: dict[str, int | str] = {
            "login": 123,
            "name": "Test User",
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        assert stats["username"] == ""

    def test_build_user_stats_non_int_contributions(self) -> None:
        """Test handling non-int contributions."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test",
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": "not_an_int",
        }

        stats = build_user_stats(data)
        assert stats["total_contributions"] == 0


class TestBuildLanguageStatsTypeValidation:
    """Tests for build_language_stats type validation."""

    def test_build_language_stats_non_str_color(self) -> None:
        """Test handling non-str color."""
        languages: list[dict[str, int | str]] = [
            {"name": "Python", "size": 100, "color": 123},
        ]

        stats, _total = build_language_stats(languages)
        assert len(stats) == 1
        assert stats[0]["color"] == "#858585"  # Default color

    def test_build_language_stats_negative_size(self) -> None:
        """Test handling negative size."""
        languages: list[dict[str, int | str]] = [
            {"name": "Python", "size": -100, "color": "#3572A5"},
            {"name": "JavaScript", "size": 100, "color": "#f1e05a"},
        ]

        stats, _total = build_language_stats(languages)
        assert len(stats) == 1
        assert stats[0]["name"] == "JavaScript"

    def test_build_language_stats_non_int_size(self) -> None:
        """Test handling non-int size."""
        languages: list[dict[str, int | str]] = [
            {"name": "Python", "size": "not_an_int", "color": "#3572A5"},
            {"name": "JavaScript", "size": 100, "color": "#f1e05a"},
        ]

        stats, _total = build_language_stats(languages)
        assert len(stats) == 1
        assert stats[0]["name"] == "JavaScript"

    def test_build_language_stats_non_str_name(self) -> None:
        """Test handling non-str name."""
        languages: list[dict[str, int | str]] = [
            {"name": 123, "size": 100, "color": "#3572A5"},
            {"name": "JavaScript", "size": 100, "color": "#f1e05a"},
        ]

        stats, _total = build_language_stats(languages)
        assert len(stats) == 1
        assert stats[0]["name"] == "JavaScript"


class TestBuildCapabilitiesResponse:
    """Tests for build_capabilities_response function."""

    def test_build_capabilities_response_basic(self) -> None:
        """Test building capabilities response."""
        cap: Capability = {
            "name": "xgboost_tabular",
            "strength": "strong",
            "tags": ("ml", "tabular", "xgboost"),
            "description": "XGBoost gradient boosting",
        }

        response = build_capabilities_response(
            repo="owner/repo",
            capabilities=(cap,),
            ml_backends=("xgboost", "lightgbm"),
            frameworks=("fastapi",),
            data_formats=("csv", "parquet"),
            task_types=("binary_classification",),
        )

        assert response["repo"] == "owner/repo"
        assert len(response["capabilities"]) == 1
        assert response["capabilities"][0]["name"] == "xgboost_tabular"
        assert response["ml_backends"] == ("xgboost", "lightgbm")
        assert response["frameworks"] == ("fastapi",)
        assert response["data_formats"] == ("csv", "parquet")
        assert response["task_types"] == ("binary_classification",)

    def test_build_capabilities_response_empty(self) -> None:
        """Test building capabilities response with no capabilities."""
        response = build_capabilities_response(
            repo="owner/repo",
            capabilities=(),
            ml_backends=(),
            frameworks=(),
            data_formats=(),
            task_types=(),
        )

        assert response["repo"] == "owner/repo"
        assert response["capabilities"] == ()
        assert response["ml_backends"] == ()
