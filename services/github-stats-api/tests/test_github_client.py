from __future__ import annotations

from github_stats_api.github_client import get_language_color


class TestGetLanguageColor:
    """Tests for get_language_color function."""

    def test_get_language_color_known_language(self) -> None:
        """Test getting color for known language."""
        assert get_language_color("Python") == "#3572A5"
        assert get_language_color("JavaScript") == "#f1e05a"
        assert get_language_color("TypeScript") == "#3178c6"
        assert get_language_color("Rust") == "#dea584"
        assert get_language_color("Go") == "#00ADD8"

    def test_get_language_color_unknown_language(self) -> None:
        """Test getting color for unknown language returns default."""
        assert get_language_color("UnknownLanguage123") == "#858585"
        assert get_language_color("") == "#858585"
