from __future__ import annotations

from github_stats_api.themes import get_theme


class TestGetTheme:
    """Tests for get_theme function."""

    def test_get_theme_default(self) -> None:
        """Test getting default theme."""
        theme = get_theme("default")

        assert theme["bg_color"] == "#fffefe"
        assert theme["title_color"] == "#2f80ed"
        assert theme["text_color"] == "#434d58"
        assert theme["border_color"] == "#e4e2e2"
        assert theme["icon_color"] == "#4c71f2"

    def test_get_theme_dracula(self) -> None:
        """Test getting dracula theme."""
        theme = get_theme("dracula")

        assert theme["bg_color"] == "#282a36"
        assert theme["title_color"] == "#ff79c6"
        assert theme["text_color"] == "#f8f8f2"
        assert theme["border_color"] == "#44475a"
        assert theme["icon_color"] == "#bd93f9"

    def test_get_theme_dark(self) -> None:
        """Test getting dark theme."""
        theme = get_theme("dark")

        assert theme["bg_color"] == "#151515"
        assert theme["title_color"] == "#fff"

    def test_get_theme_github_dark(self) -> None:
        """Test getting github_dark theme."""
        theme = get_theme("github_dark")

        assert theme["bg_color"] == "#0d1117"
        assert theme["title_color"] == "#58a6ff"

    def test_get_theme_transparent(self) -> None:
        """Test getting transparent theme."""
        theme = get_theme("transparent")

        assert theme["bg_color"] == "#00000000"
        assert theme["border_color"] == "#00000000"

    def test_get_theme_unknown_returns_default(self) -> None:
        """Test that unknown theme returns default."""
        theme = get_theme("nonexistent-theme")

        assert theme["bg_color"] == "#fffefe"
        assert theme["title_color"] == "#2f80ed"
