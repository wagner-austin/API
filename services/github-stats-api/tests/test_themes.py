from __future__ import annotations

from github_stats_api.themes import get_theme, get_theme_names


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
        assert theme["gradient"] is None
        assert theme["glow_color"] is None
        assert theme["sparkle_color"] is None
        assert theme["sparkle_count"] == 0

    def test_get_theme_dracula(self) -> None:
        """Test getting dracula theme."""
        theme = get_theme("dracula")

        assert theme["bg_color"] == "#282a36"
        assert theme["title_color"] == "#ff79c6"
        assert theme["text_color"] == "#f8f8f2"
        assert theme["border_color"] == "#44475a"
        assert theme["icon_color"] == "#bd93f9"
        assert theme["gradient"] is None
        assert theme["glow_color"] is None

    def test_get_theme_dark(self) -> None:
        """Test getting dark theme."""
        theme = get_theme("dark")

        assert theme["bg_color"] == "#151515"
        assert theme["title_color"] == "#fff"
        assert theme["gradient"] is None

    def test_get_theme_github_dark(self) -> None:
        """Test getting github_dark theme."""
        theme = get_theme("github_dark")

        assert theme["bg_color"] == "#0d1117"
        assert theme["title_color"] == "#58a6ff"
        assert theme["gradient"] is None

    def test_get_theme_transparent(self) -> None:
        """Test getting transparent theme."""
        theme = get_theme("transparent")

        assert theme["bg_color"] == "#00000000"
        assert theme["border_color"] == "#00000000"
        assert theme["gradient"] is None

    def test_get_theme_unknown_returns_default(self) -> None:
        """Test that unknown theme returns default."""
        theme = get_theme("nonexistent-theme")

        assert theme["bg_color"] == "#fffefe"
        assert theme["title_color"] == "#2f80ed"


class TestGetThemeCyberpunk:
    """Tests for cyberpunk theme."""

    def test_get_theme_cyberpunk_colors(self) -> None:
        """Test cyberpunk theme base colors."""
        theme = get_theme("cyberpunk")

        assert theme["bg_color"] == "#0a0a0f"
        assert theme["title_color"] == "#00fff9"
        assert theme["text_color"] == "#e0e0e0"
        assert theme["border_color"] == "#ff00ff"
        assert theme["icon_color"] == "#ff00ff"

    def test_get_theme_cyberpunk_gradient(self) -> None:
        """Test cyberpunk theme gradient."""
        theme = get_theme("cyberpunk")

        gradient = theme["gradient"]
        # Gradient must be defined for cyberpunk theme
        assert gradient == {
            "angle": 135,
            "stops": (
                {"offset": 0, "color": "#0a0a0f"},
                {"offset": 50, "color": "#1a0a2e"},
                {"offset": 100, "color": "#0a0a0f"},
            ),
        }

    def test_get_theme_cyberpunk_effects(self) -> None:
        """Test cyberpunk theme visual effects."""
        theme = get_theme("cyberpunk")

        assert theme["glow_color"] == "#00fff9"
        assert theme["sparkle_color"] == "#ff00ff"
        assert theme["sparkle_count"] == 8


class TestGetThemeSynthwave:
    """Tests for synthwave theme."""

    def test_get_theme_synthwave_colors(self) -> None:
        """Test synthwave theme base colors."""
        theme = get_theme("synthwave")

        assert theme["bg_color"] == "#1a1a2e"
        assert theme["title_color"] == "#f72585"
        assert theme["icon_color"] == "#4cc9f0"

    def test_get_theme_synthwave_gradient(self) -> None:
        """Test synthwave theme gradient."""
        theme = get_theme("synthwave")

        gradient = theme["gradient"]
        assert gradient == {
            "angle": 180,
            "stops": (
                {"offset": 0, "color": "#1a1a2e"},
                {"offset": 50, "color": "#2d1b4e"},
                {"offset": 100, "color": "#1a1a2e"},
            ),
        }

    def test_get_theme_synthwave_effects(self) -> None:
        """Test synthwave theme visual effects."""
        theme = get_theme("synthwave")

        assert theme["glow_color"] == "#f72585"
        assert theme["sparkle_color"] == "#4cc9f0"
        assert theme["sparkle_count"] == 6


class TestGetThemeNeon:
    """Tests for neon theme."""

    def test_get_theme_neon_colors(self) -> None:
        """Test neon theme base colors."""
        theme = get_theme("neon")

        assert theme["bg_color"] == "#0d0d0d"
        assert theme["title_color"] == "#39ff14"
        assert theme["icon_color"] == "#ff073a"

    def test_get_theme_neon_gradient(self) -> None:
        """Test neon theme gradient."""
        theme = get_theme("neon")

        gradient = theme["gradient"]
        assert gradient == {
            "angle": 45,
            "stops": (
                {"offset": 0, "color": "#0d0d0d"},
                {"offset": 100, "color": "#1a1a1a"},
            ),
        }

    def test_get_theme_neon_effects(self) -> None:
        """Test neon theme visual effects."""
        theme = get_theme("neon")

        assert theme["glow_color"] == "#39ff14"
        assert theme["sparkle_color"] == "#ff073a"
        assert theme["sparkle_count"] == 10


class TestGetThemeAurora:
    """Tests for aurora theme."""

    def test_get_theme_aurora_colors(self) -> None:
        """Test aurora theme base colors."""
        theme = get_theme("aurora")

        assert theme["bg_color"] == "#0f0c29"
        assert theme["title_color"] == "#a8ff78"
        assert theme["icon_color"] == "#78ffd6"

    def test_get_theme_aurora_gradient(self) -> None:
        """Test aurora theme gradient."""
        theme = get_theme("aurora")

        gradient = theme["gradient"]
        assert gradient == {
            "angle": 135,
            "stops": (
                {"offset": 0, "color": "#0f0c29"},
                {"offset": 33, "color": "#302b63"},
                {"offset": 66, "color": "#24243e"},
                {"offset": 100, "color": "#0f0c29"},
            ),
        }

    def test_get_theme_aurora_effects(self) -> None:
        """Test aurora theme visual effects."""
        theme = get_theme("aurora")

        assert theme["glow_color"] == "#78ffd6"
        assert theme["sparkle_color"] == "#a8ff78"
        assert theme["sparkle_count"] == 12


class TestGetThemeRadical:
    """Tests for radical theme."""

    def test_get_theme_radical_colors(self) -> None:
        """Test radical theme base colors."""
        theme = get_theme("radical")

        assert theme["bg_color"] == "#141321"
        assert theme["title_color"] == "#fe428e"
        assert theme["icon_color"] == "#f8d847"

    def test_get_theme_radical_gradient(self) -> None:
        """Test radical theme gradient."""
        theme = get_theme("radical")

        gradient = theme["gradient"]
        assert gradient == {
            "angle": 160,
            "stops": (
                {"offset": 0, "color": "#141321"},
                {"offset": 50, "color": "#1e1b32"},
                {"offset": 100, "color": "#141321"},
            ),
        }

    def test_get_theme_radical_effects(self) -> None:
        """Test radical theme visual effects."""
        theme = get_theme("radical")

        assert theme["glow_color"] == "#fe428e"
        assert theme["sparkle_color"] == "#a9fef7"
        assert theme["sparkle_count"] == 8


class TestGetThemeNames:
    """Tests for get_theme_names function."""

    def test_get_theme_names_returns_all_themes(self) -> None:
        """Test that get_theme_names returns all theme names."""
        names = get_theme_names()

        assert "default" in names
        assert "dark" in names
        assert "dracula" in names
        assert "github_dark" in names
        assert "transparent" in names
        assert "cyberpunk" in names
        assert "synthwave" in names
        assert "neon" in names
        assert "aurora" in names
        assert "radical" in names

    def test_get_theme_names_returns_tuple_of_ten(self) -> None:
        """Test that get_theme_names returns exactly 10 theme names."""
        names = get_theme_names()

        # Verify exact expected output
        assert names == (
            "default",
            "dark",
            "dracula",
            "github_dark",
            "transparent",
            "cyberpunk",
            "synthwave",
            "neon",
            "aurora",
            "radical",
        )
