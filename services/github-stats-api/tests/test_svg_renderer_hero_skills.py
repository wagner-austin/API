"""Tests for svg renderer validation: RenderHeroCard."""

from __future__ import annotations

from github_stats_api.renderers import (
    render_hero_card,
    render_skills_card,
)


class TestRenderHeroCard:
    """Tests for render_hero_card function."""

    def test_render_hero_card_basic(self) -> None:
        """Test rendering basic hero card."""
        svg = render_hero_card(
            name="Austin Wagner",
            subtitle="Full-Stack Developer",
            lines=("Location: Irvine", "Education: UC Irvine"),
            theme_name="default",
            disable_animations=False,
        )

        assert "<svg" in svg
        assert "</svg>" in svg
        assert "Austin Wagner" in svg
        assert "Full-Stack Developer" in svg
        assert "Location: Irvine" in svg
        assert "Education: UC Irvine" in svg

    def test_render_hero_card_no_subtitle(self) -> None:
        """Test rendering hero card without subtitle."""
        svg = render_hero_card(
            name="Test User",
            subtitle="",
            lines=(),
            theme_name="default",
            disable_animations=False,
        )

        assert "<svg" in svg
        assert "Test User" in svg

    def test_render_hero_card_cyberpunk_has_rain(self) -> None:
        """Test that cyberpunk theme includes rain animation."""
        svg = render_hero_card(
            name="Test",
            subtitle="",
            lines=(),
            theme_name="cyberpunk",
            disable_animations=False,
        )

        assert "rainFall" in svg
        assert "rain-drop" in svg

    def test_render_hero_card_disable_animations(self) -> None:
        """Test that disabling animations removes rain."""
        svg = render_hero_card(
            name="Test",
            subtitle="",
            lines=(),
            theme_name="cyberpunk",
            disable_animations=True,
        )

        assert "rainFall" not in svg
        assert "rain-drop" not in svg

    def test_render_hero_card_cyberpunk_has_glow(self) -> None:
        """Test that cyberpunk theme includes glow effect."""
        svg = render_hero_card(
            name="Test",
            subtitle="",
            lines=(),
            theme_name="cyberpunk",
            disable_animations=False,
        )

        assert "glowPulse" in svg
        assert "glow-text" in svg

    def test_render_hero_card_escapes_special_chars(self) -> None:
        """Test that special characters are escaped."""
        svg = render_hero_card(
            name="Test & User",
            subtitle="Dev <JS>",
            lines=("Line with 'quotes'",),
            theme_name="default",
            disable_animations=True,
        )

        assert "&amp;" in svg
        assert "&lt;" in svg
        assert "&gt;" in svg
        assert "&apos;" in svg

    def test_render_hero_card_dynamic_height(self) -> None:
        """Test that height adjusts based on line count."""
        svg_few = render_hero_card(
            name="Test",
            subtitle="",
            lines=("Line 1",),
            theme_name="default",
            disable_animations=True,
        )

        svg_many = render_hero_card(
            name="Test",
            subtitle="",
            lines=("Line 1", "Line 2", "Line 3", "Line 4"),
            theme_name="default",
            disable_animations=True,
        )

        # Extract heights from viewBox - svg_few has 1 line = 184px, svg_many has 4 lines = 256px
        assert 'viewBox="0 0 495 184"' in svg_few
        assert 'viewBox="0 0 495 256"' in svg_many


class TestRenderSkillsCard:
    """Tests for render_skills_card function."""

    def test_render_skills_card_basic(self) -> None:
        """Test rendering basic skills card."""
        svg = render_skills_card(
            skills=("Python", "TypeScript", "React"),
            theme_name="default",
            hide_border=False,
            disable_animations=False,
        )

        assert "<svg" in svg
        assert "</svg>" in svg
        assert "Tech Stack" in svg
        assert "Python" in svg
        assert "TypeScript" in svg
        assert "React" in svg

    def test_render_skills_card_cyberpunk_has_glow(self) -> None:
        """Test that cyberpunk theme includes glow effect."""
        svg = render_skills_card(
            skills=("Python",),
            theme_name="cyberpunk",
            hide_border=False,
            disable_animations=False,
        )

        assert "glowPulse" in svg
        assert "glow-text" in svg

    def test_render_skills_card_cyberpunk_has_sparkles(self) -> None:
        """Test that cyberpunk theme includes sparkles."""
        svg = render_skills_card(
            skills=("Python",),
            theme_name="cyberpunk",
            hide_border=False,
            disable_animations=False,
        )

        assert "sparkle" in svg
        assert "twinkle" in svg

    def test_render_skills_card_disable_animations(self) -> None:
        """Test that disabling animations removes effects."""
        svg = render_skills_card(
            skills=("Python",),
            theme_name="cyberpunk",
            hide_border=True,
            disable_animations=True,
        )

        assert "glowPulse" not in svg
        assert "twinkle" not in svg

    def test_render_skills_card_hide_border(self) -> None:
        """Test that hide_border sets border opacity to 0."""
        svg = render_skills_card(
            skills=("Python",),
            theme_name="default",
            hide_border=True,
            disable_animations=True,
        )

        assert 'stroke-opacity="0"' in svg

    def test_render_skills_card_escapes_special_chars(self) -> None:
        """Test that special characters are escaped."""
        svg = render_skills_card(
            skills=("C++", "C#", "F#"),
            theme_name="default",
            hide_border=False,
            disable_animations=True,
        )

        assert "C++" in svg
        assert "C#" in svg
        assert "F#" in svg

    def test_render_skills_card_dynamic_height(self) -> None:
        """Test that height adjusts based on skill count."""
        svg_few = render_skills_card(
            skills=("Python", "React"),
            theme_name="default",
            hide_border=False,
            disable_animations=True,
        )

        svg_many = render_skills_card(
            skills=(
                "Python",
                "TypeScript",
                "React",
                "FastAPI",
                "Docker",
                "Redis",
                "PostgreSQL",
                "Git",
            ),
            theme_name="default",
            hide_border=False,
            disable_animations=True,
        )

        # svg_few has 2 skills = 1 row (2 per row), svg_many has 8 skills = 4 rows
        assert 'viewBox="0 0 495 114"' in svg_few
        assert 'viewBox="0 0 495 246"' in svg_many

    def test_render_skills_card_has_colored_circles(self) -> None:
        """Test that skills have colored circle icons."""
        svg = render_skills_card(
            skills=("Python", "Docker"),
            theme_name="default",
            hide_border=False,
            disable_animations=True,
        )

        # Python color is #3776ab, Docker color is #2496ed
        assert 'fill="#3776ab"' in svg
        assert 'fill="#2496ed"' in svg
        # Skills with icons use <path>, skills without use <circle>
        assert "<path" in svg or "<circle" in svg

    def test_render_icon_with_transform(self) -> None:
        """Test that _render_icon includes transform when set."""
        from github_stats_api.icons import MultiPathIcon
        from github_stats_api.renderers.skills import _render_icon

        icon: MultiPathIcon = {
            "viewbox_width": 24,
            "viewbox_height": 24,
            "paths": ({"d": "M0 0h24v24H0z", "fill": "#ff0000"},),
            "transform": "rotate(45)",
        }
        result = _render_icon(icon, 10, 20, 18)
        assert "rotate(45)" in result
        assert "translate(10, 20)" in result
        assert "scale(" in result
