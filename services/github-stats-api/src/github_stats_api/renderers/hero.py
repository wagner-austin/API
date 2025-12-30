"""Hero card renderer."""

from __future__ import annotations

from ..themes import get_theme
from ._common import escape_xml, get_glow_css, render_background


def _get_rain_css(rain_color: str, drop_count: int) -> str:
    """Get CSS for falling rain animation."""
    css_parts = [
        """
@keyframes rainFall {
  0% { transform: translateY(-20px); opacity: 0; }
  10% { opacity: 0.6; }
  90% { opacity: 0.6; }
  100% { transform: translateY(350px); opacity: 0; }
}
"""
    ]

    for i in range(drop_count):
        duration = 1.5 + (i % 5) * 0.3
        delay = (i * 0.15) % 2.0
        css_parts.append(
            f".rain-{i} {{ animation: rainFall {duration:.1f}s linear {delay:.1f}s infinite; }}"
        )

    css_parts.append(f".rain-drop {{ fill: {rain_color}; opacity: 0; }}")

    return "\n".join(css_parts)


def _render_rain_drops(width: int, height: int, rain_color: str, drop_count: int) -> str:
    """Render SVG rain drop elements."""
    drops = []
    for i in range(drop_count):
        x = 10 + ((i * 37) % (width - 20))
        length = 8 + (i % 4) * 3
        drops.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{length}" '
            f'stroke="{rain_color}" stroke-width="1" class="rain-drop rain-{i}"/>'
        )
    return "\n".join(drops)


def render_hero_card(
    name: str,
    subtitle: str,
    lines: tuple[str, ...],
    theme_name: str,
    disable_animations: bool,
) -> str:
    """Render a full-width hero card with rain animation."""
    theme = get_theme(theme_name)

    width = 495
    base_height = 120
    lines_height = len(lines) * 24
    height = base_height + lines_height + 40

    title_color = theme["title_color"]
    text_color = theme["text_color"]
    glow_color = theme["glow_color"]

    rain_color = glow_color if glow_color is not None else "#00fff9"
    drop_count = 25

    rain_css = "" if disable_animations else _get_rain_css(rain_color, drop_count)
    glow_css = "" if glow_color is None else get_glow_css(glow_color)

    defs_svg, rect_svg = render_background(width, height, theme, False, "hero-grad")

    if disable_animations:
        rain_svg = ""
    else:
        rain_svg = _render_rain_drops(width, height, rain_color, drop_count)

    header_class = "hero-name glow-text" if glow_color is not None else "hero-name"

    left_margin = 25
    line_color = "#ffffff"

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        defs_svg,
        "<style>",
        f".hero-name {{ font: 700 42px 'Segoe UI', sans-serif; fill: {title_color}; }}",
        f".hero-subtitle {{ font: 400 16px 'Segoe UI', sans-serif; fill: {text_color}; "
        "opacity: 0.9; }",
        f".hero-line {{ font: 400 14px 'Segoe UI', sans-serif; fill: {line_color}; }}",
        rain_css,
        glow_css,
        "</style>",
        rect_svg,
        rain_svg,
        f'<text x="{left_margin}" y="50" text-anchor="start" class="{header_class}">'
        f"{escape_xml(name)}</text>",
    ]

    if subtitle:
        svg_parts.append(
            f'<text x="{left_margin}" y="75" text-anchor="start" class="hero-subtitle">'
            f"{escape_xml(subtitle)}</text>"
        )

    y_offset = 110
    for line in lines:
        svg_parts.append(
            f'<text x="{left_margin}" y="{y_offset}" text-anchor="start" class="hero-line">'
            f"{escape_xml(line)}</text>"
        )
        y_offset += 24

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)
