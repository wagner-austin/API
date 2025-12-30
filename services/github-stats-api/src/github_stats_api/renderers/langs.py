"""Languages card renderer."""

from __future__ import annotations

from ..api.schemas.stats import LanguageStats
from ..themes import get_theme
from ._common import (
    escape_xml,
    get_animation_css,
    get_glow_css,
    get_sparkle_css,
    render_background,
    render_sparkles,
)


def render_langs_card(
    username: str,
    languages: list[LanguageStats],
    total_size: int,
    theme_name: str,
    hide_border: bool,
    layout: str,
    langs_count: int,
    disable_animations: bool,
) -> str:
    """Render top languages SVG card."""
    theme = get_theme(theme_name)

    langs = languages[:langs_count]

    if layout == "compact":
        width = 495
        height = 170
    elif layout in ("donut", "pie"):
        width = 495
        height = 250
    else:
        width = 495
        height = 145 + len(langs) * 25

    title_color = theme["title_color"]
    text_color = theme["text_color"]
    animation_css = "" if disable_animations else get_animation_css()
    glow_color = theme["glow_color"]
    glow_css = "" if glow_color is None else get_glow_css(glow_color)

    defs_svg, rect_svg = render_background(width, height, theme, hide_border, "langs-grad")

    sparkle_color = theme["sparkle_color"]
    sparkle_count = theme["sparkle_count"]
    sparkles_svg = ""
    sparkle_css = ""
    if sparkle_color is not None and sparkle_count > 0:
        sparkles_svg = render_sparkles(width, height, sparkle_color, sparkle_count)
        sparkle_css = "" if disable_animations else get_sparkle_css(sparkle_color)

    header_class = "header glow-text" if glow_color is not None else "header"

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        defs_svg,
        "<style>",
        f".header {{ font: 600 18px 'Segoe UI', sans-serif; fill: {title_color}; }}",
        f".lang-name {{ font: 400 12px 'Segoe UI', sans-serif; fill: {text_color}; }}",
        f".lang-pct {{ font: 400 12px 'Segoe UI', sans-serif; fill: {text_color}; opacity: 0.8; }}",
        animation_css,
        glow_css,
        sparkle_css,
        "</style>",
        rect_svg,
        sparkles_svg,
        f'<text x="25" y="35" class="{header_class}">Most Used Languages</text>',
    ]

    if layout == "compact":
        bar_width = width - 50
        bar_height = 8
        x_offset = 25
        y_offset = 55

        svg_parts.append(
            f'<rect x="{x_offset}" y="{y_offset}" width="{bar_width}" height="{bar_height}" '
            f'rx="2" fill="{theme["border_color"]}"/>'
        )

        current_x: float = float(x_offset)
        for lang in langs:
            seg_width = (lang["percentage"] / 100) * bar_width
            if seg_width > 0:
                anim_class = "" if disable_animations else " grow-width"
                svg_parts.append(
                    f'<rect x="{current_x:.1f}" y="{y_offset}" width="{seg_width:.1f}" '
                    f'height="{bar_height}" fill="{lang["color"]}" class="{anim_class.strip()}"/>'
                )
                current_x += seg_width

        y_offset += 25
        items_per_row = 3
        for i, lang in enumerate(langs):
            col = i % items_per_row
            row = i // items_per_row
            lx = x_offset + col * 100
            ly = y_offset + row * 20

            lang_color = lang["color"]
            lang_name = escape_xml(lang["name"])
            stagger_idx = min(i + 1, 5)
            stagger_class = "" if disable_animations else f"fade-in stagger-{stagger_idx}"
            svg_parts.append(
                f'<g class="{stagger_class}">'
                f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{lang_color}"/>'
                f'<text x="{lx + 15}" y="{ly}" class="lang-name">{lang_name}</text>'
                f"</g>"
            )

    else:
        y_offset = 55
        bar_width = 150

        for idx, lang in enumerate(langs):
            stagger_idx = min(idx + 1, 5)
            stagger_class = "" if disable_animations else f"fade-in stagger-{stagger_idx}"
            bar_anim_class = "" if disable_animations else " grow-width"

            svg_parts.append(
                f'<g class="{stagger_class}">'
                f'<text x="25" y="{y_offset}" class="lang-name">{escape_xml(lang["name"])}</text>'
                f"</g>"
            )

            bar_x = 120
            bar_y = y_offset + 5
            pct_width = (lang["percentage"] / 100) * bar_width
            svg_parts.append(
                f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="8" rx="2" '
                f'fill="{theme["border_color"]}"/>'
            )
            svg_parts.append(
                f'<rect x="{bar_x}" y="{bar_y}" width="{pct_width:.1f}" height="8" rx="2" '
                f'fill="{lang["color"]}" class="{bar_anim_class.strip()}"/>'
            )

            y_offset += 25

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)


def build_language_stats(
    languages: list[dict[str, int | str]],
) -> tuple[list[LanguageStats], int]:
    """Build LanguageStats list from GitHub data."""
    valid_entries: list[tuple[str, int, str]] = []
    for lang in languages:
        name = lang.get("name", "")
        size = lang.get("size", 0)
        color = lang.get("color", "#858585")

        if not isinstance(name, str) or not name:
            continue
        if not isinstance(size, int) or size <= 0:
            continue
        if not isinstance(color, str):
            color = "#858585"

        valid_entries.append((name, size, color))

    total_size = sum(entry[1] for entry in valid_entries)

    if total_size == 0:
        return [], 0

    result: list[LanguageStats] = []
    for name, size, color in valid_entries:
        percentage = (size / total_size) * 100
        result.append(
            {
                "name": name,
                "size": size,
                "percentage": percentage,
                "color": color,
            }
        )

    return result, total_size
