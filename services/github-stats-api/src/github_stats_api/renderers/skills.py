"""Skills card renderer."""

from __future__ import annotations

from ..icons import MultiPathIcon, get_skill_icon
from ..themes import get_theme
from ._common import (
    escape_xml,
    get_glow_css,
    get_sparkle_css,
    render_background,
    render_sparkles,
)


def _render_icon(icon: MultiPathIcon, x: float, y: float, size: int) -> str:
    """Render a multi-path icon at the given position and size.

    Args:
        icon: MultiPathIcon with paths, viewBox, and transform.
        x: X position for the icon.
        y: Y position for the icon.
        size: Target size in pixels.

    Returns:
        SVG group element with scaled and positioned icon paths.
    """
    scale = size / icon["viewbox_width"]
    parts = [f'<g transform="translate({x}, {y}) scale({scale})']
    if icon["transform"]:
        parts.append(f" {icon['transform']}")
    parts.append('">')
    for path in icon["paths"]:
        parts.append(f'<path d="{path["d"]}" fill="{path["fill"]}"/>')
    parts.append("</g>")
    return "".join(parts)


def render_skills_card(
    skills: tuple[str, ...],
    theme_name: str,
    hide_border: bool,
    disable_animations: bool,
) -> str:
    """Render a tech stack/skills card with colored skill icons."""
    theme = get_theme(theme_name)

    width = 495
    title = "Tech Stack"

    skill_height = 32
    skill_margin = 12
    row_height = skill_height + skill_margin
    skills_per_row = 2
    num_rows = (len(skills) + skills_per_row - 1) // skills_per_row
    content_height = num_rows * row_height
    height = 55 + content_height + 15

    title_color = theme["title_color"]
    text_color = theme["text_color"]
    border_color = theme["border_color"]
    glow_color = theme["glow_color"]

    glow_css = "" if glow_color is None or disable_animations else get_glow_css(glow_color)

    sparkle_color = theme["sparkle_color"]
    sparkle_count = theme["sparkle_count"]
    sparkle_svg = ""
    sparkle_css = ""
    if sparkle_color is not None and sparkle_count > 0:
        sparkle_svg = render_sparkles(width, height, sparkle_color, sparkle_count)
        sparkle_css = "" if disable_animations else get_sparkle_css(sparkle_color)

    defs_svg, rect_svg = render_background(width, height, theme, hide_border, "skills-grad")

    border_opacity = "0" if hide_border else "1"
    border_svg = (
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" '
        f'rx="4.5" fill="none" stroke="{border_color}" stroke-opacity="{border_opacity}"/>'
    )

    header_class = "header glow-text" if glow_color is not None else "header"

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        defs_svg,
        "<style>",
        f".header {{ font: 600 18px 'Segoe UI', sans-serif; fill: {title_color}; }}",
        f".skill-name {{ font: 600 13px 'Segoe UI', sans-serif; fill: {text_color}; }}",
        glow_css,
        sparkle_css,
        "</style>",
        rect_svg,
        border_svg,
        sparkle_svg,
        f'<text x="25" y="35" class="{header_class}">{escape_xml(title)}</text>',
    ]

    col_width = (width - 50) // skills_per_row
    start_x = 25
    start_y = 55
    icon_size = 18

    for i, skill in enumerate(skills):
        row = i // skills_per_row
        col = i % skills_per_row
        x = start_x + col * col_width
        y = start_y + row * row_height

        icon = get_skill_icon(skill)

        icon_x = x + 4
        icon_y = y + (skill_height - icon_size) // 2

        if icon is not None:
            svg_parts.append(_render_icon(icon, icon_x, icon_y, icon_size))
        else:
            circle_x = icon_x + icon_size // 2
            circle_y = icon_y + icon_size // 2
            svg_parts.append(
                f'<circle cx="{circle_x}" cy="{circle_y}" r="{icon_size // 2}" fill="#888888"/>'
            )

        text_x = icon_x + icon_size + 10
        text_y = y + skill_height // 2 + 5
        svg_parts.append(
            f'<text x="{text_x}" y="{text_y}" class="skill-name">{escape_xml(skill)}</text>'
        )

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)
