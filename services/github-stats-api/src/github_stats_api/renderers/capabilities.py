"""Capabilities card renderer."""

from __future__ import annotations

from ..api.schemas.stats import CapabilitiesResponse, Capability
from ..themes import get_theme
from ._common import (
    escape_xml,
    get_animation_css,
    get_glow_css,
    get_sparkle_css,
    render_background,
    render_sparkles,
)


def render_capabilities_card(
    response: CapabilitiesResponse,
    theme_name: str,
    hide_border: bool,
    disable_animations: bool,
) -> str:
    """Render codebase capabilities SVG card."""
    theme = get_theme(theme_name)

    capabilities = response["capabilities"]
    ml_backends = response["ml_backends"]
    task_types = response["task_types"]

    width = 495
    cap_rows = (len(capabilities) + 1) // 2
    backend_row_count = 1 if ml_backends else 0
    task_row_count = 1 if task_types else 0
    height = 80 + cap_rows * 35 + backend_row_count * 35 + task_row_count * 35

    title_color = theme["title_color"]
    text_color = theme["text_color"]
    icon_color = theme["icon_color"]
    animation_css = "" if disable_animations else get_animation_css()
    glow_color = theme["glow_color"]
    glow_css = "" if glow_color is None else get_glow_css(glow_color)

    defs_svg, rect_svg = render_background(width, height, theme, hide_border, "caps-grad")

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
        f".section {{ font: 600 14px 'Segoe UI', sans-serif; fill: {title_color}; }}",
        f".cap-name {{ font: 400 12px 'Segoe UI', sans-serif; fill: {text_color}; }}",
        f".cap-desc {{ font: 400 10px 'Segoe UI', sans-serif; fill: {text_color}; "
        "opacity: 0.7; }}",
        f".tag {{ font: 400 10px 'Segoe UI', sans-serif; fill: {icon_color}; }}",
        ".strength-strong { fill: #2ecc71; }",
        ".strength-moderate { fill: #f1c40f; }",
        ".strength-basic { fill: #95a5a6; }",
        animation_css,
        glow_css,
        sparkle_css,
        "</style>",
        rect_svg,
        sparkles_svg,
        f'<text x="25" y="35" class="{header_class}">Codebase Capabilities</text>',
    ]

    y_offset = 60

    if capabilities:
        for i, cap in enumerate(capabilities):
            col = i % 2
            row = i // 2
            x = 25 + col * 235
            y = y_offset + row * 35

            stagger_idx = min(i + 1, 5)
            stagger_class = "" if disable_animations else f"fade-in stagger-{stagger_idx}"

            strength_class = f"strength-{cap['strength']}"
            scale_class = "" if disable_animations else " scale-in"

            svg_parts.append(
                f'<g class="{stagger_class}">'
                f'<circle cx="{x + 5}" cy="{y - 4}" r="4" class="{strength_class}{scale_class}"/>'
                f'<text x="{x + 15}" y="{y}" class="cap-name">'
                f"{escape_xml(cap['name'].replace('_', ' ').title())}</text>"
                f'<text x="{x + 15}" y="{y + 12}" class="cap-desc">{cap["strength"]}</text>'
                f"</g>"
            )

        y_offset += cap_rows * 35 + 10

    if ml_backends:
        backend_class = "" if disable_animations else "fade-in stagger-3"
        svg_parts.append(
            f'<g class="{backend_class}">'
            f'<text x="25" y="{y_offset}" class="section">ML Backends</text>'
        )
        y_offset += 18
        backends_str = ", ".join(escape_xml(b) for b in ml_backends)
        svg_parts.append(f'<text x="25" y="{y_offset}" class="tag">{backends_str}</text></g>')
        y_offset += 25

    if task_types:
        task_class = "" if disable_animations else "fade-in stagger-4"
        svg_parts.append(
            f'<g class="{task_class}"><text x="25" y="{y_offset}" class="section">Task Types</text>'
        )
        y_offset += 18
        tasks_formatted = [t.replace("_", " ").title() for t in task_types]
        tasks_str = ", ".join(escape_xml(t) for t in tasks_formatted[:6])
        if len(task_types) > 6:
            tasks_str += f" +{len(task_types) - 6} more"
        svg_parts.append(f'<text x="25" y="{y_offset}" class="tag">{tasks_str}</text></g>')

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)


def build_capabilities_response(
    repo: str,
    capabilities: tuple[Capability, ...],
    ml_backends: tuple[str, ...],
    frameworks: tuple[str, ...],
    data_formats: tuple[str, ...],
    task_types: tuple[str, ...],
) -> CapabilitiesResponse:
    """Build CapabilitiesResponse from profile data."""
    return {
        "repo": repo,
        "capabilities": capabilities,
        "ml_backends": ml_backends,
        "frameworks": frameworks,
        "data_formats": data_formats,
        "task_types": task_types,
    }
