"""Stats card renderer."""

from __future__ import annotations

import math
from typing import Literal

from ..api.schemas.stats import UserStats
from ..themes import get_theme
from ._common import (
    escape_xml,
    format_number,
    get_animation_css,
    get_glow_css,
    get_sparkle_css,
    render_background,
    render_sparkles,
)

_RANK_THRESHOLDS: list[tuple[str, float]] = [
    ("S+", 1),
    ("S", 12.5),
    ("A+", 25),
    ("A", 37.5),
    ("B+", 50),
    ("B", 62.5),
    ("C", 100),
]


def _calculate_rank(
    commits: int,
    prs: int,
    issues: int,
    stars: int,
) -> tuple[Literal["S+", "S", "A+", "A", "B+", "B", "C"], float]:
    """Calculate user rank based on activity."""
    score = commits * 1 + prs * 2 + issues * 1 + stars * 4
    percentile = 100.0 if score <= 0 else max(0.0, min(100.0, 100.0 - math.log10(score + 1) * 15.0))

    if percentile <= 1:
        return "S+", percentile
    if percentile <= 12.5:
        return "S", percentile
    if percentile <= 25:
        return "A+", percentile
    if percentile <= 37.5:
        return "A", percentile
    if percentile <= 50:
        return "B+", percentile
    if percentile <= 62.5:
        return "B", percentile
    return "C", percentile


def render_stats_card(
    stats: UserStats,
    theme_name: str,
    hide_border: bool,
    show_icons: bool,
    hide: tuple[str, ...],
    disable_animations: bool,
) -> str:
    """Render user stats SVG card."""
    theme = get_theme(theme_name)

    width = 495
    height = 195

    items: list[tuple[str, str, str, str]] = []

    if "stars" not in hide:
        items.append(("⭐", "Total Stars", format_number(stats["total_stars"]), "stars"))
    if "commits" not in hide:
        items.append(("🔥", "Total Commits", format_number(stats["total_commits"]), "commits"))
    if "prs" not in hide:
        items.append(("🔀", "Total PRs", format_number(stats["total_prs"]), "prs"))
    if "issues" not in hide:
        items.append(("🐛", "Total Issues", format_number(stats["total_issues"]), "issues"))
    if "contribs" not in hide:
        contrib_val = format_number(stats["total_contributions"])
        items.append(("📊", "Contributed to", contrib_val, "contribs"))

    height = 120 + len(items) * 25

    animation_css = "" if disable_animations else get_animation_css()
    glow_color = theme["glow_color"]
    glow_css = "" if glow_color is None else get_glow_css(glow_color)

    defs_svg, rect_svg = render_background(width, height, theme, hide_border, "stats-grad")

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
        f".header {{ font: 600 18px 'Segoe UI', sans-serif; fill: {theme['title_color']}; }}",
        f".stat-label {{ font: 400 14px 'Segoe UI', sans-serif; fill: {theme['text_color']}; }}",
        f".stat-value {{ font: 600 14px 'Segoe UI', sans-serif; fill: {theme['text_color']}; }}",
        f".rank {{ font: 800 24px 'Segoe UI', sans-serif; fill: {theme['title_color']}; }}",
        f".rank-circle {{ stroke: {theme['title_color']}; fill: none; stroke-width: 6; }}",
        ".icon { font-size: 14px; }",
        animation_css,
        glow_css,
        sparkle_css,
        "</style>",
        rect_svg,
        sparkles_svg,
        f'<text x="25" y="35" class="{header_class}">'
        f"{escape_xml(stats['name'])}'s GitHub Stats</text>",
    ]

    y_offset = 65
    for idx, (icon, label, value, _) in enumerate(items):
        stagger_class = "" if disable_animations else f" fade-in stagger-{idx + 1}"
        if show_icons:
            svg_parts.append(
                f'<g class="{stagger_class.strip()}">'
                f'<text x="25" y="{y_offset}" class="icon">{icon}</text>'
                f'<text x="50" y="{y_offset}" class="stat-label">{label}:</text>'
                f'<text x="200" y="{y_offset}" class="stat-value">{value}</text>'
                f"</g>"
            )
        else:
            svg_parts.append(
                f'<g class="{stagger_class.strip()}">'
                f'<text x="25" y="{y_offset}" class="stat-label">{label}:</text>'
                f'<text x="200" y="{y_offset}" class="stat-value">{value}</text>'
                f"</g>"
            )
        y_offset += 25

    rank = stats["rank"]
    cx = width - 70
    cy = height // 2
    r = 40
    circle_class = "rank-circle" if disable_animations else "rank-circle rank-circle-anim"
    rank_text_class = "" if disable_animations else "fade-in stagger-5"
    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" class="{circle_class}"/>')
    svg_parts.append(
        f'<text x="{cx}" y="{cy + 8}" text-anchor="middle" '
        f'class="rank {rank_text_class}">{rank}</text>'
    )

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)


def build_user_stats(
    data: dict[str, int | str],
) -> UserStats:
    """Build UserStats from GitHub data."""
    commits = data.get("total_commits", 0)
    prs = data.get("total_prs", 0)
    issues = data.get("total_issues", 0)
    stars = data.get("total_stars", 0)

    if not isinstance(commits, int):
        commits = 0
    if not isinstance(prs, int):
        prs = 0
    if not isinstance(issues, int):
        issues = 0
    if not isinstance(stars, int):
        stars = 0

    rank, percentile = _calculate_rank(commits, prs, issues, stars)

    name = data.get("name", "")
    if not isinstance(name, str):
        name = ""
    login = data.get("login", "")
    if not isinstance(login, str):
        login = ""
    total_contributions = data.get("total_contributions", 0)
    if not isinstance(total_contributions, int):
        total_contributions = 0

    return {
        "username": login,
        "name": name or login,
        "total_commits": commits,
        "total_prs": prs,
        "total_issues": issues,
        "total_stars": stars,
        "total_contributions": total_contributions,
        "rank": rank,
        "rank_percentile": percentile,
    }
