from __future__ import annotations

from typing import Literal

from .api.schemas.stats import LanguageStats, UserStats
from .themes import get_theme

# Rank thresholds based on activity percentiles
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
    """Calculate user rank based on activity.

    Args:
        commits: Total commits.
        prs: Total pull requests.
        issues: Total issues.
        stars: Total stars received.

    Returns:
        Tuple of (rank, percentile).
    """
    # Weighted score formula (simplified from github-readme-stats)
    score = commits * 1 + prs * 2 + issues * 1 + stars * 4

    # Logarithmic percentile (higher = better)
    import math

    percentile = 100.0 if score <= 0 else max(0.0, min(100.0, 100.0 - math.log10(score + 1) * 15.0))

    # Find rank using threshold lookup. Last threshold is ("C", 100) and
    # percentile is clamped to [0, 100], so we always find a match.
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


def _escape_xml(text: str) -> str:
    """Escape XML special characters.

    Args:
        text: Raw text.

    Returns:
        XML-escaped text.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _format_number(num: int) -> str:
    """Format number with K/M suffix.

    Args:
        num: Number to format.

    Returns:
        Formatted string.
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num / 1_000:.1f}k"
    return str(num)


def render_stats_card(
    stats: UserStats,
    theme_name: str,
    hide_border: bool,
    show_icons: bool,
    hide: tuple[str, ...],
) -> str:
    """Render user stats SVG card.

    Args:
        stats: User statistics data.
        theme_name: Theme name.
        hide_border: Whether to hide the border.
        show_icons: Whether to show icons.
        hide: Stats to hide.

    Returns:
        SVG string.
    """
    theme = get_theme(theme_name)

    # Card dimensions
    width = 495
    height = 195

    # Build stat items
    items: list[tuple[str, str, str, str]] = []  # (icon, label, value, id)

    if "stars" not in hide:
        items.append(("⭐", "Total Stars", _format_number(stats["total_stars"]), "stars"))
    if "commits" not in hide:
        items.append(("🔥", "Total Commits", _format_number(stats["total_commits"]), "commits"))
    if "prs" not in hide:
        items.append(("🔀", "Total PRs", _format_number(stats["total_prs"]), "prs"))
    if "issues" not in hide:
        items.append(("🐛", "Total Issues", _format_number(stats["total_issues"]), "issues"))
    if "contribs" not in hide:
        contrib_val = _format_number(stats["total_contributions"])
        items.append(("📊", "Contributed to", contrib_val, "contribs"))

    # Adjust height based on items
    height = 120 + len(items) * 25

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        "<style>",
        f".header {{ font: 600 18px 'Segoe UI', sans-serif; fill: {theme['title_color']}; }}",
        f".stat-label {{ font: 400 14px 'Segoe UI', sans-serif; fill: {theme['text_color']}; }}",
        f".stat-value {{ font: 600 14px 'Segoe UI', sans-serif; fill: {theme['text_color']}; }}",
        f".rank {{ font: 800 24px 'Segoe UI', sans-serif; fill: {theme['title_color']}; }}",
        f".rank-circle {{ stroke: {theme['title_color']}; fill: none; stroke-width: 6; }}",
        ".icon { font-size: 14px; }",
        "</style>",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="4.5" '
        f'fill="{theme["bg_color"]}" stroke="{theme["border_color"]}" '
        f'stroke-opacity="{0 if hide_border else 1}"/>',
        f'<text x="25" y="35" class="header">{_escape_xml(stats["name"])}\'s GitHub Stats</text>',
    ]

    # Render stat items
    y_offset = 65
    for icon, label, value, _ in items:
        if show_icons:
            svg_parts.append(f'<text x="25" y="{y_offset}" class="icon">{icon}</text>')
            svg_parts.append(f'<text x="50" y="{y_offset}" class="stat-label">{label}:</text>')
        else:
            svg_parts.append(f'<text x="25" y="{y_offset}" class="stat-label">{label}:</text>')
        svg_parts.append(f'<text x="200" y="{y_offset}" class="stat-value">{value}</text>')
        y_offset += 25

    # Render rank circle
    rank = stats["rank"]
    cx = width - 70
    cy = height // 2
    r = 40
    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" class="rank-circle"/>')
    svg_parts.append(f'<text x="{cx}" y="{cy + 8}" text-anchor="middle" class="rank">{rank}</text>')

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)


def render_langs_card(
    username: str,
    languages: list[LanguageStats],
    total_size: int,
    theme_name: str,
    hide_border: bool,
    layout: str,
    langs_count: int,
) -> str:
    """Render top languages SVG card.

    Args:
        username: GitHub username.
        languages: List of language statistics.
        total_size: Total bytes across all languages.
        theme_name: Theme name.
        hide_border: Whether to hide the border.
        layout: Layout style.
        langs_count: Number of languages to show.

    Returns:
        SVG string.
    """
    theme = get_theme(theme_name)

    # Limit languages
    langs = languages[:langs_count]

    # Card dimensions based on layout
    if layout == "compact":
        width = 350
        height = 170
    elif layout in ("donut", "pie"):
        width = 350
        height = 250
    else:  # default
        width = 300
        height = 145 + len(langs) * 25

    title_color = theme["title_color"]
    text_color = theme["text_color"]
    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        "<style>",
        f".header {{ font: 600 18px 'Segoe UI', sans-serif; fill: {title_color}; }}",
        f".lang-name {{ font: 400 12px 'Segoe UI', sans-serif; fill: {text_color}; }}",
        f".lang-pct {{ font: 400 12px 'Segoe UI', sans-serif; fill: {text_color}; "
        "opacity: 0.8; }}",
        "</style>",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="4.5" '
        f'fill="{theme["bg_color"]}" stroke="{theme["border_color"]}" '
        f'stroke-opacity="{0 if hide_border else 1}"/>',
        '<text x="25" y="35" class="header">Most Used Languages</text>',
    ]

    if layout == "compact":
        # Compact bar layout
        bar_width = width - 50
        bar_height = 8
        x_offset = 25
        y_offset = 55

        # Draw progress bar background
        svg_parts.append(
            f'<rect x="{x_offset}" y="{y_offset}" width="{bar_width}" height="{bar_height}" '
            f'rx="2" fill="{theme["border_color"]}"/>'
        )

        # Draw language segments
        current_x: float = float(x_offset)
        for lang in langs:
            seg_width = (lang["percentage"] / 100) * bar_width
            if seg_width > 0:
                svg_parts.append(
                    f'<rect x="{current_x:.1f}" y="{y_offset}" width="{seg_width:.1f}" '
                    f'height="{bar_height}" fill="{lang["color"]}"/>'
                )
                current_x += seg_width

        # Legend
        y_offset += 25
        items_per_row = 3
        for i, lang in enumerate(langs):
            col = i % items_per_row
            row = i // items_per_row
            lx = x_offset + col * 100
            ly = y_offset + row * 20

            lang_color = lang["color"]
            lang_name = _escape_xml(lang["name"])
            lang_pct = lang["percentage"]
            svg_parts.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{lang_color}"/>')
            svg_parts.append(f'<text x="{lx + 15}" y="{ly}" class="lang-name">{lang_name}</text>')
            svg_parts.append(
                f'<text x="{lx + 15}" y="{ly + 12}" class="lang-pct">{lang_pct:.1f}%</text>'
            )

    else:
        # Default list layout
        y_offset = 55
        bar_width = 150

        for lang in langs:
            # Language name and percentage
            svg_parts.append(
                f'<text x="25" y="{y_offset}" class="lang-name">{_escape_xml(lang["name"])}</text>'
            )
            svg_parts.append(
                f'<text x="{width - 25}" y="{y_offset}" text-anchor="end" class="lang-pct">'
                f"{lang['percentage']:.2f}%</text>"
            )

            # Progress bar
            bar_x = 120
            bar_y = y_offset + 5
            pct_width = (lang["percentage"] / 100) * bar_width
            svg_parts.append(
                f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="8" rx="2" '
                f'fill="{theme["border_color"]}"/>'
            )
            svg_parts.append(
                f'<rect x="{bar_x}" y="{bar_y}" width="{pct_width:.1f}" height="8" rx="2" '
                f'fill="{lang["color"]}"/>'
            )

            y_offset += 25

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)


def build_user_stats(
    data: dict[str, int | str],
) -> UserStats:
    """Build UserStats from GitHub data.

    Args:
        data: Raw data from GitHub client.

    Returns:
        UserStats TypedDict.
    """
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


def build_language_stats(
    languages: list[dict[str, int | str]],
) -> tuple[list[LanguageStats], int]:
    """Build LanguageStats list from GitHub data.

    Args:
        languages: Raw language data from GitHub client.

    Returns:
        Tuple of (list of LanguageStats, total size).
    """
    # First pass: collect valid language entries
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

    # Calculate total from valid entries only
    total_size = sum(entry[1] for entry in valid_entries)

    if total_size == 0:
        return [], 0

    # Build result with percentages
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


__all__ = [
    "build_language_stats",
    "build_user_stats",
    "render_langs_card",
    "render_stats_card",
]
