from __future__ import annotations

from typing import Literal

from .api.schemas.stats import CapabilitiesResponse, Capability, LanguageStats, UserStats
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


def _get_animation_css() -> str:
    """Get CSS keyframe animations for SVG cards.

    Returns:
        CSS string with animation keyframes and classes.
    """
    return """
@keyframes fadeInAnimation {
  0% { opacity: 0; }
  100% { opacity: 1; }
}
@keyframes growWidthAnimation {
  0% { transform: scaleX(0); }
  100% { transform: scaleX(1); }
}
@keyframes scaleInAnimation {
  0% { transform: scale(0); }
  100% { transform: scale(1); }
}
@keyframes rankCircleAnimation {
  0% { stroke-dashoffset: 251.32; }
  100% { stroke-dashoffset: 0; }
}
.fade-in { animation: fadeInAnimation 0.3s ease-in-out forwards; }
.stagger-1 { animation-delay: 0.1s; opacity: 0; }
.stagger-2 { animation-delay: 0.2s; opacity: 0; }
.stagger-3 { animation-delay: 0.3s; opacity: 0; }
.stagger-4 { animation-delay: 0.4s; opacity: 0; }
.stagger-5 { animation-delay: 0.5s; opacity: 0; }
.grow-width { animation: growWidthAnimation 0.6s ease-in-out forwards; transform-origin: left; }
.scale-in { animation: scaleInAnimation 0.4s ease-in-out forwards; transform-origin: center; }
.rank-circle-anim {
  stroke-dasharray: 251.32;
  stroke-dashoffset: 251.32;
  animation: rankCircleAnimation 1s ease-in-out forwards;
  animation-delay: 0.5s;
}
"""


def render_stats_card(
    stats: UserStats,
    theme_name: str,
    hide_border: bool,
    show_icons: bool,
    hide: tuple[str, ...],
    disable_animations: bool,
) -> str:
    """Render user stats SVG card.

    Args:
        stats: User statistics data.
        theme_name: Theme name.
        hide_border: Whether to hide the border.
        show_icons: Whether to show icons.
        hide: Stats to hide.
        disable_animations: Whether to disable CSS animations.

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

    # Build CSS with optional animations
    animation_css = "" if disable_animations else _get_animation_css()

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
        animation_css,
        "</style>",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="4.5" '
        f'fill="{theme["bg_color"]}" stroke="{theme["border_color"]}" '
        f'stroke-opacity="{0 if hide_border else 1}"/>',
        f'<text x="25" y="35" class="header">{_escape_xml(stats["name"])}\'s GitHub Stats</text>',
    ]

    # Render stat items with staggered animation
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

    # Render rank circle with animation
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
    """Render top languages SVG card.

    Args:
        username: GitHub username.
        languages: List of language statistics.
        total_size: Total bytes across all languages.
        theme_name: Theme name.
        hide_border: Whether to hide the border.
        layout: Layout style.
        langs_count: Number of languages to show.
        disable_animations: Whether to disable CSS animations.

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
    animation_css = "" if disable_animations else _get_animation_css()

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        "<style>",
        f".header {{ font: 600 18px 'Segoe UI', sans-serif; fill: {title_color}; }}",
        f".lang-name {{ font: 400 12px 'Segoe UI', sans-serif; fill: {text_color}; }}",
        f".lang-pct {{ font: 400 12px 'Segoe UI', sans-serif; fill: {text_color}; "
        "opacity: 0.8; }}",
        animation_css,
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

        # Draw language segments with animation
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

        # Legend with staggered fade-in
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
            stagger_idx = min(i + 1, 5)
            stagger_class = "" if disable_animations else f"fade-in stagger-{stagger_idx}"
            svg_parts.append(
                f'<g class="{stagger_class}">'
                f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{lang_color}"/>'
                f'<text x="{lx + 15}" y="{ly}" class="lang-name">{lang_name}</text>'
                f'<text x="{lx + 15}" y="{ly + 12}" class="lang-pct">{lang_pct:.1f}%</text>'
                f"</g>"
            )

    else:
        # Default list layout
        y_offset = 55
        bar_width = 150

        for idx, lang in enumerate(langs):
            stagger_idx = min(idx + 1, 5)
            stagger_class = "" if disable_animations else f"fade-in stagger-{stagger_idx}"
            bar_anim_class = "" if disable_animations else " grow-width"

            # Language name and percentage
            svg_parts.append(
                f'<g class="{stagger_class}">'
                f'<text x="25" y="{y_offset}" class="lang-name">{_escape_xml(lang["name"])}</text>'
                f'<text x="{width - 25}" y="{y_offset}" text-anchor="end" class="lang-pct">'
                f"{lang['percentage']:.2f}%</text>"
                f"</g>"
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
                f'fill="{lang["color"]}" class="{bar_anim_class.strip()}"/>'
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


def render_capabilities_card(
    response: CapabilitiesResponse,
    theme_name: str,
    hide_border: bool,
    disable_animations: bool,
) -> str:
    """Render codebase capabilities SVG card.

    Args:
        response: Capabilities response data.
        theme_name: Theme name.
        hide_border: Whether to hide the border.
        disable_animations: Whether to disable CSS animations.

    Returns:
        SVG string.
    """
    theme = get_theme(theme_name)

    capabilities = response["capabilities"]
    ml_backends = response["ml_backends"]
    task_types = response["task_types"]

    # Card dimensions - dynamic based on content
    width = 495
    # Header + capabilities + ML backends + task types + padding
    cap_rows = (len(capabilities) + 1) // 2  # 2 per row
    backend_row_count = 1 if ml_backends else 0
    task_row_count = 1 if task_types else 0
    height = 80 + cap_rows * 35 + backend_row_count * 35 + task_row_count * 35

    title_color = theme["title_color"]
    text_color = theme["text_color"]
    icon_color = theme["icon_color"]
    animation_css = "" if disable_animations else _get_animation_css()

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
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
        "</style>",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="4.5" '
        f'fill="{theme["bg_color"]}" stroke="{theme["border_color"]}" '
        f'stroke-opacity="{0 if hide_border else 1}"/>',
        '<text x="25" y="35" class="header">Codebase Capabilities</text>',
    ]

    y_offset = 60

    # Render capabilities (2 per row) with staggered animation
    if capabilities:
        for i, cap in enumerate(capabilities):
            col = i % 2
            row = i // 2
            x = 25 + col * 235
            y = y_offset + row * 35

            stagger_idx = min(i + 1, 5)
            stagger_class = "" if disable_animations else f"fade-in stagger-{stagger_idx}"

            # Strength indicator
            strength_class = f"strength-{cap['strength']}"
            scale_class = "" if disable_animations else " scale-in"

            svg_parts.append(
                f'<g class="{stagger_class}">'
                f'<circle cx="{x + 5}" cy="{y - 4}" r="4" class="{strength_class}{scale_class}"/>'
                f'<text x="{x + 15}" y="{y}" class="cap-name">'
                f"{_escape_xml(cap['name'].replace('_', ' ').title())}</text>"
                f'<text x="{x + 15}" y="{y + 12}" class="cap-desc">{cap["strength"]}</text>'
                f"</g>"
            )

        y_offset += cap_rows * 35 + 10

    # Render ML backends with fade-in
    if ml_backends:
        backend_class = "" if disable_animations else "fade-in stagger-3"
        svg_parts.append(
            f'<g class="{backend_class}">'
            f'<text x="25" y="{y_offset}" class="section">ML Backends</text>'
        )
        y_offset += 18
        backends_str = ", ".join(_escape_xml(b) for b in ml_backends)
        svg_parts.append(f'<text x="25" y="{y_offset}" class="tag">{backends_str}</text></g>')
        y_offset += 25

    # Render task types with fade-in
    if task_types:
        task_class = "" if disable_animations else "fade-in stagger-4"
        svg_parts.append(
            f'<g class="{task_class}"><text x="25" y="{y_offset}" class="section">Task Types</text>'
        )
        y_offset += 18
        # Format task types nicely
        tasks_formatted = [t.replace("_", " ").title() for t in task_types]
        tasks_str = ", ".join(_escape_xml(t) for t in tasks_formatted[:6])
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
    """Build CapabilitiesResponse from profile data.

    Args:
        repo: GitHub repository string.
        capabilities: Tuple of detected capabilities.
        ml_backends: Tuple of ML backend names.
        frameworks: Tuple of framework names.
        data_formats: Tuple of data format names.
        task_types: Tuple of task type names.

    Returns:
        CapabilitiesResponse TypedDict.
    """
    return {
        "repo": repo,
        "capabilities": capabilities,
        "ml_backends": ml_backends,
        "frameworks": frameworks,
        "data_formats": data_formats,
        "task_types": task_types,
    }


__all__ = [
    "build_capabilities_response",
    "build_language_stats",
    "build_user_stats",
    "render_capabilities_card",
    "render_langs_card",
    "render_stats_card",
]
