from __future__ import annotations

import math
from typing import Literal

from .api.schemas.stats import CapabilitiesResponse, Capability, LanguageStats, UserStats
from .themes import Gradient, Theme, get_theme

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


def _get_glow_css(glow_color: str) -> str:
    """Get CSS for glow effects on text and icons with pulse animation.

    Creates an infinite pulsing glow effect that continuously animates.
    This animation runs forever so it's visible regardless of caching.

    Args:
        glow_color: Hex color for the glow effect.

    Returns:
        CSS string with glow filter, pulse animation, and classes.
    """
    # Build multi-line filter for 50% keyframe
    glow_50 = (
        f"drop-shadow(0 0 4px {glow_color}) "
        f"drop-shadow(0 0 8px {glow_color}) "
        f"drop-shadow(0 0 12px {glow_color})"
    )
    return f"""
@keyframes glowPulse {{
  0%, 100% {{
    filter: drop-shadow(0 0 2px {glow_color}) drop-shadow(0 0 4px {glow_color});
  }}
  50% {{
    filter: {glow_50};
  }}
}}
.glow-text {{
  animation: glowPulse 2s ease-in-out infinite;
}}
.glow-icon {{
  filter: drop-shadow(0 0 2px {glow_color});
}}
"""


def _render_gradient_defs(gradient: Gradient, grad_id: str) -> str:
    """Render SVG gradient definition.

    Args:
        gradient: Gradient specification with angle and stops.
        grad_id: Unique ID for the gradient element.

    Returns:
        SVG defs element containing the gradient.
    """
    # Convert angle to x1,y1,x2,y2 coordinates
    # SVG gradients use: angle 0 = left-to-right, 90 = top-to-bottom
    angle_rad = math.radians(gradient["angle"])
    x1 = 50 - 50 * math.cos(angle_rad)
    y1 = 50 - 50 * math.sin(angle_rad)
    x2 = 50 + 50 * math.cos(angle_rad)
    y2 = 50 + 50 * math.sin(angle_rad)

    stops_svg = ""
    for stop in gradient["stops"]:
        stops_svg += f'<stop offset="{stop["offset"]}%" stop-color="{stop["color"]}"/>'

    return f"""<defs>
<linearGradient id="{grad_id}" x1="{x1:.1f}%" y1="{y1:.1f}%" x2="{x2:.1f}%" y2="{y2:.1f}%">
{stops_svg}
</linearGradient>
</defs>"""


def _get_sparkle_css(sparkle_color: str) -> str:
    """Get CSS for sparkle twinkle animation.

    Creates an infinite twinkling effect for sparkle decorations.

    Args:
        sparkle_color: Hex color for sparkles (used in keyframes).

    Returns:
        CSS string with twinkle keyframes and sparkle classes.
    """
    return """
@keyframes twinkle {
  0%, 100% { opacity: 0.3; transform: scale(0.8); }
  50% { opacity: 1; transform: scale(1.2); }
}
.sparkle {
  transform-box: fill-box;
  transform-origin: center;
}
.sparkle-1 { animation: twinkle 1.5s ease-in-out infinite; }
.sparkle-2 { animation: twinkle 2s ease-in-out infinite 0.3s; }
.sparkle-3 { animation: twinkle 1.8s ease-in-out infinite 0.6s; }
.sparkle-4 { animation: twinkle 2.2s ease-in-out infinite 0.9s; }
"""


def _render_sparkles(
    width: int,
    height: int,
    sparkle_color: str,
    sparkle_count: int,
) -> str:
    """Render sparkle/star decorations with twinkle animation.

    Uses deterministic positioning based on index for consistent rendering.
    Each sparkle gets a different animation delay for staggered twinkling.
    Caller must ensure sparkle_count > 0 before calling.

    Args:
        width: Card width in pixels.
        height: Card height in pixels.
        sparkle_color: Hex color for sparkles.
        sparkle_count: Number of sparkles to render (must be > 0).

    Returns:
        SVG group element containing animated sparkle shapes.
    """
    sparkles: list[str] = []

    # Deterministic positions using golden ratio distribution
    phi = 1.618033988749895
    for i in range(sparkle_count):
        # Distribute sparkles using golden ratio for even spread
        t = (i * phi) % 1.0
        # Keep sparkles in margin areas (not over text)
        if i % 2 == 0:
            # Right side sparkles
            x = width - 30 - (t * 40)
            y = 20 + ((i * phi * 1.3) % 1.0) * (height - 40)
        else:
            # Top/bottom edge sparkles
            x = 30 + (t * (width - 100))
            y = 15 + (t * 20) if i % 4 == 1 else height - 15 - (t * 20)

        # Vary sparkle size
        size = 2 + (i % 3)

        # Assign animation class (cycles through 1-4 for varied timing)
        anim_class = f"sparkle sparkle-{(i % 4) + 1}"

        # Four-point star shape with animation class
        sparkle = (
            f'<path class="{anim_class}" '
            f'd="M{x:.1f},{y - size:.1f} L{x + size * 0.3:.1f},{y:.1f} '
            f'L{x:.1f},{y + size:.1f} L{x - size * 0.3:.1f},{y:.1f} Z" '
            f'fill="{sparkle_color}"/>'
        )
        sparkles.append(sparkle)

    return f'<g class="sparkles">{"".join(sparkles)}</g>'


def _render_background(
    width: int,
    height: int,
    theme: Theme,
    hide_border: bool,
    grad_id: str,
) -> tuple[str, str]:
    """Render background rectangle with optional gradient.

    Args:
        width: Card width in pixels.
        height: Card height in pixels.
        theme: Theme configuration.
        hide_border: Whether to hide the border.
        grad_id: Gradient ID if gradient is used.

    Returns:
        Tuple of (defs_svg, rect_svg) where defs contains gradient definition.
    """
    gradient = theme["gradient"]
    border_opacity = 0 if hide_border else 1

    if gradient is not None:
        defs_svg = _render_gradient_defs(gradient, grad_id)
        fill = f"url(#{grad_id})"
    else:
        defs_svg = ""
        fill = theme["bg_color"]

    rect_svg = (
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" '
        f'rx="4.5" fill="{fill}" stroke="{theme["border_color"]}" '
        f'stroke-opacity="{border_opacity}"/>'
    )

    return defs_svg, rect_svg


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

    # Build CSS with optional animations and glow effects
    animation_css = "" if disable_animations else _get_animation_css()
    glow_color = theme["glow_color"]
    glow_css = "" if glow_color is None else _get_glow_css(glow_color)

    # Render background with optional gradient
    defs_svg, rect_svg = _render_background(width, height, theme, hide_border, "stats-grad")

    # Render sparkle decorations with animation CSS
    sparkle_color = theme["sparkle_color"]
    sparkle_count = theme["sparkle_count"]
    sparkles_svg = ""
    sparkle_css = ""
    if sparkle_color is not None and sparkle_count > 0:
        sparkles_svg = _render_sparkles(width, height, sparkle_color, sparkle_count)
        sparkle_css = "" if disable_animations else _get_sparkle_css(sparkle_color)

    # Determine header class (with or without glow)
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
        f"{_escape_xml(stats['name'])}'s GitHub Stats</text>",
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
        width = 495
        height = 170
    elif layout in ("donut", "pie"):
        width = 495
        height = 250
    else:  # default
        width = 495
        height = 145 + len(langs) * 25

    title_color = theme["title_color"]
    text_color = theme["text_color"]
    animation_css = "" if disable_animations else _get_animation_css()
    glow_color = theme["glow_color"]
    glow_css = "" if glow_color is None else _get_glow_css(glow_color)

    # Render background with optional gradient
    defs_svg, rect_svg = _render_background(width, height, theme, hide_border, "langs-grad")

    # Render sparkle decorations with animation CSS
    sparkle_color = theme["sparkle_color"]
    sparkle_count = theme["sparkle_count"]
    sparkles_svg = ""
    sparkle_css = ""
    if sparkle_color is not None and sparkle_count > 0:
        sparkles_svg = _render_sparkles(width, height, sparkle_color, sparkle_count)
        sparkle_css = "" if disable_animations else _get_sparkle_css(sparkle_color)

    # Determine header class (with or without glow)
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
            stagger_idx = min(i + 1, 5)
            stagger_class = "" if disable_animations else f"fade-in stagger-{stagger_idx}"
            svg_parts.append(
                f'<g class="{stagger_class}">'
                f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{lang_color}"/>'
                f'<text x="{lx + 15}" y="{ly}" class="lang-name">{lang_name}</text>'
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

            # Language name
            svg_parts.append(
                f'<g class="{stagger_class}">'
                f'<text x="25" y="{y_offset}" class="lang-name">{_escape_xml(lang["name"])}</text>'
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
    glow_color = theme["glow_color"]
    glow_css = "" if glow_color is None else _get_glow_css(glow_color)

    # Render background with optional gradient
    defs_svg, rect_svg = _render_background(width, height, theme, hide_border, "caps-grad")

    # Render sparkle decorations with animation CSS
    sparkle_color = theme["sparkle_color"]
    sparkle_count = theme["sparkle_count"]
    sparkles_svg = ""
    sparkle_css = ""
    if sparkle_color is not None and sparkle_count > 0:
        sparkles_svg = _render_sparkles(width, height, sparkle_color, sparkle_count)
        sparkle_css = "" if disable_animations else _get_sparkle_css(sparkle_color)

    # Determine header class (with or without glow)
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


def _get_rain_css(rain_color: str, drop_count: int) -> str:
    """Get CSS for falling rain animation.

    Creates an infinite falling rain effect with randomized drops.

    Args:
        rain_color: Hex color for rain drops.
        drop_count: Number of rain drops to animate.

    Returns:
        CSS string with rain animation keyframes and classes.
    """
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

    # Generate unique animation for each drop with different durations/delays
    for i in range(drop_count):
        duration = 1.5 + (i % 5) * 0.3  # 1.5s to 2.7s
        delay = (i * 0.15) % 2.0  # Staggered delays
        css_parts.append(
            f".rain-{i} {{ animation: rainFall {duration:.1f}s linear {delay:.1f}s infinite; }}"
        )

    css_parts.append(f".rain-drop {{ fill: {rain_color}; opacity: 0; }}")

    return "\n".join(css_parts)


def _render_rain_drops(width: int, height: int, rain_color: str, drop_count: int) -> str:
    """Render SVG rain drop elements.

    Args:
        width: Card width.
        height: Card height.
        rain_color: Hex color for drops.
        drop_count: Number of drops.

    Returns:
        SVG elements for rain drops.
    """
    drops = []
    for i in range(drop_count):
        # Distribute drops across width with some randomness via modulo
        x = 10 + ((i * 37) % (width - 20))
        # Vary drop lengths
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
    """Render a full-width hero card with rain animation.

    Args:
        name: Display name (large title).
        subtitle: Subtitle text.
        lines: Info lines to display.
        theme_name: Color theme name.
        disable_animations: Whether to disable animations.

    Returns:
        SVG string.
    """
    theme = get_theme(theme_name)

    width = 495
    # Dynamic height based on content
    base_height = 120  # Name + subtitle
    lines_height = len(lines) * 24
    height = base_height + lines_height + 40

    title_color = theme["title_color"]
    text_color = theme["text_color"]
    glow_color = theme["glow_color"]

    # Rain uses glow color or a default cyan
    rain_color = glow_color if glow_color is not None else "#00fff9"
    drop_count = 25

    # Get animation CSS
    rain_css = "" if disable_animations else _get_rain_css(rain_color, drop_count)
    glow_css = "" if glow_color is None else _get_glow_css(glow_color)

    # Render background with gradient
    defs_svg, rect_svg = _render_background(width, height, theme, False, "hero-grad")

    # Render rain drops
    if disable_animations:
        rain_svg = ""
    else:
        rain_svg = _render_rain_drops(width, height, rain_color, drop_count)

    # Header class with glow
    header_class = "hero-name glow-text" if glow_color is not None else "hero-name"

    # Left-align text with white color for lines
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
        # Name
        f'<text x="{left_margin}" y="50" text-anchor="start" class="{header_class}">'
        f"{_escape_xml(name)}</text>",
    ]

    # Subtitle
    if subtitle:
        svg_parts.append(
            f'<text x="{left_margin}" y="75" text-anchor="start" class="hero-subtitle">'
            f"{_escape_xml(subtitle)}</text>"
        )

    # Info lines
    y_offset = 110
    for line in lines:
        svg_parts.append(
            f'<text x="{left_margin}" y="{y_offset}" text-anchor="start" class="hero-line">'
            f"{_escape_xml(line)}</text>"
        )
        y_offset += 24

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)


_SKILL_COLORS: dict[str, str] = {
    "python": "#3776ab",
    "typescript": "#3178c6",
    "javascript": "#f7df1e",
    "react": "#61dafb",
    "fastapi": "#009688",
    "pytorch": "#ee4c2c",
    "docker": "#2496ed",
    "redis": "#dc382d",
    "postgresql": "#4169e1",
    "postgres": "#4169e1",
    "git": "#f05032",
    "rust": "#dea584",
    "go": "#00add8",
    "java": "#ed8b00",
    "c++": "#00599c",
    "c#": "#512bd4",
    "node": "#339933",
    "nodejs": "#339933",
    "vue": "#4fc08d",
    "angular": "#dd0031",
    "svelte": "#ff3e00",
    "tailwind": "#06b6d4",
    "css": "#1572b6",
    "html": "#e34f26",
    "aws": "#ff9900",
    "azure": "#0078d4",
    "gcp": "#4285f4",
    "kubernetes": "#326ce5",
    "linux": "#fcc624",
    "nginx": "#009639",
    "mongodb": "#47a248",
    "mysql": "#4479a1",
    "graphql": "#e10098",
    "flask": "#000000",
    "django": "#092e20",
}


def _get_skill_color(skill: str) -> str:
    """Get color for a skill name.

    Args:
        skill: Skill name (case-insensitive).

    Returns:
        Hex color string.
    """
    return _SKILL_COLORS.get(skill.lower(), "#888888")


def render_skills_card(
    skills: tuple[str, ...],
    theme_name: str,
    hide_border: bool,
    disable_animations: bool,
) -> str:
    """Render a tech stack/skills card with colored skill icons.

    Args:
        skills: Tuple of skill names.
        theme_name: Color theme name.
        hide_border: Whether to hide border.
        disable_animations: Whether to disable animations.

    Returns:
        SVG string.
    """
    theme = get_theme(theme_name)

    width = 495
    title = "Tech Stack"

    # Calculate layout - skills in rows with icon + text
    skill_height = 32
    skill_margin = 12
    row_height = skill_height + skill_margin
    skills_per_row = 2
    num_rows = (len(skills) + skills_per_row - 1) // skills_per_row
    content_height = num_rows * row_height
    height = 55 + content_height + 15  # title + content + padding

    title_color = theme["title_color"]
    text_color = theme["text_color"]
    border_color = theme["border_color"]
    glow_color = theme["glow_color"]

    # Get animation CSS
    glow_css = "" if glow_color is None or disable_animations else _get_glow_css(glow_color)

    # Render sparkle decorations with animation CSS
    sparkle_color = theme["sparkle_color"]
    sparkle_count = theme["sparkle_count"]
    sparkle_svg = ""
    sparkle_css = ""
    if sparkle_color is not None and sparkle_count > 0:
        sparkle_svg = _render_sparkles(width, height, sparkle_color, sparkle_count)
        sparkle_css = "" if disable_animations else _get_sparkle_css(sparkle_color)

    # Render background
    defs_svg, rect_svg = _render_background(width, height, theme, hide_border, "skills-grad")

    # Border
    border_opacity = "0" if hide_border else "1"
    border_svg = (
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" '
        f'rx="4.5" fill="none" stroke="{border_color}" stroke-opacity="{border_opacity}"/>'
    )

    # Header class with glow
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
        f'<text x="25" y="35" class="{header_class}">{_escape_xml(title)}</text>',
    ]

    # Render skills with colored circle icons
    col_width = (width - 50) // skills_per_row
    start_x = 25
    start_y = 55
    icon_radius = 8

    for i, skill in enumerate(skills):
        row = i // skills_per_row
        col = i % skills_per_row
        x = start_x + col * col_width
        y = start_y + row * row_height

        skill_color = _get_skill_color(skill)

        # Colored circle icon
        circle_x = x + icon_radius + 4
        circle_y = y + skill_height // 2
        svg_parts.append(
            f'<circle cx="{circle_x}" cy="{circle_y}" r="{icon_radius}" fill="{skill_color}"/>'
        )

        # Skill name text
        text_x = circle_x + icon_radius + 10
        text_y = y + skill_height // 2 + 5
        svg_parts.append(
            f'<text x="{text_x}" y="{text_y}" class="skill-name">{_escape_xml(skill)}</text>'
        )

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)


__all__ = [
    "build_capabilities_response",
    "build_language_stats",
    "build_user_stats",
    "render_capabilities_card",
    "render_hero_card",
    "render_langs_card",
    "render_skills_card",
    "render_stats_card",
]
