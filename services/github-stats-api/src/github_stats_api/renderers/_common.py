"""Common utilities for SVG rendering."""

from __future__ import annotations

import math

from ..themes import Gradient, Theme


def escape_xml(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_number(num: int) -> str:
    """Format number with K/M suffix."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num / 1_000:.1f}k"
    return str(num)


def get_animation_css() -> str:
    """Get CSS keyframe animations for SVG cards."""
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


def get_glow_css(glow_color: str) -> str:
    """Get CSS for glow effects with pulse animation."""
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


def render_gradient_defs(gradient: Gradient, grad_id: str) -> str:
    """Render SVG gradient definition."""
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


def get_sparkle_css(sparkle_color: str) -> str:
    """Get CSS for sparkle twinkle animation."""
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


def render_sparkles(
    width: int,
    height: int,
    sparkle_color: str,
    sparkle_count: int,
) -> str:
    """Render sparkle/star decorations with twinkle animation."""
    sparkles: list[str] = []
    phi = 1.618033988749895

    for i in range(sparkle_count):
        t = (i * phi) % 1.0
        if i % 2 == 0:
            x = width - 30 - (t * 40)
            y = 20 + ((i * phi * 1.3) % 1.0) * (height - 40)
        else:
            x = 30 + (t * (width - 100))
            y = 15 + (t * 20) if i % 4 == 1 else height - 15 - (t * 20)

        size = 2 + (i % 3)
        anim_class = f"sparkle sparkle-{(i % 4) + 1}"

        sparkle = (
            f'<path class="{anim_class}" '
            f'd="M{x:.1f},{y - size:.1f} L{x + size * 0.3:.1f},{y:.1f} '
            f'L{x:.1f},{y + size:.1f} L{x - size * 0.3:.1f},{y:.1f} Z" '
            f'fill="{sparkle_color}"/>'
        )
        sparkles.append(sparkle)

    return f'<g class="sparkles">{"".join(sparkles)}</g>'


def render_background(
    width: int,
    height: int,
    theme: Theme,
    hide_border: bool,
    grad_id: str,
) -> tuple[str, str]:
    """Render background rectangle with optional gradient."""
    gradient = theme["gradient"]
    border_opacity = 0 if hide_border else 1

    if gradient is not None:
        defs_svg = render_gradient_defs(gradient, grad_id)
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
