"""SVG card rendering module.

This module re-exports all rendering functions from the renderers package
for backwards compatibility.
"""

from .renderers import (
    build_capabilities_response,
    build_language_stats,
    build_user_stats,
    render_capabilities_card,
    render_hero_card,
    render_langs_card,
    render_skills_card,
    render_stats_card,
)

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
