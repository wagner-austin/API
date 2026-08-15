"""SVG icon paths for skill icons.

The icon table is grouped across the private _icons_* modules by what each
icon depicts; this module merges them into one mapping, applies the aliases,
and exposes the lookup. Icon paths come from Simple Icons
(https://simpleicons.org/).
"""

from __future__ import annotations

from github_stats_api._icon_types import IconPath, MultiPathIcon
from github_stats_api._icons_backend import BACKEND_ICONS
from github_stats_api._icons_data import DATA_ICONS
from github_stats_api._icons_frontend import FRONTEND_ICONS
from github_stats_api._icons_infrastructure import INFRASTRUCTURE_ICONS
from github_stats_api._icons_languages import LANGUAGES_ICONS

SKILL_ICONS: dict[str, MultiPathIcon] = {
    **LANGUAGES_ICONS,
    **FRONTEND_ICONS,
    **BACKEND_ICONS,
    **DATA_ICONS,
    **INFRASTRUCTURE_ICONS,
}

# Aliases
SKILL_ICONS["transformers"] = SKILL_ICONS["huggingface"]
SKILL_ICONS["node"] = SKILL_ICONS["nodejs"]
SKILL_ICONS["postgres"] = SKILL_ICONS["postgresql"]
SKILL_ICONS["tailwindcss"] = SKILL_ICONS["tailwind"]
SKILL_ICONS["sklearn"] = SKILL_ICONS["scikitlearn"]
SKILL_ICONS["scikit-learn"] = SKILL_ICONS["scikitlearn"]
SKILL_ICONS["apachekafka"] = SKILL_ICONS["kafka"]
SKILL_ICONS["traefikproxy"] = SKILL_ICONS["traefik"]


def get_skill_icon(skill: str) -> MultiPathIcon | None:
    """Get SVG icon data for a skill.

    Args:
        skill: Skill name (case-insensitive).

    Returns:
        MultiPathIcon with paths, viewBox, and transform, or None if no icon available.
    """
    return SKILL_ICONS.get(skill.lower())


__all__ = ["SKILL_ICONS", "IconPath", "MultiPathIcon", "get_skill_icon"]
