"""Interest-based filtering for hackathons.

This module provides functions to filter hackathons by user interests
including themes, states, and featured status.
"""

from __future__ import annotations

from platform_devpost.types import (
    Hackathon,
    InterestFilter,
)


def _hackathon_has_theme(hackathon: Hackathon, theme_name: str) -> bool:
    """Check if hackathon has a theme matching the given name.

    Args:
        hackathon: Hackathon to check.
        theme_name: Theme name to match (case-insensitive).

    Returns:
        True if hackathon has matching theme.
    """
    theme_lower = theme_name.lower()
    return any(theme_lower in theme.name.lower() for theme in hackathon.themes)


def _matches_include_themes(hackathon: Hackathon, include_themes: tuple[str, ...]) -> bool:
    """Check if hackathon matches any include theme.

    Args:
        hackathon: Hackathon to check.
        include_themes: Tuple of theme names to include.

    Returns:
        True if hackathon has at least one matching theme, or if include_themes is empty.
    """
    if len(include_themes) == 0:
        return True

    return any(_hackathon_has_theme(hackathon, theme_name) for theme_name in include_themes)


def _matches_exclude_themes(hackathon: Hackathon, exclude_themes: tuple[str, ...]) -> bool:
    """Check if hackathon should be excluded based on themes.

    Args:
        hackathon: Hackathon to check.
        exclude_themes: Tuple of theme names to exclude.

    Returns:
        True if hackathon does NOT have any excluded themes.
    """
    return all(not _hackathon_has_theme(hackathon, theme_name) for theme_name in exclude_themes)


def filter_hackathons(
    hackathons: tuple[Hackathon, ...],
    interests: InterestFilter,
) -> tuple[Hackathon, ...]:
    """Filter hackathons by user interests.

    Args:
        hackathons: Tuple of hackathons to filter.
        interests: Interest filter configuration.

    Returns:
        Tuple of hackathons matching all filter criteria.
    """
    result: list[Hackathon] = []

    for hackathon in hackathons:
        # Check include themes
        if not _matches_include_themes(hackathon, interests.include_themes):
            continue

        # Check exclude themes
        if not _matches_exclude_themes(hackathon, interests.exclude_themes):
            continue

        # Check states
        if interests.states is not None and hackathon.open_state not in interests.states:
            continue

        # Check featured
        if interests.featured_only and not hackathon.featured:
            continue

        result.append(hackathon)

    return tuple(result)
