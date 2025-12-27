"""Interest-based competition filtering.

Filters competitions based on user interests including tags,
categories, and reward thresholds.
"""

from __future__ import annotations

import re

from .types import (
    Competition,
    InterestFilter,
)

# -----------------------------------------------------------------------------
# Reward Parsing
# -----------------------------------------------------------------------------


def _parse_reward_amount(reward: str) -> int | None:
    """Parse monetary reward amount from string.

    Args:
        reward: Reward string like "$100,000" or "Knowledge".

    Returns:
        Parsed amount in dollars, or None if not monetary.
    """
    # Check for non-monetary rewards
    reward_lower = reward.lower()
    if reward_lower in ("knowledge", "kudos", "swag", "medals"):
        return None

    # Extract numeric value
    # Handle formats like "$100,000", "€50,000", "100000"
    match = re.search(r"[\d,]+", reward)
    if match:
        amount_str = match.group().replace(",", "")
        return int(amount_str)

    return None


# -----------------------------------------------------------------------------
# Tag Matching
# -----------------------------------------------------------------------------


def _normalize_tag(tag: str) -> str:
    """Normalize tag for comparison.

    Args:
        tag: Tag to normalize.

    Returns:
        Lowercase, hyphen-normalized tag.
    """
    return tag.lower().replace("_", "-").strip()


def _has_any_tag(
    competition: Competition,
    tags: tuple[str, ...],
) -> bool:
    """Check if competition has any of the specified tags.

    Args:
        competition: Competition to check.
        tags: Tags to look for.

    Returns:
        True if competition has at least one matching tag.
    """
    if not tags:
        return True

    comp_tags = {_normalize_tag(t) for t in competition.tags}
    filter_tags = {_normalize_tag(t) for t in tags}

    return bool(comp_tags & filter_tags)


def _has_excluded_tag(
    competition: Competition,
    exclude_tags: tuple[str, ...],
) -> bool:
    """Check if competition has any excluded tags.

    Args:
        competition: Competition to check.
        exclude_tags: Tags that should not be present.

    Returns:
        True if competition has any excluded tag.
    """
    if not exclude_tags:
        return False

    comp_tags = {_normalize_tag(t) for t in competition.tags}
    excluded = {_normalize_tag(t) for t in exclude_tags}

    return bool(comp_tags & excluded)


# -----------------------------------------------------------------------------
# Filter Application
# -----------------------------------------------------------------------------


def _passes_filter(
    competition: Competition,
    filter_: InterestFilter,
) -> bool:
    """Check if competition passes all filter criteria.

    Args:
        competition: Competition to check.
        filter_: Filter criteria to apply.

    Returns:
        True if competition passes all criteria.
    """
    # Check include tags (must have at least one)
    if filter_.include_tags and not _has_any_tag(competition, filter_.include_tags):
        return False

    # Check exclude tags (must not have any)
    if _has_excluded_tag(competition, filter_.exclude_tags):
        return False

    # Check minimum reward
    if filter_.min_reward is not None:
        reward_amount = _parse_reward_amount(competition.reward)
        if reward_amount is None:
            # Non-monetary rewards don't meet minimum
            return False
        if reward_amount < filter_.min_reward:
            return False

    # Check categories - return True if categories is None or competition is in allowed categories
    return filter_.categories is None or competition.category in filter_.categories


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def filter_competitions(
    competitions: tuple[Competition, ...],
    interests: InterestFilter,
) -> tuple[Competition, ...]:
    """Filter competitions by user interests.

    Args:
        competitions: Competitions to filter.
        interests: Interest filter criteria.

    Returns:
        Tuple of competitions matching the filter.
    """
    result: list[Competition] = []

    for comp in competitions:
        if _passes_filter(comp, interests):
            result.append(comp)

    return tuple(result)


def make_interest_filter(
    *,
    include_tags: tuple[str, ...] = (),
    exclude_tags: tuple[str, ...] = (),
    min_reward: int | None = None,
    categories: tuple[str, ...] | None = None,
) -> InterestFilter:
    """Create an interest filter with the given criteria.

    Args:
        include_tags: Tags to include (must have at least one).
        exclude_tags: Tags to exclude (must not have any).
        min_reward: Minimum monetary reward.
        categories: Allowed categories (None = all).

    Returns:
        InterestFilter instance.
    """
    # Convert category strings to CompetitionCategory
    from .types import CompetitionCategory

    validated_categories: tuple[CompetitionCategory, ...] | None = None
    if categories is not None:
        cat_list: list[CompetitionCategory] = []
        for cat in categories:
            if cat == "Featured":
                cat_list.append("Featured")
            elif cat == "Research":
                cat_list.append("Research")
            elif cat == "Playground":
                cat_list.append("Playground")
            elif cat == "Getting Started":
                cat_list.append("Getting Started")
            elif cat == "Masters":
                cat_list.append("Masters")
            elif cat == "Kudos":
                cat_list.append("Kudos")
        validated_categories = tuple(cat_list)

    return InterestFilter(
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        min_reward=min_reward,
        categories=validated_categories,
    )


__all__ = [
    "filter_competitions",
    "make_interest_filter",
]
