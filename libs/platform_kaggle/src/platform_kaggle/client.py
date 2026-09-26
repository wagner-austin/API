"""Kaggle API client wrapper.

Provides a typed interface to the Kaggle API for listing and retrieving
competition metadata. Authentication is handled via the KAGGLE_API_TOKEN
environment variable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from platform_core.members import as_member

from .testing import hooks
from .types import (
    Competition,
    CompetitionCategory,
    KaggleApiProtocol,
)

# -----------------------------------------------------------------------------
# Category Mapping
# -----------------------------------------------------------------------------

# The SDK's valid_competition_categories, keyed by the category each selects.
# Every member but COMMUNITY has one; the test suite pins that key set.
_SDK_CATEGORY_WORDS: Final[Mapping[CompetitionCategory, str]] = {
    CompetitionCategory.FEATURED: "featured",
    CompetitionCategory.RESEARCH: "research",
    CompetitionCategory.RECRUITMENT: "recruitment",
    CompetitionCategory.GETTING_STARTED: "gettingStarted",
    CompetitionCategory.MASTERS: "masters",
    CompetitionCategory.PLAYGROUND: "playground",
}


def _to_api_category(category: CompetitionCategory) -> str:
    """Convert CompetitionCategory to the word ``competitions_list`` filters on.

    The words are the SDK's ``valid_competition_categories``; it refuses any
    other with ValueError.

    Args:
        category: CompetitionCategory to convert.

    Returns:
        Kaggle API category string.

    Raises:
        ValueError: For COMMUNITY, which the SDK lists as a group rather than
            a category, so no category word selects it.
    """
    if category is CompetitionCategory.COMMUNITY:
        raise ValueError(
            "Kaggle lists community competitions as a group, not a category; "
            "competitions_list has no category word for 'Community'"
        )
    return _SDK_CATEGORY_WORDS[category]


def _extract_ref_slug(url: str) -> str:
    """Extract competition slug from Kaggle URL.

    Kaggle API 1.8.3 returns full URLs in the ref field:
    'https://www.kaggle.com/competitions/gemini-3'

    This function extracts the slug ('gemini-3').

    Args:
        url: Full Kaggle competition URL.

    Returns:
        Competition slug.

    Raises:
        ValueError: If URL format is invalid.
    """
    marker = "/competitions/"
    idx = url.find(marker)
    if idx == -1:
        raise ValueError(f"Invalid Kaggle competition URL: {url}")
    slug = url[idx + len(marker) :]
    return slug.rstrip("/")


# -----------------------------------------------------------------------------
# KaggleClient Implementation
# -----------------------------------------------------------------------------


class KaggleClient:
    """Kaggle API client for competition discovery.

    Wraps the Kaggle Python API to provide typed access to competition
    metadata. Requires authentication via KAGGLE_API_TOKEN environment
    variable.

    Attributes:
        _api: Authenticated Kaggle API instance.
    """

    __slots__ = ("_api",)

    def __init__(self) -> None:
        """Initialize Kaggle client with authentication."""
        factory = hooks.kaggle_api_factory
        self._api: KaggleApiProtocol = factory()

    def list_competitions(
        self,
        *,
        search: str | None = None,
        category: CompetitionCategory | None = None,
    ) -> tuple[Competition, ...]:
        """List active competitions with optional filters.

        Args:
            search: Optional search query to filter by title/description.
            category: Optional category filter.

        Returns:
            Tuple of matching competitions.
        """
        category_str = _to_api_category(category) if category is not None else None

        response = self._api.competitions_list(
            search=search,
            category=category_str,
        )

        # API can return None
        if response is None:
            return ()

        # New Kaggle API returns wrapper with .competitions property
        # Items are ApiCompetition objects with attribute access
        # Both the list and individual items can be None

        result: list[Competition] = []
        competitions = response.competitions
        if competitions is None:
            return ()

        for comp in competitions:
            if comp is None:
                continue

            # Extract fields using attribute access
            # Kaggle API 1.8.3 returns URLs in ref field, extract slug
            ref_url = str(comp.ref)
            ref = _extract_ref_slug(ref_url)
            title = str(comp.title)
            category_raw = str(comp.category)
            reward = str(comp.reward)
            deadline = str(comp.deadline)
            team_count = int(comp.team_count)

            # Tags can be None or contain None items
            raw_tags = comp.tags
            if raw_tags is None:
                tags: tuple[str, ...] = ()
            else:
                tags = tuple(str(t.ref) for t in raw_tags if t is not None)

            description = str(comp.description)
            url = str(comp.url)

            competition = Competition(
                ref=ref,
                title=title,
                category=as_member(category_raw, "category", CompetitionCategory),
                reward=reward,
                deadline=deadline,
                team_count=team_count,
                tags=tags,
                description=description,
                url=url,
            )
            result.append(competition)

        return tuple(result)

    def get_competition(self, ref: str) -> Competition | None:
        """Get a specific competition by ref.

        Args:
            ref: Competition reference slug (e.g., "amex-default-prediction").

        Returns:
            Competition if found, None otherwise.
        """
        # Search for the competition by ref
        response = self._api.competitions_list(search=ref)

        if response is None:
            return None

        competitions = response.competitions
        if competitions is None:
            return None

        for comp in competitions:
            if comp is None:
                continue

            # Kaggle API 1.8.3 returns URLs in ref field, extract slug
            ref_url = str(comp.ref)
            comp_ref = _extract_ref_slug(ref_url)
            if comp_ref == ref:
                # Tags can be None or contain None items
                raw_tags = comp.tags
                if raw_tags is None:
                    tags: tuple[str, ...] = ()
                else:
                    tags = tuple(str(t.ref) for t in raw_tags if t is not None)

                return Competition(
                    ref=comp_ref,
                    title=str(comp.title),
                    category=as_member(str(comp.category), "category", CompetitionCategory),
                    reward=str(comp.reward),
                    deadline=str(comp.deadline),
                    team_count=int(comp.team_count),
                    tags=tags,
                    description=str(comp.description),
                    url=str(comp.url),
                )

        return None


__all__ = [
    "KaggleClient",
    "_extract_ref_slug",
    "_to_api_category",
]
