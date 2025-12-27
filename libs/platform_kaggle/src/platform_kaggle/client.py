"""Kaggle API client wrapper.

Provides a typed interface to the Kaggle API for listing and retrieving
competition metadata. Authentication is handled via kaggle.json or
environment variables.
"""

from __future__ import annotations

from .testing import hooks
from .types import (
    Competition,
    CompetitionCategory,
    KaggleApiProtocol,
)

# -----------------------------------------------------------------------------
# Category Mapping
# -----------------------------------------------------------------------------


def _normalize_category(raw_category: str) -> CompetitionCategory:
    """Normalize Kaggle API category string to CompetitionCategory.

    Args:
        raw_category: Raw category string from Kaggle API.

    Returns:
        Normalized CompetitionCategory.
    """
    if raw_category == "Featured":
        return "Featured"
    if raw_category == "Research":
        return "Research"
    if raw_category == "Playground":
        return "Playground"
    if raw_category == "Getting Started":
        return "Getting Started"
    if raw_category == "Masters":
        return "Masters"
    if raw_category == "Kudos":
        return "Kudos"
    # Default to Playground for unknown categories
    return "Playground"


def _to_api_category(category: CompetitionCategory) -> str:
    """Convert CompetitionCategory to Kaggle API category string.

    Args:
        category: CompetitionCategory to convert.

    Returns:
        Kaggle API category string.
    """
    if category == "Featured":
        return "featured"
    if category == "Research":
        return "research"
    if category == "Playground":
        return "playground"
    if category == "Getting Started":
        return "gettingStarted"
    if category == "Masters":
        return "masters"
    # Kudos is the only remaining valid category
    return "kudos"


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
    metadata. Requires authentication via kaggle.json or environment
    variables (KAGGLE_USERNAME, KAGGLE_KEY).

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
        search_str = search if search is not None else ""
        category_str = _to_api_category(category) if category is not None else ""

        response = self._api.competitions_list(
            search=search_str,
            category=category_str,
        )

        # New Kaggle API returns wrapper with .competitions property
        # Items are ApiCompetition objects with attribute access

        result: list[Competition] = []
        for comp in response.competitions:
            # Extract fields using attribute access
            # Kaggle API 1.8.3 returns URLs in ref field, extract slug
            ref_url = str(comp.ref)
            ref = _extract_ref_slug(ref_url)
            title = str(comp.title)
            category_raw = str(comp.category)
            reward = str(comp.reward)
            deadline = str(comp.deadline)
            team_count = int(comp.team_count)
            tags = tuple(str(t.ref) for t in comp.tags)
            description = str(comp.description)
            url = str(comp.url)

            competition = Competition(
                ref=ref,
                title=title,
                category=_normalize_category(category_raw),
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
        response = self._api.competitions_list(search=ref, category="")

        for comp in response.competitions:
            # Kaggle API 1.8.3 returns URLs in ref field, extract slug
            ref_url = str(comp.ref)
            comp_ref = _extract_ref_slug(ref_url)
            if comp_ref == ref:
                return Competition(
                    ref=comp_ref,
                    title=str(comp.title),
                    category=_normalize_category(str(comp.category)),
                    reward=str(comp.reward),
                    deadline=str(comp.deadline),
                    team_count=int(comp.team_count),
                    tags=tuple(str(t.ref) for t in comp.tags),
                    description=str(comp.description),
                    url=str(comp.url),
                )

        return None


__all__ = [
    "KaggleClient",
    "_extract_ref_slug",
    "_normalize_category",
    "_to_api_category",
]
