"""Core types for platform_kaggle.

This module defines Kaggle-specific types and re-exports shared types
from platform_codebase.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from platform_codebase import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    LibInfo,
    MatchRecommendation,
    ServiceInfo,
    decode_capability,
    decode_profile,
    encode_capability,
    encode_profile,
    require_recommendation,
    require_strength,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_int,
    require_float,
    require_int,
    require_list,
    require_str,
)

# -----------------------------------------------------------------------------
# Re-exports from platform_codebase
# -----------------------------------------------------------------------------

__all__ = [
    "CapabilityStrength",
    "CodebaseCapability",
    "CodebaseProfile",
    "Competition",
    "CompetitionCategory",
    "CompetitionMatch",
    "CompetitionPage",
    "CompetitionPages",
    "CompetitionsResponseProtocol",
    "InterestFilter",
    "KaggleApiClassProtocol",
    "KaggleApiFactoryProtocol",
    "KaggleApiProtocol",
    "KaggleClientProtocol",
    "KaggleCompetitionProtocol",
    "KaggleModuleProtocol",
    "KagglePageFetcherProtocol",
    "KagglePreAuthModuleProtocol",
    "KaggleTagProtocol",
    "LibInfo",
    "MatchRecommendation",
    "ServiceInfo",
    "decode_capability",
    "decode_competition",
    "decode_competition_page",
    "decode_competition_pages",
    "decode_filter",
    "decode_match",
    "decode_profile",
    "encode_capability",
    "encode_competition",
    "encode_competition_page",
    "encode_competition_pages",
    "encode_filter",
    "encode_match",
    "encode_profile",
    "require_recommendation",
    "require_strength",
]


# -----------------------------------------------------------------------------
# Kaggle-Specific Literal Types
# -----------------------------------------------------------------------------

CompetitionCategory = Literal[
    "Featured", "Research", "Playground", "Getting Started", "Masters", "Kudos"
]


# -----------------------------------------------------------------------------
# Internal Validation Helpers
# -----------------------------------------------------------------------------


def _require_list_str(obj: JSONObject, key: str) -> list[str]:
    """Extract required list of strings from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        List of strings.

    Raises:
        JSONTypeError: If field is missing or contains non-strings.
    """
    items = require_list(obj, key)
    result: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise JSONTypeError(f"Field '{key}[{i}]' must be a string, got {type(item).__name__}")
        result.append(item)
    return result


def _require_dict_value(value: JSONValue, context: str) -> JSONObject:
    """Require value to be a dict.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        The value as JSONObject.

    Raises:
        JSONTypeError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{context} must be an object, got {type(value).__name__}")
    return value


def _require_category(obj: JSONObject, key: str) -> CompetitionCategory:
    """Extract and validate CompetitionCategory from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        Validated CompetitionCategory.

    Raises:
        JSONTypeError: If field is missing or not a valid category.
    """
    value = require_str(obj, key)
    if value == "Featured":
        return "Featured"
    if value == "Research":
        return "Research"
    if value == "Playground":
        return "Playground"
    if value == "Getting Started":
        return "Getting Started"
    if value == "Masters":
        return "Masters"
    if value == "Kudos":
        return "Kudos"
    raise JSONTypeError(f"Field '{key}' must be a valid category, got '{value}'")


def _require_category_value(value: JSONValue, context: str) -> CompetitionCategory:
    """Require value to be a valid CompetitionCategory.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        Validated CompetitionCategory.

    Raises:
        JSONTypeError: If value is not a valid category.
    """
    if not isinstance(value, str):
        raise JSONTypeError(f"{context} must be a string, got {type(value).__name__}")
    if value == "Featured":
        return "Featured"
    if value == "Research":
        return "Research"
    if value == "Playground":
        return "Playground"
    if value == "Getting Started":
        return "Getting Started"
    if value == "Masters":
        return "Masters"
    if value == "Kudos":
        return "Kudos"
    raise JSONTypeError(f"{context} must be a valid category, got '{value}'")


# -----------------------------------------------------------------------------
# Competition
# -----------------------------------------------------------------------------


class Competition:
    """Kaggle competition metadata.

    Attributes:
        ref: Competition reference slug (e.g., "amex-default-prediction").
        title: Competition title.
        category: Competition category.
        reward: Prize description (e.g., "$100,000" or "Knowledge").
        deadline: Deadline as ISO 8601 date string.
        team_count: Number of teams participating.
        tags: Tuple of competition tags.
        description: Short description.
        url: Full Kaggle URL.
    """

    __slots__ = (
        "category",
        "deadline",
        "description",
        "ref",
        "reward",
        "tags",
        "team_count",
        "title",
        "url",
    )

    def __init__(
        self,
        *,
        ref: str,
        title: str,
        category: CompetitionCategory,
        reward: str,
        deadline: str,
        team_count: int,
        tags: tuple[str, ...],
        description: str,
        url: str,
    ) -> None:
        """Initialize competition.

        Args:
            ref: Competition reference slug.
            title: Competition title.
            category: Competition category.
            reward: Prize description.
            deadline: Deadline as ISO 8601 date string.
            team_count: Number of teams participating.
            tags: Tuple of competition tags.
            description: Short description.
            url: Full Kaggle URL.
        """
        self.ref = ref
        self.title = title
        self.category = category
        self.reward = reward
        self.deadline = deadline
        self.team_count = team_count
        self.tags = tags
        self.description = description
        self.url = url


def encode_competition(comp: Competition) -> JSONObject:
    """Encode Competition to JSON-serializable dict.

    Args:
        comp: Competition to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "ref": comp.ref,
        "title": comp.title,
        "category": comp.category,
        "reward": comp.reward,
        "deadline": comp.deadline,
        "team_count": comp.team_count,
        "tags": list(comp.tags),
        "description": comp.description,
        "url": comp.url,
    }
    return result


def decode_competition(data: JSONObject) -> Competition:
    """Decode Competition from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Competition.

    Raises:
        JSONTypeError: If validation fails.
    """
    return Competition(
        ref=require_str(data, "ref"),
        title=require_str(data, "title"),
        category=_require_category(data, "category"),
        reward=require_str(data, "reward"),
        deadline=require_str(data, "deadline"),
        team_count=require_int(data, "team_count"),
        tags=tuple(_require_list_str(data, "tags")),
        description=require_str(data, "description"),
        url=require_str(data, "url"),
    )


# -----------------------------------------------------------------------------
# CompetitionMatch
# -----------------------------------------------------------------------------


class CompetitionMatch:
    """A competition scored against codebase capabilities.

    Attributes:
        competition: The matched competition.
        match_score: Score from 0.0 to 1.0.
        matched_capabilities: Names of matched capabilities.
        missing_capabilities: Names of capabilities that would help.
        recommendation: Match recommendation level.
    """

    __slots__ = (
        "competition",
        "match_score",
        "matched_capabilities",
        "missing_capabilities",
        "recommendation",
    )

    def __init__(
        self,
        *,
        competition: Competition,
        match_score: float,
        matched_capabilities: tuple[str, ...],
        missing_capabilities: tuple[str, ...],
        recommendation: MatchRecommendation,
    ) -> None:
        """Initialize match.

        Args:
            competition: The matched competition.
            match_score: Score from 0.0 to 1.0.
            matched_capabilities: Names of matched capabilities.
            missing_capabilities: Names of capabilities that would help.
            recommendation: Match recommendation level.
        """
        self.competition = competition
        self.match_score = match_score
        self.matched_capabilities = matched_capabilities
        self.missing_capabilities = missing_capabilities
        self.recommendation = recommendation


def encode_match(match: CompetitionMatch) -> JSONObject:
    """Encode CompetitionMatch to JSON-serializable dict.

    Args:
        match: CompetitionMatch to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "competition": encode_competition(match.competition),
        "match_score": match.match_score,
        "matched_capabilities": list(match.matched_capabilities),
        "missing_capabilities": list(match.missing_capabilities),
        "recommendation": match.recommendation,
    }
    return result


def decode_match(data: JSONObject) -> CompetitionMatch:
    """Decode CompetitionMatch from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CompetitionMatch.

    Raises:
        JSONTypeError: If validation fails.
    """
    comp_raw = data.get("competition")
    return CompetitionMatch(
        competition=decode_competition(_require_dict_value(comp_raw, "competition")),
        match_score=require_float(data, "match_score"),
        matched_capabilities=tuple(_require_list_str(data, "matched_capabilities")),
        missing_capabilities=tuple(_require_list_str(data, "missing_capabilities")),
        recommendation=require_recommendation(data, "recommendation"),
    )


# -----------------------------------------------------------------------------
# InterestFilter
# -----------------------------------------------------------------------------


class InterestFilter:
    """User interest filter for competitions.

    Attributes:
        include_tags: Tags to include (must have at least one).
        exclude_tags: Tags to exclude (must not have any).
        min_reward: Minimum prize amount (None = include Knowledge).
        categories: Allowed categories (None = all).
    """

    __slots__ = ("categories", "exclude_tags", "include_tags", "min_reward")

    def __init__(
        self,
        *,
        include_tags: tuple[str, ...],
        exclude_tags: tuple[str, ...],
        min_reward: int | None,
        categories: tuple[CompetitionCategory, ...] | None,
    ) -> None:
        """Initialize filter.

        Args:
            include_tags: Tags to include.
            exclude_tags: Tags to exclude.
            min_reward: Minimum prize amount.
            categories: Allowed categories.
        """
        self.include_tags = include_tags
        self.exclude_tags = exclude_tags
        self.min_reward = min_reward
        self.categories = categories


def encode_filter(f: InterestFilter) -> JSONObject:
    """Encode InterestFilter to JSON-serializable dict.

    Args:
        f: InterestFilter to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "include_tags": list(f.include_tags),
        "exclude_tags": list(f.exclude_tags),
        "min_reward": f.min_reward,
        "categories": list(f.categories) if f.categories is not None else None,
    }
    return result


def decode_filter(data: JSONObject) -> InterestFilter:
    """Decode InterestFilter from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated InterestFilter.

    Raises:
        JSONTypeError: If validation fails.
    """
    cats_raw = data.get("categories")
    categories: tuple[CompetitionCategory, ...] | None = None
    if cats_raw is not None:
        if not isinstance(cats_raw, list):
            raise JSONTypeError(
                f"Field 'categories' must be an array, got {type(cats_raw).__name__}"
            )
        categories = tuple(
            _require_category_value(c, f"categories[{i}]") for i, c in enumerate(cats_raw)
        )

    return InterestFilter(
        include_tags=tuple(_require_list_str(data, "include_tags")),
        exclude_tags=tuple(_require_list_str(data, "exclude_tags")),
        min_reward=optional_int(data, "min_reward"),
        categories=categories,
    )


# -----------------------------------------------------------------------------
# CompetitionPage
# -----------------------------------------------------------------------------


class CompetitionPage:
    """A single page of competition content from Kaggle's internal API.

    Attributes:
        id: Numeric page ID.
        name: Page name (e.g., "Description", "Evaluation", "Timeline").
        content: Markdown content of the page.
    """

    __slots__ = ("content", "id", "name")

    def __init__(
        self,
        *,
        id: int,
        name: str,
        content: str,
    ) -> None:
        """Initialize competition page.

        Args:
            id: Numeric page ID.
            name: Page name.
            content: Markdown content.
        """
        self.id = id
        self.name = name
        self.content = content


def encode_competition_page(page: CompetitionPage) -> JSONObject:
    """Encode CompetitionPage to JSON-serializable dict.

    Args:
        page: CompetitionPage to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "id": page.id,
        "name": page.name,
        "content": page.content,
    }
    return result


def decode_competition_page(data: JSONObject) -> CompetitionPage:
    """Decode CompetitionPage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CompetitionPage.

    Raises:
        JSONTypeError: If validation fails.
    """
    return CompetitionPage(
        id=require_int(data, "id"),
        name=require_str(data, "name"),
        content=require_str(data, "content"),
    )


# -----------------------------------------------------------------------------
# CompetitionPages
# -----------------------------------------------------------------------------


class CompetitionPages:
    """Collection of competition pages with convenient accessors.

    Provides quick access to common pages (description, evaluation, etc.)
    while also exposing the full list of pages.

    Attributes:
        competition_id: Numeric Kaggle competition ID.
        pages: Tuple of all pages.
        description: Content of the Description page (empty if not found).
        evaluation: Content of the Evaluation page (empty if not found).
        timeline: Content of the Timeline page (empty if not found).
        rules: Content of the Rules page (empty if not found).
    """

    __slots__ = (
        "competition_id",
        "description",
        "evaluation",
        "pages",
        "rules",
        "timeline",
    )

    def __init__(
        self,
        *,
        competition_id: int,
        pages: tuple[CompetitionPage, ...],
        description: str,
        evaluation: str,
        timeline: str,
        rules: str,
    ) -> None:
        """Initialize competition pages collection.

        Args:
            competition_id: Numeric Kaggle competition ID.
            pages: Tuple of all pages.
            description: Content of the Description page.
            evaluation: Content of the Evaluation page.
            timeline: Content of the Timeline page.
            rules: Content of the Rules page.
        """
        self.competition_id = competition_id
        self.pages = pages
        self.description = description
        self.evaluation = evaluation
        self.timeline = timeline
        self.rules = rules


def encode_competition_pages(pages: CompetitionPages) -> JSONObject:
    """Encode CompetitionPages to JSON-serializable dict.

    Args:
        pages: CompetitionPages to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "competition_id": pages.competition_id,
        "pages": [encode_competition_page(p) for p in pages.pages],
        "description": pages.description,
        "evaluation": pages.evaluation,
        "timeline": pages.timeline,
        "rules": pages.rules,
    }
    return result


def decode_competition_pages(data: JSONObject) -> CompetitionPages:
    """Decode CompetitionPages from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CompetitionPages.

    Raises:
        JSONTypeError: If validation fails.
    """
    pages_raw = require_list(data, "pages")
    pages: list[CompetitionPage] = []
    for i, page_data in enumerate(pages_raw):
        pages.append(decode_competition_page(_require_dict_value(page_data, f"pages[{i}]")))

    return CompetitionPages(
        competition_id=require_int(data, "competition_id"),
        pages=tuple(pages),
        description=require_str(data, "description"),
        evaluation=require_str(data, "evaluation"),
        timeline=require_str(data, "timeline"),
        rules=require_str(data, "rules"),
    )


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


class KaggleTagProtocol(Protocol):
    """Protocol for Kaggle API tag object (ApiCategory)."""

    ref: str


class KaggleCompetitionProtocol(Protocol):
    """Protocol for Kaggle API competition object (ApiCompetition).

    The real Kaggle API returns ApiCompetition objects with these attributes.
    Types match the real kagglesdk.competitions.types.ApiCompetition.
    """

    @property
    def ref(self) -> str:
        """Competition reference URL (full Kaggle URL)."""
        ...

    @property
    def title(self) -> str:
        """Competition title."""
        ...

    @property
    def category(self) -> str:
        """Competition category."""
        ...

    @property
    def reward(self) -> str:
        """Prize description."""
        ...

    @property
    def deadline(self) -> datetime:
        """Deadline (datetime object from Kaggle API)."""
        ...

    @property
    def team_count(self) -> int:
        """Number of teams."""
        ...

    @property
    def tags(self) -> Sequence[KaggleTagProtocol | None] | None:
        """Competition tags (may contain None items or be None)."""
        ...

    @property
    def description(self) -> str:
        """Short description."""
        ...

    @property
    def url(self) -> str:
        """Full Kaggle URL."""
        ...


class CompetitionsResponseProtocol(Protocol):
    """Protocol for competitions_list response (new Kaggle API format)."""

    @property
    def competitions(self) -> Sequence[KaggleCompetitionProtocol | None] | None:
        """Get sequence of competition objects (may contain None or be None)."""
        ...


class KaggleApiProtocol(Protocol):
    """Protocol for Kaggle API instance."""

    def authenticate(self) -> None:
        """Authenticate with Kaggle API using credentials."""
        ...

    def competitions_list(
        self,
        group: str | None = None,
        category: str | None = None,
        sort_by: str | None = None,
        page: int | None = None,
        search: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> CompetitionsResponseProtocol | None:
        """List competitions with optional filters.

        Args:
            group: Competition group filter.
            category: Category filter.
            sort_by: Sort order.
            page: Page number.
            search: Search query.
            page_size: Results per page.
            page_token: Pagination token.

        Returns:
            Response wrapper with competitions list, or None.
        """
        ...


class KaggleApiClassProtocol(Protocol):
    """Protocol for KaggleApi class (not instance)."""

    def __call__(self) -> KaggleApiProtocol:
        """Instantiate a new KaggleApi.

        Returns:
            New unauthenticated KaggleApiProtocol instance.
        """
        ...


class KaggleModuleProtocol(Protocol):
    """Protocol for the kaggle API module."""

    KaggleApi: KaggleApiClassProtocol


class KagglePreAuthModuleProtocol(Protocol):
    """Protocol for kaggle module with pre-authenticated global api.

    The kaggle package creates and authenticates a global `api` object
    at import time in its __init__.py. This protocol allows typed access.
    """

    api: KaggleApiProtocol


class KaggleApiFactoryProtocol(Protocol):
    """Protocol for Kaggle API factory."""

    def __call__(self) -> KaggleApiProtocol:
        """Create authenticated Kaggle API instance.

        Returns:
            Authenticated KaggleApiProtocol instance.
        """
        ...


@runtime_checkable
class KaggleClientProtocol(Protocol):
    """Protocol for Kaggle API client."""

    def list_competitions(
        self,
        *,
        search: str | None = None,
        category: CompetitionCategory | None = None,
    ) -> tuple[Competition, ...]:
        """List active competitions with optional filters.

        Args:
            search: Optional search query.
            category: Optional category filter.

        Returns:
            Tuple of matching competitions.
        """
        ...

    def get_competition(self, ref: str) -> Competition | None:
        """Get a specific competition by ref.

        Args:
            ref: Competition reference slug.

        Returns:
            Competition if found, None otherwise.
        """
        ...


@runtime_checkable
class KagglePageFetcherProtocol(Protocol):
    """Protocol for fetching competition pages from Kaggle's internal API."""

    def fetch_pages(self, competition_id: int) -> CompetitionPages:
        """Fetch all pages for a competition.

        Args:
            competition_id: Numeric Kaggle competition ID.

        Returns:
            CompetitionPages containing all page content.

        Raises:
            RuntimeError: If the API request fails.
        """
        ...

    def get_competition_id(self, slug: str) -> int:
        """Get the numeric competition ID from a slug.

        Args:
            slug: Competition slug (e.g., "google-gemma-3n-hackathon").

        Returns:
            Numeric competition ID.

        Raises:
            RuntimeError: If the API request fails.
            JSONTypeError: If response parsing fails.
        """
        ...
