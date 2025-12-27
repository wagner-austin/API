"""Core types for platform_kaggle.

This module defines Kaggle-specific types and re-exports shared types
from platform_codebase.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    "CompetitionsResponseProtocol",
    "InterestFilter",
    "KaggleApiClassProtocol",
    "KaggleApiFactoryProtocol",
    "KaggleApiProtocol",
    "KaggleClientProtocol",
    "KaggleCompetitionProtocol",
    "KaggleModuleProtocol",
    "KaggleTagProtocol",
    "LibInfo",
    "MatchRecommendation",
    "ServiceInfo",
    "decode_capability",
    "decode_competition",
    "decode_filter",
    "decode_match",
    "decode_profile",
    "encode_capability",
    "encode_competition",
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
# Protocols
# -----------------------------------------------------------------------------


class KaggleTagProtocol(Protocol):
    """Protocol for Kaggle API tag object (ApiCategory)."""

    ref: str


class KaggleCompetitionProtocol(Protocol):
    """Protocol for Kaggle API competition object (ApiCompetition).

    The real Kaggle API returns ApiCompetition objects with these attributes.
    """

    @property
    def ref(self) -> str:
        """Competition reference slug."""
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
    def deadline(self) -> str:
        """Deadline as ISO 8601 date string."""
        ...

    @property
    def team_count(self) -> int:
        """Number of teams."""
        ...

    @property
    def tags(self) -> Sequence[KaggleTagProtocol]:
        """Competition tags."""
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
    def competitions(self) -> Sequence[KaggleCompetitionProtocol]:
        """Get sequence of competition objects."""
        ...


class KaggleApiProtocol(Protocol):
    """Protocol for Kaggle API instance."""

    def authenticate(self) -> None:
        """Authenticate with Kaggle API using credentials."""
        ...

    def competitions_list(
        self,
        search: str = "",
        category: str = "",
    ) -> CompetitionsResponseProtocol:
        """List competitions with optional filters.

        Args:
            search: Optional search query.
            category: Optional category filter.

        Returns:
            Response wrapper with competitions list.
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
