"""types: CompetitionMatch and related definitions."""

from __future__ import annotations

from platform_codebase import (
    MatchRecommendation,
    require_recommendation,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    optional_int,
    require_float,
    require_str_list,
)

from platform_kaggle._types_competition import Competition, decode_competition, encode_competition
from platform_kaggle._types_validation import (
    CompetitionCategory,
    _require_category_value,
    _require_dict_value,
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
        matched_capabilities=tuple(require_str_list(data, "matched_capabilities")),
        missing_capabilities=tuple(require_str_list(data, "missing_capabilities")),
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
        include_tags=tuple(require_str_list(data, "include_tags")),
        exclude_tags=tuple(require_str_list(data, "exclude_tags")),
        min_reward=optional_int(data, "min_reward"),
        categories=categories,
    )
