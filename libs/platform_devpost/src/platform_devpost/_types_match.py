"""platform_devpost Hackathon match and interest-filter payloads."""

from __future__ import annotations

from platform_codebase import (
    MatchRecommendation,
    require_recommendation,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_bool,
    require_float,
    require_str_list,
)

from platform_devpost._types_hackathon import (
    Hackathon,
    decode_hackathon,
    encode_hackathon,
)
from platform_devpost._types_validation import (
    HackathonState,
    _require_dict_value,
    _require_state_value,
)

# -----------------------------------------------------------------------------
# HackathonMatch
# -----------------------------------------------------------------------------


class HackathonMatch:
    """A hackathon scored against codebase capabilities.

    Attributes:
        hackathon: The matched hackathon.
        match_score: Score from 0.0 to 1.0.
        matched_capabilities: Names of matched capabilities.
        missing_capabilities: Names of capabilities that would help.
        recommendation: Match recommendation level.
    """

    __slots__ = (
        "hackathon",
        "match_score",
        "matched_capabilities",
        "missing_capabilities",
        "recommendation",
    )

    def __init__(
        self,
        *,
        hackathon: Hackathon,
        match_score: float,
        matched_capabilities: tuple[str, ...],
        missing_capabilities: tuple[str, ...],
        recommendation: MatchRecommendation,
    ) -> None:
        """Initialize match.

        Args:
            hackathon: The matched hackathon.
            match_score: Score from 0.0 to 1.0.
            matched_capabilities: Names of matched capabilities.
            missing_capabilities: Names of capabilities that would help.
            recommendation: Match recommendation level.
        """
        self.hackathon = hackathon
        self.match_score = match_score
        self.matched_capabilities = matched_capabilities
        self.missing_capabilities = missing_capabilities
        self.recommendation = recommendation


def encode_match(match: HackathonMatch) -> JSONObject:
    """Encode HackathonMatch to JSON-serializable dict.

    Args:
        match: HackathonMatch to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "hackathon": encode_hackathon(match.hackathon),
        "match_score": match.match_score,
        "matched_capabilities": list(match.matched_capabilities),
        "missing_capabilities": list(match.missing_capabilities),
        "recommendation": match.recommendation,
    }
    return result


def decode_match(data: JSONObject) -> HackathonMatch:
    """Decode HackathonMatch from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated HackathonMatch.

    Raises:
        JSONTypeError: If validation fails.
    """
    hackathon_raw = data.get("hackathon")
    return HackathonMatch(
        hackathon=decode_hackathon(_require_dict_value(hackathon_raw, "hackathon")),
        match_score=require_float(data, "match_score"),
        matched_capabilities=tuple(require_str_list(data, "matched_capabilities")),
        missing_capabilities=tuple(require_str_list(data, "missing_capabilities")),
        recommendation=require_recommendation(data, "recommendation"),
    )


# -----------------------------------------------------------------------------
# InterestFilter
# -----------------------------------------------------------------------------


class InterestFilter:
    """User interest filter for hackathons.

    Attributes:
        include_themes: Theme names to include (must have at least one).
        exclude_themes: Theme names to exclude (must not have any).
        states: Allowed states (None = all).
        featured_only: If True, only return featured hackathons.
    """

    __slots__ = ("exclude_themes", "featured_only", "include_themes", "states")

    def __init__(
        self,
        *,
        include_themes: tuple[str, ...],
        exclude_themes: tuple[str, ...],
        states: tuple[HackathonState, ...] | None,
        featured_only: bool,
    ) -> None:
        """Initialize filter.

        Args:
            include_themes: Theme names to include.
            exclude_themes: Theme names to exclude.
            states: Allowed states (None = all).
            featured_only: If True, only return featured hackathons.
        """
        self.include_themes = include_themes
        self.exclude_themes = exclude_themes
        self.states = states
        self.featured_only = featured_only


def encode_filter(f: InterestFilter) -> JSONObject:
    """Encode InterestFilter to JSON-serializable dict.

    Args:
        f: InterestFilter to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "include_themes": list(f.include_themes),
        "exclude_themes": list(f.exclude_themes),
        "states": list(f.states) if f.states is not None else None,
        "featured_only": f.featured_only,
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
    states_raw = data.get("states")
    states: tuple[HackathonState, ...] | None = None
    if states_raw is not None:
        if not isinstance(states_raw, list):
            raise JSONTypeError(f"Field 'states' must be an array, got {type(states_raw).__name__}")
        states = tuple(_require_state_value(s, f"states[{i}]") for i, s in enumerate(states_raw))

    return InterestFilter(
        include_themes=tuple(require_str_list(data, "include_themes")),
        exclude_themes=tuple(require_str_list(data, "exclude_themes")),
        states=states,
        featured_only=require_bool(data, "featured_only"),
    )
