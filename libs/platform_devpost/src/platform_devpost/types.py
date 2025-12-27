"""Core types for platform_devpost.

This module defines Devpost-specific types and re-exports shared types
from platform_codebase.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from platform_codebase import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    MatchRecommendation,
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
    require_bool,
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
    "DevpostApiProtocol",
    "DevpostClientProtocol",
    "DisplayedLocation",
    "Hackathon",
    "HackathonListMeta",
    "HackathonListResponse",
    "HackathonMatch",
    "HackathonState",
    "InterestFilter",
    "MatchRecommendation",
    "Theme",
    "decode_capability",
    "decode_displayed_location",
    "decode_filter",
    "decode_hackathon",
    "decode_list_meta",
    "decode_list_response",
    "decode_match",
    "decode_profile",
    "decode_theme",
    "encode_capability",
    "encode_displayed_location",
    "encode_filter",
    "encode_hackathon",
    "encode_list_meta",
    "encode_list_response",
    "encode_match",
    "encode_profile",
    "encode_theme",
    "require_recommendation",
    "require_strength",
]


# -----------------------------------------------------------------------------
# Devpost-Specific Literal Types
# -----------------------------------------------------------------------------

HackathonState = Literal["open", "upcoming", "ended", "submissions"]


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


def _require_state(obj: JSONObject, key: str) -> HackathonState:
    """Extract and validate HackathonState from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        Validated HackathonState.

    Raises:
        JSONTypeError: If field is missing or not a valid state.
    """
    value = require_str(obj, key)
    if value == "open":
        return "open"
    if value == "upcoming":
        return "upcoming"
    if value == "ended":
        return "ended"
    if value == "submissions":
        return "submissions"
    raise JSONTypeError(f"Field '{key}' must be a valid state, got '{value}'")


def _require_state_value(value: JSONValue, context: str) -> HackathonState:
    """Require value to be a valid HackathonState.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        Validated HackathonState.

    Raises:
        JSONTypeError: If value is not a valid state.
    """
    if not isinstance(value, str):
        raise JSONTypeError(f"{context} must be a string, got {type(value).__name__}")
    if value == "open":
        return "open"
    if value == "upcoming":
        return "upcoming"
    if value == "ended":
        return "ended"
    if value == "submissions":
        return "submissions"
    raise JSONTypeError(f"{context} must be a valid state, got '{value}'")


# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------


class Theme:
    """A hackathon theme/category.

    Attributes:
        id: Theme identifier.
        name: Theme display name.
    """

    __slots__ = ("id", "name")

    def __init__(self, *, id: int, name: str) -> None:
        """Initialize theme.

        Args:
            id: Theme identifier.
            name: Theme display name.
        """
        self.id = id
        self.name = name


def encode_theme(theme: Theme) -> JSONObject:
    """Encode Theme to JSON-serializable dict.

    Args:
        theme: Theme to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "id": theme.id,
        "name": theme.name,
    }
    return result


def decode_theme(data: JSONObject) -> Theme:
    """Decode Theme from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Theme.

    Raises:
        JSONTypeError: If validation fails.
    """
    return Theme(
        id=require_int(data, "id"),
        name=require_str(data, "name"),
    )


# -----------------------------------------------------------------------------
# DisplayedLocation
# -----------------------------------------------------------------------------


class DisplayedLocation:
    """Location information for a hackathon.

    Attributes:
        icon: Icon name for display.
        location: Location text description.
    """

    __slots__ = ("icon", "location")

    def __init__(self, *, icon: str, location: str) -> None:
        """Initialize displayed location.

        Args:
            icon: Icon name for display.
            location: Location text description.
        """
        self.icon = icon
        self.location = location


def encode_displayed_location(loc: DisplayedLocation) -> JSONObject:
    """Encode DisplayedLocation to JSON-serializable dict.

    Args:
        loc: DisplayedLocation to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "icon": loc.icon,
        "location": loc.location,
    }
    return result


def decode_displayed_location(data: JSONObject) -> DisplayedLocation:
    """Decode DisplayedLocation from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated DisplayedLocation.

    Raises:
        JSONTypeError: If validation fails.
    """
    return DisplayedLocation(
        icon=require_str(data, "icon"),
        location=require_str(data, "location"),
    )


# -----------------------------------------------------------------------------
# Hackathon
# -----------------------------------------------------------------------------


class Hackathon:
    """Devpost hackathon metadata.

    Attributes:
        id: Unique hackathon identifier.
        title: Hackathon title.
        url: Full Devpost URL.
        thumbnail_url: URL for hackathon thumbnail image.
        organization_name: Name of organizing organization.
        displayed_location: Location information.
        open_state: Current state (open, upcoming, ended, submissions).
        time_left_to_submission: Human-readable time remaining.
        submission_period_dates: Date range string.
        themes: Tuple of hackathon themes.
        prize_amount: Prize amount as HTML string.
        registrations_count: Number of registrations.
        featured: Whether hackathon is featured.
        winners_announced: Whether winners have been announced.
        invite_only: Whether hackathon is invite-only.
    """

    __slots__ = (
        "displayed_location",
        "featured",
        "id",
        "invite_only",
        "open_state",
        "organization_name",
        "prize_amount",
        "registrations_count",
        "submission_period_dates",
        "themes",
        "thumbnail_url",
        "time_left_to_submission",
        "title",
        "url",
        "winners_announced",
    )

    def __init__(
        self,
        *,
        id: int,
        title: str,
        url: str,
        thumbnail_url: str,
        organization_name: str,
        displayed_location: DisplayedLocation,
        open_state: HackathonState,
        time_left_to_submission: str,
        submission_period_dates: str,
        themes: tuple[Theme, ...],
        prize_amount: str,
        registrations_count: int,
        featured: bool,
        winners_announced: bool,
        invite_only: bool,
    ) -> None:
        """Initialize hackathon.

        Args:
            id: Unique hackathon identifier.
            title: Hackathon title.
            url: Full Devpost URL.
            thumbnail_url: URL for hackathon thumbnail image.
            organization_name: Name of organizing organization.
            displayed_location: Location information.
            open_state: Current state.
            time_left_to_submission: Human-readable time remaining.
            submission_period_dates: Date range string.
            themes: Tuple of hackathon themes.
            prize_amount: Prize amount as HTML string.
            registrations_count: Number of registrations.
            featured: Whether hackathon is featured.
            winners_announced: Whether winners have been announced.
            invite_only: Whether hackathon is invite-only.
        """
        self.id = id
        self.title = title
        self.url = url
        self.thumbnail_url = thumbnail_url
        self.organization_name = organization_name
        self.displayed_location = displayed_location
        self.open_state = open_state
        self.time_left_to_submission = time_left_to_submission
        self.submission_period_dates = submission_period_dates
        self.themes = themes
        self.prize_amount = prize_amount
        self.registrations_count = registrations_count
        self.featured = featured
        self.winners_announced = winners_announced
        self.invite_only = invite_only


def encode_hackathon(hackathon: Hackathon) -> JSONObject:
    """Encode Hackathon to JSON-serializable dict.

    Args:
        hackathon: Hackathon to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "id": hackathon.id,
        "title": hackathon.title,
        "url": hackathon.url,
        "thumbnail_url": hackathon.thumbnail_url,
        "organization_name": hackathon.organization_name,
        "displayed_location": encode_displayed_location(hackathon.displayed_location),
        "open_state": hackathon.open_state,
        "time_left_to_submission": hackathon.time_left_to_submission,
        "submission_period_dates": hackathon.submission_period_dates,
        "themes": [encode_theme(t) for t in hackathon.themes],
        "prize_amount": hackathon.prize_amount,
        "registrations_count": hackathon.registrations_count,
        "featured": hackathon.featured,
        "winners_announced": hackathon.winners_announced,
        "invite_only": hackathon.invite_only,
    }
    return result


def decode_hackathon(data: JSONObject) -> Hackathon:
    """Decode Hackathon from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Hackathon.

    Raises:
        JSONTypeError: If validation fails.
    """
    loc_raw = data.get("displayed_location")
    themes_raw = require_list(data, "themes")

    return Hackathon(
        id=require_int(data, "id"),
        title=require_str(data, "title"),
        url=require_str(data, "url"),
        thumbnail_url=require_str(data, "thumbnail_url"),
        organization_name=require_str(data, "organization_name"),
        displayed_location=decode_displayed_location(
            _require_dict_value(loc_raw, "displayed_location")
        ),
        open_state=_require_state(data, "open_state"),
        time_left_to_submission=require_str(data, "time_left_to_submission"),
        submission_period_dates=require_str(data, "submission_period_dates"),
        themes=tuple(
            decode_theme(_require_dict_value(t, f"themes[{i}]")) for i, t in enumerate(themes_raw)
        ),
        prize_amount=require_str(data, "prize_amount"),
        registrations_count=require_int(data, "registrations_count"),
        featured=require_bool(data, "featured"),
        winners_announced=require_bool(data, "winners_announced"),
        invite_only=require_bool(data, "invite_only"),
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
        matched_capabilities=tuple(_require_list_str(data, "matched_capabilities")),
        missing_capabilities=tuple(_require_list_str(data, "missing_capabilities")),
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
        include_themes=tuple(_require_list_str(data, "include_themes")),
        exclude_themes=tuple(_require_list_str(data, "exclude_themes")),
        states=states,
        featured_only=require_bool(data, "featured_only"),
    )


# -----------------------------------------------------------------------------
# API Response Types
# -----------------------------------------------------------------------------


class HackathonListMeta:
    """Metadata for hackathon list response.

    Attributes:
        total_count: Total number of hackathons matching query.
        per_page: Number of hackathons per page.
    """

    __slots__ = ("per_page", "total_count")

    def __init__(self, *, total_count: int, per_page: int) -> None:
        """Initialize meta.

        Args:
            total_count: Total number of hackathons.
            per_page: Number per page.
        """
        self.total_count = total_count
        self.per_page = per_page


def encode_list_meta(meta: HackathonListMeta) -> JSONObject:
    """Encode HackathonListMeta to JSON-serializable dict.

    Args:
        meta: Meta to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "total_count": meta.total_count,
        "per_page": meta.per_page,
    }
    return result


def decode_list_meta(data: JSONObject) -> HackathonListMeta:
    """Decode HackathonListMeta from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated HackathonListMeta.

    Raises:
        JSONTypeError: If validation fails.
    """
    return HackathonListMeta(
        total_count=require_int(data, "total_count"),
        per_page=require_int(data, "per_page"),
    )


class HackathonListResponse:
    """Response from hackathon list API.

    Attributes:
        hackathons: Tuple of hackathons.
        meta: Pagination metadata.
    """

    __slots__ = ("hackathons", "meta")

    def __init__(
        self,
        *,
        hackathons: tuple[Hackathon, ...],
        meta: HackathonListMeta,
    ) -> None:
        """Initialize response.

        Args:
            hackathons: Tuple of hackathons.
            meta: Pagination metadata.
        """
        self.hackathons = hackathons
        self.meta = meta


def encode_list_response(resp: HackathonListResponse) -> JSONObject:
    """Encode HackathonListResponse to JSON-serializable dict.

    Args:
        resp: Response to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "hackathons": [encode_hackathon(h) for h in resp.hackathons],
        "meta": encode_list_meta(resp.meta),
    }
    return result


def decode_list_response(data: JSONObject) -> HackathonListResponse:
    """Decode HackathonListResponse from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated HackathonListResponse.

    Raises:
        JSONTypeError: If validation fails.
    """
    hackathons_raw = require_list(data, "hackathons")
    meta_raw = data.get("meta")

    return HackathonListResponse(
        hackathons=tuple(
            decode_hackathon(_require_dict_value(h, f"hackathons[{i}]"))
            for i, h in enumerate(hackathons_raw)
        ),
        meta=decode_list_meta(_require_dict_value(meta_raw, "meta")),
    )


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


class DevpostApiProtocol(Protocol):
    """Protocol for Devpost API client."""

    def fetch_hackathons(
        self,
        *,
        page: int = 1,
        search: str | None = None,
    ) -> HackathonListResponse:
        """Fetch hackathons from Devpost API.

        Args:
            page: Page number (1-indexed).
            search: Optional search query.

        Returns:
            HackathonListResponse with hackathons and metadata.
        """
        ...


@runtime_checkable
class DevpostClientProtocol(Protocol):
    """Protocol for Devpost client."""

    def list_hackathons(
        self,
        *,
        search: str | None = None,
        state: HackathonState | None = None,
    ) -> tuple[Hackathon, ...]:
        """List hackathons with optional filters.

        Args:
            search: Optional search query.
            state: Optional state filter.

        Returns:
            Tuple of matching hackathons.
        """
        ...

    def get_hackathon(self, hackathon_id: int) -> Hackathon | None:
        """Get a specific hackathon by ID.

        Args:
            hackathon_id: Hackathon identifier.

        Returns:
            Hackathon if found, None otherwise.
        """
        ...
