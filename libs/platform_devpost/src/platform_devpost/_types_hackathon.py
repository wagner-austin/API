"""platform_devpost Hackathon payloads."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_int,
    require_list,
    require_str,
)

from platform_devpost._types_theme import (
    DisplayedLocation,
    Theme,
    decode_displayed_location,
    decode_theme,
    encode_displayed_location,
    encode_theme,
)
from platform_devpost._types_validation import (
    HackathonState,
    _require_dict_value,
    _require_state,
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
