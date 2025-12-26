"""TypedDict definitions for Google Calendar API and competition tracking."""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_str,
    require_bool,
    require_int,
    require_list,
    require_str,
)

# =============================================================================
# Literal Types
# =============================================================================

EventStatus = Literal["confirmed", "tentative", "cancelled"]
ReminderMethod = Literal["email", "popup"]
CompetitionSource = Literal["kaggle", "devpost", "manual"]
TokenType = Literal["Bearer"]
CalendarAccessRole = Literal["freeBusyReader", "reader", "writer", "owner"]

# Default reminders: 1 day (1440 min) + 1 hour (60 min) before deadline
DEFAULT_REMINDERS: tuple[int, int] = (1440, 60)


# =============================================================================
# Validation Helpers
# =============================================================================


def _require_event_status(obj: JSONObject, key: str) -> EventStatus:
    """Extract and validate EventStatus from JSON object."""
    value = require_str(obj, key)
    if value == "confirmed":
        return "confirmed"
    if value == "tentative":
        return "tentative"
    if value == "cancelled":
        return "cancelled"
    raise JSONTypeError(f"Field '{key}' must be confirmed/tentative/cancelled, got '{value}'")


def _require_reminder_method(obj: JSONObject, key: str) -> ReminderMethod:
    """Extract and validate ReminderMethod from JSON object."""
    value = require_str(obj, key)
    if value == "email":
        return "email"
    if value == "popup":
        return "popup"
    raise JSONTypeError(f"Field '{key}' must be email/popup, got '{value}'")


def _require_competition_source(obj: JSONObject, key: str) -> CompetitionSource:
    """Extract and validate CompetitionSource from JSON object."""
    value = require_str(obj, key)
    if value == "kaggle":
        return "kaggle"
    if value == "devpost":
        return "devpost"
    if value == "manual":
        return "manual"
    raise JSONTypeError(f"Field '{key}' must be kaggle/devpost/manual, got '{value}'")


def _require_token_type(obj: JSONObject, key: str) -> TokenType:
    """Extract and validate TokenType from JSON object."""
    value = require_str(obj, key)
    if value == "Bearer":
        return "Bearer"
    raise JSONTypeError(f"Field '{key}' must be Bearer, got '{value}'")


def _require_access_role(obj: JSONObject, key: str) -> CalendarAccessRole:
    """Extract and validate CalendarAccessRole from JSON object."""
    value = require_str(obj, key)
    if value == "freeBusyReader":
        return "freeBusyReader"
    if value == "reader":
        return "reader"
    if value == "writer":
        return "writer"
    if value == "owner":
        return "owner"
    raise JSONTypeError(f"Field '{key}' must be freeBusyReader/reader/writer/owner, got '{value}'")


def _require_dict_value(value: JSONValue, context: str) -> JSONObject:
    """Require value to be a dict."""
    if not isinstance(value, dict):
        raise JSONTypeError(f"{context} must be an object, got {type(value).__name__}")
    return value


def _require_list_int(obj: JSONObject, key: str) -> list[int]:
    """Extract required list of ints from JSON object."""
    items = require_list(obj, key)
    result: list[int] = []
    for i, item in enumerate(items):
        if not isinstance(item, int):
            raise JSONTypeError(f"Field '{key}[{i}]' must be an int, got {type(item).__name__}")
        result.append(item)
    return result


# =============================================================================
# Calendar Event Types
# =============================================================================


class EventDateTime(TypedDict):
    """DateTime for calendar event."""

    dateTime: str  # RFC 3339: "2025-12-26T14:00:00-08:00"
    timeZone: str  # e.g., "America/Los_Angeles"


def encode_event_datetime(dt: EventDateTime) -> JSONObject:
    """Encode EventDateTime to JSON-serializable dict."""
    result: JSONObject = {
        "dateTime": dt["dateTime"],
        "timeZone": dt["timeZone"],
    }
    return result


def decode_event_datetime(data: JSONObject) -> EventDateTime:
    """Decode EventDateTime from dict with validation."""
    return EventDateTime(
        dateTime=require_str(data, "dateTime"),
        timeZone=require_str(data, "timeZone"),
    )


class ReminderOverride(TypedDict):
    """Reminder override for calendar event."""

    method: ReminderMethod
    minutes: int


def encode_reminder_override(r: ReminderOverride) -> JSONObject:
    """Encode ReminderOverride to JSON-serializable dict."""
    result: JSONObject = {
        "method": r["method"],
        "minutes": r["minutes"],
    }
    return result


def decode_reminder_override(data: JSONObject) -> ReminderOverride:
    """Decode ReminderOverride from dict with validation."""
    return ReminderOverride(
        method=_require_reminder_method(data, "method"),
        minutes=require_int(data, "minutes"),
    )


class EventReminders(TypedDict):
    """Reminders configuration for calendar event."""

    useDefault: bool
    overrides: tuple[ReminderOverride, ...]


def encode_event_reminders(r: EventReminders) -> JSONObject:
    """Encode EventReminders to JSON-serializable dict."""
    overrides_list: list[JSONValue] = []
    for override in r["overrides"]:
        overrides_list.append(encode_reminder_override(override))
    result: JSONObject = {
        "useDefault": r["useDefault"],
        "overrides": overrides_list,
    }
    return result


def decode_event_reminders(data: JSONObject) -> EventReminders:
    """Decode EventReminders from dict with validation."""
    overrides_raw = require_list(data, "overrides")
    return EventReminders(
        useDefault=require_bool(data, "useDefault"),
        overrides=tuple(
            decode_reminder_override(_require_dict_value(o, f"overrides[{i}]"))
            for i, o in enumerate(overrides_raw)
        ),
    )


class CalendarEvent(TypedDict):
    """Google Calendar event.

    Attributes:
        id: Unique event identifier.
        summary: Event title.
        description: Event description.
        start: Event start time.
        end: Event end time.
        status: Event status (confirmed, tentative, cancelled).
        reminders: Reminder configuration.
        location: Location string (e.g., "123 Main St, City").
        recurrence: Tuple of RRULE strings for recurring events.
    """

    id: str
    summary: str
    description: str
    start: EventDateTime
    end: EventDateTime
    status: EventStatus
    reminders: EventReminders
    location: str
    recurrence: tuple[str, ...]


def encode_calendar_event(e: CalendarEvent) -> JSONObject:
    """Encode CalendarEvent to JSON-serializable dict.

    Args:
        e: CalendarEvent to encode.

    Returns:
        JSON-serializable dict representation.
    """
    recurrence_list: list[JSONValue] = list(e["recurrence"])
    result: JSONObject = {
        "id": e["id"],
        "summary": e["summary"],
        "description": e["description"],
        "start": encode_event_datetime(e["start"]),
        "end": encode_event_datetime(e["end"]),
        "status": e["status"],
        "reminders": encode_event_reminders(e["reminders"]),
        "location": e["location"],
        "recurrence": recurrence_list,
    }
    return result


def _require_recurrence(data: JSONObject, key: str) -> tuple[str, ...]:
    """Extract and validate recurrence list from JSON object.

    Args:
        data: JSON object to extract from.
        key: Key to extract.

    Returns:
        Tuple of RRULE strings.

    Raises:
        JSONTypeError: If value is not a list of strings.
    """
    raw_list = data.get(key, [])
    if not isinstance(raw_list, list):
        raise JSONTypeError(f"Field '{key}' must be a list, got {type(raw_list).__name__}")
    result: list[str] = []
    for i, item in enumerate(raw_list):
        if not isinstance(item, str):
            raise JSONTypeError(f"Field '{key}[{i}]' must be a string, got {type(item).__name__}")
        result.append(item)
    return tuple(result)


def decode_calendar_event(data: JSONObject) -> CalendarEvent:
    """Decode CalendarEvent from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CalendarEvent.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    start_raw = data.get("start")
    end_raw = data.get("end")
    reminders_raw = data.get("reminders")
    # Location defaults to empty string if not present
    location_raw = data.get("location", "")
    location = location_raw if isinstance(location_raw, str) else ""
    return CalendarEvent(
        id=require_str(data, "id"),
        summary=require_str(data, "summary"),
        description=require_str(data, "description"),
        start=decode_event_datetime(_require_dict_value(start_raw, "start")),
        end=decode_event_datetime(_require_dict_value(end_raw, "end")),
        status=_require_event_status(data, "status"),
        reminders=decode_event_reminders(_require_dict_value(reminders_raw, "reminders")),
        location=location,
        recurrence=_require_recurrence(data, "recurrence"),
    )


class CalendarListItem(TypedDict):
    """Calendar in user's calendar list."""

    id: str
    summary: str
    description: str
    primary: bool
    accessRole: CalendarAccessRole
    timeZone: str


def encode_calendar_list_item(c: CalendarListItem) -> JSONObject:
    """Encode CalendarListItem to JSON-serializable dict."""
    result: JSONObject = {
        "id": c["id"],
        "summary": c["summary"],
        "description": c["description"],
        "primary": c["primary"],
        "accessRole": c["accessRole"],
        "timeZone": c["timeZone"],
    }
    return result


def decode_calendar_list_item(data: JSONObject) -> CalendarListItem:
    """Decode CalendarListItem from dict with validation."""
    return CalendarListItem(
        id=require_str(data, "id"),
        summary=require_str(data, "summary"),
        description=require_str(data, "description"),
        primary=require_bool(data, "primary"),
        accessRole=_require_access_role(data, "accessRole"),
        timeZone=require_str(data, "timeZone"),
    )


# =============================================================================
# OAuth Types
# =============================================================================


class OAuthCredentials(TypedDict):
    """OAuth 2.0 client credentials."""

    client_id: str
    client_secret: str
    redirect_uri: str


def encode_oauth_credentials(c: OAuthCredentials) -> JSONObject:
    """Encode OAuthCredentials to JSON-serializable dict."""
    result: JSONObject = {
        "client_id": c["client_id"],
        "client_secret": c["client_secret"],
        "redirect_uri": c["redirect_uri"],
    }
    return result


def decode_oauth_credentials(data: JSONObject) -> OAuthCredentials:
    """Decode OAuthCredentials from dict with validation."""
    return OAuthCredentials(
        client_id=require_str(data, "client_id"),
        client_secret=require_str(data, "client_secret"),
        redirect_uri=require_str(data, "redirect_uri"),
    )


class OAuthTokens(TypedDict):
    """OAuth 2.0 access and refresh tokens."""

    access_token: str
    refresh_token: str
    expires_at: int  # Unix timestamp
    token_type: TokenType


def encode_oauth_tokens(t: OAuthTokens) -> JSONObject:
    """Encode OAuthTokens to JSON-serializable dict."""
    result: JSONObject = {
        "access_token": t["access_token"],
        "refresh_token": t["refresh_token"],
        "expires_at": t["expires_at"],
        "token_type": t["token_type"],
    }
    return result


def decode_oauth_tokens(data: JSONObject) -> OAuthTokens:
    """Decode OAuthTokens from dict with validation."""
    return OAuthTokens(
        access_token=require_str(data, "access_token"),
        refresh_token=require_str(data, "refresh_token"),
        expires_at=require_int(data, "expires_at"),
        token_type=_require_token_type(data, "token_type"),
    )


# =============================================================================
# Competition Tracking Types
# =============================================================================


class TrackedCompetition(TypedDict):
    """A competition being tracked for calendar sync."""

    id: str  # Unique ID
    source: CompetitionSource
    name: str  # Competition name
    deadline: str  # ISO 8601 datetime
    url: str  # Link to competition
    project_path: str | None  # e.g., "libs/cleargbm"
    calendar_event_id: str | None  # Created event ID (None if not synced)
    reminders: tuple[int, ...]  # Minutes before deadline


def encode_tracked_competition(c: TrackedCompetition) -> JSONObject:
    """Encode TrackedCompetition to JSON-serializable dict."""
    reminders_list: list[JSONValue] = list(c["reminders"])
    result: JSONObject = {
        "id": c["id"],
        "source": c["source"],
        "name": c["name"],
        "deadline": c["deadline"],
        "url": c["url"],
        "project_path": c["project_path"],
        "calendar_event_id": c["calendar_event_id"],
        "reminders": reminders_list,
    }
    return result


def decode_tracked_competition(data: JSONObject) -> TrackedCompetition:
    """Decode TrackedCompetition from dict with validation."""
    return TrackedCompetition(
        id=require_str(data, "id"),
        source=_require_competition_source(data, "source"),
        name=require_str(data, "name"),
        deadline=require_str(data, "deadline"),
        url=require_str(data, "url"),
        project_path=optional_str(data, "project_path"),
        calendar_event_id=optional_str(data, "calendar_event_id"),
        reminders=tuple(_require_list_int(data, "reminders")),
    )


class CompetitionsFile(TypedDict):
    """Root structure for competitions JSON file."""

    competitions: tuple[TrackedCompetition, ...]


def encode_competitions_file(f: CompetitionsFile) -> JSONObject:
    """Encode CompetitionsFile to JSON-serializable dict."""
    comps_list: list[JSONValue] = []
    for c in f["competitions"]:
        comps_list.append(encode_tracked_competition(c))
    result: JSONObject = {
        "competitions": comps_list,
    }
    return result


def decode_competitions_file(data: JSONObject) -> CompetitionsFile:
    """Decode CompetitionsFile from dict with validation."""
    comps_raw = require_list(data, "competitions")
    return CompetitionsFile(
        competitions=tuple(
            decode_tracked_competition(_require_dict_value(c, f"competitions[{i}]"))
            for i, c in enumerate(comps_raw)
        ),
    )


# =============================================================================
# Google API Response Types (for decoding API responses)
# =============================================================================


class GoogleCredentialsFile(TypedDict):
    """Structure of Google OAuth credentials JSON file (downloaded from console)."""

    installed: GoogleInstalledCredentials


class GoogleInstalledCredentials(TypedDict):
    """The 'installed' section of Google credentials file."""

    client_id: str
    client_secret: str
    redirect_uris: tuple[str, ...]


def decode_google_credentials_file(data: JSONObject) -> GoogleCredentialsFile:
    """Decode Google credentials file with validation."""
    installed_raw = data.get("installed")
    installed = _require_dict_value(installed_raw, "installed")
    redirect_uris_raw = require_list(installed, "redirect_uris")
    redirect_uris: list[str] = []
    for i, uri in enumerate(redirect_uris_raw):
        if not isinstance(uri, str):
            raise JSONTypeError(
                f"Field 'redirect_uris[{i}]' must be a string, got {type(uri).__name__}"
            )
        redirect_uris.append(uri)
    return GoogleCredentialsFile(
        installed=GoogleInstalledCredentials(
            client_id=require_str(installed, "client_id"),
            client_secret=require_str(installed, "client_secret"),
            redirect_uris=tuple(redirect_uris),
        ),
    )


class GoogleTokenResponse(TypedDict):
    """Response from Google OAuth token endpoint."""

    access_token: str
    refresh_token: str | None  # Only on initial auth, not refresh
    expires_in: int  # Seconds until expiration
    token_type: str


def decode_google_token_response(data: JSONObject) -> GoogleTokenResponse:
    """Decode Google token response with validation."""
    return GoogleTokenResponse(
        access_token=require_str(data, "access_token"),
        refresh_token=optional_str(data, "refresh_token"),
        expires_in=require_int(data, "expires_in"),
        token_type=require_str(data, "token_type"),
    )
