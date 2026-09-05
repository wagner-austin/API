"""Google Calendar API integration for competition deadline tracking.

This library provides:
- OAuth 2.0 authentication with Google Calendar API
- Calendar event creation and management
- Competition deadline tracking with automatic reminders
- TypedDict-based strict typing throughout

Example usage:
    from platform_calendar import (
        load_or_authorize,
        google_calendar_client,
        make_competition,
        sync_competition,
        load_competitions,
        save_competitions,
    )

    # Authenticate (opens browser first time)
    tokens = load_or_authorize()
    client = google_calendar_client(tokens=tokens)

    # Create and sync a competition
    comp = make_competition(
        competition_id="devpost-visaverse-2025",
        source="devpost",
        name="VisaVerse AI Hackathon",
        deadline="2025-12-26T22:00:00Z",
        url="https://devpost.com/...",
        project_path="services/grandma-api",
    )
    synced = sync_competition(client, competition=comp)
"""

# Errors - from platform_core
from platform_core.errors import AppError, CalendarErrorCode

# OAuth types - from platform_core
from platform_core.oauth_types import (
    OAuthCredentials,
    OAuthTokenResponse,
    OAuthTokens,
    TokenType,
    decode_oauth_credentials,
    decode_oauth_token_response,
    decode_oauth_tokens,
    encode_oauth_credentials,
    encode_oauth_token_response,
    encode_oauth_tokens,
)

# Auth
from platform_calendar.auth import (
    authorize,
    build_auth_url,
    exchange_code_for_tokens,
    get_valid_tokens,
    is_token_expired,
    load_or_authorize,
    refresh_access_token,
)

# Client
from platform_calendar.client import google_calendar_client

# Competitions
from platform_calendar.competitions import (
    add_competition,
    create_competition_event,
    get_competition,
    load_competitions,
    make_competition,
    remove_competition,
    save_competitions,
    sync_all_competitions,
    sync_competition,
    update_competition,
)

# Config
from platform_calendar.config import (
    CALENDAR_SCOPES,
    GOOGLE_AUTH_URL,
    GOOGLE_CALENDAR_API_BASE,
    GOOGLE_TOKEN_URL,
    get_competitions_path,
    get_credentials_path,
    get_tokens_path,
    load_credentials,
)

# Testing
from platform_calendar.fakes import (
    FakeCalendarClient,
    make_fake_calendar,
    make_fake_console,
    make_fake_credentials,
    make_fake_current_time,
    make_fake_event,
    make_fake_file_system,
    make_fake_http_get,
    make_fake_http_send,
    make_fake_no_tokens,
    make_fake_tokens,
    make_raising_http_get,
    make_raising_http_send,
)
from platform_calendar.testing import (
    CalendarClientProtocol,
    HooksContainer,
    HTTPErrorProtocol,
    hooks,
    reset_hooks,
)
from platform_calendar.types import (
    DEFAULT_REMINDERS,
    CalendarAccessRole,
    CalendarEvent,
    CalendarListItem,
    CompetitionsFile,
    CompetitionSource,
    EventDateTime,
    EventReminders,
    EventStatus,
    GoogleCredentialsFile,
    GoogleInstalledCredentials,
    ReminderMethod,
    ReminderOverride,
    TrackedCompetition,
    decode_calendar_event,
    decode_calendar_list_item,
    decode_competitions_file,
    decode_event_datetime,
    decode_event_reminders,
    decode_google_credentials_file,
    decode_reminder_override,
    decode_tracked_competition,
    encode_calendar_event,
    encode_calendar_list_item,
    encode_competitions_file,
    encode_event_datetime,
    encode_event_reminders,
    encode_reminder_override,
    encode_tracked_competition,
    is_all_day_event,
)

__all__ = [
    "CALENDAR_SCOPES",
    "DEFAULT_REMINDERS",
    "GOOGLE_AUTH_URL",
    "GOOGLE_CALENDAR_API_BASE",
    "GOOGLE_TOKEN_URL",
    "AppError",
    "CalendarAccessRole",
    "CalendarClientProtocol",
    "CalendarErrorCode",
    "CalendarEvent",
    "CalendarListItem",
    "CompetitionSource",
    "CompetitionsFile",
    "EventDateTime",
    "EventReminders",
    "EventStatus",
    "FakeCalendarClient",
    "GoogleCredentialsFile",
    "GoogleInstalledCredentials",
    "HTTPErrorProtocol",
    "HooksContainer",
    "OAuthCredentials",
    "OAuthTokenResponse",
    "OAuthTokens",
    "ReminderMethod",
    "ReminderOverride",
    "TokenType",
    "TrackedCompetition",
    "add_competition",
    "authorize",
    "build_auth_url",
    "create_competition_event",
    "decode_calendar_event",
    "decode_calendar_list_item",
    "decode_competitions_file",
    "decode_event_datetime",
    "decode_event_reminders",
    "decode_google_credentials_file",
    "decode_oauth_credentials",
    "decode_oauth_token_response",
    "decode_oauth_tokens",
    "decode_reminder_override",
    "decode_tracked_competition",
    "encode_calendar_event",
    "encode_calendar_list_item",
    "encode_competitions_file",
    "encode_event_datetime",
    "encode_event_reminders",
    "encode_oauth_credentials",
    "encode_oauth_token_response",
    "encode_oauth_tokens",
    "encode_reminder_override",
    "encode_tracked_competition",
    "exchange_code_for_tokens",
    "get_competition",
    "get_competitions_path",
    "get_credentials_path",
    "get_tokens_path",
    "get_valid_tokens",
    "google_calendar_client",
    "hooks",
    "is_all_day_event",
    "is_token_expired",
    "load_competitions",
    "load_credentials",
    "load_or_authorize",
    "make_competition",
    "make_fake_calendar",
    "make_fake_console",
    "make_fake_credentials",
    "make_fake_current_time",
    "make_fake_event",
    "make_fake_file_system",
    "make_fake_http_get",
    "make_fake_http_send",
    "make_fake_no_tokens",
    "make_fake_tokens",
    "make_raising_http_get",
    "make_raising_http_send",
    "refresh_access_token",
    "remove_competition",
    "reset_hooks",
    "save_competitions",
    "sync_all_competitions",
    "sync_competition",
    "update_competition",
]
