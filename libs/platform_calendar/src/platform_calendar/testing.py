"""Test utilities, fakes, and hooks for platform_calendar.

Re-exports common OAuth test utilities from platform_core.oauth_testing
for convenience. Calendar-specific fakes and hooks are defined here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from platform_core.errors import AppError, CalendarErrorCode
from platform_core.json_utils import JSONObject

# Re-export OAuth testing utilities from platform_core for convenience.
# Note: platform_calendar has its own make_fake_credentials/make_fake_tokens
# that create hook functions, so we don't re-export the platform_core versions
# which return data directly.
from platform_core.oauth_testing import make_error_response_json, make_token_response_json
from rich.console import Console

from platform_calendar.types import (
    CalendarEvent,
    CalendarListItem,
    EventDateTime,
    EventReminders,
    OAuthCredentials,
    OAuthTokens,
    ReminderOverride,
)

# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class CalendarClientProtocol(Protocol):
    """Protocol for Google Calendar client.

    Defines the interface for interacting with Google Calendar API.
    """

    def list_calendars(self) -> tuple[CalendarListItem, ...]:
        """List all calendars for the authenticated user.

        Returns:
            Tuple of CalendarListItem for each accessible calendar.
        """
        ...

    def get_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> CalendarEvent:
        """Get a single event by ID.

        Args:
            calendar_id: Calendar containing the event.
            event_id: ID of the event to retrieve.

        Returns:
            The CalendarEvent.

        Raises:
            AppError[CalendarErrorCode]: If event not found.
        """
        ...

    def get_events(
        self,
        *,
        calendar_id: str,
        time_min: str,
        time_max: str,
    ) -> tuple[CalendarEvent, ...]:
        """Get events in a time range.

        Args:
            calendar_id: Calendar to query.
            time_min: Start of time range (RFC 3339).
            time_max: End of time range (RFC 3339).

        Returns:
            Tuple of CalendarEvent within the time range.
        """
        ...

    def create_event(
        self,
        *,
        calendar_id: str,
        summary: str,
        description: str,
        start: EventDateTime,
        end: EventDateTime,
        reminders: tuple[int, ...],
        location: str = "",
        recurrence: tuple[str, ...] = (),
    ) -> CalendarEvent:
        """Create a new calendar event.

        Args:
            calendar_id: Calendar to create event in.
            summary: Event title.
            description: Event description.
            start: Event start time.
            end: Event end time.
            reminders: Reminder times in minutes before event.
            location: Event location string.
            recurrence: RRULE strings for recurring events.

        Returns:
            The created CalendarEvent.
        """
        ...

    def update_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
        summary: str | None = None,
        description: str | None = None,
        start: EventDateTime | None = None,
        end: EventDateTime | None = None,
        reminders: tuple[int, ...] | None = None,
        location: str | None = None,
        recurrence: tuple[str, ...] | None = None,
    ) -> CalendarEvent:
        """Update an existing calendar event.

        Args:
            calendar_id: Calendar containing the event.
            event_id: ID of the event to update.
            summary: New event title (None to keep existing).
            description: New description (None to keep existing).
            start: New start time (None to keep existing).
            end: New end time (None to keep existing).
            reminders: New reminder times (None to keep existing).
            location: New location (None to keep existing).
            recurrence: New recurrence rules (None to keep existing).

        Returns:
            The updated CalendarEvent.

        Raises:
            AppError[CalendarErrorCode]: If event not found.
        """
        ...

    def delete_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> None:
        """Delete a calendar event.

        Args:
            calendar_id: Calendar containing the event.
            event_id: ID of the event to delete.

        Raises:
            AppError[CalendarErrorCode]: If event not found.
        """
        ...


# =============================================================================
# HTTP Error Protocol
# =============================================================================


@runtime_checkable
class HTTPErrorProtocol(Protocol):
    """Protocol for HTTP error with code and response body."""

    @property
    def code(self) -> int:
        """HTTP status code."""
        ...

    def read(self) -> bytes:
        """Read response body."""
        ...


# =============================================================================
# Hook Type Definitions
# =============================================================================

HttpGetHook = Callable[[str, dict[str, str]], str]
HttpPostHook = Callable[[str, dict[str, str], str], str]
HttpPatchHook = Callable[[str, dict[str, str], str], str]
HttpDeleteHook = Callable[[str, dict[str, str]], None]
LoadTokensHook = Callable[[], OAuthTokens | None]
SaveTokensHook = Callable[[OAuthTokens], None]
LoadCredentialsHook = Callable[[], OAuthCredentials]
OpenBrowserHook = Callable[[str], None]
CurrentTimeHook = Callable[[], int]
ReadFileHook = Callable[[str], str]
WriteFileHook = Callable[[str, str], None]
FileExistsHook = Callable[[str], bool]
ConsoleOutputHook = Callable[[str], None]
ConsoleInputHook = Callable[[str], str]

# CLI-specific hooks
CliApiGetHook = Callable[[str, str], JSONObject]
CliApiPostHook = Callable[[str, str, JSONObject], JSONObject]
CliApiDeleteHook = Callable[[str, str], None]
CliGetEnvHook = Callable[[str], str | None]
CliSetEnvHook = Callable[[str, str], None]
CliGetNowHook = Callable[[], datetime]
CliPromptAskHook = Callable[[str], str]
CliConfirmAskHook = Callable[[str], bool]
CliGetConsoleHook = Callable[[], Console]


# =============================================================================
# Hooks Container
# =============================================================================


class HooksContainer:
    """Container for dependency injection hooks."""

    http_get: HttpGetHook
    http_post: HttpPostHook
    http_patch: HttpPatchHook
    http_delete: HttpDeleteHook
    load_tokens: LoadTokensHook
    save_tokens: SaveTokensHook
    load_credentials: LoadCredentialsHook
    open_browser: OpenBrowserHook
    current_time: CurrentTimeHook
    read_file: ReadFileHook
    write_file: WriteFileHook
    file_exists: FileExistsHook
    console_output: ConsoleOutputHook
    console_input: ConsoleInputHook

    # CLI-specific hooks
    cli_api_get: CliApiGetHook
    cli_api_post: CliApiPostHook
    cli_api_delete: CliApiDeleteHook
    cli_get_env: CliGetEnvHook
    cli_set_env: CliSetEnvHook
    cli_get_now: CliGetNowHook
    cli_prompt_ask: CliPromptAskHook
    cli_confirm_ask: CliConfirmAskHook
    cli_get_console: CliGetConsoleHook

    def reset(self) -> None:
        """Restore every hook to its production implementation.

        The restoration `reset_hooks()` performs, exposed as a method so an
        autouse fixture can name the container it protects.
        """
        reset_hooks()


hooks = HooksContainer()


# =============================================================================
# Production Implementations
# =============================================================================


def _prod_http_get(url: str, headers: dict[str, str]) -> str:
    """Production HTTP GET using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url)
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    try:
        body = response.read()
        return body.decode("utf-8")
    finally:
        response.close()


def _prod_http_post(url: str, headers: dict[str, str], body: str) -> str:
    """Production HTTP POST using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, data=body.encode("utf-8"), method="POST")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    try:
        response_body = response.read()
        return response_body.decode("utf-8")
    finally:
        response.close()


def _prod_http_patch(url: str, headers: dict[str, str], body: str) -> str:
    """Production HTTP PATCH using urllib.

    Args:
        url: URL to send PATCH request to.
        headers: HTTP headers to include.
        body: Request body as string.

    Returns:
        Response body as string.
    """
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, data=body.encode("utf-8"), method="PATCH")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    try:
        response_body = response.read()
        return response_body.decode("utf-8")
    finally:
        response.close()


def _prod_http_delete(url: str, headers: dict[str, str]) -> None:
    """Production HTTP DELETE using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, method="DELETE")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    response.close()


def _prod_load_tokens(path: str | None = None) -> OAuthTokens | None:
    """Production token loader.

    Loads OAuth tokens from environment variables or file.

    Environment variables (checked first):
        GOOGLE_CALENDAR_ACCESS_TOKEN: OAuth access token
        GOOGLE_CALENDAR_REFRESH_TOKEN: OAuth refresh token
        GOOGLE_CALENDAR_TOKEN_EXPIRES_AT: Token expiry (Unix timestamp as string)

    If any token env var is set, all must be set.
    If no env vars are set, reads from file path.

    Args:
        path: Optional file path. Defaults to ~/.google/calendar_tokens.json

    Returns:
        OAuthTokens if found, None if no tokens configured.

    Raises:
        AppError[CalendarErrorCode]: If tokens are partially configured in environment.
    """

    from platform_core.config import config_test_hooks
    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_calendar.types import OAuthTokens, decode_oauth_tokens

    # Check environment variables first using centralized hook
    env_access_token = config_test_hooks.get_env("GOOGLE_CALENDAR_ACCESS_TOKEN")
    env_refresh_token = config_test_hooks.get_env("GOOGLE_CALENDAR_REFRESH_TOKEN")
    env_expires_at = config_test_hooks.get_env("GOOGLE_CALENDAR_TOKEN_EXPIRES_AT")

    # If any token env var is set, validate all are present
    if env_access_token is not None or env_refresh_token is not None or env_expires_at is not None:
        missing: list[str] = []
        if env_access_token is None:
            missing.append("GOOGLE_CALENDAR_ACCESS_TOKEN")
        if env_refresh_token is None:
            missing.append("GOOGLE_CALENDAR_REFRESH_TOKEN")
        if env_expires_at is None:
            missing.append("GOOGLE_CALENDAR_TOKEN_EXPIRES_AT")
        if missing:
            msg = f"Partial tokens in environment. Missing: {', '.join(missing)}"
            raise AppError(CalendarErrorCode.AUTH_FAILED, msg, http_status=401)
        # All env vars present - narrow types for mypy
        assert env_access_token is not None
        assert env_refresh_token is not None
        assert env_expires_at is not None
        return OAuthTokens(
            access_token=env_access_token,
            refresh_token=env_refresh_token,
            expires_at=int(env_expires_at),
            token_type="Bearer",
        )

    # No env vars set - read from file
    tokens_path = Path(path) if path else Path.home() / ".google" / "calendar_tokens.json"
    if not tokens_path.exists():
        return None
    content = tokens_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError):
        return None
    return decode_oauth_tokens(data)


def _prod_save_tokens(tokens: OAuthTokens, path: str | None = None) -> None:
    """Production token saver - writes to ~/.google/calendar_tokens.json."""

    from platform_core.json_utils import dump_json_str

    from platform_calendar.types import encode_oauth_tokens

    tokens_path = Path(path) if path else Path.home() / ".google" / "calendar_tokens.json"
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    content = dump_json_str(encode_oauth_tokens(tokens), indent=2)
    tokens_path.write_text(content, encoding="utf-8")


def _prod_load_credentials(path: str | None = None) -> OAuthCredentials:
    """Production credentials loader.

    Loads OAuth credentials from environment variables or file.

    Environment variables (checked first):
        GOOGLE_CALENDAR_CLIENT_ID: OAuth client ID
        GOOGLE_CALENDAR_CLIENT_SECRET: OAuth client secret
        GOOGLE_CALENDAR_REDIRECT_URI: Redirect URI (defaults to "http://localhost")

    If any credential env var is set, all required ones must be set.
    If no env vars are set, reads from file path.

    Args:
        path: Optional file path. Defaults to ~/.google/calendar_credentials.json

    Returns:
        OAuthCredentials with client_id, client_secret, redirect_uri.

    Raises:
        AppError[CalendarErrorCode]: If credentials not found or partially configured.
    """

    from platform_core.config import config_test_hooks
    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_calendar.types import OAuthCredentials, decode_google_credentials_file

    # Check environment variables first using centralized hook
    env_client_id = config_test_hooks.get_env("GOOGLE_CALENDAR_CLIENT_ID")
    env_client_secret = config_test_hooks.get_env("GOOGLE_CALENDAR_CLIENT_SECRET")
    env_redirect_uri = config_test_hooks.get_env("GOOGLE_CALENDAR_REDIRECT_URI")

    # If any credential env var is set, validate all required ones are present
    if env_client_id is not None or env_client_secret is not None:
        missing: list[str] = []
        if env_client_id is None:
            missing.append("GOOGLE_CALENDAR_CLIENT_ID")
        if env_client_secret is None:
            missing.append("GOOGLE_CALENDAR_CLIENT_SECRET")
        if missing:
            msg = f"Partial credentials in environment. Missing: {', '.join(missing)}"
            raise AppError(CalendarErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401)
        # All required env vars present - narrow types for mypy
        assert env_client_id is not None
        assert env_client_secret is not None
        return OAuthCredentials(
            client_id=env_client_id,
            client_secret=env_client_secret,
            redirect_uri=env_redirect_uri if env_redirect_uri is not None else "http://localhost",
        )

    # No env vars set - read from file
    creds_path = Path(path) if path else Path.home() / ".google" / "calendar_credentials.json"
    if not creds_path.exists():
        msg = f"Credentials file not found at {creds_path}"
        raise AppError(CalendarErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401)
    content = creds_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Credentials file is not valid JSON: {e}"
        raise AppError(CalendarErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401) from e
    google_creds = decode_google_credentials_file(data)
    installed = google_creds["installed"]
    redirect_uri = installed["redirect_uris"][0] if installed["redirect_uris"] else ""
    return OAuthCredentials(
        client_id=installed["client_id"],
        client_secret=installed["client_secret"],
        redirect_uri=redirect_uri,
    )


def _prod_open_browser(
    url: str,
    _opener: Callable[[str], bool] | None = None,
) -> None:
    """Production browser opener."""
    import webbrowser

    opener = _opener if _opener is not None else webbrowser.open
    opener(url)


def _prod_current_time() -> int:
    """Production current time in seconds since epoch."""
    import time

    return int(time.time())


def _prod_read_file(path: str) -> str:
    """Production file reader."""

    return Path(path).read_text(encoding="utf-8")


def _prod_write_file(path: str, content: str) -> None:
    """Production file writer."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _prod_file_exists(path: str) -> bool:
    """Production file exists check."""

    return Path(path).exists()


def _prod_console_output(message: str) -> None:
    """Production console output using print."""
    import sys

    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _prod_console_input(
    prompt: str,
    _input_func: Callable[[str], str] | None = None,
) -> str:
    """Production console input using input."""
    input_func = _input_func if _input_func is not None else input
    return input_func(prompt)


# =============================================================================
# CLI Production Implementations
# =============================================================================

# Module-level cache for CLI environment and console
_cli_env_loaded: bool = False
_cli_env_cache: dict[str, str] = {}
_cli_default_console: Console | None = None


def _prod_cli_api_get(access_token: str, url: str) -> JSONObject:
    """Production CLI API GET request.

    Args:
        access_token: OAuth access token.
        url: Full API URL.

    Returns:
        Parsed JSON response.
    """
    import http.client
    import urllib.request

    from platform_core.json_utils import load_json_str, narrow_json_to_dict

    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    resp: http.client.HTTPResponse = urllib.request.urlopen(req)
    body = resp.read().decode("utf-8")
    raw = load_json_str(body)
    return narrow_json_to_dict(raw)


def _prod_cli_api_post(access_token: str, url: str, request_body: JSONObject) -> JSONObject:
    """Production CLI API POST request.

    Args:
        access_token: OAuth access token.
        url: Full API URL.
        request_body: JSON request body.

    Returns:
        Parsed JSON response.
    """
    import http.client
    import urllib.request

    from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

    data = dump_json_str(request_body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    resp: http.client.HTTPResponse = urllib.request.urlopen(req)
    body = resp.read().decode("utf-8")
    raw = load_json_str(body)
    return narrow_json_to_dict(raw)


def _prod_cli_api_delete(access_token: str, url: str) -> None:
    """Production CLI API DELETE request.

    Args:
        access_token: OAuth access token.
        url: Full API URL.
    """
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        method="DELETE",
    )
    urllib.request.urlopen(req)


def _prod_cli_get_env(key: str) -> str | None:
    """Production CLI environment variable getter.

    Loads from .env file in the platform_calendar package directory.

    Args:
        key: Environment variable name.

    Returns:
        Value if found, None otherwise.
    """
    import os

    global _cli_env_loaded, _cli_env_cache

    if not _cli_env_loaded:
        # Load from .env file relative to this module
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        k, v = line.strip().split("=", 1)
                        _cli_env_cache[k] = v
        _cli_env_loaded = True

    return _cli_env_cache.get(key)


def _prod_cli_set_env(key: str, value: str) -> None:
    """Production CLI environment variable setter.

    Updates the in-memory cache with the new value.

    Args:
        key: Environment variable name.
        value: Value to set.
    """
    global _cli_env_cache
    _cli_env_cache[key] = value


def _prod_cli_get_now() -> datetime:
    """Production CLI current datetime.

    Returns:
        Current datetime.
    """
    return datetime.now()


def _prod_cli_prompt_ask(
    message: str,
    _prompt_func: Callable[[str], str] | None = None,
) -> str:
    """Production CLI prompt using Rich.

    Args:
        message: Prompt message.
        _prompt_func: Optional override for testing.

    Returns:
        User input.
    """
    from rich.prompt import Prompt

    prompt_func = _prompt_func if _prompt_func is not None else Prompt.ask
    return prompt_func(message)


def _prod_cli_confirm_ask(
    message: str,
    _confirm_func: Callable[[str], bool] | None = None,
) -> bool:
    """Production CLI confirm using Rich.

    Args:
        message: Prompt message.
        _confirm_func: Optional override for testing.

    Returns:
        True if confirmed.
    """
    from rich.prompt import Confirm

    confirm_func = _confirm_func if _confirm_func is not None else Confirm.ask
    return confirm_func(message)


def _prod_cli_get_console() -> Console:
    """Production CLI console getter.

    Returns:
        Rich Console instance (cached).
    """
    global _cli_default_console

    if _cli_default_console is None:
        _cli_default_console = Console()
    return _cli_default_console


def _init_production_hooks() -> None:
    """Initialize hooks with production implementations."""
    hooks.http_get = _prod_http_get
    hooks.http_post = _prod_http_post
    hooks.http_patch = _prod_http_patch
    hooks.http_delete = _prod_http_delete
    hooks.load_tokens = _prod_load_tokens
    hooks.save_tokens = _prod_save_tokens
    hooks.load_credentials = _prod_load_credentials
    hooks.open_browser = _prod_open_browser
    hooks.current_time = _prod_current_time
    hooks.read_file = _prod_read_file
    hooks.write_file = _prod_write_file
    hooks.file_exists = _prod_file_exists
    hooks.console_output = _prod_console_output
    hooks.console_input = _prod_console_input
    # CLI hooks
    hooks.cli_api_get = _prod_cli_api_get
    hooks.cli_api_post = _prod_cli_api_post
    hooks.cli_api_delete = _prod_cli_api_delete
    hooks.cli_get_env = _prod_cli_get_env
    hooks.cli_set_env = _prod_cli_set_env
    hooks.cli_get_now = _prod_cli_get_now
    hooks.cli_prompt_ask = _prod_cli_prompt_ask
    hooks.cli_confirm_ask = _prod_cli_confirm_ask
    hooks.cli_get_console = _prod_cli_get_console


# Initialize on module load
_init_production_hooks()


def reset_hooks() -> None:
    """Reset all hooks to production implementations (for test teardown)."""
    global _cli_env_loaded, _cli_env_cache, _cli_default_console
    _cli_env_loaded = False
    _cli_env_cache = {}
    _cli_default_console = None
    _init_production_hooks()


# =============================================================================
# Fake Calendar Client
# =============================================================================


class FakeCalendarClient(CalendarClientProtocol):
    """In-memory fake calendar client for testing."""

    def __init__(self) -> None:
        """Initialize the fake client with empty state."""
        self._calendars: list[CalendarListItem] = []
        self._events: dict[str, list[CalendarEvent]] = {}
        self._next_event_id: int = 1
        self._created_events: list[CalendarEvent] = []
        self._updated_events: list[CalendarEvent] = []
        self._deleted_events: list[tuple[str, str]] = []

    # -------------------------------------------------------------------------
    # Test Helpers
    # -------------------------------------------------------------------------

    def add_calendar(
        self,
        *,
        calendar_id: str,
        summary: str,
        description: str = "",
        primary: bool = False,
        time_zone: str = "UTC",
    ) -> None:
        """Add a fake calendar for testing.

        Args:
            calendar_id: Unique calendar ID.
            summary: Calendar name.
            description: Calendar description.
            primary: Whether this is the primary calendar.
            time_zone: Calendar timezone.
        """
        item = CalendarListItem(
            id=calendar_id,
            summary=summary,
            description=description,
            primary=primary,
            accessRole="owner",
            timeZone=time_zone,
        )
        self._calendars.append(item)
        self._events[calendar_id] = []

    def add_event(self, *, calendar_id: str, event: CalendarEvent) -> None:
        """Add a fake event for testing.

        Args:
            calendar_id: Calendar to add event to.
            event: Event to add.
        """
        if calendar_id not in self._events:
            self._events[calendar_id] = []
        self._events[calendar_id].append(event)

    def get_created_events(self) -> list[CalendarEvent]:
        """Get all events created via create_event().

        Returns:
            List of created events.
        """
        return list(self._created_events)

    def get_updated_events(self) -> list[CalendarEvent]:
        """Get all events updated via update_event().

        Returns:
            List of updated events.
        """
        return list(self._updated_events)

    def get_deleted_events(self) -> list[tuple[str, str]]:
        """Get all (calendar_id, event_id) pairs deleted via delete_event().

        Returns:
            List of (calendar_id, event_id) tuples.
        """
        return list(self._deleted_events)

    # -------------------------------------------------------------------------
    # Protocol Implementation
    # -------------------------------------------------------------------------

    def list_calendars(self) -> tuple[CalendarListItem, ...]:
        """List all calendars.

        Returns:
            Tuple of all calendars.
        """
        return tuple(self._calendars)

    def get_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> CalendarEvent:
        """Get a single event by ID.

        Args:
            calendar_id: Calendar containing the event.
            event_id: ID of the event to retrieve.

        Returns:
            The CalendarEvent.

        Raises:
            AppError[CalendarErrorCode]: If event not found.
        """
        events = self._events.get(calendar_id, [])
        for event in events:
            if event["id"] == event_id:
                return event
        msg = f"Event '{event_id}' not found in calendar '{calendar_id}'"
        raise AppError(CalendarErrorCode.EVENT_NOT_FOUND, msg, http_status=404)

    def get_events(
        self,
        *,
        calendar_id: str,
        time_min: str,
        time_max: str,
    ) -> tuple[CalendarEvent, ...]:
        """Get events in a time range.

        Args:
            calendar_id: Calendar to query.
            time_min: Start of time range.
            time_max: End of time range.

        Returns:
            Tuple of events (simple filtering, no datetime comparison).
        """
        events = self._events.get(calendar_id, [])
        return tuple(events)

    def create_event(
        self,
        *,
        calendar_id: str,
        summary: str,
        description: str,
        start: EventDateTime,
        end: EventDateTime,
        reminders: tuple[int, ...],
        location: str = "",
        recurrence: tuple[str, ...] = (),
    ) -> CalendarEvent:
        """Create a new calendar event.

        Args:
            calendar_id: Calendar to create event in.
            summary: Event title.
            description: Event description.
            start: Event start time.
            end: Event end time.
            reminders: Reminder times in minutes.
            location: Event location string.
            recurrence: RRULE strings for recurring events.

        Returns:
            The created CalendarEvent.
        """
        event_id = f"fake_event_{self._next_event_id}"
        self._next_event_id += 1

        overrides: list[ReminderOverride] = []
        for minutes in reminders:
            overrides.append(ReminderOverride(method="popup", minutes=minutes))

        event = CalendarEvent(
            id=event_id,
            summary=summary,
            description=description,
            start=start,
            end=end,
            status="confirmed",
            reminders=EventReminders(
                useDefault=False,
                overrides=tuple(overrides),
            ),
            location=location,
            recurrence=recurrence,
        )

        if calendar_id not in self._events:
            self._events[calendar_id] = []
        self._events[calendar_id].append(event)
        self._created_events.append(event)

        return event

    def update_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
        summary: str | None = None,
        description: str | None = None,
        start: EventDateTime | None = None,
        end: EventDateTime | None = None,
        reminders: tuple[int, ...] | None = None,
        location: str | None = None,
        recurrence: tuple[str, ...] | None = None,
    ) -> CalendarEvent:
        """Update an existing calendar event.

        Args:
            calendar_id: Calendar containing the event.
            event_id: ID of the event to update.
            summary: New event title (None to keep existing).
            description: New description (None to keep existing).
            start: New start time (None to keep existing).
            end: New end time (None to keep existing).
            reminders: New reminder times (None to keep existing).
            location: New location (None to keep existing).
            recurrence: New recurrence rules (None to keep existing).

        Returns:
            The updated CalendarEvent.

        Raises:
            AppError[CalendarErrorCode]: If event not found.
        """
        events = self._events.get(calendar_id, [])
        for i, event in enumerate(events):
            if event["id"] == event_id:
                # Build new reminders if provided
                new_reminders: EventReminders
                if reminders is not None:
                    overrides: list[ReminderOverride] = []
                    for minutes in reminders:
                        overrides.append(ReminderOverride(method="popup", minutes=minutes))
                    new_reminders = EventReminders(
                        useDefault=False,
                        overrides=tuple(overrides),
                    )
                else:
                    new_reminders = event["reminders"]

                updated = CalendarEvent(
                    id=event["id"],
                    summary=summary if summary is not None else event["summary"],
                    description=description if description is not None else event["description"],
                    start=start if start is not None else event["start"],
                    end=end if end is not None else event["end"],
                    status=event["status"],
                    reminders=new_reminders,
                    location=location if location is not None else event["location"],
                    recurrence=recurrence if recurrence is not None else event["recurrence"],
                )
                self._events[calendar_id][i] = updated
                self._updated_events.append(updated)
                return updated

        # Event not found - raise error
        msg = f"Event '{event_id}' not found in calendar '{calendar_id}'"
        raise AppError(CalendarErrorCode.EVENT_NOT_FOUND, msg, http_status=404)

    def delete_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> None:
        """Delete a calendar event.

        Args:
            calendar_id: Calendar containing the event.
            event_id: ID of the event to delete.
        """
        self._deleted_events.append((calendar_id, event_id))
        events = self._events.get(calendar_id, [])
        self._events[calendar_id] = [e for e in events if e["id"] != event_id]


# =============================================================================
# Factory Helpers for Tests
# =============================================================================


def make_fake_http_get(response: str) -> HttpGetHook:
    """Create a hook that returns a fixed response."""

    def _hook(url: str, headers: dict[str, str]) -> str:
        return response

    return _hook


def make_fake_http_post(response: str) -> HttpPostHook:
    """Create a hook that returns a fixed response."""

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        return response

    return _hook


def make_raising_http_get(exc: BaseException) -> HttpGetHook:
    """Create a hook that raises an exception."""

    def _hook(url: str, headers: dict[str, str]) -> str:
        raise exc

    return _hook


def make_raising_http_post(exc: BaseException) -> HttpPostHook:
    """Create a hook that raises an exception."""

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        raise exc

    return _hook


def make_fake_http_delete() -> HttpDeleteHook:
    """Create a hook that does nothing (successful delete)."""

    def _hook(url: str, headers: dict[str, str]) -> None:
        pass

    return _hook


def make_raising_http_delete(exc: BaseException) -> HttpDeleteHook:
    """Create a hook that raises an exception."""

    def _hook(url: str, headers: dict[str, str]) -> None:
        raise exc

    return _hook


def make_fake_http_patch(response: str) -> HttpPatchHook:
    """Create a hook that returns a fixed response.

    Args:
        response: Response body to return.

    Returns:
        HttpPatchHook that returns the fixed response.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        return response

    return _hook


def make_raising_http_patch(exc: BaseException) -> HttpPatchHook:
    """Create a hook that raises an exception.

    Args:
        exc: Exception to raise.

    Returns:
        HttpPatchHook that raises the exception.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        raise exc

    return _hook


def make_fake_tokens(tokens: OAuthTokens) -> LoadTokensHook:
    """Create a hook that returns fixed tokens."""

    def _hook() -> OAuthTokens | None:
        return tokens

    return _hook


def make_fake_no_tokens() -> LoadTokensHook:
    """Create a hook that returns None (no cached tokens)."""

    def _hook() -> OAuthTokens | None:
        return None

    return _hook


def make_fake_credentials(creds: OAuthCredentials) -> LoadCredentialsHook:
    """Create a hook that returns fixed credentials."""

    def _hook() -> OAuthCredentials:
        return creds

    return _hook


def make_fake_current_time(timestamp: int) -> CurrentTimeHook:
    """Create a hook that returns a fixed timestamp."""

    def _hook() -> int:
        return timestamp

    return _hook


def make_fake_file_system(
    files: dict[str, str],
) -> tuple[ReadFileHook, WriteFileHook, FileExistsHook]:
    """Create hooks that use an in-memory file system."""
    storage = dict(files)

    def _read(path: str) -> str:
        if path not in storage:
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        return storage[path]

    def _write(path: str, content: str) -> None:
        storage[path] = content

    def _exists(path: str) -> bool:
        return path in storage

    return _read, _write, _exists


def make_fake_console(inputs: list[str]) -> tuple[ConsoleOutputHook, ConsoleInputHook]:
    """Create hooks for fake console I/O.

    Args:
        inputs: List of strings to return from console_input in order.

    Returns:
        Tuple of (output_hook, input_hook).
    """
    outputs: list[str] = []
    input_index = [0]  # Use list for closure mutation

    def _output(message: str) -> None:
        outputs.append(message)

    def _input(prompt: str) -> str:
        if input_index[0] >= len(inputs):
            return ""
        result = inputs[input_index[0]]
        input_index[0] += 1
        return result

    return _output, _input


def make_fake_event(
    *,
    event_id: str = "test_event_1",
    summary: str = "Test Event",
    description: str = "Test description",
    start_datetime: str = "2025-12-26T14:00:00-08:00",
    end_datetime: str = "2025-12-26T15:00:00-08:00",
    time_zone: str = "America/Los_Angeles",
    status: str = "confirmed",
    location: str = "",
    recurrence: tuple[str, ...] = (),
) -> CalendarEvent:
    """Create a fake CalendarEvent for testing.

    Args:
        event_id: Event ID.
        summary: Event title.
        description: Event description.
        start_datetime: Start time in RFC 3339 format.
        end_datetime: End time in RFC 3339 format.
        time_zone: Timezone for start/end.
        status: Event status (confirmed, tentative, cancelled).
        location: Event location string.
        recurrence: RRULE strings for recurring events.

    Returns:
        CalendarEvent with the specified values.
    """
    event_status: str = status
    return CalendarEvent(
        id=event_id,
        summary=summary,
        description=description,
        start=EventDateTime(dateTime=start_datetime, timeZone=time_zone),
        end=EventDateTime(dateTime=end_datetime, timeZone=time_zone),
        status=(
            "confirmed"
            if event_status == "confirmed"
            else ("tentative" if event_status == "tentative" else "cancelled")
        ),
        reminders=EventReminders(useDefault=True, overrides=()),
        location=location,
        recurrence=recurrence,
    )


def make_fake_calendar(
    *,
    calendar_id: str = "primary",
    summary: str = "Primary Calendar",
    description: str = "",
    primary: bool = True,
    time_zone: str = "America/Los_Angeles",
) -> CalendarListItem:
    """Create a fake CalendarListItem for testing."""
    return CalendarListItem(
        id=calendar_id,
        summary=summary,
        description=description,
        primary=primary,
        accessRole="owner",
        timeZone=time_zone,
    )


__all__ = [
    "CalendarClientProtocol",
    "ConsoleInputHook",
    "ConsoleOutputHook",
    "CurrentTimeHook",
    "FakeCalendarClient",
    "FileExistsHook",
    "HTTPErrorProtocol",
    "HooksContainer",
    "HttpDeleteHook",
    "HttpGetHook",
    "HttpPatchHook",
    "HttpPostHook",
    "LoadCredentialsHook",
    "LoadTokensHook",
    "OpenBrowserHook",
    "ReadFileHook",
    "SaveTokensHook",
    "WriteFileHook",
    "_prod_http_patch",
    "hooks",
    "make_error_response_json",
    "make_fake_calendar",
    "make_fake_console",
    "make_fake_credentials",
    "make_fake_current_time",
    "make_fake_event",
    "make_fake_file_system",
    "make_fake_http_delete",
    "make_fake_http_get",
    "make_fake_http_patch",
    "make_fake_http_post",
    "make_fake_no_tokens",
    "make_fake_tokens",
    "make_raising_http_delete",
    "make_raising_http_get",
    "make_raising_http_patch",
    "make_raising_http_post",
    "make_token_response_json",
    "reset_hooks",
]
