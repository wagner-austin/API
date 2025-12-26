"""Test utilities, fakes, and hooks for platform_calendar."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from platform_core.errors import AppError, CalendarErrorCode

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
    """Protocol for Google Calendar client."""

    def list_calendars(self) -> tuple[CalendarListItem, ...]:
        """List all calendars for the authenticated user."""
        ...

    def get_events(
        self,
        *,
        calendar_id: str,
        time_min: str,
        time_max: str,
    ) -> tuple[CalendarEvent, ...]:
        """Get events in a time range."""
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
    ) -> CalendarEvent:
        """Create a new calendar event."""
        ...

    def update_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
        summary: str | None = None,
        description: str | None = None,
    ) -> CalendarEvent:
        """Update an existing calendar event."""
        ...

    def delete_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> None:
        """Delete a calendar event."""
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


# =============================================================================
# Hooks Container
# =============================================================================


class HooksContainer:
    """Container for dependency injection hooks."""

    http_get: HttpGetHook
    http_post: HttpPostHook
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


def _prod_load_tokens(path: str | None = None) -> OAuthTokens | None:
    """Production token loader - reads from ~/.google/calendar_tokens.json."""
    from pathlib import Path

    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_calendar.types import decode_oauth_tokens

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
    from pathlib import Path

    from platform_core.json_utils import dump_json_str

    from platform_calendar.types import encode_oauth_tokens

    tokens_path = Path(path) if path else Path.home() / ".google" / "calendar_tokens.json"
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    content = dump_json_str(encode_oauth_tokens(tokens), indent=2)
    tokens_path.write_text(content, encoding="utf-8")


def _prod_load_credentials(path: str | None = None) -> OAuthCredentials:
    """Production credentials loader - reads from ~/.google/calendar_credentials.json."""
    from pathlib import Path

    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_calendar.types import OAuthCredentials, decode_google_credentials_file

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
    from pathlib import Path

    return Path(path).read_text(encoding="utf-8")


def _prod_write_file(path: str, content: str) -> None:
    """Production file writer."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _prod_file_exists(path: str) -> bool:
    """Production file exists check."""
    from pathlib import Path

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


def _init_production_hooks() -> None:
    """Initialize hooks with production implementations."""
    hooks.http_get = _prod_http_get
    hooks.http_post = _prod_http_post
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


# Initialize on module load
_init_production_hooks()


def reset_hooks() -> None:
    """Reset all hooks to production implementations (for test teardown)."""
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
        """Add a fake calendar for testing."""
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
        """Add a fake event for testing."""
        if calendar_id not in self._events:
            self._events[calendar_id] = []
        self._events[calendar_id].append(event)

    def get_created_events(self) -> list[CalendarEvent]:
        """Get all events created via create_event()."""
        return list(self._created_events)

    def get_updated_events(self) -> list[CalendarEvent]:
        """Get all events updated via update_event()."""
        return list(self._updated_events)

    def get_deleted_events(self) -> list[tuple[str, str]]:
        """Get all (calendar_id, event_id) pairs deleted via delete_event()."""
        return list(self._deleted_events)

    # -------------------------------------------------------------------------
    # Protocol Implementation
    # -------------------------------------------------------------------------

    def list_calendars(self) -> tuple[CalendarListItem, ...]:
        """List all calendars."""
        return tuple(self._calendars)

    def get_events(
        self,
        *,
        calendar_id: str,
        time_min: str,
        time_max: str,
    ) -> tuple[CalendarEvent, ...]:
        """Get events in a time range."""
        events = self._events.get(calendar_id, [])
        # Simple filtering - in real implementation would compare datetimes
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
    ) -> CalendarEvent:
        """Create a new calendar event."""
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
    ) -> CalendarEvent:
        """Update an existing calendar event."""
        events = self._events.get(calendar_id, [])
        for i, event in enumerate(events):
            if event["id"] == event_id:
                updated = CalendarEvent(
                    id=event["id"],
                    summary=summary if summary is not None else event["summary"],
                    description=description if description is not None else event["description"],
                    start=event["start"],
                    end=event["end"],
                    status=event["status"],
                    reminders=event["reminders"],
                )
                self._events[calendar_id][i] = updated
                self._updated_events.append(updated)
                return updated

        # Event not found - return a placeholder
        placeholder = CalendarEvent(
            id=event_id,
            summary=summary if summary is not None else "",
            description=description if description is not None else "",
            start=EventDateTime(dateTime="", timeZone="UTC"),
            end=EventDateTime(dateTime="", timeZone="UTC"),
            status="confirmed",
            reminders=EventReminders(useDefault=True, overrides=()),
        )
        self._updated_events.append(placeholder)
        return placeholder

    def delete_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> None:
        """Delete a calendar event."""
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
) -> CalendarEvent:
    """Create a fake CalendarEvent for testing."""
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
    "HttpGetHook",
    "HttpPostHook",
    "LoadCredentialsHook",
    "LoadTokensHook",
    "OpenBrowserHook",
    "ReadFileHook",
    "SaveTokensHook",
    "WriteFileHook",
    "hooks",
    "make_fake_calendar",
    "make_fake_console",
    "make_fake_credentials",
    "make_fake_current_time",
    "make_fake_event",
    "make_fake_file_system",
    "make_fake_http_get",
    "make_fake_http_post",
    "make_fake_no_tokens",
    "make_fake_tokens",
    "make_raising_http_get",
    "make_raising_http_post",
    "reset_hooks",
]
