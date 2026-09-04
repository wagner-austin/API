"""Test utilities, fakes, and hooks for platform_calendar.

Re-exports common OAuth test utilities from platform_core.oauth_testing
for convenience. Calendar-specific fakes and hooks are defined here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol, runtime_checkable

from platform_core.json_utils import JSONObject

# Re-export OAuth testing utilities from platform_core for convenience.
# Note: platform_calendar has its own make_fake_credentials/make_fake_tokens
# that create hook functions, so we don't re-export the platform_core versions
# which return data directly.
from rich.console import Console

from platform_calendar._prod_hooks import (
    _prod_cli_api_delete,
    _prod_cli_api_get,
    _prod_cli_api_post,
    _prod_cli_confirm_ask,
    _prod_cli_get_console,
    _prod_cli_get_env,
    _prod_cli_get_now,
    _prod_cli_prompt_ask,
    _prod_cli_set_env,
    _prod_console_input,
    _prod_console_output,
    _prod_current_time,
    _prod_file_exists,
    _prod_http_delete,
    _prod_http_get,
    _prod_http_patch,
    _prod_http_post,
    _prod_load_credentials,
    _prod_load_tokens,
    _prod_open_browser,
    _prod_read_file,
    _prod_save_tokens,
    _prod_write_file,
)
from platform_calendar.types import (
    CalendarEvent,
    CalendarListItem,
    EventDateTime,
    OAuthCredentials,
    OAuthTokens,
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
    from platform_calendar import _prod_hooks

    _prod_hooks._cli_env_loaded = False
    _prod_hooks._cli_env_cache = {}
    _prod_hooks._cli_default_console = None
    _prod_hooks._cli_env_path = _prod_hooks._default_cli_env_path()
    _init_production_hooks()


# =============================================================================
# Fake Calendar Client
# =============================================================================


__all__ = [
    "CalendarClientProtocol",
    "ConsoleInputHook",
    "ConsoleOutputHook",
    "CurrentTimeHook",
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
    "hooks",
    "reset_hooks",
]
