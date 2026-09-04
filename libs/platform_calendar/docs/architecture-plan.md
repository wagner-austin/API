# Architecture: platform_calendar Library

## Overview

The `platform_calendar` library provides Google Calendar API integration for tracking competition deadlines with automatic reminders. It handles OAuth 2.0 authentication, calendar event CRUD operations, competition-to-calendar synchronization, and a CLI for managing events across multiple accounts.

## Dependencies

- `platform-core` - OAuth types and utilities:
  - `oauth_types.py`: `OAuthCredentials`, `OAuthTokens`, `OAuthTokenResponse`, `TokenType` with encode/decode
  - `oauth.py`: PKCE functions (`generate_code_verifier`, `generate_code_challenge`, `generate_state`), `is_token_expired`
  - `oauth_testing.py`: `make_token_response_json`, `make_error_response_json`
  - `json_utils.py`: `require_*` helpers, `load_json_str`, `dump_json_str`
  - `errors.py`: `AppError`, `CalendarErrorCode`
- `rich` - Terminal formatting for CLI

No external HTTP libraries - uses Python stdlib (`urllib.request`, `webbrowser`) for OAuth and API calls.

## Directory Structure

```
libs/platform_calendar/
├── pyproject.toml
├── README.md
├── Makefile
├── .gitignore
├── .env.example
├── .env                      # Local config (gitignored)
├── docs/
│   └── architecture-plan.md
├── scripts/
│   ├── __init__.py
│   └── guard.py
├── src/platform_calendar/
│   ├── __init__.py           # Public exports
│   ├── py.typed              # PEP 561 marker
│   ├── types.py              # TypedDicts + encode/decode functions
│   ├── config.py             # API base URL constant
│   ├── auth.py               # OAuth flow + token management
│   ├── client.py             # Google Calendar API wrapper
│   ├── competitions.py       # Competition → Event mapping
│   ├── cli.py                # Command-line interface
│   └── testing.py            # Protocols, hooks, fakes, factories
└── tests/
    ├── __init__.py
    ├── conftest.py           # Autouse hook reset fixture
    ├── test_types.py
    ├── test_client.py
    ├── test_auth.py
    ├── test_competitions.py
    ├── test_config.py
    ├── test_cli.py           # CLI tests
    ├── test_fake_client.py
    ├── test_hooks.py
    ├── test_prod_hooks.py
    └── test_guard_entrypoint.py
```

## Google Calendar API

The library wraps the Google Calendar REST API v3:

```
Base URL: https://www.googleapis.com/calendar/v3

Endpoints:
  GET  /users/me/calendarList              - List calendars
  GET  /calendars/{id}/events              - List events (with time range)
  GET  /calendars/{id}/events/{eventId}    - Get single event
  POST /calendars/{id}/events              - Create event
  PATCH /calendars/{id}/events/{eventId}   - Update event (partial)
  DELETE /calendars/{id}/events/{eventId}  - Delete event
```

## Core Types (types.py)

### OAuth Types (re-exported from platform_core)

OAuth types are defined in `platform_core.oauth_types` and re-exported here for convenience:

```python
# Re-exported from platform_core.oauth_types
from platform_core.oauth_types import OAuthCredentials as OAuthCredentials
from platform_core.oauth_types import OAuthTokens as OAuthTokens
from platform_core.oauth_types import OAuthTokenResponse as OAuthTokenResponse
from platform_core.oauth_types import TokenType as TokenType
from platform_core.oauth_types import decode_oauth_credentials as decode_oauth_credentials
from platform_core.oauth_types import decode_oauth_tokens as decode_oauth_tokens
from platform_core.oauth_types import encode_oauth_credentials as encode_oauth_credentials
from platform_core.oauth_types import encode_oauth_tokens as encode_oauth_tokens


class OAuthCredentials(TypedDict):
    client_id: str
    client_secret: str
    redirect_uri: str


class OAuthTokens(TypedDict):
    access_token: str
    refresh_token: str
    expires_at: int  # Unix timestamp
    token_type: Literal["Bearer"]
```

### Calendar Types

```python
EventStatus = Literal["confirmed", "tentative", "cancelled"]


class EventDateTime(TypedDict):
    dateTime: str  # RFC 3339: "2025-12-26T14:00:00Z"
    timeZone: str  # e.g., "UTC", "America/Los_Angeles"


class ReminderOverride(TypedDict):
    method: Literal["email", "popup"]
    minutes: int


class EventReminders(TypedDict):
    useDefault: bool
    overrides: tuple[ReminderOverride, ...]


class CalendarEvent(TypedDict):
    id: str
    summary: str
    description: str
    start: EventDateTime
    end: EventDateTime
    status: EventStatus
    reminders: EventReminders
    location: str
    recurrence: tuple[str, ...]  # RRULE strings


class CalendarListItem(TypedDict):
    id: str
    summary: str
    description: str
    primary: bool
    timeZone: str
```

### Competition Types

```python
CompetitionSource = Literal["kaggle", "devpost", "manual"]
DEFAULT_REMINDERS: tuple[int, int] = (1440, 60)  # 1 day + 1 hour


class TrackedCompetition(TypedDict):
    id: str
    source: CompetitionSource
    name: str
    deadline: str  # ISO 8601 datetime
    url: str
    project_path: str | None
    calendar_event_id: str | None  # None if not synced
    reminders: tuple[int, ...]
```

## Key Modules

### 1. auth.py - OAuth 2.0 Flow

PKCE-based OAuth for desktop applications. Uses centralized PKCE utilities from `platform_core.oauth`:

```python
# PKCE functions imported from platform_core
from platform_core.oauth import (
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
)
from platform_core.oauth import is_token_expired as _core_is_token_expired


def build_auth_url(credentials: OAuthCredentials, code_verifier: str) -> str:
    """Build Google OAuth authorization URL with PKCE."""


def exchange_code_for_tokens(
    credentials: OAuthCredentials,
    code: str,
    code_verifier: str,
) -> OAuthTokens:
    """Exchange authorization code for tokens."""


def refresh_access_token(
    credentials: OAuthCredentials,
    refresh_token: str,
) -> OAuthTokens:
    """Refresh expired access token."""


def authorize(credentials: OAuthCredentials) -> OAuthTokens:
    """Run full OAuth flow - opens browser, waits for code."""


def load_or_authorize() -> OAuthTokens:
    """Load cached tokens or run auth flow if needed."""


def get_valid_tokens(credentials: OAuthCredentials, tokens: OAuthTokens) -> OAuthTokens:
    """Get tokens, refreshing if expired."""


def is_token_expired(tokens: OAuthTokens) -> bool:
    """Check if tokens need refresh."""
```

### 2. client.py - Calendar API Wrapper

```python
class _GoogleCalendarClient(CalendarClientProtocol):
    def __init__(self, *, access_token: str) -> None: ...

    # Internal HTTP methods
    def _api_get(self, endpoint: str) -> JSONObject: ...
    def _api_post(self, endpoint: str, body: JSONObject) -> JSONObject: ...
    def _api_patch(self, endpoint: str, body: JSONObject) -> JSONObject: ...
    def _normalize_event_response(self, data: JSONObject) -> None: ...
    def _handle_error(self, status_code: int, response_body: str, context: str) -> None: ...

    # Protocol methods
    def list_calendars(self) -> tuple[CalendarListItem, ...]: ...
    def get_event(self, *, calendar_id: str, event_id: str) -> CalendarEvent: ...
    def get_events(
        self, *, calendar_id: str, time_min: str, time_max: str
    ) -> tuple[CalendarEvent, ...]: ...
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
    ) -> CalendarEvent: ...
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
    ) -> CalendarEvent: ...
    def delete_event(self, *, calendar_id: str, event_id: str) -> None: ...


def google_calendar_client(*, tokens: OAuthTokens) -> CalendarClientProtocol:
    """Factory function to create calendar client."""
```

### 3. competitions.py - Competition Tracking

```python
def make_competition(...) -> TrackedCompetition:
    """Create a TrackedCompetition with defaults."""

def sync_competition(
    client: CalendarClientProtocol,
    *,
    competition: TrackedCompetition,
    calendar_id: str = "primary",
) -> TrackedCompetition:
    """Create calendar event for competition, return updated with event_id."""

def sync_all_competitions(
    client: CalendarClientProtocol,
    *,
    competitions: tuple[TrackedCompetition, ...],
    calendar_id: str = "primary",
) -> tuple[TrackedCompetition, ...]:
    """Sync all unsynced competitions."""

def load_competitions(path: Path | None = None) -> tuple[TrackedCompetition, ...]:
    """Load from ~/.competitions/tracked.json."""

def save_competitions(competitions: tuple[TrackedCompetition, ...], path: Path | None = None) -> None:
    """Save to ~/.competitions/tracked.json."""

def add_competition(competitions: tuple[TrackedCompetition, ...], competition: TrackedCompetition) -> tuple[TrackedCompetition, ...]: ...
def remove_competition(competitions: tuple[TrackedCompetition, ...], competition_id: str) -> tuple[TrackedCompetition, ...]: ...
def get_competition(competitions: tuple[TrackedCompetition, ...], competition_id: str) -> TrackedCompetition | None: ...
def update_competition(competitions: tuple[TrackedCompetition, ...], competition_id: str, **updates) -> tuple[TrackedCompetition, ...]: ...
```

### 4. cli.py - Command-Line Interface

The CLI provides multi-account calendar management with Rich formatting.

#### TypedDict Argument Structures

```python
class ListArgs(TypedDict):
    """Arguments for list command."""

    date: str


class CreateArgs(TypedDict):
    """Arguments for create command."""

    title: str
    time: str
    date: str
    duration: int
    location: str
    account: str


class DeleteArgs(TypedDict):
    """Arguments for delete command."""

    date: str
```

#### Token Refresh Types

```python
class TokenRefreshResponse(TypedDict):
    """Response from Google OAuth token refresh endpoint."""

    access_token: str
    expires_in: int
    token_type: str


def require_str(data: JSONObject, key: str) -> str:
    """Require a string value from a JSON object."""


def require_int(data: JSONObject, key: str) -> int:
    """Require an int value from a JSON object."""


def decode_token_refresh_response(data: JSONObject) -> TokenRefreshResponse:
    """Decode token refresh response from JSON."""
```

#### Account Configuration

```python
class Account:
    """Account configuration."""

    name: str  # Display name (e.g., "Personal", "Interns")
    email: str  # Email address
    token_env: str  # Environment variable name for access token
    refresh_token_env: str  # Environment variable name for refresh token
    expires_at_env: str  # Environment variable name for token expiration
    default_calendar: str  # Default calendar ID (usually "primary")


ACCOUNTS = [
    Account(
        name="Personal",
        email="austin.o.wagner@gmail.com",
        token_env="GOOGLE_CALENDAR_ACCESS_TOKEN",
        refresh_token_env="GOOGLE_CALENDAR_REFRESH_TOKEN",
        expires_at_env="GOOGLE_CALENDAR_TOKEN_EXPIRES_AT",
    ),
    Account(
        name="Interns",
        email="interns@liuforirvine.com",
        token_env="GOOGLE_CALENDAR_INTERNS_ACCESS_TOKEN",
        refresh_token_env="GOOGLE_CALENDAR_INTERNS_REFRESH_TOKEN",
        expires_at_env="GOOGLE_CALENDAR_INTERNS_EXPIRES_AT",
    ),
]
```

#### Token Refresh Functions

```python
def _is_token_expired(expires_at_str: str) -> bool:
    """Check if token is expired or will expire within 60 seconds."""


def _refresh_token(client_id: str, client_secret: str, refresh_token: str) -> TokenRefreshResponse:
    """Refresh an access token using the refresh token."""


def _get_valid_token_for_account(account: Account) -> str | None:
    """Get a valid access token for an account, refreshing if expired."""
```

#### Commands

```python
def cmd_list(cmd_args: ListArgs) -> None:
    """List events for a date across all accounts."""


def cmd_calendars() -> None:
    """List all calendars for all accounts."""


def cmd_create(cmd_args: CreateArgs) -> None:
    """Create a new event."""


def cmd_delete(cmd_args: DeleteArgs) -> None:
    """Delete an event (interactive selection)."""


def cmd_tomorrow() -> None:
    """Show tomorrow's events."""


def cmd_week() -> None:
    """Show this week's events."""


def main() -> None:
    """Main entry point - parses args and dispatches to commands."""
```

#### CLI Argument Parsing

```python
def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser with subcommands."""


def decode_list_args(args: argparse.Namespace) -> ListArgs:
    """Decode argparse.Namespace to typed ListArgs."""


def decode_create_args(args: argparse.Namespace) -> CreateArgs:
    """Decode argparse.Namespace to typed CreateArgs."""


def decode_delete_args(args: argparse.Namespace) -> DeleteArgs:
    """Decode argparse.Namespace to typed DeleteArgs."""
```

### 5. testing.py - Protocols, Hooks, Fakes

#### Protocol

```python
@runtime_checkable
class CalendarClientProtocol(Protocol):
    def list_calendars(self) -> tuple[CalendarListItem, ...]: ...
    def get_event(self, *, calendar_id: str, event_id: str) -> CalendarEvent: ...
    def get_events(self, *, calendar_id: str, time_min: str, time_max: str) -> tuple[CalendarEvent, ...]: ...
    def create_event(self, *, calendar_id: str, summary: str, description: str,
                     start: EventDateTime, end: EventDateTime, reminders: tuple[int, ...],
                     location: str = "", recurrence: tuple[str, ...] = ()) -> CalendarEvent: ...
    def update_event(self, *, calendar_id: str, event_id: str, ...) -> CalendarEvent: ...
    def delete_event(self, *, calendar_id: str, event_id: str) -> None: ...
```

#### Hooks Container

All external dependencies are accessed through hooks for testability:

```python
class HooksContainer:
    # HTTP hooks (for auth and client modules)
    http_get: HttpGetHook
    http_post: HttpPostHook
    http_patch: HttpPatchHook
    http_delete: HttpDeleteHook

    # Token/credential hooks (for auth module)
    load_tokens: LoadTokensHook
    save_tokens: SaveTokensHook
    load_credentials: LoadCredentialsHook

    # System hooks
    open_browser: OpenBrowserHook
    console_output: ConsoleOutputHook
    console_input: ConsoleInputHook
    current_time: CurrentTimeHook
    read_file: ReadFileHook
    write_file: WriteFileHook
    file_exists: FileExistsHook

    # CLI-specific hooks
    cli_api_get: CliApiGetHook  # GET requests with access token
    cli_api_post: CliApiPostHook  # POST requests with access token
    cli_api_delete: CliApiDeleteHook  # DELETE requests with access token
    cli_get_env: CliGetEnvHook  # Environment variable access
    cli_get_now: CliGetNowHook  # Current datetime
    cli_prompt_ask: CliPromptAskHook  # Rich Prompt.ask wrapper
    cli_confirm_ask: CliConfirmAskHook  # Rich Confirm.ask wrapper
    cli_get_console: CliGetConsoleHook  # Rich Console instance


hooks = HooksContainer()


def reset_hooks() -> None:
    """Reset all hooks to production implementations."""
```

#### Hook Type Definitions

```python
from datetime import datetime
from platform_core.json_utils import JSONObject
from rich.console import Console

# HTTP hooks
HttpGetHook = Callable[[str, dict[str, str]], str]
HttpPostHook = Callable[[str, dict[str, str], str], str]
HttpPatchHook = Callable[[str, dict[str, str], str], str]
HttpDeleteHook = Callable[[str, dict[str, str]], None]

# Token hooks
LoadTokensHook = Callable[[], OAuthTokens | None]
SaveTokensHook = Callable[[OAuthTokens], None]
LoadCredentialsHook = Callable[[], OAuthCredentials]

# System hooks
OpenBrowserHook = Callable[[str], None]
CurrentTimeHook = Callable[[], int]
ReadFileHook = Callable[[str], str]
WriteFileHook = Callable[[str, str], None]
FileExistsHook = Callable[[str], bool]
ConsoleOutputHook = Callable[[str], None]
ConsoleInputHook = Callable[[str], str]

# CLI-specific hooks
CliApiGetHook = Callable[[str, str], JSONObject]  # (access_token, url) -> response
CliApiPostHook = Callable[
    [str, str, JSONObject], JSONObject
]  # (access_token, url, body) -> response
CliApiDeleteHook = Callable[[str, str], None]  # (access_token, url) -> None
CliGetEnvHook = Callable[[str], str | None]  # (key) -> value or None
CliGetNowHook = Callable[[], datetime]  # () -> current datetime
CliPromptAskHook = Callable[[str], str]  # (message) -> user input
CliConfirmAskHook = Callable[[str], bool]  # (message) -> True/False
CliGetConsoleHook = Callable[[], Console]  # () -> Console instance
```

#### Fake Client

```python
class FakeCalendarClient(CalendarClientProtocol):
    def __init__(self) -> None:
        self._calendars: list[CalendarListItem] = []
        self._events: dict[str, list[CalendarEvent]] = {}
        self._created_events: list[CalendarEvent] = []
        self._updated_events: list[CalendarEvent] = []
        self._deleted_events: list[tuple[str, str]] = []

    # Test helpers
    def add_calendar(self, *, calendar_id: str, summary: str, ...) -> None: ...
    def add_event(self, *, calendar_id: str, event: CalendarEvent) -> None: ...
    def get_created_events(self) -> list[CalendarEvent]: ...
    def get_updated_events(self) -> list[CalendarEvent]: ...
    def get_deleted_events(self) -> list[tuple[str, str]]: ...
```

#### Factory Functions

```python
# Calendar-specific factories
def make_fake_event(...) -> CalendarEvent
def make_fake_calendar(...) -> CalendarListItem
def make_fake_tokens(tokens: OAuthTokens) -> LoadTokensHook
def make_fake_credentials(creds: OAuthCredentials) -> LoadCredentialsHook
def make_fake_http_get(response: str) -> HttpGetHook
def make_fake_http_post(response: str) -> HttpPostHook
def make_fake_http_patch(response: str) -> HttpPatchHook
def make_fake_http_delete() -> HttpDeleteHook
def make_raising_http_get(exc: BaseException) -> HttpGetHook
def make_raising_http_post(exc: BaseException) -> HttpPostHook
def make_raising_http_patch(exc: BaseException) -> HttpPatchHook
def make_raising_http_delete(exc: BaseException) -> HttpDeleteHook

# Re-exported from platform_core.oauth_testing
from platform_core.oauth_testing import make_token_response_json
from platform_core.oauth_testing import make_error_response_json
```

#### Production Hook Implementations

Production hooks are set at module load and restored by `reset_hooks()`:

```python
def _prod_cli_api_get(access_token: str, url: str) -> JSONObject:
    """Make authenticated GET request using urllib."""


def _prod_cli_api_post(access_token: str, url: str, request_body: JSONObject) -> JSONObject:
    """Make authenticated POST request using urllib."""


def _prod_cli_api_delete(access_token: str, url: str) -> None:
    """Make authenticated DELETE request using urllib."""


def _prod_cli_get_env(key: str) -> str | None:
    """Get environment variable from .env file cache."""


def _prod_cli_get_now() -> datetime:
    """Get current datetime."""


def _prod_cli_prompt_ask(message: str, _prompt_func: Callable[[str], str] | None = None) -> str:
    """Prompt for user input using Rich Prompt.ask."""


def _prod_cli_confirm_ask(message: str, _confirm_func: Callable[[str], bool] | None = None) -> bool:
    """Prompt for confirmation using Rich Confirm.ask."""


def _prod_cli_get_console() -> Console:
    """Get cached Rich Console instance."""
```

## CLI Testing Patterns

### Setting Up CLI Hooks for Tests

```python
import io
from datetime import datetime
from platform_core.json_utils import JSONObject
from rich.console import Console
from platform_calendar.testing import hooks, reset_hooks
from platform_calendar.cli import cmd_list, ListArgs


def test_cmd_list() -> None:
    """Test listing events."""
    # Capture output
    output = io.StringIO()
    console = Console(file=output, force_terminal=True)
    hooks.cli_get_console = lambda: console

    # Set up fake environment
    hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
    hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

    # Set up fake API responses
    def fake_api_get(token: str, url: str) -> JSONObject:
        if "calendarList" in url:
            return {"items": [{"id": "primary", "summary": "Main"}]}
        return {
            "items": [
                {
                    "id": "evt1",
                    "summary": "Meeting",
                    "start": {"dateTime": "2026-02-20T14:00:00-08:00"},
                }
            ]
        }

    hooks.cli_api_get = fake_api_get

    # Execute command
    cmd_list(ListArgs(date="2026-02-20"))

    # Verify output
    result = output.getvalue()
    assert "Meeting" in result
```

### Testing Event Creation

```python
def test_cmd_create() -> None:
    """Test creating an event."""
    output = io.StringIO()
    console = Console(file=output, force_terminal=True)
    hooks.cli_get_console = lambda: console

    hooks.cli_get_env = lambda key: "token" if "ACCESS_TOKEN" in key else None
    hooks.cli_get_now = lambda: datetime(2026, 2, 20, 10, 0, 0)

    created_events: list[JSONObject] = []

    def fake_api_post(token: str, url: str, body: JSONObject) -> JSONObject:
        created_events.append(body)
        return {"id": "new_event", "summary": body.get("summary", "")}

    hooks.cli_api_post = fake_api_post

    from platform_calendar.cli import cmd_create, CreateArgs

    cmd_create(
        CreateArgs(
            title="Team Meeting",
            time="14:00",
            date="2026-02-20",
            duration=60,
            location="Room A",
            account="Personal",
        )
    )

    result = output.getvalue()
    assert "Created:" in result
    assert len(created_events) == 1
    assert created_events[0]["summary"] == "Team Meeting"
```

## Environment Configuration

### Development (File-based)

```
~/.google/calendar_credentials.json  - OAuth client ID/secret
~/.google/calendar_tokens.json       - Access/refresh tokens
~/.competitions/tracked.json         - Tracked competitions
libs/platform_calendar/.env          - CLI environment variables
```

### Production (Environment Variables)

```bash
# Credentials (from Google Cloud Console)
GOOGLE_CALENDAR_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CALENDAR_CLIENT_SECRET=GOCSPX-your_secret
GOOGLE_CALENDAR_REDIRECT_URI=http://localhost

# Tokens (after OAuth authorization)
GOOGLE_CALENDAR_ACCESS_TOKEN=ya29.your_access_token
GOOGLE_CALENDAR_REFRESH_TOKEN=1//your_refresh_token
GOOGLE_CALENDAR_TOKEN_EXPIRES_AT=1735200000

# Additional accounts
GOOGLE_CALENDAR_INTERNS_ACCESS_TOKEN=ya29.work_access_token
GOOGLE_CALENDAR_INTERNS_REFRESH_TOKEN=1//work_refresh_token
GOOGLE_CALENDAR_INTERNS_EXPIRES_AT=1735200000
```

Environment variables take precedence over files. Partial configuration raises `AppError`.

## Test Coverage

- 100% statement and branch coverage required
- Tests for each encode/decode pair with round-trip validation
- Tests for OAuth flow with fake HTTP responses
- Tests for calendar client with fake HTTP hooks
- Tests for FakeCalendarClient protocol compliance
- Tests for competition sync with fake client
- Tests for environment variable loading
- Tests for production HTTP hooks with local server
- Tests for CLI commands with fake hooks
- Tests for CLI argument parsing and TypedDict decoding
- Tests for guard.py using runpy
