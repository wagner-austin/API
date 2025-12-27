# Architecture: platform_calendar Library

## Overview

The `platform_calendar` library provides Google Calendar API integration for tracking competition deadlines with automatic reminders. It handles OAuth 2.0 authentication, calendar event CRUD operations, and competition-to-calendar synchronization.

## Dependencies

- `platform-core` - OAuth types and utilities:
  - `oauth_types.py`: `OAuthCredentials`, `OAuthTokens`, `OAuthTokenResponse`, `TokenType` with encode/decode
  - `oauth.py`: PKCE functions (`generate_code_verifier`, `generate_code_challenge`, `generate_state`), `is_token_expired`
  - `oauth_testing.py`: `make_token_response_json`, `make_error_response_json`
  - `json_utils.py`: `require_*` helpers, `load_json_str`, `dump_json_str`
  - `errors.py`: `AppError`, `CalendarErrorCode`

No external HTTP libraries - uses Python stdlib (`urllib.request`, `webbrowser`) for OAuth and API calls.

## Directory Structure

```
libs/platform_calendar/
├── pyproject.toml
├── README.md
├── Makefile
├── .gitignore
├── .env.example
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
│   └── testing.py            # Protocols, hooks, fakes, factories
└── tests/
    ├── __init__.py
    ├── conftest.py           # Autouse hook reset fixture
    ├── test_types.py
    ├── test_client.py
    ├── test_auth.py
    ├── test_competitions.py
    ├── test_config.py
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
    expires_at: int          # Unix timestamp
    token_type: Literal["Bearer"]
```

### Calendar Types

```python
EventStatus = Literal["confirmed", "tentative", "cancelled"]

class EventDateTime(TypedDict):
    dateTime: str      # RFC 3339: "2025-12-26T14:00:00Z"
    timeZone: str      # e.g., "UTC", "America/Los_Angeles"

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
    deadline: str                    # ISO 8601 datetime
    url: str
    project_path: str | None
    calendar_event_id: str | None    # None if not synced
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
    def get_events(self, *, calendar_id: str, time_min: str, time_max: str) -> tuple[CalendarEvent, ...]: ...
    def create_event(self, *, calendar_id: str, summary: str, description: str,
                     start: EventDateTime, end: EventDateTime, reminders: tuple[int, ...],
                     location: str = "", recurrence: tuple[str, ...] = ()) -> CalendarEvent: ...
    def update_event(self, *, calendar_id: str, event_id: str,
                     summary: str | None = None, description: str | None = None,
                     start: EventDateTime | None = None, end: EventDateTime | None = None,
                     reminders: tuple[int, ...] | None = None, location: str | None = None,
                     recurrence: tuple[str, ...] | None = None) -> CalendarEvent: ...
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

### 4. testing.py - Protocols, Hooks, Fakes

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

```python
class HooksContainer:
    http_get: HttpGetHook
    http_post: HttpPostHook
    http_patch: HttpPatchHook
    http_delete: HttpDeleteHook
    load_tokens: LoadTokensHook
    save_tokens: SaveTokensHook
    load_credentials: LoadCredentialsHook
    open_browser: OpenBrowserHook
    console_output: ConsoleOutputHook
    console_input: ConsoleInputHook
    current_time: CurrentTimeHook
    read_file: ReadFileHook
    write_file: WriteFileHook
    file_exists: FileExistsHook

hooks = HooksContainer()

def reset_hooks() -> None:
    """Reset all hooks to production implementations."""
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

## FastAPI Service Integration

### Startup Initialization

```python
# app/dependencies.py
from platform_calendar import (
    CalendarClientProtocol,
    google_calendar_client,
    load_or_authorize,
    OAuthTokens,
)

_calendar_client: CalendarClientProtocol | None = None

def get_calendar_client() -> CalendarClientProtocol:
    """FastAPI dependency for calendar client."""
    if _calendar_client is None:
        raise RuntimeError("Calendar client not initialized")
    return _calendar_client

def init_calendar_client() -> None:
    """Initialize calendar client at startup."""
    global _calendar_client
    tokens = load_or_authorize()
    _calendar_client = google_calendar_client(tokens=tokens)
```

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.dependencies import init_calendar_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_calendar_client()
    yield

app = FastAPI(lifespan=lifespan)
```

### Route Handlers

```python
# app/routes/calendar.py
from fastapi import APIRouter, Depends, HTTPException
from platform_calendar import CalendarClientProtocol, EventDateTime
from platform_core.errors import AppError, CalendarErrorCode
from app.dependencies import get_calendar_client

router = APIRouter(prefix="/calendar", tags=["calendar"])

@router.get("/events")
def list_events(
    calendar_id: str = "primary",
    time_min: str = Query(...),
    time_max: str = Query(...),
    client: CalendarClientProtocol = Depends(get_calendar_client),
):
    try:
        events = client.get_events(
            calendar_id=calendar_id,
            time_min=time_min,
            time_max=time_max,
        )
        return {"events": [dict(e) for e in events]}
    except AppError as e:
        raise HTTPException(status_code=e.http_status, detail=e.message)

@router.post("/events")
def create_event(
    calendar_id: str = "primary",
    summary: str = Body(...),
    description: str = Body(""),
    start: str = Body(...),
    end: str = Body(...),
    location: str = Body(""),
    client: CalendarClientProtocol = Depends(get_calendar_client),
):
    try:
        event = client.create_event(
            calendar_id=calendar_id,
            summary=summary,
            description=description,
            start=EventDateTime(dateTime=start, timeZone="UTC"),
            end=EventDateTime(dateTime=end, timeZone="UTC"),
            reminders=(60,),
            location=location,
        )
        return dict(event)
    except AppError as e:
        raise HTTPException(status_code=e.http_status, detail=e.message)

@router.get("/events/{event_id}")
def get_event(
    event_id: str,
    calendar_id: str = "primary",
    client: CalendarClientProtocol = Depends(get_calendar_client),
):
    try:
        event = client.get_event(calendar_id=calendar_id, event_id=event_id)
        return dict(event)
    except AppError as e:
        if e.code == CalendarErrorCode.EVENT_NOT_FOUND:
            raise HTTPException(status_code=404, detail="Event not found")
        raise HTTPException(status_code=e.http_status, detail=e.message)
```

### Testing FastAPI Routes

```python
# tests/test_routes_calendar.py
import pytest
from fastapi.testclient import TestClient
from platform_calendar import FakeCalendarClient, EventDateTime

from app.main import app
from app import dependencies

@pytest.fixture
def fake_client():
    client = FakeCalendarClient()
    client.add_calendar(calendar_id="primary", summary="Test")
    return client

@pytest.fixture
def test_client(fake_client):
    # Override dependency
    def override_get_calendar_client():
        return fake_client
    app.dependency_overrides[dependencies.get_calendar_client] = override_get_calendar_client
    yield TestClient(app)
    app.dependency_overrides.clear()

def test_create_event(test_client, fake_client):
    response = test_client.post("/calendar/events", json={
        "summary": "Test Event",
        "start": "2025-12-26T14:00:00Z",
        "end": "2025-12-26T15:00:00Z",
    })
    assert response.status_code == 200
    assert response.json()["summary"] == "Test Event"
    assert len(fake_client.get_created_events()) == 1

def test_get_event_not_found(test_client):
    response = test_client.get("/calendar/events/nonexistent")
    assert response.status_code == 404
```

## Environment Configuration

### Development (File-based)

```
~/.google/calendar_credentials.json  - OAuth client ID/secret
~/.google/calendar_tokens.json       - Access/refresh tokens
~/.competitions/tracked.json         - Tracked competitions
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
```

Environment variables take precedence over files. Partial configuration raises `AppError`.

## Public API (__init__.py)

```python
# Auth functions
from platform_calendar.auth import (
    authorize, build_auth_url, exchange_code_for_tokens,
    get_valid_tokens, is_token_expired, load_or_authorize,
    refresh_access_token,
)

# Client factory
from platform_calendar.client import google_calendar_client

# Competition functions
from platform_calendar.competitions import (
    add_competition, get_competition, load_competitions,
    make_competition, remove_competition, save_competitions,
    sync_all_competitions, sync_competition, update_competition,
)

# Types
from platform_calendar.types import (
    CalendarEvent, CalendarListItem, CompetitionSource,
    DEFAULT_REMINDERS, EventDateTime, EventReminders, EventStatus,
    OAuthCredentials, OAuthTokens, ReminderOverride, TrackedCompetition,
)

# Testing utilities
from platform_calendar.testing import (
    CalendarClientProtocol, FakeCalendarClient, hooks,
    make_fake_calendar, make_fake_event, make_fake_http_delete,
    make_fake_http_get, make_fake_http_patch, make_fake_http_post,
    make_fake_tokens, make_raising_http_delete, make_raising_http_get,
    make_raising_http_patch, make_raising_http_post, reset_hooks,
)
```

## Test Coverage

- 100% statement and branch coverage required
- Tests for each encode/decode pair with round-trip validation
- Tests for OAuth flow with fake HTTP responses
- Tests for calendar client with fake HTTP hooks
- Tests for FakeCalendarClient protocol compliance
- Tests for competition sync with fake client
- Tests for environment variable loading
- Tests for production HTTP hooks with local server
- Tests for guard.py using runpy
