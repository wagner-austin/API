# platform_calendar Implementation Plan

## Overview

Google Calendar API integration for tracking competition deadlines (Kaggle, Devpost, etc.) with automatic reminders.

## Design Decisions

- **Standalone library**: No integration with platform_kaggle - competitions added manually via JSON
- **Default reminders**: 1440 minutes (1 day) + 60 minutes (1 hour) before deadline
- **No external deps**: OAuth + HTTP via stdlib (urllib.request, webbrowser)
- **Strict typing**: mypy strict mode, no Any, no cast, 100% coverage

---

## Google Calendar API Setup

### Step 1: Create Google Cloud Project
1. Go to https://console.cloud.google.com/
2. Create new project: "CompetitionTracker"
3. Enable "Google Calendar API" in APIs & Services

### Step 2: Create OAuth 2.0 Credentials
1. APIs & Services → Credentials → Create Credentials → OAuth client ID
2. Application type: "Desktop app"
3. Download JSON → save as `~/.google/calendar_credentials.json`

### Step 3: Required Scopes
```
https://www.googleapis.com/auth/calendar.events
https://www.googleapis.com/auth/calendar.readonly
```

---

## Directory Structure

```
libs/platform_calendar/
├── pyproject.toml
├── Makefile
├── README.md
├── docs/
│   └── plan.md                   # This file
├── src/platform_calendar/
│   ├── __init__.py               # Public exports
│   ├── py.typed                  # PEP 561 marker
│   ├── types.py                  # TypedDicts for events, calendars, competitions
│   ├── protocols.py              # CalendarClientProtocol
│   ├── errors.py                 # CalendarAPIError, AuthenticationError
│   ├── config.py                 # Credentials loading from ~/.google/
│   ├── auth.py                   # OAuth flow + token management
│   ├── client.py                 # Google Calendar API wrapper
│   ├── competitions.py           # Competition → Event mapping
│   └── testing.py                # Fakes, hooks, factories
├── scripts/
│   ├── __init__.py
│   └── guard.py                  # Monorepo guard runner
└── tests/
    ├── __init__.py
    ├── conftest.py               # Autouse hook reset fixture
    ├── test_types.py
    ├── test_client.py
    ├── test_auth.py
    ├── test_competitions.py
    └── test_guard_entrypoint.py
```

---

## Core Types

### Calendar Event (from Google API)
```python
EventStatus = Literal["confirmed", "tentative", "cancelled"]

class EventDateTime(TypedDict):
    dateTime: str      # RFC 3339: "2025-12-26T14:00:00-08:00"
    timeZone: str      # e.g., "America/Los_Angeles"

class CalendarEvent(TypedDict):
    id: str
    summary: str       # Title
    description: str
    start: EventDateTime
    end: EventDateTime
    status: EventStatus
    reminders: EventReminders

class EventReminders(TypedDict):
    useDefault: bool
    overrides: tuple[ReminderOverride, ...]

class ReminderOverride(TypedDict):
    method: Literal["email", "popup"]
    minutes: int
```

### Competition Tracking
```python
CompetitionSource = Literal["kaggle", "devpost", "manual"]

DEFAULT_REMINDERS: tuple[int, int] = (1440, 60)  # 1 day + 1 hour

class TrackedCompetition(TypedDict):
    id: str                          # Unique ID
    source: CompetitionSource
    name: str                        # Competition name
    deadline: str                    # ISO 8601 datetime
    url: str                         # Link to competition
    project_path: str | None         # e.g., "libs/cleargbm"
    calendar_event_id: str | None    # Created event ID (None if not synced)
    reminders: tuple[int, ...]       # Minutes before deadline
```

### OAuth Tokens
```python
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

---

## Protocol

```python
@runtime_checkable
class CalendarClientProtocol(Protocol):
    def list_calendars(self) -> tuple[CalendarListItem, ...]: ...

    def get_events(
        self,
        *,
        calendar_id: str,
        time_min: str,
        time_max: str,
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
    ) -> CalendarEvent: ...

    def update_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
        summary: str | None = None,
        description: str | None = None,
    ) -> CalendarEvent: ...

    def delete_event(
        self,
        *,
        calendar_id: str,
        event_id: str,
    ) -> None: ...
```

---

## Auth Flow

### Token Management
- Credentials: `~/.google/calendar_credentials.json` (OAuth client ID/secret)
- Tokens: `~/.google/calendar_tokens.json` (access/refresh tokens)
- Auto-refresh when `expires_at < current_time`

### Functions
```python
def load_credentials(path: Path | None = None) -> OAuthCredentials:
    """Load OAuth client credentials from JSON file."""

def authorize(credentials: OAuthCredentials) -> OAuthTokens:
    """Run OAuth flow - opens browser for user consent."""

def refresh_access_token(credentials: OAuthCredentials, refresh_token: str) -> OAuthTokens:
    """Refresh expired access token."""

def load_or_authorize(credentials_path: Path | None = None) -> OAuthTokens:
    """Load cached tokens or run auth flow if needed."""
```

---

## Competition Storage

File: `~/.competitions/tracked.json`
```json
{
  "competitions": [
    {
      "id": "devpost-build-from-scratch-2025",
      "source": "devpost",
      "name": "Build From Scratch - Season I",
      "deadline": "2025-12-25T22:00:00Z",
      "url": "https://devpost.com/...",
      "project_path": "libs/cleargbm",
      "calendar_event_id": "abc123",
      "reminders": [1440, 60]
    },
    {
      "id": "devpost-visaverse-2025",
      "source": "devpost",
      "name": "VisaVerse AI Hackathon",
      "deadline": "2025-12-26T22:00:00Z",
      "url": "https://devpost.com/...",
      "project_path": "services/grandma-api",
      "calendar_event_id": null,
      "reminders": [1440, 60]
    }
  ]
}
```

---

## Testing Pattern

### Hooks Container
```python
class HooksContainer:
    calendar_client: Callable[[], CalendarClientProtocol]
    http_get: Callable[[str, dict[str, str]], str]
    http_post: Callable[[str, dict[str, str], str], str]
    load_tokens: Callable[[], OAuthTokens | None]
    save_tokens: Callable[[OAuthTokens], None]
    open_browser: Callable[[str], None]
    current_time: Callable[[], int]

hooks = HooksContainer()
```

### Fake Client
```python
class FakeCalendarClient(CalendarClientProtocol):
    def __init__(self) -> None:
        self._events: dict[str, list[CalendarEvent]] = {}
        self._calendars: list[CalendarListItem] = []
        self._next_event_id: int = 1

    # Test helpers
    def add_calendar(self, *, id: str, summary: str) -> None: ...
    def add_event(self, *, calendar_id: str, event: CalendarEvent) -> None: ...
    def get_created_events(self) -> list[CalendarEvent]: ...
```

---

## Implementation Order

1. **Infrastructure**: pyproject.toml, Makefile, scripts/guard.py, py.typed
2. **types.py**: All TypedDicts with encode/decode functions
3. **errors.py**: CalendarAPIError, AuthenticationError, TokenExpiredError
4. **testing.py**: HooksContainer, FakeCalendarClient, factory helpers
5. **config.py**: load_credentials(), paths
6. **auth.py**: OAuth flow with browser, token refresh
7. **client.py**: GoogleCalendarClient implementing protocol
8. **competitions.py**: load/save/sync competitions
9. **__init__.py**: Public exports
10. **Tests**: 100% statement + branch coverage

---

## Example Usage

```python
from platform_calendar import (
    load_or_authorize,
    google_calendar_client,
    TrackedCompetition,
    create_competition_event,
    DEFAULT_REMINDERS,
)

# One-time auth (opens browser first time)
tokens = load_or_authorize()

# Create client
client = google_calendar_client(tokens=tokens)

# Track a competition
comp: TrackedCompetition = {
    "id": "devpost-visaverse-2025",
    "source": "devpost",
    "name": "VisaVerse AI Hackathon",
    "deadline": "2025-12-26T22:00:00Z",
    "url": "https://devpost.com/...",
    "project_path": "services/grandma-api",
    "calendar_event_id": None,
    "reminders": DEFAULT_REMINDERS,
}

# Create calendar event
event = create_competition_event(client, competition=comp)
print(f"Created event: {event['id']}")
```

---

## Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.11"
platform-core = { path = "../platform_core", develop = true }
# No external HTTP lib - use urllib.request
# No external OAuth lib - implement with stdlib
```
