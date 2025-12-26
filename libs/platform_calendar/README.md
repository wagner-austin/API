# platform_calendar

Google Calendar API integration for tracking competition deadlines with automatic reminders.

## Installation

```bash
poetry add platform-calendar
```

## Features

- **OAuth 2.0 Authentication**: Secure PKCE-based authentication with Google Calendar API
- **Competition Tracking**: Track Kaggle, Devpost, and other competition deadlines
- **Automatic Reminders**: Default 1 day + 1 hour before deadline (configurable)
- **Strict Typing**: mypy strict mode, no Any, 100% test coverage
- **Project Mapping**: Link competitions to your codebase projects
- **Testable Design**: Hooks-based dependency injection for full testability

## Setup

### 1. Create Google Cloud Project

1. Go to https://console.cloud.google.com/
2. Create new project: "CompetitionTracker"
3. Enable "Google Calendar API" in APIs & Services

### 2. Create OAuth 2.0 Credentials

1. Go to APIs & Services → Credentials
2. Create Credentials → OAuth client ID
3. Application type: "Desktop app"
4. Download JSON → save as `~/.google/calendar_credentials.json`

## Quick Start

```python
from platform_calendar import (
    load_or_authorize,
    google_calendar_client,
    make_competition,
    sync_competition,
    add_competition,
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
print(f"Created event: {synced['calendar_event_id']}")

# Save for later
comps = load_competitions()
comps = add_competition(comps, synced)
save_competitions(comps)
```

## Calendar Client

Direct calendar operations without competition tracking:

```python
from platform_calendar import (
    google_calendar_client,
    load_or_authorize,
    CalendarEvent,
    EventDateTime,
)

tokens = load_or_authorize()
client = google_calendar_client(tokens=tokens)

# List calendars
calendars = client.list_calendars()
for cal in calendars:
    print(f"{cal['id']}: {cal['summary']}")

# Get events in time range
events = client.get_events(
    calendar_id="primary",
    time_min="2025-01-01T00:00:00Z",
    time_max="2025-12-31T23:59:59Z",
)

# Create event
event = client.create_event(
    calendar_id="primary",
    summary="Project Deadline",
    description="Submit final code",
    start=EventDateTime(dateTime="2025-12-26T22:00:00Z", timeZone="UTC"),
    end=EventDateTime(dateTime="2025-12-26T23:00:00Z", timeZone="UTC"),
    reminders=(1440, 60),  # 1 day and 1 hour before
)
```

## Competition Storage

Competitions are stored at `~/.competitions/tracked.json`:

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
    }
  ]
}
```

## Testing

The library uses a hooks-based design for full testability without mocking:

```python
from platform_calendar import (
    hooks,
    reset_hooks,
    FakeCalendarClient,
    make_fake_tokens,
    make_fake_http_get,
)

# Reset to production hooks
reset_hooks()

# Use fake client for testing
client = FakeCalendarClient()
client.add_calendar(calendar_id="primary", summary="Test Calendar")

# Use fake HTTP responses
hooks.http_get = make_fake_http_get('{"items": []}')
hooks.load_tokens = make_fake_tokens(tokens)
```

## API Reference

### Auth Functions

| Function | Description |
|----------|-------------|
| `load_or_authorize` | Load cached tokens or run OAuth flow |
| `authorize` | Run OAuth authorization flow |
| `build_auth_url` | Build Google OAuth URL |
| `exchange_code_for_tokens` | Exchange auth code for tokens |
| `refresh_access_token` | Refresh expired access token |
| `get_valid_tokens` | Get tokens, refreshing if expired |
| `is_token_expired` | Check if tokens need refresh |

### Client Functions

| Function | Description |
|----------|-------------|
| `google_calendar_client` | Create calendar client from tokens |

### Competition Functions

| Function | Description |
|----------|-------------|
| `make_competition` | Create TrackedCompetition |
| `add_competition` | Add to competition list |
| `remove_competition` | Remove from competition list |
| `get_competition` | Find by ID |
| `update_competition` | Update competition fields |
| `load_competitions` | Load from JSON file |
| `save_competitions` | Save to JSON file |
| `sync_competition` | Create calendar event |
| `sync_all_competitions` | Sync all unsynced |

### Types

| Type | Description |
|------|-------------|
| `OAuthTokens` | Access/refresh token pair |
| `OAuthCredentials` | OAuth client credentials |
| `CalendarEvent` | Calendar event data |
| `CalendarListItem` | Calendar metadata |
| `TrackedCompetition` | Competition with calendar link |
| `EventDateTime` | Event start/end time |
| `EventReminders` | Reminder configuration |

### Error Handling

```python
from platform_calendar import AppError, CalendarErrorCode

try:
    client.get_events(...)
except AppError as e:
    if e.code == CalendarErrorCode.CALENDAR_NOT_FOUND:
        print("Calendar does not exist")
    elif e.code == CalendarErrorCode.AUTHENTICATION_FAILED:
        print("Need to re-authenticate")
```

| Error Code | Description |
|------------|-------------|
| `AUTHENTICATION_FAILED` | OAuth token invalid/expired |
| `CALENDAR_NOT_FOUND` | Calendar ID not found |
| `EVENT_NOT_FOUND` | Event ID not found |
| `RATE_LIMIT_EXCEEDED` | API quota exceeded |
| `INVALID_CREDENTIALS` | Bad OAuth credentials |
| `NETWORK_ERROR` | HTTP request failed |

## Configuration

Default paths:

| Path | Description |
|------|-------------|
| `~/.google/calendar_credentials.json` | OAuth client credentials |
| `~/.google/calendar_tokens.json` | Access/refresh tokens |
| `~/.competitions/tracked.json` | Tracked competitions |

## Development

```bash
cd libs/platform_calendar
make check  # Run lint + tests
make lint   # Run linting only
make test   # Run tests only
```

## Requirements

- Python 3.11+
- platform-core (for error handling and JSON utilities)
- 100% test coverage enforced
