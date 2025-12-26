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

# Get a single event by ID
event = client.get_event(calendar_id="primary", event_id="abc123")

# Create event with optional location and recurrence
event = client.create_event(
    calendar_id="primary",
    summary="Weekly Team Sync",
    description="Discuss project progress",
    start=EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC"),
    end=EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC"),
    reminders=(1440, 60),  # 1 day and 1 hour before
    location="Conference Room A",  # Optional
    recurrence=("RRULE:FREQ=WEEKLY;COUNT=10",),  # Optional: recurring event
)

# Update event (partial update - only specified fields are changed)
updated = client.update_event(
    calendar_id="primary",
    event_id=event["id"],
    summary="Updated Title",
    description="New description",
    start=EventDateTime(dateTime="2025-12-27T14:00:00Z", timeZone="UTC"),
    end=EventDateTime(dateTime="2025-12-27T15:00:00Z", timeZone="UTC"),
    reminders=(60, 30),  # Update reminders
    location="New Location",
    recurrence=("RRULE:FREQ=DAILY;COUNT=5",),
)

# Delete event
client.delete_event(calendar_id="primary", event_id=event["id"])
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
    make_fake_http_post,
    make_fake_http_patch,
    make_fake_http_delete,
    make_fake_event,
)

# Reset to production hooks
reset_hooks()

# Use fake client for testing
client = FakeCalendarClient()
client.add_calendar(calendar_id="primary", summary="Test Calendar")

# Create events with the fake client
from platform_calendar import EventDateTime
start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

event = client.create_event(
    calendar_id="primary",
    summary="Test Event",
    description="Test",
    start=start,
    end=end,
    reminders=(60,),
    location="Room A",
    recurrence=("RRULE:FREQ=DAILY",),
)

# Get event by ID
fetched = client.get_event(calendar_id="primary", event_id=event["id"])

# Update event
updated = client.update_event(
    calendar_id="primary",
    event_id=event["id"],
    summary="Updated Event",
    reminders=(30, 15),
)

# Use fake HTTP responses for real client testing
hooks.http_get = make_fake_http_get('{"items": []}')
hooks.http_post = make_fake_http_post('{"id": "new123"}')
hooks.http_patch = make_fake_http_patch('{"id": "updated123"}')
hooks.http_delete = make_fake_http_delete()
hooks.load_tokens = make_fake_tokens(tokens)

# Create fake events for testing
fake_event = make_fake_event(
    event_id="test123",
    summary="Fake Event",
    location="Test Location",
    recurrence=("RRULE:FREQ=WEEKLY",),
)
```

### Testing Helpers

| Helper | Description |
|--------|-------------|
| `FakeCalendarClient` | In-memory calendar client implementing full protocol |
| `make_fake_http_get(response)` | Returns fixed response for GET requests |
| `make_fake_http_post(response)` | Returns fixed response for POST requests |
| `make_fake_http_patch(response)` | Returns fixed response for PATCH requests |
| `make_fake_http_delete()` | No-op for DELETE requests |
| `make_raising_http_get(error)` | Raises specified exception on GET |
| `make_raising_http_post(error)` | Raises specified exception on POST |
| `make_raising_http_patch(error)` | Raises specified exception on PATCH |
| `make_raising_http_delete(error)` | Raises specified exception on DELETE |
| `make_fake_event(...)` | Create a CalendarEvent with defaults |
| `make_fake_calendar(...)` | Create a CalendarListItem with defaults |
| `make_fake_tokens(tokens)` | Returns fixed tokens |
| `make_fake_credentials(creds)` | Returns fixed credentials |
| `reset_hooks()` | Reset all hooks to production implementations |

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

### Client Methods

The `CalendarClientProtocol` provides the following methods:

| Method | Description |
|--------|-------------|
| `list_calendars()` | List all calendars for the authenticated user |
| `get_events(calendar_id, time_min, time_max)` | Get events in a time range |
| `get_event(calendar_id, event_id)` | Get a single event by ID |
| `create_event(...)` | Create a new event with optional location/recurrence |
| `update_event(...)` | Partial update (PATCH) - only updates specified fields |
| `delete_event(calendar_id, event_id)` | Delete an event |

### Client Factory

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
| `CalendarEvent` | Calendar event (id, summary, description, start, end, status, reminders, location, recurrence) |
| `CalendarListItem` | Calendar metadata |
| `TrackedCompetition` | Competition with calendar link |
| `EventDateTime` | Event start/end time (dateTime, timeZone) |
| `EventReminders` | Reminder configuration (useDefault, overrides) |
| `ReminderOverride` | Individual reminder (method, minutes) |

### CalendarEvent Fields

```python
class CalendarEvent(TypedDict):
    id: str                      # Event ID
    summary: str                 # Event title
    description: str             # Event description
    start: EventDateTime         # Start time
    end: EventDateTime           # End time
    status: EventStatus          # "confirmed", "tentative", or "cancelled"
    reminders: EventReminders    # Reminder settings
    location: str                # Location string (optional in API, always present in type)
    recurrence: tuple[str, ...]  # RRULE strings for recurring events
```

### Error Handling

```python
from platform_core.errors import AppError, CalendarErrorCode

try:
    client.get_events(...)
except AppError as e:
    if e.code == CalendarErrorCode.CALENDAR_NOT_FOUND:
        print("Calendar does not exist")
    elif e.code == CalendarErrorCode.AUTH_FAILED:
        print("Need to re-authenticate")
```

| Error Code | Description |
|------------|-------------|
| `CREDENTIALS_NOT_FOUND` | OAuth credentials file missing or invalid |
| `TOKEN_EXPIRED` | OAuth token needs refresh |
| `AUTH_FAILED` | OAuth authentication failed |
| `CALENDAR_API_ERROR` | General API error (includes network errors) |
| `CALENDAR_NOT_FOUND` | Calendar ID not found |
| `EVENT_NOT_FOUND` | Event ID not found |

## Configuration

### File-Based (Development)

Default paths:

| Path | Description |
|------|-------------|
| `~/.google/calendar_credentials.json` | OAuth client credentials |
| `~/.google/calendar_tokens.json` | Access/refresh tokens |
| `~/.competitions/tracked.json` | Tracked competitions |

### Environment Variables (Production/Railway)

For deployment, configure via environment variables instead of files:

**Credentials** (from Google Cloud Console):
```bash
GOOGLE_CALENDAR_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CALENDAR_CLIENT_SECRET=GOCSPX-your_secret
GOOGLE_CALENDAR_REDIRECT_URI=http://localhost  # optional, defaults to http://localhost
```

**Tokens** (after OAuth authorization):
```bash
GOOGLE_CALENDAR_ACCESS_TOKEN=ya29.your_access_token
GOOGLE_CALENDAR_REFRESH_TOKEN=1//your_refresh_token
GOOGLE_CALENDAR_TOKEN_EXPIRES_AT=1735200000  # Unix timestamp
```

Environment variables take precedence over files. If any credential env var is set, all required ones must be present (partial config raises an error).

See `.env.example` for a template.

## Try It Out

### Step 1: Install Dependencies

```bash
cd libs/platform_calendar
poetry install
```

### Step 2: Authorize (First Time)

Run the OAuth flow to get tokens:

```bash
poetry run python -c "
from platform_calendar import load_or_authorize
tokens = load_or_authorize()
print('Access Token:', tokens['access_token'][:50] + '...')
print('Refresh Token:', tokens['refresh_token'])
print('Expires At:', tokens['expires_at'])
print()
print('Add these to your .env:')
print(f\"GOOGLE_CALENDAR_ACCESS_TOKEN={tokens['access_token']}\")
print(f\"GOOGLE_CALENDAR_REFRESH_TOKEN={tokens['refresh_token']}\")
print(f\"GOOGLE_CALENDAR_TOKEN_EXPIRES_AT={tokens['expires_at']}\")
"
```

This opens your browser for Google sign-in. After authorizing, tokens are saved to `~/.google/calendar_tokens.json`.

### Step 3: List Your Calendars

```bash
poetry run python -c "
from platform_calendar import load_or_authorize, google_calendar_client

tokens = load_or_authorize()
client = google_calendar_client(tokens=tokens)

print('Your calendars:')
for cal in client.list_calendars():
    primary = ' (primary)' if cal['primary'] else ''
    print(f\"  - {cal['summary']}{primary}: {cal['id']}\")
"
```

### Step 4: Create a Test Event

```bash
poetry run python -c "
from platform_calendar import (
    load_or_authorize,
    google_calendar_client,
    EventDateTime,
)

tokens = load_or_authorize()
client = google_calendar_client(tokens=tokens)

event = client.create_event(
    calendar_id='primary',
    summary='Test Event from platform_calendar',
    description='This is a test event created by the library',
    start=EventDateTime(dateTime='2025-12-27T14:00:00Z', timeZone='UTC'),
    end=EventDateTime(dateTime='2025-12-27T15:00:00Z', timeZone='UTC'),
    reminders=(60,),  # 1 hour before
)
print(f\"Created event: {event['id']}\")
print(f\"Summary: {event['summary']}\")
"
```

### Step 5: Track a Competition

```bash
poetry run python -c "
from platform_calendar import (
    load_or_authorize,
    google_calendar_client,
    make_competition,
    sync_competition,
    add_competition,
    load_competitions,
    save_competitions,
)

tokens = load_or_authorize()
client = google_calendar_client(tokens=tokens)

# Create a competition
comp = make_competition(
    competition_id='devpost-test-2025',
    source='devpost',
    name='Test Competition',
    deadline='2025-12-31T23:59:00Z',
    url='https://devpost.com/example',
    project_path='libs/platform_calendar',
)

# Sync to calendar
synced = sync_competition(client, competition=comp)
print(f\"Created calendar event: {synced['calendar_event_id']}\")

# Save to tracked competitions
comps = load_competitions()
comps = add_competition(comps, synced)
save_competitions(comps)
print(f\"Saved to ~/.competitions/tracked.json\")
"
```

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
