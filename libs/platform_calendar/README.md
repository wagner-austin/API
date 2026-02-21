# platform_calendar

Google Calendar API integration for tracking competition deadlines with automatic reminders, plus a CLI for managing events across multiple accounts.

## Installation

### Within this monorepo

Add to your package's `pyproject.toml`:

```toml
[tool.poetry.dependencies]
platform-calendar = { path = "../platform_calendar", develop = true }
```

Then install:

```bash
poetry install
```

### External (if published to PyPI)

```bash
poetry add platform-calendar
```

## Features

- **OAuth 2.0 Authentication**: Secure PKCE-based authentication with Google Calendar API (uses `platform_core.oauth` utilities)
- **Competition Tracking**: Track Kaggle, Devpost, and other competition deadlines
- **Automatic Reminders**: Default 1 day + 1 hour before deadline (configurable)
- **Strict Typing**: mypy strict mode, no Any, 100% test coverage
- **Project Mapping**: Link competitions to your codebase projects
- **Testable Design**: Hooks-based dependency injection for full testability
- **CLI Tool**: Rich terminal interface for managing events across multiple accounts

## Dependencies

- `platform-core` - OAuth types and PKCE utilities, JSON utilities, error handling
- `rich` - Terminal formatting for CLI

## Setup

### 1. Create Google Cloud Project

1. Go to https://console.cloud.google.com/
2. Create new project: "CompetitionTracker"
3. Enable "Google Calendar API" in APIs & Services

### 2. Create OAuth 2.0 Credentials

1. Go to APIs & Services -> Credentials
2. Create Credentials -> OAuth client ID
3. Application type: "Desktop app"
4. Download JSON -> save as `~/.google/calendar_credentials.json`

## CLI Usage

The CLI provides commands for managing calendar events across multiple Google accounts.

### Commands

```bash
# List today's events (default)
poetry run python -m platform_calendar.cli list
poetry run python -m platform_calendar.cli ls
poetry run python -m platform_calendar.cli l

# List events for a specific date
poetry run python -m platform_calendar.cli list --date 2026-02-25

# List tomorrow's events
poetry run python -m platform_calendar.cli tomorrow
poetry run python -m platform_calendar.cli tm

# List this week's events
poetry run python -m platform_calendar.cli week
poetry run python -m platform_calendar.cli w

# List all calendars
poetry run python -m platform_calendar.cli calendars
poetry run python -m platform_calendar.cli cals

# Create an event
poetry run python -m platform_calendar.cli create "Team Meeting" 14:00 --duration 60 --location "Room A"
poetry run python -m platform_calendar.cli add "Lunch" 12:00 --date 2026-02-25

# Delete an event (interactive)
poetry run python -m platform_calendar.cli delete
poetry run python -m platform_calendar.cli rm --date 2026-02-25
```

### CLI Configuration

The CLI reads OAuth tokens from environment variables. Create a `.env` file in the package directory:

```bash
# OAuth Credentials (required for token refresh)
GOOGLE_CALENDAR_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CALENDAR_CLIENT_SECRET=GOCSPX-your_secret
GOOGLE_CALENDAR_REDIRECT_URI=http://localhost

# Account 1: Personal
GOOGLE_CALENDAR_ACCESS_TOKEN=ya29.your_access_token
GOOGLE_CALENDAR_REFRESH_TOKEN=1//your_refresh_token
GOOGLE_CALENDAR_TOKEN_EXPIRES_AT=1735200000

# Account 2: Work (example)
GOOGLE_CALENDAR_INTERNS_ACCESS_TOKEN=ya29.work_access_token
GOOGLE_CALENDAR_INTERNS_REFRESH_TOKEN=1//work_refresh_token
GOOGLE_CALENDAR_INTERNS_EXPIRES_AT=1735200000
```

### Automatic Token Refresh

The CLI automatically refreshes expired access tokens using the refresh token. When a token expires:

1. The CLI checks `TOKEN_EXPIRES_AT` before each API call
2. If expired (or expiring within 60 seconds), it uses the refresh token to get a new access token
3. The new token is cached in memory for subsequent calls

This means you only need to authenticate once - the CLI handles token refresh automatically.

### CLI Output

The CLI uses Rich for formatted output with color-coded styling:
- **Headers**: Bold cyan
- **Account names**: Bold yellow
- **Times**: Green (regular events) or Blue (all-day events)
- **Errors**: Bold red
- **Success messages**: Bold green

## Quick Start (Python API)

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
    location="Conference Room A",
    recurrence=("RRULE:FREQ=WEEKLY;COUNT=10",),
)

# Update event (partial update - only specified fields are changed)
updated = client.update_event(
    calendar_id="primary",
    event_id=event["id"],
    summary="Updated Title",
    description="New description",
    start=EventDateTime(dateTime="2025-12-27T14:00:00Z", timeZone="UTC"),
    end=EventDateTime(dateTime="2025-12-27T15:00:00Z", timeZone="UTC"),
    reminders=(60, 30),
    location="New Location",
    recurrence=("RRULE:FREQ=DAILY;COUNT=5",),
)

# Delete event
client.delete_event(calendar_id="primary", event_id=event["id"])
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

# Use fake HTTP responses for real client testing
hooks.http_get = make_fake_http_get('{"items": []}')
hooks.http_post = make_fake_http_post('{"id": "new123"}')
hooks.load_tokens = make_fake_tokens(tokens)

# Create fake events for testing
fake_event = make_fake_event(
    event_id="test123",
    summary="Fake Event",
    location="Test Location",
    recurrence=("RRULE:FREQ=WEEKLY",),
)
```

### Hooks Architecture

The `HooksContainer` provides dependency injection for all external dependencies:

| Hook | Type | Description |
|------|------|-------------|
| `http_get` | `HttpGetHook` | HTTP GET requests |
| `http_post` | `HttpPostHook` | HTTP POST requests |
| `http_patch` | `HttpPatchHook` | HTTP PATCH requests |
| `http_delete` | `HttpDeleteHook` | HTTP DELETE requests |
| `load_tokens` | `LoadTokensHook` | Load OAuth tokens |
| `save_tokens` | `SaveTokensHook` | Save OAuth tokens |
| `load_credentials` | `LoadCredentialsHook` | Load OAuth credentials |
| `open_browser` | `OpenBrowserHook` | Open browser for OAuth |
| `current_time` | `CurrentTimeHook` | Get current timestamp |
| `read_file` | `ReadFileHook` | Read file contents |
| `write_file` | `WriteFileHook` | Write file contents |
| `file_exists` | `FileExistsHook` | Check file existence |
| `console_output` | `ConsoleOutputHook` | Console output |
| `console_input` | `ConsoleInputHook` | Console input |
| `cli_api_get` | `CliApiGetHook` | CLI API GET requests |
| `cli_api_post` | `CliApiPostHook` | CLI API POST requests |
| `cli_api_delete` | `CliApiDeleteHook` | CLI API DELETE requests |
| `cli_get_env` | `CliGetEnvHook` | CLI environment variables |
| `cli_get_now` | `CliGetNowHook` | CLI current datetime |
| `cli_prompt_ask` | `CliPromptAskHook` | CLI user prompts |
| `cli_confirm_ask` | `CliConfirmAskHook` | CLI confirmations |
| `cli_get_console` | `CliGetConsoleHook` | CLI Rich console |

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

### CLI Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `list` | `ls`, `l` | List events for a date |
| `calendars` | `cals` | List all calendars |
| `create` | `add`, `new` | Create a new event |
| `delete` | `rm`, `del` | Delete an event (interactive) |
| `tomorrow` | `tm` | Show tomorrow's events |
| `week` | `w` | Show this week's events |

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

## Development

```bash
cd libs/platform_calendar
make check  # Run lint + tests
make lint   # Run linting only
make test   # Run tests only
```

## Requirements

- Python 3.11+
- platform-core (OAuth types, PKCE utilities, error handling, JSON utilities)
- rich (terminal formatting for CLI)
- 100% test coverage enforced
