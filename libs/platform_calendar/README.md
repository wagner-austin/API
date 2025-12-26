# platform_calendar

Google Calendar API integration for tracking competition deadlines with automatic reminders.

## Features

- **OAuth 2.0 Authentication**: Secure authentication with Google Calendar API
- **Competition Tracking**: Track Kaggle, Devpost, and other competition deadlines
- **Automatic Reminders**: Default 1 day + 1 hour before deadline
- **Strict Typing**: mypy strict mode, no Any, 100% test coverage
- **Project Mapping**: Link competitions to your codebase projects

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

## Usage

```python
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
print(f"Created event: {synced['calendar_event_id']}")

# Save for later
comps = load_competitions()
comps = add_competition(comps, synced)
save_competitions(comps)
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

## Development

```bash
cd libs/platform_calendar
make check  # Run lint + tests
make lint   # Run linting only
make test   # Run tests only
```
