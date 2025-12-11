# platform-music

Core music analytics library for Music Wrapped: service integrations, analytics, and wrapped result generation.

## Installation

```bash
poetry add platform-music
```

No external dependencies - this is a pure domain library.

## Quick Start

```python
from platform_music import (
    WrappedGenerator,
    ListeningHistory,
    WrappedResult,
    FakeLastFm,
)

# Generate wrapped result from history
generator = WrappedGenerator()
result: WrappedResult = generator.generate(history)

print(result["total_scrobbles"])
print(result["top_artists"])
print(result["top_songs"])
```

## Models

All models are TypedDicts with no mutable state:

```python
from platform_music import (
    PlayRecord,
    ListeningHistory,
    TopArtist,
    TopSong,
    WrappedResult,
)

# Play record - a single track play
play: PlayRecord = {
    "track": {
        "id": "abc123",
        "title": "Song Title",
        "artist_name": "Artist Name",
        "duration_ms": 210000,
        "service": "lastfm",
    },
    "played_at": "2024-12-25T14:30:00Z",
    "service": "lastfm",
}

# Listening history for a year
history: ListeningHistory = {
    "year": 2024,
    "plays": [play1, play2, ...],
    "total_plays": 1500,
}

# Top artist entry
top_artist: TopArtist = {
    "artist_name": "Artist Name",
    "play_count": 150,
}

# Top song entry
top_song: TopSong = {
    "title": "Song Title",
    "artist_name": "Artist Name",
    "play_count": 50,
}

# Wrapped result
result: WrappedResult = {
    "service": "lastfm",
    "year": 2024,
    "generated_at": "2024-12-31T23:59:59Z",
    "total_scrobbles": 1500,
    "top_artists": [top_artist, ...],
    "top_songs": [top_song, ...],
    "top_by_month": [...],
}
```

## Wrapped Generator

Generate wrapped results from listening history:

```python
from platform_music import WrappedGenerator, ListeningHistory, WrappedResult

generator = WrappedGenerator()

# Generate wrapped from history
result: WrappedResult = generator.generate(
    history=history,
    top_n_artists=10,
    top_n_songs=10,
)

# Access results
print(f"Total scrobbles: {result['total_scrobbles']}")
for artist in result["top_artists"]:
    print(f"{artist['artist_name']}: {artist['play_count']} plays")
```

## Music Service Protocol

Protocol for music service integrations (Last.fm-first design):

```python
from platform_music import MusicServiceProto

class MyMusicService(MusicServiceProto):
    def fetch_history(self, year: int) -> ListeningHistory:
        # Fetch listening history for year
        ...
```

## Service Credentials

Typed credentials for each music service:

```python
from platform_music import (
    LastFmCredentials,
    SpotifyCredentials,
    AppleMusicCredentials,
    YouTubeMusicCredentials,
    ServiceCredentials,
)

# Last.fm credentials
lastfm_creds: LastFmCredentials = {
    "username": "myuser",
    "api_key": "...",
}

# Spotify credentials
spotify_creds: SpotifyCredentials = {
    "access_token": "...",
    "refresh_token": "...",
}

# Union type for any service
creds: ServiceCredentials = lastfm_creds  # or spotify_creds, etc.
```

## Job Processing

Process wrapped jobs from a queue:

```python
from platform_music import WrappedJobPayload, process_wrapped_job

payload: WrappedJobPayload = {
    "user_id": 123,
    "service": "lastfm",
    "year": 2024,
    "credentials": lastfm_creds,
}

result = process_wrapped_job(payload)
```

## Error Codes

Domain-specific error codes:

```python
from platform_music import MusicWrappedErrorCode

# Available error codes
MusicWrappedErrorCode.INVALID_CREDENTIALS
MusicWrappedErrorCode.SERVICE_UNAVAILABLE
MusicWrappedErrorCode.NO_HISTORY_FOUND
MusicWrappedErrorCode.RATE_LIMITED
```

## Testing

Fake Last.fm service for unit tests:

```python
from platform_music import FakeLastFm

# Create fake service with predefined history
fake = FakeLastFm(history=mock_history)

# Use in tests
result = fake.fetch_history(2024)
```

## API Reference

### Models

| Type | Description |
|------|-------------|
| `PlayRecord` | Single track play event |
| `ListeningHistory` | Year's listening history |
| `TopArtist` | Artist with play count |
| `TopSong` | Song with play count |
| `WrappedResult` | Complete wrapped result |

### Generator

| Type | Description |
|------|-------------|
| `WrappedGenerator` | Wrapped result generator |

### Credentials

| Type | Description |
|------|-------------|
| `LastFmCredentials` | Last.fm API credentials |
| `SpotifyCredentials` | Spotify OAuth tokens |
| `AppleMusicCredentials` | Apple Music credentials |
| `YouTubeMusicCredentials` | YouTube Music credentials |
| `ServiceCredentials` | Union of all credential types |

### Jobs

| Type | Description |
|------|-------------|
| `WrappedJobPayload` | Job payload for processing |
| `process_wrapped_job` | Process a wrapped job |

### Protocols

| Protocol | Description |
|----------|-------------|
| `MusicServiceProto` | Music service interface |

### Testing

| Type | Description |
|------|-------------|
| `FakeLastFm` | Fake Last.fm for tests |

### Error Codes

| Type | Description |
|------|-------------|
| `MusicWrappedErrorCode` | Domain error codes |

## Design Principles

- **Strict typing**: TypedDict and Protocol only
- **No Any, cast, or type: ignore**
- **Parse at edges**: Validate/decode at service boundaries
- **Pure analytics**: Side-effect-free computations
- **100% test coverage**: Statement and branch coverage

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- No external dependencies
- 100% test coverage enforced
