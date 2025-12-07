# music-wrapped-api

Music listening analytics service that aggregates data from Spotify, Apple Music, YouTube Music, and Last.fm to generate yearly "Wrapped" reports.

## Features

- **Multi-Service Support**: Spotify, Apple Music, YouTube Music, Last.fm
- **OAuth Integration**: Full OAuth flow for Spotify and Last.fm
- **Token Storage**: Secure credential storage in Redis
- **YouTube Takeout Import**: Parse Google Takeout data for YouTube Music
- **Background Processing**: RQ workers for async report generation
- **PNG Export**: Visual wrapped cards via image rendering

## Quick Start

```bash
# Start infrastructure (from repo root)
make infra

# Start the service
cd services/music-wrapped-api
docker compose up -d

# Check health
curl http://localhost:8006/healthz
```

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `REDIS_URL` | Yes | Redis connection string |
| `LASTFM_API_KEY` | Yes | Last.fm API key |
| `LASTFM_API_SECRET` | Yes | Last.fm API secret |
| `SPOTIFY_CLIENT_ID` | Yes | Spotify app client ID |
| `SPOTIFY_CLIENT_SECRET` | Yes | Spotify app client secret |
| `APPLE_DEVELOPER_TOKEN` | Yes | Apple Music developer token |
| `PORT` | No | Server port (default: 8000) |

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  music-wrapped-api                                          │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ FastAPI         │───▶│ Redis                           │ │
│  │ - Auth routes   │    │ - Token storage                 │ │
│  │ - Generate      │    │ - Job queue (RQ)                │ │
│  │ - Status/Result │    │ - Result cache                  │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
│           │                           │                     │
│           ▼                           ▼                     │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ OAuth Flows     │    │ RQ Worker                       │ │
│  │ - Spotify       │    │ - Fetch listening history       │ │
│  │ - Last.fm       │    │ - Aggregate statistics          │ │
│  └─────────────────┘    │ - Store result in Redis         │ │
│                         └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Endpoints

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Liveness probe |
| GET | `/readyz` | Readiness probe (checks Redis) |

### Authentication

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/wrapped/auth/spotify/start` | Start Spotify OAuth flow |
| GET | `/v1/wrapped/auth/spotify/callback` | Spotify OAuth callback |
| GET | `/v1/wrapped/auth/lastfm/start` | Start Last.fm OAuth flow |
| GET | `/v1/wrapped/auth/lastfm/callback` | Last.fm OAuth callback |
| POST | `/v1/wrapped/auth/youtube/store` | Store YouTube Music credentials |
| POST | `/v1/wrapped/auth/apple/store` | Store Apple Music token |

### Wrapped Generation

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/wrapped/generate` | Start wrapped generation job |
| POST | `/v1/wrapped/import/youtube-takeout` | Import YouTube Takeout file |
| GET | `/v1/wrapped/status/{job_id}` | Get job status and progress |
| GET | `/v1/wrapped/result/{result_id}` | Get JSON result |
| GET | `/v1/wrapped/download/{result_id}` | Download PNG image |
| GET | `/v1/wrapped/schema` | Get WrappedResult JSON schema |

## Usage Examples

### Spotify OAuth Flow

```bash
# 1. Start OAuth (get auth URL)
curl "http://localhost:8006/v1/wrapped/auth/spotify/start?callback=http://localhost:3000/callback"

# Response: {"auth_url": "https://accounts.spotify.com/authorize?...", "state": "abc123"}

# 2. User visits auth_url, authorizes, redirected to callback with code

# 3. Exchange code for token
curl "http://localhost:8006/v1/wrapped/auth/spotify/callback?code=AUTH_CODE&state=abc123&callback=http://localhost:3000/callback"

# Response: {"token_id": "abc123def456...", "expires_in": 3600}
```

### Last.fm OAuth Flow

```bash
# 1. Start OAuth
curl "http://localhost:8006/v1/wrapped/auth/lastfm/start?callback=http://localhost:3000/callback"

# Response: {"auth_url": "https://www.last.fm/api/auth/?..."}

# 2. User authorizes, callback receives token

# 3. Exchange token
curl "http://localhost:8006/v1/wrapped/auth/lastfm/callback?token=LASTFM_TOKEN"

# Response: {"session_key": "abc123...", "username": "user123"}
```

### Store YouTube Music Credentials

```bash
curl -X POST http://localhost:8006/v1/wrapped/auth/youtube/store \
  -H "Content-Type: application/json" \
  -d '{"sapisid": "YOUR_SAPISID", "cookies": "YOUR_COOKIES"}'

# Response: {"token_id": "abc123..."}
```

### Store Apple Music Token

```bash
curl -X POST http://localhost:8006/v1/wrapped/auth/apple/store \
  -H "Content-Type: application/json" \
  -d '{"music_user_token": "YOUR_MUSIC_USER_TOKEN"}'

# Response: {"token_id": "abc123..."}
```

### Generate Wrapped Report

```bash
# Using stored token
curl -X POST http://localhost:8006/v1/wrapped/generate \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2024,
    "service": "spotify",
    "credentials": {"token_id": "abc123..."}
  }'

# Using full credentials (Last.fm)
curl -X POST http://localhost:8006/v1/wrapped/generate \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2024,
    "service": "lastfm",
    "credentials": {"session_key": "YOUR_SESSION_KEY"}
  }'

# Response: {"job_id": "job123...", "status": "queued"}
```

### Import YouTube Takeout

```bash
curl -X POST http://localhost:8006/v1/wrapped/import/youtube-takeout \
  -F "file=@watch-history.json" \
  -F "year=2024"

# Response: {"job_id": "job123...", "status": "queued", "token_id": "abc..."}
```

### Check Job Status

```bash
curl http://localhost:8006/v1/wrapped/status/job123

# Response:
# {
#   "job_id": "job123",
#   "status": "finished",
#   "progress": 100,
#   "result_id": "result456..."
# }
```

### Get Result

```bash
# JSON result
curl http://localhost:8006/v1/wrapped/result/result456

# PNG image
curl http://localhost:8006/v1/wrapped/download/result456 -o wrapped.png
```

## Supported Services

### Spotify
- OAuth 2.0 authorization flow
- Scopes: `user-read-recently-played`, `user-top-read`
- Credentials: `access_token`, `refresh_token`, `expires_in`

### Last.fm
- Web authentication flow
- Credentials: `session_key` (optional: `api_key`, `api_secret` from env)

### Apple Music
- Developer token + Music User Token
- Credentials: `music_user_token`, `developer_token`

### YouTube Music
- Cookie-based authentication (SAPISID + cookies)
- Credentials: `sapisid`, `cookies`
- Also supports Google Takeout import

## Error Codes

| HTTP | Code | Description |
|------|------|-------------|
| 400 | `INVALID_INPUT` | Invalid request body or parameters |
| 404 | `NOT_FOUND` | Token or result not found |
| 502 | `EXTERNAL_SERVICE_ERROR` | OAuth provider error |

## Worker

The worker processes wrapped generation jobs asynchronously:

```bash
# Run worker
poetry run music-wrapped-worker
```

Worker entry point: `music_wrapped_api.worker_entry:main`

## Development

```bash
# Install dependencies
poetry install --with dev

# Run checks
make check

# Run tests
make test

# Run linting
make lint
```

## Docker

```bash
# Build and run
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

Network: Requires `platform-network` and `platform-redis` from root docker-compose.

## Related

- [API Documentation](docs/api.md) - Full endpoint specifications
- [platform_music](../../libs/platform_music) - Core music analytics library
