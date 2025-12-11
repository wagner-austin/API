# Music Wrapped API

Music listening analytics service that aggregates data from Spotify, Apple Music, YouTube Music, and Last.fm to generate yearly "Wrapped" reports. Features OAuth integration, background job processing, and PNG export.

## Features

- **Multi-Service Support**: Spotify, Apple Music, YouTube Music, Last.fm
- **OAuth Integration**: Full OAuth flow for Spotify and Last.fm
- **Token Storage**: Secure credential storage in Redis
- **YouTube Takeout Import**: Parse Google Takeout data for YouTube Music
- **Background Processing**: RQ workers for async report generation
- **PNG Export**: Visual wrapped cards via image rendering
- **Type Safety**: mypy strict mode, zero `Any` types
- **100% Test Coverage**: Statements and branches

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- Redis (for job queue and token storage)

### Installation

```bash
cd services/music-wrapped-api
poetry install --with dev
```

### Start with Docker (from repository root)

```bash
# Start infrastructure
make infra

# Start the service
cd services/music-wrapped-api
docker compose up -d

# Verify
curl http://localhost:8006/healthz
curl http://localhost:8006/readyz
```

### Local Development

```bash
# Start Redis
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Run API
poetry run hypercorn music_wrapped_api.asgi:app --bind 0.0.0.0:8006

# Run Worker (separate terminal)
poetry run music-wrapped-worker
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe (checks Redis) |
| `/v1/wrapped/auth/spotify/start` | GET | Start Spotify OAuth flow |
| `/v1/wrapped/auth/spotify/callback` | GET | Spotify OAuth callback |
| `/v1/wrapped/auth/lastfm/start` | GET | Start Last.fm OAuth flow |
| `/v1/wrapped/auth/lastfm/callback` | GET | Last.fm OAuth callback |
| `/v1/wrapped/auth/youtube/store` | POST | Store YouTube Music credentials |
| `/v1/wrapped/auth/apple/store` | POST | Store Apple Music token |
| `/v1/wrapped/generate` | POST | Start wrapped generation job |
| `/v1/wrapped/import/youtube-takeout` | POST | Import YouTube Takeout file |
| `/v1/wrapped/status/{job_id}` | GET | Get job status and progress |
| `/v1/wrapped/result/{result_id}` | GET | Get JSON result |
| `/v1/wrapped/download/{result_id}` | GET | Download PNG image |
| `/v1/wrapped/schema` | GET | Get WrappedResult JSON schema |

---

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_URL` | string | **Required** | Redis connection string |
| `LASTFM_API_KEY` | string | **Required** | Last.fm API key |
| `LASTFM_API_SECRET` | string | **Required** | Last.fm API secret |
| `SPOTIFY_CLIENT_ID` | string | **Required** | Spotify app client ID |
| `SPOTIFY_CLIENT_SECRET` | string | **Required** | Spotify app client secret |
| `APPLE_DEVELOPER_TOKEN` | string | **Required** | Apple Music developer token |
| `PORT` | int | `8006` | Server port |
| `LOGGING__LEVEL` | string | `INFO` | Log level |

### Example .env

```bash
REDIS_URL=redis://localhost:6379/0
LASTFM_API_KEY=your-lastfm-api-key
LASTFM_API_SECRET=your-lastfm-api-secret
SPOTIFY_CLIENT_ID=your-spotify-client-id
SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
APPLE_DEVELOPER_TOKEN=your-apple-developer-token
PORT=8006
```

---

## Architecture

### Component Overview

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

### Supported Services

#### Spotify
- OAuth 2.0 authorization flow
- Scopes: `user-read-recently-played`, `user-top-read`
- Credentials: `access_token`, `refresh_token`, `expires_in`

#### Last.fm
- Web authentication flow
- Credentials: `session_key` (optional: `api_key`, `api_secret` from env)

#### Apple Music
- Developer token + Music User Token
- Credentials: `music_user_token`, `developer_token`

#### YouTube Music
- Cookie-based authentication (SAPISID + cookies)
- Credentials: `sapisid`, `cookies`
- Also supports Google Takeout import

---

## Development

### Commands

```bash
make install      # Install dependencies
make install-dev  # Install with dev dependencies
make lint         # Run guards + ruff + mypy
make test         # Run pytest with coverage
make check        # Run lint + test
```

### Quality Gates

All code must pass:

1. **Guard Scripts**: No `Any`, no `cast`, no `type: ignore`
2. **Ruff**: Linting and formatting
3. **Mypy**: Strict type checking
4. **Pytest**: 100% statement and branch coverage

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/test_auth.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

---

## Project Structure

```
music-wrapped-api/
├── src/music_wrapped_api/
│   ├── __init__.py
│   ├── api/
│   │   ├── main.py           # App factory and routes
│   │   ├── auth.py           # OAuth flows
│   │   ├── generate.py       # Wrapped generation
│   │   └── schemas.py        # Request/response models
│   ├── services/
│   │   ├── spotify.py        # Spotify integration
│   │   ├── lastfm.py         # Last.fm integration
│   │   ├── apple.py          # Apple Music integration
│   │   └── youtube.py        # YouTube Music integration
│   ├── worker/
│   │   └── jobs.py           # RQ job handlers
│   └── worker_entry.py       # Worker entry point
├── tests/
├── scripts/
│   └── guard.py
├── docs/
│   ├── api.md
│   └── CONTRIBUTING.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

---

## Deployment

### Docker

```bash
# Build
docker build -t music-wrapped-api:latest .

# Run
docker run -p 8006:8006 \
  -e REDIS_URL=redis://redis:6379/0 \
  -e LASTFM_API_KEY=your-key \
  -e LASTFM_API_SECRET=your-secret \
  -e SPOTIFY_CLIENT_ID=your-client-id \
  -e SPOTIFY_CLIENT_SECRET=your-client-secret \
  -e APPLE_DEVELOPER_TOKEN=your-token \
  music-wrapped-api:latest
```

### Docker Compose

```yaml
version: "3.8"
services:
  music-wrapped:
    build: .
    ports:
      - "8006:8006"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis

  worker:
    build: .
    command: music-wrapped-worker
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
```

Network: Requires `platform-network` and `platform-redis` from root docker-compose.

### Railway

```bash
# Set environment variables in Railway dashboard
railway up
```

**Health Check Path:** `/healthz`

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `redis` | Token storage and job queue |
| `rq` | Background job processing |
| `python-multipart` | File upload handling |
| `platform-core` | Logging, errors, config |
| `platform-workers` | RQ worker harness |
| `platform-music` | Music analytics library |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-asyncio` | Async test support |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel tests |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |

---

## Error Handling

All errors return JSON with consistent format:

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable description",
  "request_id": "uuid-for-tracing"
}
```

### Error Codes

| HTTP | Code | Description |
|------|------|-------------|
| 400 | `INVALID_INPUT` | Invalid request body or parameters |
| 404 | `NOT_FOUND` | Token or result not found |
| 502 | `EXTERNAL_SERVICE_ERROR` | OAuth provider error |

---

## Quality Standards

- **Type Safety**: mypy strict mode, no `Any`, no `cast`
- **Coverage**: 100% statements and branches
- **Guard Rules**: Enforced via `scripts/guard.py`
- **Logging**: Structured JSON via platform_core
- **Errors**: Consistent `{code, message, request_id}` format

---

## License

Apache-2.0
