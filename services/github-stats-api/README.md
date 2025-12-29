# GitHub Stats API

Strictly typed GitHub statistics SVG card generation API. A Python reimplementation of [github-readme-stats](https://github.com/anuraghazra/github-readme-stats) designed for Railway deployment.

## Features

- **SVG Card Generation**: Dynamic GitHub stats and language cards
- **10 Themes**: Basic (default, dark, dracula, github_dark, transparent) and premium with animations (cyberpunk, synthwave, neon, aurora, radical)
- **Animated Effects**: Premium themes include pulsing glow, twinkling sparkles, and gradient backgrounds
- **Layout Options**: Default, compact, donut, and pie layouts for language cards
- **Type Safety**: mypy strict mode, zero `Any` types, Protocol-based DI via `_test_hooks.py` pattern
- **100% Test Coverage**: Statements and branches
- **Caching**: Configurable response caching to reduce GitHub API load

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- GitHub personal access token with `read:user` scope

### Installation

```bash
cd services/github-stats-api
poetry install --with dev
```

### Run Locally

```bash
# Set environment variables
export GITHUB_TOKEN=ghp_xxx

# Development (with reload)
poetry run hypercorn github_stats_api.asgi:app --bind 0.0.0.0:8000 --reload

# Production
poetry run hypercorn github_stats_api.asgi:app --bind [::]:${PORT:-8000}

# Verify
curl http://localhost:8000/health
```

### Docker

```bash
# With docker-compose (integrated with platform network)
cp .env.example .env
# Edit .env with your GITHUB_TOKEN
docker compose up -d

# Or standalone build from monorepo root
docker build -t github-stats-api -f services/github-stats-api/Dockerfile ../..
docker run -e GITHUB_TOKEN=ghp_xxx -p 8000:8000 github-stats-api

# Verify
curl http://localhost:8000/healthz
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe |
| `/health` | GET | Health check |
| `/api` | GET | Generate user stats SVG card |
| `/api/top-langs` | GET | Generate top languages SVG card |

### Example Usage

**Stats Card (with animated theme):**
```
/api?username=wagner-austin&theme=cyberpunk&hide_border=true&show_icons=true
```

**Languages Card (with animated theme):**
```
/api/top-langs?username=wagner-austin&theme=cyberpunk&layout=compact&langs_count=8
```

### Available Themes

| Theme | Type | Effects |
|-------|------|---------|
| `default` | Basic | Light theme, no animations |
| `dark` | Basic | Dark theme, no animations |
| `dracula` | Basic | Dracula colors, no animations |
| `github_dark` | Basic | GitHub dark mode, no animations |
| `transparent` | Basic | Transparent background, no animations |
| `cyberpunk` | Premium | Cyan/magenta with gradient, glow pulse, sparkles |
| `synthwave` | Premium | Pink/blue 80s with gradient, glow pulse, sparkles |
| `neon` | Premium | Green/red neon with gradient, glow pulse, sparkles |
| `aurora` | Premium | Green/teal with gradient, glow pulse, sparkles |
| `radical` | Premium | Pink/yellow with gradient, glow pulse, sparkles |

---

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GITHUB_TOKEN` | string | Required | GitHub PAT with `read:user` scope |
| `CACHE_TTL_SECONDS` | int | `1800` | Response cache duration (30 min) |
| `PORT` | int | `8000` | HTTP server port |

### Example .env

```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CACHE_TTL_SECONDS=1800
PORT=8000
```

---

## Architecture

### Component Overview

```
github_stats_api/
├── api/                    # FastAPI routes
│   ├── main.py            # App factory
│   ├── routes/            # Endpoint handlers
│   │   ├── health.py      # Health endpoints
│   │   └── stats.py       # Stats/langs endpoints
│   ├── schemas/           # TypedDict definitions
│   │   └── stats.py       # Request/response types
│   └── validators/        # Request validation
│       └── stats.py       # Query param decoders
├── _test_hooks.py         # DI hooks for testing
├── client.py              # GitHub GraphQL client
├── github_client.py       # Protocol + query strings
├── svg_renderer.py        # SVG card generation
├── themes.py              # Color theme definitions
├── settings.py            # Environment config
└── asgi.py                # Production entrypoint
```

### Test Hooks Pattern (`_test_hooks.py`)

The service uses dependency injection via a `_test_hooks.py` module for testability without mocks:

```python
# In _test_hooks.py
_build_client_hook: Callable[[float], HttpxAsyncClient] = _default_build_client

def get_client_hook() -> Callable[[float], HttpxAsyncClient]:
    """Get current client builder hook."""
    return _build_client_hook

def set_client_hook(hook: Callable[[float], HttpxAsyncClient]) -> None:
    """Set client builder hook for testing."""
    global _build_client_hook
    _build_client_hook = hook

def reset_client_hook() -> None:
    """Reset client builder hook to default."""
    global _build_client_hook
    _build_client_hook = _default_build_client

# In routes/stats.py
build_client = get_client_hook()
client = build_client(_HTTP_TIMEOUT_SECONDS)
```

Tests install fake implementations; production uses the default real implementation.

### Design Documentation

- [Architecture](./docs/architecture.md) - Full architectural design
- [API Reference](./docs/api.md) - Complete API documentation

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
poetry run pytest tests/test_validators.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

---

## Project Structure

```
github-stats-api/
├── src/github_stats_api/
│   ├── __init__.py
│   ├── _test_hooks.py          # DI hooks for testing
│   ├── asgi.py                 # ASGI entrypoint
│   ├── client.py               # GitHub API client
│   ├── github_client.py        # Protocol + queries
│   ├── settings.py             # Config loader
│   ├── svg_renderer.py         # SVG generation
│   ├── themes.py               # Color themes
│   └── api/
│       ├── __init__.py
│       ├── main.py             # App factory
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── health.py       # Health endpoints
│       │   └── stats.py        # Stats endpoints
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── stats.py        # TypedDicts
│       └── validators/
│           ├── __init__.py
│           └── stats.py        # Request validation
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_asgi.py
│   ├── test_client.py
│   ├── test_github_client.py
│   ├── test_health_routes.py
│   ├── test_hooks.py
│   ├── test_scripts_guard.py
│   ├── test_settings.py
│   ├── test_stats_routes.py
│   ├── test_svg_renderer.py
│   └── test_validators.py
├── scripts/
│   ├── __init__.py
│   └── guard.py                # Monorepo guard harness
├── docs/
│   ├── api.md
│   └── architecture.md
├── .env.example
├── DEPLOYING_RAILWAY.md
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── pyproject.toml
```

---

## Deployment

See [DEPLOYING_RAILWAY.md](./DEPLOYING_RAILWAY.md) for detailed Railway deployment instructions.

### Quick Railway Setup

1. **Create service** from GitHub repository (monorepo root)
2. **Set environment variables**:
   ```
   RAILWAY_DOCKERFILE_PATH=services/github-stats-api/Dockerfile
   GITHUB_TOKEN=ghp_xxx
   CACHE_TTL_SECONDS=1800
   ```
3. **Port** automatically configured via `PORT` env var

The `RAILWAY_DOCKERFILE_PATH` tells Railway to use the service's Dockerfile while building from the monorepo root (required to access `libs/platform_core`).

### Health Checks

- **Liveness**: `/healthz`
- **Readiness**: `/readyz`
- **General**: `/health`

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `httpx` | Async HTTP client |
| `platform-core` | Logging, errors, config, HTTP client protocols |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel tests |
| `pytest-asyncio` | Async test support |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |

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
