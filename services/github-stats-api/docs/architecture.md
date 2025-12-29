# Architecture: github-stats-api Service

## Overview

The `github-stats-api` service generates dynamic SVG cards displaying GitHub user statistics. It's a Python reimplementation of [github-readme-stats](https://github.com/anuraghazra/github-readme-stats), built with:

- FastAPI for the HTTP API
- GitHub GraphQL API for fetching user data
- Custom SVG rendering with theme support
- Protocol-based dependency injection for testability
- 100% test coverage (statements and branches)

## Dependencies

| Package | Purpose |
|---------|---------|
| `platform-core` | Exception handlers, logging, HTTP client protocols, JSON utilities |
| `httpx` | Async HTTP client for GitHub API (via platform-core protocols) |
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |

## Directory Structure

```
services/github-stats-api/
├── pyproject.toml
├── README.md
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── DEPLOYING_RAILWAY.md
├── docs/
│   ├── api.md                 # API reference
│   └── architecture.md        # This file
├── scripts/
│   ├── __init__.py
│   └── guard.py               # Monorepo guard harness
├── src/github_stats_api/
│   ├── __init__.py
│   ├── _test_hooks.py         # DI hooks for testing
│   ├── asgi.py                # Production ASGI entrypoint
│   ├── settings.py            # Environment config
│   ├── themes.py              # Color theme definitions
│   ├── github_client.py       # Protocol + GraphQL queries
│   ├── client.py              # GitHub API client implementation
│   ├── svg_renderer.py        # SVG card generation
│   └── api/
│       ├── __init__.py
│       ├── main.py            # FastAPI app factory
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── stats.py       # Request/response TypedDicts
│       ├── validators/
│       │   ├── __init__.py
│       │   └── stats.py       # Query param validation
│       └── routes/
│           ├── __init__.py
│           ├── health.py      # Health check endpoints
│           └── stats.py       # Stats card endpoints
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_asgi.py
    ├── test_client.py
    ├── test_github_client.py
    ├── test_health_routes.py
    ├── test_hooks.py
    ├── test_scripts_guard.py
    ├── test_settings.py
    ├── test_stats_routes.py
    ├── test_svg_renderer.py
    └── test_validators.py
```

---

## GitHub GraphQL API

The service uses GitHub's GraphQL API (v4) for efficient data fetching:

### User Stats Query

```graphql
query userInfo($login: String!) {
  user(login: $login) {
    login
    name
    contributionsCollection {
      totalCommitContributions
      restrictedContributionsCount
    }
    pullRequests(first: 1) { totalCount }
    openIssues: issues(states: OPEN) { totalCount }
    closedIssues: issues(states: CLOSED) { totalCount }
    repositories(first: 100, ownerAffiliations: OWNER) {
      nodes { stargazerCount }
    }
    repositoriesContributedTo(first: 1) { totalCount }
  }
}
```

### Languages Query

```graphql
query userLanguages($login: String!) {
  user(login: $login) {
    repositories(first: 100, ownerAffiliations: OWNER, isFork: false) {
      nodes {
        languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
          edges {
            size
            node { name, color }
          }
        }
      }
    }
  }
}
```

---

## Core Types (schemas/stats.py)

### Request Types

```python
# Theme literal type for all requests
ThemeName = Literal[
    "default", "dark", "dracula", "github_dark", "transparent",
    "cyberpunk", "synthwave", "neon", "aurora", "radical",
]

class StatsRequest(TypedDict, total=True):
    username: str
    theme: ThemeName
    hide_border: bool
    show_icons: bool
    include_all_commits: bool
    hide: tuple[str, ...]  # stats to hide
    disable_animations: bool

class LangsRequest(TypedDict, total=True):
    username: str
    theme: ThemeName
    hide_border: bool
    layout: Literal["default", "compact", "donut", "pie"]
    langs_count: int
    hide: tuple[str, ...]  # languages to hide
    disable_animations: bool
```

### Response Types

```python
class UserStats(TypedDict, total=True):
    username: str
    name: str
    total_commits: int
    total_prs: int
    total_issues: int
    total_stars: int
    total_contributions: int
    rank: Literal["S+", "S", "A+", "A", "B+", "B", "C"]
    rank_percentile: float

class LanguageStats(TypedDict, total=True):
    name: str
    size: int
    percentage: float
    color: str
```

---

## Key Modules

### 1. github_client.py - Protocols & Queries

Defines the protocol and GraphQL query strings:

```python
class GitHubClientProto(Protocol):
    async def fetch_user_stats(self, username: str) -> GitHubUserData: ...
    async def fetch_languages(self, username: str) -> list[GitHubLanguageData]: ...

_USER_STATS_QUERY = """..."""
_LANGUAGES_QUERY = """..."""
```

### 2. client.py - GitHub API Client

Production implementation using platform_core HTTP client:

```python
class GitHubClient:
    def __init__(self, token: str, client: HttpxAsyncClient) -> None:
        self._token = token
        self._client = client

    async def fetch_user_stats(self, username: str) -> GitHubUserData:
        # POST to https://api.github.com/graphql
        # Parse response and aggregate stats
        ...

    async def fetch_languages(self, username: str) -> list[GitHubLanguageData]:
        # POST to https://api.github.com/graphql
        # Aggregate languages across repos
        ...
```

### 3. themes.py - Color Themes

```python
class GradientStop(TypedDict, total=True):
    offset: int    # Percentage 0-100
    color: str     # Hex color

class Gradient(TypedDict, total=True):
    angle: int                       # Degrees (0=right, 90=down)
    stops: tuple[GradientStop, ...]  # Color stops

class Theme(TypedDict, total=True):
    bg_color: str           # Background color (fallback if no gradient)
    title_color: str        # Title text color
    text_color: str         # Body text color
    border_color: str       # Border color
    icon_color: str         # Icon color
    gradient: Gradient | None      # Optional gradient background
    glow_color: str | None         # Optional glow color for text
    sparkle_color: str | None      # Optional sparkle decoration color
    sparkle_count: int             # Number of sparkle decorations (0=none)

# Basic themes (no visual effects)
_THEMES: dict[str, Theme] = {
    "default": {..., "gradient": None, "glow_color": None, ...},
    "dark": {...},
    "dracula": {...},
    "github_dark": {...},
    "transparent": {...},
    # Premium themes with visual effects
    "cyberpunk": {..., "gradient": {...}, "glow_color": "#00fff9", "sparkle_count": 8},
    "synthwave": {..., "gradient": {...}, "glow_color": "#f72585", "sparkle_count": 6},
    "neon": {..., "gradient": {...}, "glow_color": "#39ff14", "sparkle_count": 10},
    "aurora": {..., "gradient": {...}, "glow_color": "#78ffd6", "sparkle_count": 12},
    "radical": {..., "gradient": {...}, "glow_color": "#fe428e", "sparkle_count": 8},
}

def get_theme(name: str) -> Theme: ...
def get_theme_names() -> tuple[str, ...]: ...
```

### 4. svg_renderer.py - SVG Generation

```python
def render_stats_card(
    stats: UserStats,
    theme_name: str,
    hide_border: bool,
    show_icons: bool,
    hide: tuple[str, ...],
) -> str:
    # Generate SVG string with:
    # - Card background/border
    # - Title with user name
    # - Stat rows (stars, commits, PRs, issues)
    # - Rank circle (S+, S, A+, A, B+, B, C)

def render_langs_card(
    username: str,
    languages: list[LanguageStats],
    total_size: int,
    theme_name: str,
    hide_border: bool,
    layout: str,
    langs_count: int,
) -> str:
    # Generate SVG string with:
    # - Compact bar layout or list layout
    # - Donut or pie chart layouts
    # - Language colors and percentages
```

### 5. validators/stats.py - Request Validation

```python
def decode_stats_request(
    username: str | None,
    theme: str | None,
    hide_border: str | None,
    ...
) -> StatsRequest:
    # Validates:
    # - Username format (alphanumeric + hyphens, max 39 chars)
    # - Theme is valid literal
    # - Boolean params are "true"/"false"/"1"/"0"
    # - Hide list contains valid stat names

def decode_langs_request(...) -> LangsRequest: ...
```

### 6. routes/stats.py - HTTP Endpoints

```python
async def get_stats(request: Request, ...) -> Response:
    req = decode_stats_request(...)
    build_client = get_client_hook()
    async with build_client(_HTTP_TIMEOUT_SECONDS) as http_client:
        gh = GitHubClient(settings["github_token"], http_client)
        data = await gh.fetch_user_stats(req["username"])
    stats = build_user_stats(data)
    svg = render_stats_card(...)
    return Response(content=svg, media_type="image/svg+xml")

async def get_top_langs(request: Request, ...) -> Response:
    ...

def build_router(settings_provider) -> APIRouter:
    router.add_api_route("/api", get_stats, ...)
    router.add_api_route("/api/top-langs", get_top_langs, ...)
```

---

## Test Hooks Pattern (`_test_hooks.py`)

The service uses dependency injection via a `_test_hooks.py` module for testability without mocks:

```python
# In _test_hooks.py
from collections.abc import Callable
from platform_core.http_client import HttpxAsyncClient, build_async_client

def _default_build_client(timeout_seconds: float) -> HttpxAsyncClient:
    """Default client builder using real httpx."""
    return build_async_client(timeout_seconds=timeout_seconds)

# Hook for building HTTP clients - tests can replace with fake
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
```

### Usage in Production Code

```python
# In routes/stats.py
from ..._test_hooks import get_client_hook

async def get_stats(request: Request, ...) -> Response:
    build_client = get_client_hook()
    async with build_client(_HTTP_TIMEOUT_SECONDS) as client:
        gh = GitHubClient(settings["github_token"], client)
        ...
```

### Usage in Tests

```python
# In tests/test_stats_routes.py
from github_stats_api._test_hooks import reset_client_hook, set_client_hook
from platform_core.testing import FakeHttpxAsyncClient, FakeHttpxResponse

@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    yield
    reset_client_hook()

async def test_get_stats_returns_svg() -> None:
    fake_response = FakeHttpxResponse(200, _make_fake_user_response())

    def build_fake_client(timeout: float) -> HttpxAsyncClient:
        return FakeHttpxAsyncClient(fake_response)

    set_client_hook(build_fake_client)
    # ... test code
```

---

## Ranking Algorithm

User rank is calculated based on activity:

```python
score = commits * 1 + prs * 2 + issues * 1 + stars * 4
percentile = 100 - (log10(score + 1) * 15)

# Rank thresholds:
# S+: top 1%     (percentile <= 1)
# S:  top 12.5%  (percentile <= 12.5)
# A+: top 25%    (percentile <= 25)
# A:  top 37.5%  (percentile <= 37.5)
# B+: top 50%    (percentile <= 50)
# B:  top 62.5%  (percentile <= 62.5)
# C:  rest       (percentile > 62.5)
```

---

## Caching Strategy

Responses include cache headers to reduce GitHub API load:

```python
headers={
    "Cache-Control": f"max-age={settings['cache_ttl_seconds']}, s-maxage={settings['cache_ttl_seconds']}",
}
```

Default TTL is 30 minutes (1800 seconds).

---

## Error Handling

Uses `platform_core.errors.AppError`:

- `400 Bad Request` - Invalid query params (INVALID_INPUT)
- `404 Not Found` - User not found (NOT_FOUND)
- `502 Bad Gateway` - GitHub API errors (EXTERNAL_SERVICE_ERROR)

All errors include:
- `code` - Error code string
- `message` - Human-readable description
- `request_id` - UUID for tracing

---

## Testing Strategy

### Protocol-Based DI

All external dependencies use protocols from platform_core:
- `HttpxAsyncClient` - Async HTTP client protocol
- `HttpxResponse` - Response protocol

### Fake Implementations (from platform_core.testing)

```python
from platform_core.testing import (
    FakeHttpxAsyncClient,
    FakeHttpxResponse,
)

# Create fake response
fake_response = FakeHttpxResponse(200, {"data": {...}})

# Create fake client
fake_client = FakeHttpxAsyncClient(fake_response)
```

### Test Categories

1. **Unit Tests**: Individual functions (validators, renderers, helpers)
2. **Integration Tests**: Full endpoint tests with fake HTTP clients
3. **Guard Tests**: Verify guard script execution

### Coverage Requirements

- 100% statement and branch coverage
- Tests for each endpoint with various query param combinations
- Tests for validation edge cases (all themes, layouts, boolean formats)
- Tests for SVG rendering (structure validation)
- Tests for error handling (user not found, API errors)
- Tests for rank calculation (all rank levels)

---

## Deployment

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GITHUB_TOKEN` | Yes | - | GitHub PAT with `read:user` scope |
| `CACHE_TTL_SECONDS` | No | `1800` | Response cache duration |
| `PORT` | No | `8000` | HTTP server port |

### Docker Compose (Local)

```bash
# With platform network integration (port 8009)
cp .env.example .env
# Edit .env with your GITHUB_TOKEN
docker compose up -d
```

### Docker (Standalone)

```bash
# Build from monorepo root
docker build -t github-stats-api -f services/github-stats-api/Dockerfile .

# Run
docker run -e GITHUB_TOKEN=ghp_xxx -p 8000:8000 github-stats-api
```

### Railway

Deploy with environment variables:
- `RAILWAY_DOCKERFILE_PATH=services/github-stats-api/Dockerfile` (required for monorepo)
- `GITHUB_TOKEN` - GitHub PAT with `read:user` scope
- Port automatically configured via `PORT` env var
- Health checks on `/healthz` and `/readyz`

See [DEPLOYING_RAILWAY.md](../DEPLOYING_RAILWAY.md) for detailed instructions.

---

## Quality Standards

- **Type Safety**: mypy strict mode, no `Any`, no `cast`, no `type: ignore`
- **Coverage**: 100% statements and branches
- **Guard Rules**: Enforced via `scripts/guard.py`
- **No Mocks**: Tests use fake implementations from platform_core.testing
- **No Dataclasses**: All structured types use TypedDict
