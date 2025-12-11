# Contributing to Music Wrapped API

Guidelines for contributing to the music-wrapped-api service.

## Project Overview

Music Wrapped API is a FastAPI-based microservice that:
- Aggregates music listening data from Spotify, Apple Music, YouTube Music, and Last.fm
- Generates yearly "Wrapped" analytics reports
- Provides OAuth integration for Spotify and Last.fm
- Processes jobs asynchronously via Redis Queue (RQ)
- Renders visual wrapped cards as PNG images

## Project Structure

```
music-wrapped-api/
├── src/
│   └── music_wrapped_api/
│       ├── api/
│       │   ├── main.py           # FastAPI app factory
│       │   ├── health.py         # Health check helpers
│       │   └── routes/
│       │       ├── wrapped.py    # Main wrapped endpoints
│       │       ├── health.py     # Health check routes
│       │       └── _decoders.py  # Request parsing/validation
│       ├── _test_hooks.py        # Dependency injection hooks
│       ├── asgi.py               # ASGI entrypoint
│       ├── health.py             # Health utilities
│       └── worker_entry.py       # RQ worker entrypoint
├── tests/                        # 91 test cases
├── scripts/
│   └── guard.py                  # Monorepo rule enforcement
├── Makefile                      # PowerShell-based commands
├── pyproject.toml                # Poetry dependencies
└── Dockerfile                    # Multi-stage build
```

## Core Principles

This project maintains **zero technical debt** standards:

- **No Any types:** All code uses precise types (TypedDict, Protocols, Literals)
- **No type: ignore:** All type errors resolved at source
- **No casts:** Type narrowing via isinstance checks only
- **No mocks:** Test hooks pattern for dependency injection
- **100% test coverage:** All statements, branches, and edge cases covered
- **Platform library integration:** Uses centralized libs for Redis, RQ, config, logging

## Development Setup

### Prerequisites

- **Python 3.11+**
- **Poetry** (package management)
- **Redis** (local development or Docker)
- **PowerShell** (Windows) or compatible shell

### Initial Setup

1. **Navigate to the service:**
   ```bash
   cd services/music-wrapped-api
   ```

2. **Install dependencies:**
   ```bash
   poetry install --with dev
   ```

3. **Run checks:**
   ```bash
   make check
   ```

This runs:
- Guard script validation
- Ruff linting and formatting
- Mypy strict type checking
- Pytest with 100% coverage enforcement

## Development Workflow

### Makefile Commands

```bash
make check      # Run all checks (guard, lint, test)
make lint       # Run guard, ruff, mypy
make test       # Run pytest with coverage
```

### Type System Standards

#### Use TypedDict for Data Models

```python
from typing import TypedDict

class SpotifyCredentials(TypedDict):
    access_token: str
    refresh_token: str
    expires_in: int
```

#### Use Protocols for Dependencies

```python
from typing import Protocol

class RedisStrProto(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    def close(self) -> None: ...
```

#### Use JSONValue for External Data

```python
from platform_core.json_utils import JSONValue

def parse_response(data: dict[str, JSONValue]) -> SpotifyCredentials:
    access_token = data.get("access_token")
    if not isinstance(access_token, str):
        raise TypeError("access_token must be string")
    # ... validate all fields
    return {"access_token": access_token, ...}
```

## Test Hooks Pattern

This project uses **test hooks** for dependency injection instead of mocks or monkeypatch. The `_test_hooks.py` module provides module-level hooks that tests override.

### How It Works

**In production code:**

```python
# src/music_wrapped_api/api/routes/wrapped.py
from music_wrapped_api import _test_hooks

def get_job_status(job_id: str) -> dict[str, str]:
    conn = _test_hooks.rq_conn(redis_url)
    job = _test_hooks.get_job(job_id, conn)
    return {"status": job.get_status()}
```

**In tests:**

```python
# tests/test_wrapped_status.py
from music_wrapped_api import _test_hooks
from platform_workers.testing import FakeRedisBytesClient

class _FakeJob:
    def get_status(self) -> str:
        return "finished"

def test_get_job_status() -> None:
    def _fake_get_job(job_id: str, connection: _RedisBytesClient) -> _FakeJob:
        return _FakeJob()

    _test_hooks.get_job = _fake_get_job

    result = get_job_status("job-123")
    assert result["status"] == "finished"
```

### Available Hooks

The `_test_hooks.py` module provides hooks for:

| Hook | Purpose |
|------|---------|
| `get_env` | Read environment variables |
| `require_env` | Read required environment variables |
| `redis_factory` | Create Redis KV client |
| `rq_conn` | Create RQ Redis connection |
| `rq_queue_factory` | Create RQ queue |
| `get_job` | Fetch RQ job by ID |
| `build_renderer` | Create wrapped image renderer |
| `urlopen_get` | HTTP GET requests |
| `urlopen_post` | HTTP POST requests |
| `make_request` | Create HTTP request objects |
| `lfm_get_session_json` | Last.fm session API |
| `spotify_exchange_code` | Spotify OAuth token exchange |
| `rand_state` | Generate random OAuth state |
| `guard_find_monorepo_root` | Find monorepo root (for guard script) |
| `guard_load_orchestrator` | Load guard orchestrator |

### Conftest Auto-Reset

The `tests/conftest.py` includes an autouse fixture that:

1. Sets up a fake environment with test values
2. Configures default hook implementations for tests
3. Automatically resets all hooks after each test

```python
# tests/conftest.py
@pytest.fixture(autouse=True)
def _default_test_env() -> None:
    """Set up default test environment and hooks."""
    env = make_fake_env({
        "REDIS_URL": "redis://test-redis:6379/0",
        "SPOTIFY_CLIENT_ID": "test-spotify-id",
        # ... other test values
    })
    config_test_hooks.get_env = env
    # ... configure other hooks
```

### Why Test Hooks?

- **No dynamic patching:** Avoids runtime monkey-patching complexity
- **Type-safe:** Hooks are typed with Protocols
- **Explicit:** Dependencies are visible in function signatures
- **Fast:** No mock framework overhead
- **Parallel-safe:** Each test can configure its own hooks

**Never use:** `unittest.mock`, `monkeypatch`, `@patch`, or any dynamic patching.

## Platform Library Integration

### Centralized Redis/RQ

Use the centralized Redis and RQ utilities from `platform_workers`:

```python
from platform_workers.redis import RedisStrProto, redis_for_kv
from platform_workers.rq_harness import rq_queue, redis_raw_for_rq, RQClientQueue
from platform_workers.testing import FakeRedis, FakeQueue, FakeRedisBytesClient
```

### Centralized Configuration

Use `platform_core.config` for environment variables:

```python
from platform_core.config import _optional_env_str, _require_env_str
```

In tests, use `FakeEnv`:

```python
from platform_core.testing import FakeEnv, make_fake_env
from platform_core.config import _test_hooks as config_test_hooks

env = make_fake_env({"REDIS_URL": "redis://test:6379/0"})
config_test_hooks.get_env = env
```

### Centralized JSON Utilities

```python
from platform_core.json_utils import JSONValue, load_json_str, dump_json_str
from platform_core.errors import AppError, ErrorCode
```

## Testing Requirements

### 100% Coverage Enforcement

All code must have:
- **Statement coverage:** 100%
- **Branch coverage:** 100%
- **Edge case coverage:** All error paths, validation, type checks

```bash
make test  # Fails if coverage < 100%
```

### Test Organization

```
tests/
├── test_<feature>.py           # Main functionality tests
├── test_<feature>_<scenario>.py  # Specific scenarios
├── test_hooks_defaults.py      # Tests for production hook implementations
└── conftest.py                 # Pytest fixtures, hook reset
```

### Writing Tests

1. **Override hooks** for the dependencies you need to control
2. **Use centralized fakes** from `platform_workers.testing` and `platform_core.testing`
3. **Avoid weak assertions** (no `hasattr`, `isinstance` checks as assertions)
4. **Test production defaults** to achieve coverage of `_test_hooks.py`

Example:

```python
def test_spotify_auth_callback() -> None:
    def _fake_exchange(
        code: str, redirect_uri: str, client_id: str, client_secret: str
    ) -> dict[str, JSONValue]:
        return {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}

    _test_hooks.spotify_exchange_code = _fake_exchange

    client = TestClient(create_app())
    resp = client.get("/v1/wrapped/auth/spotify/callback?code=abc&state=xyz&callback=http://x")
    assert resp.status_code == 200
```

## Pull Request Process

### Before Submitting

1. **Run full checks:**
   ```bash
   make check
   ```

2. **Verify 100% coverage:**
   - All statements covered
   - All branches covered

3. **Ensure no violations:**
   - Guard scripts pass (0 violations)
   - Mypy passes (no errors)
   - Ruff passes (no lint errors)

### PR Requirements

- All tests pass (91+ tests)
- 100% coverage maintained
- Guard scripts pass (all rule categories)
- Mypy strict checks pass
- Ruff formatting applied
- No type: ignore comments
- No mocks or monkeypatch
- Documentation updated (if adding features)

## Related Documentation

- [README.md](../README.md) - Service overview and quick start
- [API Reference](api.md) - Complete endpoint documentation
- [platform_music](../../../libs/platform_music) - Core music analytics library
- [platform_workers](../../../libs/platform_workers) - RQ and Redis utilities
- [platform_core](../../../libs/platform_core) - Configuration and utilities
