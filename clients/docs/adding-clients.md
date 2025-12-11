# Adding New Clients

This guide covers how to add a new client application to the monorepo, following the patterns established by DiscordBot.

## Directory Structure

Create the following structure:

```
clients/MyClient/
├── src/my_client/
│   ├── __init__.py
│   ├── _test_hooks.py      # Dependency injection hooks
│   ├── config.py           # Configuration re-exports
│   ├── container.py        # Service container (DI composition)
│   └── main.py             # Entry point
├── tests/
│   ├── conftest.py         # Pytest fixtures and hook resets
│   └── support/
│       └── settings.py     # Test settings factory
├── scripts/
│   └── guard.py            # Code quality guard script
├── pyproject.toml
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Step 1: Initialize Poetry Project

```bash
cd clients/MyClient
poetry init --name "my-client" --python "^3.11"
```

## Step 2: Configure pyproject.toml

### Dependencies

Add shared library dependencies as path references:

```toml
[tool.poetry.dependencies]
python = "^3.11"
monorepo-guards = { path = "../../libs/monorepo_guards", develop = true }
platform-core = { path = "../../libs/platform_core", develop = true }
# Add other platform libs as needed:
# platform-discord = { path = "../../libs/platform_discord", develop = true }
# platform-workers = { path = "../../libs/platform_workers", develop = true }

[tool.poetry.group.dev.dependencies]
pytest = "^9.0.0"
pytest-asyncio = "^1.3.0"
pytest-cov = "^7.0.0"
pytest-xdist = "^3.6.1"
ruff = "^0.14.4"
mypy = "^1.13.0"
```

### Strict Type Checking

Configure mypy with strict settings (no `Any`, no `cast`):

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
warn_unused_configs = true
disallow_subclassing_any = true
disallow_any_generics = true
disallow_any_unimported = true
disallow_any_expr = true
disallow_any_decorated = true
disallow_any_explicit = true
no_implicit_optional = true
check_untyped_defs = true
no_implicit_reexport = true
show_error_codes = true
pretty = true
files = ["src", "tests", "scripts"]
mypy_path = ["src"]
explicit_package_bases = true
```

### Ruff Configuration

Ban `Any`, `cast`, and `TypeAlias`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "scripts", "tests"]
exclude = [".venv"]

[tool.ruff.lint]
select = ["E","F","I","B","BLE","UP","N","C4","SIM","RET","C90","RUF","ANN"]
ignore = []
fixable = ["ALL"]

[tool.ruff.lint.flake8-annotations]
allow-star-arg-any = false
mypy-init-return = true

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.Any" = { msg = "Do not use typing.Any; prefer precise types or Protocols/TypedDicts." }
"typing.cast" = { msg = "Do not use typing.cast; prefer adapters or precise types." }
"typing.TypeAlias" = { msg = "Do not use TypeAlias; use Literal types or expand unions explicitly." }
```

### Coverage Requirements

Enforce 100% statement and branch coverage:

```toml
[tool.coverage.run]
source = ["src", "scripts"]
omit = []
branch = true

[tool.coverage.report]
precision = 2
show_missing = true
fail_under = 100
```

## Step 3: Create Test Hooks Module

The `_test_hooks.py` pattern enables dependency injection without mocking frameworks. Production code calls hooks; tests replace them with fakes.

```python
# src/my_client/_test_hooks.py
"""Test hooks for my_client - allows injecting test dependencies.

Production code calls hooks directly. Tests assign fake implementations.

Usage in production:
    from my_client import _test_hooks
    settings = _test_hooks.load_settings()

Usage in tests:
    from my_client import _test_hooks
    _test_hooks.load_settings = lambda: build_settings()
"""
from __future__ import annotations

from typing import Protocol

from platform_core.config import MyClientSettings
from platform_core.config import load_my_client_settings as _real_load_settings


class LoadSettingsProtocol(Protocol):
    """Protocol for settings loader function."""
    def __call__(self) -> MyClientSettings: ...


def _default_load_settings() -> MyClientSettings:
    """Production implementation - loads from environment."""
    return _real_load_settings()


# Hook for settings loading. Tests override to return test settings.
load_settings: LoadSettingsProtocol = _default_load_settings


# Add more hooks as needed for:
# - HTTP clients
# - Redis connections
# - External service clients
# - File system operations
```

### Hook Pattern Guidelines

1. **Define a Protocol** for each hookable dependency
2. **Create a default implementation** that calls production code
3. **Export a module-level callable** that production code uses
4. **Tests override** by assigning a different callable

Example for HTTP client:

```python
class BuildClientProtocol(Protocol):
    def __call__(self, timeout: float) -> HttpxClient: ...

def _default_build_client(timeout: float) -> HttpxClient:
    return _real_build_client(timeout)

build_client: BuildClientProtocol = _default_build_client
```

## Step 4: Create Configuration Module

Re-export configuration types from `platform_core`:

```python
# src/my_client/config.py
from __future__ import annotations

from platform_core.config import (
    MyClientSettings,
    RedisConfig,
    # ... other config types
)

from . import _test_hooks


def load_my_client_settings() -> MyClientSettings:
    """Load settings via hook (allows test injection)."""
    return _test_hooks.load_settings()


__all__ = [
    "MyClientSettings",
    "RedisConfig",
    "load_my_client_settings",
]
```

## Step 5: Create Service Container

The container composes services and wires dependencies:

```python
# src/my_client/container.py
from __future__ import annotations

from typing import ClassVar

from platform_core.logging import get_logger

from . import _test_hooks
from .config import MyClientSettings, load_my_client_settings


class ServiceContainer:
    """Application container for dependency injection."""

    __slots__ = ("cfg", "my_service")

    _logger: ClassVar = get_logger(__name__)

    def __init__(
        self,
        *,
        cfg: MyClientSettings,
        my_service: MyService,
    ) -> None:
        self.cfg = cfg
        self.my_service = my_service

    @classmethod
    def from_env(cls) -> ServiceContainer:
        """Create container from environment configuration."""
        cfg = load_my_client_settings()
        my_service = MyService(cfg)
        return cls(cfg=cfg, my_service=my_service)
```

## Step 6: Create Test Fixtures

### conftest.py

```python
# tests/conftest.py
from __future__ import annotations

from collections.abc import Generator

import pytest

from my_client import _test_hooks
from my_client.config import MyClientSettings
from tests.support.settings import build_settings


def reset_hooks() -> None:
    """Reset all test hooks to their default implementations."""
    _test_hooks.load_settings = _test_hooks._default_load_settings
    # Reset other hooks...


@pytest.fixture(autouse=True)
def _reset_hooks_fixture() -> Generator[None, None, None]:
    """Autouse fixture that resets hooks after each test."""
    test_settings = build_settings()

    def _test_load_settings() -> MyClientSettings:
        return test_settings

    _test_hooks.load_settings = _test_load_settings
    yield
    reset_hooks()
```

### Test Settings Factory

```python
# tests/support/settings.py
from __future__ import annotations

from typing import Protocol

from my_client.config import MyClientSettings


class SettingsFactory(Protocol):
    """Protocol for settings factory callable."""
    def __call__(
        self,
        *,
        api_url: str | None = None,
        # ... other overrides
    ) -> MyClientSettings: ...


def build_settings(
    *,
    api_url: str | None = None,
    # ... other overrides with defaults
) -> MyClientSettings:
    """Build settings with test defaults."""
    return {
        "api": {
            "url": api_url or "http://localhost:8000",
            # ... other fields with test defaults
        },
        # ... other config sections
    }
```

## Step 7: Create Guard Script

```python
# scripts/guard.py
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from my_client import _test_hooks


def main(argv: Sequence[str] | None = None) -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    monorepo_root = _test_hooks.guard_find_monorepo_root(project_root)
    run_for_project = _test_hooks.guard_load_orchestrator(monorepo_root)

    args = list(argv) if argv is not None else list(sys.argv[1:])
    root_override: Path | None = None
    verbose = False
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--root" and idx + 1 < len(args):
            root_override = Path(args[idx + 1]).resolve()
            idx += 2
        elif token in ("-v", "--verbose"):
            verbose = True
            idx += 1
        else:
            idx += 1

    target_root = root_override if root_override is not None else project_root
    rc = run_for_project(monorepo_root=monorepo_root, project_root=target_root)
    if verbose:
        sys.stdout.write(f"guard_exit_code code={rc}\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(None))
```

Add hooks to `_test_hooks.py`:

```python
class GuardFindMonorepoRootProtocol(Protocol):
    def __call__(self, start: Path) -> Path: ...

class GuardRunForProjectProtocol(Protocol):
    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int: ...

class GuardLoadOrchestratorProtocol(Protocol):
    def __call__(self, monorepo_root: Path) -> GuardRunForProjectProtocol: ...

def _default_guard_find_monorepo_root(start: Path) -> Path:
    current = start
    while True:
        if (current / "libs").is_dir():
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root not found")
        current = current.parent

def _default_guard_load_orchestrator(monorepo_root: Path) -> GuardRunForProjectProtocol:
    import sys
    libs_path = monorepo_root / "libs"
    guards_src = libs_path / "monorepo_guards" / "src"
    sys.path.insert(0, str(guards_src))
    sys.path.insert(0, str(libs_path))
    mod = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    return mod.run_for_project

guard_find_monorepo_root: GuardFindMonorepoRootProtocol = _default_guard_find_monorepo_root
guard_load_orchestrator: GuardLoadOrchestratorProtocol = _default_guard_load_orchestrator
```

## Step 8: Create Makefile

```makefile
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check

lint:
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; poetry env remove --all | Out-Null }; exit 0
	if ((Test-Path ".\\scripts\\guard.py") -or (Test-Path ".\\scripts\\guard\\__main__.py")) { poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE } }
	poetry lock
	poetry install --with dev
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

test:
	poetry lock
	poetry install --with dev
	$$covArgs = @("--cov-branch","--cov-report=term-missing"); $$cands = @("src","scripts"); foreach ($$c in $$cands) { if (Test-Path (Join-Path "." $$c)) { $$covArgs += "--cov=$$c" } }; poetry run pytest -v @covArgs

check: lint | test
```

## Step 9: Create Docker Configuration

### Dockerfile

Use a multi-stage build with builder and runtime stages. The build context is the monorepo root to access shared libs:

```dockerfile
# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9-slim-bookworm
ARG APP_DIR=clients/MyClient

# ---------- Builder: resolve deps and build wheel ----------
FROM python:${PYTHON_VERSION} AS builder

ARG APP_DIR=clients/MyClient

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "poetry==${POETRY_VERSION}"

WORKDIR /workspace

# Copy project and shared libs in monorepo layout
COPY ${APP_DIR}/pyproject.toml ${APP_DIR}/poetry.lock ${APP_DIR}/README.md ./clients/MyClient/
COPY ${APP_DIR}/src ./clients/MyClient/src
COPY ${APP_DIR}/scripts ./clients/MyClient/scripts
COPY libs ./libs

WORKDIR /workspace/clients/MyClient

RUN poetry build -f wheel

# ---------- Runtime base: install the wheel ----------
FROM python:${PYTHON_VERSION} AS runtime-base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /workspace

COPY --from=builder /workspace/clients/MyClient/dist/*.whl /workspace/dist/
COPY --from=builder /workspace/libs /workspace/libs
RUN pip install /workspace/dist/*.whl /workspace/libs/platform_core \
    && rm -rf /workspace/dist /workspace/libs /root/.cache

WORKDIR /app

USER app

# ---------- Main target ----------
FROM runtime-base AS main

CMD ["python", "-m", "my_client.main"]
```

### docker-compose.yml

Note the build context is `../..` (monorepo root) to access shared libs:

```yaml
services:
  my-client:
    build:
      context: ../..
      dockerfile: clients/MyClient/Dockerfile
      target: main
    env_file:
      - .env
    environment:
      - REDIS_URL=${REDIS_URL:-redis://platform-redis:6379/0}
    networks:
      - platform-network
    restart: unless-stopped

networks:
  platform-network:
    external: true
```

## Step 10: Document Environment Variables

### .env.example

```bash
# =============================================================================
# My Client Configuration
# =============================================================================

# --- Required ---
MY_SERVICE_API_URL=http://my-service:8000

# --- Optional ---
LOG_LEVEL=INFO

# --- Redis (for background jobs) ---
REDIS_URL=redis://platform-redis:6379/0
```

## Connecting to Backend Services

### HTTP Clients

Use `platform_core.http_client` types with test hooks for injection:

```python
# src/my_client/services/my_api_client.py
from __future__ import annotations

from platform_core.http_client import HttpxAsyncClient
from platform_core.json_utils import JSONValue

from my_client import _test_hooks


class MyServiceClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: int = 30,
        client: HttpxAsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = float(timeout_seconds)
        # Use hook for testability - tests inject fake clients
        self._client: HttpxAsyncClient = (
            _test_hooks.build_async_client(self._timeout) if client is None else client
        )

    async def get_data(self, item_id: str) -> MyResponse:
        resp = await self._client.get(f"{self._base}/items/{item_id}")
        if resp.status_code >= 400:
            raise MyAPIError(resp.status_code, resp.text)
        body: JSONValue = resp.json()
        if not isinstance(body, dict):
            raise MyAPIError(500, "Invalid response")
        return _decode_response(body)

    async def aclose(self) -> None:
        await self._client.aclose()
```

Add the hook to `_test_hooks.py`:

```python
from platform_core.http_client import HttpxAsyncClient
from platform_core.http_client import build_async_client as _real_build_async_client

class BuildAsyncClientProtocol(Protocol):
    def __call__(self, timeout: float) -> HttpxAsyncClient: ...

def _default_build_async_client(timeout: float) -> HttpxAsyncClient:
    return _real_build_async_client(timeout)

build_async_client: BuildAsyncClientProtocol = _default_build_async_client
```

### Event Subscriptions

For Redis pub/sub event subscriptions, use `platform_discord.bot_subscriber`:

```python
from platform_discord.bot_subscriber import BotEventSubscriber

class MyEventSubscriber(BotEventSubscriber[MyEventV1]):
    def __init__(self, *, bot: BotProto, redis_url: str) -> None:
        super().__init__(
            bot,
            redis_url=redis_url,
            events_channel="my-service:events",
            task_name="my-event-subscriber",
            decode=decode_my_event,
        )

    async def _handle_event(self, ev: MyEventV1) -> None:
        # Process event
        pass
```

### Background Jobs

For RQ job queues, use `platform_workers.rq_harness` with test hooks:

```python
# src/my_client/services/jobs/my_enqueuer.py
from __future__ import annotations

from typing import Protocol, TypedDict

from platform_workers.rq_harness import (
    RQClientQueue,
    RQJobLike,
    RQRetryLike,
    _RedisBytesClient,
)
from platform_workers.rq_harness import _JsonValue as _RQJsonValue

from my_client import _test_hooks


class MyEnqueuer(Protocol):
    def enqueue_job(self, *, request_id: str, data: str) -> str: ...


class RQMyEnqueuerConfig(TypedDict):
    redis_url: str
    queue_name: str
    job_timeout_s: int
    result_ttl_s: int


class RQMyEnqueuer:
    def __init__(
        self,
        *,
        redis_url: str,
        queue_name: str,
        job_timeout_s: int = 3600,
        result_ttl_s: int = 86400,
    ) -> None:
        self._config: RQMyEnqueuerConfig = {
            "redis_url": redis_url,
            "queue_name": queue_name,
            "job_timeout_s": job_timeout_s,
            "result_ttl_s": result_ttl_s,
        }

    def enqueue_job(self, *, request_id: str, data: str) -> str:
        # Use hooks for testability
        conn: _RedisBytesClient = _test_hooks.redis_raw_for_rq(self._config["redis_url"])
        queue: RQClientQueue = _test_hooks.rq_queue(
            self._config["queue_name"], connection=conn
        )
        payload: _RQJsonValue = {
            "type": "my_job.v1",
            "request_id": request_id,
            "data": data,
        }
        job: RQJobLike = queue.enqueue(
            "my_worker.process_job",
            payload,
            job_timeout=self._config["job_timeout_s"],
            result_ttl=self._config["result_ttl_s"],
            description=f"my-job:{request_id}",
        )
        return job.get_id()
```

Add the hooks to `_test_hooks.py`:

```python
from platform_workers.rq_harness import (
    RQClientQueue,
    RQRetryLike,
    _RedisBytesClient,
)
from platform_workers.rq_harness import redis_raw_for_rq as _real_redis_raw_for_rq
from platform_workers.rq_harness import rq_queue as _real_rq_queue

class RqBytesClientFactoryProtocol(Protocol):
    def __call__(self, url: str) -> _RedisBytesClient: ...

class RqQueueProtocol(Protocol):
    def __call__(self, name: str, *, connection: _RedisBytesClient) -> RQClientQueue: ...

def _default_redis_raw_for_rq(url: str) -> _RedisBytesClient:
    return _real_redis_raw_for_rq(url)

def _default_rq_queue(name: str, *, connection: _RedisBytesClient) -> RQClientQueue:
    return _real_rq_queue(name, connection=connection)

redis_raw_for_rq: RqBytesClientFactoryProtocol = _default_redis_raw_for_rq
rq_queue: RqQueueProtocol = _default_rq_queue
```

## Quality Checklist

Before merging a new client:

- [ ] All tests pass with 100% statement and branch coverage
- [ ] `make check` passes (guards + ruff + mypy + pytest)
- [ ] No `Any`, `cast`, or `type: ignore` in codebase
- [ ] All hooks have corresponding Protocol definitions
- [ ] All hooks have `_default_*` implementations exported
- [ ] conftest.py resets all hooks after each test
- [ ] README.md documents all environment variables
- [ ] .env.example includes all configuration options
- [ ] Dockerfile builds successfully (from monorepo root context)
- [ ] docker-compose.yml connects to platform-network
- [ ] scripts/guard.py runs successfully
- [ ] `__all__` exports defined in all public modules
