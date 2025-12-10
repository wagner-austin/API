# Testing Patterns

This document covers testing patterns used across the monorepo to achieve 100% coverage without mocks.

## Core Principles

1. **No mocks** - No `unittest.mock`, `MagicMock`, `@patch`, or `monkeypatch.setattr`
2. **No coverage exclusions** - No `# pragma: no cover` or `exclude_lines` config
3. **Real code paths** - Tests execute actual production code
4. **Dependency injection** - Use test hooks for external dependencies

The only acceptable `monkeypatch` usage is `setenv`/`delenv` for environment variables.

## Test Hooks Pattern

### Problem: Testing `if __name__ == "__main__"`

The standard `if __name__ == "__main__": main()` guard cannot be tested by simply calling `main()` - coverage won't see the guard line as executed.

Using `runpy.run_module("module", run_name="__main__")` seems like a solution, but it **re-executes the module fresh**. Any module-level state you set is lost:

```python
# This DOES NOT work:
import worker_entry
worker_entry._test_runner = fake_runner  # Set on imported module

runpy.run_module("worker_entry", run_name="__main__")
# ^ Fresh execution creates NEW _test_runner = None, ignores our fake
```

### Solution: Separate Test Hooks Module

Put test hooks in a **separate module** that gets **imported**. When `runpy` re-executes the main module, it imports the hooks module which is already in `sys.modules` with our test values:

```
worker/
├── __init__.py
├── _test_hooks.py    # <-- Separate module for test injection
└── worker_entry.py   # <-- Imports _test_hooks
```

**`_test_hooks.py`**:
```python
"""Test hooks for worker entry - allows injecting test runner before module load."""

from __future__ import annotations

from typing import Protocol

from platform_workers.rq_harness import WorkerConfig


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None: ...


# Tests set this BEFORE running worker_entry as __main__
test_runner: WorkerRunnerProtocol | None = None
```

**`worker_entry.py`**:
```python
from . import _test_hooks

def _get_default_runner() -> WorkerRunnerProtocol:
    """Get the default worker runner."""
    if _test_hooks.test_runner is not None:
        return _test_hooks.test_runner
    return run_rq_worker

def main() -> None:
    # ... setup ...
    runner = _get_default_runner()
    runner(config)

if __name__ == "__main__":
    main()
```

**Test**:
```python
def test_main_guard_executes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the if __name__ == '__main__' guard executes main()."""
    import runpy
    import sys

    monkeypatch.setenv("REDIS_URL", "redis://test:6379/0")

    received_configs: list[WorkerConfig] = []

    def _recording_runner(config: WorkerConfig) -> None:
        received_configs.append(config)

    # Set test hook BEFORE running as __main__
    _test_hooks.test_runner = _recording_runner

    # Remove module from sys.modules to avoid RuntimeWarning
    module_name = "my_service.worker.worker_entry"
    saved_module = sys.modules.pop(module_name, None)

    # Run as __main__ - this executes the guard
    runpy.run_module(module_name, run_name="__main__", alter_sys=False)

    # Restore
    if saved_module is not None:
        sys.modules[module_name] = saved_module
    _test_hooks.test_runner = None

    assert len(received_configs) == 1
```

### Why This Works

1. Test sets `_test_hooks.test_runner = fake_runner`
2. `runpy.run_module` re-executes `worker_entry.py` fresh
3. Fresh execution does `from . import _test_hooks`
4. Python finds `_test_hooks` already in `sys.modules` (from step 1)
5. Fresh execution sees `_test_hooks.test_runner = fake_runner`
6. Guard line executes, calling `main()` which uses our fake runner

## Container Factory Pattern

### Problem: Testing Code That Creates Real Connections

Container classes often create real Redis/database connections:

```python
class ServiceContainer:
    @classmethod
    def from_settings(cls, settings: Settings) -> ServiceContainer:
        redis = redis_for_kv(settings["redis_url"])      # Real Redis!
        db = psycopg.connect(settings["database_url"])   # Real DB!
        return cls(redis=redis, db=db)
```

### Solution: Factory Test Hooks

Create a separate `_test_hooks.py` with injectable factories:

**`core/_test_hooks.py`**:
```python
"""Test hooks for container - allows injecting factory functions."""

from __future__ import annotations

from typing import Protocol

from covenant_persistence import ConnectionProtocol
from platform_workers.redis import RedisStrProto
from platform_workers.rq_harness import _RedisBytesClient


class RedisFactoryProtocol(Protocol):
    def __call__(self, url: str) -> RedisStrProto: ...


class ConnectionFactoryProtocol(Protocol):
    def __call__(self, dsn: str) -> ConnectionProtocol: ...


class RedisRqFactoryProtocol(Protocol):
    def __call__(self, url: str) -> _RedisBytesClient: ...


# Injectable factories - tests set these before calling from_settings()
redis_factory: RedisFactoryProtocol | None = None
connection_factory: ConnectionFactoryProtocol | None = None
redis_rq_factory: RedisRqFactoryProtocol | None = None
```

**`container.py`**:
```python
from . import _test_hooks

def _get_redis_for_kv(url: str) -> RedisStrProto:
    if _test_hooks.redis_factory is not None:
        return _test_hooks.redis_factory(url)
    return redis_for_kv(url)

def _get_psycopg_connect(dsn: str) -> ConnectionProtocol:
    if _test_hooks.connection_factory is not None:
        return _test_hooks.connection_factory(dsn)
    psycopg = __import__("psycopg")
    connect_fn: ConnectCallable = psycopg.connect
    return connect_fn(dsn)
```

**Test**:
```python
def test_from_settings_uses_injected_factories() -> None:
    fake_redis = FakeRedis()
    fake_conn = _FakeConnection()

    def _redis_factory(url: str) -> FakeRedis:
        return fake_redis

    def _connection_factory(dsn: str) -> _FakeConnection:
        return fake_conn

    # Set hooks
    _test_hooks.redis_factory = _redis_factory
    _test_hooks.connection_factory = _connection_factory

    container = ServiceContainer.from_settings(settings)

    # Restore
    _test_hooks.redis_factory = None
    _test_hooks.connection_factory = None

    assert container.redis is fake_redis
```

## Testing Production Paths (No Hook Set)

To achieve 100% coverage, you must also test the production code paths where hooks are `None`. These paths call real external services, so tests must handle connection errors:

```python
def test_get_psycopg_connect_without_hook_attempts_real_connection() -> None:
    """Test production path calls real psycopg.connect."""
    psycopg = __import__("psycopg")
    operational_error: type[Exception] = psycopg.OperationalError

    _test_hooks.connection_factory = None  # Ensure hook is not set

    # Invalid DSN triggers real psycopg, which raises
    with pytest.raises(operational_error):
        _get_psycopg_connect("host= dbname=x")


def test_get_redis_for_kv_without_hook_attempts_real_connection() -> None:
    """Test production path calls real redis_for_kv."""
    from platform_workers.redis import _load_redis_error_class

    redis_error: type[BaseException] = _load_redis_error_class()

    _test_hooks.redis_factory = None

    client = _get_redis_for_kv("redis://nonexistent-host:6379/0")
    with pytest.raises((redis_error, OSError)):
        client.ping()  # Triggers actual connection attempt
```

## Error Type Handling

When catching dynamically-loaded exception types, use typed variable assignments:

```python
# Wrong - mypy complains about Any
from psycopg import OperationalError  # Contains Any in type

# Right - typed assignment
psycopg = __import__("psycopg")
operational_error: type[Exception] = psycopg.OperationalError

# For redis errors
from platform_workers.redis import _load_redis_error_class
redis_error: type[BaseException] = _load_redis_error_class()
```

## Centralized Test Doubles

Use centralized fake implementations from `libs/platform_workers/testing.py`:

```python
from platform_workers.testing import FakeRedis, FakeRedisBytesClient

# FakeRedis implements RedisStrProto
# FakeRedisBytesClient implements RedisBytesProto
```

**Do not create custom Redis stubs.** The `monorepo_guards` package enforces this rule.

## Summary

| Pattern | Use Case |
|---------|----------|
| Separate `_test_hooks` module | Testing `if __name__ == "__main__"` guard |
| Factory test hooks | Testing code that creates real connections |
| `type[Exception]` annotations | Catching dynamically-loaded exception types |
| `FakeRedis` / `FakeRedisBytesClient` | Redis test doubles |
| `monkeypatch.setenv` | Environment variable testing (only acceptable monkeypatch) |
