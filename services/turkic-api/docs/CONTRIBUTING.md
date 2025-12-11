# Contributing to Turkic API

Thank you for your interest in contributing to the Turkic API project! This document provides guidelines to help you contribute effectively to this production-grade REST API service for Turkic language corpus processing.

## Project Overview

Turkic API is a FastAPI-based microservice that:
- Processes corpus data for 7 Turkic languages (Kazakh, Kyrgyz, Uzbek, Turkish, Uyghur, Finnish, Azerbaijani)
- Provides asynchronous job processing via Redis Queue (RQ)
- Integrates with platform libraries for centralized logging, configuration, and event streaming
- Maintains 100% test coverage with zero technical debt
- Deploys to Railway as separate API and worker services

## Project Structure

```
turkic-api/
├── src/
│   └── turkic_api/
│       ├── api/              # FastAPI application (9 modules)
│       │   ├── main.py       # API entrypoint, creates FastAPI app
│       │   ├── services.py   # Endpoint handlers
│       │   ├── jobs.py       # Job processing implementation
│       │   ├── worker_entry.py  # RQ worker entrypoint
│       │   ├── config.py     # Settings TypedDict
│       │   ├── models.py     # Request/response parsers
│       │   ├── types.py      # Protocols (LoggerProtocol, QueueProtocol)
│       │   ├── dependencies.py  # FastAPI dependency injection
│       │   ├── errors.py     # HTTP exception handlers
│       │   └── job_store.py  # Redis-backed job state persistence
│       └── core/             # Business logic (12 modules)
│           ├── translit.py   # IPA transliteration engine
│           ├── langid.py     # Language identification
│           ├── corpus.py     # Corpus service
│           ├── corpus_download.py  # Corpus file filtering
│           ├── transliteval.py     # Rule-based transliteration
│           └── models.py     # Core TypedDict models
├── tests/                    # 52 test modules, 503 test cases
├── scripts/                  # Development tools
│   ├── guard.py             # Monorepo rule enforcement
│   └── setup_dev.py         # Development setup automation
├── Makefile                 # PowerShell-based commands
├── pyproject.toml           # Poetry dependencies, strict mypy config
├── railway.toml             # Railway deployment config
└── Dockerfile               # Multi-stage build (api + worker targets)
```

## Core Principles

This project maintains **zero technical debt** standards:

- **No Any types:** All code uses precise types (TypedDict, Protocols, Literals)
- **No type: ignore:** All type errors resolved at source
- **No casts:** Type narrowing via isinstance checks only
- **No try/except fallbacks:** Errors handled explicitly with typed exceptions
- **100% test coverage:** All statements, branches, and edge cases covered
- **No backwards compatibility code:** Clean, modern patterns only
- **Platform library integration:** Centralized logging, config, events via platform_core/platform_workers

## Development Setup

### Prerequisites

- **Python 3.11+**
- **Poetry** (package management)
- **Redis** (local development or Docker)
- **PowerShell** (Windows) or compatible shell

### Initial Setup

1. **Clone the monorepo:**
   ```bash
   git clone <monorepo-url>
   cd services/turkic-api
   ```

2. **Install dependencies:**
   ```bash
   poetry install --with dev
   ```

3. **Run guard scripts:**
   ```bash
   poetry run python scripts/guard.py
   ```

4. **Verify installation:**
   ```bash
   make check
   ```

This runs:
- Guard script validation (11 rule categories)
- Ruff linting and formatting
- Mypy strict type checking
- Pytest with 100% coverage enforcement

## Development Workflow

### Makefile Commands

All development commands use PowerShell syntax:

```bash
make check      # Run all checks (guard, lint, test)
make lint       # Run guard, ruff, mypy
make test       # Run pytest with coverage
```

### Guard Script Enforcement

The `scripts/guard.py` enforces 11 monorepo rules:

1. **No stdlib logging imports:** Use `platform_core.logging` only
2. **No pydantic imports:** Use TypedDict for data models
3. **No dataclasses:** Use TypedDict for data models
4. **No werkzeug imports:** Use FastAPI patterns
5. **No flask imports:** FastAPI-only service
6. **No environment variable reads:** Use `platform_core.config`
7. **No subprocess calls:** Prohibited for security
8. **No os.system calls:** Prohibited for security
9. **No shell=True:** Prohibited for security
10. **No relative imports in tests:** Use absolute imports
11. **No type: ignore comments:** Resolve at source

Run guard before every commit:
```bash
poetry run python scripts/guard.py
```

### Type System Standards

#### Use TypedDict for Data Models

```python
from typing_extensions import TypedDict

class JobParams(TypedDict):
    """Parameters for job processing from queue."""
    source: str
    language: str
    script: str | None
    max_sentences: int
    transliterate: bool
    confidence_threshold: float
```

**Never use:** Pydantic models, dataclasses, or plain dicts with no types.

#### Use JSONValue for External Data

```python
from platform_core.json_utils import JSONValue

def parse_external(data: dict[str, JSONValue]) -> JobParams:
    """Parse and validate external data with runtime checks."""
    source = data.get("source")
    if not isinstance(source, str):
        raise TypeError("source must be string")
    # ... validation for all fields
    return {"source": source, ...}
```

**Never use:** `Any` types or unchecked casts.

#### Use Protocols for Dependencies

```python
from typing import Protocol
from turkic_api.api.types import LoggerProtocol

def process_job(logger: LoggerProtocol) -> None:
    """Process job with injected logger dependency."""
    logger.info("Processing started")
```

**Never use:** Concrete classes or `Any` for dependency injection.

### Logging Standards

#### Always Use platform_core.logging

```python
from platform_core.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing job", extra={"job_id": job_id, "status": "started"})
```

**Never use:** `import logging`, `logging.basicConfig`, or stdlib logging patterns.

#### Structured Logging Only

```python
# CORRECT: Structured extra fields
logger.info("Upload complete", extra={"job_id": job_id, "file_id": file_id, "size": size})

# WRONG: String interpolation
logger.info(f"Upload complete for job {job_id}")
```

### Testing Requirements

#### 100% Coverage Enforcement

All code must have:
- **Statement coverage:** 100% (1417/1417 statements)
- **Branch coverage:** 100% (472/472 branches)
- **Edge case coverage:** All error paths, validation, type checks

```bash
# Run tests with coverage
make test

# Coverage fails if < 100%
poetry run pytest -n auto -v --cov-branch --cov=src --cov=scripts
```

#### Test Organization

```
tests/
├── test_<module>_<scenario>.py    # Main functionality tests
├── test_<module>_error_paths.py   # Error handling tests
├── test_<module>_edge_cases.py    # Boundary conditions
└── conftest.py                    # Pytest fixtures
```

#### Test Hooks Pattern

This project uses **test hooks** for dependency injection instead of mocks or monkeypatch. The `_test_hooks.py` module provides module-level hooks that tests override:

```python
# In production code (e.g., api/jobs.py)
from turkic_api import _test_hooks

def process_job(job_id: str) -> None:
    client = _test_hooks.redis_factory(url)  # Uses hook
    # ...

# In tests (e.g., tests/test_jobs.py)
from turkic_api import _test_hooks
from platform_workers.testing import FakeRedis

def test_process_job() -> None:
    # Override hook with fake
    orig = _test_hooks.redis_factory
    _test_hooks.redis_factory = lambda url: FakeRedis()
    try:
        result = process_job("job-123")
        assert result["status"] == "completed"
    finally:
        _test_hooks.redis_factory = orig  # Restore original
```

The `conftest.py` auto-resets all hooks after each test via an autouse fixture.

**Never use:** `unittest.mock`, `monkeypatch`, `@patch`, or any dynamic patching.

#### Parallel Test Execution

Tests run in parallel using pytest-xdist:
```bash
pytest -n auto -v  # Automatically uses all CPU cores
```

All tests must be:
- **Isolated:** No shared state between tests
- **Deterministic:** Same input always produces same output
- **Fast:** Entire suite runs in ~20 seconds

### Code Quality Standards

#### Type Checking (mypy)

All code must pass strict mypy checks:

```toml
[tool.mypy]
strict = true
disallow_any_expr = true
disallow_any_decorated = true
disallow_any_explicit = true
disallow_any_generics = true
warn_unreachable = true
warn_unused_ignores = true
```

**Common patterns:**

```python
# Type narrowing with isinstance
def process(value: str | int) -> str:
    if isinstance(value, str):
        return value.upper()  # mypy knows value is str here
    return str(value)  # mypy knows value is int here

# Literal types for constrained strings
from typing import Literal

Source = Literal["oscar", "wikipedia", "bible", "quran", "udhr", "cc100", "flores"]

def validate_source(s: str) -> Source:
    if s not in ("oscar", "wikipedia", "bible", "quran", "udhr", "cc100", "flores"):
        raise ValueError(f"Invalid source: {s}")
    return s  # type: ignore[return-value]  # OK: We validated at runtime
```

#### Formatting and Linting (ruff)

```bash
# Auto-fix and format
ruff check . --fix
ruff format .
```

Configuration in `pyproject.toml`:
- Line length: 100
- Python 3.11+ syntax
- Import sorting (isort-compatible)

### Platform Library Integration

#### Centralized Configuration

```python
from platform_core.config import get_env_str, get_env_int

redis_url = get_env_str("REDIS_URL", default="redis://localhost:6379/0")
max_workers = get_env_int("MAX_WORKERS", default=4)
```

**Never use:** `os.getenv()`, `os.environ[]`, or direct environment access.

#### Event Publishing

```python
from platform_core.job_events import make_progress_event, encode_job_event

event = make_progress_event(domain="turkic", job_id=job_id, user_id=user_id, progress=50, message="processing")
redis.publish(channel, encode_job_event(event))
```

#### Redis Protocols

```python
from platform_workers.redis import RedisStrProto, redis_for_kv

client: RedisStrProto = redis_for_kv(redis_url)
client.set("key", "value")
client.close()
```

### API Development

#### Endpoint Pattern

```python
from fastapi import APIRouter, Depends
from turkic_api.api.dependencies import get_queue
from turkic_api.api.types import QueueProtocol, JsonDict

router = APIRouter()

@router.post("/jobs")
def create_job(
    payload: JsonDict,
    queue: QueueProtocol = Depends(get_queue),
) -> dict[str, str]:
    """Create new corpus processing job."""
    job = queue.enqueue("process_corpus", payload)
    return {"job_id": job.id, "status": "queued"}
```

#### Request Validation

```python
from turkic_api.api.models import parse_job_create

def create_job_handler(raw_payload: JsonDict) -> JobCreateResponse:
    """Parse and validate job creation request."""
    validated = parse_job_create(raw_payload)  # Raises HTTPException on invalid
    # validated is now typed as JobCreate
    return enqueue_job(validated)
```

### Worker Development

#### Job Processing Pattern

```python
from turkic_api.api.jobs import process_corpus_impl, JobParams, JobResult
from turkic_api.api.config import Settings
from turkic_api.api.types import LoggerProtocol
from platform_workers.redis import RedisStrProto

def process_job(
    job_id: str,
    params: JobParams,
    *,
    redis: RedisStrProto,
    settings: Settings,
    logger: LoggerProtocol,
) -> JobResult:
    """Process corpus job with explicit dependency injection."""
    return process_corpus_impl(job_id, params, redis=redis, settings=settings, logger=logger)
```

**Key principles:**
- Explicit dependency injection (no globals, no module-level state)
- All deps passed as keyword arguments
- Return typed results (JobResult TypedDict)
- Logging via injected LoggerProtocol

## Pull Request Process

### Before Submitting

1. **Run guard scripts:**
   ```bash
   poetry run python scripts/guard.py
   ```

2. **Run full checks:**
   ```bash
   make check
   ```

3. **Verify 100% coverage:**
   ```bash
   poetry run pytest -n auto -v --cov-branch --cov=src --cov=scripts --cov-report=term-missing
   ```

4. **Check git status:**
   ```bash
   git status  # Ensure no unintended changes
   ```

### PR Requirements

- ✅ All tests pass (503+ tests)
- ✅ 100% coverage maintained (statements + branches)
- ✅ Guard scripts pass (all 11 rules)
- ✅ Mypy strict checks pass
- ✅ Ruff formatting applied
- ✅ No type: ignore comments added
- ✅ No backwards compatibility code
- ✅ Documentation updated (if adding features)

### PR Description Template

```markdown
## Changes

Brief description of what changed and why.

## Testing

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] 100% coverage maintained
- [ ] Guard scripts pass
- [ ] Mypy strict checks pass

## Type Safety

- [ ] No Any types introduced
- [ ] No type: ignore added
- [ ] All external data validated with isinstance checks
- [ ] TypedDict used for all data models

## Platform Integration

- [ ] Uses platform_core.logging (not stdlib logging)
- [ ] Uses platform_core.config (not os.getenv)
- [ ] Event publishing uses typed events
```

## Common Patterns

### Error Handling

```python
from fastapi import HTTPException

def validate_source(source: str) -> Literal["oscar", "wikipedia", ...]:
    """Validate source with typed error."""
    valid_sources = ("oscar", "wikipedia", "bible", "quran", "udhr", "cc100", "flores")
    if source not in valid_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source. Must be one of: {', '.join(valid_sources)}"
        )
    return source  # type: ignore[return-value]  # Runtime validation ensures correctness
```

### File Operations

```python
from pathlib import Path

def load_corpus(data_dir: str, source: str, language: str) -> Path:
    """Load corpus file with type-safe path handling."""
    corpus_path = Path(data_dir) / "corpus" / f"{source}_{language}.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    return corpus_path
```

### JSON Parsing

```python
from platform_core.json_utils import JSONValue, load_json_str

def parse_response(payload: str) -> dict[str, str]:
    """Parse JSON response with runtime validation."""
    parsed: JSONValue = load_json_str(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")

    result: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Expected string keys and values")
        result[key] = value
    return result
```

## Questions?

- Check `DESIGN.md` for architecture details
- Check `docs/setup_guide.md` for development environment setup
- Check `src/turkic_api/core/rules/README.md` for transliteration rules
- Open an issue for bugs or feature requests
