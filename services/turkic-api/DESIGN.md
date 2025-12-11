# Turkic API - Architecture & Design

Production-grade REST API for Turkic language corpus processing with **zero technical debt**, **100% test coverage**, and **strict type safety**.

---

## Core Principles

### Zero Compromise on Quality
- **No Any types** - Explicit typing throughout
- **No casts** - Type narrowing via runtime checks
- **No type: ignore** - All type errors resolved
- **No try/except** - Explicit error propagation
- **No backwards compatibility** - Clean, modern patterns only
- **No fallbacks** - Single, correct implementation path

### Type Safety (Enforced by mypy --strict)
```toml
[tool.mypy]
strict = true
disallow_any_expr = true          # No Any escapes
disallow_any_decorated = true     # No decorator Any
disallow_any_explicit = true      # No typing.Any
disallow_any_generics = true      # No unparameterized generics
```

### Test Coverage (Enforced by pytest-cov)
```toml
[tool.coverage.report]
fail_under = 100     # 100% statement coverage required
branch = true        # 100% branch coverage required
```

### Code Quality (Enforced by guard scripts)
- No Pydantic models (TypedDict only)
- No `TYPE_CHECKING` imports
- Proper `UnknownJson` usage at API boundaries
- Platform library integration (platform_core, platform_workers)

---

## Technology Stack

### Runtime
- **Python**: 3.11+
- **Framework**: FastAPI 0.121+
- **ASGI Server**: Hypercorn 0.18+
- **Job Queue**: Redis 5.2+ with RQ 2.0+
- **Data Bank Client**: platform_core.data_bank_client (httpx under the hood)

### Platform Libraries (Monorepo)
- **platform_core**: Shared types, logging, config, events
- **platform_workers**: RQ harness, Redis protocols, worker utilities

### Development
- **Package Manager**: Poetry 1.8+
- **Type Checker**: mypy 1.13+ (strict mode)
- **Linter**: Ruff 0.14+
- **Formatter**: Ruff + Black 25.11+
- **Testing**: pytest 9.0+ with coverage, asyncio, xdist, timeout

### Deployment
- **Platform**: Railway
- **Container**: Multi-stage Dockerfile (api + worker targets)
- **Storage**: Railway Persistent Volumes
- **Queue**: Railway Redis addon

---

## Architecture Overview

```
┌─────────────────────── Railway Platform ───────────────────────┐
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  API Service (Hypercorn + FastAPI)                       │  │
│  │  - POST /api/v1/jobs         (create corpus job)         │  │
│  │  - GET  /api/v1/jobs/{id}    (get job status)            │  │
│  │  - GET  /api/v1/jobs/{id}/result (stream result)         │  │
│  │  - GET  /healthz             (liveness probe)            │  │
│  │  - GET  /readyz              (readiness probe)           │  │
│  └─────────────┬────────────────────────────────────────────┘  │
│                │                                                │
│                ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Redis (Railway Addon)                                   │  │
│  │  - RQ job queue (TURKIC_QUEUE)                           │  │
│  │  - Job status storage (TurkicJobStatus hashes)           │  │
│  │  - Event publishing (turkic:events via job_events)      │  │
│  └─────────────┬────────────────────────────────────────────┘  │
│                │                                                │
│                ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RQ Worker Service (python -m worker_entry)              │  │
│  │  - Processes jobs from queue                             │  │
│  │  - Downloads corpus (OSCAR/Wikipedia)                    │  │
│  │  - Filters by language & confidence                      │  │
│  │  - Transliterates to IPA                                 │  │
│  │  - Uploads to data-bank-api                              │  │
│  │  - Publishes events (started/progress/completed/failed)  │  │
│  └─────────────┬────────────────────────────────────────────┘  │
│                │                                                │
│                ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Persistent Volume (/data)                               │  │
│  │  - corpus/{source}_{lang}.txt (cached datasets)          │  │
│  │  - models/lid.176.bin (FastText language ID model)       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │  data-bank-api       │
                 │  (External Service)  │
                 │  - File uploads      │
                 │  - File metadata     │
                 └──────────────────────┘
```

---

## Project Structure

```
turkic-api/
├── src/
│   └── turkic_api/                 # Main package
│       ├── api/                    # FastAPI application layer
│       │   ├── main.py            # App factory
│       │   ├── config.py          # Settings (wraps platform_core)
│       │   ├── dependencies.py    # Dependency injection
│       │   ├── services.py        # Business logic (JobService)
│       │   ├── jobs.py            # RQ job implementation
│       │   ├── job_store.py       # Redis storage adapter
│       │   ├── health.py          # Health check logic
│       │   ├── models.py          # TypedDict models
│       │   ├── types.py           # Type aliases & protocols
│       │   ├── validators.py      # Input validation
│       │   ├── streaming.py       # Data bank streaming
│       │   ├── logging_fields.py  # Logging configuration
│       │   ├── provider_context.py # Dependency providers
│       │   └── routes/            # Route handlers
│       │       ├── health.py      # Health endpoints
│       │       └── jobs.py        # Job endpoints
│       ├── worker_entry.py        # RQ worker entry point
│       └── core/                  # Core business logic
│           ├── translit.py        # Transliteration engine
│           ├── transliteval.py    # Rule evaluation
│           ├── langid.py          # FastText language detection
│           ├── corpus.py          # Corpus streaming protocol
│           ├── corpus_download.py # OSCAR/Wikipedia download
│           ├── models.py          # Core type aliases
│           └── rules/             # Transliteration rules (*.rules)
│               ├── kk_ipa.rules   # Kazakh → IPA
│               ├── ky_ipa.rules   # Kyrgyz → IPA
│               ├── uz_ipa.rules   # Uzbek → IPA
│               ├── tr_ipa.rules   # Turkish → IPA
│               ├── ug_ipa.rules   # Uyghur → IPA
│               ├── fi_ipa.rules   # Finnish → IPA
│               ├── az_ipa.rules   # Azerbaijani → IPA
│               └── ...            # Latin conversion rules
├── scripts/
│   ├── guard.py                   # Code quality enforcement
│   └── README.md
├── tests/                         # Test suite (52 files, 503 tests)
│   ├── test_api.py               # API endpoint tests
│   ├── test_jobs_*.py            # Job processing tests
│   ├── test_worker*.py           # Worker tests
│   ├── test_*_ipa_letters.py     # Transliteration tests
│   └── ...
├── docs/                          # Documentation
│   ├── index.md
│   ├── CONTRIBUTING.md
│   └── setup_guide.md
├── Makefile                       # Development commands
├── Dockerfile                     # Multi-stage build (api + worker)
├── railway.toml                   # Railway deployment config
├── pyproject.toml                 # Poetry + tool config
└── README.md

Stats: 21 source files, 52 test files, 1417 statements, 100% coverage
```

---

## Type System Architecture

### Core Types (No Pydantic, No Dataclasses)

All models use **TypedDict** for strict, structural typing:

```python
# api/models.py - Request/Response models
from typing_extensions import TypedDict
from typing import Literal, NotRequired

class JobCreate(TypedDict):
    """Job creation request (validated at API boundary)."""
    source: Literal["oscar", "wikipedia"]
    language: Literal["kk", "ky", "uz", "tr", "ug", "fi", "az"]
    script: Literal["Latn", "Cyrl", "Arab"] | None
    max_sentences: int
    transliterate: bool
    confidence_threshold: float

class JobStatus(TypedDict):
    """Job status response."""
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int
    message: str | None
    result_url: str | None
    file_id: str | None
    upload_status: Literal["uploaded"] | None
    created_at: datetime
    updated_at: datetime
    error: str | None
```

### UnknownJson Pattern (API Boundaries)

Type-safe handling of untyped JSON at system boundaries:

```python
# core/models.py
UnknownJson = (
    None
    | bool
    | int
    | float
    | str
    | list["UnknownJson"]
    | dict[str, "UnknownJson"]
)

# api/types.py
JsonDict = dict[str, str | int | float | bool | None | list[str | int | float | bool | None]]

# Usage: Parse untrusted input
def parse_job_create(payload: JsonDict) -> JobCreate:
    """Convert JsonDict → UnknownJson → validated JobCreate."""
    converted: dict[str, UnknownJson] = {}
    for k, v in payload.items():
        # Narrow to UnknownJson, then validate
        converted[k] = v if isinstance(v, (str, int, float, bool, type(None))) else list(v)
    return _decode_job_create_from_unknown(converted)
```

### Protocol-Based Dependency Injection

Type-safe interfaces without concrete dependencies:

```python
# api/types.py
from typing import Protocol
from collections.abc import Mapping

class LoggerProtocol(Protocol):
    """Minimal logger interface (compatible with platform_core.logging)."""
    def info(
        self,
        msg: str,
        *args: UnknownJson,
        extra: Mapping[str, UnknownJson] | None = None,
    ) -> None: ...
    # debug, warning, error...

class QueueProtocol(Protocol):
    """Minimal queue interface (compatible with RQ)."""
    def enqueue(
        self, func: str | _EnqCallable, *args: UnknownJson, **kwargs: UnknownJson
    ) -> JobLike: ...

# Usage: Inject protocols, not concrete types
class JobService:
    def __init__(
        self,
        *,
        redis: RedisStrProto,
        logger: LoggerProtocol,
        queue: QueueProtocol,
    ) -> None:
        self._redis = redis
        self._logger = logger
        self._queue = queue
```

---

## Platform Library Integration

### Centralized Logging (platform_core.logging)

**Single source of truth** - no stdlib `logging` imports:

```python
# All modules use platform_core logging
from platform_core.logging import get_logger, setup_logging

# API initialization (main.py)
def _init_logging() -> None:
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="turkic-api",
        instance_id=None,
        extra_fields=[],
    )

# Worker initialization (worker_entry.py)
def main() -> None:
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="turkic-worker",
        instance_id=None,
        extra_fields=[],
    )
    logger = get_logger(__name__)
    run_rq_worker(config)
```

**No legacy code:**
- ✅ Zero `import logging` from stdlib
- ✅ Zero `logging.basicConfig()` calls
- ✅ Zero local logger configuration
- ✅ All logging via `platform_core.logging.get_logger()`

### Event-Driven Architecture (platform_core.job_events + platform_workers.job_context)

Type-safe event publishing to Redis using the shared generic infrastructure:

```python
from platform_core.job_events import JobDomain, default_events_channel
from platform_workers.job_context import JobContext, make_job_context

TURKIC_DOMAIN: JobDomain = "turkic"

# jobs.py - Publishing events via generic context
def _make_ctx(redis: RedisStrProto, job_id: str, user_id: int, queue: str) -> JobContext:
    return make_job_context(
        redis=redis,
        domain=TURKIC_DOMAIN,
        events_channel=default_events_channel(TURKIC_DOMAIN),
        job_id=job_id,
        user_id=user_id,
        queue_name=queue,
    )

def publish_started(redis: RedisStrProto, job_id: str, user_id: int, queue: str) -> None:
    _make_ctx(redis, job_id, user_id, queue).publish_started()

def publish_completed(
    redis: RedisStrProto, job_id: str, user_id: int, queue: str, file_id: str, size: int
) -> None:
    ctx = _make_ctx(redis, job_id, user_id, queue)
    ctx.publish_completed(result_id=file_id, result_bytes=size)
```

### Configuration (platform_core.config)

Centralized settings from environment:

```python
# api/config.py
from platform_core.config import TurkicApiSettings as Settings
from platform_core.config import load_turkic_api_settings

def settings_from_env() -> Settings:
    """Load settings via shared platform_core config helpers."""
    return load_turkic_api_settings()

# Settings TypedDict (from platform_core)
# - redis_url: str
# - data_dir: str
# - environment: str
# - data_bank_api_url: str
# - data_bank_api_key: str
```

### Worker Harness (platform_workers.rq_harness)

RQ worker lifecycle management:

```python
# worker_entry.py
from platform_workers.rq_harness import WorkerConfig, run_rq_worker
from platform_core.job_events import default_events_channel
from platform_core.queues import TURKIC_QUEUE

def _build_config() -> WorkerConfig:
    settings = settings_from_env()
    return {
        "redis_url": settings["redis_url"],
        "queue_name": TURKIC_QUEUE,
        "events_channel": default_events_channel("turkic"),
    }

def main() -> None:
    setup_logging(...)
    run_rq_worker(_build_config())  # Blocks until worker stops
```

---

## API Endpoints

### POST /api/v1/jobs

Create a new corpus processing job.

**Request Body (JsonDict → JobCreate):**
```json
{
  "source": "oscar",
  "language": "kk",
  "script": null,
  "max_sentences": 1000,
  "transliterate": true,
  "confidence_threshold": 0.95
}
```

**Response (JobResponse):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Business Logic:**
1. Validate input via `parse_job_create()`
2. Generate UUID job_id
3. Save initial status to Redis (`TurkicJobStore`)
4. Enqueue job to RQ (`api.jobs._decode_process_corpus`)
5. Publish `StartedV1` event
6. Return job metadata

---

### GET /api/v1/jobs/{job_id}

Get job status and metadata.

**Response (JobStatus):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "message": "done",
  "result_url": "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/result",
  "file_id": "a3f8b2c1d4e5f6g7h8i9j0k1l2m3n4o5",
  "upload_status": "uploaded",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:32:15Z",
  "error": null
}
```

**States:**
- `queued` - In RQ queue, not started
- `processing` - Worker executing job
- `completed` - Success, result available via data-bank-api
- `failed` - Error occurred, see `error` field

---

### GET /api/v1/jobs/{job_id}/result

Stream result file from data-bank-api.

**Response:**
- **Status**: 200 OK (if completed)
- **Content-Type**: `text/plain; charset=utf-8`
- **Transfer-Encoding**: `chunked` (streamed from data-bank)
- **Body**: Processed sentences, one per line (UTF-8)

**Error Responses:**
- `404` - Job not found
- `425` - Job not completed yet (queued/processing)
- `410` - Job failed
- `500` - Data bank streaming error

**Implementation:**
```python
# Proxies to data-bank-api GET /files/{file_id} using DataBankClient
from platform_core.data_bank_client import DataBankClient

client = DataBankClient(settings["data_bank_api_url"], settings["data_bank_api_key"], timeout_seconds=120.0)
head = client.head(file_id, request_id=job_id)
resp = client.stream_download(file_id, request_id=job_id)
try:
    for chunk in resp.iter_bytes():
        yield chunk
finally:
    resp.close()
```

---

### GET /api/v1/health

Health check endpoint (used by Railway).

**Response (HealthResponse):**
```json
{
  "status": "healthy",
  "redis": true,
  "volume": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Logic:**
- `healthy` - Redis ping success AND volume accessible
- `degraded` - One of Redis/volume failing
- `unhealthy` - Both Redis and volume failing

---

## Job Processing Pipeline

### Worker Execution Flow

```
1. RQ worker picks job from TURKIC_QUEUE
   ↓
2. Calls: api.jobs._decode_process_corpus(job_id, params)
   ↓
3. Decode params: dict[str, UnknownJson] → JobParams
   ↓
4. Call: process_corpus_impl(job_id, params, redis, settings, logger)
   ↓
5. Steps:
   a. Update status: "processing" (TurkicJobStore)
   b. Publish: StartedV1 event
   c. Ensure corpus: ensure_corpus_file() - downloads/caches if needed
   d. Stream corpus: LocalCorpusService.stream(spec)
   e. Transliterate: to_ipa() if transliterate=true
   f. Upload result: DataBankClient.upload(...)
   g. Save metadata: file_id, size, sha256, content_type
   h. Update status: "completed"
   i. Publish: CompletedV1 event
   ↓
6. Return: JobResult (job_id, status, result_url)
```

### Error Handling (No try/except)

All errors propagate explicitly:

```python
# jobs.py - No fallback logic
def _upload_and_record(...) -> str:
    """Upload to data-bank-api; raises DataBankClientError on failure."""
    if url_cfg.strip() == "" or key_cfg.strip() == "":
        logger.error("data-bank configuration missing", extra={...})
        _fail_upload(redis, job_id, "config_missing", store, emit)
        raise DataBankClientError("data-bank configuration missing")

    client = DataBankClient(url_cfg, key_cfg, timeout_seconds=600.0)
    response = client.upload(
        file_id=f"{job_id}.txt", stream=buffer, content_type="text/plain; charset=utf-8", request_id=job_id
    )
    # Persist typed FileUploadResponse fields
    store.save_upload_metadata(job_id, response)
    return response["file_id"]
```

**No recovery mechanisms:**
- ❌ No retries
- ❌ No fallbacks
- ❌ No best-effort modes
- ✅ Fail fast, log clearly, propagate explicitly

---

## Data Bank Integration

### Upload Flow (Strict, No Fallback)

```python
# After job completes processing, BEFORE marking "completed":
1. POST {data_bank_api_url}/files
   Headers: X-API-Key, X-Request-ID
   Body: multipart/form-data with result file

2. Parse response: FileUploadResponse
   {
     "file_id": "sha256_hash",
     "size": 12345,
     "sha256": "abc123...",
     "content_type": "text/plain",
     "created_at": "2024-01-15T10:30:00Z"
   }

3. Save to Redis: job:{job_id}:upload
   - file_id, size, sha256, content_type, created_at

4. Update job status: "completed", file_id, upload_status="uploaded"

5. Publish: CompletedV1 event with file_id
```

**Error Cases (All raise UploadError):**
- Missing/invalid config (TURKIC_DATA_BANK_API_URL/KEY)
- Network failure
- Non-2xx status code
- Malformed JSON response
- Missing/invalid file_id in response

**No local fallback:**
- ❌ No local file serving via /result
- ❌ No "upload failed but job succeeded" state
- ✅ Upload failure = job failure

---

## Development Workflow

### Makefile Commands

```makefile
make check     # Full quality check (lint + test)
make lint      # Ruff + mypy + guard scripts
make test      # pytest with 100% coverage
```

**`make check` execution order:**
1. Clean stale venv if needed
2. Run guard scripts (code quality rules)
3. `poetry install --with dev`
4. `ruff check . --fix`
5. `ruff format .`
6. `mypy src tests scripts` (strict mode)
7. `poetry install --with dev` (again for test deps)
8. `pytest -n auto -v --cov-branch --cov=src --cov=scripts`

**Exit on failure:**
- Any step fails → entire `make check` fails
- 100% coverage required
- Zero type errors allowed
- Zero guard violations allowed

### Guard Script Enforcement

```python
# scripts/guard.py → monorepo_guards.orchestrator

Rules enforced:
- config: No hardcoded secrets
- typing: No Any, cast, type: ignore
- imports: No TYPE_CHECKING, no Pydantic
- patterns: Proper UnknownJson usage
- logging: No stdlib logging imports
- suppress: No type suppression comments
- exceptions: No try/except blocks
- env: Proper environment variable usage
- json: Correct JSON parsing patterns
- dataclass: No @dataclass (TypedDict only)
```

**Example violation:**
```python
# ❌ FAILS guard check
from typing import Any

def process(data: Any) -> dict:  # Any type
    return {}

# ✅ PASSES guard check
def process(data: UnknownJson) -> dict[str, str]:
    if not isinstance(data, dict):
        raise TypeError("Expected dict")
    return {k: str(v) for k, v in data.items()}
```

---

## Testing Strategy

### Coverage Requirements

**pyproject.toml:**
```toml
[tool.coverage.report]
fail_under = 100
branch = true
show_missing = true
```

**Current stats (all passing):**
- 503 tests
- 1417 statements
- 472 branches
- 100% statement coverage
- 100% branch coverage

### Test Categories

**API Tests (`test_api.py`, `test_main_*.py`):**
```python
def test_create_job_enqueues_and_returns_id(client: TestClient) -> None:
    """Test POST /api/v1/jobs returns valid job ID."""
    response = client.post("/api/v1/jobs", json={...})
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert UUID(data["job_id"])
```

**Job Processing Tests (`test_jobs_*.py`, `test_worker.py`):**
```python
def test_process_corpus_impl_creates_file_and_updates_status(tmp_path: Path) -> None:
    """Test full job processing pipeline with mocked dependencies."""
    redis = _RedisStub()
    settings = Settings(...)
    logger = logging.getLogger(__name__)

    result = process_corpus_impl("job1", params, redis=redis, settings=settings, logger=logger)

    assert result["status"] == "completed"
    assert redis.hashes[turkic_job_key("job1")]["status"] == "completed"
```

**Transliteration Tests (`test_*_ipa_letters.py`, `test_*_northwind.py`):**
```python
@pytest.mark.parametrize("letter,ipa", [("а", "ɑ"), ("б", "b"), ...])
def test_letter_to_ipa(letter: str, ipa: str) -> None:
    """Test individual letter transliteration."""
    assert to_ipa(letter, "kk") == ipa
```

**Edge Case Tests (`test_api_strict_coverage.py`):**
```python
def test_transliteval_edge_cases_and_rule_files(tmp_path: Path) -> None:
    """Test error paths and all production rule files."""
    # Test unclosed macro error (line 227)
    bad_path.write_text("$A = [abc\n", encoding="utf-8")
    with pytest.raises(te.RuleParseError, match="macro definition missing closing"):
        te.load_rules("unclosed_macro_test.rules")

    # Test partial truncation (lines 288-289)
    out2: list[str] = ["abc", "defgh"]
    te._truncate_output(out2, 2)
    assert out2 == ["abc", "def"]

    # Verify all production rules parse
    production_files: tuple[Path, ...] = tuple(
        p for p in rule_dir.glob("*.rules") if "_" in p.stem
    )
    for path in production_files:
        rules = te.load_rules(path.name)
        te.apply_rules("test", rules)
```

### Parallel Testing

**pytest-xdist configuration:**
```toml
[tool.pytest.ini_options]
addopts = "-v -n auto --dist loadscope"
timeout = 60
```

**Benefits:**
- `n auto` - Uses all CPU cores
- `loadscope` - Distributes by module
- Runs 503 tests in ~20 seconds (vs ~60 sequential)

**Parallel-safe patterns:**
- Tests create temp files in `tmp_path` (per-test isolation)
- Production rule file filter excludes test files by naming convention
- Redis stubs are per-test instances

---

## Deployment

### Multi-Stage Dockerfile

```dockerfile
# Stage 1: Builder - creates wheel with all deps
FROM python:3.11-slim-bookworm AS builder
RUN pip install "poetry==1.8.3"
COPY pyproject.toml poetry.lock ./services/turkic-api/
COPY src ./services/turkic-api/src
COPY libs ./libs
RUN poetry build -f wheel

# Stage 2: Runtime base - installs wheel once
FROM python:3.11-slim-bookworm AS runtime-base
COPY --from=builder /workspace/services/turkic-api/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm -rf /tmp/*.whl

# Stage 3a: API target (default)
FROM runtime-base AS api
EXPOSE 8000
CMD ["sh", "-c", "exec hypercorn 'turkic_api.api.main:create_app' --bind [::]:${PORT:-8000}"]

# Stage 3b: Worker target
FROM runtime-base AS worker
CMD ["sh", "-c", "exec python -m turkic_api.worker_entry"]
```

**Benefits:**
- Single build produces both API and worker images
- Wheel caching speeds up rebuilds
- No dev dependencies in runtime image
- Platform libraries (platform_core, platform_workers) bundled in wheel

### Railway Configuration

**railway.toml (API service):**
```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "sh -c \"exec hypercorn 'turkic_api.api.main:create_app' --bind [::]:${PORT:-8000}\""
healthcheckPath = "/api/v1/health"
healthcheckTimeout = 30
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
```

**Worker service (separate Railway service):**
```bash
# Docker build target: worker
# Start command: python -m turkic_api.worker_entry
```

**Environment Variables:**
```bash
# Auto-provided by Railway
PORT=8000
RAILWAY_ENVIRONMENT=production

# Configured in Railway dashboard
TURKIC_REDIS_URL=redis://default:xxx@redis.railway.internal:6379
TURKIC_DATA_DIR=/data
TURKIC_DATA_BANK_API_URL=https://data-bank-api.railway.app
TURKIC_DATA_BANK_API_KEY=<secret>

# Alternative: Use gateway URL (auto-appends /data-bank)
# API_GATEWAY_URL=https://gateway.railway.app
```

---

## Supported Languages

### Current Support

| Language     | Code | IPA | Latin | Notes                |
|-------------|------|-----|-------|----------------------|
| Kazakh      | kk   | ✅  | ✅    | Cyrillic & Latin     |
| Kyrgyz      | ky   | ✅  | ✅    | Cyrillic & Latin     |
| Uzbek       | uz   | ✅  | ✅    | Cyrillic & Latin     |
| Turkish     | tr   | ✅  | ✅    | Latin only           |
| Uyghur      | ug   | ✅  | ❌    | Arabic script        |
| Finnish     | fi   | ✅  | ❌    | Latin only           |
| Azerbaijani | az   | ✅  | ❌    | Latin only           |

### Adding New Languages

**Steps:**
1. Add rules file to `src/turkic_api/core/rules/{lang}_ipa.rules`
2. Follow ICU transliterator syntax (see existing files)
3. Add test file `tests/test_{lang}_ipa_letters.py`
4. Language automatically detected by `get_supported_languages()`

**No code changes needed** - rule files are discovered at runtime via `glob("*.rules")`.

---

## Security

### Input Validation

All inputs validated at API boundary:

```python
# models.py
def parse_job_create(payload: JsonDict) -> JobCreate:
    """Parse and validate JobCreate from request body."""
    # 1. Convert JsonDict → UnknownJson
    # 2. Validate each field with runtime checks
    # 3. Narrow to Literal types
    # 4. Return typed JobCreate
```

**Validation patterns:**
- Literal types for enums (source, language, script, status)
- Range checks for integers (`max_sentences: 1-100000`)
- Range checks for floats (`confidence_threshold: 0.0-1.0`)
- No arbitrary strings (UUID job_id only)

### No SQL/Command Injection

- **No SQL database** - Redis KV store only
- **No shell commands** - Python subprocess not used
- **No path traversal** - Job IDs are UUIDs, not filenames
- **No code execution** - No eval/exec

### API Security

- **CORS**: Configured in `create_app()` (allowed origins only)
- **Rate limiting**: Future enhancement (not yet implemented)
- **Authentication**: Not required (public API)
- **HTTPS**: Enforced by Railway platform

---

## Monitoring & Observability

### Structured Logging (JSON)

All logs output as JSON via `platform_core.logging`:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "turkic-api",
  "logger": "turkic_api.api.services",
  "message": "Job created",
  "extra": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "language": "kk",
    "source": "oscar"
  }
}
```

**Benefits:**
- Easily parseable by log aggregators
- Structured extra fields for filtering
- Consistent format across API and worker

### Health Checks

Railway pings `/api/v1/health` every 30 seconds:

```python
async def health_endpoint(...) -> HealthResponse:
    redis_ok = await redis.ping()
    volume_ok = Path(settings["data_dir"]).is_dir()

    status: HealthStatus
    if redis_ok and volume_ok:
        status = "healthy"
    elif redis_ok or volume_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return {...}
```

**Railway actions:**
- `healthy` - No action
- `degraded` - Log warning
- `unhealthy` - Restart service (3 retries)

### Event Stream

All job lifecycle events published to Redis via the generic job channel:

```
channel: default_events_channel("turkic")  # turkic:events
  - turkic.job.started.v1
  - turkic.job.progress.v1 (every 50 sentences)
  - turkic.job.completed.v1
  - turkic.job.failed.v1
```

Payloads use the generic `platform_core.job_events` shape with a `domain` field, ErrorKind-tagged failures, and `result_id`/`result_bytes` on completion.

External services can subscribe to track job progress in real-time.

---

## Performance Characteristics

### Throughput

- **API**: Handles 100+ req/sec (FastAPI async)
- **Worker**: 1000 sentences/sec transliteration
- **Streaming**: 64KB chunks from data-bank

### Latency

- **POST /jobs**: <50ms (enqueue only)
- **GET /jobs/{id}**: <10ms (Redis lookup)
- **GET /jobs/{id}/result**: <100ms first byte (data-bank stream)

### Resource Usage

- **API**: 50MB RAM, <1% CPU (idle)
- **Worker**: 200MB RAM, 20% CPU (active job)
- **Corpus cache**: ~500MB per language on disk

---

## Future Enhancements

**Not planned for v1.0:**
- Rate limiting (slowapi)
- Authentication (JWT)
- Metrics (Prometheus)
- Caching (Redis cache for common corpora)
- Web UI (separate project)

**Quality bar remains:**
- 100% test coverage required for all changes
- Zero type errors allowed
- Guard scripts must pass
- No technical debt introduced

---

## License

Apache-2.0
