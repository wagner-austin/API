# Job and Event System Refactor

## Document Status

| Field | Value |
|-------|-------|
| **Author** | Platform Engineering |
| **Created** | 2025-11-28 |
| **Updated** | 2025-11-30 |
| **Status** | COMPLETE (all 8 services migrated) |
| **Scope** | platform_core, platform_workers, clients/DiscordBot, and 8 services |

---

## Migration Progress

| Phase | Component | Status | Notes |
|-------|-----------|--------|-------|
| 1 | Generic Infrastructure | COMPLETE | `job_events.py`, `job_context.py`, `testing.py`, RQ harness cleanup |
| 2 | Event Consumers | COMPLETE | Discord runtime + DiscordBot decoding generic job events |
| 3 | turkic-api | COMPLETE | Generic job_events/job_context with `result_id`/`result_bytes` payload |
| 4 | transcript-api | COMPLETE | Sync + async use job_events/job_context/BaseJobStore (`result_id`/`result_bytes`) |
| 5 | data-bank-api | COMPLETE | Worker uses job_events channel; no lifecycle publishers today |
| 6 | qr-api | COMPLETE | Worker uses job_events channel; no lifecycle publishers today |
| 7 | Model-Trainer | COMPLETE | Lifecycle via job_events; metrics remain domain-specific (trainer.metrics.*) |
| 8 | handwriting-ai | COMPLETE | Lifecycle via job_events; metrics remain domain-specific (digits.metrics.*) |
| 9 | covenant-radar-api | COMPLETE | Worker uses job_events channel (domain: covenant) |
| 10 | music-wrapped-api | COMPLETE | Worker uses job_events channel (domain: music_wrapped) |

---

## Executive Summary

This document proposes a comprehensive refactor to consolidate the fragmented job and event system across the monorepo. Currently, each service implements its own:

1. Event types (`*_events.py` files in platform_core)
2. Job store (Redis-backed status tracking)
3. Job context (event publishing helpers)
4. Worker entry points

The refactor will:

1. **Consolidate event types** into a single generic `platform_core/job_events.py`
2. **Create reusable JobStore** in `platform_workers`
3. **Parameterize job context** by domain instead of duplicating code
4. **Standardize worker patterns** across all services
5. **Achieve 100% type coverage** with no `Any`, `cast`, or `type: ignore`

---

## Current State Analysis

### Event Files in platform_core

| File | Domain | Event Types | Pattern |
|------|--------|-------------|---------|
| `job_events.py` | generic | StartedV1, ProgressV1, CompletedV1, FailedV1 | `{domain}.job.*.v1` |
| `transcript_events.py` | transcript | CompletedV1, FailedV1 (legacy sync API; deprecated) | `transcript.*.v1` |
| `trainer_events.py` | trainer | TrainStartedV1, TrainProgressV1, etc. | `trainer.*.v1` |
| `digits_events.py` | digits | StartedV1, BatchV1, EpochV1, CompletedV1, etc. | `digits.train.*.v1` |
| `data_bank_events.py` | databank | (upload/download events) | `databank.*.v1` |

**Problem**: trainer/digits still carry lifecycle variants; transcript sync events remain domain-specific; lifecycle job events should converge on `job_events.py`.

### Job Store Implementations

| Service | File | Pattern |
|---------|------|---------|
| turkic-api | `api/job_store.py` | TurkicJobStore, TurkicJobStatus TypedDict |
| transcript-api | `job_store.py` | TranscriptJobStore, TranscriptJobStatus TypedDict |
| Model-Trainer | (uses trainer_events) | RunStatus in runs store |
| handwriting-ai | (inline in digits.py) | No dedicated store |
| data-bank-api | (none) | No lifecycle store (no lifecycle events published) |
| qr-api | (none) | No lifecycle store (no lifecycle events published) |

**Problem**: Model-Trainer and handwriting-ai still own bespoke lifecycle storage/decoders; data-bank-api/qr-api lack shared lifecycle storage if/when they emit events.

### Job Context Implementations

**Location**: `platform_workers/job_context.py`

- Generic `make_job_context` publishes Started/Progress/Completed/Failed via `job_events.py` for any domain.
- No domain-specific contexts remain in `rq_harness.py` (cleaned).

### Service Job Processing

| Service | Entry Point | Events | Store |
|---------|-------------|--------|-------|
| turkic-api | `api/jobs.py:process_corpus_job` | job_events | TurkicJobStore |
| transcript-api | `jobs.py:process_stt` | job_events (async + sync endpoints) | TranscriptJobStore |
| Model-Trainer | `worker/*.py` | trainer_events | (inline) |
| handwriting-ai | `jobs/digits.py:process_train_job` | digits_events | (inline) |
| data-bank-api | `worker_entry.py` | job_events channel only (no lifecycle publishers) | (none) |
| qr-api | `worker_entry.py` | job_events channel only (no lifecycle publishers) | (none) |

---

## Problem Statement

### Core Issues

1. **Code Duplication**: Trainer/digits lifecycle events still mirror the generic lifecycle with domain naming.
2. **Inconsistent Naming**: Lifecycle fields differ from `job_events.py` naming and payload shape.
3. **Type Drift**: Trainer/digits lifecycle payloads omit generic fields (domain/result metadata) and diverge over time.
4. **Maintenance Burden**: Lifecycle changes require touching multiple domain files instead of `job_events.py`.
5. **No Generic Store**: Model-Trainer/handwriting-ai keep bespoke lifecycle storage/decoders.

### Technical Debt Risks

1. **Drift**: New services may copy trainer/digits patterns instead of generic job events.
2. **Bugs**: Inconsistent serialization logic leads to subtle parsing errors.
3. **Testing**: Domain-specific lifecycle code requires its own coverage instead of reusing generic tests.
4. **Onboarding**: Multiple event patterns (job_events + trainer/digits + transcript sync) increase cognitive load.

---

## Proposed Solution: Generic Job Infrastructure

### Design Principles

1. **Strict Typing**: All types use `TypedDict`/`Protocol` - no `Any`
2. **Parameterized by Domain**: Single implementation with domain string parameter
3. **Fail-Fast**: No fallbacks, best-effort, or try/except recovery
4. **Parse at Boundary**: Validation happens at decode time
5. **100% Test Coverage**: Branch coverage required for all code
6. **No External Dependencies**: Pure Python, no third-party job libraries

### Architecture Overview

```
platform_core/
  job_events.py          # Generic job event types and codecs
  job_types.py           # Shared TypedDicts for job status

platform_workers/
  job_store.py           # Generic Redis-backed job store
  job_context.py         # Generic job context factory
  rq_harness.py          # RQ integration (uses job_context)

services/*/
  jobs.py                # Service-specific job logic (minimal)
  worker_entry.py        # Worker entry point
```

---

## Detailed Design

### 1. Generic Job Events (platform_core/job_events.py)

```python
from __future__ import annotations

from typing import Final, Literal, NotRequired, TypedDict

from .json_utils import JSONValue, dump_json_str, load_json_str

# Domain type - extensible via Literal union
JobDomain = Literal["turkic", "transcript", "trainer", "digits", "databank", "qr"]

# Event type suffixes
EventSuffix = Literal["started", "progress", "completed", "failed"]

ErrorKind = Literal["user", "system"]


class JobStartedV1(TypedDict):
    """Generic job started event."""
    type: str  # "{domain}.job.started.v1"
    domain: JobDomain
    job_id: str
    user_id: int
    queue: str


class JobProgressV1(TypedDict):
    """Generic job progress event."""
    type: str  # "{domain}.job.progress.v1"
    domain: JobDomain
    job_id: str
    user_id: int
    progress: int
    message: NotRequired[str]
    # Optional domain hint; not for metrics
    payload: NotRequired[JSONValue]


class JobCompletedV1(TypedDict):
    """Generic job completed event."""
    type: str  # "{domain}.job.completed.v1"
    domain: JobDomain
    job_id: str
    user_id: int
    result_id: str  # file_id, video_id, model_id, etc.
    result_bytes: int


class JobFailedV1(TypedDict):
    """Generic job failed event."""
    type: str  # "{domain}.job.failed.v1"
    domain: JobDomain
    job_id: str
    user_id: int
    error_kind: ErrorKind
    message: str


JobEventV1 = JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1


def make_event_type(domain: JobDomain, suffix: EventSuffix) -> str:
    """Construct event type string."""
    return f"{domain}.job.{suffix}.v1"


def encode_job_event(event: JobEventV1) -> str:
    """Encode job event to JSON string."""
    return dump_json_str(event)


def make_started_event(
    *,
    domain: JobDomain,
    job_id: str,
    user_id: int,
    queue: str,
) -> JobStartedV1:
    """Factory for started events."""
    return {
        "type": make_event_type(domain, "started"),
        "domain": domain,
        "job_id": job_id,
        "user_id": user_id,
        "queue": queue,
    }


def make_progress_event(
    *,
    domain: JobDomain,
    job_id: str,
    user_id: int,
    progress: int,
    message: str | None = None,
    payload: JSONValue | None = None,
) -> JobProgressV1:
    """Factory for progress events."""
    ev: JobProgressV1 = {
        "type": make_event_type(domain, "progress"),
        "domain": domain,
        "job_id": job_id,
        "user_id": user_id,
        "progress": progress,
    }
    if message is not None:
        ev["message"] = message
    if payload is not None:
        ev["payload"] = payload
    return ev


def make_completed_event(
    *,
    domain: JobDomain,
    job_id: str,
    user_id: int,
    result_id: str,
    result_bytes: int,
) -> JobCompletedV1:
    """Factory for completed events."""
    return {
        "type": make_event_type(domain, "completed"),
        "domain": domain,
        "job_id": job_id,
        "user_id": user_id,
        "result_id": result_id,
        "result_bytes": result_bytes,
    }


def make_failed_event(
    *,
    domain: JobDomain,
    job_id: str,
    user_id: int,
    error_kind: ErrorKind,
    message: str,
) -> JobFailedV1:
    """Factory for failed events."""
    return {
        "type": make_event_type(domain, "failed"),
        "domain": domain,
        "job_id": job_id,
        "user_id": user_id,
        "error_kind": error_kind,
        "message": message,
    }


# Default channel factory
def default_events_channel(domain: JobDomain) -> str:
    """Get default Redis pub/sub channel for domain."""
    return f"{domain}:events"
```

### 2. Generic Job Types (platform_core/job_types.py)

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict

JobStatusLiteral = Literal["queued", "processing", "completed", "failed"]


class BaseJobStatus(TypedDict):
    """Base fields for all job status types."""
    job_id: str
    user_id: int
    status: JobStatusLiteral
    progress: int
    message: str | None
    created_at: datetime
    updated_at: datetime
    error: str | None


def job_key(domain: str, job_id: str) -> str:
    """Generate Redis key for job status."""
    return f"{domain}:job:{job_id}"
```

### 3. Generic Job Store (platform_workers/job_store.py)

```python
from __future__ import annotations

from datetime import datetime
from typing import Protocol, TypeVar

from platform_core.job_types import BaseJobStatus, JobStatusLiteral, job_key


class RedisHashProto(Protocol):
    """Minimal Redis interface for hash operations."""
    def hset(self, key: str, mapping: dict[str, str]) -> int: ...
    def hgetall(self, key: str) -> dict[str, str]: ...


T = TypeVar("T", bound=BaseJobStatus)


class JobStoreEncoder(Protocol[T]):
    """Protocol for encoding job status to Redis hash."""
    def encode(self, status: T) -> dict[str, str]: ...
    def decode(self, job_id: str, raw: dict[str, str]) -> T: ...


class BaseJobStore:
    """Generic Redis-backed job store.

    Subclasses provide domain-specific encoder via Protocol.
    """

    def __init__(
        self,
        redis: RedisHashProto,
        domain: str,
        encoder: JobStoreEncoder[BaseJobStatus],
    ) -> None:
        self._redis = redis
        self._domain = domain
        self._encoder = encoder

    def save(self, status: BaseJobStatus) -> None:
        key = job_key(self._domain, status["job_id"])
        self._redis.hset(key, self._encoder.encode(status))

    def load(self, job_id: str) -> BaseJobStatus | None:
        key = job_key(self._domain, job_id)
        raw = self._redis.hgetall(key)
        if not raw:
            return None
        return self._encoder.decode(job_id, raw)


# Shared parsing utilities
def parse_status(raw: dict[str, str]) -> JobStatusLiteral:
    """Parse status field from Redis hash."""
    status_raw = raw.get("status")
    if status_raw is None:
        raise ValueError("missing status in redis store")
    if status_raw == "queued":
        return "queued"
    if status_raw == "processing":
        return "processing"
    if status_raw == "completed":
        return "completed"
    if status_raw == "failed":
        return "failed"
    raise ValueError("invalid status in redis store")


def parse_int_field(raw: dict[str, str], key: str) -> int:
    """Parse integer field from Redis hash."""
    value = raw.get(key)
    if value is None:
        raise ValueError(f"missing {key} in redis store")
    stripped = value.strip()
    if stripped == "" or not stripped.lstrip("-").isdigit():
        raise ValueError(f"invalid {key} in redis store")
    return int(stripped)


def parse_datetime_field(raw: dict[str, str], key: str) -> datetime:
    """Parse datetime field from Redis hash."""
    value = raw.get(key)
    if value is None or value.strip() == "":
        raise ValueError(f"missing {key} in redis store")
    return datetime.fromisoformat(value)


def parse_optional_str(raw: dict[str, str], key: str) -> str | None:
    """Parse optional string field from Redis hash."""
    val = raw.get(key)
    if val is None:
        return None
    stripped = val.strip()
    return stripped if stripped != "" else None
```

### 4. Generic Job Context (platform_workers/job_context.py)

```python
from __future__ import annotations

from typing import Protocol

from platform_core.job_events import (
    JobDomain,
    JobEventV1,
    encode_job_event,
    make_completed_event,
    make_failed_event,
    make_progress_event,
    make_started_event,
)


class RedisPublishProto(Protocol):
    """Minimal Redis interface for pub/sub."""
    def publish(self, channel: str, message: str) -> int: ...


class JobContext(Protocol):
    """Protocol for job event publishing."""
    def publish_started(self) -> None: ...
    def publish_progress(self, progress: int, message: str | None) -> None: ...
    def publish_completed(self, result_id: str, result_bytes: int) -> None: ...
    def publish_failed(self, error_kind: str, message: str) -> None: ...


def make_job_context(
    *,
    redis: RedisPublishProto,
    domain: JobDomain,
    events_channel: str,
    job_id: str,
    user_id: int,
    queue_name: str,
) -> JobContext:
    """Create a job context for publishing events.

    Args:
        redis: Redis client with publish capability.
        domain: Job domain (e.g., "turkic", "transcript").
        events_channel: Redis pub/sub channel for events.
        job_id: Unique job identifier.
        user_id: User who owns this job.
        queue_name: Name of the RQ queue.

    Returns:
        JobContext that publishes domain-specific events.
    """

    class _Ctx:
        def publish_started(self) -> None:
            ev = make_started_event(
                domain=domain,
                job_id=job_id,
                user_id=user_id,
                queue=queue_name,
            )
            redis.publish(events_channel, encode_job_event(ev))

        def publish_progress(self, progress: int, message: str | None) -> None:
            ev = make_progress_event(
                domain=domain,
                job_id=job_id,
                user_id=user_id,
                progress=progress,
                message=message,
            )
            redis.publish(events_channel, encode_job_event(ev))

        def publish_completed(self, result_id: str, result_bytes: int) -> None:
            ev = make_completed_event(
                domain=domain,
                job_id=job_id,
                user_id=user_id,
                result_id=result_id,
                result_bytes=result_bytes,
            )
            redis.publish(events_channel, encode_job_event(ev))

        def publish_failed(self, error_kind: str, message: str) -> None:
            kind = "user" if error_kind == "user" else "system"
            ev = make_failed_event(
                domain=domain,
                job_id=job_id,
                user_id=user_id,
                error_kind=kind,
                message=message,
            )
            redis.publish(events_channel, encode_job_event(ev))

    return _Ctx()
```

---

## Migration Plan

### Phase 1: Create Generic Infrastructure

**Deliverables:**

1. `platform_core/job_events.py` - Generic event types and factories ✅
2. `platform_core/job_types.py` - Shared job status types ✅
3. `platform_workers/job_store.py` - Generic Redis store ✅
4. `platform_workers/job_context.py` - Generic context factory ✅
5. Tests with 100% coverage for all new code ✅

**Validation (ran):**

```bash
cd libs/platform_core && make check
cd libs/platform_workers && make check
```

**Validation:**

```bash
cd libs/platform_core && make check
cd libs/platform_workers && make check
```

### Phase 2: Migrate turkic-api (Reference Implementation) ✅ COMPLETE

**Changes Made:**

1. Updated `api/jobs.py` to use `platform_workers.job_context.make_job_context` with `domain="turkic"`
2. Updated `worker_entry.py` to use `default_events_channel("turkic")`
3. Replaced all local `_RedisStub`, `_QueueStub` in tests with `FakeRedis`, `FakeQueue` from `platform_workers.testing`
4. Updated `publish_completed` calls to include `result_bytes` parameter

**Validation (passed):**

```bash
cd services/turkic-api && make check
# 514 tests passed, 100% coverage
```

### Phase 3: Migrate transcript-api (COMPLETE) ✅ COMPLETE

**Changes Made:**

1. Updated `jobs.py` to use generic `make_job_context` with `domain="transcript"`
2. Updated `worker_entry.py` to use `default_events_channel("transcript")`
3. Replaced all local test stubs with centralized `FakeRedis`, `FakeQueue`, `FakeLogger`
4. Updated route tests to use provider pattern with injected fakes

**Validation (passed):**

```bash
cd services/transcript-api && make check
# 254 tests passed, 100% coverage
```

### Phase 4: Migrate Model-Trainer ✅ COMPLETE

**Changes Made:**

1. Uses `platform_core.job_events` for lifecycle (started/progress/completed/failed) with domain `trainer`.
2. Keeps rich trainer metrics as domain-specific events published alongside lifecycle.
3. Worker entry uses `default_events_channel("trainer")`.

### Phase 5: Migrate handwriting-ai ✅ COMPLETE

**Changes Made:**

1. Publishes lifecycle via `platform_core.job_events` + `platform_workers.job_context`.
2. Worker entry uses `default_events_channel("digits")`.
3. Keeps specialized digits metrics events (BatchV1, EpochV1, BestV1, etc.) alongside lifecycle.

### Phase 8: Migrate covenant-radar-api ✅ COMPLETE

**Changes Made:**

1. Worker entry uses `default_events_channel("covenant")`.

### Phase 9: Migrate music-wrapped-api ✅ COMPLETE

**Changes Made:**

1. Worker entry uses `default_events_channel("music_wrapped")`.

### Phase 6: Migrate data-bank-api ✅ COMPLETE

**Changes Made:**

1. Updated `worker_entry.py` to use `default_events_channel("databank")`
2. Replaced local `_FakeRedis` stubs in tests with `FakeRedis` from `platform_workers.testing`
3. Updated worker entry test to use new channel format

**Validation (passed):**

```bash
cd services/data-bank-api && make check
# 90 tests passed, 100% coverage
```

### Phase 6b: Migrate qr-api ✅ COMPLETE

**Changes Made:**

1. Worker entry already using `default_events_channel("qr")` (was ahead of the curve)
2. Replaced local `_FakeRedis` stubs in tests with `FakeRedis` from `platform_workers.testing`
3. Used `sadd("rq:workers", ...)` to control worker count for readyz tests

**Validation (passed):**

```bash
cd services/qr-api && make check
# 32 tests passed, 100% coverage
```

### Phase 7: Cleanup

**Deliverables:**

1. Remove deprecated domain-specific event files
2. Remove duplicate code in services
3. Update documentation
4. Final validation across all repos

---

## Affected Repositories

### Libraries (2)

| Library | Changes |
|---------|---------|
| `platform_core` | Add `job_events.py`, `job_types.py`; deprecate domain-specific lifecycle files; retain strictly typed metrics modules |
| `platform_workers` | Add `job_store.py`, `job_context.py`; update `rq_harness.py` |

### Services (8)

| Service | Status | Domain | Files Changed |
|---------|--------|--------|---------------|
| `turkic-api` | ✅ COMPLETE | `turkic` | `api/jobs.py`, `worker_entry.py`, 8 test files |
| `transcript-api` | ✅ COMPLETE | `transcript` | `jobs.py`, `worker_entry.py`, `routes/jobs.py`, 12 test files |
| `data-bank-api` | ✅ COMPLETE | `databank` | `worker_entry.py`, 6 test files |
| `qr-api` | ✅ COMPLETE | `qr` | 2 test files (worker already used generic) |
| `Model-Trainer` | ✅ COMPLETE | `trainer` | `worker/*.py`, orchestrators |
| `handwriting-ai` | ✅ COMPLETE | `digits` | `jobs/digits.py`, `worker_entry.py` |
| `covenant-radar-api` | ✅ COMPLETE | `covenant` | `worker_entry.py` |
| `music-wrapped-api` | ✅ COMPLETE | `music_wrapped` | `worker_entry.py` |

---

## Metrics Publishing Standard

1. Lifecycle events (started, progress, completed, failed) are standardized via `platform_core.job_events` and published to `{domain}:events` with strict `TypedDict` contracts and encode/decode functions.
2. Domain metrics remain strictly typed and domain-specific (e.g., `trainer.metrics.*`, `digits.metrics.*`) and must be published as top-level events on the same domain channel. Do not embed metrics inside lifecycle payloads.
3. `JobProgressV1.payload` is reserved for lightweight, optional domain hints; it is not a transport for metrics.
4. Consumers decode metrics events first, then lifecycle as needed. Trainer code follows this; digits consumers adopt the same pattern.

Rationale: Preserves strict typing per domain, avoids drift, keeps contracts explicit, and allows 100% coverage on well-defined event unions.

## Key Migration Patterns

### Using Centralized Test Stubs

Replace local `_FakeRedis`, `_RedisStub`, `_QueueStub` with centralized stubs:

```python
# Before (local stubs)
class _FakeRedis:
    def ping(self, **kwargs) -> bool: ...
    def scard(self, key: str) -> int: ...
    # ... 40+ lines of boilerplate

# After (centralized)
from platform_workers.testing import FakeRedis, FakeQueue, FakeLogger

r = FakeRedis()
q = FakeQueue()
log = FakeLogger()
```

### Controlling Worker Count for /readyz Tests

Use `sadd` to simulate workers:

```python
def _fake_redis_provider(url: str) -> FakeRedis:
    r = FakeRedis()
    # Simulate 1 worker for "ready" state
    r.sadd("rq:workers", "worker-1")
    return r

# For "degraded" state (no workers), don't add any workers
def _fake_redis_no_workers(url: str) -> FakeRedis:
    return FakeRedis()  # scard("rq:workers") returns 0
```

### Using Generic Job Context

Publish lifecycle events with the generic factory:

```python
from platform_core.job_events import default_events_channel
from platform_workers.job_context import make_job_context

ctx = make_job_context(
    redis=redis,
    domain="turkic",
    events_channel=default_events_channel("turkic"),
    job_id=job_id,
    user_id=user_id,
    queue_name=queue,
)
ctx.publish_started()
ctx.publish_progress(progress=50, message="halfway")
ctx.publish_completed(result_id=file_id, result_bytes=file_size)
ctx.publish_failed(error_kind="user", message="bad input")
```

### Worker Entry Pattern

```python
from platform_core.job_events import default_events_channel
from platform_core.queues import DOMAIN_QUEUE
from platform_workers.rq_harness import WorkerConfig, run_rq_worker

def _build_config() -> WorkerConfig:
    return {
        "redis_url": settings["redis_url"],
        "queue_name": DOMAIN_QUEUE,
        "events_channel": default_events_channel("domain"),  # generic
    }
```

---

## Code Standards Compliance

### Typing Rules

- All types explicit; no `Any` (explicit or implicit) anywhere.
- No `cast()` usage.
- No `type: ignore` comments.
- No `.pyi` stub files or untyped shims.
- `TypedDict` for all structured data.
- `Protocol` for all interfaces (minimal, purpose-built).
- Dynamic imports: use `__import__()` and assign `getattr` results directly to variables annotated with the Protocol to prevent `Any` leakage.

### Error Handling

- Core libraries validate and raise; no try/except for recovery or “best‑effort”.
- All failure modes documented in docstrings.
- Tests assert error branches explicitly.

### Testing

- 100% statement coverage
- 100% branch coverage
- No `noqa` comments
- No coverage exclusions

### Import Patterns

- `__import__()` for dynamic module loading
- `getattr()` with Protocol type at assignment (never leave untyped)
- JSON parsing via `platform_core.json_utils` then strict `_decode*` validation
- No Pydantic, no dataclasses in src/

---

## Backward Compatibility

### During Migration

No back-compat in core libraries (no re-exports/aliases). Services migrate in lockstep. Existing Redis key patterns and event type strings remain unchanged to preserve external contracts.

### Post-Migration

1. Remove legacy modules and any transitional aliases.
2. No changes to external API contracts (event types/keys preserved).
3. Event consumers see same JSON for lifecycle; metrics remain domain-specific.

---

## Testing Requirements

### Unit Tests

1. `job_events.py`: Event factory functions, encoding/decoding
2. `job_types.py`: Key generation, status literals
3. `job_store.py`: Encode/decode roundtrip, parse helpers
4. `job_context.py`: Event publishing, error propagation

### Integration Tests

1. Redis store save/load roundtrip
2. Event publish/subscribe flow
3. Full job lifecycle (started → progress → completed/failed)

### Coverage Requirements

- **Statements**: 100%
- **Branches**: 100%
- **No coverage exclusions**

---

## Success Metrics

1. **Line count reduction**: ~60% fewer lines in event/job code
2. **File count reduction**: 6 event files → 2 generic files
3. **100% test coverage** for all job infrastructure
4. **Zero type errors** with strict mypy
5. **All services pass** `make check` after migration

---

## Rollback Plan

If issues discovered:

1. **Immediate**: Revert to domain-specific implementations (code still exists during migration)
2. **Short-term**: Fix issues in generic code, redeploy
3. **Investigation**: Compare event output between old and new implementations

---

## Appendix A: Domain-Specific Event Modules (Metrics Only)

- `platform_core.trainer_metrics_events`: training metrics for Model‑Trainer (config/progress/completed).
- `platform_core.digits_metrics_events`: training metrics for Handwriting‑AI (config/batch/epoch/best/artifact/upload/completed).
- `platform_core.data_bank_events`: legacy module with inconsistent domain naming; deprecated and scheduled for removal. Lifecycle is standardized via `job_events`.

---

## Appendix B: Redis Key Patterns

| Domain | Current Pattern | Proposed Pattern |
|--------|-----------------|------------------|
| turkic | `turkic:job:{job_id}` | `turkic:job:{job_id}` (unchanged) |
| transcript | `transcript:job:{job_id}` | `transcript:job:{job_id}` (unchanged) |
| trainer | `trainer:run:{run_id}` | `trainer:job:{job_id}` (standardize) |
| digits | (none) | `digits:job:{job_id}` (new) |
| databank | (none) | `databank:job:{job_id}` (new) |
| qr | (none) | `qr:job:{job_id}` (aligned with generic) |

---

## Appendix C: Event Channel Patterns

| Domain | Current Channel | Proposed Channel |
|--------|-----------------|------------------|
| turkic | `turkic:events` | `turkic:events` (unchanged) |
| transcript | `transcript:events` | `transcript:events` (unchanged) |
| trainer | `trainer:events` | `trainer:events` (unchanged) |
| digits | `digits:events` | `digits:events` (unchanged) |
| databank | `databank:events` | `databank:events` (unchanged) |
| qr | `qr:events` | `qr:events` (unchanged) |

---

## Appendix D: Domain-Specific Extensions

Some domains have additional event types beyond the core four (started, progress, completed, failed). These will remain domain-specific:

### digits_events.py (Keep)

- `BatchV1` - Per-batch training metrics
- `EpochV1` - Per-epoch training metrics
- `BestV1` - New best validation accuracy
- `ArtifactV1` - Model artifact saved
- `UploadV1` - Artifact uploaded

### trainer_events.py (Keep)

- Evaluation events
- Tokenizer training events
- Checkpoint events

### Approach

Generic job events handle the common lifecycle (started → progress → completed/failed). Domain-specific events remain in their own files for specialized use cases.
