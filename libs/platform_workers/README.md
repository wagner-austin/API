# platform-workers

Typed Redis helpers and RQ (Redis Queue) harness for background job processing.

## Installation

```bash
poetry add platform-workers
```

Requires `platform-core` and optionally `redis`, `rq` for runtime.

## Quick Start

```python
from platform_workers.redis import redis_for_kv
from platform_workers.rq_harness import rq_queue, rq_retry
from platform_workers.health import readyz_redis

# Create Redis client
redis = redis_for_kv("redis://localhost:6379")

# Enqueue a job
queue = rq_queue("my-queue", redis_raw_for_rq("redis://localhost:6379"))
job = queue.enqueue("mymodule.task", {"input": "data"})

# Health check
response = readyz_redis(redis)
```

## Redis Clients

Typed Redis client factories that avoid module-level imports:

```python
from platform_workers.redis import (
    redis_for_kv,      # String key-value operations
    redis_for_rq,      # Binary client for RQ
    redis_raw_for_rq,  # Raw client for RQ (full API)
    redis_for_pubsub,  # Async client for pub/sub
)

# String operations
redis = redis_for_kv("redis://localhost:6379")
redis.set("key", "value")
value = redis.get("key")
redis.hset("hash", {"field": "value"})
redis.publish("channel", "message")

# For RQ workers (needs raw client)
from platform_workers.rq_harness import run_rq_worker, WorkerConfig

config: WorkerConfig = {
    "redis_url": "redis://localhost:6379",
    "queue_name": "default",
    "events_channel": "events:jobs",
}
run_rq_worker(config)
```

### Redis Protocols

```python
from platform_workers.redis import (
    RedisStrProto,     # String client protocol
    RedisBytesProto,   # Binary client protocol
    RedisAsyncProto,   # Async client protocol
    RedisPubSubProto,  # Pub/sub protocol
    RedisError,        # Platform Redis error
    is_redis_error,    # Type-safe error check
    PubSubMessage,     # Pub/sub message TypedDict
)

# Type-safe error handling
try:
    redis.ping()
except Exception as e:
    if is_redis_error(e):
        print("Redis connection failed")
```

## RQ Harness

Typed RQ queue and worker utilities:

```python
from platform_workers.rq_harness import (
    rq_queue,
    rq_retry,
    get_current_job,
    run_rq_worker,
    WorkerConfig,
)

# Create queue client
from platform_workers.redis import redis_raw_for_rq
conn = redis_raw_for_rq("redis://localhost:6379")
queue = rq_queue("my-queue", conn)

# Enqueue job with retry
job = queue.enqueue(
    "mymodule.process_task",
    {"input": "data"},
    job_timeout=300,
    retry=rq_retry(max_retries=3, intervals=[10, 30, 60]),
)
print(job.get_id())

# Inside worker job
job = get_current_job()
if job:
    print(job.get_id())
```

### Worker Configuration

```python
from platform_workers.rq_harness import WorkerConfig, run_rq_worker

config: WorkerConfig = {
    "redis_url": "redis://localhost:6379",
    "queue_name": "default",
    "events_channel": "events:jobs",
}

run_rq_worker(config)
```

## Job Context

Publish job lifecycle events to Redis using the generic event schema:

```python
from platform_core.job_events import default_events_channel
from platform_workers.job_context import JobContext, make_job_context

ctx: JobContext = make_job_context(
    redis=redis_client,
    domain="turkic",
    events_channel=default_events_channel("turkic"),
    job_id="abc123",
    user_id=42,
    queue_name="default",
)

# Publish lifecycle events
ctx.publish_started()
ctx.publish_progress(50, "processing")
ctx.publish_completed(result_id="output.txt", result_bytes=1234)
ctx.publish_failed(error_kind="user", message="invalid input")
```

## Job Store

Store and retrieve job results:

```python
from platform_workers.job_store import JobStore

store = JobStore(redis=redis_client, prefix="jobs:")

# Store result
store.set_result(job_id="abc123", result={"status": "done"}, ttl_seconds=3600)

# Retrieve result
result = store.get_result(job_id="abc123")
```

## Health Checks

Readiness probes that check Redis connectivity:

```python
from platform_workers.health import readyz_redis, readyz_redis_with_workers
from platform_workers.redis import redis_for_kv

redis = redis_for_kv("redis://localhost:6379")

# Basic Redis check
@app.get("/readyz")
def ready() -> ReadyResponse:
    return readyz_redis(redis)

# Redis + worker presence check
@app.get("/readyz")
def ready() -> ReadyResponse:
    return readyz_redis_with_workers(redis, workers_key="rq:workers")
```

## API Reference

### Redis Module

| Type | Description |
|------|-------------|
| `redis_for_kv` | Create string Redis client |
| `redis_for_rq` | Create binary Redis client |
| `redis_raw_for_rq` | Create raw Redis client for RQ |
| `redis_for_pubsub` | Create async pub/sub client |
| `RedisError` | Platform Redis error |
| `is_redis_error` | Check if exception is Redis error |
| `PubSubMessage` | Pub/sub message TypedDict |

### Redis Protocols

| Protocol | Description |
|----------|-------------|
| `RedisStrProto` | String client protocol |
| `RedisBytesProto` | Binary client protocol |
| `RedisAsyncProto` | Async client protocol |
| `RedisPubSubProto` | Pub/sub protocol |

### RQ Harness

| Type | Description |
|------|-------------|
| `rq_queue` | Create RQ queue |
| `rq_retry` | Create retry configuration |
| `get_current_job` | Get current job in worker |
| `run_rq_worker` | Run RQ worker |
| `WorkerConfig` | Worker configuration TypedDict |

### Job Context

| Type | Description |
|------|-------------|
| `JobContext` | Job context with event publishing |
| `make_job_context` | Create job context |

### Job Store

| Type | Description |
|------|-------------|
| `JobStore` | Job result storage |

### Health

| Function | Description |
|----------|-------------|
| `readyz_redis` | Basic Redis readiness check |
| `readyz_redis_with_workers` | Redis + workers readiness check |

## Testing

Fake implementations for unit tests:

```python
from platform_workers.testing import FakeRedis, FakeJobContext

# Fake Redis client
redis = FakeRedis()
redis.set("key", "value")
assert redis.get("key") == "value"

# Fake job context
ctx = FakeJobContext()
ctx.publish_started()
assert ctx.events == [{"type": "started", ...}]
```

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- platform-core
- redis 5.2.0+ (optional)
- rq 2.0.0+ (optional)
- 100% test coverage enforced
