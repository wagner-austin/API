# platform-workers

Typed Redis helpers and RQ (Redis Queue) harness for background job processing.

## Installation

```bash
poetry add platform-workers
```

Requires `platform-core` and optionally `redis`, `rq` for runtime.

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
)
```

## RQ Harness

Typed RQ queue and worker utilities:

```python
from platform_workers.rq_harness import (
    rq_queue,
    rq_retry,
    get_current_job,
    run_rq_worker,
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

### Job Context

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

ctx.publish_started()
ctx.publish_progress(50, "processing")
ctx.publish_completed(result_id="output.txt", result_bytes=1234)
ctx.publish_failed(error_kind="user", message="invalid input")
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
