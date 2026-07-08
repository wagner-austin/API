---
title: platform_workers — RQ + Redis background job pattern
tags: [platform-workers, rq, redis, background-jobs, libs]
related: [[monorepo-discipline]]
sources:
  - libs/platform_workers/
  - libs/platform_workers/README.md
fact_checked: 2026-07-07
confidence: high
---

# platform_workers — RQ + Redis background job pattern

`platform_workers` is the shared library for background job processing across every service that needs one. Wraps [RQ (Redis Queue)](https://python-rq.org/) with typed Redis client factories, RQ harness helpers, and readyz-check plumbing. Services consume this lib — they do not talk to Redis or RQ directly.[^1]

## Typed Redis client factories

Instead of module-level `redis.from_url(...)` (which forces import-time side effects and makes testing harder), `platform_workers.redis` exposes typed factory functions:

- `redis_for_kv(url)` — string key-value operations
- `redis_for_rq(url)` — binary client for RQ
- `redis_raw_for_rq(url)` — raw client for RQ (full API surface)
- `redis_for_pubsub(url)` — async client for pub/sub

Each factory returns a typed client with the right serialization shape for its use case. This makes it a mypy error to accidentally pass a `redis_for_kv` client to code that expects the raw RQ client.

## RQ harness — `rq_queue` + `rq_retry`

Two thin helpers cover the common job shape:

```python
from platform_workers.rq_harness import rq_queue, rq_retry

queue = rq_queue("my-queue", redis_raw_for_rq("redis://localhost:6379"))
job = queue.enqueue("mymodule.task", {"input": "data"})
```

`rq_retry` wires the retry policy in a single call so every service uses the same backoff shape.

## Health check plumbing

`readyz_redis(redis)` is the canonical readiness check for Redis-dependent services. Returns the FastAPI response shape (`{"status": "ready"}` on success, `503` with a specific `reason` on failure). Services use it in their `/readyz` route so the cross-service readiness contract stays consistent.

## Why the wrapper

Without `platform_workers`:
- Every service imports `redis` at module scope, making it impossible to swap the client for testing.
- Every service picks its own retry policy — some retry aggressively, some don't retry at all.
- Every readyz check has a slightly different failure shape, making cross-service alerting inconsistent.
- Every service duplicates the RQ boilerplate around queue creation.

With `platform_workers`:
- DI-friendly factories → tests inject a fake Redis without patching modules.
- One retry policy → cross-service backoff behavior is consistent.
- One readyz shape → alerting rules are shared.
- One RQ harness → job wiring is boilerplate-free.

## When a service does NOT need this lib

Services that don't queue background work — pure request/response APIs like `qr-api` — don't need `platform_workers`. Adding it as a dependency "just in case" pulls in Redis + RQ transitively for no benefit. If you're not enqueuing jobs, don't import it.

The corollary: if you find yourself needing background work in a service that doesn't have `platform_workers`, add the lib to that service's `pyproject.toml` — don't reach for `redis` directly.

## Related — `platform_core` `AsyncHttpClient` + `platform_ml` artifact storage

`platform_workers` handles the "queue → work → result" backbone. For the "make an HTTP call to a downstream service" side, use `platform_core`'s `AsyncHttpClient` (typed, OAuth-aware, retry-aware). For persisting model artifacts produced by workers, use `platform_ml` (manifests + device/precision auto-detection). These three libs are the standard trio a service pulls in for anything more than a stateless request handler.

[^1]: [`libs/platform_workers/README.md`](../../libs/platform_workers/README.md) — "Typed Redis helpers and RQ (Redis Queue) harness for background job processing."
