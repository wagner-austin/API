---
title: platform_workers — RQ + Redis background job pattern
tags: [platform-workers, rq, redis, background-jobs, libs]
related:
  - "[[monorepo-discipline]]"
source_paths:
  - libs/platform_workers
  - libs/platform_workers/README.md
source_git_blobs:
  "libs/platform_workers": 273a2b2e989a14f12d19ca7f1ae79a3d1eb89725
  "libs/platform_workers/README.md": b5204d1d51000d9ee256eb8f26973dc917ed2cb4
fact_checked: "2026-08-14"
confidence: high
hubs: [libs]
---

# platform_workers — RQ + Redis background job pattern

`platform_workers` is the shared library for background job processing across every service that needs one. Wraps [RQ (Redis Queue)](https://python-rq.org/) with typed Redis client factories, RQ harness helpers, and readyz-check plumbing. Services consume this lib — they do not talk to Redis or RQ directly.[^1]

## Typed Redis client factories

Instead of module-level `redis.from_url(...)` (which forces import-time side effects and makes testing harder), `platform_workers.redis` exposes typed factory functions[^2]:

- `redis_for_kv(url)` — string key-value operations
- `redis_for_rq(url)` — binary client for RQ
- `redis_raw_for_rq(url)` — raw client for RQ (full API surface)
- `redis_for_pubsub(url)` — async client for pub/sub

Each factory returns a typed client with the right serialization shape for its use case — `RedisStrProto`, `RedisBytesProto`, `RedisAsyncProto`, and the concrete `_RedisBytesClient`[^2]. This makes it a mypy error to accidentally pass a `redis_for_kv` client to code that expects the raw RQ client.

## RQ harness — `rq_queue` + `rq_retry` (+ worker entry helpers)

The client-side helpers cover the common enqueue shape[^3]:

```python
from platform_workers.rq_harness import rq_queue, rq_retry

queue = rq_queue("my-queue", redis_raw_for_rq("redis://localhost:6379"))
job = queue.enqueue("mymodule.task", {"input": "data"})
```

`rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike` wires the retry policy in a single call so every service uses the same backoff shape[^3].

For the worker side, `run_rq_worker(config: WorkerConfig)`[^3] is the canonical entry point — services stand up a `worker_entry.py` module that instantiates a `WorkerConfig` and calls `run_rq_worker`; the runtime side (rq module load, connection binding, simple-worker construction) is handled by the harness. `get_current_job()` and `rq_fetch_job(job_id, connection)` cover the inside-a-job and by-id lookup shapes[^3].

## Health check plumbing

`readyz_redis(redis)`[^4] is the canonical readiness check for Redis-dependent services. Returns a `ReadyResponse` TypedDict: `{"status": "ready", "reason": None}` on success, `{"status": "degraded", "reason": "<code>"}` on Redis ping failure (`"redis error"` / `"redis no-pong"`). The consuming `/readyz` route maps `"degraded"` to HTTP 503. Services use it so the cross-service readiness contract stays consistent. The `readyz_redis_with_workers(redis, *, workers_key="rq:workers")` variant additionally probes that at least one RQ worker is registered by counting `SCARD` on the workers set — reporting `"no-worker"` if the count is zero. Services with a `worker_entry.py` prefer this one (qr-api, data-bank-api, ...)[^4].

## Why the wrapper

Without `platform_workers`[^1]:
- Every service imports `redis` at module scope, making it impossible to swap the client for testing.
- Every service picks its own retry policy — some retry aggressively, some don't retry at all.
- Every readyz check has a slightly different failure shape, making cross-service alerting inconsistent.
- Every service duplicates the RQ boilerplate around queue creation.

With `platform_workers`[^1]:
- DI-friendly factories → tests inject a fake Redis without patching modules.
- One retry policy → cross-service backoff behavior is consistent.
- One readyz shape → alerting rules are shared.
- One RQ harness → job wiring is boilerplate-free.

## When a service does NOT need this lib

Services that don't queue background work — pure stateless request/response APIs like `github-stats-api` (SVG rendering), `grandma-api`, and `opportunity-radar-api` — don't depend on `platform_workers`[^5]. Adding it as a dependency "just in case" pulls in Redis + RQ transitively for no benefit. If you're not enqueuing jobs or serving `/readyz` off a shared Redis, don't import it.

The corollary: if you find yourself needing background work in a service that doesn't have `platform_workers`, add the lib to that service's `pyproject.toml` — don't reach for `redis` directly. Services that DO consume it today: `data-bank-api`, `Model-Trainer`, `Art-Trainer`, `qr-api` (has an RQ worker via `run_rq_worker` + a KV Redis for cache), `music-wrapped-api`, `covenant-radar-api`, `transcript-api`, `turkic-api`, `handwriting-ai`[^5].

## Related — `platform_core` `AsyncHttpClient` + `platform_ml` artifact storage

`platform_workers` handles the "queue → work → result" backbone[^1]. For the "make an HTTP call to a downstream service" side, use `platform_core`'s `AsyncHttpClient` (typed, OAuth-aware, retry-aware). For persisting model artifacts produced by workers, use `platform_ml` (manifests + device/precision auto-detection). These three libs are the standard trio a service pulls in for anything more than a stateless request handler.

[^1]: [`libs/platform_workers/README.md`](../../libs/platform_workers/README.md) — "Typed Redis helpers and RQ (Redis Queue) harness for background job processing."
[^2]: `libs/platform_workers/src/platform_workers/redis.py:282,315,324,336` — `redis_raw_for_rq(url) -> _RedisBytesClient`, `redis_for_kv(url) -> RedisStrProto`, `redis_for_rq(url) -> RedisBytesProto`, `redis_for_pubsub(url) -> RedisAsyncProto`. Four distinct return protocols is what makes the mis-pass a type error.
[^3]: `libs/platform_workers/src/platform_workers/rq_harness.py` — `WorkerConfig(TypedDict)` at `:19`, `rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike` at `:102`, `get_current_job() -> CurrentJobProto | None` at `:229`, `run_rq_worker(config: WorkerConfig) -> None` at `:240`, `rq_fetch_job(job_id: str, connection: _RedisBytesClient) -> FetchedJobProto` at `:290`. All five re-verified exact 2026-08-14. The re-check was prompted by the trailing-slash under-pin: this page's `libs/platform_workers/` entry had been pinned to the package's `.gitignore` rather than to the tree, so nothing under it was watched. What changed since the 2026-07-20 check is `_RQJobInternal`, which declared `get_id()`; rq has removed that method in favour of an `id` property, so the protocol as written raised `AttributeError` against any modern rq while the test fakes — which implemented the protocol as declared — kept passing. Protocol and fake now both mirror `id`. This page asserts nothing about `get_id`, so no claim here was wrong; the anchors below the protocol did all shift by 8 lines and are re-taken above.
[^4]: `libs/platform_workers/src/platform_workers/health.py` — `readyz_redis(redis: RedisStrProto) -> ReadyResponse` at `:17` returning `{"status": "degraded", "reason": "redis error"}` (`:34`), `{"status": "degraded", "reason": "redis no-pong"}` (`:38`), else `{"status": "ready", "reason": None}` (`:40`); `readyz_redis_with_workers(redis, *, workers_key: str = "rq:workers")` at `:43-47`, which additionally calls `redis.scard(workers_key)` (`:75`) and returns `{"status": "degraded", "reason": "no-worker"}` when the count is `<= 0` (`:84-86`). `ReadyResponse` is imported from `platform_core.health` (`:9`).
[^5]: Verified 2026-07-31 by grepping every `pyproject.toml` under `services/` and `libs/` for a `platform_workers` dependency. The ten services listed are exactly the matches; `github-stats-api`, `grandma-api`, and `opportunity-radar-api` return no match. (`libs/platform_discord` and `libs/platform_music` also consume it, but they are libs rather than services.)
