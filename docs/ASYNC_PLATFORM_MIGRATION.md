# Async Platform Migration (API Monorepo)

## Document Status
| Field | Value |
|-------|-------|
| **Author** | Platform Engineering |
| **Created** | 2025-11-28 |
| **Status** | Draft (proposed) |
| **Scope** | All services and shared libs |

## Why Change
- Current stack is synchronous end-to-end (Redis, RQ, HTTP clients), with only surface-level async handlers in a few routes. We block threads on I/O and carry duplicate sync-only patterns per service.
- We want a single, consistent async runtime to improve concurrency on I/O-bound workloads (Redis, HTTP, storage) and to eliminate drift across services.
- No back-compat or dual paths: migrate libraries and services to async and remove sync pathways to prevent divergence.

## Target State (Strict, No Fallbacks)
- Shared async primitives live in libs and are the only allowed interfaces:
  - `platform_workers.async_redis`: Typed Protocols for async Redis (publish, hash ops, kv), no Any/cast/type: ignore, no shims.
  - `platform_workers.async_queue`: Async enqueue/worker abstraction (RQ-compatible adapter or async-native queue) with typed job payloads.
  - `platform_core.async_events`: Async publish helpers around `job_events` using async Redis clients.
  - Async health/readiness helpers that verify Redis connectivity without best-effort retries.
- Services use only async I/O paths: async routes, async Redis publish, async enqueue, async worker entrypoints, async health.
- Tests enforce 100% statement and branch coverage for all async paths (libs and services). No coverage exclusions, no noqa.
- Exception handling is explicit and propagates; no silent retries, no “best effort.”
- No dual sync/async stacks after migration; sync helpers removed or guarded in docs as deprecated/removed.

## Design Pillars
1. **Strict typing only**: Protocols for external clients, no Any, no casts, no type: ignore, no stubs/.pyi.
2. **Parse at boundaries**: Validate inputs (JSON, env, Redis payloads) immediately; fail fast with precise errors.
3. **No fallback paths**: Async operations must succeed or raise; no hidden try/except softening errors.
4. **One path per concern**: A single async helper per concern in libs; services cannot reimplement Redis/event logic.
5. **Total coverage**: 100% statements and branches for new async code and updated services.

## Library Work (Phase 1)
- Add `platform_workers.async_redis`:
  - Protocols for publish, hset/hgetall, get/set, close; no generic `Redis[Any]` references.
  - Factory to build an asyncio Redis client (choice: redis-py asyncio). No dynamic Any leakage; use typed wrappers.
- Add `platform_workers.async_queue`:
  - Async enqueue interface and worker loop abstraction. Either:
    - RQ async adapter if viable, or
    - Async-native worker (e.g., arq-like) wrapped in strict Protocols.
  - Typed job payload and result codecs (no Any), explicit failure modes.
- Add `platform_core.async_events`:
  - Async publish helpers for `job_events` with explicit channels.
  - No best-effort publish; failures propagate.
- Add async health helpers that hit Redis via async client and raise on failure (no retries).
- Tests: Full branch coverage for all new modules; guard rules enforced.

## Service Migration Plan (Phase 2+)
- Order (smallest async surface first): transcript-api → qr-api → turkic-api → handwriting-ai → Model-Trainer → data-bank-api.
- For each service:
  - Replace sync Redis usage with `async_redis` helpers.
  - Replace sync publish with `async_events`.
  - Replace sync enqueue/worker entry with `async_queue` worker entry.
  - Update health/ready checks to async Redis helpers.
  - Ensure routes await all I/O; remove blocking calls in request paths.
  - Tests: add/adjust fixtures for async Redis/queue; keep 100% cov/branch.
- Remove sync RQ wiring and sync Redis publish from services once migrated.

## Clients / Subscribers
- If DiscordBot or other clients consume pub/sub streams, add an async subscriber helper (Protocol-based) or keep them sync explicitly documented. Decide per client; avoid mixing models silently.

## Risks and Mitigations
- **Queue library fit**: If RQ async support is insufficient, switch to an async-native worker. Mitigation: prototype in Phase 1 with strict adapters and tests.
- **Migration blast radius**: Touches every service. Mitigation: migrate sequentially; run `make check` per lib/service; remove sync code only after the service switches.
- **Runtime dependencies**: New redis asyncio extra may change Docker base images. Mitigation: pin deps and update Dockerfiles during each service migration.

## Validation
- `make check` in libs after Phase 1, then per service after each migration.
- 100% coverage enforced; no exclusions.
- Explicit negative tests for async publish/queue error paths (exceptions propagate).

## Deliverables Checklist
- [ ] `platform_workers.async_redis` with tests (100% cov/branch).
- [ ] `platform_workers.async_queue` (adapter or native) with tests.
- [ ] `platform_core.async_events` with tests.
- [ ] Async health helpers.
- [ ] transcript-api migrated and green `make check`.
- [ ] qr-api migrated and green `make check`.
- [ ] turkic-api migrated and green `make check`.
- [ ] handwriting-ai migrated and green `make check`.
- [ ] Model-Trainer migrated and green `make check`.
- [ ] data-bank-api migrated and green `make check`.
- [ ] Docs updated; sync paths removed or marked removed.
