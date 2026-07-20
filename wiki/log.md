# Wiki Operation Log

Append-only. Log structural operations (new hubs, decomposition, audits, cleanups). Routine page edits don't need a log entry — git history covers those.

## [2026-07-06] init | api monorepo wiki scaffolded
Hubs created: services, clients, libs, infrastructure
Notes: initial scaffold via /wiki-init. Empty pages/ — content added as subsystems get documented.

## [2026-07-07] first-batch | 3 starter pages
Pages written: monorepo-discipline, platform-workers-rq-pattern, service-port-map
Hubs updated: services (+1), libs (+1), infrastructure (+1)
Notes: all pages audited claim-by-claim against the code before landing — Redis client factory names, RQ harness helpers, monorepo-guards.toml location, service port assignments all verified in the source. Skipped writing more api pages this batch because deeper subsystem context (Kafka streaming in covenant-radar-api, Kohya-ss backend in Art-Trainer, MMS-LID in platform_langid) would require reading service internals; those pages should be written by someone who's touched the subsystem code, not paraphrased from READMEs.

## [2026-07-20] audit | all pages verified against current code
Pages audited: monorepo-discipline, platform-workers-rq-pattern, service-port-map (plus all 4 hubs + index)
Pages updated: index.md, hubs/services.md, hubs/libs.md, hubs/clients.md, pages/service-port-map.md, pages/platform-workers-rq-pattern.md, pages/monorepo-discipline.md (fact_checked bump only)

Findings and fixes (all applied):
1. **`doc-extract-api` service was missing from the wiki entirely.** Real service under `services/doc-extract-api/` with its own Dockerfile, poetry env, README, and layered `docker-compose.yml`. Uses port 8012 (host) → 8000 (container) per `services/doc-extract-api/docker-compose.yml:24`; runs `hypercorn` via Dockerfile CMD; depends on `psycopg` (postgres) and `platform_workers`. Added to: `index.md` services enumeration; `hubs/services.md` services list; `pages/service-port-map.md` port table with an inline note that the assignment is authoritative in the service's own compose file, not the root README. Root cause: root `README.md` Services table still doesn't list this service — the wiki inherited the omission because its citation chain terminates at the root README.
2. **"8012 is free" claim in service-port-map was wrong.** Corrected to "8013 is free". Also expanded step 3 of "Adding a new service" to describe the layered-compose pattern that doc-extract-api uses.
3. **`platform_devpost` lib was missing from `index.md` libs enumeration.** Added. (Already present in `hubs/libs.md`; added a one-line description parenthetical there to match the pattern of surrounding libs.)
4. **`qr-api` was cited as the canonical "doesn't need `platform_workers`" example — actually a heavy consumer.** `services/qr-api/src/qr_api/*` imports `redis_for_kv`, `run_rq_worker`, `WorkerConfig`, `readyz_redis_with_workers`, `RedisStrProto` across six modules. Corrected example to `github-stats-api` / `grandma-api` / `opportunity-radar-api` (grep-verified they have zero `platform_workers` or `rq` imports). Added an explicit "services that DO consume it today" list so the boundary is unambiguous.
5. **`readyz_redis` response shape was mis-described.** Wiki claimed "returns `{status: ready}` on success, 503 with a reason on failure". Actual: returns a `ReadyResponse` TypedDict `{"status": "ready"|"degraded", "reason": None|str}`; the consuming route maps `degraded` → HTTP 503. Fixed with the exact success/degraded shapes verified in `libs/platform_workers/src/platform_workers/health.py`.
6. **`platform_workers` RQ surface was under-documented.** The page named only `rq_queue` + `rq_retry` and called them "two thin helpers", but every service with a worker actually goes through `run_rq_worker(config: WorkerConfig)` as the entry point (see qr-api's `worker_entry.py`, etc.). Added a paragraph naming `run_rq_worker`, `WorkerConfig`, `get_current_job`, `rq_fetch_job` — verified as `__all__` / top-level `def`s in `rq_harness.py`. Added `readyz_redis_with_workers` alongside `readyz_redis` with its actual `SCARD`-on-workers-set behaviour.
7. **`hubs/clients.md` looked empty ("0 pages") — but the client `TankpitBot` maintains its own full three-tier wiki at `clients/TankpitBot/wiki/`.** Not a wiki bug per se, but a navigation gap: a new AI reading `api/wiki` would never find 50+ pages of TankpitBot knowledge. Added a section pointing at that wiki as the source of truth for tankpit facts, and defined what this hub SHOULD hold (monorepo-integration surfaces — none yet written).

Verified (no changes needed):
- `platform_workers` typed Redis factories — `redis_for_kv`, `redis_for_rq`, `redis_raw_for_rq`, `redis_for_pubsub` — all present at `libs/platform_workers/src/platform_workers/redis.py:{282,315,324,336}`.
- `rq_queue` + `rq_retry` — present at `rq_harness.py:{185,94}`.
- `monorepo_guards` rule count — the wiki says "20+"; actual is 25 `*_rules.py` files in `libs/monorepo_guards/src/monorepo_guards/`, each containing multiple rules. "20+" is correct and intentionally conservative.
- `monorepo-guards.toml` location — at repo root, path referenced in wiki resolves correctly.
- Every port `8000-8011` matched between README table, root `docker-compose.yml` port-map comment, and the wiki table.
- covenant-radar-api Kafka claim — `confluent-kafka = "^2.12"` in `services/covenant-radar-api/pyproject.toml:33`.
- PostgreSQL claim — root `docker-compose.yml` runs `postgres:16-alpine`; consumed by covenant-radar-api, doc-extract-api, and `covenant_persistence` lib.

Root causes recorded for the misses: (a) the wiki cites the root `README.md` as its primary for service enumeration, which itself omits doc-extract-api — future audits should cross-check `ls services/` against the README, not trust the README alone; (b) the qr-api "doesn't need workers" example was a guess based on the service name, not a grep of its imports.

`fact_checked` on all three content pages bumped to 2026-07-20.
