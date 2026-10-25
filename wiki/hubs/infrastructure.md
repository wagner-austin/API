# Infrastructure

The cross-service platform that makes the monorepo run — Redis (queues + RQ), PostgreSQL (persistence for covenant + services that need it), Traefik (reverse proxy for the FastAPI fleet), docker-compose orchestration, and the monorepo-wide build + test + lint conventions (strict mypy, 100% coverage, `monorepo-guards.toml`). Covers boot + lifecycle patterns, per-service Dockerfile conventions, CI + local dev topology, and the cross-service contracts that keep the fleet consistent.

[Monorepo Discipline](../pages/monorepo-discipline.md) -- the three cross-cutting rules every service and lib enforces: strict mypy (zero `Any`), 100% coverage (statements + branches), `monorepo_guards` static analysis (20+ Python + Rust checks)

[Reduction order is an environment variable read once](../pages/determinism-env-read-once-at-library-load.md) -- why the cuBLAS / cuBLASLt / BLAS-thread strings live in `platform_core`; setting them after library load is accepted in silence; zero cuBLASLt workspace removes split-K, what that measurably buys across cards, and what it explicitly does not buy

[Guard-pass as a fine-tune metric](../pages/code-style-guard-pass-instrument-limits.md) -- what code-style-eval can and cannot measure: truncation gates the metric before any checker runs, the sandbox's import boundary, why installing the corpus's deps changes the instrument and must enter the fingerprint, and the exact-McNemar power the 392-item corpus does not have

[Runner listeners and dropped jobs](../pages/fleet-runner-listener-isolation-and-dropped-jobs.md) -- why lavender's runner listeners sit in an unbounded `listeners.slice` with each job moved into a scope of the bounded `runners.slice`, why `MemoryLow` could not protect them, and how the hub's reap pass ends a Worker only when GitHub reads its runner not busy (the offline busy=true incident)

[loki's tunnel stall is the hub starting cloudflared](../pages/fleet-loki-tunnel-stall-cloudflared-start.md) -- why fleet ssh to loki timed out "during banner exchange": loki's connector and the hub's tunnels recorded nothing, a quiet machine on the same uplink saw no stall, and the seconds went into each per-call `cloudflared` hashing its own 40 MB executable on the CPU-starved hub; the A/B against one long-lived forwarder

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
