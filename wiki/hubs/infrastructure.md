# Infrastructure

The cross-service platform that makes the monorepo run — Redis (queues + RQ), PostgreSQL (persistence for covenant + services that need it), Traefik (reverse proxy for the FastAPI fleet), docker-compose orchestration, and the monorepo-wide build + test + lint conventions (strict mypy, 100% coverage, `monorepo-guards.toml`). Covers boot + lifecycle patterns, per-service Dockerfile conventions, CI + local dev topology, and the cross-service contracts that keep the fleet consistent.

[Monorepo Discipline](../pages/monorepo-discipline.md) -- the three cross-cutting rules every service and lib enforces: strict mypy (zero `Any`), 100% coverage (statements + branches), `monorepo_guards` static analysis (20+ Python + Rust checks)

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
