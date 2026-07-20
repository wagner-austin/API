---
title: Monorepo Discipline — strict mypy, 100% coverage, monorepo-guards
tags: [monorepo, typing, testing, guards, workspace-discipline]
related: [[platform-workers-rq-pattern]]
sources:
  - monorepo-guards.toml
  - libs/monorepo_guards/
  - README.md
fact_checked: 2026-07-20
confidence: high
---

# Monorepo Discipline

The api monorepo enforces three cross-cutting rules on every service and library: **strict mypy (zero `Any`)**, **100% test coverage (statements + branches)**, and **`monorepo_guards` static analysis** (20+ Python + Rust checks). Any service or lib that violates the rules fails `make check` and cannot merge.[^1]

## Strict mypy — zero `Any`

Every `pyproject.toml` in `services/` and `libs/` runs mypy in strict mode. Explicit `Any` — as annotation, cast target, or return type — is a lint failure. Third-party libraries without type stubs get typed wrappers in the appropriate `platform_*` lib; the service consumes the wrapper, not the untyped import.

This makes function signatures the ground truth: reading a service's `main.py` and `api/*.py` tells you the request/response shape without running the code. It also makes refactors safe — mypy catches every consumer at edit time.

## 100% coverage — statements + branches

Coverage runs statement-level AND branch-level. A partially-covered `if x is None` branch fails even when the statement lines are hit. This forces test authors to think through the negative and edge paths, not just the happy path.

Coverage is per-package. `services/data-bank-api` covers its own code; it doesn't count coverage of `libs/platform_core` (which has its own 100%). Tests that reach into another package to hit its lines are a smell — mock the shared lib's public interface (via DI hooks, not `unittest.mock`) and test the wrapper's behavior locally.

## `monorepo_guards` — the static analysis rule engine

Lives in [`libs/monorepo_guards/`](../../libs/monorepo_guards/). Config in [`monorepo-guards.toml`](../../monorepo-guards.toml). 20+ rules that fire from every service's `make lint`. Categories:

- **Anti-patterns** — banned imports, banned function calls, banned dependency shapes
- **Contract enforcement** — required helpers, required error handlers, required lifecycle wiring
- **Rust guards** — parallel rule set for `cleargbm_rs` (Rust core) that catches unsafe patterns per-crate

Adding a new rule is a small PR in `libs/monorepo_guards/`. Any service or lib that starts violating the new rule after the merge fails `make lint` — which is the point.

## Why the three together

Any one of these alone leaves gaps. Together:
- **mypy** catches type-shape drift at edit time (before the tests run).
- **coverage** catches logic paths that no test exercises (before the drift ships).
- **guards** catch anti-patterns that pass typecheck AND tests but produce production incidents (bare error capture, missing lifecycle, forked helpers).

Each layer catches drift the others miss. Removing any one opens a class of incident that historically produced a real fire.

## Practical implications

- **Never `Any`** — not as a shortcut, not as "I'll type it later." Use `TypedDict`, `Protocol`, or a typed wrapper.
- **Never `# type: ignore`** without a comment naming the specific mypy bug or third-party gap it works around.
- **Never `pragma: no cover`** except at genuine unreachable boundaries (e.g. `if __name__ == "__main__":` in a library).
- **Never `# noqa`** to silence a monorepo_guard — fix the underlying pattern.

If your change makes any of the three rules seem too restrictive, that's usually a signal the change is fighting the architecture, not that the rule needs an exception.

[^1]: [README.md § Services table](../../README.md) — "Type Safety: mypy strict mode, zero `Any` types. 100% Test Coverage: Statements and branches."
