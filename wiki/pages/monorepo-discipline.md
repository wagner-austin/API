---
title: Monorepo Discipline — strict mypy, 100% coverage, monorepo-guards
tags: [monorepo, typing, testing, guards, workspace-discipline]
related:
  - "[[platform-workers-rq-pattern]]"
source_paths:
  - monorepo-guards.toml
  - libs/monorepo_guards
  - README.md
source_git_blobs:
  "monorepo-guards.toml": 496d1d68863cdea918fb7a0f94153179d711805e
  "libs/monorepo_guards": 853af883055eeeb77d69532d503a6bbbe12093ef
  "README.md": a5bbc13914ac4cd4428f57ad76b987502990bfe7
fact_checked: "2026-08-14"
confidence: high
hubs: [infrastructure]
---

# Monorepo Discipline

The api monorepo enforces three cross-cutting rules on every service and library: **strict mypy (zero `Any`)**, **100% test coverage (statements + branches)**, and **`monorepo_guards` static analysis** (20+ Python + Rust checks). Any service or lib that violates the rules fails `make check` and cannot merge.[^1]

## Strict mypy — zero `Any`

Every `pyproject.toml` in `services/` and `libs/` runs mypy in strict mode — verified across all 36 of them[^2]. Explicit `Any` — as annotation, cast target, or return type — is a lint failure[^2]. Third-party libraries without type stubs get typed wrappers in the appropriate `platform_*` lib; the service consumes the wrapper, not the untyped import.

This makes function signatures the ground truth: reading a service's `main.py` and `api/*.py` tells you the request/response shape without running the code. It also makes refactors safe — mypy catches every consumer at edit time[^1].

## 100% coverage — statements + branches

Coverage runs statement-level AND branch-level — `branch = true` and `fail_under = 100` in all 36 packages[^3]. A partially-covered `if x is None` branch fails even when the statement lines are hit. This forces test authors to think through the negative and edge paths, not just the happy path.

Coverage is per-package[^3]. `services/data-bank-api` covers its own code; it doesn't count coverage of `libs/platform_core` (which has its own 100%). Tests that reach into another package to hit its lines are a smell — mock the shared lib's public interface (via DI hooks, not `unittest.mock`) and test the wrapper's behavior locally.

## `monorepo_guards` — the static analysis rule engine

Lives in [`libs/monorepo_guards/`](../../libs/monorepo_guards/). Config in [`monorepo-guards.toml`](../../monorepo-guards.toml). 20+ rules that fire from every service's `make lint`[^4]. Categories:

- **Anti-patterns** — banned imports, banned function calls, banned dependency shapes
- **Contract enforcement** — required helpers, required error handlers, required lifecycle wiring
- **Rust guards** — parallel rule set for `cleargbm_rs` (Rust core) that catches unsafe patterns per-crate

Adding a new rule is a small PR in `libs/monorepo_guards/`[^4]. Any service or lib that starts violating the new rule after the merge fails `make lint` — which is the point.

## Why the three together

Any one of these alone leaves gaps[^1]. Together:
- **mypy** catches type-shape drift at edit time (before the tests run).
- **coverage** catches logic paths that no test exercises (before the drift ships).
- **guards** catch anti-patterns that pass typecheck AND tests but produce production incidents (bare error capture, missing lifecycle, forked helpers).

Each layer catches drift the others miss[^5]. Removing any one opens a class of incident that historically produced a real fire.

## Practical implications

- **Never `Any`** — not as a shortcut, not as "I'll type it later." Use `TypedDict`, `Protocol`, or a typed wrapper.
- **Never `# type: ignore`** without a comment naming the specific mypy bug or third-party gap it works around.
- **Never `pragma: no cover`** except at genuine unreachable boundaries (e.g. `if __name__ == "__main__":` in a library).
- **Never `# noqa`** to silence a monorepo_guard — fix the underlying pattern.

If your change makes any of the three rules seem too restrictive, that's usually a signal the change is fighting the architecture, not that the rule needs an exception[^5].

[^1]: [README.md](../../README.md) `:171-172` — verbatim: "**Type Safety**: mypy strict mode, zero `Any` types, zero `cast`, zero `type: ignore`" and "**Test Coverage**: 100% statement + branch coverage enforced". **Quote corrected 2026-08-05:** this footnote previously read "Type Safety: mypy strict mode, zero `Any` types. 100% Test Coverage: Statements and branches" and cited a "§ Services table" it does not come from. Neither half matched the file — the Type Safety line has since been extended with `zero cast, zero type: ignore`, and the coverage line was reworded, so the string "100% Test Coverage" appears nowhere in README.md. Re-read from the file rather than repinned.
[^2]: Verified 2026-07-31 by sweeping every `pyproject.toml` at depth ≤ 2 under `libs/` and `services/`: 36 of 36 declare `strict = true`. `libs/covenant_ml/pyproject.toml:44,49-52` is representative — `strict = true` plus `disallow_any_unimported`, `disallow_any_expr`, `disallow_any_decorated`, and `disallow_any_explicit` all `true`, which is what makes an explicit `Any` a failure rather than a warning.
[^3]: Same sweep, 2026-07-31; re-checked 2026-08-14 after the trailing-slash under-pin was found — this page's `libs/monorepo_guards/` entry had been pinned to that package's `.gitignore` rather than to its tree, so nothing under it was watched. The only change since is `ExceptionsRule._log_call_any` in `exceptions_rules.py`, which added `write_line` to the set of calls that count as surfacing a failure in an `except` body (the sanctioned output channel of the stdlib-only clients). This page states the three rules' intent and asserts nothing about that regex, so no claim moved. 36 of 36 `pyproject.toml` files declare both `branch = true` and `fail_under = 100`, with no exceptions. `libs/covenant_ml/pyproject.toml:88,93` is representative. Per-package scoping follows from each package carrying its own `[tool.coverage]` block rather than a repo-level one.
[^4]: `libs/monorepo_guards/src/monorepo_guards/` — 30 modules defining 33 `*Rule` classes across 31 distinct rule names (counted 2026-07-31), so "20+ rules" holds with margin. Config at `monorepo-guards.toml` in the repo root.
[^5]: [synthesis] over the three verified layers — `pyproject.toml` at depth ≤ 2 under `libs/` and `services/` (36 of 36 declaring `strict = true`, and 36 of 36 declaring both `branch = true` and `fail_under = 100`), plus `libs/monorepo_guards/src/monorepo_guards/` (30 modules, 33 `*Rule` classes, 31 distinct rule names). Those measurements are [^2], [^3] and [^4]; this footnote adds no source of its own. **Documented negative:** the assertion that removing a layer "historically produced a real fire" is an editorial judgement — no incident record exists for it in this repo, and none is cited.
