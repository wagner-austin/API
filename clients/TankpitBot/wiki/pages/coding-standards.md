---
title: Coding Standards
tags: [architecture, testing, quality]
related:
  - "[[inheritance-chain]]"
source_paths:
  - "src/tankpit_bot"
  - "tests"
source_git_blobs:
  "src/tankpit_bot": "744443e972df97d609f4c2530ef056e25c877db5"
  "tests": "2170c901aac496262dd7b981773e87d97f54a0b6"
fact_checked: "2026-07-31"
confidence: high
hubs: [architecture]
---

# Coding Standards

## Type safety — zero escape hatches

- No `Any`, `object` cast, `type: ignore`, `.pyi`, `noqa`[^1]
- No `TYPE_CHECKING` blocks — all imports at module level[^1]
- No `TypeAlias` — use Literal types or expand unions[^1]
- Ruff bans enforced in `pyproject.toml` `[tool.ruff.lint.flake8-tidy-imports.banned-api]`[^1]
- mypy strict mode with all `disallow_any_*` flags enabled[^1]

## Testing — no fakes, no mocks, no monkey-patching

- No `unittest.mock` anywhere in the test suite[^2]
- `MonkeyPatchBanRule` guard enforces save-and-restore DI pattern, 0 violations[^2]
- `_test_hooks` DI pattern: `tankpit_bot/_test_hooks/` is a package of 8 domain submodules exposing module-level injectable callables and Protocols; tests swap in protocol-matching implementations by save-and-restore ([[testing-patterns]])[^2]
- No weak assertions (`assert len(x) >= 0` etc.) — concrete assertions on specific values[^2]
- 100% coverage required (`fail_under = 100` in `pyproject.toml`)[^2]

## Code style

- No back-compat shims, no thin wrappers, no fallbacks, no legacy code[^1]
- No duplicate code — keep codebase DRY[^1]
- No placeholder code, no stubs[^1]
- No `try/except` for flow control — only at system boundaries[^1]
- Google-style docstrings[^1]
- TypedDict with encode/decode and `require_*` validation[^1]
- **File size 400-600 lines max, src AND tests** (user ruling 2026-07-31, verbatim: "we need modular, clear sepraration of concerns, no monolithic files. 400 - 600 lines, including test files too" — supersedes the earlier "under 400 where possible"). Split any over-bar file you touch into cohesive modules; never grow a file already past 600. No re-export shims when splitting (the wrapper ban above applies) — move call-site imports properly. Backlog at ruling time: 45 files over 600, listed in the 2026-07-31 log entry.[^1]

## Git discipline

- NEVER run `git checkout`/`clean`/`restore` without explicit user confirmation[^3]
- NEVER run any git command without explicit user permission[^3]
- Create new commits rather than amending[^3]

[^1]: user (Austin), enforced across all sessions — "no back compat shims, no thin wrappers, no fallbacks, no legacy code, no type alias, no duplicate code"
[^2]: MonkeyPatchBanRule in monorepo-guards; _test_hooks pattern verified across 5,631 tests at 100% coverage. `poetry run python -m scripts.guard` 2026-07-31: 0 violations across every rule group, `mock-ban` and `monkey-patch-ban` included
[^3]: user (Austin), 2026-06-13 — incident destroyed uncommitted work; absolute rule since
