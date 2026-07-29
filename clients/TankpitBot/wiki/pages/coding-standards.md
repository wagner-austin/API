---
title: Coding Standards
tags: [architecture, testing, quality]
related:
  - "[[inheritance-chain]]"
source_paths:
  - "src/tankpit_bot"
  - "tests"
source_git_blobs:
  "src/tankpit_bot": "471868c4ca9b6f7837650c3dc482784be43c9a0f"
  "tests": "a49bca58707400e57a5c473586840a17528030ea"
fact_checked: "2026-06-16"
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
- `_test_hooks` DI pattern: production code declares `_test_hooks` dict with injectable callables; tests swap in protocol-matching implementations[^2]
- No weak assertions (`assert len(x) >= 0` etc.) — concrete assertions on specific values[^2]
- 100% coverage required (`fail_under = 100` in `pyproject.toml`)[^2]

## Code style

- No back-compat shims, no thin wrappers, no fallbacks, no legacy code[^1]
- No duplicate code — keep codebase DRY[^1]
- No placeholder code, no stubs[^1]
- No `try/except` for flow control — only at system boundaries[^1]
- Google-style docstrings[^1]
- TypedDict with encode/decode and `require_*` validation[^1]
- Files under 400 lines where possible (Phase 4 god-file splits)[^1]

## Git discipline

- NEVER run `git checkout`/`clean`/`restore` without explicit user confirmation[^3]
- NEVER run any git command without explicit user permission[^3]
- Create new commits rather than amending[^3]

[^1]: user (Austin), enforced across all sessions — "no back compat shims, no thin wrappers, no fallbacks, no legacy code, no type alias, no duplicate code"
[^2]: MonkeyPatchBanRule in monorepo-guards; _test_hooks pattern verified across 3,923 tests at 100% coverage
[^3]: user (Austin), 2026-06-13 — incident destroyed uncommitted work; absolute rule since
