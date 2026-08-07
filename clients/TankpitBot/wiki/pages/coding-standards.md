---
title: Coding Standards
tags: [architecture, testing, quality]
related:
  - "[[inheritance-chain]]"
source_paths:
  - "src/tankpit_bot"
  - "tests"
source_git_blobs:
  "src/tankpit_bot": "fee6f258a2eca770e7edb72e4a4911af56ea8cd1"
  "tests": "ecc452024fb1528bf007907381ce3306952479c2"
fact_checked: "2026-08-07"
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
- **No exclusions, no omissions, no exemptions, no exceptions** (user ruling 2026-08-07, verbatim). No coverage `omit` entry, no `exclude_lines`, no guard allowlist, no `# pragma`, no `noqa`, no `type: ignore`, no xfail or skip to get a gate green — and an existing one is a defect to delete, not a precedent to extend. An exempted gate measures the files someone already chose to test, not the codebase, and the exemption is self-perpetuating: `combat_probe.py` sat at 69% and its script at 0% purely because being on the list meant nobody wrote the tests, and both reached 100% through the `_test_hooks` seams every other probe already used. Code that looks untestable is missing a seam — add it.[^2]

## Code style

- No back-compat shims, no thin wrappers, no fallbacks, no legacy code[^1]
- No duplicate code — keep codebase DRY[^1]
- No placeholder code, no stubs[^1]
- No `try/except` for flow control — only at system boundaries[^1]
- Google-style docstrings[^1]
- TypedDict with encode/decode and `require_*` validation[^1]
- **File size 400-600 lines max, src AND tests** (user ruling 2026-07-31, as recorded: "we need modular, clear separation of concerns, no monolithic files. 400 - 600 lines, including test files too" — supersedes the earlier "under 400 where possible"). Split any over-bar file you touch into cohesive modules; never grow a file already past 600. No re-export shims when splitting (the wrapper ban above applies) — move call-site imports properly. Backlog at ruling time: **40** files over 600, enumerated in the log entry; measured again 2026-08-06 it was **77**, and all 40 originals had GROWN (+6,272 lines, zero splits). The rule is now machine-checked -- `scripts/file_size_rules.py`, wired into `scripts.guard` -- with no allowlist and no baseline, so the backlog is 0 and a file crossing the line fails the gate on the commit that crosses it.[^4]

## Git discipline

- NEVER run `git checkout`/`clean`/`restore` without explicit user confirmation[^3]
- NEVER run any git command without explicit user permission[^3]
- Create new commits rather than amending[^3]

[^1]: Standing user (Austin) rules, summarized in `CLAUDE.md:24-28` (§ Coding standards), which points here for the full list. The mechanically enforced subset is in `pyproject.toml`: the three banned imports at `:108-111` (`typing.Any`, `typing.cast`, `typing.TypeAlias`, each with its rejection message) and mypy strict mode at `:74-91` — `strict = true` plus `disallow_any_unimported`, `disallow_any_expr`, `disallow_any_decorated`, `disallow_any_explicit`, `disallow_any_generics`, over `files = ["src", "tests", "scripts"]`. Re-verified 2026-08-07 against the current tree, now counting `scripts/` as well as `src/` + `tests/`: zero `TYPE_CHECKING`, zero `TypeAlias`, zero `.pyi`, zero `noqa`, zero `unittest.mock`, and zero real `type: ignore` pragmas. The grep's only four non-binary `type: ignore` hits are prose restating the ban -- `src/tankpit_bot/_test_hooks/__init__.py:26`, `tests/test_state_decoder.py:3`, `scripts/_test_hooks.py:6` -- plus the guard's own fixture string at `tests/test_guard_checks.py:271`. The style rules in this section that are *not* machine-checked — no back-compat shims, no thin wrappers, no fallbacks, no duplicate code, no placeholders, Google-style docstrings — are user rulings held across sessions with no enforcing artifact in this repo.
[^2]: `MonkeyPatchBanRule` is defined in the api monorepo at `libs/monorepo_guards/src/monorepo_guards/monkey_patch_rules.py:79` (`name = "monkey-patch-ban"` at `:82`) and registered in the orchestrator's rule list at `libs/monorepo_guards/src/monorepo_guards/orchestrator.py:71`; the companion `mock-ban` rule is at `mock_rules.py:25`. TankpitBot runs them through `scripts/guard.py`, invoked by `make lint` at `Makefile:77` (`poetry run python -m scripts.guard`) and thus by `make check`. Paths are relative to the api monorepo root, one level above this project's own tree. `fail_under = 100` with `branch = true` in `[tool.coverage.run]`. The omit list that once exempted four probe modules is **deleted** (2026-08-07): an exempted gate measures nothing, so coverage now sees every file. The 5,631-test gate is the one recorded at `wiki/log.md:2444`; later entries report 5,835 (`log.md:2805`). `src/tankpit_bot/_test_hooks/` holds exactly 8 domain submodules (`bot`, `browser`, `cdp`, `env`, `fs`, `playwright_loader`, `runtime`, `terrain`), counted 2026-08-05. Re-verified the same day: zero `unittest.mock` / `import mock` occurrences under `src/` or `tests/`.
[^3]: Standing user (Austin) rule. The enforcing artifact in this workspace is the permission allowlist at `.claude/settings.local.json:6-7`, whose only git entries are `Bash(git add:*)` and `Bash(git commit:*)` — `checkout`, `clean`, and `restore` are absent, so each invocation requires explicit confirmation. **The 2026-06-13 incident this rule is attributed to is not recorded anywhere in this workspace**: `wiki/log.md` opens at 2026-06-16 (`log.md:7`), three days later, and no entry mentions destroyed or uncommitted work. The rule is real and enforced; the originating date and the claim that it "destroyed uncommitted work" are uncorroborated here and are carried as attribution only. (`.claude/` is gitignored at the monorepo's `.gitignore:78`, so this file cannot be pinned in `source_git_blobs`.)
[^4]: `wiki/log.md:2426-2428` — the 2026-07-31 ruling entry, which records the quotation. `log.md:2437` enumerates the backlog: **40** files over 600 lines, counted from that line's own list 2026-08-05. **Corrected 2026-08-05:** this page previously said 45 and rendered the quotation with "sepraration"; the log — written the same day as the ruling — has 40 and "separation". No other over-600 backlog entry exists in the log, so the 45 had no source.
