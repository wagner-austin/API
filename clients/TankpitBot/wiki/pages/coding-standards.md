---
title: Coding Standards
tags: [architecture, testing, quality]
related:
  - "[[inheritance-chain]]"
source_paths:
  - "src/tankpit_bot"
  - "tests"
source_git_blobs:
  "src/tankpit_bot": "84cbcda90e100889c9d16e40960b053388c0a9bb"
  "tests": "630bf9c4018a0f345bded2bff0f1a336648a870e"
fact_checked: "2026-08-14"
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
- **A test that swaps a `_test_hooks` attribute must restore it, and the guard now checks.** A swap left in place leaks to every later test on the same xdist worker. The autouse `_restore_hooks` fixture resets 16 attrs centrally; anything outside that list must be put back under a recognised guard — a `finally` body, a post-`yield` fixture, a `teardown_*` method, or an ancestor `conftest.py`. `scripts/hook_restore_rules.py` fails the build otherwise, with no allowlist. Found exactly one real leak across 391 assignment sites (`remove_file`, fixed centrally); the 6,171 tests passing at 100% coverage on 2026-08-08 had never noticed it, because coverage proves a line ran and says nothing about what happens after the test ends.[^5]
- No weak assertions (`assert len(x) >= 0` etc.) — concrete assertions on specific values[^2]
- 100% coverage required (`fail_under = 100` in `pyproject.toml`)[^2]
- **No exclusions, no omissions, no exemptions, no exceptions** (user ruling 2026-08-07, verbatim). No coverage `omit` entry, no `exclude_lines`, no guard allowlist, no `# pragma`, no `noqa`, no `type: ignore`, no xfail or skip to get a gate green — and an existing one is a defect to delete, not a precedent to extend. An exempted gate measures the files someone already chose to test, not the codebase, and the exemption is self-perpetuating: `combat_probe.py` sat at 69% and its script at 0% purely because being on the list meant nobody wrote the tests, and both reached 100% through the `_test_hooks` seams every other probe already used. Code that looks untestable is missing a seam — add it.[^2]

## Code style

- No back-compat shims, no thin wrappers, no fallbacks, no legacy code[^1]
- **Machine-checked subset, and the one part that deliberately is not.** `scripts/shim_rules.py` enforces legacy vocabulary, self-named aliases (`X = X`) and renamed re-exports (`NEW = OLD` in `__all__`). Fallbacks audit clean: **0** `except ImportError`, **0** `getattr(_, _, default)` under `src/`. **Thin wrappers are NOT enforced and must not be**: an AST sweep finds 60 pure pass-throughs, but they are Protocol implementations and domain naming — `SurfaceRouteTerrain.get_terrain` must exist to satisfy `TerrainMapProtocol`, and `ProtocolCodec.encode → xor_bytes` *is* the abstraction. Telling those from a pointless alias needs intent, not syntax, so a rule would need the allowlist this project refuses. Do not "fix" those 60 sites.[^6]
- No duplicate code — keep codebase DRY[^1]
- No placeholder code, no stubs[^1]
- No `try/except` for flow control — only at system boundaries[^1]
- Google-style docstrings[^1]
- TypedDict with encode/decode and `require_*` validation — **at serialization boundaries**, resolved 2026-09-02[^7]
- **File size 400-600 lines max, src AND tests** (user ruling 2026-07-31, as recorded: "we need modular, clear separation of concerns, no monolithic files. 400 - 600 lines, including test files too" — supersedes the earlier "under 400 where possible"). Split any over-bar file you touch into cohesive modules; never grow a file already past 600. No re-export shims when splitting (the wrapper ban above applies) — move call-site imports properly. Backlog at ruling time: **40** files over 600, enumerated in the log entry; measured again 2026-08-06 it was **77**, and all 40 originals had GROWN (+6,272 lines, zero splits). The rule is now machine-checked monorepo-wide -- `FileSizeRule` in `libs/monorepo_guards` (lifted 2026-08-22 from this repo's `scripts/file_size_rules.py`, which is deleted; board task 21e173d7 cleared the 137-file cross-package backlog first) -- with no allowlist and no baseline, so a file crossing the line fails the gate on the commit that crosses it, in every package.[^4]

## Verification discipline (2026-08-20, after the gatherer livelock)

Born from the planner/veto feedback-gap class ([[fleet-coordination]]):
two behavior holes and two analyzer holes, every one an instance of an
assumption that had already been falsified somewhere else in the tree.

- **Falsification sweep:** when a live run falsifies an assumption,
  grep the whole repo for every other site encoding it — the SAME
  session, before the fix commits. Precedent for the cost of not
  doing it: the kill-attribution split landed in the live registry
  2026-08-14 and the analyzer's copy of the same assumption survived
  six more days, until `test_analyzer_consistency.py` existed to
  compare them. Its twin: the fuel lock's capacity gate (2026-07-06)
  whose equipment sibling, 60 lines up in the same file, stayed
  ungated for six weeks.
- **Scenario matrix before live:** a new capability's sim soak runs
  its new cells (role × inventory seed × fuel seed), not just the
  default scenario, BEFORE the first live run. The gatherer livelock
  was reachable only in the gatherer × full-stock cell — a cell no
  soak had ever exercised, so the first live run was the first
  execution of the code path.
- **Liveness is an instrumented dimension:** the ledger's
  `zero_dispatch_streaks` counter emits a `liveness_stall` diagnostic
  at `LIVENESS_STALL_STREAK` consecutive zero-dispatch replans of one
  kind, and the issue report scans the same signal post-hoc.
  Thresholds are empirical (459-run archive sweep: healthy ceiling 7,
  the one recorded livelock 93) — re-measure before changing them.
- **Contracts state invariants, not caller snapshots** (2026-08-20,
  the scope-pending radar drop): "nothing downstream correlates an
  attempt against it" was true of scope's callers on 08-01 and
  written as a property of the command; eleven days later a new
  consumer (the harvest frame shift) inherited it as a guarantee and
  bought half of all scan stalls ever recorded. Phrase a module's
  assumptions as invariants of the command/world; when you add a NEW
  CONSUMER to a module, its stated assumptions are claims to
  re-verify, not guarantees to trust. Corollary: fire-and-forget
  dispatch is reserved for commands with NO dependent server state —
  after the scope promotion, chat is its only member, by contract
  (the flood mute forbids waiting on chat, and nothing reads its
  outcome).

## Git discipline

- NEVER run `git checkout`/`clean`/`restore` without explicit user confirmation[^3]
- NEVER run any git command without explicit user permission[^3]
- Create new commits rather than amending[^3]

[^1]: Standing user (Austin) rules, summarized in `CLAUDE.md:24-28` (§ Coding standards), which points here for the full list. The mechanically enforced subset is in `pyproject.toml`: the three banned imports at `:108-111` (`typing.Any`, `typing.cast`, `typing.TypeAlias`, each with its rejection message) and mypy strict mode at `:74-91` — `strict = true` plus `disallow_any_unimported`, `disallow_any_expr`, `disallow_any_decorated`, `disallow_any_explicit`, `disallow_any_generics`, over `files = ["src", "tests", "scripts"]`. Re-verified 2026-08-07 against the current tree, now counting `scripts/` as well as `src/` + `tests/`: zero `TYPE_CHECKING`, zero `TypeAlias`, zero `.pyi`, zero `noqa`, zero `unittest.mock`, and zero real `type: ignore` pragmas. The grep's only four non-binary `type: ignore` hits are prose restating the ban -- `src/tankpit_bot/_test_hooks/__init__.py:26`, `tests/test_state_decoder.py:3`, `scripts/_test_hooks.py:6` -- plus the guard's own fixture string at `tests/test_guard_checks.py:271`. **Corrected 2026-08-10:** this footnote previously listed "no back-compat shims" among the rules with *no enforcing artifact*. `scripts/shim_rules.py` enforces it (legacy vocabulary, `X = X`, renamed re-exports), so that was wrong. The style rules genuinely without an enforcing artifact are: no thin wrappers (deliberately unenforceable — see [^6]), no duplicate code, no placeholders, and Google-style docstrings. Fallbacks have no rule but audit clean at 0 occurrences.
[^2]: `MonkeyPatchBanRule` is defined in the api monorepo at `libs/monorepo_guards/src/monorepo_guards/monkey_patch_rules.py:79` (`name = "monkey-patch-ban"` at `:82`) and registered in the orchestrator's rule list at `libs/monorepo_guards/src/monorepo_guards/orchestrator.py:71`; the companion `mock-ban` rule is at `mock_rules.py:25`. TankpitBot runs them through `scripts/guard.py`, invoked by `make lint` at `Makefile:77` (`poetry run python -m scripts.guard`) and thus by `make check`. Paths are relative to the api monorepo root, one level above this project's own tree. `fail_under = 100` with `branch = true` in `[tool.coverage.run]`. The omit list that once exempted four probe modules is **deleted** (2026-08-07): an exempted gate measures nothing, so coverage now sees every file. The 5,631-test gate is the one recorded at `wiki/log.md:2444`; later entries report 5,835 (`log.md:2805`). `src/tankpit_bot/_test_hooks/` holds exactly 8 domain submodules (`bot`, `browser`, `cdp`, `env`, `fs`, `playwright_loader`, `runtime`, `terrain`), counted 2026-08-05. Re-verified the same day: zero `unittest.mock` / `import mock` occurrences under `src/` or `tests/`.
[^3]: Standing user (Austin) rule. The enforcing artifact in this workspace is the permission allowlist at `.claude/settings.local.json:6-7`, whose only git entries are `Bash(git add:*)` and `Bash(git commit:*)` — `checkout`, `clean`, and `restore` are absent, so each invocation requires explicit confirmation. **The 2026-06-13 incident this rule is attributed to is not recorded anywhere in this workspace**: `wiki/log.md` opens at 2026-06-16 (`log.md:7`), three days later, and no entry mentions destroyed or uncommitted work. The rule is real and enforced; the originating date and the claim that it "destroyed uncommitted work" are uncorroborated here and are carried as attribution only. (`.claude/` is gitignored at the monorepo's `.gitignore:78`, so this file cannot be pinned in `source_git_blobs`.)
[^4]: `wiki/log.md:2426-2428` — the 2026-07-31 ruling entry, which records the quotation. `log.md:2437` enumerates the backlog: **40** files over 600 lines, counted from that line's own list 2026-08-05. **Corrected 2026-08-05:** this page previously said 45 and rendered the quotation with "sepraration"; the log — written the same day as the ruling — has 40 and "separation". No other over-600 backlog entry exists in the log, so the 45 had no source.
[^5]: `scripts/hook_restore_rules.py`, wired into the local rule sum at `scripts/guard.py:139` and therefore run by `make lint` (`Makefile:77`) and `make check`. The autouse fixture is `tests/conftest.py:127` (`_restore_hooks`), resetting 16 attrs as counted by AST 2026-08-08: `append_text`, `find_best_static_byte`, `force_exit`, `get_argv`, `get_current_time_ms`, `get_env`, `get_sync_playwright`, `install_signal_handlers`, `load_terrain_map`, `path_exists`, `process_received_message_hook`, `read_text`, `remove_file`, `start_watchdog`, `sync_playwright`, `write_text`. **Re-counted 2026-08-12: still 16, but no longer all on one object.** Fifteen are `_test_hooks` attributes; `process_received_message_hook` moved to `replay_test_hooks` (`tests/conftest.py:165`) because, per the inline comment, "it names WorldService, which _test_hooks cannot ([[session-state-deglobalisation]] step 8)". The analysis seams are no longer restored attribute-by-attribute here either — the fixture calls `analysis_test_hooks.reset_analysis_hooks()` (`src/tankpit_bot/analysis/_test_hooks.py:101`) instead. The rule's own suite is `tests/scripts/test_hook_restore_rules.py` (13 tests, 100% of 87 statements / 56 branches), and it ships a paired negative control — it must fire on a planted unrestored swap *and* stay silent on all four legitimate restore shapes, because a check that has never failed on a known-bad input is not evidence. The leak it was written for was `tests/bot/test_tick_loop_lifecycle.py:55`, which set `remove_file` inline with no restore; `path_exists` on the adjacent line was already safe because it was in the reset list. **Scope limit, stated honestly:** the rule covers `_test_hooks` attribute swaps only. The wider leaked-process-state class — `logging` handlers, `sys.modules`, in-place mutation of module containers — was swept by hand the same day and found clean (all three `addHandler` sites paired with `removeHandler` in a `finally`, both `sys.modules` mutations restored), but it is **not** machine-enforced, so that half is true as of 2026-08-08 rather than guaranteed.
[^7]: **Scope resolved 2026-09-02.** Read literally, "every TypedDict needs encode/decode" would add codecs to ~30 in-process structs in `sim/` alone against the 3 modules that have them (`world.py`, `commands.py`, `fuel_pickup.py`), and every one would be reachable only from its own round-trip test. The rule's SAFETY property is already enforced structurally, and by three independent artifacts: the `json` guard rule bans `json.loads` outside `platform_core.json_utils`, so all parsing returns `JSONValue`; the `typing` guard rule bans `Any`, `cast` and `TypeAlias`; and mypy runs `disallow_any_expr`. Together those make an unvalidated `JSONValue -> TypedDict` conversion **unwritable** — the only way across is `require_*` / `narrow_json_to_*`. So the codec requirement binds where a type actually crosses a boundary (an artifact on disk, the wire, a config file), and there it is mandatory and mechanically unavoidable; for a struct that never leaves the process it adds code kept alive by its own test, which the no-placeholder and DRY rules exclude. The `fuel_pickup` pair written 2026-09-01 under the literal reading was deleted the same week as its only caller — see [[capture-differ]] for the analysis modules where the boundary version is genuinely load-bearing (a diff is written to an artifact and read back).
[^6]: Audited 2026-08-10 against the current tree. `scripts/shim_rules.py` states the governing principle in its own docstring: "A rule that needs a human to adjudicate would need an allowlist, and an allowlist is the thing this project refuses." The 60 pure pass-throughs are functions whose body is exactly `return other(<own params>)`; 11 are methods and the rest module-level. Representative legitimate cases: `bot/ai/ferry.py:276` and `:360` (`SurfaceRouteTerrain` satisfying `TerrainMapProtocol`), `bot/command_service.py:64`, `browser/cdp_service.py:113`, `protocol/codec.py:146` and `:161`, `terrain.py:151`. `action_lab/equipment_probe.py`'s nine `_x() -> x_for_probe()` methods look mechanical but bind `self`-derived arguments, so they are not pure aliases. The banned-escape counts (0 `TypeAlias` / `Any` / `cast` / `noqa`) exclude docstring prose restating the ban and `tests/test_guard_checks.py:270-273`, which is the guard's own negative-control fixture — a known-bad input proving the rule can fail.
