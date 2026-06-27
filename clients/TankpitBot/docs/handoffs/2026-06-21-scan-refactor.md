# Scan System Refactor — Handoff Brief

**Audience:** A second AI engineer who has not been part of the prior conversation.
**Date:** 2026-06-21
**Status:** COMPLETED 2026-06-21 — see `wiki/log.md` for the final delta. `make check` green at 100% coverage. The notes below are kept as the original brief for traceability.
**Owner of remaining work:** the next session.

## Read these first

1. `CLAUDE.md` (project root) — strict coding standards: no Any/cast/type:ignore/.pyi/noqa, `_test_hooks.py` DI exclusively, 100% statement + branch coverage, no fallbacks/back-compat/legacy code, no mocks in tests.
2. `wiki/index.md` — wiki is the single source of truth for game mechanics, wire protocol, architecture decisions.
3. `wiki/pages/radar-mechanics.md` — the radar facts you're implementing against (updated 2026-06-21).
4. `wiki/pages/bot-behavior-contract.md` §3.2 and §3.4 — the contract rows your refactor must satisfy.

## Hard rules

- **No back-compat shims.** Delete the old paths outright; do not leave `# legacy` blocks.
- **No divergent code paths.** One scan/forage path. One coverage record. Same code regardless of extras count.
- **No mocks.** Use `_test_hooks` DI for any fakes in tests.
- **100% statement + branch coverage** in `make check`. Add tests as you go.
- **Don't run live `make bot`** — the user will run it. You verify via `make check` and unit/scenario tests only.

## Game-mechanic ground truth (confirmed by user 2026-06-21)

Read [[radar-mechanics]] for the full version. The facts you need:

1. **Radar reveals fuel containers, equipment containers, and mines only.** These entities are hidden by default on spawn; radar reveals them. Radar **never** reveals enemies — enemies arrive via the normal wire stream.
2. **One wire command (`CMD_RADAR` 0x66), two server-side resolutions.** When `extras > 0` the server consumes one extra and reveals the entire viewport. When `extras == 0` it reveals a 5×5 chebyshev-radius-2 block centered on the tank.
3. **Both scans are clamped to viewport bounds.** Free radar at a viewport corner reveals fewer than 25 tiles — the intersection of `(tank±2)` with the viewport. Coverage tracking must mark exactly the revealed tiles.
4. **Viewport shifting is OFF.** Walking never moves the viewport. The only way to a new viewport is teleport. So once every tile in the current viewport is scanned, the bot's only forward action is `teleport`.
5. **Pickup is server-routed.** `pickup_equipment(x, y)` / `pickup_fuel(x, y)` dispatched once from within the viewport tells the server to walk the tank to the tile and complete the pickup. Bot should NOT issue per-tile `move` commands toward a pickup target.

## Current state (where I left off)

- `bot/ai/scan_coverage.py` has been **rewritten** to tile-level primitives. Cell-based functions (`FORAGE_CELL_SIZE`, `cell_center`, `is_cell_covered`, `local_scan_cell_key`, `record_local_scan`, `record_viewport_scan`) are deleted. New API:
  - `tile_key(x, y)`
  - `is_tile_covered(tiles, x, y, now_ms)`
  - `record_tile_scan(tiles, scanned_tiles, now_ms)` — accepts an iterable of `(x, y)` tuples
  - `viewport_tiles(left, top, right, bottom)`
  - `free_radar_revealed_tiles(tank_x, tank_y, viewport_left, viewport_top, viewport_right, viewport_bottom)` — returns the radius-2 block ∩ viewport
  - `is_viewport_fully_covered(tiles, left, top, right, bottom, now_ms)`
  - `select_best_free_radar_position(tiles, tank_x, tank_y, left, top, right, bottom, now_ms)` — picks the destination whose next free radar reveals the most uncovered ground (retargeted 2026-06-22; previously `nearest_uncovered_tile_in_viewport`, which picked the closest unscanned tile -- inefficient because the next radar mostly re-covered already-scanned ground)
- `bot/ai/types.py` `AIStateDict` has been renamed `local_scan_cells` → `local_scan_tiles`. Factory updated.
- `bot/ai/context.py` `mark_scan_dispatched` has been updated to call the new tile-level helpers. When `extras > 0` it records every viewport tile; otherwise it records the free-radar footprint clipped to viewport.
- `bot/ai/forage.py` has been rewritten to operate on tiles (one path, no extras-gated branch). `select_forage_target` returns the nearest unscanned tile in the current viewport.

**`make check` is broken right now.** The following still reference deleted symbols or the renamed field — fix them as part of this refactor:

- `bot/ai/recover_equipment_mode.py` still has `from tankpit_bot.bot.ai.scan_coverage import is_cell_covered` (line ~32) and the `_plan_equipment_sense_or_search` body still has the `extras == 0` gate and the `radar_for_equipment` branch. Delete both. After deletion, the function should just call `plan_forage_search` and fall through to `_plan_equipment_search` (teleport hop) when forage returns None.
- Anywhere that reads `ai_state["local_scan_cells"]` — search and replace to `local_scan_tiles`.
- `bot/ai/recover_fuel_mode.py` `_plan_fuel_sense_or_search` (line ~595) has a `radar_for_fuel` branch that fires `make_radar_command()` without tile-coverage tracking. Same fix as equipment: replace with a call to `plan_forage_search`, fall through to fuel teleport hop on None. The forager doesn't care whether the caller is equipment or fuel recovery — it just keeps scanning unscanned viewport tiles.
- `bot/ai/hunt_mode.py` `search_for_enemies` lines 54-75 dispatch radar with reason `"radar to search for enemies"`. **Delete this branch outright.** Radar does not reveal enemies. After deletion, the function falls through to `map_open` or viewport-edge walking (both correct). The radar dispatch in `_decide_hunt_close` (line ~184) IS legitimate — it scans for mines/containers around the engagement tile — but the comment/reason string says "scanning viewport first" which is fine; just keep it.

## Files you will touch

- `src/tankpit_bot/bot/ai/recover_equipment_mode.py` — delete `extras == 0` gate, delete `radar_for_equipment` branch, drop the `is_cell_covered` import. Resulting `_plan_equipment_sense_or_search` is ~10 lines.
- `src/tankpit_bot/bot/ai/recover_fuel_mode.py` — delete `radar_for_fuel` branch in `_plan_fuel_sense_or_search`. Add a call to `plan_forage_search` for fuel mode. (You may need to thread mode/reason through `plan_forage_search` or accept that fuel recovery shares the equipment reason — pick whichever stays simplest.)
- `src/tankpit_bot/bot/ai/hunt_mode.py` — delete the enemy-search radar branch in `search_for_enemies`.
- Anywhere in `src/` that references `local_scan_cells` — rename to `local_scan_tiles`. Use ripgrep / Grep to find all callers.
- Tests — see "Test surface" below.

## Test surface (large)

The cell-based code had a lot of test coverage. Migrate or rewrite each:

- `tests/bot/ai/test_recover_equipment_mode.py`
  - `test_try_search_critical_equipment_returns_radar_when_scan_is_needed` — assertions about `cmd_type == "radar"` and `reason == "radar_for_equipment"` need to become `reason == "forage_radar"` (or whatever you settle on).
  - `test_try_search_critical_equipment_uses_regular_radar_when_extra_is_empty` — was testing the extras=0 forage path. Should now be redundant with the unified path; either delete or convert to a generic forage radar assertion.
  - `test_try_search_critical_equipment_does_not_spam_radar_in_covered_cell` — I added this 2026-06-21 to lock the cell-coverage gate. The tile-aware forager satisfies the same contract by different mechanism; rename and rewrite to set `local_scan_tiles` covering the whole viewport and assert no radar fires.
  - `_exhausted_forage_cells(...)` helper (line 23) — rewrite as `_exhausted_forage_tiles(viewport_bounds, now_ms)` returning a tile map.
  - Several tests use `"local_scan_cells": _exhausted_forage_cells(...)` — migrate to `"local_scan_tiles"` with viewport-tile maps.
- `tests/bot/ai/test_recover_fuel_mode.py` — `radar_for_fuel` assertions become `forage_radar`. Any tests of the no-coverage radar fire path need adjusting.
- `tests/bot/ai/test_hunt_mode.py` — any test asserting `reason == "radar_for_enemies"` or similar should be deleted; HUNT does not fire radar to search for enemies.
- `tests/bot/ai/test_forage.py` (if it exists) or any forager tests — rewrite for tile semantics. `select_forage_cell_target` is gone; use `select_forage_target` returning a tile.
- `tests/bot/ai/test_scan_coverage.py` (if it exists) — rewrite all cell-based tests as tile-based.
- `tests/replay/test_real_session_regressions.py` — replay tests assert specific command counts. After this refactor, equipment/fuel recovery will fire fewer radars and walk more; the counts will shift. Run the replays, update the asserted numbers, document the shift in the test docstring.
- Any other test that constructs an `AIStateDict` literal with `local_scan_cells=...` — rename.

## Acceptance criteria

`make check` passes with:
- Zero references to `local_scan_cells`, `is_cell_covered`, `FORAGE_CELL_SIZE`, `cell_center`, `local_scan_cell_key`, `record_local_scan`, `record_viewport_scan`, `_FORAGE_SEARCH_RING_LIMIT`, `_nearest_uncovered_in_ring`, `select_forage_cell_target`, or any other cell-grid symbol in `src/` or `tests/`.
- Zero references to the deleted radar reasons `"radar_for_equipment"`, `"radar_for_fuel"`, `"radar_for_enemies"` in `src/`.
- 100% statement + branch coverage.
- 4000+ existing tests still pass.

Live verification (NOT your job; user will run): `make bot TANKPIT_BOT_SESSION_SECONDS=90` should show:
- Bot does not fire radar in HUNT mode (radar count = 0 unless `_decide_hunt_close` scan-on-landing fires).
- Bot does not spam the same radar at the same position — once a viewport tile becomes covered it stays covered (TTL).
- Bot walks between scans within a viewport, then teleports when fully covered.

## Anti-patterns to avoid (would each be wrong)

- Adding a `mode` enum to `plan_forage_search` so equipment and fuel call different branches inside — that's reintroducing divergence. Both modes should call the same function with different reasons/scores; the forager itself should be agnostic.
- Trying to preserve "free radar marks just the bot's cell" as a backward-compat behavior. Delete it.
- Adding helper `make_radar_for_enemies()` or similar. Radar is one command; whoever calls `make_radar_command()` must be scanning for hidden entities (containers/mines), never enemies.
- Keeping `local_scan_cells` aliased to `local_scan_tiles` for "smooth migration". Delete the old name.

## Things you can skip

- `tests/scenarios/` — the new `BotScenario` harness exists but the failure-mode scenarios were never written. You do not need to add scenario tests for this refactor; existing unit + replay tests are sufficient.
- The wiki has already been updated with the relevant game-mechanic facts. You do not need to write wiki pages — just make the code match.
- The "Inventory full" code 7 issue (separate problem). `tick_loop_actions.py:44` `_ACTION_BLOCKING_COMMAND_ERRORS` does not include `SUPERVISOR_ERROR_INVENTORY_FULL`. That's a separate fix — out of scope for this refactor.
- Removing the `EnemyTrackingProbe` (action_lab) — keep it, it's working and proved its value finding the viewport-presence bug.

## Recommended order

1. **Audit imports.** Grep for every reference to the deleted symbols and `local_scan_cells`. Make a list. (5 min)
2. **Fix `recover_equipment_mode.py`** — delete the dead branches. Run `make check`; expect test failures that name the dead branches. (15 min)
3. **Fix `recover_fuel_mode.py`** — same pattern. (10 min)
4. **Fix `hunt_mode.py`** — delete the enemy-search radar branch. (5 min)
5. **Rename `local_scan_cells` → `local_scan_tiles` everywhere.** This is mechanical. (10 min)
6. **Update tests.** Bulk of the work. Walk through each failing test, decide: rewrite for tile semantics or delete as obsolete. (60-120 min)
7. **Replay-test number updates** — `tests/replay/test_real_session_regressions.py` command counts will shift. Run, get the new counts, update + document. (15 min)
8. **`make check` until 100% coverage and 0 failures.** (variable)
9. **Update `wiki/log.md`** with a new entry describing the refactor result (which branches were deleted, which tests rewrote, the final command-count shift). One paragraph.

## Where to find things

- Tile-level primitives: `src/tankpit_bot/bot/ai/scan_coverage.py` (already rewritten — read it first)
- Forager: `src/tankpit_bot/bot/ai/forage.py` (already rewritten)
- Mark dispatch: `src/tankpit_bot/bot/ai/context.py::mark_scan_dispatched` (already rewritten)
- Equipment recovery owner: `src/tankpit_bot/bot/ai/recover_equipment_mode.py` (still has dead code)
- Fuel recovery owner: `src/tankpit_bot/bot/ai/recover_fuel_mode.py` (still has dead code)
- Hunt mode: `src/tankpit_bot/bot/ai/hunt_mode.py` (still has enemy-search radar branch)
- AIStateDict: `src/tankpit_bot/bot/ai/types.py` (already renamed — but check encoders/decoders if they exist)

## Open questions you might ask the user

- Does the bot need a separate `radar_for_fuel` priority that's higher than equipment when fuel is critical? My read: no, the forager is mode-agnostic. But if it has to differ, ask before adding branches.
- After a successful pickup the container disappears from world state — should the tile remain marked covered? My read: yes, the tile WAS scanned; pickup doesn't unscan it. The TTL ages it out anyway.

That's the brief. Good luck.
