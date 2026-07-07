# Handoff: Rank-derived fuel capacity + radar radius, and killing the tank-full pickup loop

**Branch:** `combat-rework`
**Date authored:** 2026-07-06
**Author of this handoff:** prior AI session (mining + wiki); user reviewed the ban list at the bottom of this doc
**For:** the next AI implementing this
**Read first:** `wiki/index.md`, then `wiki/pages/game-economy.md`, `wiki/pages/radar-mechanics.md`, `wiki/pages/client-constants.md`. Everything below is scoped by the facts those pages already record; do not re-derive them.

## Ban list (user directive, non-negotiable)

The user asked for this fix explicitly with **no back-compat shims, no thin wrappers, no fallbacks, no legacy code, no type alias**. Interpret aggressively:

- No "for now, keep the old function returning 0 so callers still work" — delete callers.
- No `def fuel_capacity(rank): return capacity_from_rank(rank)` — one function, called at the call site.
- No `if learned_capacity > 0 and learned_capacity != formula: log(...)` — the formula is the truth, the watermark is deleted.
- No `FuelCapacity = int` / `Rank = int` type aliases — plain `int`.
- No `_deprecated`, no `# TODO remove`, no re-exports for old import paths — delete the symbol; fix every import site.
- No `is_fuel_at_capacity` helper for one caller — inline `ctx.fuel >= fuel_capacity(rank)` at the call site.

If a rename is worth doing, do it and fix every call site in the same commit.

## The problem being fixed

Live run 2026-07-06 18:20:55 exited `out_of_fuel` at fuel 1100 with real fuel loot on screen. Root cause chain:

1. A 30 s container freshness TTL silently dropped real in-viewport containers. **(already fixed in this branch — TTL removed)**
2. A wire 0x52 code-5 "Tank full" on `pickup_fuel` blacklisted the good fuel container. **(already fixed — code-5 now calls `learn_fuel_capacity` instead of `increment_container_failed_pickups`, see `bot/tick_loop_actions.py:177-184`)**
3. When the COLLECT cascade found nothing, it raised `SessionExitError("out_of_fuel")` even at full tank. **(already fixed — collect_mode yields to hunt at healthy fuel)**

But those three fixes together introduced a **new** loop: with blacklisting gone and TTL gone, a full-tank bot with a held fuel lock retries `pickup_fuel` every ~2 s indefinitely. Each retry draws code-5, updates `_learned_fuel_capacity`, and the lock survives to next tick — because `_continue_or_release_fuel_lock` has no capacity gate. `_select_and_pickup_fuel` has a capacity gate (`is_fuel_at_learned_capacity`), the lock path does not. Full tank, real loot, wire rejection every tick — the accidental loop-breaker (blacklisting) was removed before the real gate (capacity on the lock path) existed.

**The current working tree is a known-bad intermediate state.** Do not commit as-is.

## The fix (mechanics-first, watermark deleted)

While diagnosing the loop the prior session cracked the client's math. **Fuel capacity = 1000 + 100·rank.** Never on the wire; the client derives it in `Gc` (fuel-gauge draw). Rank IS on the wire (`self_state["rank"]`, values 0-8), so the bot knows capacity at tick 1 with no probe pickup, no game-log line, no watermark.

Verified at four ranks by user max-deposit measurements (private 1000+100, sergeant 1200+100, major 1500+100, colonel 1598+100+~2 walked). See `wiki/pages/game-economy.md` for the full table and the client-mining citations.

**Consequence:** the entire `_learned_fuel_capacity` machinery in `game_log_feedback.py` is dead. Delete it. Compute capacity from rank at the call site. This closes the pickup loop at the root because:

- Fuel selection has a real capacity gate (was already there, but keyed off a probe).
- Fuel-lock continuation gets the same gate (was missing entirely).
- Rank-ups handled automatically — no watermark to poison, no stale-read edge case, no "raise-outgrown" oscillation logic.

Same story on radar. **Built-in radar radius = 2 + floor(rank/3)** (5x5 / 7x7 / 9x9 at rank bands 0-2 / 3-5 / 6-8). Verified at 5 of 9 ranks. `REGULAR_RADAR_RADIUS = 2` in `state/viewport_geometry.py` is only correct for private and below; sergeant+ sees a wider footprint that the scan-coverage planner currently under-counts. Make it rank-derived.

## Concrete changes

### 1. New module `src/tankpit_bot/state/rank_formulas.py`

Two functions, no class, no dataclass, no config, no back-compat layer.

```python
"""Rank-derived game constants.

Derived from client mining 2026-07-06 (tpclient.js Gc/Cc/ce) plus user
measurements at 4 ranks; see wiki/pages/game-economy.md and
wiki/pages/radar-mechanics.md. Rank ranges 0..8 (recruit..general),
sourced from ``self_state["rank"]`` on the wire.
"""

from __future__ import annotations


def fuel_capacity(rank: int) -> int:
    """Return the tank's fuel capacity at the given rank.

    ``capacity = 1000 + 100 * rank`` -- the client's gauge math,
    verified at ranks 1/3/6/7 via max-deposit arithmetic
    (deposit floor = 100).
    """
    return 1000 + 100 * rank


def free_radar_radius(rank: int) -> int:
    """Return the chebyshev radius of the built-in radar at this rank.

    ``radius = 2 + rank // 3`` -- steps at sergeant (3) and major (6),
    verified at ranks 1/3/4/6/7 via manual axial reveals.
    """
    return 2 + rank // 3
```

No validation, no clamping. Rank is a wire field on `self_state`; if it's out of range that's a protocol regression, not something this module papers over. Do not add `assert 0 <= rank <= 8`.

Add `__all__` and a unit test in `tests/state/test_rank_formulas.py` that hits every rank 0-8 for both functions (18 assertions). No parametrize magic, no property tests — this is a two-line formula.

### 2. Delete `_learned_fuel_capacity` + all its machinery

File: `src/tankpit_bot/diagnostics/game_log_feedback.py`

Delete:
- module-level `_learned_fuel_capacity` state
- `get_learned_fuel_capacity`
- `is_fuel_at_learned_capacity`
- `learn_fuel_capacity`
- `_raise_outgrown_capacity`
- the call to `_raise_outgrown_capacity(world)` at the top of `register_world_feedback_from_game_log`
- the `elif text == _TANK_FULL_TEXT:` branch (the game-log "Tank full" line is now a no-op — capacity is rank-derived, so the line teaches nothing)
- the `_TANK_FULL_TEXT` constant
- `_learned_fuel_capacity = 0` from `reset_game_log_feedback` (module state is gone)
- both names from `__all__`

What stays in this module: `_last_pickup_target`, `_last_move_target`, `record_pickup_dispatch`, `record_move_dispatch`, `_consume_empty_container`, `_consume_blocked_move`, `reset_game_log_feedback`, `register_world_feedback_from_game_log`. Those handle the empty-container and blocked-move signals which are wire-orthogonal and still needed.

Update the module docstring at the top: strike the "Tank full" bullet. Capacity is no longer a learned signal.

### 3. Delete the wire-code-5 → `learn_fuel_capacity` bridge

File: `src/tankpit_bot/bot/tick_loop_actions.py` at line 177-184.

The block that runs on `kind == "collect" and error_code == _COMMAND_ERROR_TANK_FULL` currently calls `learn_fuel_capacity(bot.get_world_state())`. Delete that call and the accompanying `emit_sync` (`"tank full at (%d,%d); capacity learned, container kept"`).

What's the replacement? **Nothing.** With rank-derived capacity, `_select_and_pickup_fuel` and the fuel-lock continuation will never dispatch a `pickup_fuel` at full tank in the first place — the code-5 becomes unreachable in normal flow. If it fires anyway (race between rank-up and dispatch, protocol quirk, etc.), the right response is the same as any other `collect` rejection: `increment_container_failed_pickups` + IDLE transition. So merge the branch:

```python
if kind == "collect":
    increment_container_failed_pickups(get_world_service(), tx, ty)
    emit_sync("marked container at (%d,%d) as failed pickup", tx, ty)
if kind in ("move", "teleport"):
    ...
```

Remove the `_COMMAND_ERROR_TANK_FULL` constant if this file is its only consumer (grep first). Remove the `learn_fuel_capacity` import.

### 4. Gate fuel selection AND fuel-lock continuation on rank-derived capacity

File: `src/tankpit_bot/bot/ai/collect_mode.py`

Two call sites:

**`_select_and_pickup_fuel`** (currently at line 411, uses `is_fuel_at_learned_capacity`). Replace with:

```python
def _select_and_pickup_fuel(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    if ctx.fuel >= fuel_capacity(ctx.self_state["rank"]):
        return None
    ...  # rest unchanged
```

Import: `from tankpit_bot.state.rank_formulas import fuel_capacity` at the top. Delete the `from tankpit_bot.diagnostics.game_log_feedback import is_fuel_at_learned_capacity` line.

**`_continue_or_release_fuel_lock`** (around line 340-360). Add the capacity check at the top of the function, before the `_superior_fuel_candidate` check:

```python
def _continue_or_release_fuel_lock(
    ctx: DecideCtx,
    base_state: AIStateDict,
    locked_target: ContainerStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    if ctx.fuel >= fuel_capacity(ctx.self_state["rank"]):
        emit_ai(
            "releasing fuel lock at (%d,%d): tank at capacity",
            locked_target["x"],
            locked_target["y"],
        )
        return None, clear_resource_target(base_state)
    if _superior_fuel_candidate(ctx, locked_target) is not None:
        ...
```

**This is the actual bug fix.** Without this, the tank-full pickup loop persists even after the watermark is deleted.

### 5. Rank-derived radar radius

File: `src/tankpit_bot/state/viewport_geometry.py`

Delete `REGULAR_RADAR_RADIUS = 2` and remove it from `__all__`. Change `regular_radar_bounds` to take a rank:

```python
def regular_radar_bounds(
    center_x: int,
    center_y: int,
    rank: int,
) -> tuple[int, int, int, int]:
    """Return inclusive bounds for the built-in radar scan.

    The built-in radar radius is rank-scaled: 2 + rank // 3 chebyshev
    (5x5/7x7/9x9 at rank bands 0-2/3-5/6-8). Only an extra radar
    sweeps the whole viewport regardless of rank.
    """
    radius = free_radar_radius(rank)
    return (
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    )
```

Import `free_radar_radius` from `rank_formulas`. Do not add a default value for `rank`; make every caller pass it explicitly.

Update the sole caller at `src/tankpit_bot/sniffer/world_service.py:411`:

```python
return regular_radar_bounds(self_state["x"], self_state["y"], self_state["rank"])
```

### 6. Unify `FREE_RADAR_RADIUS` with the rank formula

File: `src/tankpit_bot/state/scan_coverage.py`

`FREE_RADAR_RADIUS = 2` (line 42) is the same concept as `REGULAR_RADAR_RADIUS`, just held by the scan-coverage planner. Delete it and thread `rank` through the four functions that use it:

- `free_radar_revealed_tiles(tank_x, tank_y, viewport_left, viewport_top, viewport_right, viewport_bottom, rank)` — add rank parameter
- `free_radar_new_coverage(scanned_tiles, tile_x, tile_y, ..., rank)` — add rank parameter
- `select_best_free_radar_position(scanned_tiles, tank_x, tank_y, ..., rank)` — add rank parameter

Each computes `radius = free_radar_radius(rank)` locally and uses it in place of the module constant. Remove `FREE_RADAR_RADIUS` from `__all__`.

Update the four call sites:
- `src/tankpit_bot/bot/ai/forage.py:150` (`free_radar_new_coverage(...)` inside `plan_forage_search`)
- `src/tankpit_bot/state/scan_coverage.py:258` (`free_radar_new_coverage` inside `select_best_free_radar_position` — pass its own `rank` arg through)
- Any other grep hits for `free_radar_revealed_tiles`, `free_radar_new_coverage`, `select_best_free_radar_position`

For `forage.py`, the rank is `ctx.self_state["rank"]`. The tests in `tests/world_state/test_scan_coverage.py` and `tests/bot/ai/test_forage*.py` need matching rank args.

### 7. Wiki + comments cleanup

Two wire-verified 5x5 assumptions in code comments need to become rank-scaled:

- `src/tankpit_bot/state/scan_coverage.py:40-41` docstring — "Built-in radar reveals a 5x5 around the tank (chebyshev radius 2, wire-verified 2026-06-12)"
- `src/tankpit_bot/state/scan_coverage.py:113-115, 177-181, 222-224, 251-253` — every "5x5" in a docstring becomes "(2·radius+1)² footprint (radius rank-derived)"
- Any comment in `bot/ai/forage.py` that says "5x5" or "25 tiles"

The wiki already reflects the fix (radar-mechanics.md, game-economy.md, client-constants.md were updated in the prior session's log entry — check `wiki/log.md`'s 2026-07-06 entries). No wiki edits needed for this code change unless a footnote points at a code path you actually rename.

## Tests to update / write / delete

### Delete outright

- `tests/diagnostics/test_game_log_feedback.py`:
  - `test_tank_full_learns_fuel_capacity`
  - `test_tank_full_without_self_state_learns_nothing`
  - `test_tank_full_with_zero_fuel_learns_nothing`
  - `test_tank_full_stale_read_never_lowers_capacity`
  - `test_observed_fuel_above_capacity_raises_belief`
  - `test_fuel_at_capacity_does_not_invalidate`
  - `test_capacity_check_skipped_without_self_state`
  - the `_learned_fuel_capacity` and `_last_pickup_target` teardown in the module fixture only clears what still exists after the delete
  - Update `test_reset_clears_targets_and_capacity` → rename to `test_reset_clears_targets`, drop the capacity assertion
  - Update `test_unrelated_lines_consume_nothing` if it asserts against `_TANK_FULL_TEXT`
- `tests/bot/test_tick_loop_coverage.py`:
  - `test_command_error_tank_full_keeps_container_and_learns_capacity` (line 739) — the new behavior is code-5 marks the container as failed like any other collect rejection. Rewrite the test to that assertion, or delete if `test_command_error_treats_generic_collect_reject_as_failed_pickup` already covers it (grep the file first).

### New

- `tests/state/test_rank_formulas.py` — 18 assertions covering ranks 0-8 for both `fuel_capacity` and `free_radar_radius`. Reference values in `wiki/pages/game-economy.md` and `wiki/pages/radar-mechanics.md`.
- `tests/bot/ai/test_collect_mode_fuel.py` — add:
  - `test_locked_fuel_released_at_rank_derived_capacity` — bot at rank 3 (sergeant), fuel 1300, held fuel lock → lock releases, `clear_resource_target` applied, no `pickup_fuel` command in the decision
  - `test_select_fuel_returns_none_at_rank_derived_capacity` — same setup, direct call to `_select_and_pickup_fuel`, expects `None`
  - Both tests must NOT touch `_learned_fuel_capacity` or `learn_fuel_capacity` (they don't exist anymore).

### Update

- `tests/bot/ai/test_collect_mode_equipment.py::test_collect_mode_skips_opportunistic_fuel_at_learned_capacity` (line 281) — rename to `test_collect_mode_skips_opportunistic_fuel_at_rank_capacity`, delete the `reset_game_log_feedback` calls, set self-state rank so `fuel_capacity(rank) == 1100`, remove imports of the deleted feedback symbols.
- Any test that constructs a `SelfStateDict` for a collect-mode scenario now needs `rank` set to the value the test wants. Most tests already set `rank=2` (corporal, capacity 1200) via `_support.make_world`; verify no test at fuel ≥ capacity accidentally gates behavior on the deleted watermark.

### Grep pass before running the suite

```
grep -rn "learn_fuel_capacity\|is_fuel_at_learned_capacity\|get_learned_fuel_capacity\|_learned_fuel_capacity\|_raise_outgrown_capacity\|REGULAR_RADAR_RADIUS\|FREE_RADAR_RADIUS" src/ tests/ scripts/
```

Should return zero hits after this change.

## Verification gate

`make check` must pass — guard + ruff + mypy + tests + 100% coverage. Then a `make run` (or the shorter `TANKPIT_BOT_SESSION_SECONDS=60 make run` diagnostic version, per `feedback_short_live_runs.md`) with the user watching to confirm the tank-full pickup loop is closed at the source. Do not commit without both.

## Non-goals (out of scope for this change)

- Deposit command support (encoder, dispatch path). The wiki now knows the wire shape (`'D'` 0x44, u16 LE amount) but shipping deposit as a bot action is a separate feature.
- The `ca = 50` radar cooldown mining. Units unconfirmed; the wiki flags this and the current in-code cooldown gate is orthogonal.
- Dual-shot and teleport exact fuel costs. Still open rows in `game-economy.md`; no code impact today.

## Where the mining came from (so future-you can trust it without re-mining)

- `wiki/pages/game-economy.md` — capacity table + deposit mechanics, citations to `tpclient.js Gc/Wb/ce` and user measurements
- `wiki/pages/radar-mechanics.md` — rank-scaled radius table, footnote 14 has the four axial measurements
- `wiki/pages/client-constants.md` — mined constants block with the `Gc`/`Cc`/`ce` function bodies
- `wiki/log.md` — 2026-07-06 entry at the bottom summarizes the whole discovery, plus prior 2026-07-06 log for what's already in-tree

Do not re-run the mining. The user did the measurements once; the values are settled.
