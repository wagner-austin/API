---
title: Executor Rejection Silent Loops
tags: [architecture, bug, executor, planner]
related: [[self-observing-architecture]], [[combat-chase-bug]], [[bot-behavior-contract]], [[mine-mechanics]]
sources: [see footnotes]
fact_checked: 2026-07-17
confidence: high
---

# Executor Rejection Silent Loops

## Symptom

Executor client-side validation rejects a planner-produced command every tick with an `emit_ai("rejecting ...")` log line and no wire dispatch. The bot appears frozen from the outside — no server exchanges, no state change — while the AI log fills silently. The 2026-07-06 20:47:31 deadlock (26 s of `_is_valid_shoot` self-rejections) was one instance; Phase 0 of the [[self-observing-architecture]] retired that specific check. The **structural pattern** that produced it is still live and can fire from at least three other rejection sites.

## Structural cause

Two independent mechanisms combine to make executor rejections invisible to the planner:

1. **AI state rolls back on rejection.** `tick_loop.py:490-491` persists `bot._ai_state` only when `command_sent` is True. When `execute` returns False, the updated AI state (including `blocked_combat_targets`, `resource_target_kind`, `last_shot_target_id`) is thrown away. Next tick's planner runs against the same base state and re-produces the same or similar command.[^1]
2. **Rejections are not wired to the block/replan machinery.** `mark_move_target_failed` is called from `tick_loop_actions.py` (walk failures and wire 0x52 code-0/1 rejections; the `game_log_feedback.py` DOM call site was deleted 2026-07-19 when the channel went witness-only) and `completions.py:232` (completion timeouts), but never from any of the executor's nine `emit_ai("rejecting ...")` sites. The `is_move_target_failed` guard at `combat_strategy.py:440` that would route into `block_combat_target_and_replan` is unreachable from executor rejections.[^2]

Together: rejection → no state persisted → no tile marked failed → same tick next iteration → same rejection. The loop breaks only when world state itself changes (target moves, mine detonates, container regenerates).

## Known live instances (as of 2026-07-17)

### 1. Combat teleport onto a hostile-mine tile

`choose_combat_landing_tile` (`combat_landing.py:46-69`) returns the enemy's exact coordinates by design — server displaces on landing. Line 68 explicitly discards `world`, `self_state`, and `terrain`, so it never consults `hostile_mines`. If an enemy stands on a same-team mine (defensive-minefield scenario — friendly mines are passable per [[mine-mechanics]]), the enemy's coord is a hostile-mine tile from our POV. `_is_valid_move_destination` (`executor.py:360`) rejects every tick until the enemy moves or the mine detonates.[^3]

### 2. Combat target stale (position drift)

`_tracked_combat_target` (`executor.py:392`) still carries a position-match check: `if tank["x"] != ai_state["combat_target_x"] or tank["y"] != ai_state["combat_target_y"]: return None`. Same shape as the Phase 0 shoot bug, but on the teleport path. If the enemy moves between planner-decide and executor-dispatch, `_is_valid_teleport` (`executor.py:452`) rejects with "combat target is stale". Usually recovers next tick when the planner reads fresh position, but silent per-tick self-rejection until it does.[^4]

### 3. Container pickup race

`_is_valid_pickup` rejections at `executor.py:307/:314/:326/:333` fire when the world-state container view disagrees between planner and executor (container consumed, kind mismatch). Same silent-loop shape as #1 if the disagreement persists across ticks — e.g., a stale planner cache or a wire-order race where the pickup response arrives after the planner already re-picked the same container.

## Latent (not a loop today but same shape)

`find_teleport_landing_tile` (`equipment_search.py:35-73`) accepts a `blocked_mines` parameter and deletes it at line 62. The parameter is dead code across five callers. Container-teleport landing therefore never consults mines; if a container ever ended up on a mine tile the executor's mine rejection would silently loop the pickup pursuit.[^5]

## Fix options

- **A.** Wire `_is_valid_move_destination` rejection to `mark_move_target_failed(target_x, target_y, now)`. Plugs the mine loop (and every future tile-rejection loop) into the existing block-and-replan pathway.
- **B.** Same for the two `_is_valid_teleport` rejection paths.
- **C.** Delete the dead `blocked_mines` parameter on `find_teleport_landing_tile` (or wire it through).
- **D.** Add a per-target-id block for `_is_valid_shoot` rejections — kills the 20:47:31 class fully.
- **E.** Remove the position-match in `_tracked_combat_target` (matches Phase 0 philosophy: aim is a hint, `target_id` is the truth channel).

The structural fix is Phase 1 of [[self-observing-architecture]] — every rejection becomes a contract violation with the planner's `Decision` correlated, so silent loops are impossible by construction. A/B are the minimal wiring fix that closes the currently-known instances.

[^1]: `src/tankpit_bot/bot/tick_loop.py:490-491` — `if command_sent: bot._ai_state = decision["updated_ai_state"]`; when `execute` returns False the updated AI state is discarded and the next tick re-plans from the same base state.
[^2]: grep of `mark_move_target_failed` across `src/`, 2026-07-17 (call-site list re-verified 2026-07-19 after the game-log teardown): `tick_loop_actions.py` walk-failure and 0x52-rejection sites plus `completions.py:232` — none in `executor.py`. The former `game_log_feedback.py:93` site is gone with the module.
[^3]: `choose_combat_landing_tile` at `src/tankpit_bot/bot/ai/combat_landing.py:46-69` (line 68: `del world, self_state, terrain`); executor rejection at `src/tankpit_bot/bot/executor.py:360`; commit `4d11980b` narrowed the mine check from `world["mines"]` to `hostile_mines(world)` for same-team passability but did not consider planner-executor consistency on hostile tiles.
[^4]: `_tracked_combat_target` at `src/tankpit_bot/bot/executor.py:371-394`, position-match at line 392; called from `_is_valid_teleport` at line 449.
[^5]: `find_teleport_landing_tile` at `src/tankpit_bot/bot/ai/equipment_search.py:35-73`; line 62: `del start_x, start_y, blocked_mines`; five callers pass the arg (`bot/ai/movement.py:262/:391/:451/:538`, `bot/ai/equipment_search.py:146/:212/:262/:304/:380`).
