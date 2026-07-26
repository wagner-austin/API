---
title: Executor Rejection Silent Loops
tags: [architecture, bug, executor, planner]
related:
  - "[[self-observing-architecture]]"
  - "[[combat-chase-bug]]"
  - "[[bot-behavior-contract]]"
  - "[[mine-mechanics]]"
source_paths:
  - "src/tankpit_bot/bot"
  - "runs/bot"
source_git_blobs:
  "src/tankpit_bot/bot": "012f907aad43446928b1f85f0e628a390d00fdb4"
fact_checked: "2026-07-17"
confidence: high
hubs: [architecture]
---

# Executor Rejection Silent Loops

## Symptom

Executor client-side validation rejects a planner-produced command every tick with an `emit_ai("rejecting ...")` log line and no wire dispatch. The bot appears frozen from the outside — no server exchanges, no state change — while the AI log fills silently. The 2026-07-06 20:47:31 deadlock (26 s of `_is_valid_shoot` self-rejections) was one instance; Phase 0 of the [[self-observing-architecture]] retired that specific check. The **structural pattern** that produced it is still live and can fire from at least three other rejection sites.

## Structural cause

Two independent mechanisms combine to make executor rejections invisible to the planner:[^1][^2]

1. **AI state rolls back on rejection.** `tick_loop.py:490-491` persists `bot._ai_state` only when `command_sent` is True. When `execute` returns False, the updated AI state (including `blocked_combat_targets`, `resource_target_kind`, `last_shot_target_id`) is thrown away. Next tick's planner runs against the same base state and re-produces the same or similar command.[^1]
2. **Rejections are not wired to the block/replan machinery.** `mark_move_target_failed` is called from `tick_loop_actions.py` (walk failures and wire 0x52 code-0/1 rejections; the `game_log_feedback.py` DOM call site was deleted 2026-07-19 when the channel went witness-only) and `completions.py:232` (completion timeouts), but never from any of the executor's nine `emit_ai("rejecting ...")` sites. The `is_move_target_failed` guard at `combat_strategy.py:440` that would route into `block_combat_target_and_replan` is unreachable from executor rejections.[^2]

Together: rejection → no state persisted → no tile marked failed → same tick next iteration → same rejection. The loop breaks only when world state itself changes (target moves, mine detonates, container regenerates).[^1][^2]

## Resolution (2026-07-20): the mine class is dead by construction

The latent instance below FIRED on 2026-07-20 17:16 exactly as
predicted — the dot-hop selector proposed a mined fuel dot at
(37,153) and `_is_valid_move_destination` discarded the teleport 23
consecutive ticks until session end. The fix was neither A nor B
(feedback wiring) but a root cut, per the user's ruling ("not anti
loop, but something that addresses the root of the issue" — quoted
with its receipt in [[terrain-composition]], which documents the
loop capture, the cut, and the verification soak):

- **Hostile mines are composed into the decision terrain**
  (`compose_decision_terrain` → `FerryAwareTerrain.hostile_mine_keys`),
  so every passability consumer — pathfinding, reachability,
  selectors, clamps, the surface-route gate — shares ONE walkability
  answer. The parallel `blocked_mines` parameter threading was
  deleted end-to-end (pathfinding, reachability, equipment_search,
  movement, ferry clamp, tick_loop_actions, action_lab).
- **`_is_valid_move_destination` is deleted.** For teleports it was
  wrong physics — the server displaces off mined tiles on landing
  (see [[teleport-mechanics]] Placement), so instance #1 below was a
  veto of a perfectly safe command. For walks it is unreachable: the
  planner cannot produce a mined destination from a terrain view in
  which mined tiles do not exist.
- The `discarded_hostile_mine` outcome labels are gone from the
  ledger. Instances #2 and #3 (stale combat anchor, pickup race) and
  their discard labels remain — they guard planner cross-tick state,
  a separate audit.

Side effect: `tick_loop_actions`' stall-clearing reachability checks
now use the composed view too — they were previously raw static
terrain, neither ferry- nor mine-aware ([[terrain-composition]] §The
cut, item 4).

## Known live instances (as of 2026-07-17)

### 1. Combat teleport onto a hostile-mine tile — RESOLVED 2026-07-20 (veto deleted; the teleport is safe, server displaces)

`choose_combat_landing_tile` (`combat_landing.py:46-69`) returns the enemy's exact coordinates by design — server displaces on landing. Line 68 explicitly discards `world`, `self_state`, and `terrain`, so it never consults `hostile_mines`. If an enemy stands on a same-team mine (defensive-minefield scenario — friendly mines are passable per [[mine-mechanics]]), the enemy's coord is a hostile-mine tile from our POV. `_is_valid_move_destination` (`executor.py:360`) rejects every tick until the enemy moves or the mine detonates.[^3]

### 2. Combat target stale (position drift)

`_tracked_combat_target` (`executor.py:392`) still carries a position-match check: `if tank["x"] != ai_state["combat_target_x"] or tank["y"] != ai_state["combat_target_y"]: return None`. Same shape as the Phase 0 shoot bug, but on the teleport path. If the enemy moves between planner-decide and executor-dispatch, `_is_valid_teleport` (`executor.py:452`) rejects with "combat target is stale". Usually recovers next tick when the planner reads fresh position, but silent per-tick self-rejection until it does.[^4]

### 3. Container pickup race

`_is_valid_pickup` rejections at `executor.py:307/:314/:326/:333` fire when the world-state container view disagrees between planner and executor (container consumed, kind mismatch). Same silent-loop shape as #1 if the disagreement persists across ticks — e.g., a stale planner cache or a wire-order race where the pickup response arrives after the planner already re-picked the same container.[^6]

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
[^6]: rejection sites verified by direct read of `executor.py` at fact-check 2026-07-17 (line refs in the paragraph); the whole validator family was deleted 2026-07-21 by commit `59fce8e1` ("bot: executor is pure dispatch"), so these line refs are historical — the deletion diff in git history is the current receipt.
[^7]: commit `59fce8e1` (2026-07-21) — the deletion diff carries every symbol named here; zero-discard archive confirmation re-derivable by scanning `runs/bot/*.events.jsonl` for `discarded_` outcomes after 2026-07-20; deletion recorded in the wiki-log entry "[2026-07-21] refactor | Executor is pure dispatch"; erratum recorded in "[2026-07-21] erratum + finding | The pursuit volley already exists — and has been firing all along".


## Resolution 2026-07-21 — the class is CLOSED: executor is pure dispatch

Instances #2 and #3 died the same death as #1: deletion, not feedback.
The unreachability proof: the tick is SYNCHRONOUS (drain → decide →
execute on one thread — `world_sync.drain_messages` is the only world
mutation point), so nothing can change between planner-decide and
dispatch; resource locks are normalized at `DecideCtx` construction
with the SAME pursuability predicate selection uses (a surviving lock
guarantees its container exists); combat releases reset the anchor to
−1 before any veto could see it; and the teleport source check
guarded a container source ("world_state") that no creation site can
produce since the 0x4C map-container path was deleted. Archive
confirmation: zero validator discards in any run since the mine fix.[^7]

Deleted: `_is_valid_shoot` / `_is_valid_pickup` / `_is_valid_teleport`
/ `_is_dispatchable` and their `_tracked_*` helpers, the six
`emit_*_discarded_*` ledger emitters, six `discarded_*` outcome
literals, and the ledger-audit discard analytics. The AI-state
persistence gate survives for genuine CDP dispatch failures only.[^7]

ERRATUM (same day): the first version of this entry (and commit
59fce8e1's message) claimed the shoot veto had been "silently
blocking the reroute-TTL pursuit window." Wrong — `remove_tank` is
deliberately a NO-OP (2026-06-22: 0x58 fires on tracking churn, not
just death; 0x41 is the only authoritative death signal), so the
registry keeps departed tanks and `_tracked_tank` always passed.
The pursuit volley ([[shoot-event-format]]) was never blocked; it
has been firing all along (`find_locked_target_pursuit` → homing at
the frozen position → server reroutes → hits confirm → first
genuine miss at TTL expiry trips the stationary-miss classifier →
release). Soaks 2026-07-21 show 4–12 pursuit shots per run. The
veto was dead code either way — the deletion stands unchanged.
