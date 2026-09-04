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
  "src/tankpit_bot/bot": "6816b2568371fabe3982cf4adddba153e1ad44b9"
fact_checked: "2026-08-07"
confidence: high
hubs: [architecture]
---

# Executor Rejection Silent Loops

## Symptom

Executor client-side validation rejects a planner-produced command every tick with an `emit_ai("rejecting ...")` log line and no wire dispatch. The bot appears frozen from the outside — no server exchanges, no state change — while the AI log fills silently. The 2026-07-06 20:47:31 deadlock (26 s of `_is_valid_shoot` self-rejections) was one instance; Phase 0 of the [[self-observing-architecture]] retired that specific check. The **structural pattern** that produced it is still live and can fire from at least three other rejection sites.

## Structural cause

Two independent mechanisms combine to make executor rejections invisible to the planner:[^1][^2]

1. **AI state rolls back on rejection.** `tick_body.py:172,177` persists `bot._ai_state` only when `command_sent` is True. When `execute` returns False, the updated AI state (including `blocked_combat_targets`, `resource_target_kind`, `last_shot_target_id`) is thrown away. Next tick's planner runs against the same base state and re-produces the same or similar command.[^1]
2. **Rejections are not wired to the block/replan machinery.** `mark_move_target_failed` is called from `tick_loop_actions.py:208` (walk failures), `tick_loop_command_errors.py:144` (the wire 0x52 code-0/1 rejections, split out of the tick loop; the `game_log_feedback.py` DOM call site was deleted 2026-07-19 when the channel went witness-only) and `completions.py:256` (completion timeouts), but never from any of the executor's nine `emit_ai("rejecting ...")` sites. The `is_move_target_failed` guard — now at `collect_locks.py:137,206` — that would route into `block_combat_target_and_replan` is unreachable from executor rejections.[^2]

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

> **Read this section in the past tense.** Every line number and symbol
> below is the 2026-07-17 tree, kept because it is the diagnosis the
> deletion was based on. None of it describes today's code: the
> validator family (`_tracked_combat_target`, `_is_valid_teleport`,
> `_is_valid_move_destination`, `_is_valid_pickup`) was deleted
> 2026-07-21 by `59fce8e1`, and `choose_combat_landing_tile` moved to
> `combat_landing.py:80` and now consults `world`, `self_state` and
> `terrain` rather than discarding them. See **Resolution 2026-07-21**
> at the foot of this page.[^pasttense]

### 1. Combat teleport onto a hostile-mine tile — RESOLVED 2026-07-20 (veto deleted; the teleport is safe, server displaces)

`choose_combat_landing_tile` (`combat_landing.py:46-69`) returns the enemy's exact coordinates by design — server displaces on landing. Line 68 explicitly discards `world`, `self_state`, and `terrain`, so it never consults `hostile_mines`. If an enemy stands on a same-team mine (defensive-minefield scenario — friendly mines are passable per [[mine-mechanics]]), the enemy's coord is a hostile-mine tile from our POV. `_is_valid_move_destination` (`executor.py:360`) rejects every tick until the enemy moves or the mine detonates.[^3]

### 2. Combat target stale (position drift)

`_tracked_combat_target` (`executor.py:392`) still carries a position-match check: `if tank["x"] != ai_state["combat_target_x"] or tank["y"] != ai_state["combat_target_y"]: return None`. Same shape as the Phase 0 shoot bug, but on the teleport path. If the enemy moves between planner-decide and executor-dispatch, `_is_valid_teleport` (`executor.py:452`) rejects with "combat target is stale". Usually recovers next tick when the planner reads fresh position, but silent per-tick self-rejection until it does.[^4]

### 3. Container pickup race

`_is_valid_pickup` rejections at `executor.py:307/:314/:326/:333` fire when the world-state container view disagrees between planner and executor (container consumed, kind mismatch). Same silent-loop shape as #1 if the disagreement persists across ticks — e.g., a stale planner cache or a wire-order race where the pickup response arrives after the planner already re-picked the same container.[^6]

## Latent — RESOLVED 2026-07-20 (fix option C landed with the root cut)

Formerly: `find_teleport_landing_tile` accepted a `blocked_mines`
parameter and deleted it unread, dead code across five callers, so
container-teleport landing never consulted mines[^8].

`blocked_mines` no longer exists anywhere in the tree — the end-to-end
deletion described under the 2026-07-20 resolution above took it out
along with the rest of the parallel threading. The function's signature
is now `find_teleport_landing_tile(terrain, goal_x, goal_y)`: it
receives the **composed** decision terrain, in which hostile mines are
already impassable, so the landing selector consults mines by
construction rather than through a side parameter.[^5]

## Fix options

- **A.** Wire `_is_valid_move_destination` rejection to `mark_move_target_failed(target_x, target_y, now)`. Plugs the mine loop (and every future tile-rejection loop) into the existing block-and-replan pathway.
- **B.** Same for the two `_is_valid_teleport` rejection paths.
- **C.** Delete the dead `blocked_mines` parameter on `find_teleport_landing_tile` (or wire it through). — **DONE** 2026-07-20, as part of the root cut rather than as a standalone fix.
- **D.** Add a per-target-id block for `_is_valid_shoot` rejections — kills the 20:47:31 class fully.
- **E.** Remove the position-match in `_tracked_combat_target` (matches Phase 0 philosophy: aim is a hint, `target_id` is the truth channel).

The structural fix is Phase 1 of [[self-observing-architecture]] — every rejection becomes a contract violation with the planner's `Decision` correlated, so silent loops are impossible by construction. A/B are the minimal wiring fix that closes the currently-known instances.

[^1]: `src/tankpit_bot/bot/tick_body.py:172,177` — `command_sent = executor.execute(bot, decision, snapshot)` then `if command_sent:` persists `decision["updated_ai_state"]`; when `execute` returns False the updated AI state is discarded and the next tick re-plans from the same base state. (Was `:490-491` at the 2026-07-17 fact-check, then `tick_loop.py:817-822`; the gate itself is unchanged, only its position — the tick loop has since been split, and the body that runs one tick is now `tick_body.py`.)
[^2]: grep of `mark_move_target_failed` across `src/`, 2026-07-17 (call-site list re-verified 2026-08-07 after the tick-loop split): the production callers are `tick_loop_actions.py:205` (walk failure), `tick_loop_command_errors.py:141` (the 0x52-rejection site, split out of the tick loop), and `completions.py:256` — none in `executor.py`. The former `game_log_feedback.py:93` site is gone with the module. **Re-run 2026-08-12:** the same three callers, two of them re-anchored, and the indirection they went through has been removed. This footnote previously said the name "resolves through `sniffer/world_state.py:140`"; that module no longer exists, and each caller now reaches `sniffer/world_service_movement.py:36` directly off the session's own world — `self.world.` in `completions.py`, `bot.world.` in `tick_loop_actions.py`, `ws.` in `tick_loop_command_errors.py` ([[session-state-deglobalisation]] step 8).
[^3]: `choose_combat_landing_tile` at `src/tankpit_bot/bot/ai/combat_landing.py:82` (signature `:82-89`; anchors re-taken 2026-08-12, the signature having gained the `ws: WorldService` parameter that [[session-state-deglobalisation]] step 8 threads through); commit `4d11980b` narrowed the mine check from `world["mines"]` to `hostile_mines(world)` for same-team passability but did not consider planner-executor consistency on hostile tiles. **Repinned 2026-08-07:** the old citation `:46-69` and its "line 68: `del world, self_state, terrain`" are both gone. The function no longer discards those parameters — it uses all of them: `world` and `self_state` for stand-off occupancy and the tie-break toward self, `terrain` for `is_landing_legal`, which is exactly the planner-executor consistency this page asked for. `hostile_mines` is still the mine check, now at `:266`.
[^4]: **Historical.** `_tracked_combat_target` was at `src/tankpit_bot/bot/executor.py:371-394` with the position-match at line 392, called from `_is_valid_teleport` at line 449. None of the three symbols exists today — the whole validator family was deleted 2026-07-21 by commit `59fce8e1` ("bot: executor is pure dispatch"), which is the resolution recorded at the end of this page. Kept as the description of the bug that motivated the deletion; see [^6] and [^7] for the receipts.
[^5]: historical shape recorded at the 2026-07-17 fact-check: `find_teleport_landing_tile` at `equipment_search.py:35-73`, line 62 `del start_x, start_y, blocked_mines`, five callers passing the arg. Current state re-verified 2026-07-31: the function is at `equipment_search.py:33` with signature `(terrain, goal_x, goal_y)`, and `grep -rn blocked_mines src/` returns zero hits tree-wide.
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
[^8]: `src/tankpit_bot/bot/ai/equipment_search.py:33-37` — the current signature is `find_teleport_landing_tile(terrain: TerrainMapProtocol, goal_x: int, goal_y: int) -> tuple[int, int] | None`, with no `blocked_mines` parameter. A grep for `blocked_mines` across `src/` returns **zero** matches, confirming the parameter and its dead plumbing were removed rather than merely bypassed. Verified 2026-07-31.
[^pasttense]: Marker added because the section's line numbers and symbols were written against a pre-refactor tree. Checked 2026-08-07: the predicates it names — `_is_valid_shoot`, `_is_valid_move_destination`, `_is_valid_teleport`, `_is_valid_pickup`, `_tracked_combat_target`, `blocked_mines` — return zero matches anywhere under `src/tankpit_bot/`, which is what establishes the past tense rather than asserting it. The surviving validation of this shape is `validate_collect_plan` at `src/tankpit_bot/bot/ai/intent.py:257`, with the state-machine guards `is_valid_transition` and `validate_transition` at `src/tankpit_bot/bot/states.py:392` and `:406`.
