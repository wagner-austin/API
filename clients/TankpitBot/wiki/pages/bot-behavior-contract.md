---
title: Bot Behavior Contract
tags: [bot, contract, verification]
related: [[client-state-machine]], [[decode-coverage]], [[tank-freshness-model]], [[gameplay-loop]]
sources: [tankpit_bot/bot/, tankpit_bot/sniffer/, runs/bot/latest.events.jsonl format]
fact_checked: 2026-06-20
confidence: high
verified: 2026-06-20 (anchored to specific code paths + integration tests)
---

# Bot Behavior Contract

The single source of truth for *what the bot must do* and *how each behavior is verified*. When proposing a fix, consult this page first to know what else might break.

This contract complements [[client-state-machine]] (the JS client's state machine) and [[tank-freshness-model]] (our wire-presence semantic). The state machine describes the *server protocol*; this page describes *our bot's obligations*.

Format: each row in each section has **MUST / MUST NOT / Verified by**. "Verified by" names the smoke assertion and integration test that locks the behavior in. If a behavior is not yet verified, that's flagged explicitly — every claim here must eventually point to a test.

## 1. Lifecycle

### 1.1 Startup

| Aspect | Contract |
|---|---|
| MUST | Transition `INITIALIZING → WAITING_FOR_POSITION → IDLE` within `action_stall_timeout_ms` (10 s default). |
| MUST | Establish `self_state` (tank id, position, team) before leaving `WAITING_FOR_POSITION`. |
| MUST NOT | Enter HUNT or any decision mode while `self_state` is None. |
| Verified by | `smoke[1]` (login completes); `tests/integration/test_bot_login.py` (TBD). |

### 1.2 Session-end (graceful)

| Aspect | Contract |
|---|---|
| MUST | When `TANKPIT_BOT_SESSION_SECONDS` elapses, flush `runs/bot/latest.events.jsonl`, write `latest.summary.txt`, append to `runs/bot/_index.tsv`. |
| MUST | Exit cleanly (return code 0) even mid-action. |
| MUST NOT | Leave `latest.*` symlinks pointing at a prior run's data. |
| Verified by | `tests/integration/test_session_shutdown.py` (TBD); `runs/bot/_index.tsv` row appended (Tier 3.3, handed off). |

### 1.3 Session-end (interrupted: SIGINT / SIGTERM / crash)

| Aspect | Contract |
|---|---|
| MUST | Flush events JSONL before exit. |
| MUST | Write `latest.summary.txt` with `exit_reason=interrupted` (or `crashed`). |
| MUST | Append to `_index.tsv` so the run is discoverable. |
| MUST NOT | Lose events emitted in the final tick. |
| Verified by | `tests/integration/test_signal_handler.py` (TBD; Tier 3.4, handed off). |

## 2. Perception (world state)

### 2.1 Tank tracking

| Aspect | Contract |
|---|---|
| MUST | Every tank in `world_state["tanks"]` carries the three freshness timestamps (`timestamp_ms`, `last_wire_seen_ms`, `last_position_update_ms`) plus a `liveness` field. See [[tank-freshness-model]]. |
| MUST | `timestamp_ms` advances on every observation (wire or map). |
| MUST | `last_wire_seen_ms` advances ONLY on `is_wire_sourced=True` observations. |
| MUST | `last_position_update_ms` advances ONLY on `is_wire_sourced=True` observations that also carry `position`. |
| MUST | `liveness` is one of `alive` / `deactivated`. 0x41 Deactivation → `deactivated`; corpse-direction wire (`direction >= 32`) → `deactivated`; wire-sourced position update with alive direction → `alive`. |
| MUST | 0x58 TankRemove is a NO-OP at the registry level. 0x58 is *tracking removal*, not a kill — verified 2026-06-20 (orange-5 got 5 TankRemove events across 2 actual kills). The earlier behaviour deleted the entry from `tanks`; that caused the bot to abandon pursuit of locked targets that merely teleported out of viewport (live capture 2026-06-22 — bot fired 1 homing then dropped the lock). Keeping the entry lets `find_locked_target_pursuit` keep firing toward the cached coords until 0x41 Deactivation arrives or `timestamp_ms` goes stale. |
| MUST NOT | Treat 0x58 as a death signal. Use `0x41 Deactivation` (it flips `liveness="deactivated"` and routes through the kill cooldown). |
| Verified by | `tests/world_state/test_mutations.py::TestRemoveTank::test_keeps_tank_in_registry` (0x58 is no-op); `tests/world_state/test_mutations.py::TestDeactivateTank` (0x41 sets liveness); `tests/bot/ai/test_hunt_mode.py::test_hunt_engage_fires_homing_when_locked_target_left_viewport` (pursuit fires while target is out of viewport). |

### 2.2 Position correctness

| Aspect | Contract |
|---|---|
| MUST | Tank positions reflect the latest authoritative source: MovementResponse (0x3D) > MapData (0x4C) > TankEntry (0x28) > TankInfo (0x21 — no position) > viewport overlay. |
| MUST | MapData updates positions for ALL tanks regardless of liveness. (A `deactivated` tank's position can still be informational; a `removed` tank doesn't exist in the registry, so this is moot.) |
| MUST NOT | Believe a `(0, 0)` position as live — that's the unsynced-tank sentinel (`analyze_threats` filters tanks at the origin). |
| Verified by | `tests/sniffer/test_world_state_dispatch_tank.py::TestDispatchMapData::test_map_data_lifts_tank_positions`. |

### 2.3 MapData processing

| Aspect | Contract |
|---|---|
| MUST | Mark `world_service.mark_map_data_processed()` after applying MapData tank observations. This is what `_clear_completed_map_open` polls. |
| MUST | Apply MapData position updates for every listed tank (no skip for any liveness state). MapData is authoritative for "where this tank is right now". |
| Verified by | `tests/sniffer/test_world_state_dispatch_tank.py::TestDispatchMapData::test_map_data_marks_action_complete` (locked in 2026-06-20). |

## 3. Decision-making

### 3.1 Mode selection

| Aspect | Contract |
|---|---|
| MUST | Mode selection is deterministic given (world_state, self_state, config). No randomness. |
| MUST | COLLECT entry uses `should_enter_collect`, which fires when ANY of: (1) `fuel <= fuel_low_threshold` (interrupts even an active combat target); (2) **Weapon emergency** -- any reserve below its *break* threshold (dual / homing < 4 or radar < 5) -- interrupts even with an active combat target; (3) **Between kills** -- any reserve below its *resume* threshold (25 / 25 / 20) AND `combat_target_id == -1`. The unified COLLECT mode replaced the historical `RECOVER_FUEL` + `RECOVER_EQUIPMENT` split 2026-06-24. Fuel-low still interrupts an active engagement; weapon-low waits until between kills. |
| MUST | HUNT is the default mode when no other mode triggers. |
| MUST NOT | Enter HUNT while `self_state` is None (lifecycle 1.1). |
| Verified by | `tests/bot/ai/test_mode_controller.py` (existing). |

### 3.2 HUNT acquisition

| Aspect | Contract |
|---|---|
| MUST | Use `analyze_threats` to score candidate enemies. Sort by distance, then finish-priority, then freshness. |
| MUST | Filter out: self, allies, unsynced `(0,0)`, `liveness != "alive"` (catches both direct 0x41 deactivations and corpse-direction wire arrivals via `apply_tank_observation`), stale `timestamp_ms` older than `WIRE_PRESENCE_TTL_MS`. |
| MUST | Open the map (`map_open` command) when no candidate is found and `map_open_cooldown_ms` has elapsed. |
| MUST | After `map_open`, wait for the authoritative completion signal `map_data_processed` (set by `_dispatch_map_data` via `ws.mark_map_data_processed()`). |
| MUST NOT | Re-issue `map_open` while a prior `map_open` is in flight and within `action_stall_timeout_ms`. |
| MUST NOT | Fire `make_radar_command()` to "search for enemies". Radar does not reveal enemies (see [[radar-mechanics]]); enemy discovery is map-open + viewport-edge walking only. |
| Verified by | `smoke[2,3]` (map_open clears, HUNT acquires); `tests/integration/test_hunt_acquires_wire_confirmed_enemy.py` (Tier 1, handed off). |

### 3.3 Combat shoot gates (`_combat_shoot`)

| Aspect | Contract |
|---|---|
| MUST | Gate 1: `is_wire_present(target["last_wire_seen_ms"], now)` — guard against firing at wire-stale ghosts. |
| MUST | Gate 2: `is_position_fresh(target["last_position_update_ms"], now)` — guard against firing at a stale position. |
| MUST | On gate failure, call `block_combat_target_and_replan` (does NOT fire, picks a different target after cooldown). |
| MUST | On miss against a moved target, re-aim at the new position. |
| MUST | On miss against a stationary target, mark the target as blocked. |
| MUST NOT | Fire if either gate fails. |
| RESOLVED | Stationary practice-room bots pass these gates correctly under the 2-state liveness model. The earlier "ghost cache" concern (#75/#77/#80) was a false reading of the corpus — verified 2026-06-20 that tanks disappear from MapData immediately on kill; no server-side cache of dead tanks at kill tiles exists. |
| Verified by | `tests/integration/test_combat_fires_when_gates_pass.py` (Tier 1, handed off); `tests/integration/test_combat_blocks_on_wire_stale_target.py` (Tier 1, handed off). |

### 3.4 COLLECT (unified fuel + equipment collection)

| Aspect | Contract |
|---|---|
| MUST | COLLECT runs a single cascade per tick: (1) continue a held equipment or fuel lock from a previous tick; (2) pick up the best equipment in the current viewport; (3) pick up the best fuel in the current viewport (skipped at learned capacity); (4) **Sense** -- radar when the viewport has unscanned tiles, or walk toward an unscanned tile so the next free radar covers it; (5) **Hop** -- teleport to a fresh viewport when nothing actionable remains here. Equipment ranks ahead of fuel by design: the user's gameplay loop is "pick up all equipment, then maybe the biggest fuel container, then hop". |
| MUST | Exit (`should_exit_collect`) holds the mode until BOTH `fuel >= fuel_full_threshold` AND combat reserves are restored (dual ≥ 25 AND homing ≥ 25 AND radar ≥ 20). The break/resume gap gives hysteresis -- entry at the low break, exit only at the higher resume. |
| MUST | `self_state["fuel"]` is updated **only** from the wire's absolute-fuel messages (0x44 FuelGain, 0x2E TankStatusSync, 0x64 FuelDeposit). `pickup_container` is registry-only -- it does NOT add `transferred = prior_volume - remaining_volume` locally. The local-delta branch was a double-count on top of the wire's already-correct absolute fuel; removed 2026-06-23 after live observation of a 438-volume container producing a +438 ghost. See [[fuel-system#fuel-data-flow-single-source-of-truth]]. |
| MUST | A pickup is NEVER pre-filtered as wasted -- the server picks the slot you're most behind on at pickup time (see [[equipment-system]]). The bot dispatches `pickup_equipment` whenever any equipment container is in range; only the all-25 case fails with code 7. |
| MUST | Recognise server 0x52 `SUPERVISOR_ERROR_INVENTORY_FULL` (code 7) as an action-blocking error in `_ACTION_BLOCKING_COMMAND_ERRORS` (`bot/tick_loop_actions.py:44`). The in-flight pickup clears immediately on the wire signal instead of stalling the full 10 s timeout, and the container's `failed_pickups` counter is bumped so the blacklist takes over. Closed 2026-06-21 with the empirical guard `tests/bot/test_tick_loop_coverage.py::test_command_error_clears_collect_on_inventory_full`. |
| MUST | After firing a radar, mark exactly the tiles the radar revealed in `AIStateDict.local_scan_tiles`. Free radar = intersection of `(tank ± 2)` with viewport bounds; extra radar = every tile in the viewport. The bot picks its next forage action from this map (see `bot/ai/scan_coverage.py`, refactor 2026-06-21). |
| Verified by | `tests/bot/ai/test_mode_controller.py`, `tests/bot/ai/test_collect_mode_fuel.py`, `tests/bot/ai/test_collect_mode_equipment.py`, `tests/bot/ai/test_collect_mode_integration.py`, `tests/world_state/test_mutations.py::TestPickupContainer`. |

## 4. Action execution

### 4.1 Action lifecycle

| Aspect | Contract |
|---|---|
| MUST | Every action has a START (`emit_wire`), a WAITING phase (`emit_sync` repeated), and a COMPLETION (`emit_wire_complete` with `signal=`). |
| MUST | Completion signal is one of: `map_data_processed`, `teleport_landed`, `radar_scan_complete`, `position_reached`, `container_consumed_or_reached`, `stall_timeout`, `movement_rejected`. |
| MUST | If no authoritative signal arrives within `action_stall_timeout_ms`, emit `signal=stall_timeout` and replan to IDLE. |
| MUST NOT | Leave an action "in-flight" forever — `_clear_stalled_action` must fire. |
| Verified by | `tests/bot/test_completion_events.py` (existing); `smoke[5]` (zero stalls in first 10 s). |

### 4.2 Anti-loop protection

| Aspect | Contract |
|---|---|
| MUST | `map_open_cooldown_ms` (5 s default) prevents repeated map opens. |
| MUST | `kill_cooldown_ms` (30 s default) prevents re-targeting a recently killed tank. |
| MUST | `scan_cooldown_ms` (5 s default) prevents radar thrashing. |
| MUST NOT | Stall + replan + re-issue the same action within one cooldown window (the open-close-map loop pattern). |
| Verified by | `smoke[5]`; `tests/integration/test_stall_timeout_replans_to_idle.py` (Tier 2, handed off). |

## 5. Anti-patterns (must never re-emerge)

These are the historical bug-shapes the bot has suffered. Every fix in `combat_strategy.py`, `world_state_dispatch.py`, or `tick_loop_actions.py` should be checked against this list.

| Anti-pattern | What it looks like | Prevented by |
|---|---|---|
| Open-close-map loop | `WIRE: map_open` → `SYNC: waiting for map open sync` (× N) → `stall_timeout` → repeat. | `mark_map_data_processed()` must be called by `_dispatch_map_data`. Test: `test_map_data_marks_action_complete` (2026-06-20). |
| Ghost firing | Firing at a tile where the tank already left, just because MapData still lists it. | `is_wire_present` gate in `_combat_shoot`. Test: `test_ghost_wire_presence_regression.py` (currently under review per #77). |
| Stationary-target reject | Practice-room bots fail the kill gate because they emit no per-tank wire after join. | OPEN — design decision pending (#75). |
| Stale-position fire | Wire-presence fresh but position not updated → fire at old tile. | `is_position_fresh` gate in `_combat_shoot`. |
| Same-tile re-engage | Bot shoots the same tile 12 times after misses. | `block_combat_target_and_replan` cooldown + stationary-target detection. |
| Action amnesia | Action emits no completion event because the consumer of the wire signal never wires it up. | Authoritative-completion contract (§4.1) + Tier 2 integration tests for every action lifecycle. |
| Radar spam in covered viewport | Bot fires the radar every 2 s in the same spot. Diagnosed 2026-06-21 (live capture 19:46:33+): the old server-side `scanned_viewports` gate did not close for a free 5x5 scan, so the bot re-fired forever at extras=0 after a failed pickup. | Tile-aware forager (`bot/ai/forage.py::plan_forage_search`) uses `AIStateDict.local_scan_tiles` and `is_viewport_fully_covered(...)` as the gate. Each radar dispatch marks exactly the revealed tiles via `mark_scan_dispatched`. Tests: `tests/bot/ai/test_forage.py::TestForageSearch`. |
| Radar to find enemies | HUNT acquire dispatches a `radar` command when no target is visible. Diagnosed 2026-06-21: radar reveals only hidden entities (fuel / equipment / mines); enemies arrive through the wire stream. | `search_for_enemies` in `bot/ai/hunt_mode.py` dispatches map_open only -- the radar branch was deleted 2026-06-21. Tests: `tests/bot/ai/test_hunt_mode.py::test_hunt_search_dispatches_map_open_not_radar_during_acquire`, `tests/bot/ai/test_enemy_search.py::TestDecideMapOpen::test_fallback_opens_map_even_when_recently_opened`. |
| Edge-walk fuel burn during HUNT | Bot walked or teleported to viewport-edge tiles every tick the map was on cooldown. Diagnosed live 2026-06-22 (60 s run): 14 of 30 ticks were `edge_for_enemies`, 10 of those were terrain-blocked teleports at ~131 fuel each. Two reasons it was waste: (a) viewport shifting is OFF in this game configuration so a walk to the edge reveals no new ground, and (b) the teleport fallback aimed at a random edge tile rather than a known enemy. Walks also cost fuel (per-tile) so even the "free" edge walks burned the reserve. | `search_for_enemies` dispatches map_open unconditionally -- the cooldown-gated edge-walk branch was deleted 2026-06-22. Tests: `tests/bot/ai/test_hunt_mode.py::test_hunt_search_dispatches_map_open_not_radar_during_acquire`, `tests/bot/ai/test_enemy_search.py::TestDecideMapOpen::test_fallback_opens_map_even_when_recently_opened`. |

## 6. What is NOT in this contract (yet)

These behaviors exist in the code but lack a verified contract entry. Adding them is Tier 2 integration test work.

- Inventory restock thresholds and timing
- Patrol waypoint cycling
- Combat target switching when a higher-value enemy enters range
- Bridge-build vs obstacle-drop decision (carrying state)
- Teleport-target selection (heuristic)
- Self-rank promotion handling (0x2B reception)

When you add tests for these, also add a contract row here. The contract grows with the test suite.

## 7. How to use this page

**Before proposing a bot-behavior fix:** read the relevant section. If the change affects a MUST/MUST NOT, list which one and what test will be updated. If the change affects an anti-pattern's prevention mechanism, confirm the test still passes.

**When `make smoke` fails:** the failing assertion maps to a section here (`smoke[N]` references). Open that section to know what the bot was supposed to be doing and which other behaviors might be entangled.

**When `make check` integration tests fail:** the test name maps to a row here. The contract row tells you the broader behavior that test guards.

**When you add a new behavior:** add a row here first (MUST / MUST NOT / Verified by), then write the test, then implement. Contract-first prevents drift.

## Open items tracked elsewhere

None at end of 2026-06-20. The last decoder gap (#72 13-byte 0x43) was a 3-record multi-pickup CacheUpdate; see [[decode-coverage]] for the corrected wire format and the ContainerPickup multi-record dispatch.
