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
| MUST | 0x58 TankRemove DELETES the tank from `tanks`. 0x58 is *tracking removal*, not necessarily a kill — verified 2026-06-20 (orange-5 got 5 TankRemove events across 2 actual kills). The next MapData or per-tank wire re-adds the tank. |
| MUST NOT | Treat 0x58 as a death signal. Use `0x41 Deactivation` for that. |
| Verified by | `tests/world_state/test_mutations.py::TestRemoveTank` (0x58 deletes); `tests/world_state/test_mutations.py::TestDeactivateTank` (0x41 sets liveness); `tests/state/test_tank_observation.py::test_freshness_invariants`. |

### 2.2 Position correctness

| Aspect | Contract |
|---|---|
| MUST | Tank positions reflect the latest authoritative source: MovementResponse (0x3D) > MapData (0x4C) > TankEntry (0x28) > TankInfo (0x21 — no position) > viewport overlay. |
| MUST | MapData updates positions for ALL tanks regardless of liveness. (A `deactivated` tank's position can still be informational; a `removed` tank doesn't exist in the registry, so this is moot.) |
| MUST NOT | Believe a `(0, 0)` position as live — that's the unsynced-tank sentinel (`analyze_threats` filters tanks at the origin). |
| Verified by | `tests/sniffer/test_world_state_dispatch_tank.py::TestDispatchMapData::test_map_data_replaces_fuel_dots_and_lifts_tank_positions`. |

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
| MUST | RECOVER_FUEL takes precedence over HUNT when `fuel < fuel_low_threshold`. |
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

### 3.4 Equipment & fuel recovery

| Aspect | Contract |
|---|---|
| MUST | RECOVER_FUEL targets the nearest fuel depot (`map_fuel_dots`) when `fuel < fuel_low_threshold`. |
| MUST | Collect adjacent containers opportunistically when path passes them. |
| MUST | Refill equipment to `dual_resume_threshold` before resuming combat after a `dual_break_threshold` event. |
| Verified by | `tests/integration/test_refuel_triggers_below_threshold.py` (Tier 1, handed off). |

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
