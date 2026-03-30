# HFSM Refactor Plan

## Status

This document is the concrete implementation plan for refactoring the bot from
its current flat per-tick priority planner into a hierarchical state machine
with better replay/debug tooling and stricter command validation.

This is not a brainstorming note.
Each phase below has:

- scope
- files to change
- explicit deliverables
- acceptance criteria
- things that are out of scope for that phase

## Why This Refactor Exists

The current bot works, but control is spread across too many overlapping state
systems:

- execution state in `src/tankpit_bot/bot/states.py`
- in-flight action lifecycle in `src/tankpit_bot/bot/states.py`
- AI memory in `src/tankpit_bot/bot/ai/types.py`
- strategic arbitration in `src/tankpit_bot/bot/ai_strategy.py`

That creates these concrete problems:

- the planner re-litigates the whole world every tick instead of holding a
  durable goal
- combat, fuel recovery, equipment recovery, search, and transport sequencing
  are mixed together
- the executor mostly sends what the planner asked for without enough
  last-second validation
- stale target data can still drive actions
- live iteration is weak because there is no proper replay harness

## Goals

1. Introduce a durable top-level AI mode instead of relying on a flat priority
   chain alone.
2. Separate world modeling, planning, and command execution more cleanly.
3. Make target freshness/source explicit enough to block stale actions.
4. Add replay/debug tooling so real capture sessions can become regression
   fixtures.
5. Reduce map-open dependence where possible, but keep it as the known fallback
   for global enemy refresh until a better protocol path is discovered.

## Non-Goals

- This refactor does not assume a new protocol command exists for “full world
  state on demand”.
- This refactor does not attempt multi-bot coordination yet.
- This refactor does not rewrite the sniffer/decoder stack unless required to
  expose source/freshness metadata.
- This refactor does not remove the existing execution FSM in one shot.

## Desired End State

### Control Layers

1. World model layer
   - authoritative observed facts only
   - target/container entries carry freshness metadata
   - no planning logic

2. Behavior controller layer
   - durable top-level modes
   - explicit substate per mode
   - mode transitions based on explicit entry/exit conditions

3. Execution layer
   - translate intent into commands
   - validate commands against current freshness/visibility constraints
   - track in-flight action lifecycle

### Target Top-Level Modes

- `HUNT`
- `RECOVER_FUEL`
- `RECOVER_EQUIPMENT`

There is no separate `SEARCH_ENEMY` mode. Enemy search is handled by
`HUNT.ACQUIRE`, which covers local refresh, nearest-enemy queries, and
map-open fallback. Separating it would duplicate target acquisition logic.

### Expected Substates

- `HUNT`
  - `ACQUIRE`
  - `REFRESH`
  - `CLOSE`
  - `ENGAGE`
  - `CONFIRM_KILL`

- `RECOVER_FUEL`
  - `SENSE`
  - `SEARCH`
  - `APPROACH`
  - `PICKUP`
  - `DONE`

- `RECOVER_EQUIPMENT`
  - `SENSE`
  - `SEARCH`
  - `APPROACH`
  - `PICKUP`
  - `DONE`

## Phase 0: Tooling First

### Scope

Build the minimum replay/debug tooling required to stop iterating blindly.

### Files

- new `scripts/replay_bot.py`
- likely new helper module under `src/tankpit_bot/bot/` or `src/tankpit_bot/sniffer/`
- tests under `tests/`

### Deliverables

1. A replay script that:
   - loads a captured session
   - feeds received frames through the existing decode/world-state path
   - runs bot decision logic tick by tick
   - records the commands the bot would send

2. A structured decision trace format, at minimum:
   - tick index
   - timestamp
   - self position/fuel
   - active AI mode/state
   - visible threat summary
   - selected command
   - selected reason

3. At least 3 replay fixtures from real problem cases:
   - corpse-targeting case
   - equipment search loop case
   - bad fuel/equipment recovery case

### Acceptance Criteria

- we can run one command against a saved capture and inspect per-tick decisions
- a real bad session can be replayed without launching the live bot
- at least one known bug is represented as a replay regression test

### Out of Scope

- HFSM mode implementation
- planner rewrite

## Phase 1: World Model Freshness and Validation

### Scope

Make target trust explicit enough that stale actions can be blocked.

### Files

- `src/tankpit_bot/state/types.py`
- `src/tankpit_bot/sniffer/world_state.py`
- `src/tankpit_bot/bot/ai/threats.py`
- `src/tankpit_bot/bot/executor.py`
- tests in `tests/world_state/` and `tests/bot/`

### Deliverables

1. Add source/freshness metadata for tanks and containers.
   Minimum acceptable fields:
   - `timestamp_ms`
   - `source`

2. Valid `source` values must be explicit, not implied.
   Support existing observed sources first:
   - `viewport`
   - `radar`
   - `world_state`
   Add richer source categories (e.g. `nearest_enemy`, `registry`) only when
   the code actually uses them. Do not pre-build categories that have no
   producer yet.

3. Executor-side validation before dispatch:
   - do not shoot stale non-viewport targets
   - do not teleport to targets whose source is not valid for teleporting
   - do not pickup a container that no longer exists in world state

4. Fix the stale planner-snapshot bug in `_tick_once`.
   Currently the decision input (`world`) is read before in-flight action
   handling, which can mutate world state (e.g. `increment_container_failed_pickups`).
   The fix: the decision input snapshot must be created after all in-flight
   completion and timeout mutations have run.

5. Add mine awareness to the world model.
   Not full mine tactics — just:
   - known mine tiles enter world-model source/freshness like any other entity
   - movement validation rejects known mine tiles as walk/teleport destinations
   This belongs here because it is world-model validity, not later behavior
   polish.

6. Add tests for rejected stale actions and mine-blocked movement.

### Acceptance Criteria

- stale targets can be identified in code, not guessed from logs
- executor can reject a planner command when the target is no longer trustworthy
- planner snapshot is never stale with respect to in-flight action mutations
- known mine tiles block movement planning
- tests cover at least one stale-shoot rejection, one stale-pickup rejection,
  and one mine-blocked movement rejection

### Out of Scope

- durable HFSM modes
- mine-aware tactical behavior (e.g. mine avoidance pathfinding)

## Phase 2: Introduce Durable Top-Level AI Mode

### Scope

Add explicit persistent AI mode and substate without deleting the old planner in
one shot.

### Files

- `src/tankpit_bot/bot/ai/types.py`
- `src/tankpit_bot/bot/ai_strategy.py`
- `tests/bot/ai/test_types.py`
- `tests/bot/test_ai_strategy.py`

### Deliverables

1. Add persistent AI mode fields to AI state.
   Minimum:
   - `mode` (Literal of top-level modes plus `"UNSET"`)
   - `mode_state` (substate within the active mode)
   - `mode_started_ms`

2. Define entry/exit conditions for each top-level mode.

3. Implement the mode-lock migration rule:
   - if `mode != "UNSET"`, the HFSM mode owns planning for that tick
   - the old flat priority chain runs only when `mode == "UNSET"`
   - if the active mode reaches an invalid or unrecoverable state, it clears
     to `"UNSET"` and the old planner runs as fallback

   This rule is the migration contract. It means:
   - new HFSM modes can be introduced one at a time
   - the old planner is never deleted until all modes are stable
   - both paths never run in the same tick

### Acceptance Criteria

- AI state contains a durable top-level mode that survives across ticks
- when a mode is active, the flat priority chain does not run
- when a mode clears to `"UNSET"`, the flat priority chain runs as fallback
- invalid mode state clears cleanly instead of crashing

### Out of Scope

- full replacement of all current behavior branches

## Phase 3: Implement `RECOVER_EQUIPMENT`

### Scope

Turn equipment recovery into the first real HFSM mode.

### Files

- `src/tankpit_bot/bot/ai_strategy.py`
- `src/tankpit_bot/bot/ai/types.py`
- possibly `src/tankpit_bot/bot/executor.py`
- tests in `tests/bot/test_ai_strategy.py`

### Deliverables

1. `RECOVER_EQUIPMENT` mode with substates:
   - `SENSE`
   - `SEARCH`
   - `APPROACH`
   - `PICKUP`
   - `DONE`

2. Entry conditions:
   - dual below break threshold
   - radar below break threshold

3. Exit conditions:
   - dual at or above resume threshold
   - radar at or above resume threshold

4. Search rules:
   - prefer local actionable equipment
   - use radar if appropriate
   - use local short teleport hops only
   - no static global patrol teleport sectors
   - bail out after repeated empty search attempts

### Acceptance Criteria

- bot does not leave equipment recovery after one small pickup unless reserve is
  actually restored
- bot does not use old cross-map sector teleports
- replay fixture for the old radar/teleport loop now passes

## Phase 4: Implement `RECOVER_FUEL`

### Scope

Turn fuel recovery into the second real HFSM mode.

### Files

- `src/tankpit_bot/bot/ai_strategy.py`
- `src/tankpit_bot/bot/ai/pathfinding.py`
- tests in `tests/bot/`

### Deliverables

1. `RECOVER_FUEL` mode with substates:
   - `SENSE`
   - `SEARCH`
   - `APPROACH`
   - `PICKUP`
   - `DONE`

2. Fuel search rules:
   - prefer fresh actionable fuel
   - use radar when needed
   - avoid wasteful search when no meaningful fuel is available

3. Eliminate the tiny-step crawl behavior by using durable navigation goals or a
   longer validated path chunk.

### Acceptance Criteria

- bot can stay in fuel recovery until fuel is genuinely stabilized
- fuel collection no longer depends on repeated 1-2 tile replans for ordinary
  pathing cases

## Phase 5: Implement `HUNT`

### Scope

Move combat into a durable HFSM mode.

### Files

- `src/tankpit_bot/bot/ai_strategy.py`
- `src/tankpit_bot/bot/executor.py`
- tests in `tests/bot/`

### Deliverables

1. `HUNT` mode with substates:
   - `ACQUIRE`
   - `REFRESH`
   - `CLOSE`
   - `ENGAGE`
   - `CONFIRM_KILL`

2. Replace ad hoc `combat_phase` behavior with explicit state transitions.

3. Add target freshness rules:
   - viewport-fresh target can be shot
   - nearest-enemy target can be closed on, but not blindly shot if stale
   - map-derived target needs refresh before blind engagement after delay

4. Prefer `nearest_enemy` refresh before `map_open` when appropriate.

### Acceptance Criteria

- miss/reacquire loop is simpler and more deterministic
- post-kill recovery transition is explicit
- stale enemy positions do not directly produce blind shots

## Phase 6: Retire Old Arbitration Paths

### Scope

Remove obsolete flat-priority branches once equivalent HFSM modes are stable.

### Files

- `src/tankpit_bot/bot/ai_strategy.py`
- tests in `tests/bot/`
- docs

### Deliverables

1. Delete dead or duplicated planner branches.
2. Remove old assumptions from tests.
3. Update architecture documentation to reflect the new control model.

### Acceptance Criteria

- no duplicated legacy and HFSM logic for the same behavior path
- README and docs match the actual code

## Documentation Cleanup Plan

### Current Problems

1. `README.md` is stale.
   It still describes behavior modes like `DEFEND`, `DEPOSIT_FUEL`, and
   `PATROL` that do not match the current planner.

2. Protocol docs are useful, but they are not control-architecture docs.

3. There is currently no single document that explains:
   - the current control model
   - the refactor target
   - the migration sequence

### Required Documentation Changes

1. Keep:
   - `docs/protocol.md`
   - `docs/protocol_reference.md`
   - `docs/decoding_status.md`

2. Add:
   - this document as the implementation plan
   - later, a separate `docs/bot-architecture.md` once the refactor is stable

3. Update:
   - `README.md` to stop describing obsolete behavior modes

4. Do not use progress notes as architecture docs.

## Immediate Next Actions

1. Build `scripts/replay_bot.py`.
2. Convert 2-3 real failure sessions into replay fixtures.
3. Fix the stale planner-snapshot bug in `_tick_once`.
4. Add target `source` metadata to state (existing sources only).
5. Add mine tiles to world model; reject mine landings in movement validation.
6. Add executor-side stale-action rejection.
7. Introduce persistent top-level AI mode with mode-lock migration rule.
8. Implement `RECOVER_EQUIPMENT` as the first HFSM mode.

## Definition of Done

This refactor is only done when all of the following are true:

- the bot has a durable top-level AI mode
- fuel/equipment recovery are not per-tick branch accidents
- combat no longer blindly acts on stale targets
- replay-driven regression tests exist for real failures
- README/docs describe the real control architecture instead of obsolete modes
