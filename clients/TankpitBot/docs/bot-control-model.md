# Bot Control Model

## Purpose

This document describes the bot control model as it exists today.

This is a current-state document, not the desired future architecture.
For the planned refactor target, see:

- `docs/bot-hfsm-refactor-plan.md`

## High-Level Shape

The current bot is a tick-based planner with an execution state machine.

Each tick:

1. drain received WebSocket messages
2. update world state
3. update execution state from world state
4. block if an in-flight action is still resolving
5. run the planner
6. execute one command

Primary files:

- `src/tankpit_bot/bot/tick_loop.py`
- `src/tankpit_bot/bot/base.py`
- `src/tankpit_bot/bot/ai_strategy.py` (orchestration)
- `src/tankpit_bot/bot/ai/context.py` (shared decision context and helpers)
- `src/tankpit_bot/bot/ai/movement.py` (walk/teleport/exploration)
- `src/tankpit_bot/bot/ai/combat_strategy.py` (combat phases and targeting)
- `src/tankpit_bot/bot/executor.py`
- `src/tankpit_bot/sniffer/world_state.py` (core state and accessors)
- `src/tankpit_bot/sniffer/world_state_dispatch.py` (protocol message routing)
- `src/tankpit_bot/sniffer/world_state_radar.py` (radar scan and cache promotion)
- `src/tankpit_bot/sniffer/world_state_tanks.py` (tank state updates)
- `src/tankpit_bot/sniffer/world_state_tiles.py` (viewport and tile patches)
- `src/tankpit_bot/sniffer/world_state_containers.py` (container and fuel updates)
- `src/tankpit_bot/sniffer/world_state_inventory.py` (inventory tracking)
- `src/tankpit_bot/sniffer/world_state_combat.py` (combat hit and kill tracking)

## Layers

### 1. World-State Layer

Files:

- `src/tankpit_bot/sniffer/world_state.py` (core state, accessors, reset)
- `src/tankpit_bot/sniffer/world_state_dispatch.py` (protocol message routing)
- `src/tankpit_bot/sniffer/world_state_radar.py` (radar and resource reconciliation)
- `src/tankpit_bot/sniffer/world_state_tanks.py` (tank state mutations)
- `src/tankpit_bot/sniffer/world_state_tiles.py` (viewport and tile patches)
- `src/tankpit_bot/sniffer/world_state_containers.py` (container CRUD)
- `src/tankpit_bot/sniffer/world_state_inventory.py` (inventory tracking)
- `src/tankpit_bot/sniffer/world_state_combat.py` (combat hit and kill tracking)

Responsibility:

- decode-driven world-state mutation
- inventory tracking from protocol
- combat hit tracking
- killed-tank tracking
- failed move target tracking
- ASCII rendering for debug output

Important point:

- this layer merges multiple observation sources with different freshness and
  scope

Main sources:

- `world_state` blob
- `radar_response`
- `movement_response`
- `movement`
- `position_update`
- `viewport_update`
- `enemy_detection`
- `deactivation`

## 2. Execution State Layer

Files:

- `src/tankpit_bot/bot/states.py`
- `src/tankpit_bot/bot/base.py`
- `src/tankpit_bot/bot/tick_loop.py`

This is not the strategic AI mode.
It is the command/execution lifecycle.

### Bot States

Current execution states:

- `INITIALIZING`
- `WAITING_FOR_POSITION`
- `IDLE`
- `SCANNING`
- `MOVING`
- `TELEPORTING`
- `COLLECTING`
- `COMBAT`
- `LOW_FUEL`
- `DISCONNECTED`

### In-Flight Action

The more authoritative command lifecycle record is `InFlightActionDict`.

Kinds:

- `none`
- `move`
- `collect`
- `teleport`
- `scan`
- `shoot`
- `map_open`

Important point:

- the bot state name and the in-flight action are not the same thing
- action lifecycle is the stronger indicator of what the bot is waiting on

## 3. Planner State Layer

Files:

- `src/tankpit_bot/bot/ai/types.py`
- `src/tankpit_bot/bot/ai/context.py`
- `src/tankpit_bot/bot/ai_strategy.py`

This is the strategic memory used by the planner.

Examples:

- config
- `mode`
- `mode_state`
- `mode_started_ms`
- killed tank IDs
- `last_scan_ms`
- `last_map_open_ms`
- `combat_target_id`
- `combat_target_x`
- `combat_target_y`
- `patrol_waypoint_index`
- `equipment_search_failures`
- `blocked_combat_targets`

Important point:

- teleport affordability is now computed from exact source/target coordinates
  using the in-game distance formula, not a flat estimated cost

### Current Behavior Labels

The planner currently emits one of these behavior labels per decision:

- `HUNT`
- `COLLECT_FUEL`
- `COLLECT_EQUIPMENT`

These labels are decision outputs for the current tick.
They are not the same thing as the durable top-level control mode stored in AI
state.

### Durable Mode Migration

The planner state now also carries a durable top-level mode lock:

- `mode`
- `mode_state`
- `mode_started_ms`

Current migration contract:

- `mode == "UNSET"`: choose the top-level owner from current recovery
  priorities
- `mode == "HUNT"`: HUNT owns planning until a recovery mode takes priority
- `mode == "RECOVER_FUEL"`: fuel recovery owns planning until the full-fuel
  exit threshold is restored
- `mode == "RECOVER_EQUIPMENT"`: equipment recovery owns planning until combat
  reserves are restored to resume thresholds
- unsupported or invalid durable modes are ignored for the tick, then owner
  selection re-runs from the current world state

## Tick Flow

### Sync

Implemented in:

- `src/tankpit_bot/bot/world_sync.py`
- `src/tankpit_bot/sniffer/decoders.py`

Behavior:

- drain buffered CDP payloads
- decode them
- dispatch them into world-state mutation

### State Update

Implemented in:

- `Bot._update_state_from_world()` in `src/tankpit_bot/bot/base.py`

This stage:

- advances startup states
- completes teleport, walk, scan, and collection actions
- may transition to `LOW_FUEL`

### Wait/Block

Implemented in:

- `_has_in_flight_action()` in `src/tankpit_bot/bot/tick_loop.py`

This stage blocks replanning while waiting on:

- move
- collect
- teleport
- scan
- map_open

Shoot is currently treated as non-blocking for replanning.

### Plan

Implemented in:

- `decide(...)` in `src/tankpit_bot/bot/ai_strategy.py`

The current planner is now a durable-owner selector, not a flat priority chain.

Current shape:

1. choose the active top-level owner (`RECOVER_FUEL`, `RECOVER_EQUIPMENT`, or
   `HUNT`)
2. run exactly one owner route for the tick
3. derive the durable substate from the owner's returned decision
4. persist the chosen owner in AI state

### Execute

Implemented in:

- `src/tankpit_bot/bot/executor.py`

Behavior:

- apply equipment toggles
- dispatch chosen command

Important point:

- executor is currently thin
- most command validity assumptions are made upstream in the planner

## Combat Model

Combat now runs under a durable `HUNT` owner. The combat target fields and the
durable `mode_state` are the authoritative combat memory.

Current durable HUNT substates:

- `ACQUIRE`
- `REFRESH`
- `CLOSE`
- `ENGAGE`
- `CONFIRM_KILL`

Main functions:

- `decide_hunt_mode(...)`
- `search_for_enemies(...)`
- `open_map_for_target(...)`
- `teleport_to_target(...)`
- `close_target(...)`
- `engage_target(...)`

Current broad sequence:

1. `HUNT.ACQUIRE` picks a viable target or searches for enemies
2. `HUNT.REFRESH` refreshes target information and re-evaluates geometry
3. `HUNT.CLOSE` teleports near the target and verifies firing geometry
4. `HUNT.ENGAGE` shoots or refreshes on miss
5. `HUNT.CONFIRM_KILL` explicitly clears a vanished/killed target before reacquiring

If teleport finds no passable adjacent landing tile (e.g. enemy is on water with
all 4 cardinal neighbors impassable), the target is added to
`blocked_combat_targets` with a TTL. The planner then skips that target and
engages the next viable threat. Blocked targets expire after `kill_cooldown_ms`.

## Fuel and Equipment Model

Fuel and equipment are controlled by threshold-driven planner branches.

Fuel:

- below low threshold triggers fuel recovery
- below critical threshold can interrupt more aggressively
- the durable owner now clears stale combat target locks when fuel recovery
  takes control
- the default full-fuel exit threshold is `1100`, matching the live tank fuel
  cap observed on April 6, 2026

Equipment:

- break thresholds trigger durable equipment recovery ownership
- resume thresholds determine when equipment recovery may release control
- the owner now runs explicit recovery substates:
  - `SENSE`
  - `SEARCH`
  - `APPROACH`
  - `PICKUP`

Important point:

- fuel recovery is now a durable owner with the same explicit recovery
  substates:
  - `SENSE`
  - `SEARCH`
  - `APPROACH`
  - `PICKUP`
- equipment recovery is also a durable owner with the same substate vocabulary
  and direct ownership of route planning

## Map Open in the Current Model

`map_open` is currently a sequencing tool and a refresh trigger.

Important points:

- there is no reliable authoritative “map is open” flag from the game
- the bot records `map_open` as an in-flight action
- tick loop clears that action after a fresh sync arrives
- the planner uses map-open because it is the current known way to provoke the
  useful global enemy-position blob

## Former Stale World-Reference Bug

This was a real bug in the earlier flat planner.

The tick loop in `_tick_once` used to read the world state snapshot at the top
of the tick, before in-flight action handling ran:

```
world = bot.get_world_state()         # snapshot taken here
...
if _has_in_flight_action(bot):        # may mutate global _world_state
    return
...
decision = ai_strategy.decide(world, ...)  # uses stale snapshot
```

When `_has_in_flight_action` cleared a stalled collection, it called
`increment_container_failed_pickups`, which mutated the global `_world_state`.
But the `world` variable still held the pre-mutation snapshot. The planner then
saw `failed_pickups=0` and retried the same dead container.

That specific retry path has since been fixed by preserving `failed_pickups`
across radar refreshes and clearing empty fuel containers when radar reports
`volume=0`. The architectural lesson still stands: control-relevant world
mutations around tick orchestration are easy to get wrong in the current flat
model.

## Why This Model Feels Messy

The current control model is functional but structurally mixed.

Main reasons:

1. execution state, strategic memory, and behavior intent are still separate
   layers
2. some migration glue still relies on older combat-target memory conventions
3. some route helpers still derive behavior labels from older tick-era naming
4. world-state freshness/source can still be expressed more strongly in planner
   decisions
5. executor does not yet act as a strong last-mile validator for every command
   family
6. in-flight action handling and planner memory still interact in fragile ways,
   even after the stale-container retry bug above was fixed

## What This Model Is Good At

- fast iteration on tactical heuristics
- easy unit testing of small decision helpers
- straightforward command lifecycle tracking
- pragmatic handling of a partially reverse-engineered game protocol

## What This Model Is Bad At

- durable intent
- clean interruption/resume behavior
- multi-step recovery loops
- extensibility for multi-bot coordination
- proving that planner behavior improved after refactors without replay tooling

## Planned Replacement Direction

The intended replacement is a hierarchical control model with durable top-level
modes such as:

- `HUNT`
- `RECOVER_FUEL`
- `RECOVER_EQUIPMENT`

That migration plan is documented in:

- `docs/bot-hfsm-refactor-plan.md`
