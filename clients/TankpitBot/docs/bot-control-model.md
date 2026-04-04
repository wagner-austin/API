# Bot Control Model

## Purpose

This document describes the bot control model as it exists today.

This is a current-state document, not the desired future architecture.
For the planned refactor target, see:

- `docs/hfsm-refactor-plan.md`

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
- killed tank IDs
- `last_scan_ms`
- `last_map_open_ms`
- `combat_target_id`
- `combat_target_x`
- `combat_target_y`
- `combat_phase`
- `patrol_waypoint_index`
- `equipment_search_failures`
- `blocked_combat_targets`

### Current Behavior Labels

The planner currently emits one of these behavior labels per decision:

- `HUNT`
- `COLLECT_FUEL`
- `COLLECT_EQUIPMENT`

These labels are decision outputs for the current tick.
They are not yet durable top-level control modes.

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

The current planner is a flat priority chain with some tactical substate.

Current shape:

1. critical fuel
2. critical equipment
3. critical equipment search
4. combat if already locked
5. normal fuel if combat not locked
6. normal equipment if combat not locked
7. combat acquisition/continuation
8. fallback enemy search

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

Combat currently uses a subphase model inside AI state rather than a full
hierarchical controller.

Current phases:

- `none` (no combat engagement)
- `acquiring` (opening map to locate target)
- `closing` (teleporting to target, then verifying firing position)
- `engaging` (shooting at target)

Main functions:

- `_try_combat(...)`
- `_combat_open_map(...)`
- `_combat_teleport(...)`
- `_combat_close(...)` (verifies cardinal adjacency before shooting; re-teleports if not)
- `_combat_shoot(...)`

Current broad sequence:

1. open map to refresh enemy positions
2. teleport near target
3. verify cardinally adjacent (re-teleport if not)
4. shoot
5. on miss, reopen map and reacquire

If teleport finds no passable adjacent landing tile (e.g. enemy is on water with
all 4 cardinal neighbors impassable), the target is added to
`blocked_combat_targets` with a TTL. The planner then skips that target and
engages the next viable threat. Blocked targets expire after `kill_cooldown_ms`.

## Fuel and Equipment Model

Fuel and equipment are controlled by threshold-driven planner branches.

Fuel:

- below low threshold triggers fuel recovery
- below critical threshold can interrupt more aggressively

Equipment:

- break thresholds trigger emergency recovery
- resume thresholds determine when new fights may start again

Important point:

- these are not yet fully durable recovery modes
- they are planner branches with some retained state

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

1. execution state, strategic memory, and behavior intent are separate but not
   hierarchical
2. planner re-arbitrates much of the world every tick
3. combat uses retained subphase, while fuel/equipment mostly use threshold
   branches
4. world-state freshness/source is not yet expressed strongly enough in planner
   decisions
5. executor does not yet act as a strong last-mile validator
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
- `SEARCH_ENEMY`

That migration plan is documented in:

- `docs/hfsm-refactor-plan.md`
