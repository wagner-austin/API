---
title: Tank Registry Freshness Model
tags: [architecture, world-state, freshness, combat]
related: [[decode-coverage]], [[combat-chase-bug]], [[shoot-event-format]]
sources: [src/tankpit_bot/state/types/tank.py, src/tankpit_bot/state/types/tank_observation.py, src/tankpit_bot/state/mutations.py, runs/bot/bot-20260619-050303 stale-position miss loop]
fact_checked: 2026-06-19
confidence: high
---

# Tank Registry Freshness Model

Every tank in `world_state.tanks` carries **three independent freshness
timestamps**. Each gates a different decision tier; conflating them
caused the historical stale-registry combat bug.

## The three timestamps

| Field | Advances on | Gate |
|---|---|---|
| `timestamp_ms` | ANY observation (wire OR map snapshot) | Acquisition (HUNT candidate) |
| `last_wire_seen_ms` | WIRE-sourced observations only | Wire presence (anti-ghost) |
| `last_position_update_ms` | WIRE-sourced observations that carried fresh `(x, y)` | Kill-shot |

Production cadences differ by message kind, which is why one timestamp
is not enough:

- **0x2E TankStatusSync** (status + fuel) broadcasts globally every ~2 s
  for every active tank, regardless of viewport. Refreshes
  `last_wire_seen_ms` but carries no position.
- **0x3D MovementResponse** / **0x47 Movement** / **0x28 TankEntry** /
  container TankUpdate* carry `(x, y)`. Refresh all three timestamps.
- **0x4C MapData** (map snapshot) carries positions for every tank on
  the map but is NOT wire presence proof -- a departed tank lingers in
  the snapshot for minutes. Advances `timestamp_ms` only.

## The single mutator

Every tank-state mutation flows through
`apply_tank_observation(state, obs)` in `state/mutations.py`. The
observation -- a `TankObservation` TypedDict -- declares:

```
is_wire_sourced: bool        # True for wire messages; False for map
storage_source: EntitySource # 'viewport' | 'radar' | 'world_state'
position: tuple[int, int] | None
team: int | None             # plus rank / damage / direction / name / is_bot
```

The mutator enforces the freshness rules in code:

```python
if obs.is_wire_sourced:
    last_wire_seen_ms = obs.timestamp_ms
    if obs.position is not None:
        last_position_update_ms = obs.timestamp_ms
```

Field aspects merge cleanly: present overwrites, `None` preserves. A
non-existent tank is created on first observation; `is_self` is set
from the registry self-tank id.

## Gates

- **`is_wire_present(last_wire_seen_ms, now_ms)`** -- TTL 7000 ms (two
  fight-cadence periods). Acquisition / HUNT candidate selection.
- **`is_position_fresh(last_position_update_ms, now_ms)`** -- TTL
  3000 ms. **Kill-shot gate.** A wire-fresh but position-stale target
  is blocked, not fired at.

Both gates live in `bot/ai/threats.py`. Combat strategy reads them in
order: first `is_wire_present` (ghost gate), then `is_position_fresh`
(stale-position gate), then the miss-on-moved-target re-aim path.

## How the historical bug manifested

Before 2026-06-19, `update_tank_damage` (now deleted) advanced
`last_wire_seen_ms` on every damage-only TankStatusSync. A tank that
teleported out of viewport stopped producing position-bearing messages,
but the global status broadcast kept the wire-presence stamp fresh.
The kill-shot gate -- which had only `last_wire_seen_ms` to check --
never tripped, and the bot kept firing at the stale registry tile.

Run `bot-20260619-050303` recorded **25 combat_miss events on the same
target (orange-8) at the same tile (155,155) over ~100 seconds**, all
marked `target_moved=false` because the registry kept agreeing with
the bot's stale belief about the target's location.

The three-timestamp model + the `apply_tank_observation` chokepoint
+ the kill-shot position-freshness gate make this class of bug
impossible to reintroduce without breaking a locked invariant test in
`tests/world_state/test_tank_observation.py`.

## Per-message contract

For each wire message kind, the dispatcher in
`sniffer/world_state_tanks.py` builds a `TankObservation` with:

| Message | is_wire_sourced | position | advances |
|---|---|---|---|
| 0x21 TankInfo | True | None | `timestamp_ms`, `last_wire_seen_ms` |
| 0x28 TankEntry | True | `(x, y)` | all three |
| 0x2E TankStatusSync (damage) | True | None | `timestamp_ms`, `last_wire_seen_ms` |
| 0x3D MovementResponse | True | `(x, y)` | all three |
| 0x3E TankStatusFull | True | None | `timestamp_ms`, `last_wire_seen_ms` |
| 0x47 Movement (waypoint final) | True | `(x, y)` | all three |
| 0x48 EnemyDetect (radar) | False | `(x, y)` | `timestamp_ms` only |
| 0x53 ShootEvent (enemy source tile) | True | `(x, y)` | all three |
| container `tank_update_*` | True | `(x, y)` | all three |
| container `tank_status_short` | True | None | `timestamp_ms`, `last_wire_seen_ms` |
| container `tank_registry` | True | `(x, y)` | all three |
| 0x4C MapData (map snapshot) | False | `(x, y)` | `timestamp_ms` only |
| client-side registry refinement | False | `(x, y)` | `timestamp_ms` only |

0x48 EnemyDetect routes through the non-wire path on purpose: radar
returns a tile-coarse estimate that may not match the target's actual
wire position by the next tick. The kill-shot gate must continue to
require a fresh **wire-bearing** position; radar alone does not
suffice.

## Tests that lock the contract

Every rule above is pinned by a test in
`tests/world_state/test_tank_observation.py`:

- `TestInvariantTimestampAlwaysAdvances` -- rule 1.
- `TestInvariantWireSeenRequiresWire` -- rule 2.
- `TestInvariantPositionFreshnessRequiresBoth` -- rule 3.
- `TestFieldMergeSemantics` -- present overwrites, `None` preserves.
- `TestTankCreationOnFirstObservation` -- the create path.

The combat-strategy regression test for the kill-shot gate lives in
`tests/bot/ai/test_combat_strategy.py::TestWirePresenceGate::test_position_stale_adjacent_target_is_blocked_not_shot`.
Removing or weakening any of these tests is a deliberate contract
change and must come with a docstring + this wiki page update.
