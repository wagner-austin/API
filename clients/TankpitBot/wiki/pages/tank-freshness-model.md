---
title: Tank Registry Freshness Model
tags: [architecture, world-state, freshness, combat]
related: [[decode-coverage]], [[combat-chase-bug]], [[shoot-event-format]]
sources: [src/tankpit_bot/state/types/tank.py, src/tankpit_bot/state/types/tank_observation.py, src/tankpit_bot/state/mutations.py, runs/bot/bot-20260619-050303 stale-position miss loop, runs/bot/bot-20260620-191622 target-block loop]
fact_checked: 2026-06-20
confidence: high
---

# Tank Registry Freshness Model

Every tank in `world_state.tanks` carries **three independent freshness
timestamps**. Each gates a different decision tier; conflating them
caused the historical stale-registry combat bug.

## The three timestamps

| Field | Advances on | Gate |
|---|---|---|
| `timestamp_ms` | ANY observation (wire, map snapshot, radar, DOM refinement) | Acquisition (HUNT candidate) |
| `last_wire_seen_ms` | WIRE-sourced observations only | Wire presence (anti-ghost) |
| `last_position_update_ms` | Observations with `position_is_authoritative=True` AND non-null `position` | Kill-shot |

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
is_wire_sourced: bool              # True for wire; drives last_wire_seen_ms
position_is_authoritative: bool    # True when the carried position is the
                                   # server's own statement (wire-with-pos
                                   # OR MAP_DATA); drives last_position_update_ms
storage_source: EntitySource       # 'viewport' | 'radar' | 'world_state'
position: tuple[int, int] | None
team: int | None                   # plus rank / damage / direction / name / is_bot
```

The two flags are intentionally independent so MAP_DATA -- the server's
own snapshot of the global tank roster -- can advance the kill-shot
position gate without claiming wire presence (which it does not prove
-- a departed tank can linger in the snapshot for minutes). Radar
EnemyDetect and DOM-scraped client-registry refinements set both
flags False: they are tile-coarse or out-of-band estimates that must
not gate a kill shot. `position_is_authoritative` defaults to
`is_wire_sourced` when omitted, so existing wire call sites compose
the historical semantics for free.

The mutator enforces the freshness rules in code:

```python
if obs["is_wire_sourced"]:
    last_wire_seen_ms = obs["timestamp_ms"]
if obs["position_is_authoritative"] and obs["position"] is not None:
    last_position_update_ms = obs["timestamp_ms"]
```

Field aspects merge cleanly: present overwrites, `None` preserves. A
non-existent tank is created on first observation; `is_self` is set
from the registry self-tank id.

## Gates

- **`is_wire_present(last_wire_seen_ms, now_ms)`** -- TTL 7000 ms (two
  fight-cadence periods). Acquisition / HUNT candidate selection.
- **`is_position_fresh(last_position_update_ms, now_ms)`** -- TTL
  7000 ms (matched to wire-presence TTL after the 2026-06-20
  target-block loop). **Kill-shot gate.** A wire-fresh but
  position-stale target is blocked, not fired at.

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

| Message | is_wire_sourced | position_is_authoritative | position | advances |
|---|---|---|---|---|
| 0x21 TankInfo | True | True (unused, position is None) | None | `timestamp_ms`, `last_wire_seen_ms` |
| 0x28 TankEntry | True | True | `(x, y)` | all three |
| 0x2E TankStatusSync (damage) | True | True (unused, position is None) | None | `timestamp_ms`, `last_wire_seen_ms` |
| 0x3D MovementResponse | True | True | `(x, y)` | all three |
| 0x3E TankStatusFull | True | True (unused, position is None) | None | `timestamp_ms`, `last_wire_seen_ms` |
| 0x47 Movement (waypoint final) | True | True | `(x, y)` | all three |
| 0x48 EnemyDetect (radar) | False | **False** | `(x, y)` | `timestamp_ms` only |
| 0x53 ShootEvent (enemy source tile) | True | True | `(x, y)` | all three |
| container `tank_update_*` | True | True | `(x, y)` | all three |
| container `tank_status_short` | True | True (unused, position is None) | None | `timestamp_ms`, `last_wire_seen_ms` |
| container `tank_registry` | True | True | `(x, y)` | all three |
| 0x4C MapData (map snapshot) | False | **True** | `(x, y)` | `timestamp_ms`, `last_position_update_ms` |
| client-side registry refinement | False | **False** | `(x, y)` | `timestamp_ms` only |

0x48 EnemyDetect and client-side registry refinements set
`position_is_authoritative=False` on purpose: the first is a
tile-coarse radar estimate that may not match the target's actual wire
position by the next tick; the second is a DOM scrape with no server
proof. The kill-shot gate requires the server's own statement of
position -- wire-with-position or MAP_DATA snapshot.

## Registry lifecycle: 0x58 TankRemove is a no-op (changed 2026-06-22)

Tanks enter `world["tanks"]` on first observation (`apply_tank_observation`)
and leave only when `0x41 Deactivation` flips `liveness="deactivated"` and
the kill cooldown elapses. **`0x58 TankRemove` is a no-op** -- the entry
stays put. Earlier behaviour deleted the entry; that caused the bot to
abandon pursuit of locked targets that merely teleported out of viewport
(live capture 2026-06-22: bot fired exactly one homing then dropped the
lock because `0x58` fired in the next tick).

Rationale: `0x58` carries no information that the freshness gates above
can't already derive. A tank that has truly left the world stops
broadcasting `0x2E TankStatusSync`; `timestamp_ms` ages out naturally.
A tank that simply teleported keeps broadcasting `0x2E` (which refreshes
`timestamp_ms` and `last_wire_seen_ms` but NOT position) -- pursuit fires
homing at the cached coords, server picks homing weapon, homing tracks.
The only authoritative death signal is `0x41`, and the lifecycle is now
gated entirely by `liveness`.

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
