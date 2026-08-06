---
title: Tank Registry Freshness Model
tags: [architecture, world-state, freshness, combat]
related:
  - "[[decode-coverage]]"
  - "[[combat-chase-bug]]"
  - "[[shoot-event-format]]"
  - "[[server-push-gating]]"
source_paths:
  - "runs/bot/bot-20260619-050303.capture_session.json"
  - "runs/bot/bot-20260620-191622.capture_session.json"
  - "src/tankpit_bot/state"
source_git_blobs:
  "src/tankpit_bot/state": "01f57c7928f05025a5ca0c14ab82ec4ff320031b"
fact_checked: "2026-08-04"
confidence: high
hubs: [architecture]
---

# Tank Registry Freshness Model

Every tank in `world_state.tanks` carries **four independent freshness
timestamps**.[^1] Each gates a different decision tier; conflating them
caused the historical stale-registry combat bug.[^2]

## The four timestamps

| Field | Advances on | Gate |
|---|---|---|
| `timestamp_ms` | ANY observation (wire, map snapshot, radar, DOM refinement) | Registry retention |
| `last_wire_seen_ms` | WIRE-sourced observations only | Wire presence (anti-ghost) |
| `last_position_update_ms` | Observations with `position_is_authoritative=True` AND non-null `position` | Kill-shot |
| `last_viewport_observation_ms` | Observations with `storage_source == "viewport"` only | **HUNT acquisition** |

The fourth was added after the 2026-06-21 tracking probe showed the
first three were not sufficient to answer "can I actually see this
tank?": **26 of 27 tanks passed `timestamp_ms`, `last_wire_seen_ms`,
AND `last_position_update_ms` while the JS client's own registry had
none of them in view.** 0x4C MapData refreshes everyone's position-and-
wire stamps and 0x2E TankStatusSync broadcasts globally for every alive
tank, so without a viewport-scoped stamp the threat list is the global
roster rather than the visible one. `analyze_threats` filters on this
timestamp first.[^1]

Production cadences differ by message kind, which is why one timestamp
is not enough:[^1]

- **0x2E TankStatusSync** (status + fuel) syncs every ~2 s per ACTING
  tank, regardless of viewport — the "global broadcast" reading was
  revised 2026-07-24: a playing observer adjacent to an undisturbed
  bot received ZERO 0x2E for that bot in ten minutes, so the cadence
  is per-tank activity-conditional (the archive measured it during
  combat, where every tank acts; see [[server-push-gating]]).[^8]
  Refreshes `last_wire_seen_ms` but carries no position.
  Shadow-measured (2026-07-22, `make shadow` over 245 sessions):
  OTHER-tank median inter-sync gaps sit at 1981-2010 ms — dead on
  the 2 s tick — but the SELF tank drifts to 3-4 s+ medians in ~8%
  of sessions (17 of 219), a mode other tanks never show. Your own
  truth also rides 0x44/0x64/0x49, so the self 0x2E cadence is
  evidently not load-bearing. Narrowed 2026-07-24: the drift is
  ACTIVITY-CORRELATED — 8/21 human sniff sessions are sparse (38%)
  vs 9/198 bot sessions (4.5%), and sparse sessions average half
  the command rate (16 vs 29 cmds/min). Mechanism now confirmed at
  the stream level: periodic push traffic flows only around real
  gameplay actions (the walking observer's own 0x2E arrived every
  ~3 s against its 1.5 s action beat).[^3][^8]
- **0x3D MovementResponse** / **0x47 Movement** / **0x28 TankEntry** /
  container TankUpdate* carry `(x, y)`. Refresh all three timestamps.
- **0x4C MapData** (map snapshot) carries positions for every tank on
  the map but is NOT wire presence proof -- a departed tank lingers in
  the snapshot for minutes. Advances `timestamp_ms` only.

## The single mutator

Every tank-state mutation flows through
`apply_tank_observation(state, obs)` in `state/mutations.py`. The
observation -- a `TankObservation` TypedDict -- declares:[^1]

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
the historical semantics for free.[^1]

The mutator enforces the freshness rules in code:[^1]

```python
if obs["is_wire_sourced"]:
    last_wire_seen_ms = obs["timestamp_ms"]
if obs["position_is_authoritative"] and obs["position"] is not None:
    last_position_update_ms = obs["timestamp_ms"]
```

Field aspects merge cleanly: present overwrites, `None` preserves. A
non-existent tank is created on first observation; `is_self` is set
from the registry self-tank id.[^1]

## Gates

The wire/position TTLs live in `bot/ai/threats.py`, applied inline at
their decision sites; `VIEWPORT_PRESENCE_TTL_MS` moved to
`state/types/tank.py` (2026-08-04) beside the field it gates, because
tile occupancy (`state/occupancy.py`) became its second consumer and
the two must never drift:[^1]

- **`VIEWPORT_PRESENCE_TTL_MS`** -- 5000 ms against
  `last_viewport_observation_ms`. **Acquisition gate**, applied first in
  `analyze_threats`; also gates the greeting approach
  (`bot/ai/greeting.py`) and tank-body occupancy
  ([[terrain-composition]]).
- **`WIRE_PRESENCE_TTL_MS`** -- 7000 ms (two fight-cadence periods)
  against `last_wire_seen_ms`. Ghost gate.
- **`POSITION_FRESHNESS_TTL_MS`** -- 7000 ms against
  `last_position_update_ms`, matched to the wire TTL after the
  2026-06-20 target-block loop. **Kill-shot gate.** A wire-fresh but
  position-stale target is blocked, not fired at.

Order at the combat site: viewport presence, then wire presence, then
position freshness, then the miss-on-moved-target re-aim path.[^1]

## Login choreography: every tank starts position-less (measured 2026-08-04)

The server opens every session with a **full-roster 0x21 TankInfo
dump** -- name and team for every tank on the map, NO coordinates --
and positions only arrive with the first position-bearing sync.
Measured across three captures (113 tanks): every first sighting was
0x21 (one 0x3D), and the first-sight-to-first-position window was
uniform per session -- 10.9 s (bot-20260802-205105), 9.1 s
(bot-20260803-180918), 45.7 s (sniff-20260620-190228, the user did
not open the map for 45 s). The window ends at one event for the
whole roster, so it is login choreography, not a per-tank race.[^7]

Until that first position message a registry entry sits at the
``(0, 0)`` construction default (`apply_tank_observation` creates
unknown tanks from any observation, defaulting absent fields). The
``(0, 0)`` state is therefore the NORMAL opening state of every tank
the bot ever meets -- and (0, 0) is also a legal tile.

**The canonical predicate** is
`state.types.has_known_position(tank)`: coordinates differ from the
default, OR `last_position_update_ms > 0` (covers a tank
authoritatively placed exactly on (0, 0); the coordinate check alone
covers radar EnemyDetect, which writes real coords without advancing
the kill-shot timestamp). Seven modules used to hand-copy the inline
`x == 0 and y == 0` comparison and an eighth consumer
(`state/occupancy.py`, 2026-08-04) shipped without it -- walling off
the map corner with the whole roster's phantom bodies for the
session's first 5 s. All sites now consume the predicate, and the
guard rule `scripts/state_sentinel_rules.py` bans the inline idiom
outside `state/types/tank.py`.

One deliberate non-consumer: the HELLO greeting
(`bot/ai/greeting.py`). User ruling 2026-07-31 -- "hello can run
anytime... as long as the other player is on the map logged in" -- a
human still at the roster default gets greeted the moment their
identity broadcast lands; the predicate gates targeting and the
stand-off visit, never the chat.

Team and rank share the same defaulting mechanism (`team=0` is a real
team) but NOT the same exposure: the only creation route without a
team is `_update_tank_position` (0x47/0x53/0x42 sources), and the
measurement shows the 0x21 roster dump -- which carries team --
always wins the creation race. Latent, undefended, empirically never
fires. Mines and containers are immune by construction: their
factories require every field.[^7]

[^7]: First-sight probe over runs/bot/bot-20260802-205105,
    runs/bot/bot-20260803-180918, runs/sniff/sniff-20260620-190228
    (2026-08-04): per-tank first message type and
    first-sight-to-first-position gap, decoded through
    `protocol.decode_message` -- method in wiki log 2026-08-04.

## How the historical bug manifested

Before 2026-06-19, `update_tank_damage` (now deleted) advanced
`last_wire_seen_ms` on every damage-only TankStatusSync. A tank that
teleported out of viewport stopped producing position-bearing messages,
but the global status broadcast kept the wire-presence stamp fresh.
The kill-shot gate -- which had only `last_wire_seen_ms` to check --
never tripped, and the bot kept firing at the stale registry tile.[^2]

Run `bot-20260619-050303` recorded **25 combat_miss events on the same
target (orange-8) at the same tile (155,155) over ~100 seconds**, all
marked `target_moved=false` because the registry kept agreeing with
the bot's stale belief about the target's location.[^2]

The three-timestamp model + the `apply_tank_observation` chokepoint
+ the kill-shot position-freshness gate make this class of bug
impossible to reintroduce without breaking a locked invariant test in
`tests/world_state/test_tank_observation.py`.[^1]

## Per-message contract

For each wire message kind, the dispatcher in
`sniffer/world_state_tanks.py` builds a `TankObservation` with:[^1]

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
position -- wire-with-position or MAP_DATA snapshot.[^1]

## Registry lifecycle: 0x58 TankRemove is a no-op (changed 2026-06-22)

Tanks enter `world["tanks"]` on first observation (`apply_tank_observation`)
and leave only when `0x41 Deactivation` flips `liveness="deactivated"` and
the kill cooldown elapses. **`0x58 TankRemove` is a no-op** -- the entry
stays put. Earlier behaviour deleted the entry; that caused the bot to
abandon pursuit of locked targets that merely teleported out of viewport
(live capture 2026-06-22: bot fired exactly one homing then dropped the
lock because `0x58` fired in the next tick).[^4]

Rationale: `0x58` carries no information that the freshness gates above
can't already derive. A tank that has truly left the world stops
broadcasting `0x2E TankStatusSync`; `timestamp_ms` ages out naturally.
A tank that simply teleported keeps broadcasting -- pursuit fires
homing toward it, server picks homing weapon, homing tracks.
The only authoritative death signal is `0x41`, and the lifecycle is now
gated entirely by `liveness` ([[deactivation-format]]).[^1]

**Correction (2026-07-03):** the original rationale claimed off-viewport
tanks stop producing position-bearing messages, so the registry would
hold the last on-viewport coordinate and pursuit aims would stay legal
by construction. That is false: `0x3D MovementResponse` broadcasts
position for **every tank on the map** every ~2 s, so a pursued
target's registry coordinates track its true off-viewport tile. Live
run 2026-07-03 20:34: the registry followed orange-4 to (143,237), 5
rows below the viewport, and five pursuit shots at that aim drew 0x52
code-0 rejections. Aim legality is now enforced at the dispatch
boundary instead (`_clamp_aim_into_viewport` in `combat_strategy.py`
-- see [[shot-range]] and [[bot-behavior-contract]] 3.3); the registry
keeps the truth.

## Tests that lock the contract

Every rule above is pinned by a test in
`tests/world_state/test_tank_observation.py`:[^1]

- `TestInvariantTimestampAlwaysAdvances` -- rule 1.
- `TestInvariantWireSeenRequiresWire` -- rule 2.
- `TestInvariantPositionFreshnessRequiresAuthoritativePosition` -- rule 3.
- `TestStorageSourceIsRecorded` -- the `storage_source` that drives rule 4.
- `TestFieldMergeSemantics` -- present overwrites, `None` preserves.
- `TestTankCreationOnFirstObservation` -- the create path.
- `TestOuterTimestampAdvances`, `TestTankObservationCodec` -- outer-stamp
  and round-trip coverage.

The combat-side regression tests for the kill-shot gate live in
`tests/bot/ai/test_combat_strategy.py::TestKillShotWireGate` and
`tests/integration/test_combat_gates.py` (which exercises
`WIRE_PRESENCE_TTL_MS` and `POSITION_FRESHNESS_TTL_MS` directly);
target stickiness across the wire TTL is pinned by
`tests/scenarios/test_target_stickiness.py`. Removing or weakening any
of these tests is a deliberate contract change and must come with a
docstring and an update to this page[^9].

[^1]: code truth: `state/mutations.py` (`apply_tank_observation`) + `state/types/tank_observation.py` + `bot/ai/threats.py` gates + `sniffer/world_state_tanks.py` dispatcher (`src/tankpit_bot/state` blob-pinned in frontmatter); invariants locked by `tests/world_state/test_tank_observation.py` and `tests/bot/ai/test_combat_strategy.py`
[^2]: runs/bot/bot-20260619-050303.capture_session.json (frontmatter-pinned) — the 25-miss stale-registry loop; three-timestamp fix landed 2026-06-19/20
[^3]: `make shadow` sync-cadence law (`src/tankpit_bot/validate/shadow_laws.py`), calibration sweep 2026-07-22 over 245 archive sessions
[^8]: decisive watch capture `bot_watch_probe.capture_session.json` (2026-07-24): 617 s adjacent to purple-2; received t>60 s = 188 self 0x2E, 0 other-tank 0x2E; see [[server-push-gating]] for the seven-run proof
[^4]: live capture 2026-06-22 — one-homing lock-drop incident that motivated the 0x58 no-op change
[^9]: Verified present 2026-07-31: `tests/bot/ai/test_combat_strategy.py` (with `class TestKillShotWireGate` at `:984`), `tests/integration/test_combat_gates.py` (which imports `POSITION_FRESHNESS_TTL_MS` and `WIRE_PRESENCE_TTL_MS` at `:26-27` and describes the gate at `:10`), and `tests/scenarios/test_target_stickiness.py`.
