---
title: Terrain Composition — Single-Owner Walkability
tags: [architecture, terrain, mines, ferry, planner, executor]
related:
  - "[[ferry-mechanics]]"
  - "[[mine-mechanics]]"
  - "[[teleport-mechanics]]"
  - "[[executor-rejection-loops]]"
  - "[[self-observing-architecture]]"
  - "[[bot-behavior-contract]]"
source_paths:
  - "src/tankpit_bot/terrain.py"
  - "tpclient.js"
source_git_blobs:
  "src/tankpit_bot/terrain.py": "32901c5ed745e8b8df212b5f388961b461cfcd3c"
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-07-31"
confidence: high
hubs: [architecture]
---

# Terrain Composition — Single-Owner Walkability

The question **"can the tank walk onto this tile?"** has exactly one
owner: the composed decision terrain built once per tick by
`compose_decision_terrain` (`bot/ai/ferry.py`). Every passability
consumer — A* pathfinding, viewport reachability, hop selectors,
surface clamps, the pickup routing gate, stall-clearing — reads the
same `is_passable`. No consumer receives walkability facts through any
other channel. This page records why that rule exists, what it
replaced, and the invariants that keep it true.[^3]

## The composed view, layer by layer

```
SurfaceRouteTerrain(view, water=riding)     <- pickup routing only:
  |  intersects base passability with          one surface per click
  |  the current routing surface               (user contract 2026-07-19/20)
  v
FerryAwareTerrain(base, wire, riding, hostile_mine_keys)
  |  1. hostile mine key -> impassable        (composed 2026-07-20)
  |  2. live wire ferry tile -> "~", passable (composed 2026-06-12)
  |  3. water -> passable iff riding
  |  4. ground -> passable, rock -> not
  v
TerrainMap (static minimap, decoded from the field GIF)
```

`compose_decision_terrain(world, terrain)` assembles this per tick
from three world-state inputs: the static map, the wire terrain
overlay (`world["terrain"]`, ferries), and
`hostile_mines(world)` (`bot/ai/equipment.py` — the team filter:
same-team mines are passable per [[mine-mechanics]], enemy mines are
not). `DecideCtx.terrain` carries the composed view into every
planner; `tick_loop_actions` composes the same view for its
stall-clearing reachability checks.

## Why mines are terrain (physics)

- **Walking onto a hostile mine detonates it: 45 fuel to the victim**
  ([[game-economy]], wire-verified 2026-06-20). A mined tile is
  therefore never walkable — the same class of fact as rock.
- **Teleporting AT a mined tile is safe: the server displaces the
  landing to the nearest open tile** ([[teleport-mechanics]]
  Placement, user contract 2026-06-16). Teleport-landing legality is
  a DIFFERENT question from walkability and is deliberately NOT
  answered by this view — `find_teleport_landing_tile` stays
  mine-blind by design, and mined teleport targets dispatch freely.
- **Pickup adjacency service crosses the mine**: a container ON a
  mined tile is collected from a cardinally adjacent tile (the
  composed view makes the container tile impassable; the reachability
  layer's adjacent-landing rule services it). Observed live: the bot
  drained the mined fuel dot at (37,153) from adjacency in run
  bot-20260720-165935 while its planner was simultaneously failing to
  hop to it — see History below.

## What this replaced (the two-owner era)

Before 2026-07-20, hostile mines traveled beside terrain as a
`blocked_mines` parameter that every consumer had to remember to
thread into pathfinding and reachability calls. Pathfinding
remembered. Reachability remembered. The dot-hop selector
(`_pick_fresh_dot_hop`) did not — it consulted `terrain.is_passable`
only. The executor compensated with a pre-dispatch veto
(`_is_valid_move_destination`, added 2026-04-10) that discarded any
move/teleport whose destination carried a hostile mine.[^3]

That fork produced the fixed-point loop of run bot-20260720-165935
(17:16): planner proposes the mined dot (its rules pass), executor
vetoes (its rule fails), nothing feeds back, world unchanged, repeat —
23 consecutive ticks, ~46 s, until the session clock ended it.
[[executor-rejection-loops]] had predicted exactly this instance three
days earlier ("if a container ever ended up on a mine tile the
executor's mine rejection would silently loop the pickup pursuit").

The user's ruling set the fix scope: *"no no no. not anti loop, but
something that addresses the root of the issue."*[^5] Feedback wiring
(attempt marks, discard-marks-failed) would have made the loop short;
only removing the second owner makes it impossible.

## The cut (2026-07-20, commit 6d2afdbe)

1. `FerryAwareTerrain` gained a **required** `hostile_mine_keys:
   frozenset[str]` constructor argument. Required, not defaulted: a
   caller cannot construct the view while forgetting mines exist.
   `is_passable` checks the mine set before terrain class. Display
   (`get_terrain` / `render_viewport`) is unchanged — a mine is a
   passability fact, not a terrain character.
2. `blocked_mines` / `blocked_coords` parameter threading deleted
   end-to-end: `pathfinding.py` (including `_is_blocked_coord`),
   `reachability.py` (including per-tile mine checks on pickup targets
   and landings — now implied by `is_passable`),
   `equipment_search.py`, `movement.py`, the ferry clamp,
   `tick_loop_actions.py`, `action_lab` targeting probes.
3. `SurfaceRouteTerrain.is_passable` now **intersects** the wrapped
   view's passability with the surface class, so mine-blocking (and
   any future composed fact) propagates through pickup routing.
4. `tick_loop_actions` stall-clearing switched from the raw static
   map to `compose_decision_terrain` — it had been a third private
   physics (neither ferry- nor mine-aware) and is now the same view
   the planner used to make the decision being checked.
5. **Executor `_is_valid_move_destination` deleted**, with its
   `discarded_hostile_mine` ledger outcomes and emitters. For walks
   it became unreachable (the planner cannot emit a destination that
   does not exist in its terrain). For teleports it was wrong physics
   (server displacement) — deleting it also resolved
   [[executor-rejection-loops]] instance #1, the combat teleport at a
   mined enemy tile, which the veto would have looped in exactly the
   same way.

Net diff: 26 files, +358/−318 — the system got smaller.[^3]

## Invariants (and where they are pinned)

| Invariant | Pinned by |
|---|---|
| Hostile mine tile impassable on ground, water, and ferry cells | `test_ferry.py::test_hostile_mine_tile_is_impassable_on_any_surface` |
| Composition takes enemy mines, skips same-team mines | `test_ferry.py::test_compose_folds_hostile_mines_but_not_friendly` |
| Mine-blocking propagates through both pickup routing surfaces | `test_ferry.py::TestSurfaceRouteTerrain::test_mined_tile_blocks_both_surfaces` |
| A* routes around composed mines; direct-line and segment helpers see them | `test_pathfinding.py` (3 rewritten mine tests) |
| Pickup on a mined container tile services from adjacency | `test_reachability.py::test_collection_reachable_from_adjacent_landing_tile` |
| Dot-hop selector never proposes a mined dot (the 17:16 loop) | `test_resource_search.py::test_skips_dot_on_hostile_mine_tile` |
| A teleport at a mined tile IS dispatchable (displacement physics) | `test_executor.py::test_teleport_to_mined_tile_is_dispatchable` |

The general rule for future dynamic impassability (new obstacle
classes, temporary hazards): compose them into
`compose_decision_terrain`, never add a parameter beside it. The ferry
overlay (2026-06-12), the mine overlay (2026-07-20), and the
movable-block overlay (2026-07-20 same day — wire terrain values
1/2/3 collapse to ground/rock by walkability, see [[movable-blocks]])
are the precedents; the required-kwarg pattern is the enforcement.

## Verification soak (bot-20260720-192320)

150 ticks: 2 kills, 23/23 hits, 0 rejected, 0 blocked, **0 discards,
0 error-code-6 move refusals (previous soak: 77), 0 mine detonations,
0 terrain-blocked replans**, analyzer clean ("no top-level issues
detected" — first clean analyzer verdict in three soaks). 77 mine
tiles appeared in viewport renders during the session; the bot
navigated all of them without incident. Longest identical-decision
run: 13 consecutive shoot commands at one target — sustained combat,
not a stall.[^2]

## What the executor still validates (RESOLVED 2026-07-21)

`_is_valid_pickup` (container gone / kind mismatch), `_is_valid_shoot`
(target no longer tracked), `_is_valid_teleport` (stale combat or
resource anchors). These guard the planner's **cross-tick carried
state** (locks and anchors that can outlive their world-state
justification), not same-snapshot physics — a different problem from
the one this page closes. They were instances #2 and #3 of [[executor-rejection-loops]] and
were deleted 2026-07-21 after their unreachability was proven
(synchronous tick + decide-time lock normalization + impossible
source value). The executor is now pure dispatch — the end state
[[self-observing-architecture]] Phase 1 named.

[^1]: Loop capture: `runs/bot/bot-20260720-165935.events.jsonl` ticks 128–150 — 23 × `hop_selected` (37,153) → `action_outcome outcome=discarded_hostile_mine duration_ms=0`, fuel and inventory unchanged throughout.
[^2]: Verification soak: `runs/bot/bot-20260720-192320.events.jsonl` (latest.* at time of writing); analyzer output "no top-level issues detected"; discard/error/replan counts measured by direct event scan 2026-07-20.
[^3]: Executor veto origin: commit dc696282 (2026-04-10) "Add executor-side command validation against current world state". Mine-veto deletion + composition: commit 6d2afdbe (2026-07-20). Ferry composition precedent: 2026-06-12, run 131003 ("marooned island" that was a ferry).
[^4]: Displacement physics: user (Austin) 2026-06-16 — "you get moved off if there are mines, or if there is terrain in the way"; documented in [[teleport-mechanics]] Placement.
[^5]: user (Austin), 2026-07-20 — root-cause ruling on the fixed-point loop, quoted verbatim above.
