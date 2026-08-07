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
  "src/tankpit_bot/terrain.py": "ba99f673e65e33e17bb0cd81eacb8165a676d066"
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-08-06"
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
FerryAwareTerrain(base, wire, riding, hostile_mine_keys,
  |                                     occupied_tank_keys)
  |  1. hostile mine key -> unwalkable        (composed 2026-07-20)
  |  2. tank-body key -> unwalkable           (composed 2026-08-04)
  |  3. wire block 1 -> bridge, passable      (composed 2026-07-20)
  |     wire block 2/3 -> obstacle, not
  |  4. live wire ferry tile -> "~", passable (composed 2026-06-12)
  |  5. water -> passable iff riding
  |  6. ground -> passable, rock -> not
  v
TerrainMap (static minimap, decoded from the field GIF)
```

`compose_decision_terrain(world, terrain, now_ms)` assembles this per
tick from four world-state inputs: the static map, the wire terrain
overlay (`world["terrain"]` — ferries AND movable blocks),
`hostile_mines(world)` (`bot/ai/equipment.py` — the team filter:
same-team mines are passable per [[mine-mechanics]], enemy mines are
not), and `occupied_tank_keys(world, now_ms)`
(`state/occupancy.py` — other tanks' bodies, viewport-fresh only).
`DecideCtx.terrain` carries the composed view into every planner;
`tick_loop_actions` composes the same view for its stall-clearing
reachability checks.

## Two questions, not one

The view answers **two** questions, and they are not the same
question ([[walk-mechanics]], [[flag-triage-20260729]] F6):

| Method | Question | Blockers respected |
|---|---|---|
| `is_passable` | Can the tank WALK onto this tile? | terrain, blocks, mines, tank bodies |
| `is_landing_legal` | May the server PLACE the tank here? | terrain, blocks only |
| `is_landing_attainable` | Will a teleport aimed here actually STAND here? | terrain, blocks, team-scoped hostile mines |

**Three questions as of 2026-08-06**, not two. `is_landing_attainable`
(`bot/ai/ferry.py:187`, composed-view override at `:324`) is landing
legality intersected with the composed view's TEAM-SCOPED hostile-mine
set — the same set the walk side consumes, built once per tick from the
self model's team. Own-colour mines never displace a landing and are
absent from that set by construction ([[mine-mechanics]] § team scope;
archive 2026-08-06 measured 1,227 enemy against 2 friendly). The
distinction from `is_landing_legal` is aim versus outcome: a teleport at
an enemy-mined tile is still *legal* to dispatch and is not refused, but
it will not leave the tank on that tile.

The split exists because a teleport aimed at a mined or occupied tile
is **not refused** — the server displaces the landing to an adjacent
tile and charges the plain cost ([[mine-mechanics]], live-proven
2026-07-28). Landing selection therefore must never ask the walk
question: an enemy always occupies its own tile, so doing so silently
downgrades every direct approach teleport into a stand-off and
abandons enemies whose neighbours are terrain-blocked.

`find_teleport_landing_tile` and both choosers in `combat_landing.py`
ask `is_landing_legal`; pathfinding, reachability, and the move clamps
ask `is_passable`. Both are on `TerrainMapProtocol`, so an
implementation cannot satisfy the protocol while answering only one.[^land]

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
[^4]: Displacement physics. Originally a user (Austin) statement of 2026-06-16 — "you get moved off if there are mines, or if there is terrain in the way" — made in conversation, with no transcript in the repo. The law it states is now carried by the code and readable there: `src/tankpit_bot/terrain.py:151-165`, whose docstring gives the reason the two predicates can share an implementation on the static view — "The static minimap carries no mines and no tank bodies -- the only blockers a landing is exempt from -- so on this view the two questions have the same answer. Views that compose dynamic blockers (`FerryAwareTerrain`) are where they diverge." The diverging views are `src/tankpit_bot/bot/ai/ferry.py:161` and `:306`. Also documented at [[teleport-mechanics]] Placement.
[^5]: User (Austin) root-cause ruling of 2026-07-20 on the fixed-point loop, quoted verbatim above; the conversation is not recorded in the repo. What IS on record: `wiki/log.md:1390`, which narrates the two-owner era and its fixed-point loop and pins the fix as commit `6d2afdbe` (26 files, +358/−318). **The run this section names, `bot-20260720-165935`, is NOT in the archive** — the nearest retained 2026-07-20 captures are `bot-20260720-005424` and `bot-20260720-171140` — so the "23 consecutive ticks, ~46 s" figures cannot be re-derived from an artifact and rest on the contemporaneous narrative alone.
[^land]: `find_teleport_landing_tile` at `src/tankpit_bot/bot/ai/equipment_search.py:33`; the `is_landing_legal` question both it and the combat choosers ask is implemented per-surface, e.g. `src/tankpit_bot/bot/ai/ferry.py:163` and `:308`. The combat chooser is `choose_combat_landing_tile` at `src/tankpit_bot/bot/ai/combat_landing.py:80`, whose docstring states why the question is landing-legality and never passability. Verified present 2026-08-07.
