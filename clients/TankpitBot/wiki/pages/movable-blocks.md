---
title: Movable Concrete Blocks
tags: [terrain, blocks, game-mechanics, combat]
related:
  - "[[ferry-mechanics]]"
  - "[[terrain-composition]]"
  - "[[mine-mechanics]]"
  - "[[weapon-selection]]"
  - "[[js-source-map]]"
source_paths:
  - "runs/sniff/sniff-20260720-214839.capture_session.json"
  - "runs/sniff/sniff-20260720-215930.capture_session.json"
  - "src/tankpit_bot/sim/blocks.py"
fact_checked: "2026-07-20"
confidence: high
hubs: [combat, game-mechanics]
---

# Movable Concrete Blocks

Pickup-and-place terrain objects. The wire encoding is **fully
cracked** (2026-07-20, two manual captures) — the 0x42 / 0x4A
decoders existed from the JS reverse-engineering but had never once
fired in 204 bot sessions until the user's manual sessions. The BOT
still has no state tracking or planner awareness of blocks (open
work below).

## Wire encoding (verified 2026-07-20)

- **Client command**: `type=4 id=98 ('b')` with target (x,y) — one
  command for both pickup and drop (long-press in the client; the
  official quick-help binds it as "Click and Hold: Pick Up / Drop").
  Server decides pickup vs drop from carry state.
- **0x42 BlockAction** (msg_type 66): `tank_id`, `source_x/y` (the
  tank's tile), `drop_x/y` (the block's tile), `direction`,
  `obstacle_type`, `flag`. `direction` is an ASCII compass letter on
  PICKUP (`e`/`s`/`n`/`w` = which side of the tank the block attached
  from) and `0` on DROP.
- **`obstacle_type` enum** (12 labeled drops + prior session):
  **1 = placed in water** (walkable bridge; DOM "Bridge module
  built"), **2 = placed on land** (obstacle; DOM "Obstacle
  dropped"), **3 = stacked on a water block** (impassable terrain).
- **0x4A tile updates** `[x, y, state]` use the same enum plus
  0 = cleared. Dragging emits transient `2`→`0` pairs along the towed
  path each step; unstacking shows the tile going `3`→`1` (terrain
  back to bridge).
- **Teleport while towing is refused**: three attempts each drew an
  immediate `0x52 error_code=0, reset_action=1` ("You can't do
  this"). A block press out of reach draws `0x52 code=1`.
- **Cost: free.** Stationary same-tile pickup/re-drop pairs produced
  zero fuel delta; all session deltas decompose into normal 1/tile
  walking while towing. Same-tile immediate re-place works, including
  on sequential ticks.
- **Mine destruction is wire-silent**: placing on a mined land tile
  destroys the mine with NO 0x45 detonation and no dedicated message
  — the mine simply ceases. And it destroys **ANY team's mine**:
  blue (own-team!), red, and purple mines all destroyed in the
  labeled capture. This refines the original contract's "enemy
  mines" — friendly mines die too.
- **Containers under blocks survive** (verified on fuel and equipment
  containers; a fuel container in water became a bridge tile with
  the container intact — and its volume stays visible in 0x5A
  `cache_value`, draining as it is picked: 730→257→0 observed).

## Arrival encoding — how resting blocks reach a fresh client (verified 2026-07-20)

Capture sniff-20260720-221239 (user-narrated: known block coords,
map open/close ×2, radars at stated positions, viewport scrolled away
and back, bridge-only equipment pickup):

- **Resting blocks arrive via ordinary 0x5A viewport patches as
  `terrain_type` values** (with the known +1/+1 entity-alignment
  offset). The three channels share ONE enum — `0x42.obstacle_type`
  ≡ `0x4A` tile value ≡ `0x5A.terrain_type`: **1 = block in water
  (walkable bridge, = TERRAIN_ROCK_A), 2 = block on land (obstacle,
  = TERRAIN_ROCK_B), 3 = stacked (terrain, = TERRAIN_ROCK_AB)**.
  Blocks are dynamic rock-family terrain; ferries (5/7) complete the
  same vocabulary.
- **Radar (0x4F) does NOT carry blocks** — containers and mines only.
- **The map (0x4C) does NOT show blocks** — user-confirmed visually
  and wire-confirmed (fuel dots only).
- **Ambiguity + disambiguation rule**: wire value 1 is shared by
  genuine rock-A (impassable) and water-blocks (walkable). The static
  map resolves it: **rock-family wire tile over static WATER =
  walkable bridge; over static land = impassable obstacle either
  way.** This is the exact rule a future ingestion needs.
- Walking on a bridge is ordinary movement (normal 0x47 waypoints,
  1/tile); pickups reachable only via a bridge dispatch normally
  (equipment at (78,183) collected across the user's built bridge).
- Caveat on the "never seen in bot sessions" sweep: bot captures DO
  contain thousands of 0x5A tiles with values 1/2/3 — genuine
  mountains, indistinguishable from land blocks by value alone (and
  equally impassable, so the ambiguity is harmless on land). The
  zero-occurrence claim stands precisely for: 0x42 events, 0x4A
  block values, and rock-family tiles over static water.

## Placement rules

- **On land** → becomes a terrain obstacle, a DIFFERENT tile type
  from the mountain-range terrain.
- **In water** → becomes WALKABLE terrain (a bridge tile).
- **On another movable block in water** → becomes terrain
  (impassable). "you can only stack them on water ofc. on land you
  can place one on the ground."
- **On land, placement destroys any enemy mines on that exact tile**
  — "not adjacent mines."
- **Can be placed on top of fuel or equipment containers** — "they
  are not destroyed."

## Moving them

- Long-press to pick one up; it drags BEHIND the tank.
- Long-press on a point to drop it at that x,y.
- "you cant turn in place while pulling it. it will autopath if you
  click somewhere, but you need to go up and over and down to turn
  around, while pulling it."

## Combat interaction

- A block in the enemy's line of sight (between you and the shooter)
  BLOCKS their shots — unless they have missiles enabled
  ([[weapon-selection]]: obstructions trigger missiles, which shoot
  over it).
- "if two people are shooting at you from opposite angles, then you
  cannot block both, only one."

## Decision 2026-07-20 (revised same day): WIRED into the composed terrain

The first sweep looked only for 0x42 manipulation events and 0x4A
block values (zero in 204 sessions) and concluded blocks were absent.
The user pushed back ("why not add blocks to the terrain tho?") and a
deeper check — every wire rock-family tile in 228 room-1 captures
tested against the static practice map — flipped the verdict:

- **value 1: 4,352 sightings, ALL over static water** (bridges — a
  persistent bridge complex sits around (130–135, 145–155))
- **value 2: 2,396 sightings, ALL over static GROUND** (land blocks —
  invisible obstacles to the old model, live in real sessions)
- **value 3: 250 sightings, ALL over static water** (stacks)

Resting blocks are common map furniture where the bot plays; only
*manipulation* near the bot is unobserved. The perfect value/background
separation also means the wire value alone determines walkability —
no static-background disambiguation needed.

Wired 2026-07-20: constants renamed to truth (`TERRAIN_BLOCK_BRIDGE`
/ `_LAND` / `_STACKED`, formerly misnamed ROCK_A/B/AB),
`FerryAwareTerrain.get_terrain` collapses wire blocks to their
walkability class (bridge → ground, land/stacked → rock) so every
passability consumer inherits them ([[terrain-composition]] pattern,
third dynamic fact after ferries and mines). World state keeps the
raw terrain_type verbatim — the collapse is only the planning
projection; the world renderer draws bridges as `=`. Pinned by
`test_ferry.py::test_block_tiles_compose_by_walkability` and
`test_bridge_is_routable_for_ground_surface_pickups`.

## Implications for the bot (remaining open work)

1. ~~Terrain model~~ DONE 2026-07-20 — composed into the decision
   terrain (see revised decision above).
2. **Mine registry hygiene**: a 0x42 land drop (obstacle_type=2) on a
   tracked mine tile must delete that mine from world state — the
   destruction is wire-silent, so nothing else will. (Dormant: 0x42
   has never fired near the bot; becomes live the day it does.)
3. **Tactical uses once modeled**: bridging water gaps for pickups
   (free, vs ferry routing), mine-clearing a tile by placement
   (including our own stale mines), shot shielding in fights (one
   line of sight only; useless against missiles), denying a
   chokepoint. Towing constraints: normal walk cost, no teleporting,
   no turning in place.

[^1]: user (Austin), 2026-07-20 — verbatim contract above, delivered alongside the missile trigger rule; wire-verified the same day via sniff-20260720-214839 (7 pick/drop pairs, DOM-aligned: pickup=dir-letter, drop=dir 0; 3 towing teleports refused 0x52 code=0) and sniff-20260720-215930 (12 labeled drops: all land contexts → obstacle_type 2 including mines and containers; water → 1; stack → 3 from the prior session; blue/red/purple mines destroyed silently; stationary re-place pairs at zero fuel).
