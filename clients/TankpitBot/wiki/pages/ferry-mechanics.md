---
title: Ferry Mechanics
tags: [ferry, movement, terrain, water]
related:
  - "[[viewport-frame]]"
  - "[[teleport-mechanics]]"
  - "[[fuel-system]]"
source_paths:
  - "runs/sniff"
  - "src/tankpit_bot/sim/movement.py"
source_git_blobs:
  "src/tankpit_bot/sim/movement.py": "b94c2a79cf9896fb351d2804c6242fe8669cee2f"
fact_checked: "2026-08-07"
confidence: high
hubs: [game-mechanics]
---

# Ferry Mechanics

## Core behavior

Ferries can go **anywhere on water** — free movement, not fixed tracks. While the tank is on a ferry, water tiles are drivable; the ferry moves with the tank.[^1]

## Queue slot costs

- **Boarding** (walking onto a ferry tile): consumes one queued action[^1]
- **Ferry-to-land**: the tank takes exactly **one step onto land**, then STOP. A fresh move command is needed to continue. Bot must expect early arrival at shore, not the requested target.[^1]

## Terrain encoding

`TERRAIN_FERRY = 5`, `TERRAIN_FERRY_ROCK = 7`, ASCII `~` in world-state dumps.[^2]

## Passability rules

- Ground or ferry tile: always passable
- Water: passable only if the tank's CURRENT tile is a ferry (riding)
- Plan land targets across water expecting the one-step stop[^1]

## Single-command routing contract (2026-07-19)

One command NEVER chains surfaces — the server routes each click on
the current surface only. User (verbatim, 2026-07-19): "if youre on
land, and there is a ferry touching land, and you click onto the
water, [it'll] path you towards where you clicked and say 'you cant
reach that' — it doesnt auto route to the ferry. if you click onto
the ferry youll walk onto it. and then you can click onto water and
itll path there fine. now, if youre on water, on the ferry, and you
click onto land, the ferry will path you to the shore. you will step
off the ferry and stop at the first land tile. you would need to
click again to reach your destination land tile. it takes two
actions because you have to embark and disembark."[^4]

Planner consequence: **pickup dispatches route on the CURRENT surface**
(`SurfaceRouteTerrain` gate at `src/tankpit_bot/bot/ai/movement.py:325`,
`SurfaceRouteTerrain(terrain, water=riding)` — the PLANNER's movement
module, not the sim's `sim/movement.py` referenced elsewhere on this
page) — plain ground when
standing on land, water/ferry tiles when riding. A container floating
on water picks up normally from the ferry (user 2026-07-20: "cant you
just pick it up essentially like we were on land?"); a land container
beyond adjacency service from the water gets a piloted disembark move
first (surface-clamped to the first land tile), and the next tick
dispatches the pickup from solid ground. Cardinal-adjacency service
crosses the surface boundary in both directions (shore tile ↔ adjacent
floating container), matching the reachability layer's long-standing
adjacent-tile completion rule.[^5]

[^4]: live falsification chain, run 2026-07-19 18:19: pickup at (163,44) dispatched while riding a ferry at (167,40) (riding rule made the channel "reachable"), server routed to the disembark stop (167,44) and refused with 0x52 code 1. Offline reproduction: ferry-aware gate=True, single-surface gate=False (the container sat 4 tiles inland — beyond adjacency), matching the server. Fix pinned by `tests/bot/ai/test_movement_exploration.py::TestPickupSurfaceRouting`.
[^5]: run 2026-07-20 00:57 (bot-20260720-005424): the first fix's ground-ONLY gate was overbroad — equipment on a water tile at (226,196) was never "ground-reachable", so the disembark branch sailed the bot onto the container's own tile and then re-issued a refused move (0x52 code 6) to its own position every tick for 78 ticks (half the session). User contract: containers on water pick up normally while riding. Gate replaced by the surface-matched `SurfaceRouteTerrain`; regression pinned by `test_pickup_of_water_container_while_riding_dispatches` and `test_pickup_on_own_water_tile_while_riding_dispatches`. Whether the server honors cross-surface ADJACENT pickups (clicking a land container from the alongside ferry tile) is untested on the wire; the planner currently assumes yes, symmetric with the land→water-container case.

## Executable since 2026-07-22

The whole contract runs in the simulator (`sim/movement.py` law 2b:
surface-gated routing, boarding/disembark truncation, the ferry
following its rider; `SimServer` patches ferries as 0x5A wire terrain
with explicit reverts) and is proven against the production ingestion
over real wire bytes — see the ferries as-built in
[[physics-module-roadmap]].

## Ferry scenario in the sim (2026-08-01)

The first ferry-SEEDED sim world: `tankpit-sim-run --ferry`
(`sim/run.py::make_ferry_sim_world`) builds the real field01 lake at
(112,112) — client on the west shore at (106,112) with 400 fuel, a
700-volume container floating water-locked 6 east (inside the join
window, so the landing radar believes it), the ferry idling at
(118,112) on its own water (min land distance 3 — the doctrine's
"own area in the water" shape) OUTSIDE the join window, land stock
deliberately too lean to reach hunt readiness. The proven chain, all
through the production bot: land forage -> radar belief -> larder
`no_landing` -> **scope-scout east pan** ([[viewport-shift-protocol]])
-> ferry arrives as a terrain-5 patch -> larder hop `ferry_served`
teleports ONTO the ferry -> the held fuel lock rides 6 tiles of open
water to the container -> auto-pickup -> full tank at 1100. Pinned by
`tests/sim/test_run.py::test_ferry_session_scouts_boards_and_drains_the_water_larder`.

Sim law completed for it: a live ferry tile is a LEGAL teleport
landing even though its water is not
(`sim/actions.py::_tile_blocked_for_landing` — boarding by teleport
is the doctrine's core move). Production law completed for it (F5
completion): a container floating on water is NEVER walk territory —
the larder keeps in-viewport water containers instead of ceding them
to the walk step that can't reach them (`larder._is_walk_territory`).[^ferrylaw]

## Movement announcements and the no-drift result (measured 2026-08-04)

How the wire states ferry motion, from the archive-wide sweep
(`analysis_scripts/mine_ferry_drift.py` + the full-archive pass, 312
captures):[^7]

- A ferry's POSITION is restated by 0x5A viewport repaints (the
  2026-07-20 ride's ferry showed at (223,195) across three repaints,
  stationary for 14 s while unridden).
- A ferry's MOVEMENT is **one atomic 0x4A message** carrying both
  halves — old tile restored to water, new tile painted ferry:
  ``[(223,195,0), (226,196,5)]``. Moves are rider-move-sized LEGS
  (Manhattan 1-12 observed), not single steps.
- **148 distinct moves across the archive; 136 are rider-attributed**
  (a tank stated the departing or arriving tile within 2.5 s). The 12
  residuals are isolated singles with no cadence (mostly one per
  session, gaps 12-56 s) — the signature of riders whose positions
  happened not to be wire-stated in the window, not of an autonomous
  behavior. **No ferry drift law exists in the archive**; the sim's
  rider-following model is validated at scale and needs no drift
  term. If a future session parks beside an unridden ferry for
  minutes and its 0x4A stays silent, that closes the residual to
  zero.

**The unfinished-command close (live 2026-08-03):** a
surface-transition stop SHORT of the click gets a code-1 close in
the same batch as the truncation echo — the 08-03 run's cluster-A
collects (bot riding the ferry afloat on (59,28) water, land targets
inland) each echoed the one-step disembark then the 0x52. A
transition stop that IS the click (boarding the clicked ferry tile)
closes silently, and a mine walk-over arrest closes silently too (18
archive detonations, zero paired code-1s). The sim emits all three
shapes (`sim/emissions.py::emit_move`,
`MoveOutcomeDict.stop_reason`/`dest_reached`).[^ferrylaw]

[^7]: `analysis_scripts/mine_ferry_drift.py`, run over all 312 archive
    captures; sweep recorded at `wiki/log.md:2678-2682` (entry
    "[2026-08-04] crack + lift | Ferry movement law mined"). Classifier:
    every 0x4A leave/arrive pair (Manhattan <= 40) against every tank
    position statement (0x28/0x3D direct, 0x47 echo finals) within
    +-2.5 s. The log entry records the same 148 moves / 136
    rider-attributed split this section states, and notes that the
    first classifier pass wrongly read "all unridden" because it
    matched riders against the OLD tile only.

## Autoscroll riding doctrine (user ruling 2026-08-01, OPEN)

User, verbatim: "tbh when i use ferries i use auto scroll on so i
can ride it across multiple viewports. also when i forage with no
extra radars, i use auto scroll on." The sim scenario's ride fits
one window; a REAL long harvest ride crosses several, which needs
autoscroll ON (edge-arrival recenter, one tick per recenter) for the
duration of the ride — with it OFF the ride's move legs die at the
stored window edge. Dynamic `Ia` toggling (ON when boarding for a
beyond-window ride or foraging radar-broke, OFF otherwise) is the
queued next build; details in [[viewport-shift-protocol]].

## Forage platform doctrine (2026-07-29)

User (verbatim, flag-11 narration, 2026-07-29): "ferries are actaully
the best way to get fuel and equipment, since you can use them to
access many equipment and fuel cannisters yu other wise couldnt. you
generally will need to teleport to the ferry since many times it will
be on its own area in the water. not touching land. but they are very
good to use."[^6] Consequences the planner does not yet implement:
water-locked containers (counted `no_landing` by the larder scorer)
are harvestable by teleporting TO a ferry and riding it; ferries are
often unreachable by walking (their water body touches no shore path),
so the approach is a teleport, not a board-from-land. Tracked as F5 in
[[flag-triage-20260729]].

Combat corollary, proven live the same night: a ferry RIDER sits on a
water tile with zero passable neighbors, which cloaked a human from
acquisition until the stand-off gate replaced strict adjacency (F4 in
[[flag-triage-20260729]]).

[^6]: user (Austin), 2026-07-29, mid-run flag 11 debrief of
    bot-20260729-232252.

## Discovery (2026-06-12)

The "marooned one-tile island" from run 131003 was actually the tank standing ON A FERRY in a lake. It could have driven across the water the whole time. The walkability model treated all water as impassable. Fixed by making passability ferry-aware.[^3]

[^1]: Originally a user (Austin) statement of 2026-06-12, made in conversation and not recorded in `wiki/log.md` (which opens 2026-06-16 at `log.md:7`). Both halves are now independently held by artifacts in the repo: the go-anywhere-on-water and queue-slot rules are implemented as sim law 2b in `src/tankpit_bot/sim/movement.py:12-16` — "a ferry tile boards; while riding, water is open sea and the ferry moves with the tank. The FIRST queue-consuming surface transition — stepping onto a ferry (boarding) or from water/ferry onto land (disembarking) — STOPS the move at that tile" — and the same contract was restated verbatim by the user on 2026-07-19 and falsified live against the server, which is [^4]. Cite [^4] for the wire proof; this note records provenance only.
[^2]: `src/tankpit_bot/types/constants.py:25-26` — `TERRAIN_FERRY = 5`, `TERRAIN_FERRY_ROCK = 7`; the ASCII glyph is `ASCII_FERRY = "~"` at `:47`. **Corrected 2026-08-05:** this footnote previously pointed at `state/terrain.py`, which does not define them (the module at `src/tankpit_bot/state/types/terrain.py` is a different file, and `src/tankpit_bot/terrain.py` is a third). **Moved 2026-08-07:** the module is now `types/constants.py`, not `state/types/constants.py` — it was never state, and while it sat under `state/` it made `physics` and `state` mutually dependent ([[package-layering]]). The three line numbers are unchanged; only the package moved.
[^3]: `runs/bot/bot-20260612-131003.log:7130-7160` (run 131003, 2026-06-12 13:19:08-13:19:10). The bot teleported toward a fuel dot at (131,182) for 81 fuel (`WORLD: Fuel: 168 -> 87 (-81)`) and the server placed it at **(132,180)**, whose rendered viewport row 180 reads `W W W W @ W # #` — water on both sides, rock below, i.e. the one-tile island. **Corrected 2026-08-05:** this footnote previously gave the tank's position as (131,182); that is the fuel container's tile and the teleport TARGET, rendered `F` on row 182 of the same dump. The tank was never there. "Marooned" is the bot's own state name for the condition (`src/tankpit_bot/bot/ai/collect_hops.py:365`), not a string in this log.
[^ferrylaw]: The three-question split a ferry surface forces — passability, landing legality, landing attainability — is `is_landing_legal` at `src/tankpit_bot/bot/ai/ferry.py:163` and `:308`, beside `is_landing_attainable` at `:189` and `:326`; see [[terrain-composition]] for the full table. The 2026-08-03 unfinished-command receipts are `runs/bot/bot-20260803-180918.capture_session.json` and its `.events.jsonl`. All paths verified present 2026-08-07.
