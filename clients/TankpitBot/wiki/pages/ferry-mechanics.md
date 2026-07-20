---
title: Ferry Mechanics
tags: [ferry, movement, terrain, water]
related: [[viewport-frame]], [[teleport-mechanics]], [[fuel-system]]
sources: [see footnotes]
fact_checked: 2026-06-12
confidence: high
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
(`SurfaceRouteTerrain` gate in `movement.py`) — plain ground when
standing on land, water/ferry tiles when riding. A container floating
on water picks up normally from the ferry (user 2026-07-20: "cant you
just pick it up essentially like we were on land?"); a land container
beyond adjacency service from the water gets a piloted disembark move
first (surface-clamped to the first land tile), and the next tick
dispatches the pickup from solid ground. Cardinal-adjacency service
crosses the surface boundary in both directions (shore tile ↔ adjacent
floating container), matching the reachability layer's long-standing
adjacent-tile completion rule.[^5]

[^4]: live falsification chain, run 2026-07-19 18:19: pickup at (163,44) dispatched while riding a ferry at (167,40) (riding rule made the channel "reachable"), server routed to the disembark stop (167,44) and refused with 0x52 code 1. Offline reproduction: ferry-aware gate=True, single-surface gate=False (the container sat 4 tiles inland — beyond adjacency), matching the server. Fix pinned by `tests/bot/ai/test_movement.py::TestPickupSurfaceRouting`.
[^5]: run 2026-07-20 00:57 (bot-20260720-005424): the first fix's ground-ONLY gate was overbroad — equipment on a water tile at (226,196) was never "ground-reachable", so the disembark branch sailed the bot onto the container's own tile and then re-issued a refused move (0x52 code 6) to its own position every tick for 78 ticks (half the session). User contract: containers on water pick up normally while riding. Gate replaced by the surface-matched `SurfaceRouteTerrain`; regression pinned by `test_pickup_of_water_container_while_riding_dispatches` and `test_pickup_on_own_water_tile_while_riding_dispatches`. Whether the server honors cross-surface ADJACENT pickups (clicking a land container from the alongside ferry tile) is untested on the wire; the planner currently assumes yes, symmetric with the land→water-container case.

## Discovery (2026-06-12)

The "marooned one-tile island" from run 131003 was actually the tank standing ON A FERRY in a lake. It could have driven across the water the whole time. The walkability model treated all water as impassable. Fixed by making passability ferry-aware.[^3]

[^1]: user (Austin), 2026-06-12 — "ferries go anywhere on water; boarding + ferry-to-land each consume a queue slot"
[^2]: terrain type constants in state/terrain.py
[^3]: run 131003 2026-06-12 — tank at (131,182) on ferry tile, 87 fuel, classified as "marooned" because water was impassable
