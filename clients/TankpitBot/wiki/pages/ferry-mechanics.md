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

Planner consequence: **pickup dispatches route on plain ground only**
(`GroundOnlyTerrain` gate in `movement.py`) — the riding rule applies
to piloted moves, never to server-routed `pickup_*` clicks. A land
container that is only ferry-reachable while riding gets a piloted
disembark move first (surface-clamped to the first land tile); the
next tick dispatches the pickup from solid ground.

[^4]: live falsification chain, run 2026-07-19 18:19: pickup at (163,44) dispatched while riding a ferry at (167,40) (riding rule made the channel "reachable"), server routed to the disembark stop (167,44) and refused with 0x52 code 1. Offline reproduction: ferry-aware gate=True, ground-only gate=False, matching the server. Fix pinned by `tests/bot/ai/test_movement.py::TestPickupSurfaceRouting`.

## Discovery (2026-06-12)

The "marooned one-tile island" from run 131003 was actually the tank standing ON A FERRY in a lake. It could have driven across the water the whole time. The walkability model treated all water as impassable. Fixed by making passability ferry-aware.[^3]

[^1]: user (Austin), 2026-06-12 — "ferries go anywhere on water; boarding + ferry-to-land each consume a queue slot"
[^2]: terrain type constants in state/terrain.py
[^3]: run 131003 2026-06-12 — tank at (131,182) on ferry tile, 87 fuel, classified as "marooned" because water was impassable
