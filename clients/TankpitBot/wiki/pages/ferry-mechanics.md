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

## Discovery (2026-06-12)

The "marooned one-tile island" from run 131003 was actually the tank standing ON A FERRY in a lake. It could have driven across the water the whole time. The walkability model treated all water as impassable. Fixed by making passability ferry-aware.[^3]

[^1]: user (Austin), 2026-06-12 — "ferries go anywhere on water; boarding + ferry-to-land each consume a queue slot"
[^2]: terrain type constants in state/terrain.py
[^3]: run 131003 2026-06-12 — tank at (131,182) on ferry tile, 87 fuel, classified as "marooned" because water was impassable
