---
title: Teleport Mechanics
tags: [teleport, movement, fuel]
related: [[viewport-frame]], [[fuel-system]], [[map-mechanics]]
sources: [see footnotes]
fact_checked: 2026-06-12
confidence: high
---

# Teleport Mechanics

Teleport is the primary mobility model for search and hunting. Never propose replacing teleports with walking — if fuel economy looks bad, fix fuel acquisition or hop targets. Walking is only for short in-viewport combat closes.[^1]

## Placement

**Click directly on the target** — enemy tank, container, or tile. The server handles placement:[^2]
- If the tile is open, you land exactly there
- If a **tank** occupies the tile, the server places you adjacent (typically cardinal)
- If **mines** occupy the tile, displaced to nearest open tile
- If **terrain** (rocks or water) at the target, displaced to nearest open tile

**Do NOT compute adjacent tiles client-side.** The server is authoritative for placement. Teleporting to an enemy's exact coordinates is correct — the server places you adjacent. This is how human players play.[^8]

## Fuel cost

`cost = floor(6 * euclidean_distance)`. Cheapest: 6 fuel (1 tile). Diagonal = 8 fuel. Below ~8 fuel no teleport is affordable.[^3]

## Map requirement

The map must be open to teleport. `CMD_MAP_OPEN` (0x6c) opens it; teleport auto-closes it via `CMD_MAP_TELEPORT` (0x74). See [[map-mechanics]].[^4]

## Landing auto-pickup

Teleporting onto a container tile picks it up on landing.[^5]

## Timing

Map open → teleport → fire can all happen in one burst with no waits. The tank lands immediately and can fire on the next server tick.[^6]

## Server tick and queue

Server tick rate is 2000ms. Commands sent faster are queued by the server. Consecutive shots are ~2040ms apart — the server's actual shot cooldown. The server, not the bot, owns timing.[^7]

[^1]: user (Austin), 2026-06-11 — "no walking are you stupid lol"; teleport is the mobility primitive, walking only for short in-viewport closes
[^2]: user (Austin), 2026-06-16 — "you get moved off if there are mines, or if there is terrain in the way. or if there is water there, you get teleported to the nearest open space"
[^3]: fuel cost formula verified across multiple runs; floor(6 * euclidean) matches all observed fuel decrements
[^4]: discovery probe 2026-06-12 — map open/close wire behavior; see [[map-mechanics]] for full details
[^5]: fuel dot probe 2026-06-11 — 6/6 dots held fuel; sixth auto-picked on landing (fuel 639→1100)
[^6]: user (Austin), 2026-04-20 — protocol command timing; confirmed no waits needed between map/teleport/fire
[^7]: bot-20260614-142159.capture_session.json — server response latency 56ms-2002ms; 2000ms responses = server HOLDING queued command until cooldown elapses
[^8]: user (Austin), 2026-06-16 — "I teleport to the same exact position as the enemy tank. so the game puts me adjacent. you don't have to click on map to teleport right below them"; confirmed by official How To Play: "Open the map, then click on it to teleport"
