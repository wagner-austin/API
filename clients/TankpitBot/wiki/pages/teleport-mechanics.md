---
title: Teleport Mechanics
tags: [teleport, movement, fuel]
related:
  - "[[viewport-frame]]"
  - "[[fuel-system]]"
  - "[[map-mechanics]]"
source_paths:
  - "runs/bot"
  - "src/tankpit_bot/physics/costs.py"
source_git_blobs:
  "src/tankpit_bot/physics/costs.py": "cfd4ecc3b7dca858aebb7fdc7e4b5c9f93d77f62"
fact_checked: "2026-06-12"
confidence: high
hubs: [game-mechanics]
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

The distance is measured to the **actual landing tile**, not the clicked target: when the server displaces the landing (occupied tile, mines, terrain — see Placement above), the fuel charge matches `floor(6 × euclid(start, landing))` exactly. 624 drift hops confirm this; the planner's target-based estimate is off by only a few fuel on drifted hops, so no code change is warranted.[^3]

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
[^3]: Systematic validation 2026-07-20: every teleport dispatch in every `runs/bot/*.events.jsonl` paired with its wire fuel delta (pre-hop `Self:` position fix → post-hop `Fuel: A -> B` line; windows with intervening move/radar/shoot/pickup dispatches or a fuel-level mismatch excluded). Post-2026-06-24 era (after the fuel double-count fix): 248/248 pairs exact, costs spanning 6–654. All-era: 2,335/2,538 exact (1,711 on target coords + 624 on actual-landing coords); all 203 residuals live in pre-fix runs with broken fuel tracking. Supersedes the earlier undocumented "verified across multiple runs" assertion.
[^4]: discovery probe 2026-06-12 — map open/close wire behavior; see [[map-mechanics]] for full details
[^5]: fuel dot probe 2026-06-11 — 6/6 dots held fuel; sixth auto-picked on landing (fuel 639→1100)
[^6]: user (Austin), 2026-04-20 — protocol command timing; confirmed no waits needed between map/teleport/fire
[^7]: bot-20260614-142159.capture_session.json — server response latency 56ms-2002ms; 2000ms responses = server HOLDING queued command until cooldown elapses
[^8]: user (Austin), 2026-06-16 — "I teleport to the same exact position as the enemy tank. so the game puts me adjacent. you don't have to click on map to teleport right below them"; confirmed by official How To Play: "Open the map, then click on it to teleport"

## Displacement preference order (measured 2026-07-21)

User-piloted probe sniff-20260721-200527, 11 wire-verified
displacements (sent teleport target vs 0x3D landing fix):

- **The server tries the target's neighbors in a fixed ABSOLUTE
  order: EAST first, then NORTH, then WEST** — independent of
  approach direction (the mine at (17,63) was approached from the
  northwest, due north, and due west; all three landed at (18,63),
  its east neighbor; a solo mine and two rock targets also landed
  east).
- **A tile occupied by your own tank counts as blocked**: teleporting
  at (29,68) while standing on its east neighbor (30,68) landed
  north at (29,67); repeating from (29,67) landed east again.
- **West observed only when east and north were both rock**
  ((61,53) inside the rock mass → landed (60,53)).
- ~~South never isolated; search depth beyond ring 1 unobserved.
  Both remain open questions.~~ **CLOSED by the 2026-07-22 corpus
  sweep** (2,861 sent teleports paired with their 0x3D landing fixes
  across all 246 sessions, rejections excluded): 2,020 landed exact
  and 841 displaced — cardinal ring-1 dominates with **E 448 ≫ N 89 >
  S 31 ≈ W 28**. SOUTH IS REAL (31 samples), so the full cardinal
  set E→N→W→S stands with the E-then-N ordering probe-verified and
  the relative order of the last two unresolvable from frequencies
  alone (blockage geometry confounds). **The search also extends
  BEYOND ring 1**: ~24 % of displaced landings are ring-2 or
  diagonal offsets ((-2,-2) 30, (-2,-3) 25, (1,1) 23, (2,0) 17,
  (1,2) 16, (2,2) 11, ...), so a fully blocked ring 1 widens the
  search rather than rejecting. The sim currently models ring-1
  E→N→W→S then cant_go — the wider search is a DOCUMENTED
  simplification ([[physics-module-roadmap]]).
- Legality contract (user, same day): displacement applies to ENEMY
  mines and terrain; you can land on and walk over your OWN or
  ally-colored mines, and you cannot teleport onto enemy mines.
- Bonus re-confirmation: every probe hop's fuel delta matched
  floor(6 x euclid) to the ACTUAL landing (e.g. (12,56)->(18,63):
  55 fuel, wire 1100->1045).
