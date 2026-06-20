---
title: "Combat Chase Loop Bug (Diagnosed 2026-06-16)"
tags: [combat, bug, teleport]
related: [[shot-range]], [[enemy-bot-behavior]], [[teleport-mechanics]], [[tank-freshness-model]]
sources: [see footnotes]
fact_checked: 2026-06-16
confidence: high
---

# Combat Chase Loop Bug

## Symptom

Bot gets stuck chasing a fleeing enemy for minutes, repeatedly teleporting 1 tile behind, never firing. Burns 6-18 fuel per hop, ~4 seconds per cycle (2s map open + 2s teleport).[^1]

## Root cause

`_combat_close` in `combat_strategy.py` requires `has_cardinal_combat_shot` (Manhattan distance == 1) before it will shoot. When the enemy moves 1 tile during the ~2s teleport, the bot lands at distance 2+ instead of 1, and re-teleports instead of shooting.[^2]

## Observed sequence (run 2026-06-16 18:26)

1. Bot at (9,128), target purple-9 at (5,128) dist=4 → teleports to (6,128)
2. Bot at (6,128), target now at (4,128) dist=2 → re-teleports to (5,128)
3. Bot at (5,128), target now at (4,128) dist=1 → finally adjacent, does scan_on_landing

Each failed attempt: target position drifts 1 tile between teleport dispatch and landing. The map open refreshes positions, confirming the enemy moved.[^2]

## Root cause

`combat_landing_candidates` was computing an adjacent tile client-side, then teleporting to THAT tile. When the map position was even 1 tile stale, the bot landed 2 tiles from reality. The correct behavior is to teleport directly to the enemy's coordinates and let the server place you adjacent — exactly how human players do it.[^5]

## Fix (implemented 2026-06-16)

- `choose_combat_landing_tile` now returns `(target["x"], target["y"])` directly — the enemy's exact position
- `find_teleport_landing_tile` (for containers) also returns `(goal_x, goal_y)` directly
- The server handles displacement: occupied tile → placed adjacent, mine → nearest open, terrain → nearest open
- Removed dead `CLOSE_WALK_RANGE_TILES` constant

This eliminates the chase loop entirely. Even if the map position is 1 tile stale, the server knows the real positions and places correctly.

[^1]: run 2026-06-16 18:26 — bot chased purple-9 through 3 teleport hops at x=4-9, y=128, spending ~54 fuel over ~12s without firing
[^2]: diagnostic logs show target position shifting between dispatch and check: target=(5,128) at dispatch → target=(4,128) at check
[^5]: user (Austin), 2026-06-16 — "I teleport to the same exact position as the enemy tank. so the game puts me adjacent"; the bot was computing adjacent tiles client-side instead of letting the server handle it
