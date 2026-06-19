---
title: Core Gameplay Loop
tags: [strategy, combat, equipment, fuel]
related: [[equipment-refill-strategy]], [[enemy-bot-behavior]], [[radar-mechanics]], [[teleport-mechanics]]
sources: [see footnotes]
fact_checked: 2026-06-16
confidence: high
---

# Core Gameplay Loop

The bot's behavior should mirror how a human player plays. Three phases that cycle.[^1]

## Phase 1: Combat

1. Teleport to enemy (directly on their position, server places you adjacent)
2. Use radar on landing (scan viewport for nearby fuel/equipment)
3. Walk to pick up fuel if needed (nearby, on current viewport)
4. Walk adjacent to enemy
5. Engage — fire dual shots
6. When enemy flees/teleports away: **fire homing shots** (track off-viewport)
7. Get the kill
8. After kill → Phase 2

## Phase 2: Post-kill refill

1. Pick up any fuel/equipment containers on the current viewport first
2. Teleport to a **close, fresh viewport** (not far — just enough for new ground)
3. Use radar — full viewport sweep
4. Pick up ALL containers (equipment AND fuel)
5. Repeat steps 2-4 until full on everything
6. Radar stock target: **~20-25 extra radars** before returning to combat
7. Duals and homings fill up fast (most common in containers). Radars are the **least frequent** — they're the bottleneck
8. When stocked → Phase 1

## Phase 3: Emergency radar refill (0 extra radars)

Only when extras hit exactly 0:[^1]

1. Scan with built-in radar (free, 5x5 coverage)
2. Walk ~5 tiles in one direction (adjacent 5x5 block, no overlap)
3. Scan again
4. Repeat — covering fresh ground each free scan
5. When you collect an extra radar, use it on a **clean viewport** (teleport away from already-scanned area)
6. Resume Phase 2 pattern (teleport → extra radar sweep → collect → repeat)

## Key principles

- **Extra radars are precious** — use each one on a FRESH viewport, never on already-scanned ground[^1]
- **Pick up everything** — fuel AND equipment on every viewport sweep, not just what you need[^1]
- **Short teleport hops** for refill — close enough to cover new ground efficiently, not huge distances[^1]
- **Never abandon a kill** — use homing shots to finish fleeing enemies[^1]
- **Radars are the bottleneck** — containers give random items, radars are least frequent[^1]

[^1]: user (Austin), 2026-06-16 — full gameplay loop description: combat → refill → radar conservation cycle
