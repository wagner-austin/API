---
title: Core Gameplay Loop
tags: [strategy, combat, equipment, fuel]
related:
  - "[[equipment-refill-strategy]]"
  - "[[enemy-bot-behavior]]"
  - "[[radar-mechanics]]"
  - "[[teleport-mechanics]]"
source_paths:
  - "runs/bot"
fact_checked: "2026-08-04"
confidence: high
hubs: [combat]
---

# Core Gameplay Loop

The bot's behavior should mirror how a human player plays. Three phases that cycle.[^1]

Wire-verified against two recorded human sessions 2026-07-01 (account Artax, captures `sniff-20260701-185917` fuel-starved and `sniff-20260701-191133` full kit).[^2]

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

## Recorded human loop — wire-verified numbers (2026-07-01)

From the full-kit session `sniff-20260701-191133` (5 min, 3 kills: red-8, orange-4, purple-1):[^2]

- **Every teleport is map-targeted.** 7 teleports, every one preceded by a `map_open` (`CMD id=108`) seconds before. Zero blind hops. Teleport distances 9-60 tiles (median ~26).
- **Engagement shape:** teleport onto the enemy's map position → fire from the landing tile every ~2 s at the same coords → enemy dies or leaves → radar → restock from the viewport's containers. Sustained fights ran 7-8 consecutive shots without moving.
- **Shot cost is ~45 fuel** (wire: repeated `Fuel: N -> N-45` paired 1:1 with `You hit ...` lines), plus ~10/tick position cost during combat. A 3-kill 5-minute session is only fuel-positive because of the post-kill restock.
- **Post-kill restock funds the loop — kills drop NOTHING.**[^4] Immediately after each deactivation the wire shows large fuel gains (+386/+136, +158/+432, +603/+124) and equipment gains (`N dual shots gained`, `extra radar gained`) up to `Inventory full`. Fuel repeatedly topped to the 1100 cap. These are ordinary CONTAINER pickups from the current viewport — the standard restock-after-kill move — not drops from the corpse: the original "kill loot" reading of this window was an attribution error, corrected 2026-08-04.
- **Restock cadence:** pickups come in bursts of 3-5 within ~10 s after a kill or a restock landing; extra radar fires once per fresh area (`id=102` + `[INV:USED] extra radars`).
- **Fuel-starved variant** (session `185917`, started at fuel 0): teleporting is off the table below ~100 fuel, so the loop degrades to walk 3-15 tiles → extra radar → pickup, tolerating fuel 0 without concern — walking and pickups still work. Teleporting resumes only after topping up from containers. This is the fallback, not the preferred loop.[^3]

[^1]: user (Austin), 2026-06-16 — full gameplay loop description: combat → refill → radar conservation cycle
[^2]: capture `runs/sniff/sniff-20260701-191133.*` — [SENT] command stream + WORLD fuel deltas + Deactivation events, decoded 2026-07-01
[^3]: capture `runs/sniff/sniff-20260701-185917.log` + user clarification 2026-07-01: "that was a different case because the fuel started at zero so i couldnt tp. usually i tp"
[^4]: user (Austin) 2026-08-04, near-verbatim: "we dont loot enemy tanks. there is no loot for a kill. we just restock after a kill starting with the current viewport. many times we kill via homing shots" — supplied while reviewing the corpse-blocking measure ([[flag-triage-20260729]] F6), whose corpse-tile walk crossings had briefly been misread as looting.
