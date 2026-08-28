---
title: Equipment Refill Strategy
tags: [equipment, radar, strategy]
related:
  - "[[radar-mechanics]]"
  - "[[equipment-system]]"
source_paths:
  - "runs/bot"
fact_checked: "2026-06-16"
confidence: high
hubs: [combat]
---

# Equipment Refill Strategy

## Container randomness

Equipment containers give random items. Radars are the **least frequent** — duals and homings fill up first. This means you need to collect many equipment containers to build radar stock. Keep radars at 20+ before returning to hunting.[^1]

## Low-radar search pattern (built-in 5x5)

When extra radars are low, do NOT burn them on equipment search. Use the built-in 5x5 with a **grid walk pattern**:[^1]

1. Scan with built-in radar (free, 5x5)
2. Walk ~5 tiles in one direction (covers adjacent 5x5 block, no overlap)
3. Scan again
4. Repeat, covering fresh ground each scan

The key: **move exactly ~5 tiles between scans** so the 5x5 blocks are adjacent, not overlapping ([[radar-mechanics]] — footprint is rank-derived, 5x5 at ranks 0-2). Walking costs 1 fuel/tile and the built-in scan is free ([[game-economy]]).

## Extra radar conservation

When you collect an extra radar during the grid walk, use it on a **fresh viewport** — not one already covered by built-in scans. Teleport to uncovered ground, then use the extra for a full viewport sweep.[^1]

## Refill exit condition

Stay in equipment recovery until radars reach ~20. Don't exit early just because duals/homings are full — radars are the bottleneck and running out is a death spiral. See [[radar-mechanics]].[^1]

[^1]: user (Austin), 2026-06-16 — "when I run out of radars I scan, walk a little, scan — making sure to do adjacent 5x5 blocks. when I collect extra radar I use them on a clean viewport. radars are the least frequent item from containers." The scan-walk-scan pattern is the free-radar footprint at `free_radar_revealed_tiles` (`src/tankpit_bot/state/scan_coverage.py:120-145`), which reveals only a small envelope around the tank clipped to viewport bounds; the "use extras on a clean viewport" half is the reveal-floor economics at `src/tankpit_bot/bot/ai/context.py:382` (`RADAR_SPEND_REVEAL_FLOOR_TILES = 32`) escalating to `:408` (`RADAR_RESERVE_REVEAL_FLOOR_TILES = 128`) on the last extra.

## The atlas + hoard implementation (2026-08-28)

The 2026-08-28 income-burn deadlock (16 radars gained, 16 spent
mid-restock, hunt bar never reached, zero kills in 17 minutes) turned
this page's two clauses against each other and forced the precedence
ruling, shipped as:

- **Radar hoard rule** (`tactics.compute_desired_equipment`): outside
  HUNT the extra-radar slot is toggled OFF below the hunt bar
  (`combat_radar_min`), so every restock press serves the free
  built-in 5x5 (this page's grid-walk half) and stock provably climbs.
  The slot was previously enable-only in `apply_equipment` — once lit
  it stayed server-side enabled forever, which is WHY collected extras
  kept burning. In HUNT the slot stays on: combat scanning is what
  the bar was saved for.
- **Equipment atlas** ([[equipment-system]] hotspot law): the corpus's
  per-field persistence-weighted hotspot tiles
  (`data/equipment_atlas.json`, `make equipment-atlas`,
  `bot/ai/equipment_atlas.py`) drive a COLLECT teleport circuit — hop
  to the best unvisited hotspot, let the viewport reveal what sits
  there (equipment needs NO radar to see), collect, hop on. Empty
  hotspots tombstone for 3 minutes.
- **Quad sweep demoted, not deleted**: it declines outright below the
  hunt bar and survives only as rich-stock recon for ground the atlas
  does not know. The last-extras reserve branch died with the gate.
