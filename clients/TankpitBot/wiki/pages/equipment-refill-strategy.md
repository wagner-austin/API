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

## The atlas implementation — and the reverted hoard (2026-08-28)

The equipment ATLAS shipped and stays ([[equipment-system]] hotspot
law): `data/equipment_atlas.json` (rebuilt by `make equipment-atlas`)
drives a COLLECT teleport circuit (`bot/ai/equipment_atlas.py`) that
hops to corpus-proven equipment ground when nothing believed
collectible remains — an ADDITION to the cascade, above the quad
sweep.

The same-day radar-HOARD band (slot toggled off between one extra and
the hunt bar) was **REVERTED by operator order** ("i never said to do
the conservative radar shit"). Radar policy is unchanged from before:
the slot stays enabled while stocked, and spending is governed by the
reveal-floor economics. Two facts from the trials survive the revert
as recorded knowledge: a press with the radar slot DISABLED is a
total no-op (no extras, no fuel, NO scan — the free built-in 5x5
requires the slot ON at zero stock), and equipment income during
20-minute low-radar foraging measured ~0.45 radars/min vs ~2.6/min
during full-tempo farming.
