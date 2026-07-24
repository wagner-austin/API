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

[^1]: user (Austin), 2026-06-16 — "when I run out of radars I scan, walk a little, scan — making sure to do adjacent 5x5 blocks. when I collect extra radar I use them on a clean viewport. radars are the least frequent item from containers."
