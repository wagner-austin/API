---
title: Fuel System
tags: [fuel, containers, map-data]
related: [[teleport-mechanics]], [[radar-mechanics]], [[map-data-decode]]
sources: [see footnotes]
fact_checked: 2026-06-14
confidence: high
---

# Fuel System

## Thresholds (current, as of 2026-06-14)

- `fuel_low_threshold`: 300 (was 500; lowered to afford combat teleports)[^1]
- `fuel_critical_threshold`: 300 (matches low)[^1]
- `hunt_min_fuel`: 100 — operating reserve for radar + teleport[^1]
- Only collect containers with volume >= 500 (smaller not worth the action cost)[^2]

## Fuel dots (MAP_DATA atlas)

The MAP_DATA blob contains a skip-RLE fuel-container atlas — the map's yellow pixels. See [[map-data-decode]] for exact decode algorithm.

- ~650 dots map-wide on field01[^3]
- Atlas is server-cached — byte-identical across all map opens in one session[^3]
- **Freshness ~40%**: about 40% of dots still hold fuel minutes later[^3]
- **Dot fuel is high volume**: every verified dot held >= 762 volume[^4]
- Off-dot fuel observed at 34 and 57 volume — dots are the big containers[^4]

## Fuel search priority

1. Radar unscanned ground (sweep reveals ~10 containers)
2. Visible containers (radar-confirmed)
3. Remembered containers
4. Dot teleport (relocation primitive, replacing blind ring hops)
5. Ring hop (blind search)
6. Dot walk (free at any fuel, tile-entry auto-collects)
7. Edge walk
8. Escape (reserve-free dot teleport when marooned)
9. Map intel[^5]

## Marooning hazard

Dots in lakes can scatter-strand the tank. Run 131003: teleport to dot (131,182) landed on one-tile island at 87 fuel. Below ~8 fuel no teleport affordable. Three-layer fix: fuel-mode radar reserve (no scans below 110), `fuel_dot_walk` (free movement), `fuel_dot_escape` (last-resort reserve-free hop).[^6]

[^1]: AIConfigDict in bot/ai/types.py — thresholds lowered from 500→300 in Phase 3d (2026-06-14)
[^2]: user (Austin), 2026-06-11 — "only collect fuel containers with volume >= 500"
[^3]: 15 runs 2026-06-11 — fuel pickup correlation: 33-71% on dots by gain bucket; sum of bytes invariant (64993)
[^4]: fuel dot probe 2026-06-11 — 6/6 nearest dots held fuel, volumes 762/807/880/1042/1189; off-dot fuel 34 and 57
[^5]: fuel search priority established 2026-06-12 after run 071658 chased 29 dots for zero gain; radar-first confirmed by run 160657 (3 kills, fuel floor 405)
[^6]: run 131003 2026-06-12 — marooned at 87 fuel on one-tile island (actually a ferry; see [[ferry-mechanics]])
