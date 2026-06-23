---
title: Viewport Frame Geometry
tags: [viewport, movement, geometry]
related: [[radar-mechanics]], [[teleport-mechanics]]
sources: [see footnotes]
fact_checked: 2026-06-22
confidence: high
---

# Viewport Frame Geometry

The viewport is a **fixed frame** in the current game configuration. The player moves freely inside it -- it does NOT scroll with the player and it does NOT recenter when the player walks to an edge. The viewport changes **only on teleport**.[^4] The viewport rectangle is also not centered on the tank: the tank can be anywhere inside, including the corner.[^5]

## Sizes

- **Actionable viewport**: 16x16 -- where move/pickup commands work[^2]
- **Observable area**: 18x18 -- radar reveals 1 extra tile on each edge (the "radar fringe")[^2]
- `ViewportStateDict` uses width=18, height=18 (full observable area)

## Viewport shifting is OFF

Viewport-shifting (auto-recenter when the tank reaches an edge) is **disabled** in the current game configuration. Walking to the viewport edge does NOT scroll the world; the tank just stops at the edge tile.[^4] To see a different 16x16 region of the map the bot must teleport. This is what makes the Sense → Hop transition load-bearing: once every tile in the current viewport has been radar-revealed, the only way to discover more containers is a teleport to a fresh viewport (see [[radar-mechanics]]).

## Tank position within the viewport

The viewport does NOT center on the tank. The tank can occupy any tile within the 16x16 frame, including a literal corner.[^5] This matters for free-radar coverage: when the tank is in a corner, the 5x5 free-radar footprint clips to 3x3 (9 tiles) instead of the full 25 (see [[radar-mechanics#both-scans-are-clamped-to-viewport-bounds]]).

## Radar fringe

Containers and entities at viewport+1 (the 18x18 outer ring) are **visible but not actionable**. Move/pickup commands only work within the inner 16x16.[^2]

## Walking paths

Walkable paths must stay inside one viewport. If BFS finds no path within viewport bounds, the container is unreachable by walking and requires a teleport reposition. Walking is not free -- it consumes **1 fuel per tile**.[^6]

## Mines under containers

Mines can exist on the same tile as equipment/fuel containers.[^1]

[^1]: user (Austin), 2026-05-27 — viewport is a fixed frame, mines under containers
[^2]: ViewportStateDict and viewport_geometry.py — 18x18 observable, 16x16 actionable, verified via radar fringe reveals
[^3]: is_collection_reachable_in_viewport BFS — bounded by current viewport, verified via terrain-blocked test cases
[^4]: user (Austin), 2026-06-21 — "we have viewport shifting off. so the viewport will never move. the only way is to teleport"
[^5]: user (Austin), 2026-06-22 — "the corner of the viewport (ir doesnt xenter on pkayer btw)"
[^6]: user (Austin), 2026-06-22 — "wlak does consume 1 fuel per tile btw"
