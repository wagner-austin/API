---
title: Viewport Frame Geometry
tags: [viewport, movement, geometry]
related: [[radar-mechanics]], [[teleport-mechanics]]
sources: [see footnotes]
fact_checked: 2026-06-11
confidence: high
---

# Viewport Frame Geometry

The viewport is a **fixed frame**. The player moves freely inside it — it does NOT scroll with the player. It only recenters when the player walks to the edge.[^1]

## Sizes

- **Actionable viewport**: 16x16 — where move/pickup commands work[^2]
- **Observable area**: 18x18 — radar reveals 1 extra tile on each edge (the "radar fringe")[^2]
- `ViewportStateDict` uses width=18, height=18 (full observable area)

## Recentering

When the player walks to a viewport edge, the viewport recenters with a **1-tick delay** — movement is abated for 1 tick, then the player can act again. Computing edge targets must use `world["viewport"]` bounds (left, top, width, height), NOT player position +/- radius.[^1]

## Radar fringe

Containers and entities at viewport+1 (the 18x18 outer ring) are **visible but not actionable**. Move/pickup commands only work within the inner 16x16. Walk to the viewport edge first to trigger recenter, then the fringe positions become actionable.[^2]

## Walking paths

Walkable paths must stay inside one viewport. If BFS finds no path within viewport bounds, the container is unreachable by walking and requires a teleport reposition. The tank stops at the edge — no auto-recenter mid-walk.[^3]

## Mines under containers

Mines can exist on the same tile as equipment/fuel containers.[^1]

[^1]: user (Austin), 2026-05-27 — viewport is a fixed frame, recenters on edge walk, mines under containers
[^2]: ViewportStateDict and viewport_geometry.py — 18x18 observable, 16x16 actionable, verified via radar fringe reveals
[^3]: is_collection_reachable_in_viewport BFS — bounded by current viewport, verified via terrain-blocked test cases
