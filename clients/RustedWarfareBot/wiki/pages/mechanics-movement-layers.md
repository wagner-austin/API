---
title: "Movement Layers and Reachability"
tags: [mechanics, terrain, pathing, planner, economy]
related:
  - "[[mechanics-resource-pools]]"
  - "[[policy-threat]]"
  - "[[engine-name-oracle]]"
  - "[[mechanics-unit-catalogue]]"
  - "[[wire-contract-ndjson]]"
source_paths:
  - "runs/decompiled/com/corrodinggames/rts/gameFramework/utility/y.java:204"
  - "runs/decompiled/com/corrodinggames/rts/gameFramework/utility/y.java:388"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:175"
  - "wiki/sources/m16-enums/enum-names.txt"
  - "wiki/sources/m17-movement/reachability.txt"
  - "wiki/sources/m6-wire/world-sample.ndjson"
  - "src/rw_bot/policy/build_order.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [game-mechanics, bot-architecture]
---

# Movement Layers and Reachability

The pool selector knew how far a pool was and who could shoot the way there ([[policy-threat]]), and had no idea whether the builder could get there at all. [[mechanics-resource-pools]] recorded that as the open half of the problem, wanting "a movement-layer model the planner does not have".

The model was in the jar the whole time, and reachability turns out to be a comparison rather than a search.

## Eight layers, named by the engine

`com.corrodinggames.rts.game.units.ao` enumerates them: **`NONE`, `LAND`, `BUILDING`, `AIR`, `WATER`, `HOVER`, `OVER_CLIFF`, `OVER_CLIFF_WATER`**.[^1]

The decompile shows only `a` through `h` — the obfuscator renamed the fields. It left the constant name strings alone, which is how these are recovered ([[engine-name-oracle]]), and it is why the wire carries the layer as a name rather than an ordinal: an ordinal would have to be interpreted, and a name is already the answer.

## Reachability is a component comparison

The engine precomputes **connected components per layer** and stores them as a grid of `short` ids. A lookup takes a world point and a layer and returns the component id there.[^2] Its own reachability predicate — which names itself `pathPossible` in a log line — is then almost nothing:[^3]

```java
if (ao2 == AIR || ao2 == NONE) return true;
short s2 = y.b(x1, y1, ao2);
short s3 = y.b(x2, y2, ao2);
…
return s2 == s3;
```

Air ignores terrain entirely and answers true unconditionally. Everything else reduces to "are these two points in the same component". No search, no cost, no approximation — this is exact, and it is the same answer the engine's own AI uses when deciding whether a group can reach a target.[^4]

**Negative ids are not ids.** `-1` is impassable, `-2` is off the map, `-3` means the grids were never built for that layer. The engine's predicate rejects the first two and then compares, which leaves a hole: two `-3`s compare equal and answer *true*. The bot rejects every negative, which is strictly more conservative and costs at most a site it might have allowed.[^5]

## A pool tile is itself impassable

The first capture read `group_land: -1` for **all forty-six** pools. Taken literally that would have called every pool unreachable and stopped the economy dead.

A resource-pool tile is not walkable ground. What matters is whether a builder can stand *beside* it, so the four neighbouring tiles are sampled when the centre has no component. That is not an invention: the engine's own AI does the same thing for the same reason, testing a zone's centre and then four points around it before giving up.[^4] The tile step is read from the map rather than assumed.

With that, every pool resolves to a real component.

## What the archived map actually looks like

Thirty-four of the forty-six pools are in component **1**, the mainland the builder starts on. The other twelve sit in **six two-pool components** — six island pairs on a symmetric ten-player map — that no land unit can walk to.[^6]

So the filter is not theoretical. Distance-only selection would have aimed a builder at one of those twelve as soon as the near ground filled, and the order would have been accepted and never completed.

Two sentinel meanings confirm themselves in the same sample, from units whose situation is independently known. The Command Center reads `NONE` / `-3`: a building does not move, so no layer grid covers it. The map editor's placeholder, parked at `(-1000, -1000)`, reads `LAND` / `-2` — exactly the off-map sentinel ([[wire-contract-ndjson]]).

## What the planner does with it

Each entity carries its layer name and its component on that layer; each pool carries its component on the **land** grid. A pool is rejected when a land builder's component does not match it, counted separately from occupied and exposed so the wait reason can say which rule fired.[^7]

The land-only framing is deliberate and visible in the field name. Every builder in the base game travels on land, so this answers the question actually being asked; a hover or naval builder would need its own grid carried alongside. A builder that is not on land is **not judged at all** rather than judged wrongly — its component indexes a different grid, and comparing the two would be a confident wrong answer. Occupancy and threat still apply to it.

## What this does not do

It answers *whether* a path exists, not how long it is or what it passes. A pool in the same component can still be 4,000 units away around a headland, and the threat model still measures exposure along a straight line ([[policy-threat]]) — so the two filters remain complementary approximations rather than one true path.

Nothing here uses the layers for anything but pools. The same comparison would answer "can this tank reach that enemy", which the combat policy currently does not ask.

[^1]: `wiki/sources/m16-enums/enum-names.txt` — `com.corrodinggames.rts.game.units.ao = NONE LAND BUILDING AIR WATER HOVER OVER_CLIFF OVER_CLIFF_WATER`, read from the class's `<clinit>` with `javap`.
[^2]: `runs/decompiled/com/corrodinggames/rts/gameFramework/utility/y.java:204` — `b(float, float, ao)`, which resolves the layer's grid, converts the world point to a tile and indexes `g[x * width + y]`, returning `-3` when the grid is absent and `-2` when the tile is outside the map.
[^3]: `runs/decompiled/com/corrodinggames/rts/gameFramework/utility/y.java:388` — the predicate quoted above, including the `l.g("pathPossible: no isolatedGroups found!")` line that names it.
[^4]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:175` — the AI's zone-reachability check, which tries the zone centre and then four points at 0.4× its radius; the AI carries its own copy of the two-point predicate at `:188`.
[^5]: `src/rw_bot/policy/build_order.py` — `_can_walk_to`, and the reasoning for rejecting negatives rather than mirroring the engine's comparison.
[^6]: `wiki/sources/m17-movement/reachability.txt` — the component census of the archived capture, derived from `wiki/sources/m6-wire/world-sample.ndjson` with no game running.
[^7]: `src/rw_bot/policy/build_order.py` — `survey_pools` and the `unreachable` count it returns, rendered into the wait reason by `_no_pool_reason`.
