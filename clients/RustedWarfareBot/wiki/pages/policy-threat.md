---
title: "Threat: Choosing Ground the Builder Survives"
tags: [policy, threat, planner, perception, economy]
related:
  - "[[mechanics-resource-pools]]"
  - "[[policy-loop]]"
  - "[[perception-visibility]]"
  - "[[mechanics-unit-catalogue]]"
  - "[[wire-contract-ndjson]]"
source_paths:
  - "runs/decompiled/com/corrodinggames/rts/game/n.java:1096"
  - "runs/decompiled/com/corrodinggames/rts/game/n.java:1103"
  - "wiki/sources/m6-wire/world-sample.ndjson:2"
  - "src/rw_bot/policy/threat.py"
  - "src/rw_bot/policy/build_order.py"
  - "agent/src/rwbot/agent/Perception.java"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: medium
hubs: [bot-architecture, game-mechanics]
---

# Threat: Choosing Ground the Builder Survives

The planner chose resource pools by distance from the base and nothing else. On an empty map that is the whole answer. On this one it sent a builder to a free pool 4,293 world units out, straight through two opponents' bases, and the builder was killed before arriving.[^1] The pool was legal, unoccupied, and the nearest one left.

## The question is the route, not the destination

That failure is not fixed by screening destinations. The builder died in transit; the pool it was walking to was fine. A pool can sit in empty ground with its only approach running down an enemy frontage, and a check that looked only at where the extractor would stand would have sent the same builder to the same death.

So the test is applied to the walk. A pool is rejected when any visible hostile's attack range covers any point of the straight line from the builder to it — and because the pool is that line's endpoint, a pool inside a turret's field of fire is caught by the same test. There is no separate destination check and there does not need to be.[^4]

Two measurements, two origins, and the split is deliberate. Exposure is measured from the **builder**, because the builder is what gets shot and it starts from wherever it happens to be standing. Distance is measured from the **anchor**, because the economy should grow outward from the base rather than trail whichever pool the builder last walked past. Using one origin for both would answer one of the two questions wrong.

## Hostility is the engine's answer, not the negation of ownership

"Everything that is not mine" is wrong in two directions, and the engine has the right predicate sitting there. `n.c(n)` compares alliance group — not team number — and returns false whenever either side is the neutral team.[^2] Its sibling `n.d(n)` is the same comparison inverted, which is what pins `c` as the hostile direction rather than a coincidence of naming.[^3]

So an allied player's tank is not mine and not a threat, and a neutral map object is neither. Both would have been threats under the naive test, and a bot that treats its ally's territory as a no-go zone concedes ground for free. The flag rides on every entity record.[^5]

Reach comes from the catalogue's declared attack range, so no radius in this module is a number the bot invented.[^6] A hostile with no weapon has no reach: an enemy builder standing on the route is an obstacle, not a gun, and ruling out ground it happens to occupy would give away the map to something that cannot shoot.

## Waiting is an answer, and it says which one

When every visible pool is occupied or covered, the policy waits rather than building into a gun. The reason it records distinguishes the two, because they are different games: pools all built on is progress, pools all covered is losing ground, and a run log that said only "occupied" would be lying on exactly the runs where the distinction matters.[^7] Nothing visible yet is a third case, and reads as an unexplored map rather than a contested one.

The stall detector bounds all three. None of them is `blocked`, because the world can leave each of them on its own — fog lifts as units move, a destroyed extractor frees its pool, and a killed enemy stops covering the route to one.

## What this deliberately does not model

**The route is a straight line and the engine's pathfinder does not walk straight lines.** It steers around terrain, so the real path can enter danger this misses and can avoid danger this reports. Both directions of error are live; closing the gap means measuring the engine's own paths rather than approximating them.

**Nothing here predicts movement.** A hostile is judged where it currently stands, so a tank that drives out to meet the builder was never counted. A closure model would need a time budget for the walk and the build, and build time is not in any dump the bot reads — inventing one would be exactly the kind of guess the rest of this project refuses.

**Reachability is still missing**, and it is a separate problem from threat. On a map where the nearest free pool is across water, this module happily reports it safe and the land builder cannot get there at all.[^1] That wants a movement-layer model, not a wider radius here.

What the straight-line test does catch is the case that actually killed a builder: a fixed enemy base sitting between the bot and the pool it wanted. Confidence on this page is `medium` for that reason — the rule is derived from engine truth at every point where it reads the world, and the geometry it applies on top is an approximation that has been argued for rather than measured.

[^1]: `wiki/pages/mechanics-resource-pools.md` — the open-questions section and its footnote 13, recording the 4,293-unit order at tile (12, 52), the builder still walking at sample 370 at `(498, 1198)` against opposing bases at roughly `(370, 2430)` and `(590, 1690)`, and gone from the roster by sample 480.
[^2]: `runs/decompiled/com/corrodinggames/rts/game/n.java:1096` — `public final boolean c(n n2) { if (n2 == i || this == i) { return false; } return this.r != n2.r; }`, where `i` is the neutral team and `r` the alliance group. Note it compares `r`, not the team number `k` the wire already carried.
[^3]: `runs/decompiled/com/corrodinggames/rts/game/n.java:1103` — `d(n)`, identical but for `==` and for answering true when both sides are neutral.
[^4]: `src/rw_bot/policy/threat.py` — `route_is_exposed` and the point-to-segment distance it rests on. The segment is bounded rather than an infinite line, which is what keeps a hostile 400 units behind the builder off a walk that never goes near it.
[^5]: `wiki/sources/m6-wire/world-sample.ndjson:2` — `…"team":5,"mine":false,"hostile":true,…`. Across the capture the field partitions cleanly: 9 records at `team:0, mine:true, hostile:false` and 48 across teams 1, 3, 5 and 7 at `mine:false, hostile:true`. The map is a free-for-all, so it does not exercise the allied case; that the predicate distinguishes it is read from the engine, not from this capture. Written by `Perception.isHostileToLocalPlayer`, whose binding `make check` verifies against the jar.
[^6]: `src/rw_bot/policy/threat.py` — `reach_of`, reading `UnitStats.weapon.attack_range` from the `-printunits` catalogue ([[mechanics-unit-catalogue]]). A type the catalogue does not describe is treated as harmless, on the grounds that there is no honest range to invent for it.
[^7]: `src/rw_bot/policy/build_order.py` — `survey_pools` returns the chosen pool with the counts behind the choice, and `_no_pool_reason` turns them into the wait reason.
