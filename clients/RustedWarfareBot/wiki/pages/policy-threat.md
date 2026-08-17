---
title: "Threat: Choosing Ground the Builder Survives"
tags: [policy, threat, planner, perception, economy]
related:
  - "[[mechanics-resource-pools]]"
  - "[[policy-loop]]"
  - "[[perception-visibility]]"
  - "[[mechanics-unit-catalogue]]"
  - "[[wire-contract-ndjson]]"
  - "[[mechanics-movement-layers]]"
source_paths:
  - "runs/decompiled/com/corrodinggames/rts/game/n.java:1096"
  - "runs/decompiled/com/corrodinggames/rts/game/n.java:1103"
  - "wiki/sources/m6-wire/world-sample.ndjson:2"
  - "wiki/sources/m14-threat/threat-survey.txt"
  - "wiki/sources/m18-reach/attack-range.txt"
  - "runs/decompiled/com/corrodinggames/rts/game/units/ar.java:371"
  - "src/rw_bot/policy/threat.py"
  - "src/rw_bot/policy/build_order.py"
  - "agent/src/rwbot/agent/Perception.java"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
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

Reach comes from each type's declared attack range, so no radius in this module is a number the bot invented.[^6] A hostile the engine reports as unarmed has no reach: an enemy builder standing on the route is an obstacle, not a gun, and ruling out ground it happens to occupy would give away the map to something that cannot shoot.

**Which source that range comes from turned out to matter more than the geometry.** It was read from `-printunits`, which emits 90 of the engine's 173 registered types — it skips the bug faction by name prefix, shadowed built-ins, custom types without a listing flag, and sixteen names it blocklists outright.[^9] An absent type read as range zero, so 48 armed types were harmless to this module, and *harmless* is the one direction a threat model must never guess in.

It is now read from the registry instead, one record per type, taken off the engine's prototype for that type.[^10] Where the two sources overlap they agree on all 90 with no disagreements, which is what makes this a wider reading of the same fact rather than a competing one. A missing name is no longer absorbed: the table is indexed, so a stale dump fails loudly and names the type instead of reporting a turret as unarmed.

**The gap was reachable by ordinary play, not by mods.** Of the buildable types, 27 are armed and absent from the old source, and most are the *same unit in another mode* — the mode being the type name a live entity reports while in it. A deployed mech bunker (240), a surfaced submarine (240), a landed experimental gunship (390), an amphibious jet underwater (100). Every one of those read as unarmed. Add the bug faction on a map with bug players and the modular spider tree, which reaches 400.[^11]

Measured honestly, it changed nothing on the sample to hand: of 55 visible hostiles, 20 read as armed under either source and the pool survey was identical. The five AI players had built only priced types by that point. This removes a latent hole rather than a firing one.[^11]

## Waiting is an answer, and it says which one

When every visible pool is occupied or covered, the policy waits rather than building into a gun. The reason it records distinguishes the two, because they are different games: pools all built on is progress, pools all covered is losing ground, and a run log that said only "occupied" would be lying on exactly the runs where the distinction matters.[^7] Nothing visible yet is a third case, and reads as an unexplored map rather than a contested one.

The stall detector bounds all three. None of them is `blocked`, because the world can leave each of them on its own — fog lifts as units move, a destroyed extractor frees its pool, and a killed enemy stops covering the route to one.

## What it does on the real map

The bot's own runs never exercise the rule, because its base is far from every opponent and the pools it reaches for are clear — so the filter had to be measured rather than inferred from a green run. Connecting to a live game after a completed plan and asking the survey what it would say, without sending anything:[^8]

Tile (12, 52) — the pool from the recorded death, 4,272 units out and the farthest on the map — reads exposed, and independently of being occupied: the straight route to it passes through fire whether or not something is standing on it. Eleven free pools are rejected for threat in that sample and fourteen survive.

The near picks are untouched. Old rule and new rule choose the same tile at 694 units, because the ground around the base is clear, so the filter costs nothing in the opening and only starts removing options once the near pools are taken — which is exactly when the old rule began reaching across the map.

The counts drift between samples because the opponents are playing; an earlier probe on the same map read 2 rejected and 27 safe. What reproduces is the shape rather than the numbers: near picks unaffected, far pools falling out, and the pool from the death always among them.

**It does not cap distance.** Fourteen pools were free and safe in that sample and the farthest was 2,518 units out. Distance still only ranks the survivors, so a long walk remains possible whenever every nearer pool is gone. Whether that wants a cap is an open question — a cap is a number with no derivation behind it, which is why there is not one here.

## What this deliberately does not model

**The route is a straight line and the engine's pathfinder does not walk straight lines.** It steers around terrain, so the real path can enter danger this misses and can avoid danger this reports. Both directions of error are live; closing the gap means measuring the engine's own paths rather than approximating them.

**Nothing here predicts movement.** A hostile is judged where it currently stands, so a tank that drives out to meet the builder was never counted. A closure model would need a time budget for the walk and the build, and build time is not in any dump the bot reads — inventing one would be exactly the kind of guess the rest of this project refuses.

**Reachability is still missing**, and it is a separate problem from threat. On a map where the nearest free pool is across water, this module happily reports it safe and the land builder cannot get there at all.[^1] That wants a movement-layer model, not a wider radius here.

What the straight-line test does catch is the case that actually killed a builder: a fixed enemy base sitting between the bot and the pool it wanted, which the live survey confirms it now rejects.[^8]

Confidence here is `high` on the claims and deliberately narrow on the scope. Every point where the rule reads the world is engine truth — hostility from the engine's own predicate, reach from the engine's own catalogue — and what it does with that has been measured on a live map rather than argued from a green run. What is *not* claimed is that a pool this rule accepts is safe. It is a filter against a specific, observed way of dying, not a model of danger.

[^1]: `wiki/pages/mechanics-resource-pools.md` — the open-questions section and its footnote 13, recording the 4,293-unit order at tile (12, 52), the builder still walking at sample 370 at `(498, 1198)` against opposing bases at roughly `(370, 2430)` and `(590, 1690)`, and gone from the roster by sample 480.
[^2]: `runs/decompiled/com/corrodinggames/rts/game/n.java:1096` — `public final boolean c(n n2) { if (n2 == i || this == i) { return false; } return this.r != n2.r; }`, where `i` is the neutral team and `r` the alliance group. Note it compares `r`, not the team number `k` the wire already carried.
[^3]: `runs/decompiled/com/corrodinggames/rts/game/n.java:1103` — `d(n)`, identical but for `==` and for answering true when both sides are neutral.
[^4]: `src/rw_bot/policy/threat.py` — `route_is_exposed` and the point-to-segment distance it rests on. The segment is bounded rather than an infinite line, which is what keeps a hostile 400 units behind the builder off a walk that never goes near it.
[^5]: `wiki/sources/m6-wire/world-sample.ndjson:2` — `…"team":5,"mine":false,"hostile":true,…`. Across the capture the field partitions cleanly: 9 records at `team:0, mine:true, hostile:false` and 48 across teams 1, 3, 5 and 7 at `mine:false, hostile:true`. The map is a free-for-all, so it does not exercise the allied case; that the predicate distinguishes it is read from the engine, not from this capture. Written by `Perception.isHostileToLocalPlayer`, whose binding `make check` verifies against the jar.
[^6]: `src/rw_bot/policy/threat.py` — `reach_of`, which indexes the registry reach table rather than defaulting through a miss.
[^7]: `src/rw_bot/policy/build_order.py` — `survey_pools` returns the chosen pool with the counts behind the choice, and `_no_pool_reason` turns them into the wait reason.
[^8]: `wiki/sources/m14-threat/threat-survey.txt` — one sample taken over the agent channel after a 700-sample run on `[z;p10]Crossing Large (10p)`, with the survey run against it and no order sent. The file carries the hostility partition, the per-pool table and the old-rule/new-rule comparison, plus the drift note against an earlier probe.
[^9]: `runs/decompiled/com/corrodinggames/rts/game/units/ar.java:371` — `s()`, the `-printunits` implementation, and the `continue` filter plus sixteen-name blocklist at its head. The fifty `--- ERROR: running printForHelp()` lines in the log are a banner loop in the same method, not failures.
[^10]: `agent/src/rwbot/agent/TypeFlags.java` — `attackRange`, reading the engine's prototype for a type through a static map lookup, and `combatRecord`, which gives the answer its own record kind rather than widening the placement record.
[^11]: `wiki/sources/m18-reach/attack-range.txt` — the cross-check, the printer's filter, the 27 buildable armed types the old source omitted, and the live before/after that showed no change on this map.
