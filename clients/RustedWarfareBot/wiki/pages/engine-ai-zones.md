---
title: "The Shipped AI's Zone System"
tags: [ai, strategy, engine, zones, planner, economy]
related:
  - "[[mechanics-resource-pools]]"
  - "[[policy-threat]]"
  - "[[policy-loop]]"
  - "[[engine-entity-model]]"
  - "[[building-structures]]"
  - "[[engine-ai-probe]]"
  - "[[engine-name-oracle]]"
source_paths:
  - "runs/decompiled/com/corrodinggames/rts/game/a/o.java"
  - "runs/decompiled/com/corrodinggames/rts/game/a/h.java"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:301"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:634"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:803"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:1492"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:1540"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:529"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:1189"
  - "runs/decompiled/com/corrodinggames/rts/game/a/l.java:71"
  - "runs/decompiled/com/corrodinggames/rts/game/a/n.java:80"
  - "runs/decompiled/com/corrodinggames/rts/game/b/e.java:92"
  - "wiki/sources/m15-ai-zones/zone-dump.txt"
  - "wiki/sources/m15-ai-zones/zone-dump-330s.txt"
  - "wiki/sources/m16-enums/enum-names.txt"
source_git_blobs:
  "wiki/sources/m15-ai-zones/zone-dump.txt": "dbfff06d71e2e199976a8bd8727e163aa7d451f9"
  "wiki/sources/m15-ai-zones/zone-dump-330s.txt": "cbf8ad157b2d0d752aee3997745800351f25b62c"
  "wiki/sources/m16-enums/enum-names.txt": "919c505b3f4d41a4c8dc599a37653ddbfb4e5e1e"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: medium
hubs: [engine-internals, game-mechanics]
---

# The Shipped AI's Zone System

The bot has no spatial model. It measures distance from an anchor and puts structures on a fixed ring ([[policy-loop]]). The AI it loses to has one, and it is readable: `com.corrodinggames.rts.game.a` is the AI package, and a **zone** is its unit of place — a circle that owns units, holds state, and decides what gets built inside it.

This page covers the zone system only. What the AI does with zones — build order, attack triggers, unit composition — is a separate reading.

## Names, and where they come from

The package is obfuscated, but the AI draws a debug overlay over its own zones and the strings in it survived ProGuard.[^1] That overlay is the naming oracle for this page: nearly every field named below is named by the engine printing it, not by inference. Two class names leak the same way — `m` identifies itself as **PlainZone** in a fallback log line,[^2] and `n` as **TransporterGroup** in a load-time complaint.[^3]

Where a name is inferred rather than printed, this page says so.

## Five kinds of zone

The save format enumerates them, so the set is closed and the ids are stable:[^4]

| id | class | what it is |
|----|-------|-----------|
| 1 | `i` | base zone — the biggest class in the package, 1,222 lines |
| 2 | `g` | combat group, flagged Attack Type or Defensive Type |
| 3 | `n` | TransporterGroup |
| 4 | `m` | PlainZone — position and radius, no behaviour |
| 5 | `l` | a short-lived rally group |

A zone that is not one of those five throws on save rather than being skipped.[^4]

## The base contract

Every zone is a circle with an identity: `S`, `T`, `U` are world x, y and radius, and `Q` is an id handed out by a per-AI counter.[^5] Construction registers the zone in the AI's list and `p()` removes it, so a zone's life is explicit rather than garbage-collected.

Three containment tests, and the difference between them matters:[^5]

- `c(x, y)` — is a *point* inside. Plain squared-distance against the radius.
- `b(unit)` — is a *unit* inside. Radius inflated by that unit's own collision radius, so a large unit counts as inside sooner than its centre does.
- `a(unit, margin)` — the same with a caller-supplied slack.

Distances are squared and left squared throughout, the same convention the bot's own code uses for the same reason.

`w()` returns a random point in the zone, though not a uniform one — sampling radius linearly biases toward the centre. Whether that is deliberate is not readable.

### Placing a building inside a zone

`e(type)` is the AI's answer to the question the bot's `PLACEMENT_RING` guesses at, and it is a **rejection sampler**: up to 15 attempts, each picking a random point in the zone and accepting it only if the map says the tile exists, a terrain-passability count clears a threshold, and the engine's own placement predicate accepts that type there.[^6] Failure returns null rather than a fallback position.

Two special cases are visible in it. One unit type gets a fixed 600-unit search radius that **grows by 100 on every failed attempt**, so the search widens rather than giving up in place. Another type is placed *adjacent to an existing unit of the same type* — six of the fifteen attempts pick a random existing one, offset it 150 units on one axis, and ask the engine's line-of-placement helper for a spot. That is a wall-building or line-building behaviour; which types they are is an enum constant this page has not resolved.

## Zones that own units

`h` is the abstract zone that holds units, and `g`, `n` and `l` extend it.[^7] Its invariant is worth stealing: a unit has a back-pointer `aB` to its zone, and joining a zone detaches it from the previous one first — **a unit belongs to exactly one zone at a time**. There is no shared ownership and no arbitration; assignment is the arbitration.

`h` also prunes: dead members are dropped each tick, and a separate busy-list is pruned of anything with no current task.

## The base zone

A base zone is created with **radius 420** for the home base and **radius 360** for an expansion.[^8][^9] The overlay names its state:[^10] a three-value state enum, `unsafe` with an `unsafeBaseTimer`, `allowedUnits`, a `buildBuildingDelay`, `claimedBaseTimer`, `abandonedTimer`, `requestedBuildersDelay`, and counts of `Builders` and `Idle Builders`.

The build-failure fields are the interesting half: `lastAttemptedBuilding`, `lastAttemptedBuildingCount`, `lastAttemptedBuildingFailed`, and two separate affordability fields — `cannotAffordPrice` and `cannotAffordBy`. The AI keeps a per-zone record of what it tried to build and why it could not, which is the same diagnostic problem the produce path forced on the bot ([[mechanics-build-actions]]) solved a different way.

### How the AI picks a base

When it has no base zone at all, it makes one at the first hit of a four-step fallback: an owned Command Center, then an owned unit of a builder type, then an owned orderable matching that set, then any owned unit answering a capability predicate.[^8] All four use radius 420.

### How the AI picks expansions

Immediately after, it tries once to seed candidate expansion zones **from the map's resource-pool list**.[^9] That list is the same one the bot reads: it is populated at map load from tiles carrying the `res_pool` flag, and the engine logs `resPools point:` when it dedupes one.[^11] So the AI expands to pools, from the same 46 tiles our scan finds on this map ([[mechanics-resource-pools]]).

A candidate pool is rejected when any of these holds:[^12]

- a **hostile** player's Command Center is within **300** units
- an **allied** player's Command Center is within **320** units
- **four or more** hostile units are within 360
- **two or more** allied buildings are within 360
- a zone already covers the point, or the AI owns no extractor within 200 of it

Two things stand out. The ally exclusion is *larger* than the enemy one, so this is a don't-crowd-your-ally rule sitting in the same filter as the threat rule. And the hostile/allied split is the same pair of engine predicates the bot's own threat model now uses — `n.c` and `n.d` ([[policy-threat]]) — which is some evidence the bot picked the right pair.

**That block is unreachable in normal play, and it does not matter.** It is guarded by a once-only flag set *before* the work is attempted, and the work is additionally gated on the AI already owning an extractor. A fresh game owns none, so the gate fails, the flag is set anyway, and the branch never runs again — the flag is written in one place and read in one, with no reset in 1,910 lines, and the whole thing sits inside a `no base zones at all` test that only holds on the first tick.[^9]

The reason it does not matter is that it is a vestigial bootstrap, not the expansion path. Two other sites create base zones on recurring cooldowns, and this page originally missed them:[^15]

- **Expansion**, radius **360**: fires when a timer reaches zero, refuses if more than two zones are already in the claiming state, otherwise asks `an()` for a site and — if no zone covers it and the viability filter passes — claims it and sets the timer to 2,000. A blocked attempt retries at 300 instead.
- A third kind, radius **310**, capped at three, on a 5,000 cooldown, sited from a *unit* rather than a point and screened by a different filter.

And `an()` draws **a resource pool uniformly at random** from the same map list.[^16] So the AI does expand onto pools; it just gets there by the live path rather than the dead one. Worth noting for our own purposes: the class also carries a nearest-pool-to-a-point helper right beside it, and the expansion path does not use it. The AI expands at random and relies on the viability filter to reject bad draws.

The unreachability claim is narrow and static-verifiable — one flag, one write, one read. The rest has now been watched directly.

## Watched live

A probe dumps every zone of every AI player to the agent log ([[engine-ai-probe]]). It is an instrument and never touches the wire, for reasons that page sets out. Sampling four AI players at 40, 90 and 150 seconds:[^17]

The **radii are exactly as read**: one zone at 420 per player, and 360 for every later one. The expansion path is unambiguously live — at 40s each player holds only its home zone; by 90s one 360 zone has appeared; by 150s there are three, with ids 13, 17 and 24 climbing the same counter. That settles it: the dead bootstrap costs nothing, because expansions arrive on the cooldown path instead.

The **capacity ratio is real and bounded in [0, 1]**, which was the shakiest inference on this page. The home zone reads 0.55, 0.8 and 0.39 across the three samples; every fresh expansion reads 0.0.

The **build delay confirms the capacity penalties** by arithmetic. Fresh expansions sit at capacity 0.0, which should attract both +180 penalties on top of a ~270 reset, and the observed values top out at 643, 639, 568 and 557 — a ~630 ceiling, where the unpenalised reset alone would cap near 285.

One thing this run did **not** show: the third zone kind at radius 310 never appeared in 150 seconds, so its cap and cooldown remain unobserved.

### The states and kinds have names

The enums looked obfuscated — the decompile shows `enum j { a, b, c; }` — but only their *fields* were renamed. The constant name strings survived in the bytecode and `javap` reads them straight out ([[engine-name-oracle]]):[^18]

- **kind** is `Main`, `ResourceOutpost`, `ForwardOutpost`
- **state** is `Pre`, `Prepare`, `Active`

So the engine's own vocabulary confirms the reading and improves on it. The radius-420 home zone is `Main`; every radius-360 expansion is a `ResourceOutpost`, which settles what the pool-sited expansions are *for*; and the unobserved radius-310 third kind is a **`ForwardOutpost`** — an aggressive forward position, which explains why it is sited from a unit rather than a map point and screened by a different filter than the economic one.

The lifecycle is visible too. A freshly claimed outpost reads `Pre`, and by the next sample an established one reads `Active`.[^19] `Prepare` sits between them and has not been caught.

## The other three

**Combat group `g`** carries an Attack/Defensive flag, a target size (`Units: n / A`), `StagingForAttack`, `AttackDelay`, `StagingTimer`, `StagingTargetFound`, `attackingFor`, a `commonMovement` enum, a `seaGroup` flag, a `unitsNeedingTransport` list and a VIP mode.[^13] So attacks are *staged*: a group fills to a size, waits, then commits. That is a mechanism the bot has no equivalent of — it has just acquired attack orders and nothing that decides when a force is ready.

**TransporterGroup `n`** carries `UnitsWanted`, `readyToMoveOut`, and `CurrentlyHelping` pointing at another zone's id.[^13] The AI models "ferry these units to that zone" as a zone of its own.

**Rally group `l`** is transient: members are dropped once they are within **60 world units** of the centre and hold no build task, and the zone deletes itself when it empties or after a 5,000-unit timer.[^14] It is the smallest useful pattern here — a zone as a temporary waypoint with an arrival test and a self-destruct.

## What this is worth to the bot

Three things transfer without any strategy commitment:

1. **A zone as the placement unit.** `e(type)` rejection-samples inside a circle against the engine's own placement predicate; the bot's ring is a fixed list of eight offsets that the engine can and does refuse. Replacing the ring with a sampler that asks the engine is a strictly better answer and needs no new engine reading.
2. **One unit, one zone.** The exclusive `aB` back-pointer is how the AI avoids two jobs claiming the same builder. The bot currently selects a producer per decision with nothing stopping two decisions picking the same unit.
3. **Staging before attacking.** The AI does not attack with what it has; it fills a group to a target size first.

What does *not* transfer cleanly is the expansion filter's radii. They are base-proximity numbers — 300 and 320 from a Command Center — where the bot's threat model is weapon-range based ([[policy-threat]]). They answer different questions and the AI's would not have saved the builder that died in transit, because it screens the destination and not the walk.

## Open questions

The state enums are obfuscated to `a`, `b`, `c`, so the overlay prints `State: a` and the meaning has to come from use sites. Only one is pinned so far: the state carrying `claimedBaseTimer`.

Two unit-type constants in the placement sampler are unresolved — the one granted a growing 600-unit search and the one placed adjacent to its own kind.

And the cold-start reading above needs a live observation before it is a finding rather than a reading.

[^1]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:803` onward — the per-zone overlay block, which walks the AI's zone list and appends a labelled line per field. The labels are unobfuscated string literals.
[^2]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:318` — `"Found zone type 0, loading PlainZone instead"`, in the branch that returns a `m`.
[^3]: `runs/decompiled/com/corrodinggames/rts/game/a/n.java:80` — `"TransporterGroup:readIn: Unit is not transporterUnit"`, in `n`'s own load path.
[^4]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:255`–`266` for the write side, which maps each class to an id and throws `"zone not instance not supported:"` on anything else, and `:301` for `l(int)`, the read-side factory.
[^5]: `runs/decompiled/com/corrodinggames/rts/game/a/o.java` — fields at the head of the class; `c(float,float)` at `:64`, `b(am)` at `:70` inflating by the unit's `cj`, `w()` at `:94`.
[^6]: `runs/decompiled/com/corrodinggames/rts/game/a/o.java:102` — `e(as)`, a 15-iteration loop ending in `l2.bL.c(tx,ty) && (passability > 5 || passability == 0) && d.a(type, x, y, ai)`, returning null when no attempt succeeds.
[^7]: `runs/decompiled/com/corrodinggames/rts/game/a/h.java` — `F` members and `G` busy list; `a(y)` at `:76` performs the detach-then-attach through `y.aB`.
[^8]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:1492`–`1538` — four `new i(...)` branches, each setting `U = 420.0f` and the same state and kind constants.
[^9]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:1540`–`1556` — the once-only block. The flag is declared at `:96`, set at `:1541`, read at `:1540`, and appears nowhere else in the 1,910-line class. Zones created here get `U = 360.0f` and a different kind constant from the home zone.
[^10]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:806`–`836` — the `instanceof i` arm of the overlay, one labelled line per field.
[^11]: `runs/decompiled/com/corrodinggames/rts/game/b/e.java:92` — `this.i.A.add(new Point(n2, n3))`, guarded by the tile's `res_pool` flag and preceded by a dedupe that logs `"resPools point:… already exists"`. `A` is declared on the map class at `game/b/b.java:100` and cleared on map load.
[^12]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:634` — `b(PointF)`, with the two counters it calls at `:1789` (allied, optionally buildings-only) and `:1802` (hostile). Both use `n.d` and `n.c` respectively, the alliance-group predicates.
[^13]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:838`–`876` — the `instanceof g`, `instanceof n` and `instanceof l` arms of the overlay.
[^14]: `runs/decompiled/com/corrodinggames/rts/game/a/l.java:71` — `this.c((am)y2) < 3600.0f`, a squared distance, so 60 world units; the self-delete at `:77`.
[^15]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:1189`–`1216` for the radius-360 expansion path, gated on a timer decayed each tick and capped by a count of zones in the claiming state, and `:1217`–`1241` for the radius-310 path, capped at three and screened by a different point filter than [^12].
[^16]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:529` — `an()` returns `A.get(f.c(A.size()))` converted to world coordinates: a uniform random draw from the resource-pool list. The nearest-pool helper `a(float,float)` sits at `:541` and is not called by either expansion path.
[^17]: `wiki/sources/m15-ai-zones/zone-dump.txt` — four AI players sampled at 40, 90 and 150 seconds of a headless sandbox run, distilled from a 4,600-line dump of every declared field of every zone.
[^18]: `wiki/sources/m16-enums/enum-names.txt` — `com.corrodinggames.rts.game.a.j = Pre Prepare Active` and `com.corrodinggames.rts.game.a.k = Main ResourceOutpost ForwardOutpost`, read from each class's `<clinit>` with `javap`.
[^19]: `wiki/sources/m15-ai-zones/zone-dump-330s.txt` — at 180s one player holds a 360 zone at `j.Pre` and another at `j.Active`, both `k.ResourceOutpost`, against its 420 `k.Main`.
