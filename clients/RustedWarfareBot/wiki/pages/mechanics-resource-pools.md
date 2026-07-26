---
title: "Resource Pools and the Placement Rule"
tags: [mechanics, economy, terrain, placement, extractor, tiles]
related:
  - "[[policy-loop]]"
  - "[[building-structures]]"
  - "[[mechanics-unit-catalogue]]"
  - "[[perception-visibility]]"
  - "[[wire-contract-ndjson]]"
  - "[[policy-threat]]"
  - "[[mechanics-movement-layers]]"
source_paths:
  - "wiki/sources/m11-pools/type-flags.ndjson:1"
  - "wiki/sources/m11-pools/type-flags.ndjson:85"
  - "wiki/sources/m11-pools/pool-build-run.log:401"
  - "wiki/sources/m11-pools/pool-build-run.log:403"
  - "wiki/sources/m6-wire/world-sample.ndjson:20"
  - "wiki/sources/m11-pools/builder-travel-timing.txt:13"
  - "src/rw_bot/policy/build_order.py"
  - "agent/src/rwbot/agent/MapTiles.java"
  - ".game/assets/units/extractor/extractor_common.ini:25"
  - ".game/assets/tilesets/misc.tsx:6"
  - ".game/assets/translations/Strings.properties:557"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [game-mechanics, engine-internals]
---

# Resource Pools and the Placement Rule

Credits come from extractors, and an extractor may be built on a resource pool and nowhere else.[^3][^7] That single rule is what makes the economy a terrain problem rather than a shopping problem, and it defeats every source the bot had: pools are not units, so they appear in no entity list, and the placement rule is in no stat dump.

## A pool is a tile, not an object

A resource pool is a map tile whose tileset declares the property `res_pool`.[^1] The map loader turns that property into a boolean on the tile object,[^2] and the engine's own placement check reads that same boolean.[^3] Nothing in the chain is inferred: the property name, the field it sets, and the code that consults it were each read from the decompiled loader.

The sandbox map carries 46 of them.[^4] Each is a single tile at the centre of a 3×3 decorative sprite — the tileset's `misc` block runs tile ids 0–8 for the artwork and puts `res_pool` on id 4 alone, which is why the count of marked tiles is exactly one per visible pool rather than nine.[^1]

That count is worth stating twice because it was arrived at twice, independently. The agent walks the live tile grid reflectively and reports 46;[^5] decompressing the map file's `Items` layer and counting the marked gid also gives 46, at the same coordinates — tile (115, 6) in both.[^6] Two unrelated routes to the same answer is what makes the tile binding trustworthy rather than merely plausible.

## Where the rule is written, and where it is not

The unit definition declares it: `placeOnlyOnResPool: true` in the extractor's shared `.ini`.[^7] The loader stores that into a field on the unit type, the type's predicate reports it, and the placement check consults the predicate before anything else — rejecting with the message key `gui.cannotPlace.needsResourcePool`.[^3]

It is not in `-printunits`. The stat catalogue carries price, HP, speed, mass and weapons, and no placement rules at all ([[mechanics-unit-catalogue]]). The only *prose* mention anywhere in the shipped files is an English sentence in a translations bundle,[^8] which is a blurb rather than a fact: it is hand-written per language and has no connection to the flag the engine enforces. A bot that parsed it would be reading marketing copy.

So the rule is asked of the live engine instead. `make type-flags` enumerates every registered unit type — the built-in enum constants and everything loaded from `assets/units/` — and asks each for its own placement predicate. The dump covers 173 types and reports exactly eight as pool-bound: the five shipped extractor tiers, two bug-faction extractors, and the built-in `extractor` the asset units shadow.[^9]

One asymmetry surfaced while writing that dump and is worth recording, because it is a trap for anything that round-trips a type name. Each type is asked directly rather than looked back up by the name it just reported, because that round trip does not always close: one built-in type hardcodes the readable name `marker`, while the registry matches built-ins on their *enum constant* name, so resolving `marker` returns nothing at all.[^14] Asking the object in hand avoids inventing an answer for a discrepancy that belongs to the engine.

## Getting pools to the planner

Pools ride in the world stream as their own record kind, counted by the frame record alongside the visible entities ([[wire-contract-ndjson]]).[^10] Each carries both a tile coordinate and the world point at the tile's centre: the tile coordinate is the pool's identity — integral, fixed for the life of the map, and the unit the engine's own check works in — while the world point is what a build order has to be addressed to.

They are carried in every sample rather than sent once at connect time. Repeating static data has a cost, but a self-contained sample is what lets a captured session replay through the planner with nothing else alongside it, and 46 records at four samples a second is not the expense worth optimising first.

The tile grid is scanned once per map rather than per sample. Pool tiles never move, and 52,900 reflective lookups four times a second would be a real cost paid for an answer that cannot change. The scan is cached against the map object, so a new match invalidates it by identity without anything having to remember to. What is re-evaluated every sample is which of those pools the player can see, through the engine's own per-tile fog test — the tile counterpart of the entity visibility test, applying the identical comparison against the asking player's fog grid ([[perception-visibility]]).

## Occupancy is the planner's judgement

The engine has no "is this pool taken" test to delegate to. Its placement check answers only "is this tile a pool";[^3] a second extractor on an occupied pool fails later, on the ordinary overlap rule, and silently. So occupancy is decided in the policy, where it can be read and changed.[^15]

A pool counts as taken when an *immobile* visible entity stands within one tile of its centre. Three parts of that carry weight. Immobile, read from the catalogue's speed field, because the builder ends every build standing on the site it just used and counting it would burn the pool it had just successfully used. Visible rather than owned, because an opponent's extractor holds a pool exactly as firmly as ours. And one tile — 20 world units — because that is the grid pitch, against an extractor collision radius of 18 declared in its own definition.[^7]

A type the catalogue does not know is treated as not occupying. The two errors are not symmetric: guessing "free" wrongly costs one order the engine refuses, which the stall detector already catches, while guessing "taken" wrongly hides that pool for the rest of the run.

## What a live run does with it

Five orders, five structures, nothing wasted.[^11] Three extractors landed on pool tile centres at (4070, 2610), (4470, 2610) and (3690, 2370) — tiles (203, 130), (223, 130) and (184, 118), all three confirmed `res_pool` in the map file — and two land factories took ring offsets from the Command Center ([[policy-loop]]).

Getting there exposed a defect that had nothing to do with pools and everything to do with distance, recorded here because pools are what surfaced it. The two nearest pools are within 230 world units of the base; the third is 588 away. The stall detector's window ran from the moment an order was sent, which silently capped how far the bot could build: at a measured 11.7 world units per sample, 45 samples reach 527 units, and the order to the third pool was declared refused while the builder was still walking to it. It completed seconds after the run gave up, standing exactly on the pool tile it had been sent to.[^12]

Timing one far build settled it. Ordering an extractor 609 units away, the builder travelled for 52 samples and the structure appeared on the very sample it stopped moving — construction itself cost nothing measurable at this sampling rate.[^12] Travel is the whole of the delay, so a builder that is still moving is an order still in flight, and the stall clock now only runs while the builder stands still. That needs no speed constant, no frame rate, and no assumption about map size.

Worth noting against the catalogue: the builder's listed speed is 0.6, and the measured ground rate is 46.9 world units per second.[^12] The catalogue's speed figure is therefore not world units per frame, and nothing should treat it as a distance per unit time without calibration ([[mechanics-unit-catalogue]]).

## Open questions

Pools were chosen by distance from the base and nothing else, and that was too little. Aiming a builder at the *farthest* pool on this map — 4,293 world units out — walked it through two opponents' bases, and it was killed before arriving.[^13] Distance is not danger, and it is not reachability either: on a map where the nearest free pool is across water the same order would send a land builder somewhere it cannot go. Both want a model the planner did not have, one of threat and one of movement layers.

**Half of that is now answered.** A pool is rejected when a visible hostile's attack range covers any point of the walk to it, hostility read from the engine's own alliance test rather than from ownership, and the survivors ranked by distance as before ([[policy-threat]]). The route is what is tested rather than the destination, because the builder in [^13] died in transit and the pool it was walking to was fine.

**And now the other half.** The engine precomputes connected components per movement layer, so reachability is a comparison rather than a search, and the planner rejects a pool whose land component the builder's does not match ([[mechanics-movement-layers]]). On this map that is not a corner case: twelve of the forty-six pools sit in six island components no land unit can walk to, and distance-only selection would have aimed a builder at one of them as soon as the near ground filled.

Extractors are also never upgraded. The catalogue prices a T2 and T3 tier for each, and the engine treats an upgrade as a distinct action from a build; nothing in the order path exercises it yet ([[building-structures]]).

And the fog filter over pools has never filtered anything, because fog is disabled on this map. That is now reported on every map scan rather than assumed, so a run where it starts mattering will say so ([[perception-visibility]]).

[^1]: `.game/assets/tilesets/misc.tsx:6` — `<property name="res_pool" value=""/>` under `<tile id="4">`. The map embeds its own copy of this tileset with the same property on the same id, at `firstgid="371"`, so the marked gid is 375.
[^2]: `com/corrodinggames/rts/game/b/g.java:215` in the decompiled tree — `if (properties.getProperty("res_pool") != null) { ((g)object).i = true; }`, one arm of the block that also sets the water, lava and cliff flags.
[^3]: `com/corrodinggames/rts/game/units/y.java:4636` — `if (this.r().p()) { l2.bL.a(this.eo, this.ep); object = l2.bL.e(l2.bL.T, l2.bL.U); if (object == null || !((g)object).i) return "{2}"; }`. The `"{2}"` sentinel is mapped to the message field holding `gui.cannotPlace.needsResourcePool` at `com/corrodinggames/rts/gameFramework/f/g.java:1885`, set at `:356`.
[^4]: `wiki/sources/m11-pools/pool-build-run.log:401` — `[rw-agent] map scan: 46 resource pool(s)`.
[^5]: `wiki/sources/m6-wire/world-sample.ndjson:20` — `{"kind":"pool","frame":3569,"index":0,"tile_x":115,"tile_y":6,"x":2310.0,"y":130.0}`, the first pool record of the archived capture; the frame record at `:1` declares `"pools":46`.
[^6]: [synthesis] — the map's `Items` layer at `.game/assets/maps/skirmish/[z;p10]Crossing Large (10p).tmx` is base64 gzip; decompressed and scanned for gid 375 it yields 46 cells, the first at tile (115, 6), matching [^5]. Derived rather than archived because the map file is the primary source and ships with the game.
[^7]: `.game/assets/units/extractor/extractor_common.ini:25` — `placeOnlyOnResPool: true`, with `radius: 18` at `:20` and `isBuilding: true` at `:23`. Every shipped extractor tier inherits this file via `copyFrom`. The loader reads the key at `com/corrodinggames/rts/game/units/custom/ag.java:1710` into the field the type predicate returns at `custom/l.java:723`.
[^8]: `.game/assets/translations/Strings.properties:557` — `units.extractor.description=[[Generates credits.]]  [[Can only be built on resource pools.]] [[Upgradable to T3]]`. The same English sentence appears verbatim in the Japanese bundle, which is what shows it to be untranslated prose rather than generated text.
[^9]: `wiki/sources/m11-pools/type-flags.ndjson:85` — `{"kind":"unittype","index":84,"name":"extractorT1","needs_pool":true}`; the file's 173 records include exactly eight with `"needs_pool":true`, the first being the built-in `extractor` at `:1`.
[^10]: `agent/src/rwbot/agent/StateStream.java` — `poolRecord` and the `pools` count on the frame record; `agent/src/rwbot/agent/MapTiles.java` holds the scan, the cache and the fog filter.
[^11]: `wiki/sources/m11-pools/pool-build-run.log:403` — the five `channel: build` lines the agent logged, three `extractorT1` and two `landFactory`, against a scorecard of `completed 5/5, orders sent 5`.
[^12]: `wiki/sources/m11-pools/builder-travel-timing.txt:13` — `RESULT travel_samples=52 total_samples=52`, with `construction_samples=0` at `:14`, `units_per_sample=11.72` at `:16` and `units_per_second=46.9` at `:17`, for the 609.3-unit order named in the header at `:5`. A one-off calibration rather than a regenerable target: it needs a live game and a builder free to walk.
[^13]: [synthesis] — the same measurement aimed at the farthest pool on the map first, 4,293 units out at tile (12, 52). The builder was still walking at sample 370, having reached (498, 1198) against opposing bases at roughly (370, 2430) and (590, 1690) in the archived capture, and had left the roster by sample 480. Recorded rather than archived because what the run establishes is a negative.
[^14]: `com/corrodinggames/rts/game/units/ar$48.java` in the decompiled tree — `public String i() { return "marker"; }`, against the by-name lookup at `ar.java:291` which compares `ar2.name()`. Observed as a live failure first: the dump aborted with "the registry listed type marker but cannot resolve it by name".
[^15]: `src/rw_bot/policy/build_order.py` — `survey_pools` and `_is_occupied`, with `POOL_OCCUPIED_RADIUS` and the reasoning for each of its three conditions. The threat filter it applies before ranking lives in `src/rw_bot/policy/threat.py` ([[policy-threat]]).
