---
title: "The Build Tree, and Planning From Goals"
tags: [planner, build-tree, engine, policy, expansion]
related:
  - "[[policy-loop]]"
  - "[[building-structures]]"
  - "[[wire-contract-ndjson]]"
  - "[[mechanics-unit-catalogue]]"
  - "[[mechanics-resource-pools]]"
source_paths:
  - "wiki/sources/m11-pools/type-flags.ndjson"
  - "wiki/sources/m13-expand/expanded-run.log:410"
  - "wiki/sources/m13-expand/expanded-run.log:416"
  - "wiki/sources/m13-expand/idle-after-plan.txt"
  - "agent/src/rwbot/agent/BuildTree.java"
  - "src/rw_bot/mechanics/build_tree.py"
  - "src/rw_bot/policy/expand.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [engine-internals, bot-architecture]
---

# The Build Tree, and Planning From Goals

Ask the bot for a tank and it used to answer `blocked` — correctly, because nothing it owned could make one, and uselessly, because the way to get one was three lines away in the engine's own registry. It now derives the prerequisite instead.[^1]

## Two sources, deliberately

The world stream already carries what each owned unit can make ([[wire-contract-ndjson]]). That is the right source for **dispatch**: it carries the engine id an order is addressed to, and availability is genuinely per unit.

It cannot answer what a **plan** asks. A plan reasons about things that do not exist, and nothing the player owns can make a tank until a factory is standing — so on the question "how do I get one", the option stream is silent by construction.

So the static half is dumped from the registry: every type is asked for its own action list, and each action for the type it makes.[^2] 314 edges across 173 types.

The two cross-check, which is what makes either trustworthy. The registry says a Builder makes thirteen structures; the live option stream, reaching the same question by a completely different route, reports exactly those thirteen for the Builder the player owns.[^3]

## What it settles immediately

Nothing in the registry makes a `laboratory` at the base tech level.[^4] That is the whole of the failure that once ran three hundred samples reporting progress while the engine refused the order silently ([[policy-loop]]) — not a bug in the planner, but a plan that was never executable. It is now refused before a socket is opened.

## Tech level one, and why it is stated

The accessor takes a tech level and the answer differs between them, so a dump has to choose. One is what a match starts at and what an unupgraded building offers, which is what a plan opening from nothing needs.

An upgraded building's extra options still reach the planner live through the option stream, so nothing is lost at dispatch — only the static lookahead is limited to the base tier. That limit is stated here rather than discovered later.

## Expansion

Goals go in; an executable plan comes out, with prerequisites inserted where they are needed.[^1]

```
goals: extractorT1 -> extractorT1 -> extractorT1 -> c_tank -> c_tank
plan:  extractorT1 -> extractorT1 -> extractorT1 -> landFactory -> c_tank -> c_tank
```

Three properties are worth more than the mechanism:

**Availability accumulates.** Two tanks insert one factory, not two, because expansion runs over the whole list rather than per entry — otherwise the bot buys a second factory it does not need.

**Goal order survives.** A prerequisite goes in front of what needs it, not in front of everything, so an extractor asked for first still opens the plan and pays for the rest.

**The search terminates over a cyclic graph.** A factory makes a builder and a builder makes a factory. Expansion tracks what it is already resolving, so the cycle is a dead end rather than a hang; asked for a tank while owning nothing, it reports that `c_tank` needs one of `experimentalDropship` or `landFactory` and neither is reachable.

Cheapest producer wins, ties broken by name, so the same goals always expand to the same plan — a plan that varied run to run would make one run's failure impossible to reproduce with another's.

## Verified live

Six orders for six plan entries, no waste: three extractors and the inserted factory placed by the Builder, then **both tanks produced by that factory** — unit 267, which did not exist when the plan was made.[^5]

## What it does not do

It plans what to build, not what to do. A completed plan leaves the bot idle, and idle is measured rather than assumed: over five minutes after the plan finished, it took no damage, lost nothing, banked credits from 8,539 to 21,164, and watched visible enemy units go from 54 to 126.[^6] It survived because nothing reached it, not because it defended anything.

[^1]: `src/rw_bot/policy/expand.py` — `expand` folds goals into a plan against the tree and what is owned; `RW-EXPAND-001` is the refusal when a goal cannot be reached.
[^2]: `agent/src/rwbot/agent/BuildTree.java` — every registered type is asked for its action list at `BASE_TECH_LEVEL`, and each action for the type it makes; the records ride in the same dump as the placement flags because both are one pass over one registry.
[^3]: `wiki/sources/m11-pools/type-flags.ndjson` — the `buildedge` records for producer `builder`, thirteen of them, matching the thirteen options unit 214 reports in `wiki/sources/m6-wire/world-sample.ndjson`.
[^4]: `wiki/sources/m11-pools/type-flags.ndjson` [synthesis] — no `buildedge` record anywhere in the dump carries `"produces":"laboratory"`.
[^5]: `wiki/sources/m13-expand/expanded-run.log:410`–`:418` — three `build extractorT1`, one `build landFactory`, then two `produce c_tank by 267`, the factory the plan had just finished building.
[^6]: `wiki/sources/m13-expand/idle-after-plan.txt` — 800 samples of observation after a completed plan, with the credit and enemy-count series.
