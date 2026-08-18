---
title: "Production: Keeping the Queues Full, and Why the Bot Built 33 of the Same Unit"
tags: [policy, production, composition, measurement, verification]
related:
  - "[[policy-economy]]"
  - "[[policy-budget]]"
  - "[[policy-combat]]"
  - "[[policy-loop]]"
  - "[[mechanics-unit-value]]"
  - "[[mechanics-build-actions]]"
  - "[[ai-opponent-strategy]]"
source_paths:
  - "src/rw_bot/policy/production.py"
  - "src/rw_bot/policy/campaign.py"
  - "scripts/play.py"
  - "wiki/sources/m20-one-tick/multi-builder-ab.txt"
  - "wiki/sources/m21-losses/where-units-die.txt"
  - "wiki/sources/m22-workers/worker-ceiling-ab.txt"
source_git_blobs:
  "src/rw_bot/policy/production.py": "3ccbb9f5aec7bffa5fece236bf8a1d9684ebc110"
  "src/rw_bot/policy/campaign.py": "ae3700f5a5c413b05f2909de398d1154d8262b2f"
  "scripts/play.py": "c95267440b13b56f48a32def2a00b99e2f9efe72"
  "wiki/sources/m20-one-tick/multi-builder-ab.txt": "f4bebb002c2360618983acaf73bc6d1dace5ad31"
  "wiki/sources/m21-losses/where-units-die.txt": "280848152bf106e667622974a0477c521f3c3801"
  "wiki/sources/m22-workers/worker-ceiling-ab.txt": "d21b022d56bab638533e62826c239975125c2e3d"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture]
---

# Production

The planner executes a list and the list ends. The engine's own AI never finishes, because it samples from a weighted mix rather than working through an order.[^1] Production is what closes that gap: the plan said a tank was wanted, and production does not stop wanting one.

Three separate failures have been recorded here, and each one was invisible until something was measured.

## 1. The idle-producer test that could never fire

`production_bound` asks whether the player has money it has nowhere to spend. Its first form asked "is every producer busy?" — counting every owned unit. A Command Center offers a Builder and is therefore idle almost permanently, so the answer was *no* on every observation of every match and the rule never fired at all.

The fix is that **producing is the engine's own answer**, carried on the option stream: a unit offering a non-placed action it may use right now. Something that makes nothing is not idle capacity; it is a wall.

## 2. Buying throughput with the money that was buying income

Restricting the test to producers of a *wanted* type made it fire, and firing made the bot worse. Three 1500-sample matches on one seed:

| factories added | extractors | income | army value |
|---|---|---|---|
| 0 | 10 | 98/s | 6,450 |
| 1 | 9 | 90/s | 4,000 |
| 7 | 7 | 74/s | 3,300 |

Income is the low-variance figure and it moves monotonically the wrong way. The mechanism is not noise: **every factory the builder places is an extractor it does not**. Income gates production, so buying capacity with the money that would have bought income trades the thing that was working for the thing that was idle. The rule now requires the queues to be genuinely full *and* the surplus to be genuinely spare, and the conservatism is deliberate.

## 3. A priority list cannot express an army

`sustain` took the first wanted type a producer could make and stopped. Every idle producer reached the same first entry, so whatever stood at the head of the list was the only thing the bot ever built. `reinforcements` made this worse by collapsing duplicates, on the reading that four tanks meant one preference stated four times.

The result was structural, not accidental: **a mixed army was impossible to ask for.** Three matches ended with 33 identical `c_tank` — a unit that cannot shoot at aircraft at all, against opponents fielding ~15 visible units it could not touch, and whose 130 reach is shorter than every static defence in the game.[^2]

Repeats are now the ratio. `("c_tank", "c_tank", "hoverTank")` means two of the first for every one of the second, and each idle producer builds whatever the roster is furthest short of — measured as a **share**, so one rule covers an army of three and an army of three hundred.

Two details are load-bearing:

- **Orders decided earlier in the same tick count toward the roster.** Every producer reads the same observation; without this a batch of idle factories all see the identical shortfall and all fill it with the identical unit, which is the old bug rediscovered one tick at a time.
- **The worker fallback is not part of the composition.** A share is owed to everything in the mix, and a builder owed a share of a 34-unit roster is a land factory ordering builders. It is a separate argument, reached only by a producer that can make nothing in the mix — which on this build means the Command Center and nothing else.

Filling the widest gap each time interleaves the types (tank, scout, tank) rather than emitting the ratio in blocks. That is the better of the two: the second type exists a build sooner and the mix is never wrong.

## How many builders?

Everything above `c_tank` in the value table is built by a **builder**, not a factory.[^2] The bot's spending ceiling is therefore set by how many builders it has, and the land factory — which can only start one 350-credit unit at a time — cannot absorb 122 credits per second on its own.

A first attempt bought a builder whenever all of them were busy. That bought 33 of them in a 1500-sample match: 16,500 credits of labour, with most of the reported army value turning out to be builders rather than anything that fights. "All of them are busy" is not a shortage; it is success. This map carries 46 pools, so unclaimed work never runs out and the rule never stopped buying.

A ceiling was added on the argument that a builder is 500 credits of thing that does not fight. Measured, the argument does not hold up — capping builders caps extractors, which caps income, and the surplus then has nowhere to go:

| worker ceiling | extractors | income | worth | best rival | ratio | banked |
|---|---|---|---|---|---|---|
| 4 | 10 | 98/s | 24,600 | 25,350 | 0.97 | 17,938 |
| 8 | 10 | 98/s | 23,450 | 26,850 | 0.87 | 15,229 |
| uncapped | 13 | 122/s | 40,850 | 26,650 | 1.53 | 4,620 |

*Seed 12345; the uncapped row is from the earlier multi-builder A/B on equivalent code.* The banked column is the diagnosis: both capped arms end the match sitting on 15,000–18,000 credits they could not spend. The ceiling is kept as a parameter rather than a constant precisely so this could be asked of a run rather than of an argument.

## What production still does not decide

**Affordability.** `sustain` used to budget across the batch against `sample["credits"]`, which was correct alone and wrong in company: the expansion pass budgeted against the same field in the same observation, so the pair committed one credit twice. What a producer *could* start is this module's question; what the player can afford has exactly one owner.[^3]

**Placement.** Only produced units are ordered, never placed structures — a queue cannot express a position.

**Availability.** The unit cap and tech gating are the engine's own predicate, asked by the agent rather than modelled here.[^4]

[^1]: See [[ai-opponent-strategy]] § "Production is a weighted mix, not a build order".
[^2]: See [[mechanics-unit-value]] and `wiki/sources/m21-losses/where-units-die.txt`.
[^3]: See [[policy-budget]] § "Refusals are the informative half".
[^4]: See [[mechanics-build-actions]] § "What actually stops a production order".
