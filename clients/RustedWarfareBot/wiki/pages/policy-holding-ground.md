---
title: "Holding Ground: 44 of 46 Pools, and Why the Bot Loses"
tags: [policy, economy, combat, measurement, verification]
related:
  - "[[policy-economy]]"
  - "[[policy-combat]]"
  - "[[policy-production]]"
  - "[[mechanics-unit-value]]"
  - "[[mechanics-resource-pools]]"
  - "[[harness-parallel-matches]]"
source_paths:
  - "wiki/sources/m28-holding/extractor-survival.txt"
  - "wiki/sources/m25-duration/full-length-matches.txt"
  - "wiki/sources/m24-wave-mass/wave-mass-ab.txt"
  - "wiki/sources/m26-upgrades/structure-offers.txt"
  - "wiki/sources/m27-aggression/attack-vs-turtle.txt"
  - ".game/assets/units/extractor/extractorT2.ini"
  - ".decompiled/com/corrodinggames/rts/game/units/custom/ag.java"
  - "src/rw_bot/policy/economy.py"
  - "src/rw_bot/policy/spending.py"
  - "scripts/play.py"
source_git_blobs:
  "wiki/sources/m28-holding/extractor-survival.txt": "df5494b71e74b7d8b57273c84113d62edfcdb49c"
  "wiki/sources/m25-duration/full-length-matches.txt": "9fd11e26631af596ac476d8c1a7ff1d58f9c8314"
  "wiki/sources/m24-wave-mass/wave-mass-ab.txt": "843eea72af8a2e8029ac5a70fd341c8d30591ca1"
  "wiki/sources/m26-upgrades/structure-offers.txt": "9147f4d519e8ffb37ffbb6d06ab8536a1b53e07a"
  "wiki/sources/m27-aggression/attack-vs-turtle.txt": "e9fdde7396a6a99b6cc35b14b59e193b003b198a"
  "src/rw_bot/policy/economy.py": "ecb91c97c17306e889a9c49209bf5baa6a3efe13"
  "src/rw_bot/policy/spending.py": "cdef325f6e784124d362b1b910e3c716f4c8507c"
  "scripts/play.py": "7aa64367cc0c59a7a930eb75fd5079ffb1d466ee"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture, game-mechanics]
---

# Holding Ground

**The bot loses.** Every measurement before this page was taken at 1,500 samples, and 1,500 samples is a transient. Played to a verdict, the bot is defeated or wiped in most matches.

## What a full-length match actually does

Four seeds at 4,000 samples:[^1]

| seed | verdict | ended at | our worth | strongest rival | our army |
|---|---|---|---|---|---|
| 12345 | **defeated** | 3,692 | 1,400 | 131,450 | 0 |
| 4242 | **defeated** | 3,483 | 1,050 | 151,100 | 1 |
| 777 | **wiped** | 3,559 | 0 | 142,300 | 0 |
| 31337 | survived to the limit | 4,000 | 6,300 | 129,200 | 2 |

At sample 1,500 the same bot reads as **1.26× ahead** on total worth. By 3,500 it has been annihilated: extractors 14 → 0, workers 36 → 0, income 130/s → 0, while the strongest opponent compounds 4,700 → 131,000–151,000 and fields 500+ visible units.

Everything tuned against the 1,500-sample figure — the worker ceiling, the army composition, the wave mass — was tuned on a position that does not survive. Those measurements are not wrong; they answer a question that turned out not to be the question.

## Why: the map is taken, and we cannot hold what we buy

The build policy reports the cause itself, without any new instrumentation:

    plan 6/8 -- blocked: extractorT1 needs a resource pool: of the 46 in sight,
                44 are built on, 0 cannot be walked to,
                and 2 can only be reached through enemy fire

**The opponents own 44 of the map's 46 resource pools.** The bot ends with one.

The cost of arriving there is the finding. The same run records `expansions 275 (28 factories)` — roughly **247 extractor orders for one surviving extractor**. The bot is not failing to *order* expansion; it orders it constantly and ends with one pool.

**What those 247 orders bought is not yet measured, and the two possibilities call for opposite fixes.** An order that was granted by the budget is not a structure that went up: the builder still has to walk there, and the engine refuses a placement silently. So the run is equally consistent with 247 extractors built and destroyed — in which case the answer is defence or immunity — and with a handful built while the rest were re-issued against pools that were taken before the builder arrived, in which case the answer is claiming faster or claiming nearer. The endpoint scorecard cannot tell them apart, which is the exact distinction [[policy-trace]] was written to make: per-sample extractor counts say *when*, and the per-loss table says *where*. Until that trace is read, "loses every claim" is a reading of the data rather than a finding in it.

**So expansion without defence looks like a credit shredder**, and that reading produced two hypotheses. Covering structures was measured and refuted. Upgrading extractors turned out to be reachable after all — and the reason it had looked unreachable was a filter in our own agent rather than anything in the game.

## The bot cannot build a single defensive structure

Not by policy, but by construction. Three gates, each individually defensible, which together make almost the whole game unreachable:

1. **`economy.py` names two types.** `EXTRACTOR_TYPE = "extractorT1"` and `FACTORY_TYPE = "landFactory"` are the only structures any code path places.
2. **`reinforcements` drops every immobile type** (`speed > 0.0`), so no structure can enter the army composition and be produced.
3. **`sustain` skips placed options**, so a producer offering one is passed over.

A builder can place thirteen things. The bot places two:

| | price | placed? |
|---|---|---|
| **c_turret_t1** | 500 | never |
| **c_antiAirTurret** | 600 | never |
| laserDefence | 1,200 | never |
| repairbay | 1,500 | never |
| fabricatorT1 | 2,200 | never |
| airFactory | 1,000 | never |
| mechFactory | 1,000 | never |
| seaFactory | 1,000 | never |
| experimentalLandFactory | 11,000 | never |
| antiNukeLauncherC | 15,000 | never |
| nukeLauncherC | 45,000 | never |
| extractorT1 | 700 | **yes** |
| landFactory | 700 | **yes** |

`c_turret_t1` is the best damage per credit in the game — 16.40 against `c_tank`'s 5.71 — and the best hit points per credit at 140 against 60.[^2] It costs less than the extractor it would defend.

The build tree also lists seventeen upgrade paths that no code path reached. For the extractor's own, none of these three gates was the binding constraint: a fourth one, in the agent rather than the policy, was discarding the action before the planner ever saw it. That is measured below, and the extractor upgrade is now taken.

## What was fixed, and what it did not fix

A builder is produced by the Command Center **and by every Land Factory**. The policy asked for one only as a *fallback*, and a fallback is reached only by a producer that can make nothing in the army composition — which a Land Factory never is, because it can always make a tank. So when the Command Center died, twenty-two factories went on building tanks while the player had no builder, no way to place another extractor, and no way back: the runs end `plan blocked: nothing the player owns can make extractorT1` with `workers 0`.

At zero builders the builder now goes **into** the composition rather than behind it, where any producer that can make one will. It is self-limiting there — a share of the roster is a share — so it does not recreate the 33-worker runaway.

**Measured, it changed nothing.** The same four seeds went from one survival to two, with one seed dying 294 samples *earlier*: noise, not a result.[^1] The bug was real and the fix stays, because a permanent-death trap is indefensible whatever the scoreboard says. But it is a safety net that rarely gets the chance to fire, because by the time the last builder dies everything else is already gone.

## The upgrade path was reachable all along, and our own agent hid it

This section records three consecutive wrong answers to one question, because the sequence is the useful part. The question was: can an owned extractor upgrade itself?

1. **"No — the engine never offers it."** A probe played the real opening until four extractors were standing, asked what every owned structure offered, and got nothing.[^3] Correct observation, wrong conclusion.
2. **"No — it is gated behind a tech level the bot never has."** `extractorT2` declares `techLevel: 2`, and `ag.java:594` registers a type's build action only into the action lists at or above its tier. That is all true, and it is not why the extractor was silent.
3. **"No — it needs a tier-2 builder, costing 44,500 credits of experimental prerequisites."** The build tree says `extractorT2` is produced by `combatEngineer`, `mechEngineer` or `extractorT1`; the first two are unreachable, so the chain looked like builder → `experimentalLandFactory` (11,000) → `experimentalDropship` (30,000) → `combatEngineer` (3,500). Also true, also not the answer.

**The real answer: the agent was dropping the action before it reached the wire.** `BuildOptions` filtered out any action that neither placed something nor answered true to the engine's "makes something" predicate, on the reading that the remainder were stops and rallies. An upgrade is neither — the asset declares it as `convertTo`, a conversion — so it was discarded silently.

With every action published, all four extractors offer `extractorT2` and the engine calls it **available**, at tier one, with no prerequisites at all:

    extractors standing: 4
    holder            unit  produces      placed  available
    commandCenter      213                 False       True
    commandCenter      213  builder        False       True
    commandCenter      213  scout          False       True
    extractorT1        227  extractorT2    False       True
    extractorT1        241  extractorT2    False       True
    extractorT1        270  extractorT2    False       True
    extractorT1        277  extractorT2    False       True

Every one of the three wrong answers was reached by reasoning from real evidence — an accurate catalogue, an accurate build tree, accurate decompiled source. What settled it was removing a filter and asking the engine again.

**Two further faults surfaced on the way to a working upgrade**, and both crashed the match rather than degrading it:

* The `makesSomething` filter existed in *two* places: the listing path and `actionMaking`, the dispatch path. Removing it from one produced the worst of both — the planner was offered an upgrade it could see, and the agent then could not find it to dispatch, threw inside the engine's script thread, and killed the game with `extractorT1 has no action making 'extractorT2'; it can make nothing`.
* A conversion **does not fill the production queue**. `queued` stays at zero for as long as it runs, so the structure keeps offering the upgrade it is already performing, and the order was re-sent every observation. One duplicate arrived after the conversion had completed, addressed to a unit that was now an `extractorT2` and could only make an `extractorT3` — which is how the second crash announced that the upgrade had worked.

Measured live over 800 samples once both were fixed: **income 54/s, which is the base 18 plus three tier-two extractors at 12** — the first income this bot has produced above the tier-one ceiling.

A reporting fault was caught in the same run and is worth naming, because it is the same shape as the 1,500-sample mistake. `count_extractors` matched `extractorT1` alone, so a player holding three upgraded extractors was reported as holding **none**. A figure that quietly means something other than what it says is how a reading goes wrong.

The tiers pay 8, 12, 20 and 30 credits per second, so this is the largest economic lever in the game and the only one needing no builder, no travel and no contested ground — an extractor upgrading *itself*, in place, on ground already held. On a map where the opponents finish holding 44 of the 46 pools, and where 247 expansion orders leave the bot with one extractor, that immunity is the whole point.[^1]

**A claim from a table is not a measurement, and neither is a single negative observation.** The 2.3x figure was derived from the catalogue and the build tree, both accurate, and was wrong about availability. The probe then said "unreachable", and that was wrong about why. An absent action, a gated action and a filtered action look identical from outside.

## Defence does not save it either

Three arms across four seeds, played to a verdict:[^1]

| arm | survived | defeated | wiped |
|---|---|---|---|
| no defence | 2 | 1 | 1 |
| turrets before income | 0 | **4** | 0 |
| turrets from surplus | 1 | 2 | 1 |

Turrets ahead of income lost every match, and the scorecards name the mechanism: there is always some uncovered structure, so the rule took the builder nearly every tick and expansion collapsed from 275 orders to about 40. Funding them from the surplus instead — after income, before throughput — restored expansion and still did not help.

`c_turret_t1` really is the best damage and hit points per credit in the game. It does not matter. The bot peaks near 40,000 worth while the leader alone reaches 131,000–151,000, and no placement rule closes a three-to-fourfold economic gap.


## In a duel, the whole match is whether the extractors survive

The section above is about the four-opponent game on the ten-player map. Against **one** opponent the same question has a sharp answer, and it settles the open question this page has carried since it was written: *what did those 247 expansion orders buy?*

Twelve seeds at Hard, 4,000 samples, one opponent on `[p2]duel_lake`. Reading the per-sample extractor count rather than the endpoint:[^5]

**Every one of the twelve reaches a peak of three extractors**, all of them inside the first quarter of the match — first extractor at 1–4% of the run, peak at 9–22%. There is no race being lost. The verdict follows what happens *after* the peak:

| extractors lost | seeds | verdict |
|---|---|---|
| 0 | 4242, 555, 777 | **won** |
| 1 | 31337 | **won** |
| 2 | 60613, 8675309, 90210, 99991 | not won |
| 3 | 12345, 1337, 8128, 24601 | not won |

Zero or one loss won four matches of four. Two or more lost eight of eight. No overlap.

**So the answer to the 247 orders is "built and destroyed", not "never built".** Every run also *regains* three or four extractors, which is the same shredder seen in the duel: the bot re-buys the same ground at 700 credits a time and cannot keep it.

The final income figure that looked like the cause is a restatement of this. Ending income was 18, 38, 50 and 58 credits per second for nought, one, two and three surviving extractors — the base 18 plus the survivors' own rates, whose steps of 20, 12 and 8 are consistent with the survivors being upgraded. Reading only the endpoint made a *consequence* look like the lever.

### The extractors are raided, not lost at a collapsing front

Two readings of the per-loss table, which records the type and last position of everything that left the roster:[^5]

**The losses are real deaths rather than upgrades.** A conversion also leaves the roster, so the entries needed checking before they could be counted. The three seeds that lost nothing record **no extractor departure at all** — and their ending income of 58/s proves they upgraded, since the base 18 plus three tier-one extractors is 42. An upgrade therefore keeps its identity, and the losers' entries are deaths.

**Every extractor dies far from where the army is fighting.** Measuring each one against the centroid of that run's own tank losses — which is where the front is, by definition:

| | extractor deaths | distance from the front | army's own spread |
|---|---|---|---|
| 8 seeds that lost ground | 2–3 each | 688–1,766 | 229–829 |
| 4242, 555, 777 | none | — | — |

Not one died inside the cloud. They are picked off at sites the army is nowhere near, spread across 22–98% of the match rather than in a single late collapse. **That is the case a static defender exists for** — and it is why the answer is a turret at the pool rather than a better attack policy or a bigger army.

### Why defence had never helped: it was covering the wrong thing

`undefended` offered every immobile structure and took the one nearest the anchor, "so the base is covered before the frontier". The per-loss table refutes that ordering outright:

| distance from base | run A | run B |
|---|---|---|
| < 300 (at the base) | 0 — 0% | 0 — 0% |
| 300–900 | 0 — 0% | 3 — 17% |
| 900–2,000 | 2 — 4% | 2 — 11% |
| > 2,000 (deep) | 46 — 96% | 13 — 72% |

**Not one unit died within 900 world units of the base across either run**, and the structures among the losses are extractors, out where the pools are. Nearest-first therefore spends the defence budget on the one place never attacked, and reaches an extractor only once every base building has cover it does not need.

### And aiming it at the extractors lost anyway

That was the strongest argument this policy has had, and the measurement refused it. Cover restricted to extractors, nearest first among those; same twelve seeds, same rung, nothing else changed:[^6]

| | wins | extractor drops | defeats | wipes |
|---|---|---|---|---|
| base first | **4** | 21 | 0 | 0 |
| extractors only | **0** | **24** | 1 | 1 |

Drops did not fall, and the first two losses in fifty-two duels appeared. The rule is therefore back as it was, and the reasoning is recorded because it was good and still lost.

**Four arms have now failed: defence ahead of income, defence from surplus, defence aimed at the base, defence aimed at what actually dies.** That is enough to stop treating the turret as the answer to extractor loss, whatever the value table says about it.

**But the arm was never a fair test, and the reason is the finding worth keeping.** The new `structures` line shows what the run output had never been able to say: across all twelve matches, **three turrets were standing at the end** — two in one seed, one built and destroyed in another. Defence has never been a policy that ran, only one that was reached. Two candidates for why, both unverified: it fires only when income declines, which on a nine-pool map is seldom; and the site it picks is a bare `+60` offset from the structure, never checked against terrain, which at a pool would be refused silently by the engine at the cost of a walk and a stall window per attempt.

**A measurement gap was closed in the same change, and it is the more valuable half.** Nothing reported whether a turret had ever been built: our own buildings appeared in no report line, the trace carried no column for them, and the expander keeps the *income* reason when defence declines, so the defence reason never reached a log at all. Asked whether one turret had been built across twelve full matches, the honest answer was that the run output could not say. That question is now answerable, and answering it is what turned "defence is aimed wrongly" into "defence barely happens".

### What the army cannot do at all

`c_tank` — the only unit the opening plan builds — declares `canAttackFlyingUnits: false`, and the printunits dump says it in words: *"Can attack ground only"*. `c_turret_t1` is *"Attacks ground units."* **That is the entire army and the entire defence, and neither can touch an aircraft.**

The engage filter makes it visible in every report. In all four winning duels the surviving enemies are **0 engageable** — every last one is something the bot cannot shoot. In the losers, 20–25% of the opponent's force is.

This is not yet shown to be what kills the extractors, and it must not be assumed: 75–80% of the opponent's units are ground, and any of them could be the raiders. But no turret we can currently place would stop the other fifth, which is what makes [[policy-combat]]'s anti-air gap the next thing to test rather than a fifth variation on the turret.

## Not attacking does not save it either

The hypothesis none of the economic work touched was that the *attacking* is what kills us. It needed no code to ask -- the wave ladder's final rung is already an argument, and a mass larger than any army the bot can field releases no wave ever, so the army gathers, rallies and defends without ever entering the opponents' half.

Three seeds per arm, played to a verdict:[^4]

| arm | survived | defeated | wiped | final worth |
|---|---|---|---|---|
| attack | 0 | 3 | 0 | 500 / 500 / 350 |
| turtle | 0 | 3 | 0 | 3,750 / 1,050 / 1,700 |

Refuted. Final worth is consistently higher without attacking -- the units genuinely are not thrown away -- and it changes no verdict.

## What is now closed, and what is left

Four hypotheses have been closed by measurement rather than argument: more builders, defensive structures ahead of income, defensive structures from surplus, and not attacking at all. None of them changes the outcome.

What survives is a mechanism found in the engine rather than in a scorecard. Unit types declare a `techLevel`, and `ag.java:594` registers a type's build action **only into the action lists at or above that level**:

    for (int i2 = n2; i2 <= 3; ++i2) { object = as2.a(i2); }

So at tech 1 a tier-2 action is *absent* rather than refused, which is exactly why an owned extractor offers nothing. `extractorT2` declares `techLevel: 2` and pays 12 credits per second against T1's 8; T3 declares 3 and pays 20, and the overclocked tier 30.

An opponent at tier 3 therefore earns **3.75x per extractor** what this bot earns, which is the right order of magnitude for a 40,000-against-150,000 worth gap that no placement rule, worker count or attack policy has been able to move.

Two things are not yet established, and both are cheap: what raises a player's tech level -- no asset declares it as a grant, and the only framework reference is a map-scripting filter -- and whether the opponents actually field tier-2 types. The second is now answered by every match, because what the opponents are fielding is a reported figure.

[^1]: `wiki/sources/m25-duration/full-length-matches.txt` — four seeds before and after the builder fix, played four at a time ([[harness-parallel-matches]]).
[^2]: See [[mechanics-unit-value]] § "The table" for the value table and `.game/assets/units/extractor*/` for the generation rates: 8, 12, 20 and 30 credits per second across the tiers.
[^3]: `wiki/sources/m26-upgrades/structure-offers.txt` — `scripts/upgrade_probe.py` played the real opening and then asked the engine, rather than reading the build tree.
[^4]: `wiki/sources/m27-aggression/attack-vs-turtle.txt` — six matches in one batch, the arms differing only in the wave ladder argument.
[^6]: `runs/sweeps/duel-hard-defence/` against `runs/sweeps/duel-hard/` per `wiki/log.md:751` — twelve seeds each, differing only in which structures `undefended` offers.
[^5]: `wiki/sources/m28-holding/extractor-survival.txt`, with the two readers that produced it beside it. Read from the per-sample traces the twelve Hard duels already wrote (`runs/traces/duel-s*.ndjson`), so it cost no runs — the distinction [[policy-trace]] exists to make, finally asked of a batch that had it.
