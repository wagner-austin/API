---
title: "The Noise Floor: What a Match Measures, and How Many You Need"
tags: [harness, measurement, determinism, statistics, methodology]
related:
  - "[[harness-parallel-matches]]"
  - "[[policy-holding-ground]]"
  - "[[policy-production]]"
  - "[[ai-opponent-strategy]]"
source_paths:
  - "runs/sweeps/noise"
  - "runs/sweeps/noise-seeded"
  - "agent/src/rwbot/agent/EngineRandom.java"
  - "src/rw_bot/harness/sweep.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-27"
confidence: high
hubs: [headless-harness, bot-architecture]
---

# The Noise Floor

A match is not a measurement. Twelve of them are, barely, and only of the right figure.

## The measurement

Twelve matches from one identical job specification — same seed, same arguments, same code — played to a verdict:[^1]

| figure | unseeded | seeded | coefficient of variation |
|---|---|---|---|
| verdict | 3 survived / 9 defeated | 3 survived / 9 defeated | — |
| total worth at the end | 350 – 15,350 | 500 – 15,850 | **1.07** |
| income at the end | 0 – 54/s | 0 – 60/s | 1.00 |
| extractors at the end | 0 – 3 | 0 – 4 | 0.72 |
| expansion orders | 118 – 194 | 119 – 194 | 0.15 |
| samples seen | 3,232 – 4,000 | 3,005 – 4,000 | 0.098 |

**A 25% survival rate on an identical specification.** Any arm of six seeds reporting one or two survivals has reported the base rate.

## What this invalidated

Four arms measured before this floor was known, all of which read as results at the time and none of which was one: more builders at one survival in six against none; the army-composition arms at none in six each; the upgrade arm at three in six. All sit inside a 25% base rate.

Worse, a single run was cited as evidence that more builders worked — *survived, four extractors, 66 credits a second, total worth 13,800*. It is the same specification as these twenty-four and sits above every one of them. It was the top of the distribution read as the effect of a change.

**Two findings survive, because they fall outside the floor.** More builders raised expansion orders from 16–22 to 107–182, where the floor's own spread on that figure is 118–194 — non-overlapping against the low arm. And throughput-before-income wiped three matches in six against **zero** wipes in twenty-four here ([[policy-production]]).

## Why seeding did not fix it

The agent pins the engine's own generator, and that is not the only one. Twelve engine call sites use `java.lang.Math.random()`, a JVM-global generator nothing was seeding, and they are not incidental: the AI picks *which unit to plant a new base at* through it, positions its sites and its workers' destinations on a random disc around them, and scatters unit positions by up to eight world units.[^2] So the opponents built their bases somewhere different every run.

That is real, it is now seeded, and **it changed the distribution not at all** — three survivals in twelve either way.

Two reasons, and both matter for anything else attempted here:

* Seeding fixes the *sequence*, not which draw each consumer receives. If the number of calls before a decision varies, every consumer downstream of it shifts.
* The map settles for **22 seconds of free-running wall clock** before the planner attaches, on a simulation that advances by the millisecond delta Slick measured for the frame.[^3] Runs therefore begin from worlds that already differ.

The seeding stays, because it removes a real uncontrolled source and costs nothing. It is not a solution.

## The standing rules

**Score on survival time, or on worth share.** The endpoint figures the scorecard reports are the worst available: final worth has a standard deviation larger than its own mean. Computed from the per-sample trace instead, the mean share of worth held against the strongest rival has a coefficient of variation of **0.066**, and survival time **0.098** — the endpoint figure beside them is **0.67**.

**Pair arms across the same seeds.** Comparing per seed removes the map and the opponents from the comparison, which is most of what the spread is.

**Twelve runs an arm.** That gives a standard error near 87 samples on survival time, so a difference of about 250 samples is detectable. Detecting a change in the *verdict* rate from 25% to 50% would need roughly 58 matches an arm.

**Do not run one-match-per-arm screens.** A twelve-way screen of army compositions would report about three survivals whichever compositions it held. Twelve are written up and parked for exactly this reason.

[^1]: `runs/sweeps/noise/` and `runs/sweeps/noise-seeded/`, twelve results each from `sweeps/noise.txt` — one job line repeated twelve times under distinct labels, since results are filed by label.
[^2]: `.decompiled/com/corrodinggames/rts/game/a/a.java:1713,1737,1761`; `game/a/o.java:96-97,166-167` — `o.w()` returns a random point on a disc and `a.java:1575` hands it to a worker as a destination; `game/units/y.java:4811-4837`.
[^3]: `.decompiled/com/corrodinggames/rts/java/u.java:210-212` stores the delta, `:637` scales it, `:710` passes it to the simulation, `:714` resets it. See [[harness-parallel-matches]] for why the frame cap is not the lever it looks like.
