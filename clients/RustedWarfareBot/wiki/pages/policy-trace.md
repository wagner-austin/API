---
title: "The Per-Sample Trace: Answering When, and Where"
tags: [harness, measurement, trace, diagnosis]
related:
  - "[[policy-determinism]]"
  - "[[policy-holding-ground]]"
  - "[[policy-production]]"
  - "[[policy-verdict]]"
source_paths:
  - "src/rw_bot/policy/trace.py"
  - "src/rw_bot/policy/recorder.py"
  - "src/rw_bot/harness/sweep.py"
  - "runs/traces"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-27"
confidence: high
hubs: [headless-harness, bot-architecture]
---

# The Per-Sample Trace

The scorecard keeps about two dozen endpoint figures. What falls between the first observation and the last is every question of the form *when did it turn*, and that is the question a pair of runs always ends up posing.

## The endpoints are not merely thin — they mislead

A full-length match reported `extractors 0 -> 0`. The trace of that same match shows it **held a peak of fourteen**, and led the strongest rival on total worth at the halfway mark, before collapsing.[^1] Nothing in the scorecard hints at either. `army 4 -> 12` reads identically whether thirty-seven units died in one bad fight or bled away two at a time for six minutes, and those call for opposite fixes.

The same shape of error produced a claim that stood in this wiki for a day: that 246 of 247 extractors the bot placed were destroyed. No source contained the figure. It was `275 − 28 = 247` **expansion orders** restated as placed-and-destroyed structures, and an order granted by the budget is not a structure that went up — the builder still has to walk there, and the engine refuses a placement silently ([[policy-holding-ground]]).

## Two tables, because they answer different questions

**Per sample** — frame, army, credits, enemies, extractors, losses since the last observation, producers, idle producers, orders issued, claims refused, total worth, the strongest rival's worth, our income and that same rival's income (the engine's own per-second figures, read off the scoreboard rows that ride every sample — the race law in one column pair, [[policy-economy]]), the world digest ([[policy-determinism]]), the opening plan's outcome (`building`/`done`/`blocked`/`stalled` — the column that located the sandbox-poisoned compile within one probe, log 2026-08-06), and the worker count (every recorded economy failure runs through it, and "when did the workforce die" is now a read, not an inference). The income pair landed 2026-08-05 between `rival` and `world`, and the plan and worker columns appended after the digest on 2026-08-06, so every column an existing reader indexes by position keeps its place across all four eras of the shape. This answers *when*.

**Per loss** — the unit, its type, and where it was standing when last seen. This answers *where*, which is what separates "dying on the walk home" from "dying at the enemy front".

A loss is **inferred, not reported**: the engine sends no death event, so a unit that was ours last sample and is absent now is counted as lost. That is not quite the same claim — a unit can also leave the roster by finishing a conversion, which is how an upgrading extractor appears — and the distinction is recorded rather than hidden behind the word.

## What it found

Averaged over twelve traced runs of one identical specification, the bot **leads the strongest rival for the first sixty per cent of every match** and then loses ninety per cent of its position. The producer count is 1.0 for the whole first half and never passes 1.7 while idle producers sit at zero: one factory, permanently saturated. Income compounds to about ten extractors, and at the crossover the credits stop becoming army — 2,299 at the halfway mark, then 6,058, 12,660, and 22,429 at the end.[^2]

The bot does not lose fights. It loses because it cannot spend what it earns.

## It is also where the usable scores live

Coefficient of variation across twelve identical runs:

| score | CV |
|---|---|
| mean share of worth against the strongest rival | **0.066** |
| peak army | 0.094 |
| samples seen | 0.098 |
| peak extractors | 0.116 |
| peak worth | 0.198 |
| **final worth** (what the scorecard reports) | **0.670** |

Ten times the noise, for the figure that was being compared. Experiments are scored off the trace now ([[policy-determinism]]).

## Every match records one

Sweeps used to pass `-` for the trace path and keep only the scorecard, on the reasoning that a run not being compared against another has nothing to read a trace for. That was wrong in the way that matters: which run turns out to be worth understanding is not known until after it has been played, and re-running to recover the detail produces a different match. Every sweep match writes one now, named after its job.[^3]

[^1]: `runs/trace-12345.ndjson` against `runs/sweeps/upgrade-fixed/long-s12345.txt`.
[^2]: `runs/traces/r01..r12-s12345.ndjson`. Peak worth averages 67,650 and arrives 63% of the way through; final worth averages 7,237.
[^3]: `src/rw_bot/harness/sweep.py`, `trace_path` and `play_args`.
