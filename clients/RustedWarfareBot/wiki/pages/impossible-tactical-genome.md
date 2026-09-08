---
title: The Tactical Genome -- Stage Three as Search over the Reflex Layer
tags: [impossible, learning, design, tactics]
related: ["[[impossible-step-three-design]]", "[[impossible-build-priority-head]]", "[[policy-determinism]]", "[[harness-population-search]]", "[[engine-ai-triggers]]"]
source_paths:
  - "src/rw_bot/policy/combat.py"
  - "src/rw_bot/policy/dispatching.py"
  - "scripts/evolve.py"
source_git_blobs:
  "src/rw_bot/policy/combat.py": "8afff52d08a953cafe435d3543c337652cab3f47"
  "src/rw_bot/policy/dispatching.py": "bc9178f263566abf5bdb21cfc85b051be0ca094c"
  "scripts/evolve.py": "b33dee0d23201d670b9974e985d954e1d7bc54f6"
provenance:
  - "wiki/log.md entries 2026-09-07: the corpus label measurement and its correction; the eighth closure (wwait24); the engagement-gate prevalence refutation"
  - "wiki/log.md 2026-07-31 micro arc (kill-sized fire groups built, capped at two; artillery-standoff refuted as engine-native)"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java as() -- the uniform-random target chooser, read 2026-09-07"
fact_checked: 2026-09-07
confidence: medium
hubs: [bot-architecture]
---

# The Tactical Genome -- Stage Three as Search over the Reflex Layer

[[impossible-step-three-design]] names per-sample tactical policy as the
expensive end of stage three. This page pins HOW that policy is
parameterized and trained, after 2026-09-07 closed every cheaper
framing: survival-regression labels are all floor (leave-batch-out CV
R-squared -0.002; within-doctrine sd equals the identical-pair sd),
event labels die on prevalence (zero deep trade windows in 1,980
matches), and the last observational correlate closed flat at screen
tier with its mechanism firing 12/12.

## The decision, stated once

Stage three does NOT train a supervised head. Every label the corpus
can offer was priced and none survives the noise floor. What survives
is SEARCH: fitness needs no label, only a metric and repetition -- and
the population machinery already produced the standing base this way
([[harness-population-search]]; evolve3-g3m10, adopted at panel tier).
The bet is re-aimed, not re-founded: the same searcher, pointed at a
parameter space it has never seen -- the reflex layer's hardwired
constants -- with fitness evaluations sized against the re-priced floor
([[policy-determinism]]: 2-se MDE ~382 at n=12, ~191 at n=48).

## The genome: constants that become alleles

The loop's tactical behavior is fixed today by constants chosen once,
by hand, each at the resolution of a single measurement or none. The
build's first phase inventories and exposes them as doctrine-carried
tactical fields, each with the trivial identity default so the standing
base is byte-reproduced at the zero vector. Candidate alleles, from the
code and the closures:

* fire-group cap (`MAX_OPEN_GROUPS`, hand-set to 2 in the 2026-07-31
  micro arc on three seeds -- "pending a wins-based reason");
* target-class priority inside kill groups (artillery-first vs
  nearest-first; unmeasured -- enemy artillery is the top killer in
  Impossible death ledgers);
* engagement spacing (splash dilution against the artillery that tops
  those ledgers; no spacing rule exists anywhere in the loop);
* wave commit thresholds by context (the ladder's rungs, today global
  constants);
* retreat-regroup margin (distinct from the dead per-unit flee reflex:
  a GROUP that disengages at a measured local disadvantage);
* lottery posture (unit dispersion as target-distribution shaping --
  the chooser is uniform-random over our units, read from source, so
  our placement IS the enemy's attack distribution; wwdec36 prices the
  scatter end of this axis as a screen while this page is written).

The inventory is phase one's deliverable and this list is its seed,
not its bound. Every allele must pass the existing discipline: wired
through one chokepoint, mechanism observable in the trace or scorecard,
identity default certified byte-identical to the base before any search
runs (the det-twin shape).

## Fitness, and what the floor permits

Fitness is paired survival against the in-batch base on shared seeds --
the campaign's own metric, no proxy. The floor prices the schedule:
n=12 pairs per candidate resolves ~382 samples at 2-se, so a
17-candidate generation costs ~408 matches; the free partition played
24 matches in ~35 wall minutes tonight, so a generation is roughly a
day of cluster churn at zero SU. Selection tolerates per-candidate
noise by repetition (evolve3 produced the adopted base from n=8
fitness that detpair24 later priced as mostly noise); the halved floor
buys either half the noise or half the matches. Adoption is unchanged
and non-negotiable: laws six and nine, a 48-pair panel, fresh-seed
replication.

## What this is not

Not imitation (the corpus holds no Impossible wins), not regression
(all floor), not event-gating (no windows). Not a rewrite of the loop:
every allele parameterizes machinery that already exists and already
survives `make check`'s bar. And not a guarantee: the honest framing of
[[impossible-step-three-design]] stands -- this is a bet that a
structurally better-played match can beat a 3.7x subsidy that has
outrun eight closures. The difference after 2026-09-07 is only that
every cheaper bet is now measured dead, and this one runs on proven
machinery with the cheapest fitness evaluations the campaign has ever
had.
