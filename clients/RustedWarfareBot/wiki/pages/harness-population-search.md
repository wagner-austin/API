---
title: Population Search — The Parameterization Design
tags: [harness, search, design]
related: [[harness-doctrine-search]], [[policy-loop]], [[campaign-ledger]]
sources: [src/rw_bot/policy/production.py, src/rw_bot/policy/doctrine.py, src/rw_bot/policy/situation.py, wiki/log.md 2026-08-07 "the arm ladder"]
fact_checked: 2026-09-03
confidence: high
---

# Population Search — The Parameterization Design

The answer to "can an AI learn the game by playing 10,000 matches": not
by deep RL from scratch (wrong sample scale, wrong compute shape), but
yes by black-box optimization over a real parameter vector -- and the
harness this project already runs is the expensive half of that system.
What is missing is only the vector and the outer loop. This page is the
design; nothing here is built until the ledger says so.

## What the survey found (2026-09-03)

Three layers of numeric decision surface, in leverage order:

1. **The composition simplex.** `doctrine.goals` is an ordered unit list
   whose "repeats are a ratio, not a count" (doctrine.py L123) and
   production holds the army to that mix forever (production.py). The
   2026-08-07 arm ladder measured this exact fact -- "production is a
   ratio simplex" -- and today's champion still fights with a HAND-NAMED
   point on it. Continuous weights over the ground vocabulary
   (c_tank, hoverTank, heavyTank, c_artillery, flame, plus the unlocked
   tiers) are the highest-leverage dimensions the knob search cannot
   reach: the int machinery moves counts of special structures, never
   the mix itself. ~10-15 dims.
2. **The existing integer knobs.** The 22 `INT_FIELDS` the spec search
   already moves; they join the vector as-is. ~20 dims.
3. **The frozen tactical constants.** ~25 module constants each measured
   once by hand and never revisited: `CLOSE_HOLD` 25, `MOMENTUM_WINDOW`
   40, `FIRST_WAVE` 3, `RALLY_RADIUS` 60, `MAX_OPEN_GROUPS` 2, the
   battery/navy `PATIENCE` 40s, four lurk radii, the nuker's retry
   windows and `FUNDING_INCOME_FLOOR` 50, rush geometry, siting radii,
   the workforce retry clocks. Each is a dimension the moment it is
   lifted to a doctrine field. `economy_floor` is already map-derived
   (expander.py) and stays a function, not a gene.

## The design in one paragraph

A **genome** (~30-45 floats) compiles to an ordinary doctrine file --
composition weights round to a small-integer ratio in `goals`, knob
genes clamp to their codec ranges -- so the ENTIRE existing evaluation
chain (payload freeze, campaign arrays, node-local clones, scorecards,
paired margin) runs unchanged. The outer loop is CMA-ES or
population-based halving, shaped exactly like `run_search`: one
generation = one interleaved batch of candidates against the sitting
champion as shared control on a fresh seed block, fitness = paired
margin delta, ~15-30 members per generation, tens of generations. At 96
matches a batch that is thousands of matches per run -- the 10,000-match
budget, spent where the variance discipline already lives. Adoption is
UNCHANGED: a converged genome's doctrine graduates to laws six and nine
like any other arm; the learner proposes, the bar disposes.

## Phasing

- **v1, no policy-code changes:** genome = composition simplex + the 22
  int knobs, compiled to doctrine text. Only new code: the compiler
  (pure, tested) and the generation loop beside `run_search`.
- **v2:** lift Layer-3 constants to doctrine fields one at a time (each
  is a small, guarded change with its own regression test), widening the
  vector as they land.
- **v3, the GPU path:** corpus-trained heads (the spatial trace columns
  accumulate on every match already; the doom model proved the shape)
  trained on `free-gpu` with the checkpoint/campaign discipline the
  Turkic LSTM project forged there. Heads are policy code, not genes;
  they re-enter through ordinary adoption.

## What this is not

Not deep RL: no gradient through the game, no value network required to
start, no GPU on the evaluation path. Not a new evaluation stack: the
compiler emits doctrine files precisely so nothing downstream changes.
Not an adoption shortcut: search-phase fitness carries winner's curse by
construction, and only untouched-seed panels adopt.

## Open questions the first run must answer

- Ratio rounding: the smallest integer ratio faithful to a weight vector
  (largest-remainder over a fixed total is the candidate).
- Fitness at Impossible: ANSWERED NEGATIVELY (2026-09-03). impsearch1's
  graduate read +0.27-0.32 across two search rounds and ZERO on 48
  untouched seeds (imps5k48) -- round-to-round consistency inside one
  selection process is not independent evidence, and margin fitness at
  8-16 pairs per candidate selects noise there. A population at
  Impossible needs far larger generations, a composite fitness
  (survival depth, economy trajectory), or the corpus-trained heads
  before it can climb. Very Hard's steeper landscape (deltas 1-2 at
  similar sd) remains the right first target for v1.
- Generation seed budget: fresh blocks per generation burn ~2,500-seed
  panels fast; the seed namespace math holds to ~38 generations below
  the search floor and needs widening past that.
