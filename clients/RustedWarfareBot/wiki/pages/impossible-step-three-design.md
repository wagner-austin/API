---
title: Step three at Impossible -- what a learned head may drive, and what it may not
tags: [impossible, learning, design]
hubs: [bot-architecture]
related: ["[[impossible-economy-problem]]", "[[harness-population-search]]", "[[policy-exact-timing]]"]
source_paths:
  - "src/rw_bot/policy/doom.py"
  - "models/fleetdoom.ndjson"
source_git_blobs:
  "src/rw_bot/policy/doom.py": 69fbc15602e4f48a97c4e82fdbc4eaac4fe6866c
  "models/fleetdoom.ndjson": 7083f4c071b6a9372a4e65916fd9430ff1deeb58
provenance:
  - "runs/sweeps/impstrike48 — sweep run directory, untracked by git"
  - "runs/sweeps/impincome96 — sweep run directory, untracked by git"
  - "wiki/log.md entry dated 2026-09-04 — kept as provenance rather than a source_path: log.md is append-only, so a blob pin on it drifts on every wiki operation while the cited entry itself cannot change"
  - "wiki/log.md impincome96 verdict entry dated 2026-09-06"
fact_checked: 2026-09-06
confidence: high
---

# Step three at Impossible -- what a learned head may drive, and what it may not

The three-step plan's first two steps are done: the policy
parameterization existed (the doctrine vocabulary), and the population
search scaled it -- to a Very Hard champion adopted at p=0.00027. Step
three is "the corpus-trained heads". This page pins what today's
evidence says a head at Impossible can and cannot be, before anyone
builds one.

## The constraint the navdoom arc already proved

A head is a TRIGGER. The naval-doom arc measured, at every trigger
quality up to a learned oracle (AUC 0.75), that **arming a dead response
breaks even or loses regardless of how well it is timed** -- "even
correctly targeted arming breaks even, so no trigger quality can rescue
it" (navdoom96, closed with prejudice). And 2026-09-04 closed every
single-verb response at Impossible with the mechanism proven firing:
denial, rebuild-as-designed, and the in-band strike release
([[impossible-economy-problem]]). A doom-style head grafted onto
today's verb vocabulary at Impossible would be a well-trained trigger
wired to responses that are all measured dead. That design is refuted
before it is built.

## What remains learnable

The gap is the RESPONSE surface, so step three at Impossible means
learned decisions at a granularity the doctrine vocabulary cannot
express:

1. **Per-sample tactical policy** -- which units engage, where the army
   stands, when it trades -- replacing hand rules inside the loop
   rather than gating them. The largest change: a policy head on the
   wire loop, trained from the trace corpus (every cluster match
   already writes one, now including `rival_army`), with train/serve
   parity by the doom template (the deployed watch computes features
   through the same class the exporter fits).
2. **Sequencing above the opening** -- the goals line fixes the opening
   and the ratio; nothing learns WHAT TO BUY NEXT as the match
   unfolds. A learned build-priority head drives a response class no
   knob expresses (the income_ladder/withhold family all failed as
   STATIC rules; a contextual one is unmeasured). Measured at this
   rung 2026-09-06: the static ladder fired in 48/48 impincome96
   matches and read paired survival -201.0 (sd 878.5) -- the
   conversion's ~500-second payback does not fit a ~790-second median
   match, so a STATIC rule pays for income the match never collects.
   Exactly the trade a contextual head could refuse to make, and the
   one no static knob can.
3. **The doom template generalized to VH-and-below**, where responses
   are alive: the ledger's open "a response worth driving with the
   banked doom model" is still the cheapest head-shaped win, just not
   an Impossible one.

## What this costs, honestly

Options 1-2 are architecture, not arms: new wire-visible decision
points, a training exporter, a model registry beside
`models/fleetdoom.ndjson`, and panels for every deployment. That is a
multi-session lane touching the loop's core, against a rung where the
rival's advantage is a 5x economy subsidy that no measured lever has
dented. The honest framing for the operator: step three at Impossible
is a bet that a structurally better-played match can beat a 5x
subsidy that outran every structural verb tried so far -- possible,
unproven, and expensive; the same machinery pointed at VH-and-below
rides on live response surfaces and proven instruments.
