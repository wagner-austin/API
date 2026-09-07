---
title: The Build-Priority Head -- Step Three's First Concrete Design
tags: [impossible, learning, design, economy]
related: ["[[impossible-step-three-design]]", "[[policy-trace]]", "[[policy-determinism]]", "[[policy-budget]]"]
source_paths:
  - "src/rw_bot/policy/campaign.py"
  - "src/rw_bot/policy/trace.py"
  - "src/rw_bot/policy/doom.py"
source_git_blobs:
  "src/rw_bot/policy/campaign.py": "d32c85432385e1dc226c4a8722b5543f54bea7c2"
  "src/rw_bot/policy/trace.py": "1494fc9ecdde3c89c833f0046faef92b0ac817d7"
  "src/rw_bot/policy/doom.py": "69fbc15602e4f48a97c4e82fdbc4eaac4fe6866c"
provenance:
  - "/pub/wagnera3/rusted/runs/traces -- 7,830 per-sample trace files, 931MB, one per cluster match through impincome96 (ls | wc -l and du -sh, read 2026-09-06)"
  - "wiki/log.md verdict entries 2026-09-06 (impincome96) and 2026-09-06 (detpair24)"
  - "wiki/log.md entries 2026-09-07 (detpair24b floor re-pricing; corpus label measurement over rw_matches/data.csv, 6,643,204 rows / 1,980 matches, exporter commit aaa8f780)"
fact_checked: 2026-09-07
confidence: medium
hubs: [bot-architecture]
---

# The Build-Priority Head

[[impossible-step-three-design]] names two architecture options; this page
is option 2 made concrete enough to build, written the day its motivating
measurement landed. A design page in the roadmap style: decisions first,
open questions stated as open.

## The decision, and where it lives

Each tick the campaign's budget walk funds a fixed priority order
([[policy-budget]]). The head replaces none of the walk's *verbs* -- it
reorders and withholds: given the tick's context, emit a priority class
(army-now / income / tech / save-toward-named-target / expand). The
decision point is the walk's entry in `campaign.py`, the same chokepoint
every spend already crosses, so one wire-visible decision covers every
purchase without new order types.

## Why a contextual decision where static rules all failed

Every static point in this family is measured dead at Impossible: the
bank read -605.9 paired survival with its safe-window gate working as
built, and the income ladder read -201.0 with its mechanism firing in
48 of 48 matches -- a T3 conversion's ~500-second payback cannot fit a
~790-second median match (log, 2026-09-06). The failures share one
shape: a rule that always saves pays for futures the match never
collects, and a rule that never saves can never buy the 56k chain. Only
a decision conditioned on the match's own state can hold both ends.
navdoom's law still binds: the head must drive LIVE responses -- it
reorders spending that already works tick to tick, rather than gating a
dead verb.

## Training material, and the constraint the floor puts on labels

Every cluster match writes a per-sample trace -- 7,830 files, 931MB,
through impincome96 -- whose 25 columns include `rival_army`, income,
worth, coverage, and the `events` letters ([[policy-trace]]). Train/serve
parity follows the doom template: the deployed watch computes features
through the same class the exporter fits (`doom.py` is the worked
example).

The label is the open problem, and [[policy-determinism]] prices it:
paired outcomes at this rung carry sd ~662 samples per pair (re-priced
2026-09-07 -- the routing weaves nearly halved the original 1,205), so
any label built from ONE pair's delta is mostly noise. Candidates:
(a) outcome regression -- survival against spend-mix-by-phase across
thousands of traces; (b) within-pair contrast on shared seeds, honest
only in bulk; (c) imitation is unavailable -- the corpus holds no
Impossible wins to imitate. Whichever is chosen, the deployment gate is
unchanged: laws six and nine, 48-pair panels, effect sized against the
measured floor (2-se MDE ~191 samples at n=48).

## The offline pass answered the label question (2026-09-07)

Candidate (a) in its within-doctrine form is measured EMPTY: over the
168 standing-base matches, early-window spend-mix features fit against
remaining survival generalize to nothing (leave-batch-out CV
R-squared -0.002), and the within-doctrine survival sd (664) equals the
identical-pair floor (662) -- under a byte-fixed doctrine there is no
predictable survival component for a per-tick head to learn from at
this corpus size. The trainable signal sits on the DOCTRINE axis
instead, strong and coherent under within-generation and within-pair
controls (log, 2026-09-07): order flow up, workers grown in phase 1
not phase 0, credits off the balance sheet, raids drafted, fights kept
inside our guns' reach. CORRECTED same-day (log, 2026-09-07, second
entry): the causal record already arbitrates most of these correlates.
The raid-draft correlate's causal twin is measured DEAD at every
strength (raid8 retracted under law nine; impden48 closed denial at
raid 3/6/8 alike), so `p3_evR` is prognostic, not a lever; the
spend-don't-hold correlates AGREE with the closed hoard arms (bank
-606, ladder -201, riposte -562) rather than adding to them. What the
corpus adds beyond the causal record is one unscreened knob -- worker
TIMING (growth in phase 1, not phase 0; the VH-era workers10 arm
tested count, never timing) -- and one live road: a head trained on
PROXIMATE EVENTS rather than survival, which the brace head's AUC
0.94 razing fit proves can beat this noise. The worker-timing arm was
screened same-day and CLOSED FLAT (wwait24: mechanism 12/12, paired
survival +282 at t=0.88, under the screen's own MDE -- log,
2026-09-07), making the observational-to-causal ledger three for
three. The proximate-event head is the lane's only remaining road,
and the spend goes there.

## What is deliberately not decided here

Model class (the head template's logistic form vs anything richer),
decision cadence (every tick vs on budget-refusal), and the exact
feature list. Each is an exporter-side experiment the corpus can answer
offline before anything touches the loop -- offline first, then the
wire, then the panel.
