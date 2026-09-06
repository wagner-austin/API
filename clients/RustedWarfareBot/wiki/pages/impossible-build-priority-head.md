---
title: The Build-Priority Head -- Step Three's First Concrete Design
tags: [impossible, learning, design, economy]
related: ["[[impossible-step-three-design]]", "[[policy-trace]]", "[[policy-determinism]]", "[[policy-budget]]"]
source_paths:
  - "src/rw_bot/policy/campaign.py"
  - "src/rw_bot/policy/trace.py"
  - "src/rw_bot/policy/doom.py"
source_git_blobs:
  "src/rw_bot/policy/campaign.py": "85a4a357c27a4fbc90a1d35fb39a18bfa380c96f"
  "src/rw_bot/policy/trace.py": "1494fc9ecdde3c89c833f0046faef92b0ac817d7"
  "src/rw_bot/policy/doom.py": "69fbc15602e4f48a97c4e82fdbc4eaac4fe6866c"
provenance:
  - "/pub/wagnera3/rusted/runs/traces -- 7,830 per-sample trace files, 931MB, one per cluster match through impincome96 (ls | wc -l and du -sh, read 2026-09-06)"
  - "wiki/log.md verdict entries 2026-09-06 (impincome96) and 2026-09-06 (detpair24)"
fact_checked: 2026-09-06
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
paired outcomes at this rung carry sd ~1,205 samples per pair, so any
label built from ONE pair's delta is mostly noise. Candidates, none yet
chosen: (a) outcome regression -- survival against spend-mix-by-phase
across thousands of traces, where aggregation buys back what the floor
takes; (b) within-pair contrast on shared seeds, honest only in bulk;
(c) imitation is unavailable -- the corpus holds no Impossible wins to
imitate. Whichever is chosen, the deployment gate is unchanged: laws six
and nine, 48-pair panels, effect sized against the measured floor
(2-se MDE ~350 samples at n=48).

## What is deliberately not decided here

Model class (the head template's logistic form vs anything richer),
decision cadence (every tick vs on budget-refusal), and the exact
feature list. Each is an exporter-side experiment the corpus can answer
offline before anything touches the loop -- offline first, then the
wire, then the panel.
