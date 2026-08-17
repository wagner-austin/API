---
title: "The Doctrine Search: Screening With a Confirmatory Backstop"
tags: [harness, methodology, search, margin, design]
related:
  - "[[campaign-ledger]]"
  - "[[harness-match-service]]"
  - "[[policy-doctrine]]"
  - "[[policy-determinism]]"
source_paths:
  - "src/rw_bot/harness/margin.py"
  - "src/rw_bot/harness/search.py"
  - "scripts/margin.py"
  - "scripts/search.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [headless-harness]
---

# The Doctrine Search

Eleven-plus rejected arms at Very Hard said the single-knob vocabulary
was mined out, and law two says knob combinations are their own
measurements -- a space no evening-per-arm ladder can afford. The search
harness industrializes the ladder: it proposes doctrine variants,
screens them cheaply, and hands a ranked shortlist to the same bar every
hand-picked arm has always faced. This page records what the method is,
what it is NOT, and where it deliberately cuts corners, because the
question "is this proper?" deserves a standing answer rather than a chat
log (2026-08-11).

## The margin: one bounded number per match

A verdict is one bit, and 48 pairs of one bit is a blunt instrument.
The margin (`rw_bot/harness/margin.py`) reads three things every card
already carries:

    margin = verdict anchor + pressure + tempo

with won +2 / survived +1 / defeated -1 / wiped -2 as the anchors,
pressure the destroyed fraction of the best rival's peak worth (law
seven's own figure), and tempo rewarding fast wins and long losses. The
bands cannot cross, so ranking by margin never disagrees with ranking
by verdict; margin only separates matches the verdict calls equal.

Validation was retrospective against panels whose answers we already
paid for: navy96f reads -1.21 (about 2.6 standard errors below zero),
agreeing with its -8 rejection, and all four naval-tilt panels read
within noise of zero -- the "breaks even, trades wins for survivals"
conclusion that originally took four panels to establish, recovered
from each panel alone.

## The search: successive halving, one shared control

`scripts/search.py` runs rounds. Each round writes the surviving
candidates' doctrine files, submits ONE batch in which every candidate
arm is paired against a single shared control on fresh seeds (pairs
interleaved so the paired read fills in seed by seed), waits for the
fleet, scores every arm by paired margin delta, and keeps the top half
for a bigger round. Everything is deterministic from one rng seed:
candidate sampling, round seeds, tie-breaks. Successive halving is the
standard fixed-budget best-arm method from the bandit literature; the
shared-control pairing is the same variance reduction every panel here
has always used.

## What the search is NOT

The search adopts nothing. Its output is a graduation ORDER, and the
graduate faces the unchanged discipline: a full paired panel judged on
wins against the +4 bar (law six), then fresh-tree replication (law
nine). Selection happens on one dataset, confirmation on an independent
one. The search proposes; the bar disposes.

## The corners it cuts, named

- **Winner's curse.** The survivor of sixteen candidates was selected
  for a high noisy estimate, so its search-phase delta is biased upward
  by construction. Search numbers are never evidence for adoption;
  only the graduation panel's are.
- **Silent false negatives.** Round zero fields eight pairs per
  candidate -- deliberately underpowered as a verdict, adequate as a
  sieve. A genuinely good candidate can die to seed luck and nobody
  will know. Re-running with a different rng seed re-rolls the sieve
  and samples different pairs, but the loss is real and accepted as
  the price of affordability.
- **The margin is a proxy.** It is validated descriptively, not
  causally; a knob that dignifies losses without converting them would
  waste search budget. The failure mode is efficiency, never ledger
  corruption, because graduation judges wins.

## Operational lessons already encoded

The first search (vhsearch1) taught three within a day, each now in
the driver: a round nobody claims is named loudly after three polls
(the fleet drains-and-exits when the queue empties before submission);
database outages are outlasted with a named retry rather than dying
(Docker's fourth crash killed the first driver mid-poll); and the
report streams as it happens, because a fifteen-hour silence is
unreadable while it matters. Resubmission is idempotent, so a killed
driver relaunches and resumes its round without loss.
