---
title: "Campaign Ledger — The Standing Scoreboard"
tags: [policy, methodology, experiments, ledger]
related:
  - "[[policy-doctrine]]"
  - "[[policy-determinism]]"
  - "[[harness-match-service]]"
  - "[[policy-exact-timing]]"
source_paths:
  - "doctrines/flame-nocover.doctrine"
  - "runs/sweeps/"
source_git_blobs:
  "doctrines/flame-nocover.doctrine": "aa9e6519a791ceb948a7578a0c24d4eb089b4c02"
provenance:
  - "wiki/log.md — the chronological record this page summarises. Deliberately NOT a pinned source_path: it is append-only, so a pin on it goes stale every time any session logs anything, including edits to this page. That is drift with no information in it, and it was firing here. The page's own framing says the same thing — log.md is the journal, this page is the current position."
game_version: "1.15 (code 176, build #28)"
confidence: high
fact_checked: 2026-09-05
hubs: [bot-architecture]
---

# Campaign Ledger — The Standing Scoreboard

The at-a-glance state of the campaign: what the bot currently is, what has
been adopted and when, what is open, what is closed, and the laws every
future arm inherits. `log.md` is the chronological record; this page is the
current position. **Update this page with every adoption, closure, and law
— it goes stale the day a verdict lands without touching it.**

The goal, unchanged since 2026-08-05: 100% vs Impossible watchably, then
any AI, any human, any 1vX seating, any map.

## The champion, per rung

| Rung | Configuration | Record | Status |
|---|---|---|---|
| Very Easy (duel_lake) | `evolve1-g4m2` | 12/12 won, mean 1,736 samples (ladve12, fresh seeds, 2026-09-05) | First certification read at n=12 -- the rung had never been measured with the modern policy. Clean sweep. |
| Easy (duel_lake) | `evolve1-g4m2` | 12/12 won, mean 2,076 samples (lade12, 2026-09-05) | First certification read at n=12. Clean sweep. |
| Medium (duel_lake) | `evolve1-g4m2` | 12/12 won, mean 2,164 samples (ladm12, 2026-09-05) | First certification read at n=12. Clean sweep. |
| Hard (duel_lake, modern policy) | `evolve1-g4m2` | 11/12 won + 1 survived-at-cap, 0 losses (ladh12, 2026-09-05) | First modern read of the rung. The non-win is a true stalemate (alive at 10,000 samples, 13,700 army vs a rival compounded to 134,550): the 1.4x subsidy already shows the shape that owns Impossible, one rung above where wins stop being automatic. |
| Hard, Linux+pinned+ff10 (duel_lake) | `aa-counter-guard` | v8 detection, held to v7 across 120 paired seeds (ab24 + ab48 + ab48b, 2026-09-01): v7 102/120 won (85%) vs v8 90/120 (75%); 0 armyless, 0 losses, 0 stalls in all 240 matches | The pace question is CLOSED as noise: discordant pairs 25:13, two-sided p = 0.073, and the per-wave flips DILUTED (7:2, 9:4, 9:7) -- p rose with more data, which a real effect does not do. Trace-pinned mechanism of divergence: arms are bit-identical until the first refusal-recovery moment (the watch's ~2-sample report replacing the 45-sample clock), after which matches are different matches. Labeled hypothesis, untested: the early asymmetry tracked the contended-boot seam, which wave three (node-local clones, 96/96 zero failures) removed. v8 keeps the rung; detection is defect-critical. Windows-path figures do not compare. |
| Very Hard (duel_lake) | `evolve1-g4m2` | 72/96 paired fresh seeds (75%) vs `close0-flame4`'s 45/96 on the same seeds | ADOPTED 2026-09-04 -- **the first machine-LEARNED champion**: the CEM population search's graduate (c_artillery slot -> heavyTank, every knob inherited), replicated to the byte by an independent rng (evolve2's g3m10). Laws six and nine vs close0-flame4: **+12 then +15** against the +4 bar (34-22 then 38-23), combined flips 40:13, **p=0.00027** -- the strongest adoption in the ladder's history (prior best p=0.005), at 71%/79% panel win rates where 50% was the record. Prior champion close0-flame4 (adopted 2026-09-02, +14/+5, p=0.005; confirmed a knob-space local optimum by vhsearch4 + flame2's rejection) stands as the knob-vocabulary ceiling the simplex broke through. |
| Impossible (duel_lake) | `flame-nocover` | 0/~120 post-certification, survival floor ~3300; 0/48 vs `flame-close6` (imp48c6) AND 0/48 vs `close0-flame4` (imp48c0f4), both paired | No adoptable gain, and composition is now measured OFF the table: close6 washed (+0.051) and the full VH champion close0-flame4 washed too (-0.082, sd 0.784) -- one was a closing behavior that never acted, the other shapes the army from the first engagement, and neither moved. What wins at VH is orthogonal to what Impossible punishes -- confirmed a THIRD time by the machine-learned champion itself: g4m2imp48 read 0/48 both sides, paired margin +0.210 (sd 0.607), a small survival lean and no road to a win. The next gain requires a new capability class; rebuild-under-fire exists as code since 2026-09-04 and is the next measurement ([[impossible-economy-problem]]). Champion held by default of evidence. |
| Other maps (Impossible) | `flame-nocover` | bridge 2811 / straits 2317 / hills 2281 vs duel 3325, 0/36 | Terrain-as-parameter closed: every alternative worse, duel_lake retained BECAUSE most favorable (log 2026-08-09). |

## Adoption history (post-certification)

Every behavior the bot actually runs was once an arm that cleared a paired
panel. Nothing on this list is a guess.

| Date | Change | Evidence |
|---|---|---|
| pre-2026-08-06 | The champion stack itself (cover-off, flame 2, close 3, aa-counter guard, raid 3, tech 1, counter tilt) | The VH arm ladder, pre-certification; carried forward as the baseline the certified harness re-measured at 12/24 and 21/48 |
| 2026-08-08 | — (no adoption yet on the certified harness) | raid 8 is the first candidate; adoption awaits imp-raid8x48 |
| 2026-09-02 | `flame-close6` takes the Very Hard rung (`close 3` -> `close 6`, all else unchanged) | vhsearch2 graduation (margin +1.778 then +0.753 across two search rounds) confirmed by TWO independent 48-pair win-bar panels: close6vh +8 and close6rep +7 against the +4 bar, disjoint seeds, independently frozen trees, 0 job failures in 192 matches; combined flips 29:14, p = 0.031 (logs 2026-09-02) |
| 2026-09-02 | `close0-flame4` takes the Very Hard rung hours later (flame 2 -> 4, close 6 -> 0) | vhsearch3 graduation (the only arm whose margin GREW with depth, +1.150 -> +1.401) confirmed by c0f4vh **+14** (28-14 of 48, flips 16:2, p=0.001 -- the widest VH win delta ever) and c0f4rep **+5** (20-15 of 48), disjoint seeds, independent trees; combined 48-29 of 96, flips 31:12, p=0.005; panels rode out 60 preemption casualties via campaign converges (logs 2026-09-02) |
| 2026-09-04 | `evolve1-g4m2` takes the Very Hard rung -- the first machine-LEARNED champion (c_artillery slot -> heavyTank in goals, every knob inherited) | CEM population search graduate (best-of-generation +1.048 -> +4.197 over five generations; generation 4 put ALL 16 members over the champion), replicated to the byte by evolve2's independent rng; confirmed by e1g4m2vh48 **+12** (34-22 of 48, flips 22:10, p=0.050) and e1g4m2rep **+15** (38-23 of 48, flips 18:3, p=0.001), disjoint untouched blocks; combined 72-45 of 96, flips 40:13, **p=0.00027** -- the strongest adoption ever recorded here (logs 2026-09-04) |

## Open questions (measurement running or queued)

- The docket after the battery: the spatial layer (its first slice shipped 2026-08-15 -- coverage recorded as three trace columns, accumulating on every future match; next steps are the doom refit with spatial features once a corpus exists, and hold v2 designed against the field), a response worth driving with the banked doom model, and the exporter productionization. Composition tuning, the naval theater, the tilt, and the battery are all measured closed; the next gain must come from a capability class the bot does not yet have.
- The Impossible frontier after 2026-09-04: every NAMED road is closed on mechanism-verified measurements in one span -- denial (impden48), rebuild-as-designed (imprb48), strike-window with the release provably firing (impstrike48), and composition three times over. Single-verb answers are exhausted; the standing next step is the original three-step plan's step three, the corpus-trained heads -- learned policy above the doctrine vocabulary -- with the spatial/doom corpus and the `rival_army` trace column as its instruments ([[impossible-economy-problem]], [[harness-population-search]]).

## Closed questions (measured, with the log entry that closed them)

**Very Hard:**
- The artillery battery ([[policy-battery]]): REJECTED at -12 (battery96, 48 pairs, 6 wins vs control's 18; arm wiped 25 to 15) with the mechanism proven in 47 of 48 arm matches -- the first new-verb panel to measure the real thing on its first attempt. The standoff counter works and still loses: $2,100 plus a builder held through construction is paid out of tempo, and no trigger quality can rescue a response that drags every game it enters (law ten, third derivation). Capability banked: the channel, the quartermaster seam, five defect fixes lifted to every twin (log 2026-08-15).
- The six-knob composition space (flame, close, raid, tech, medics, decoys): read as FLAT around the champion after vhsearch1 (grad1 +4 to the digit, grad2 22-22 dead even), and composition tuning was retired 2026-08-14 -- a verdict vhsearch2 OVERTURNED on 2026-09-02: the cluster-played search found `close 6`, and it survived both graduation laws to take the rung. The 08-14 closure was a statement about that search's reach (local fleet, smaller panels), not about the space; the space holds at least one adoptable knob the earlier reach missed.
- The naval theater itself: submarines behind a guarded sea factory, REJECTED at -8 against the +4 bar (navy96f, 48 pairs, control 21 wins vs navy 13) with the mechanism proven in all 48 arm matches -- 6 genuine rescues against 14 thrown-away control wins, and the arm wiped 24 times to control's 15. The stat-sheet hard counter is real and insufficient: tempo, not the fleet, decides these games, and the ~3-5k diversion loses the land war faster than the submarine wins the water. Capability banked (walk, pinned builder, guard, headcount -- regression-locked, reusable on any water map); economics closed at Very Hard on duel_lake (log 2026-08-11).
- The naval tilt response, under EVERY driver -- ungated -2, deficit -2, blood -2, learned-oracle -5: even correctly targeted arming breaks even, so no trigger quality can rescue it. Closed with prejudice (log 2026-08-10, navdoom96).
- The blood-gated tilt calibration (log 2026-08-08, navblood96).
- Composition surgery, all channels: eight one-knob arms negative; production is a ratio simplex (log 2026-08-07, the arm ladder).
- The unconditional naval tilt: net -2 at 24 and 48 seeds, trades wins for survivals (log 2026-08-08, navpair48).
- The deficit-gated tilt: halves the damage, keeps the disease — army deficit is the normal shape of winning against a subsidized opponent (log 2026-08-08, navgate96).

**Impossible:**
- Terrain as a parameter: all three alternative maps worse than duel_lake, each by a named mechanism -- unmanned chokes, amplified navy, fog-off feeding the tilt (log 2026-08-09).
- Cadence: the whole bot at lockstep 25 reads mildly negative-to-noise; the decisions, not their frequency, are the ceiling (log 2026-08-09).
- The choke-holding verb v1: -264 on its own bridge panel, mechanism photographed off death positions -- the post trickles, and a gather point is not a mass. Verb retained behind hold 0; v2 designs logged, not built (log 2026-08-09).
- Trades (champion): 0/11, overrun ~s3200 with economy intact (log 2026-08-07).
- Masonry, both halves: static fortress 0/12 dies faster; choke-walk creep 0/12 dies fastest (logs 2026-08-07/08).
- The income ladder: funds its T3s and loses anyway — a race is not a stalemate (log 2026-08-08).
- The scout: -1008 mean survival, the worst paired arm ever; ungated input is perturbation (log 2026-08-08).
- Riposte (-562), medics (-418), flee30 (-110), strike5 (-104): four hands closed by the screen (log 2026-08-08).
- raid+kite composed (-204) and raid 12 (-214) (log 2026-08-08); then raid 8 itself retracted on fresh-tree reversal, +217 to -198 on identical seeds (log 2026-08-09).

## The laws (what every future arm inherits)

1. **The ratio simplex** — any permanent reshaping of a razor-tuned mix costs more than it pays; there is no such thing as pure addition (2026-08-07).
2. **One knob, behavioral edition** — hands do not compose freely either; every combination is its own measurement (2026-08-08, combo36).
3. **Adapt when losing, never touch what is winning** — and the gate must read the failure mode itself, not a proxy (2026-08-08, three tilt calibrations).
4. **Latch gates saturate** — naval contact is nearly universal, naval doom is rare; a gate that can only turn on eventually fires in most wins (2026-08-08, navblood96).
5. **Input without a gated response is perturbation** — richer intel fed to unconditional rules made the bot blinder, twice (2026-07-29 scouting v1; 2026-08-08 imp-scout24).
6. **Conditional arms need paired panels at scale** — effects concentrate in seed subsets; 24 seeds measure re-roll noise (2026-08-07).
7. **Erosion law** — verdicts track whether the economy was HELD; endpoints lie, peaks tell (2026-08-07, the xmap correction).
8. **Blast radius matches trigger precision** — small reversible responses may gate on present signals; match-reshaping responses require prediction, which no present-tense scalar supplies (2026-08-08, the three-calibration arc).
9. **Effects must replicate across trees** — a paired panel controls within-tree noise, not tree-to-tree variance; the same seeds flipped +217 to -198 between trees. Fresh-tree replication is the adoption bar (2026-08-09, imp-raid8x48).
10. **Measure the response before the trigger** — a response that breaks even when correctly armed is a dead end at any trigger precision; the tilt netted -2 on its own home subset under a replicated 0.75-AUC oracle (2026-08-10, navdoom96).
11. **Mechanism before measurement** — no panel runs until a pilot match's card proves the arm's mechanism EXECUTED (the `owned peak` census line: what stood, ever, at its peak). Five navy panels ran to verdict while the thing under test never existed: the budget proved payment, nothing proved existence, and a misread of truncated walks as acceptance cost two further panels. Ship the probe's own mechanics whole — every place a channel "improves" on a proven probe is where a panel dies (2026-08-10, navy96 through navy96e).

## The ML layer

First positive result 2026-08-09: **fleet doom is predictable** -- AUC 0.80,
precision 0.82 at sample 2000 (a deployable in-match moment, 600+ samples of
lead time), on the 96-match sighted corpus with seed-grouped folds all above
0.70. The first re-ask at sample 800 read chance because the fleet had not
arrived yet -- the sighted columns diagnosed their own window. Fresh-tree
replication PASSED 2026-08-09: train-A-test-B AUC 0.751, the first signal
to survive law nine. Deployed as the tilt's mode-3 driver and REJECTED at
net -5 (2026-08-10) -- the response, not the trigger, was the flaw (law
ten). The model, the mode-3 wiring and the watch/latch machinery remain
banked for any future response worth driving.

## The instrument

The match service ([[harness-match-service]]): Postgres queue + leased
clones + leased ports + HTTP door + results mirror + retry verb. ~380
matches served across its first two days, through two Docker outages, a
worker crash, and three of its own concurrency bugs — each found by
production load, fixed same-day, and regression-locked. Panels that were
an evening of babysitting are now one submission.

Grown since (2026-08-10/11): the live dashboard at ``GET /`` (lanes,
per-arm verdict tallies, whole batch history), label-scoped priority
(a paired panel's arm jumps its own controls), interleaved pair job
files (the paired read fills in seed by seed), the detached door
(``make door``) and one-command fleet recovery (``make fleet-up``),
AboveNormal match launches with a 120s channel so co-tenants cannot
starve the sample stream, the ``owned peak`` mechanism census on every
card (law eleven's instrument), the dense margin
(``scripts/margin.py``), and the search driver (``scripts/search.py``).
Two more Docker outages and one eight-hour overnight stall were
recovered without losing a row.
