---
title: "Campaign Ledger — The Standing Scoreboard"
tags: [policy, methodology, experiments, ledger]
related:
  - "[[policy-doctrine]]"
  - "[[policy-determinism]]"
  - "[[harness-match-service]]"
  - "[[policy-exact-timing]]"
source_paths:
  - "wiki/log.md"
  - "doctrines/flame-nocover.doctrine"
  - "runs/sweeps/"
source_git_blobs:
  "wiki/log.md": "f81d58fbcc4fdb9e68062c7fdaad9889cb8ecccf"
  "doctrines/flame-nocover.doctrine": "f2fef915923ca031d05cda11f83029d0e89e195e"
game_version: "1.15 (code 176, build #28)"
confidence: high
fact_checked: 2026-09-01
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
| Hard, Linux+pinned+ff10 (duel_lake) | `aa-counter-guard` | v8 detection: 15 won + 9 survived-with-army of 24, 0 armyless, 0 stalls (hard24-detect, 2026-09-01) | Standing reference for the pinned regime. v7 ledger-only read 20 won + 4 survived on the same seeds; under v8 no member lost or stalled, but five that won now reach the sample limit with armies standing -- whether that is pace or strength is unmeasured. Windows-path figures do not compare. |
| Very Hard (duel_lake) | `flame-nocover` | 21/48 fresh seeds (44%) | Standing. Ten modification arms rejected at two scales; local optimum of the current vocabulary. |
| Impossible (duel_lake) | `flame-nocover` | 0/~120 post-certification, survival floor ~3300 | No adoptable gain: raid 8's +217 REVERSED on a fresh tree (-198, same seeds) and is retracted; kite's +98 is downgraded to unreplicated. |
| Other maps (Impossible) | `flame-nocover` | bridge 2811 / straits 2317 / hills 2281 vs duel 3325, 0/36 | Terrain-as-parameter closed: every alternative worse, duel_lake retained BECAUSE most favorable (log 2026-08-09). |

## Adoption history (post-certification)

Every behavior the bot actually runs was once an arm that cleared a paired
panel. Nothing on this list is a guess.

| Date | Change | Evidence |
|---|---|---|
| pre-2026-08-06 | The champion stack itself (cover-off, flame 2, close 3, aa-counter guard, raid 3, tech 1, counter tilt) | The VH arm ladder, pre-certification; carried forward as the baseline the certified harness re-measured at 12/24 and 21/48 |
| 2026-08-08 | — (no adoption yet on the certified harness) | raid 8 is the first candidate; adoption awaits imp-raid8x48 |

## Open questions (measurement running or queued)

- The docket after the battery: the spatial layer (its first slice shipped 2026-08-15 -- coverage recorded as three trace columns, accumulating on every future match; next steps are the doom refit with spatial features once a corpus exists, and hold v2 designed against the field), a response worth driving with the banked doom model, and the exporter productionization. Composition tuning, the naval theater, the tilt, and the battery are all measured closed; the next gain must come from a capability class the bot does not yet have.

## Closed questions (measured, with the log entry that closed them)

**Very Hard:**
- The artillery battery ([[policy-battery]]): REJECTED at -12 (battery96, 48 pairs, 6 wins vs control's 18; arm wiped 25 to 15) with the mechanism proven in 47 of 48 arm matches -- the first new-verb panel to measure the real thing on its first attempt. The standoff counter works and still loses: $2,100 plus a builder held through construction is paid out of tempo, and no trigger quality can rescue a response that drags every game it enters (law ten, third derivation). Capability banked: the channel, the quartermaster seam, five defect fixes lifted to every twin (log 2026-08-15).
- The six-knob composition space (flame, close, raid, tech, medics, decoys): FLAT around the champion. vhsearch1's 16 machine-proposed candidates triaged to one graduation; grad1 read +4 to the digit, grad2 replicated at 22-22 dead even on fresh seeds and a fresh tree. Composition tuning is retired as a path to the next rung; the search machinery is banked (logs 2026-08-14).
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
