---
title: "The Very Hard Race: 96 Games, Where the 27 Losses Are Decided"
tags: [campaign, very-hard, corpus, measurement, verdict]
related:
  - "[[campaign-ledger]]"
  - "[[impossible-economy-problem]]"
  - "[[policy-raid]]"
  - "[[policy-determinism]]"
  - "[[policy-trace]]"
source_paths:
  - "src/rw_bot/policy/match_report.py"
  - "src/rw_bot/policy/scorekeeper.py"
  - "src/rw_bot/policy/dive.py"
  - "sweeps/divebar192.txt"
  - "sweeps/bloodbar96.txt"
  - "sweeps/prangebar96.txt"
source_git_blobs:
  "src/rw_bot/policy/match_report.py": "2b95059f6ed9ecd25610092f0a49ca2f72b03215"
  "src/rw_bot/policy/scorekeeper.py": "e900ce487c7565519ff5d6687e22e24cd829a482"
  "src/rw_bot/policy/dive.py": "42c9133abd8fe268ec6ecd74814c31c77dde0a06"
  "sweeps/divebar192.txt": "10368ad3ccf2a6cf91d218850403bd4b5ebb326b"
  "sweeps/bloodbar96.txt": "5c28157873019ccfd71fbc4bd1e1339b3d02f02c"
  "sweeps/prangebar96.txt": "b79981018c3fe5149651ba08e31a799bfc535f5e"
provenance:
  - "runs/sweeps/divebar192, bloodbar96, prangebar96 and their traces on hpc3 (/pub/wagnera3/rusted/runs): the 96 champion cards these figures are read from, byte-identical to condcbar192's and condbar144/b's on every seed"
  - "scratchpad corpus96.py and early96.py (session f670d9e0, 2026-09-13): the two readers; every number below is their printed output"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-09-13
confidence: high
hubs: [bot-architecture]
---

# The Very Hard Race: 96 Games, Where the 27 Losses Are Decided

The champion (`evolve1-g4m2`) wins 69 of 96 untouched Very Hard seeds, and
the 96 games exist three times over under certified determinism: the
champion arm of `condcbar192`, of `divebar192`, and the `condbar144`/`b`
blocks read byte-identical cards on every seed. Every card since
2026-09-13 carries the cumulative `enemy peak` line (the peak simultaneous
count of every hostile type ever seen), and every trace carries the
per-loss table, so the 27 losses can be read against the 69 wins the way
pinbase48's nine never could.[^1]

## Losses are the long games

| | wins (69) | losses (27) |
|---|---|---|
| samples seen, mean | 3,367 | 7,481 |
| rival income at sample 1000 / 1500 / 2000 / 2500 | 57 / 67 / 63 / 41 | 59 / 78 / 94 / 104 |
| our income at the same samples | 67 / 62 / 59 / 59 | 66 / 57 / 52 / 49 |
| rival army value at 1500 / 2000 / 2500 | 9,703 / 8,507 / 5,437 | 11,896 / 12,993 / 14,352 |
| our army at 1500 / 2000 / 2500 | 16 / 25 / 27 | 16 / 21 / 24 |
| worth / rival at 2000: q25, median, q75 | 0.94, 1.22, 1.65 | 0.66, 0.77, 0.85 |

At sample 1000 the two populations are within a few percent of each other
on every column. By 2000 the wins have won the mid-game fight -- the rival's
army value is falling and its income collapses to 41 by 2500 as the waves
raze -- and the losses have lost it: the rival's army is half again ours in
value and its income compounds from 78 to 104 while ours erodes from 57 to
49.[^2]

## What the enemy fields in a loss

The `enemy peak` line names the tier the compounding buys. Present in the
losses and absent from the wins, with the peak count per game:

| enemy type | present in wins | present in losses | peak, wins | peak, losses |
|---|---|---|---|---|
| mechFactory | 0.00 | 0.96 | 0.0 | 1.3 |
| mechMissile / mechArtillery / mechGun | 0.00 | 0.93-0.96 | 0.0 | 2.7-3.5 |
| fabricatorT1 | 0.03 | 0.96 | 0.0 | 1.0 |
| amphibiousJet | 0.12 | 0.93 | 0.1 | 2.0 |
| battleShip | 0.12 | 0.81 | 0.1 | 1.5 |
| extractorT3 | 0.00 | 0.56 | 0.0 | 1.4 |
| c_artillery (present in every game) | 1.00 | 1.00 | 2.1 | 9.3 |
| c_turret_t1 / c_antiAirTurret | 1.00 | 1.00 | 2.4 / 3.9 | 16.5 / 19.7 |
| builder | 1.00 | 1.00 | 5.6 | 14.6 |

The mech factory is the marker: it stands in 26 of 27 losses and in no
win, and everything the losses die to in bulk -- artillery at 21.3 units
lost per game against 4.0, hover tanks 18.4 against 7.2, turrets 19.3
against 7.0 -- is what a 1.8x economy buys once the game runs past the
window the champion wins in.[^1]

## What the day's verbs bought, on the same 96 seeds

Every arm below was paired against the same certified champion cards,
so a flip and a regression are seed-named facts, not estimates.

| arm | wins / 96 | flips | regressions |
|---|---|---|---|
| champion | 69 | -- | -- |
| concurrency-gated artillery share (`outranged 3`, condcbar192) | 70 | 4 | 3 |
| dive, escalating rung (divebar192) | 66 | 1 | 4 |
| dive, artillery-gated blood 3 (bloodbar96) | 63 | 3 | 9 |
| range-first target order (prangebar96, 94 in) | 9 | 4 | 63 |

Every verb that converts an opener costs winners, because the bar's 69
winners outnumber its opener-shaped seeds two to one and a verb that
changes the mid-game fight changes it on the seeds the champion was
winning too. The dive's touched-member distribution says how: of 46
members it drew on, 25 wins stayed wins 277 samples faster, 4 wins were
lost, 1 loss was won. One party was enough on s8925833 -- the two games
are identical to frame 83,400 and the dive arm's line then bleeds on its
left flank fifteen thousand frames early.[^3]

## The lesson the opener probes taught

Every verb was first priced on the nine pinbase48 openers, where
range-first read one flip and no regression and the blood-gated dive
read four flips and two. A probe with no winner in it measures half of
every verb, and the half it omits is the majority class. Nine of the 27
loss seeds flipped under some arm today -- s8925937, s8926457, s8929161,
s8931313, s8931625, s8933601, s8934017, s8934225, s8935473 -- so a policy
that chose the right arm per seed from early state would sit near 78 of
96; no fixed arm reaches it, and the columns that separate the populations
only separate them by sample 2000, when the fight is already being lost.
That is the head tier's problem statement at Very Hard, in the corpus's
own numbers.

[^1]: `runs/sweeps/divebar192/champ-*.txt` -- `enemy peak`, `units lost to`, `samples seen` on all 96 cards, read by `corpus96.py`.
[^2]: `runs/traces/divebar192/champ-*.ndjson` -- the tick table at samples 1000/1500/2000/2500 on all 96 traces, read by `early96.py`.
[^3]: `runs/sweeps/divebar192`, `bloodbar96`, `prangebar96`, `condcbar192` -- paired verdicts per seed; the s8925833 anatomy is the pair's per-loss tables in the window 80k-160k.
