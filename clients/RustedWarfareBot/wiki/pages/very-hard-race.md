---
title: "The Very Hard Race: 96 Games, Where the 27 Losses Are Decided"
tags: [campaign, very-hard, corpus, measurement, verdict]
related:
  - "[[campaign-ledger]]"
  - "[[impossible-economy-problem]]"
  - "[[policy-raid]]"
  - "[[policy-determinism]]"
  - "[[policy-trace]]"
  - "[[policy-budget]]"
source_paths:
  - "src/rw_bot/policy/spending.py"
  - "sweeps/techbar96.txt"
  - "src/rw_bot/policy/match_report.py"
  - "src/rw_bot/policy/scorekeeper.py"
  - "src/rw_bot/policy/dive.py"
  - "src/rw_bot/policy/situation.py:234"
  - "src/rw_bot/policy/situation.py:237"
  - "src/rw_bot/policy/dispatch.py:380"
  - "src/rw_bot/policy/dispatch.py:424"
  - "src/rw_bot/policy/dispatching.py:374"
  - "src/rw_bot/policy/campaign.py:274"
  - "src/rw_bot/policy/campaign.py:326"
  - "src/rw_bot/policy/doctrine_codecs.py:337"
  - "sweeps/divebar192.txt"
  - "sweeps/bloodbar96.txt"
  - "sweeps/prangebar96.txt"
  - "sweeps/pressvh48.txt"
  - "sweeps/turtlebar192.txt"
  - "doctrines/evolve1-g4m2-turtle85.doctrine"
  - "doctrines/evolve1-g4m2-turtle85r.doctrine"
  - "sweeps/attrbar96.txt"
  - "sweeps/icptbar192.txt"
  - "doctrines/evolve1-g4m2-noicpt.doctrine"
  - "doctrines/evolve1-g4m2-guard3.doctrine"
source_git_blobs:
  "src/rw_bot/policy/spending.py": "77f82b55b26d45d42b6d4479accad1d3b9e17ba5"
  "sweeps/techbar96.txt": "1623224dcd8b73ef6551ba3132d7d6ee890fd8e2"
  "sweeps/attrbar96.txt": "f2952dbcc9a7e9ed967af9aa8dad89c50eda1208"
  "sweeps/icptbar192.txt": "b4f0e2ab8b424ba8c2b5d800ad3fb565366b6582"
  "doctrines/evolve1-g4m2-noicpt.doctrine": "b789d9033e6474ea839e5a9057e939dd18835ac0"
  "doctrines/evolve1-g4m2-guard3.doctrine": "7f9cb0926233176fdcee9d85b105ef896f62f28c"
  "src/rw_bot/policy/match_report.py": "2b95059f6ed9ecd25610092f0a49ca2f72b03215"
  "src/rw_bot/policy/scorekeeper.py": "e900ce487c7565519ff5d6687e22e24cd829a482"
  "src/rw_bot/policy/dive.py": "42c9133abd8fe268ec6ecd74814c31c77dde0a06"
  "src/rw_bot/policy/situation.py": "fb274ddf0a998bf1ac710d68ff173cf7646d1f2e"
  "src/rw_bot/policy/dispatch.py": "26dbd532a0aa0d08ae952d4013e26b5dd1a4125a"
  "src/rw_bot/policy/dispatching.py": "117a59a11128484869769370583290f66a1e4382"
  "src/rw_bot/policy/campaign.py": "b04a2a2e3dbec7b0625f7e1a0f66b67ccb8f4beb"
  "src/rw_bot/policy/doctrine_codecs.py": "b4e3f2ef3532f8a370bd54c51660aa006bf70414"
  "sweeps/divebar192.txt": "10368ad3ccf2a6cf91d218850403bd4b5ebb326b"
  "sweeps/bloodbar96.txt": "5c28157873019ccfd71fbc4bd1e1339b3d02f02c"
  "sweeps/prangebar96.txt": "b79981018c3fe5149651ba08e31a799bfc535f5e"
  "sweeps/pressvh48.txt": "51d3d0a7628c3fb532f984fc320ef009c0bba5a0"
  "sweeps/turtlebar192.txt": "d5633067d5ee71b18ece1ffcee58630554f1cefb"
  "doctrines/evolve1-g4m2-turtle85.doctrine": "cf0dc0dafa3116a50c08dc92a6fc366e75474841"
  "doctrines/evolve1-g4m2-turtle85r.doctrine": "9780044df77e45dba2a4ab471ac216dfc83a9482"
provenance:
  - "runs/sweeps/divebar192, bloodbar96, prangebar96 and their traces on hpc3 (/pub/wagnera3/rusted/runs): the 96 champion cards these figures are read from, byte-identical to condcbar192's and condbar144/b's on every seed"
  - "scratchpad corpus96.py and early96.py (session f670d9e0, 2026-09-13): the two readers; every number below is their printed output"
  - "runs/sweeps/pressvh48 on hpc3: the 48 paired press verdicts the turtle section reads against"
  - "runs/sweeps/turtlebar192 on hpc3, submitted 2026-09-13 at Very Hard from commit d0ea7720: the bar the turtle section names as unread"
  - "runs/sweeps/attrbar96 and runs/traces/attrbar96 on hpc3, the champion's 96 seeds replayed from commit 619f2f07c with the kill-ledger columns; attr96.py and exchange96.py (scratchpad, session f670d9e0) are the readers of the two attrition tables"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-09-14
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
| turtle 85 with the riposte counter-punch (turtlebar192) | 62 | 7 | 14 |
| turtle 85, hold only (turtlebar192) | 57 | 0 | 12 |
| turtle 70 with the riposte (turtlethr192) | 62 | 7 | 14 |
| turtle 100 with the riposte (turtlethr192) | 62 | 7 | 14 |
| riposte alone, no hold (ripostebar96) | 62 | 7 | 14 |
| interception bounded to three units (icptbar192) | 68 | 11 | 12 |
| interception off (icptbar192) | 59 | 15 | 25 |

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
every verb, and the half it omits is the majority class.

## Can a loss be seen early, and would a selector help?

Rank AUC of single tick features for loss against win, on the 96 seeds
and on 144 with pinbase48 added (a value below 0.5 means the feature is
higher in wins):

| feature at sample | 750 | 1000 | 1250 | 1500 | 1750 | 2000 |
|---|---|---|---|---|---|---|
| worth / rival (96) | 0.52 | 0.24 | 0.26 | 0.21 | 0.14 | 0.12 |
| worth / rival (144) | 0.48 | 0.28 | 0.27 | 0.21 | 0.14 | 0.12 |
| rival army / our worth (144) | 0.45 | 0.65 | 0.66 | 0.73 | 0.84 | 0.88 |
| extractors (144) | 0.46 | 0.38 | 0.30 | 0.25 | 0.26 | 0.25 |
| rival income / our income (144) | 0.53 | 0.63 | 0.75 | 0.79 | 0.82 | 0.84 |

A loss is visible at sample 1000 with about 0.75 AUC on the worth ratio
alone and 0.8 by 1500, a thousand samples before the fight is decided --
so a head fitted at 1250-1500 would have inputs to work with.[^4]

What it would not have is a response. Nine of the 27 loss seeds flipped
under some arm today (s8925937, s8926457, s8929161, s8931313, s8931625,
s8933601, s8934017, s8934225, s8935473), but on the behind-early cohort
they scatter across arms with no structure: on the 31 seeds where the
champion sat at or below 0.9 at sample 1500, the champion wins 15, the
concurrency share 16, the dose-8 arm 16, the dose-16 arm 15, the dive 12,
the blood-gated dive 11, range-first 3 -- and on the complement every
arm is at or below the champion. No measured arm is conditionally better
by more than one seed, so a selector over this vocabulary is worth at
most one win, and the nine flips are what knife-edge seeds do under any
perturbation rather than a signal an arm carries.[^5]

The per-loss tables say what the losses are: between samples 1000 and
2500 the army dies forward in both populations (41.9 deaths per winning
game, 43.4 per losing one), but winners die to enemy STRUCTURES (11.4 per
game against 5.9) because they are assaulting the base, and losers die to
the enemy's mobile army in the field (ground 16.7 against 14.4, air 12.9
against 10.5, navy 5.1 against 3.3) before they reach it.[^6] The loss is
the field fight against an army the enemy's opening rolled larger, and
the doctrine vocabulary has no arm that wins that fight more often than
the champion's own.

## The losses at the event level: the same fights, the same raids, a bigger army

Two more readers over the 96 champion traces, after the turtle closed,
asked whether anything in HOW a loss is played differs from a win.
Army deaths in the deciding window, grouped into engagements by time
gap and distance:[^14]

| | wins (69) | losses (27) |
|---|---|---|
| engagements per game | 24.4 | 25.8 |
| army deaths per game | 46.1 | 49.8 |
| mean deaths per engagement | 1.89 | 1.93 |
| share of deaths in engagements of four or more | 0.36 | 0.39 |
| fraction of the standing army lost per engagement | 0.11 | 0.12 |
| rival army / our worth when the engagement begins | 0.36 | 0.52 |

The losses fight the same number of engagements, of the same size,
losing the same fraction of the army each time; the only column that
moves is the ratio. There is no bad-odds engagement a per-sample
engage decision could refuse -- the field fight is lost globally, not
locally.

Our own income falls behind by sample 1,500 in the losses (57 against
62), and the extractor reader names the mechanism: between samples
1,000 and 1,500 the losses lose 1.30 extractors per game against 0.68,
a full 1.00 of them on our own side of the map, killed by the AI's
tank-and-scout raiders at 0.78 per game against 0.32. The extra 0.6
dead extractors are the five credits per second. The interception
answers those raids -- 222 intercept orders per thousand samples in
the losses against 84 in the wins -- and loses the extractors anyway,
so the responders exist and lose the race or the fight against raiders
drawn from an army half again larger.[^15] The expansion rate halves
(4.1 per thousand samples against 9.5) because the pools stay occupied
and the workers are the same.

Every level of the corpus read so far said one thing: a Very Hard loss
is the rival's larger army at sample 1,500, and every downstream
difference (income, extractor drops, interceptions, engagement ratio)
follows from it. What those columns could not say is whether the
rival's army is larger because it BUILT more or because we KILLED less
of it -- our deaths were recorded per event, the rival's only as a card
total -- and that is the question that separates an economy lane from
a tactical one.

## The kill ledger answers: same deaths, a fifth fewer kills, and the numbers explain it

The engine's kill ledger now rides every sample ([[policy-trace]]), and
`attrbar96` replayed the champion's 96 seeds from that tree: all 96
cards byte-identical to divebar192's, the fourth certification, so the
columns below were read off matches that are the same matches.[^16]

| sample | rival army, wins / losses | rival units lost, wins / losses | our units lost, wins / losses | exchange, wins / losses |
|---|---|---|---|---|
| 1000 | 8,459 / 9,300 | 2.6 / 1.7 | 4.9 / 5.7 | 0.70 / 0.32 |
| 1500 | 9,703 / 11,896 | 15.7 / 12.5 | 23.1 / 24.6 | 0.70 / 0.52 |
| 2000 | 8,507 / 12,993 | 32.9 / 26.7 | 37.3 / 41.2 | 0.91 / 0.67 |
| 2500 | 5,437 / 14,352 | 51.6 / 42.9 | 54.6 / 59.3 | 0.99 / 0.75 |

At sample 1,000, before real fighting, the rival's army is already 841
larger in the losses: that is the opening roll. By 1,500 the gap is
2,193, and our side of it is on the KILL side of the exchange, not the
death side -- 23.1 of ours lost against 24.6, but 15.7 of theirs
against 12.5.

The first reading of that deficit was wrong, and the correction is the
finding. Binned by the rival's army at sample 1,000, the losses traded
worse than the wins inside every bin and the exchange appeared to RISE
with the rival's size, which read as a tactical deficit rather than
Lanchester's arithmetic. But the rival grows faster between 1,000 and
1,500 in the losses (its income is 78 there against 67), so its size
at 1,000 understates what the waves actually fought. Binned by its
size at 1,500 -- the size at fight time -- the exchange falls with the
rival's size in both groups, as numbers say it should, and the gap
between wins and losses closes wherever both have members:[^17]

| rival army at 1,500 | wins: n, exchange 1,000-1,500 | losses: n, exchange |
|---|---|---|
| under 9,000 | 29, 0.94 | 4, 0.65 |
| 9,000-10,500 | 10, 0.78 | 5, 0.57 |
| 10,500-12,000 | 14, 0.60 | 5, 0.77 |
| 12,000-14,000 | 13, 0.55 | 8, 0.55 |
| over 14,000 | 3, 0.60 | 5, 0.42 |

The fights themselves say the same thing. Clustering both sides'
losses into fights by time, the losses fight the same number of trades
(3.3 against 3.1 per game) that start on the same footing (2.5 of ours
inside their guns against 2.9, 3.0 of theirs inside ours against 3.2,
an army of 10.9 against 11.2), and the kill deficit splits three ways
with no dominant piece: trades a little worse (8.4 of theirs against
9.7 for the same 11 of ours), fewer harvests where theirs die on our
defences with none of ours lost (2.4 against 3.4), and one more of ours
per game dying forward to a gunship with nothing killed back (1.7
against 0.9).[^18] Every piece is what a tenth more enemy army does at
the fight; none is a decision the champion is making wrong.

So the corpus reading stands where the event level left it, now with
the ledger behind it: a Very Hard loss is the rival's income roll,
which builds an army a tenth larger by 1,000 and a fifth larger by
1,500, and the exchange follows the numbers. `icptbar192` priced the
interception off and bounded to three units on the same 96 seeds
before that correction landed, because the window's extra
interceptions had read as a candidate cause; both are rejected. Off
reads 59, fifteen gained and twenty-five lost: the reserve turning on
raiders is worth ten wins to the champion, so the extra interceptions
are the symptom of more raids reaching home. Bounded to three reads 68,
eleven gained and twelve lost -- the largest flip count of any arm on
this bar, twenty-three seeds moving for a net of minus one, which is
what a perturbation of the deciding fights on knife-edge seeds looks
like and not what a lever looks like.[^19]

[^19]: `pair_read.py` over `runs/sweeps/icptbar192` (all 192 cards, nine members preempted and requeued) against the divebar192 champion cards. guard3 gained s8927185 s8927289 s8929161 s8929473 s8931313 s8931625 s8932977 s8933913 s8934537 s8935473 s8935785, lost s8925001 s8925105 s8925209 s8925833 s8929057 s8929265 s8931001 s8931729 s8932041 s8932145 s8933185 s8935369; noicpt gained 15 and lost 25, the lists in the log entry of 2026-09-14.

[^16]: `attr96.py` (scratchpad, session f670d9e0; a copy at `/pub/wagnera3/rusted/`) over `runs/sweeps/attrbar96` against `runs/sweeps/divebar192` (identity on every line but the title, engine clock, dives and enemy peak lines) and over `runs/traces/attrbar96/champ-*.ndjson`, columns 24-26 at the named samples and the per-loss table for our deaths to that sample.
[^17]: `exchange96.py` over the same traces: rival units lost between samples 1,000 and 1,500 (column 25 differenced) over our per-loss deaths in the same frames, binned by column 24 at sample 1,000 (the first, confounded read: wins 0.67/0.78/0.75/0.98 against losses 0.37/0.48/0.63/0.70 for rival armies under 8,000, 8,000-9,000, 9,000-10,000 and 10,000-11,000) and at sample 1,500 (the table).
[^18]: `fights96.py` over the same traces: every loss event on either side in the window (ours from the per-loss table with killer class and `x` below 1,900 as forward on duel_lake; theirs from column 25 differenced sample to sample) on one timeline, clustered with a 1,500-frame gap; a fight with losses on both sides is a trade, on ours alone a free fight, on theirs alone a harvest; the footing is columns 22, 23 and 1 at the trade's first sample. `sweeps/icptbar192.txt` with `doctrines/evolve1-g4m2-noicpt.doctrine` (intercept 0) and `doctrines/evolve1-g4m2-guard3.doctrine` (guard_cap 3), each one line from the champion; the intercept counts are the cards' `intercepted` line per thousand samples, 84 in the wins against 222 in the losses.

[^14]: `bursts96.py` (scratchpad, session f670d9e0; a copy at `/pub/wagnera3/rusted/`) over `runs/traces/divebar192/champ-*.ndjson`: army deaths in frames 75,000-190,000 grouped with a 900-frame gap and a 400-unit radius; the ratio is `rival_army / worth` at the sample before each burst. Burst size histogram, wins: 1102 singles, 289 pairs, 124 triples, 171 of four or more; losses: 494, 102, 39, 62.
[^15]: `extract96.py` over the same traces (extractor deaths by window, side and killer; extractors and workers columns at samples 750-2,000) and the `intercepted`, `expansions`, `samples seen` lines of the 96 champion cards in `runs/sweeps/divebar192`. Extractors at 1,000 / 1,500: wins 5.68 / 5.04, losses 5.56 / 4.37; workers 7.70 / 5.84 against 7.56 / 6.11.

## The plateau: a third of the games never buy the unlock, and they are the knife-edge seeds

The ledger read said the loss is numbers at the fight. The spend surface
says where a thousand credits of army went: in the losses the champion
holds them. Over samples 1,000-1,500 the losses average 1,514 credits
against the wins' 1,002 (medians of each game's mean 2,170 against 666),
with more factories (3.6 against 3.1), more of them idle (1.47 against
0.87) and 14 percent fewer units produced per unit time (30 against 35
per thousand samples).[^20] The bank is not a save that ramps and
drops: in 32 of the 96 games credits sit at a 2,200 plateau from sample
800 to the end of the match, three claims refused every tick, four
factories standing, and not one heavy tank is ever built. The other 64
games ramp to about 2,000 by sample 900, buy the land factory's unlock
and drain to the 800 reserve, and field eleven to sixteen heavies.[^21]

| | games | champion wins | heavy tanks, peak | factories |
|---|---|---|---|---|
| plateau | 32 | 17 (53 percent) | 0.0 | 3.9 to 4.1 |
| unlock bought | 64 | 52 (81 percent) | 11.3 to 16.2 | 2.9 |

The mechanism is a deadlock in the budget, not a knob
([[policy-budget]]). The unlock is a 2,000-credit claim that was
unprotected, so it needed the price plus the 800 reserve in the bank;
refused, it withholds 2,000 from every later claim, production
included, so production spends everything above 2,000 each tick and
the bank can never climb the extra 800. Only a lull with every factory
busy long enough buys it, and four factories almost never allow one.
Fifteen of the 27 losses are plateau games, and the plateau seeds are
the knife-edge seeds: eight of those fifteen are among the seeds the
turtle, riposte and interception arms flipped to wins, and most of the
seeds those arms lost are plateau wins -- every arm's churn was the
deadlock breaking or not under a perturbed rhythm.[^22]

The fix (2026-09-14) makes the unlock's claim protected, since it is the
purchase that makes the composition's own heavies buildable. `techbar96`
replays the champion's doctrine on the same 96 seeds from the fixed
tree with a prediction no earlier arm could make: the 64 unlock-bought
seeds replay identically, and the change falls on the 32 plateau seeds,
17 wins to hold and 15 losses to win.[^23]

[^20]: `spend96.py` over `runs/traces/attrbar96` (columns 2, 6, 7, 12 at samples 750-2,500 and over the window) and the cards' `reinforced` and `samples seen` lines.
[^21]: `bank96.py` over the same traces: credits, idle producers, refusals and orders per 100-sample block from 800 to 1,900; the plateau is a game whose mean credits over samples 1,000-1,500 exceed 1,800, and its heavy count is the cards' `owned peak` line.
[^22]: plateau wins: s8925001 s8925209 s8925313 s8925833 s8926977 s8927393 s8928017 s8928121 s8928225 s8928433 s8929369 s8929681 s8931001 s8931729 s8932041 s8933185 s8933289; plateau losses: s8925729 s8926249 s8926353 s8926457 s8927185 s8927497 s8929161 s8929889 s8932249 s8933601 s8934225 s8934641 s8935057 s8935473 s8935681. Compare the gained and lost lists in footnotes 12, 13 and 19.
[^23]: `src/rw_bot/policy/spending.py`, `unlock_tech`, commit 6048b74a2; `sweeps/techbar96.txt`.

## The response built from the reading: the turtle

The press was the first arm cut from this reading and it read the wrong
way: at sample 2000, at or below its worth percent of the strongest
rival, the press forces every wave out, and on 48 paired seeds the
press-80 arm lost four more games than the champion (`pressvh48`).[^7]
The turtle is the same one-shot read at sample 1500 with the opposite
response: at or below the percent (85 on the arms built), the wave
controller withholds every size release for the rest of the match, the
wave already out is sent home, and the reserve gathers under the guard
and the anti-air. A forced release still runs: an avenge, a committed
close or a strike sets a punching latch that lets the wave out and keeps
it out until its break, so the hold can carry the counter-punch the
community corpus ranks first against the shipped AI (let the attack burn
on the defences, then push into the window before the next group
stages, [[ai-opponent-strategy]]).[^8] Every held sample carries the `W`
code in the trace's event column, beside the press's `C`.[^9]

The field is `turtle`, a percent validated under code RW-DOCTRINE-047,
zero on every tracked preset.[^10] Two arms stand on the champion:
`evolve1-g4m2-turtle85` holds with riposte off, so no forced release
exists and the arm never punches; `evolve1-g4m2-turtle85r` holds with
riposte on, so an intrusion ending inside the outpost radius releases the
whole reserve as one wave. `turtlebar192` prices both against the
champion on the 96 certified bar seeds at Very Hard, and the pair
separates the hold's own worth from the counter's. The adoption line was
69 held with wins on the behind-at-1500 cohort.[^11]

The bar rejects both. Hold plus counter-punch reads 62 on all 96 seeds,
seven flipped to wins and fourteen lost; the hold alone reads 57, twelve
lost and none gained, and every one of the twelve is a seed where the
champion was behind at the window and won anyway. The first sixty cards
read the other way (55 to 50), because the short games file first and
the long games are the losses the hold exists for: on those the recall
forfeits the field fight the champion goes on to win, and six of them
end as 10,000-sample stalemates.[^12] With
the press at -4 the response space of attacking LESS after a mid-game
read is closed in both directions.

The attribution closed the same day. The riposte alone, with no hold
in front of it, reads 62 on all 96 seeds: it gains the same seven seeds
every turtle arm gained and loses fourteen, and its fourteen are the
turtle-70 arm's fourteen seed for seed. The thresholds at 70 and 100
read 62 each with the same seven gains and fourteen losses. So the
counter-punch was the family's whole effect and it is a net minus
seven on the champion, and the hold's threshold does not matter
because the riposte's flips happen on seeds the hold never
touches.[^13] The seven
gains are real seeds (three of them, s8925937 s8929161 s8931313, among
the nine knife-edge losses the cohort table named) and would have read as a win on any nine-seed
opener probe; the fourteen losses are the majority class the probe
omits, the lesson the opener probes already taught.

[^12]: `pair_read.py` over `runs/sweeps/turtlebar192` against `runs/sweeps/divebar192`, all 192 cards; the twelve hold-only losses are s8925001 s8925833 s8926977 s8927393 s8928641 s8929681 s8932665 s8932769 s8933497 s8933809 s8934121 s8934433, ratios 0.64-0.85 at sample 1,500 in the cohort table of `cohort_turtle.py`, the champion winning every one; the counter-punch arm wins seven of the twelve back (s8925001 s8926977 s8928641 s8932665 s8932769 s8933809 s8934433).
[^13]: `pair_read.py` over `runs/sweeps/ripostebar96` and `runs/sweeps/turtlethr192` (all 96 and all 192 cards) against the divebar192 champion cards. Riposte alone: gained s8925937 s8929161 s8929473 s8931313 s8932249 s8932977 s8933913, lost s8925001 s8925105 s8925521 s8925833 s8927393 s8927913 s8928017 s8928225 s8929369 s8929681 s8932041 s8933497 s8934121 s8934329 (the last card to file, s8932041, was a champion win the riposte lost). turtle70r: the same seven gained and the same fourteen lost, seed for seed. turtle100r: gained s8925937 s8929473 s8931313 s8932249 s8932977 s8933913 s8935473, lost s8925105 s8925833 s8927393 s8927601 s8927913 s8928017 s8928225 s8928433 s8929681 s8932041 s8933289 s8933497 s8934121 s8934329.

[^7]: `sweeps/pressvh48.txt` -- control against `vh-press80` on 48 fresh seeds 8891001-8895889, judged by the pairs and margin report; the press reads at `src/rw_bot/policy/situation.py:223` (`PRESS_WINDOW`).
[^8]: `src/rw_bot/policy/situation.py:234` (`TURTLE_WINDOW = 1500`) and `:237` (`Press`, now taking its window); `src/rw_bot/policy/dispatch.py:380` (`WaveController.command`, the `withhold` flag) and `:424` (the punching latch: a withheld controller with no forced release clears its released set, and a latched one musters until `released` empties); `src/rw_bot/policy/campaign.py:274` (`turtler = Press(turtle, TURTLE_WINDOW)`) and `:326` (the read beside the press read).
[^9]: `src/rw_bot/policy/dispatching.py:374` (`_note_releases`, `W` at `:397`); `tests/test_campaign_turtle.py` pins the latch, the three recall moves and the `W` row through `play`.
[^10]: `src/rw_bot/policy/doctrine_codecs.py:337` -- `_percent(payload, "turtle", _BAD_TURTLE, ...)`, the helper `hp_floor` and `hold` share.
[^11]: `doctrines/evolve1-g4m2-turtle85.doctrine`, `doctrines/evolve1-g4m2-turtle85r.doctrine`, `sweeps/turtlebar192.txt` -- 96 seeds 8925001+104k and 8931001+104k, arm cards only, the champion's cards being divebar192's under certified determinism; commit d0ea7720.

[^4]: `auc96.py` over `runs/traces/divebar192` and `runs/traces/pinbase48` with their cards.
[^5]: `cohort96.py` over `runs/sweeps/{divebar192,condcbar192,bloodbar96,prangebar96,condbar144,condbar144b}` and the champion traces of divebar192.
[^6]: `where96.py` over the per-loss tables of `runs/traces/divebar192/champ-*.ndjson`, frames 75,000-190,000, home meaning x at or beyond 1,900 on duel_lake.

[^1]: `runs/sweeps/divebar192/champ-*.txt` -- `enemy peak`, `units lost to`, `samples seen` on all 96 cards, read by `corpus96.py`.
[^2]: `runs/traces/divebar192/champ-*.ndjson` -- the tick table at samples 1000/1500/2000/2500 on all 96 traces, read by `early96.py`.
[^3]: `runs/sweeps/divebar192`, `bloodbar96`, `prangebar96`, `condcbar192` -- paired verdicts per seed; the s8925833 anatomy is the pair's per-loss tables in the window 80k-160k.
