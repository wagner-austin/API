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
source_git_blobs:
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
| turtle 85 with the riposte counter-punch (turtlebar192) | 62 | 7 | 14 |
| turtle 85, hold only (turtlebar192) | 57 | 0 | 12 |
| turtle 70 with the riposte (turtlethr192) | 62 | 7 | 14 |
| turtle 100 with the riposte (turtlethr192) | 62 | 7 | 14 |
| riposte alone, no hold (ripostebar96, 95 in) | 62 | 7 | 13 |

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
in front of it, reads 62 on 95 seeds: it gains the same seven seeds
every turtle arm gained and loses thirteen. The thresholds at 70 and
100 read 62 each with the same seven gains and fourteen losses. So the
counter-punch was the family's whole effect and it is a net minus six
on the champion, and the hold's threshold does not matter because the
riposte's flips happen on seeds the hold never touches.[^13] The seven
gains are real seeds (three of them, s8925937 s8929161 s8931313, among
the nine knife-edge losses the cohort table named) and would have read as a win on any nine-seed
opener probe; the thirteen losses are the majority class the probe
omits, the lesson the opener probes already taught.

[^12]: `pair_read.py` over `runs/sweeps/turtlebar192` against `runs/sweeps/divebar192`, all 192 cards; the twelve hold-only losses are s8925001 s8925833 s8926977 s8927393 s8928641 s8929681 s8932665 s8932769 s8933497 s8933809 s8934121 s8934433, ratios 0.64-0.85 at sample 1,500 in the cohort table of `cohort_turtle.py`, the champion winning every one; the counter-punch arm wins seven of the twelve back (s8925001 s8926977 s8928641 s8932665 s8932769 s8933809 s8934433).
[^13]: `pair_read.py` over `runs/sweeps/ripostebar96` (95 of 96 cards; the missing seed is a champion win, so the closed figure is 62 or 63) and `runs/sweeps/turtlethr192` (all 192) against the divebar192 champion cards. Riposte alone: gained s8925937 s8929161 s8929473 s8931313 s8932249 s8932977 s8933913, lost s8925001 s8925105 s8925521 s8925833 s8927393 s8927913 s8928017 s8928225 s8929369 s8929681 s8933497 s8934121 s8934329. turtle70r: the same seven gained, the same thirteen lost plus s8932041. turtle100r: gained s8925937 s8929473 s8931313 s8932249 s8932977 s8933913 s8935473, lost s8925105 s8925833 s8927393 s8927601 s8927913 s8928017 s8928225 s8928433 s8929681 s8932041 s8933289 s8933497 s8934121 s8934329.

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
