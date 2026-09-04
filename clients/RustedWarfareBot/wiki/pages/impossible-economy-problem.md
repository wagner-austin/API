---
title: The Impossible Economy Problem
tags: [impossible, economy, design]
related: ["[[campaign-ledger]]", "[[policy-budget]]", "[[harness-population-search]]", "[[policy-raid]]", "[[policy-economy]]"]
source_paths:
  - "src/rw_bot/policy/economy.py"
  - "src/rw_bot/policy/expander.py:68"
  - "src/rw_bot/policy/raid.py"
  - "sweeps/impden48.txt"
  - "runs/sweeps/imp48c6"
  - "runs/sweeps/imp48c0f4"
  - "runs/sweeps/impden48"
source_git_blobs:
  "src/rw_bot/policy/economy.py": "90e570f62b5b7abc44386179d53d94ff70b4b1e8"
  "src/rw_bot/policy/expander.py": "e4123f9f747523e64cb4d69b78dfa984b86bb3cc"
  "src/rw_bot/policy/raid.py": "4d6240b22965277824d1476c7dc8957b0310e74f"
  "sweeps/impden48.txt": "4926d5899c862a22df7e2e91d53efe1b41f8405a"
provenance:
  - "runs/sweeps/imp48c6 + imp48c0f4 — 96 fresh scorecards, the control read"
  - "runs/sweeps/impden48 — 144 scorecards, raid 3/6/8 on 48 shared seeds, 2026-09-03"
  - "wiki/log.md entries 2026-08-08 (pre-cluster saving/withhold variants) through 2026-09-02"
fact_checked: 2026-09-04
confidence: high
hubs: [bot-architecture]
---

# The Impossible Economy Problem

The ledger names the economy as Impossible's only measured road up. This
page pins what the problem actually is, from 96 fresh scorecards
(imp48c6 + imp48c0f4), because the obvious readings are both wrong.

## What it is not

**Not a placement failure.** The end-state lines read "extractors 0 -> 0,
income 0/s, plan blocked: nothing can make extractorT1" in 35 of 48
control matches -- but that is the death rattle, not the disease. The
`owned peak` lines show the economy machinery working: 5 extractorT1, 5
landFactory, 8 builders at peak in a typical zero-extractor-at-death
match. The bot builds its economy at Impossible.

**Not a saving/withhold tuning failure.** The pre-cluster era measured
every variant: unconditional saving doubled income and lost (the army
pauses let the enemy live), the nuke's 45,000 withhold starved the
garrison, the gun ladder's 11,000 cannibalized tempo, and the income
ladder "funds its T3s and loses anyway -- a race is not a stalemate"
(log 2026-08-08). impsearch1 re-asked the composed version at cluster
scale and the confirmation panel read zero.

## What it is

**The economy is built, then razed, and the difference compounds.** The
median control match dies at sample ~3,095. Peak economy ~5 extractors
(40/s at T1 rates) against a rival that compounds from 3,500 to
~197,000 worth -- the 5x subsidy means wave N's survivors are financed
by an economy the bot's raids never dent and the bot's own economy
rebuilds slower than waves arrive. `works lost to c_tank x5, hoverTank
x4...` is the standing shape: the scaffold dies to ordinary ground
waves, not to anything exotic.

## The three candidate capability classes

1. **Rebuild-under-fire (economy resilience).** The expander's floor
   protection reasons about CLAIMING pools, not RECLAIMING them under
   pressure; a razed extractor's pool re-enters the survey but the
   builder walk dies to the same wave that razed it. A capability that
   times reclamation to wave gaps (the momentum window already measures
   them) is policy code that does not exist.
2. **Denial (symmetric razing).** The subsidy compounds through the
   rival's OWN extractors, on the map, raidable. The raid verb exists
   and raid8-at-Impossible reversed on a fresh tree -- but that was
   pre-detection-era code and 3-6 seed screens. Whether scaled,
   sustained denial moves the compounding curve is UNMEASURED at
   current code and cluster scale.
3. **Strike-window economics.** impsearch1's strike5000 was refuted as
   a margin gain, but the release-window MECHANISM (spend the army when
   the rival's compounding dips) is the only knob that engages the
   subsidy's timing rather than its size. A composed
   denial-plus-release arm is unmeasured.

## The denial road is measured closed (impden48, 2026-09-03)

The panel this page called for ran the same day: raid 6 and raid 8 vs
the champion's raid 3 on 48 shared fresh seeds at Impossible, judged on
the rival's worth curve. **The curve did not move.** Rival peak medians
201,775 / 206,225 / 196,525 (control / raid6 / raid8) -- differences
inside noise at sd ~40k -- survival medians 3,161 / 3,188 / 3,104, and
paired margins flat (+-0.14 at sd ~0.65). Doubling and near-tripling
the raid commitment does not dent the subsidy's compounding at ANY
cost level the raid verb can express. Class 2 is dead at current
mechanics; classes 1 (rebuild-under-fire) and 3 (strike-window
economics) are the remaining roads.

## Class 1 measured FLAT at both thresholds (imprb48, 2026-09-04)

Rebuild-under-fire shipped as the `rebuild` doctrine knob
(RW-DOCTRINE-029, default 0 = prior behaviour exactly): a pool we HELD
and lost is withheld from the pool survey until the rival's army value
drops at least `rebuild` below its recent peak -- the same Momentum
wave-break signal the strike release reads -- so the builder's walk back
goes through the wave's gap instead of into its face. Virgin pools claim
as always. Mechanism: `policy/reclaim.py` (`Razed` tracker + embargo
gate) -> `survey_pools`'s `embargoed` filter, threaded via the expander.

The measurement (48 shared fresh seeds, rebuild 5000 and 15000 vs the
champion's 0): **nothing moved.** Paired margins -0.106 / +0.017 (sd
~0.7), survival medians 3,299 / 3,288 / 3,293, peak economies identical
(~5.7 extractors), end-state extractors median zero in every arm (9/48
vs 5/48 vs 6/48 matches ending with any alive -- noise). The wave-gap
reclaim-timing hypothesis is refuted at both drop scales the strike
search validated as real window-openers: by the time pools are being
razed the match is in its death phase, and the reclaim walk was never
the marginal loss. Class 1 as DESIGNED here is closed; a different gate
(embargoing ALL expansion under fire, or escorting the builder) would be
a new design, not a retuning. Class 3 (strike-window economics) is the
remaining road.

## The momentum signal's range at Impossible is MEASURED (imptr12)

Every drop-gated knob reads `Momentum.drop()` -- the strongest
survivor's army value below its 40-sample peak -- and until 2026-09-04
no record of that signal's range at Impossible existed; thresholds were
carried over from Very Hard intuition. The trace now records the signal
itself (`rival_army`, the appendix column), and twelve traced control
matches read: **per-match maximum drop 2,900-7,950, median 5,550.
Fifteen thousand NEVER fires -- the whole distribution tops out at
7,950** -- so strike 15000 and rebuild 15000 were behaviourally
identical to control in every match that ever carried them. Five
thousand opens in 9 of 12 matches but only for slivers (0-39 open
samples of thousands; zero in a quarter of matches), which reframes
strike5000's refutation as substantially "the window barely ever
opened". The calibrated band no arm has ever tested is **2,000-3,000**:
open in every match, with real dwell time. Rebuild stays closed -- its
flat brackets (always-open 0 and mostly-shut 5000) pin the band between
them -- but the strike release gets its first in-band arm.
