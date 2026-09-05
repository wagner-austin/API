---
title: The Impossible Economy Problem
tags: [impossible, economy, design]
related: ["[[campaign-ledger]]", "[[policy-budget]]", "[[harness-population-search]]", "[[policy-raid]]", "[[policy-economy]]"]
source_paths:
  - "src/rw_bot/policy/economy.py"
  - "src/rw_bot/policy/expander.py"
  - "src/rw_bot/policy/floor.py"
  - "src/rw_bot/policy/raid.py"
  - "sweeps/impden48.txt"
  - "runs/sweeps/imp48c6"
  - "runs/sweeps/imp48c0f4"
  - "runs/sweeps/impden48"
source_git_blobs:
  "src/rw_bot/policy/economy.py": "ecb91c97c17306e889a9c49209bf5baa6a3efe13"
  "src/rw_bot/policy/expander.py": "eeb4f76c13011abaab836b71aec978d582e2e68c"
  "src/rw_bot/policy/floor.py": "5471ec6039bbbe7a3256e8fdfcc68708d55732ee"
  "src/rw_bot/policy/raid.py": "4d6240b22965277824d1476c7dc8957b0310e74f"
  "sweeps/impden48.txt": "4926d5899c862a22df7e2e91d53efe1b41f8405a"
provenance:
  - "runs/sweeps/imp48c6 + imp48c0f4 — 96 fresh scorecards, the control read"
  - "runs/sweeps/impden48 — 144 scorecards, raid 3/6/8 on 48 shared seeds, 2026-09-03"
  - "wiki/log.md entries 2026-08-08 (pre-cluster saving/withhold variants) through 2026-09-02"
  - "runs/sweeps/impopen96 — 96 scorecards, 8 opening arms on 12 shared seeds, 2026-09-05"
fact_checked: 2026-09-05
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
opened". The calibrated band no arm had ever tested was **2,000-3,000**:
open in every match, with real dwell time. Rebuild stays closed -- its
flat brackets (always-open 0 and mostly-shut 5000) pin the band between
them.

## Class 3 measured closed WITH the mechanism firing (impstrike48)

The in-band arms ran the same day: strike 2500 and 4000 vs the
champion's 0 on 48 shared fresh seeds. This time the release provably
happened -- the traces carry `S` events in **48 of 48** s2500 matches
(window open 100-189 samples in the deepest), 33 of 48 at s4000, zero
in control -- and the outcome did not move: **paired margins +0.075 /
+0.080 (sd ~0.65), survival medians 3,165/3,028/3,286, zero wins
anywhere.** Releasing the horde into a genuinely broken wave neither
extends survival nor converts anything: the wave that broke is replaced
faster than the release can spend the gap. All three named roads --
denial, rebuild-as-designed, strike-window -- are now closed on
measurements where each mechanism verifiably engaged. What remains is
not a knob: the subsidy's compounding outruns every single-verb answer
the current vocabulary can express, and the next road must be a
structurally different way to play the rung.

## The braced pivot closes the same way (impbrace48b, 2026-09-05)

Step three's first learned RESPONSE ran the same night it was built: the
razing head (grouped-CV AUC 0.94-0.95; [[impossible-step-three-design]])
armed in **48 of 48** braced matches at roughly sample 3,000 -- inside
its 400-sample horizon before the median death at ~3,200, exactly as
designed -- zeroing the reserve and standing expansion down. The
outcome: **paired margin -0.029 (sd 0.607), survival medians 3,152 vs
3,237, end-state economies identical** (mean 0.08 extractors, 4/48
alive on each side). A near-perfect prediction of the razing, acted on
with a match-reshaping pivot, moves nothing: by the time the razing is
foreseeable even 400 samples out, no reallocation of the bot's own
credits changes what the 5x subsidy does next. Four mechanism-verified
closures now stand -- denial, rebuild, strike, brace -- and together
they say the endgame is not where Impossible is lost. Whatever decides
the rung is set earlier than any of these levers reach, or above the
credit-allocation vocabulary entirely.

## The road, staged by the evidence (2026-09-05)

**Stage 1 -- the opening scaffold, the last unsearched decision layer.**
Every arm in the entire Impossible record opens identically: three
extractors, the inserted factory, then army -- the scaffold is fixed by
construction in every doctrine and the genome compiler deliberately
preserved it. The four closures point EARLIER than any measured lever,
and the AI-trigger page says the early window is structured: the first
attack commits only when a 3-unit group fills, and its build decisions
ride a credit ladder ([[engine-ai-triggers]]). Whether 2 or 6 openers,
a tank before the first extractor, or a forced second factory changes
the early race has never been asked. ASKED 2026-09-05 (impopen96, 8 arms
x 12 shared seeds): **no scaffold variant moves survival** -- medians
2,472-3,245 around control's 2,894, zero wins anywhere -- so the stage-2
gate is unmet on the screen. One arm earned the panel tier: **e1, one
extractor then army, margin +0.327 with rival peak 161k vs control's
205k at identical survival** -- the only lever yet measured that dents
the rival's compounding at all (impden48's tripled raids could not).
Sub-significance at n=12; `impe1v48` (48 pairs, laws six and nine) is
the verdict. Greed (e4/e5/e6) reads flat-to-negative: more opening
economy does not help, consistent with the four closures.

**Stage 2 -- fortress, bank, finisher: blocked only on stage 1.** The
community's stated Impossible path is turtle -> fortress -> nuke, and
every mechanical link is PROVEN in this harness -- the launcher places,
the warhead stockpiles, the strike erases what it is pointed at
([[community-play-strategies]]). It is refuted solely on funding: no
measured Impossible state carries the 45,000. The one non-loss ever
recorded (tech-flame, full-cap survival) came from waves dying on a
funded wall. If any opening moves survival materially, the 56k chain
becomes fundable -- with law eight satisfied this time: the withhold
gates on the proven head machinery rather than tick-one or a crude
income floor, which is exactly how the earlier nuke arms starved.

**Stage 3 -- above the vocabulary, if openings read flat.** Per-sample
tactical policy ([[impossible-step-three-design]] option 1), and the
scripted-AI plan surface itself: the AiZones instrument reads the AI's
own group targets and timers ([[engine-ai-zones]]), and a posture
designed against the wave-group ladder (keep their attack group from
filling; bait a committed group) is a capability no credit knob
expresses. Expensive, and the operator weighs it -- but it is a road,
not a wall.
