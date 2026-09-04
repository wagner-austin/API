---
title: "The Exact-Timing Regime: the Ladder Re-Founded"
tags: [policy, determinism, measurement, ladder]
related:
  - "[[policy-determinism]]"
  - "[[policy-trace]]"
  - "[[policy-economy]]"
  - "[[harness-match-service]]"
source_paths:
  - "agent/src/rwbot/agent/SplitRandom.java"
  - "agent/src/rwbot/agent/MatchSetup.java"
  - "scripts/play.py"
  - "runs/sweeps/mltrace24b"
  - "runs/sweeps/vh-nocover24"
  - "runs/sweeps/hard-nocover24"
source_git_blobs:
  "agent/src/rwbot/agent/SplitRandom.java": "f9ce0027b04a03e4ceebf4b35f168bf9152a8165"
  "agent/src/rwbot/agent/MatchSetup.java": "e251938d667a8bd5a5145c7dd4b6402439de857c"
  "scripts/play.py": "c95267440b13b56f48a32def2a00b99e2f9efe72"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture, headless-harness]
---

# The Exact-Timing Regime

On 2026-08-06 the simulation became a pure function of the seed — certified
400/400 identical world digests across realtime and 10x runs (log
2026-08-06, the RNG arc) — and in doing so removed a handicap nobody knew
the opponents carried: their think-timers had started every previous match
polluted by free ticks, delaying their first aggression. **Every win rate
measured before this date was farmed from a delayed AI and does not
transfer.** Cross-regime trace comparisons fork at the first AI decision
(~s95 on duel_lake) by construction; comparing a pre-regime scorecard to a
post-regime one is comparing two different games.

Two traps this regime opened, both paid for and fixed the same day:

* **The boot-sandbox compile.** With frame zero pinned before the match
  world loads, the planner's settle loop could sample the engine's
  ten-player boot world and compile the opening plan against furniture it
  was about to lose — no factory inserted, plan dead at its first combat
  entry, 0/24 at two rungs before the trace's `plan` column located the
  poisoning (`scripts/play.py` FRESH_ROSTER_MAX / SANDBOX_SWAP_SAMPLES is
  the fix; `owned at compile` in every run log is the guard).
* **The silent map-load fallback.** A configured match whose map file a
  game dir lacks fails its load with an engine alert nothing reads, and
  the run drifts into the boot sandbox instead -- which is how six stale
  worker clones voided the whole cross-map batch family
  ([[policy-determinism]], the seating section). Clones re-sync maps on
  every prepare, and the agent-side guard landed the same day
  (`WrongWorldGuard`): the watcher latches only on the requested map, a
  failing load crashes at its origin, and a world that never arrives
  halts the JVM at 60s naming what was live instead.
* **Cross-regime comparability.** A batch states its regime by its frozen
  tree and, since 2026-08-06, its scorecard's `match` line. Panels that
  froze different regimes are different experiments whatever their names
  say ([[harness-match-service]]: freeze, verify, go). 2026-08-07 addendum:
  every panel above certified within-batch only — same-seed runs from
  separate invocations forked until the tick bracket landed
  ([[policy-determinism]], the cross-invocation section). The panels stand
  as the within-batch experiments they were; cross-invocation replay of a
  seed is a property only bracket-regime trees have.

## The ladder as re-founded (duel_lake 1v1, champion flame-nocover)

| rung | honest baseline | note |
|------|-----------------|------|
| Medium | ~saturated | fix-probe: 2W/1S of 3 |
| Hard | 16-18/24 | cover on/off a wash (paired net -2) |
| Very Hard | **12/24** | cover-off net +4, losses halved |
| Impossible | untested | pre-regime 0/47 stands, unre-measured |

None of these is a closed rung: the standard is winning a rung outright,
not leading a coin flip, and 12/24 at Very Hard is the baseline the next
arms climb from. The erosion analysis (log 2026-08-06) read the winning
trajectory as: reach 6 extractors by s1500, close early. Both naive
operationalizations of that sentence are now measured NEGATIVE (log
2026-08-07): latching the closer at 2x instead of 3x went 1W/23L (net
-16, the premature-all-in mode at panel scale), and 10 workers instead
of 8 went 8W/12L (net -5, early credits taken from the army as the first
waves land). The trajectory is a description, not a knob: the
discriminator lives upstream, in the early skirmishes that decide whether
expansion is affordable and dominance is real. The instrument for the
next step is the baseline panel's own 12-vs-12 split, not another arm.

## Across maps (xmap3, 2026-08-07): the erosion law travels

The rerun family -- flame-nocover, two seeds per map at VH -- reads
3W/3S/2L, and the law that covers all eight cards is the one the duel_lake
erosion analysis already named, read from PEAKS rather than endpoints:
every match's economy ran (peaks 6-10 extractors everywhere, all plans
compiled), and the verdict tracks whether it was HELD. Winners and
survivors end at their peak (big_island 9-10 held, 2W; two_cold_sides 7
and 4 of peaks 7-8, 2 survived at cap without closing); losers erode to
nothing while the rival compounds (lake_2p peaks 6-7 ground to 0 with
rival income reaching 187-202/s, defeated and wiped; hills_2p s12345 peak
7 to final 1). The endpoint `extractors 0 -> 0` on the lake cards is
[[policy-trace]]'s canonical trap performing on schedule -- the first
reading of this family fell into it and stands corrected in the log.
Expansion speed and the closer threshold remain THE levers, now with
cross-map evidence; no map-conditional opening is indicated.
