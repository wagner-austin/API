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
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-08-06"
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
  ([[policy-determinism]], the seating section). Clones now re-sync maps
  on every prepare; the agent-side die-loudly guard is queued.
* **Cross-regime comparability.** A batch states its regime by its frozen
  tree and, since 2026-08-06, its scorecard's `match` line. Panels that
  froze different regimes are different experiments whatever their names
  say ([[harness-match-service]]: freeze, verify, go).

## The ladder as re-founded (duel_lake 1v1, champion flame-nocover)

| rung | honest baseline | note |
|------|-----------------|------|
| Medium | ~saturated | fix-probe: 2W/1S of 3 |
| Hard | 16-18/24 | cover on/off a wash (paired net -2) |
| Very Hard | **12/24** | cover-off net +4, losses halved |
| Impossible | untested | pre-regime 0/47 stands, unre-measured |

None of these is a closed rung: the standard is winning a rung outright,
not leading a coin flip, and 12/24 at Very Hard is the baseline the next
arms climb from. The named path (log 2026-08-06, erosion analysis):
winners reach 6 extractors by s1500 and close early; losers reach the same
peak late and erode. Expansion speed and the closer threshold are the
levers; erosion is what forecloses them.
