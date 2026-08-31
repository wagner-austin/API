---
title: "The Noise Floor: What a Match Measures, and How Many You Need"
tags: [harness, measurement, determinism, statistics, methodology]
related:
  - "[[harness-parallel-matches]]"
  - "[[policy-holding-ground]]"
  - "[[policy-production]]"
  - "[[ai-opponent-strategy]]"
source_paths:
  - "runs/sweeps/noise"
  - "runs/sweeps/noise-seeded"
  - "agent/src/rwbot/agent/EngineRandom.java"
  - "agent/src/rwbot/agent/TickBracket.java"
  - "runs/bracket-ff1-trace.ndjson"
  - "src/rw_bot/harness/sweep.py"
source_git_blobs:
  "agent/src/rwbot/agent/EngineRandom.java": "0dbdc5e44fac79c1a2a808497d79d8478406a90b"
  "agent/src/rwbot/agent/TickBracket.java": "8086893ba07baeec9c4173fc58533168e6ecf303"
  "src/rw_bot/harness/sweep.py": "2f4275d603abb31bbaf1411fe712984c5e2b34d7"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [headless-harness, bot-architecture]
---

# The Noise Floor

A match is not a measurement. Twelve of them are, barely, and only of the right figure.

## The measurement

Twelve matches from one identical job specification — same seed, same arguments, same code — played to a verdict:[^1]

| figure | unseeded | seeded | coefficient of variation |
|---|---|---|---|
| verdict | 3 survived / 9 defeated | 3 survived / 9 defeated | — |
| total worth at the end | 350 – 15,350 | 500 – 15,850 | **1.07** |
| income at the end | 0 – 54/s | 0 – 60/s | 1.00 |
| extractors at the end | 0 – 3 | 0 – 4 | 0.72 |
| expansion orders | 118 – 194 | 119 – 194 | 0.15 |
| samples seen | 3,232 – 4,000 | 3,005 – 4,000 | 0.098 |

**A 25% survival rate on an identical specification.** Any arm of six seeds reporting one or two survivals has reported the base rate.

## What this invalidated

Four arms measured before this floor was known, all of which read as results at the time and none of which was one: more builders at one survival in six against none; the army-composition arms at none in six each; the upgrade arm at three in six. All sit inside a 25% base rate.

Worse, a single run was cited as evidence that more builders worked — *survived, four extractors, 66 credits a second, total worth 13,800*. It is the same specification as these twenty-four and sits above every one of them. It was the top of the distribution read as the effect of a change.

**Two findings survive, because they fall outside the floor.** More builders raised expansion orders from 16–22 to 107–182, where the floor's own spread on that figure is 118–194 — non-overlapping against the low arm. And throughput-before-income wiped three matches in six against **zero** wipes in twenty-four here ([[policy-production]]).

## Why seeding did not fix it

The agent pins the engine's own generator, and that is not the only one. Twelve engine call sites use `java.lang.Math.random()`, a JVM-global generator nothing was seeding, and they are not incidental: the AI picks *which unit to plant a new base at* through it, positions its sites and its workers' destinations on a random disc around them, and scatters unit positions by up to eight world units.[^2] So the opponents built their bases somewhere different every run.

That is real, it is now seeded, and **it changed the distribution not at all** — three survivals in twelve either way.

Two reasons, and both matter for anything else attempted here:

* Seeding fixes the *sequence*, not which draw each consumer receives. If the number of calls before a decision varies, every consumer downstream of it shifts.
* The map settles for **22 seconds of free-running wall clock** before the planner attaches, on a simulation that advances by the millisecond delta Slick measured for the frame.[^3] Runs therefore begin from worlds that already differ.

The seeding stays, because it removes a real uncontrolled source and costs nothing. It is not a solution.

## The fork anatomy: one draw-count leak, measured to the frame

The 2026-08-06 fork matrix (ten seed-9 traces compared pairwise on the world-digest column) reduced "runs do not replicate" to one shape. Every same-config pair — realtime and 10x fast alike — agrees bit-for-bit for thousands of frames and then forks at a discrete event where the opponent AI resolves one choice differently; eight runs split into exactly two families at frame 7200 on a single build pick (+600 vs +700 rival worth). Divergence is not drift; it is a rare consequential draw landing one unit over.

Three bytecode facts explain it (javap against the pinned jar). The engine's own per-match randomness reset (`f.a()`) seeds generator `b` — which **nothing draws from**; all five draw helpers read generator `a` — which the engine **never seeds**. The agent compensates (`EngineRandom`: engine generator + `Math.random()` + since 2026-08-06 `Collections.shuffle`'s generator, all seeded at premain and reseeded at match start). And the per-sample RNG ledger (`RandomLedger`, `rng frame=` lines in the agent log) showed the remaining leak precisely: in a pinned identical pair the **engine stream's states desync at frame 150** — third sample — while the world stays bit-identical until frame 7875 (fast) / 9525 (realtime). Something draws from the engine generator at a run-varying rate whose values usually touch nothing; every consequential consumer downstream inherits the shifted stream and eventually one choice lands differently.

Consequences: the fast-forward accelerator is exonerated (fast-vs-realtime matched 293/400 — exactly as well as realtime matched itself), and the leak is universal, not load- or speed-specific. The named instrument for the leak is the draw tap (`RandomTap`, `PLAY_RNGTAP=1`): it replaces the generator with a per-caller counting one, and two tapped runs' `rngtap frame=` tallies name the call site whose count varies.

## Solved: bit-exact replication at both speeds (2026-08-06)

The tap named three wall-paced leaks and one structural race, and the fixes compose into full determinism — **400/400 identical world digests realtime-vs-realtime, fast-vs-fast at 10x, and fast-vs-REALTIME, with all three generator states identical at every sample, measured under concurrent panel load** (`runs/cert7-*`, `runs/certf-*`).

The stack, in dependency order:

1. **Three seeded streams** — the engine's own generator (whose shipped per-match reset seeds the WRONG field, bytecode-verified), `Math.random()`, and `Collections.shuffle`'s generator — seeded at premain and reseeded at match start (`EngineRandom`).
2. **The probe-race latch** (`CommandChannel`): until a planner has acked once, a departure re-enters the wait on the same frame; a readiness probe can no longer release the world for one tick.
3. **The ambient silencer** (`MatchSetup`): the effects manager's spawner accumulates the MEASURED delta and spends two sim-stream draws per ~10 wall-units; its accumulator is parked at -1e30 on seeded matches.
4. **The tick-split generator** (`SplitRandom`): simulation draws are served from the seeded sim stream, everything render-paced from a salted side stream. Kills the unit-sway redraws and every other render-paced drawer at once, with no bytecode.
5. **The pre-tick watcher** (`Orders.onEngineTick` → `game.i.k`): the match watcher rides the engine's pre-update runnable queue, so pins, reseed, frame-zero, AI-timer reset and the hold all land BEFORE the new world's first update. Zero free ticks; the opponents' think-timer floats never see a wall-valued delta.

What this retires: the noise floor above no longer binds seeded solo runs — same seed now means same match. Chaos remains real across seeds (that is the game), so paired-seed comparison stays the standard; what changed is that a seed is now a controlled experiment at either speed, and fast panels are bit-exact replays of realtime panels at a third of the wall clock.

## Solved again, one level deeper: the same seed across INVOCATIONS (2026-08-07)

The 400/400 above only ever compared runs within one batch, and the property
it certified was narrower than the sentence claimed: same-seed runs from
SEPARATE invocations forked — always in the opponent's behavior (an extra
anti-air turret, different factory queues), always surfacing at a
consequential roll (frame 7050 on duel_lake, realtime and 10x alike) — while
parallel replicas agreed for whole panels. The hunt disproved, each with an
artifact: the load-drawn synced seed `bJ` (a real hole — the load assigns it
from a generator draw the menu world races; pinned from the match seed at
liveness — but cross-pairs matching with different `bJ` killed it as the
cause), entity ids, menu rotation, per-dir file state, identity hashes
(`-XX:hashCode=2` armed, fork unchanged, flag removed), and the draw tap's
per-window counts, which turned out to be wall-cut windows over bit-identical
worlds — a measurement artifact, not a signal.

What survived every test was the split itself: item 4 classified each draw by
**walking its stack**, and a per-draw classification that consults the
JIT-shaped frame stream is process-varying in a way no seed can reach. One
draw routed differently shifts the sim stream; the world forks at the next
behavioral roll; twin processes share a JIT timeline, which is exactly why
within-batch certification passed for a day while sequential invocations
never once agreed. The walk is deleted. Routing now asks a phase flag
(`TickBracket`): raised at the top of each tick by a ride on the pre-tick
queue, lowered after the simulation by a ride on the script queue — ordering
the engine itself guarantees — self-sustaining from the latch, fast-forward's
extra ticks bracketed explicitly. Certified: separate invocations bit-exact
at 10x (250 samples) and realtime (150), and 10x-vs-realtime bit-exact
(`runs/bracket-*-trace.ndjson`). Regime note: the bracket changes draw
routing, so pre-bracket seeds do not replay under it; batches state their
regime by frozen tree, and every panel to date stands as the within-batch
experiment it actually was.

### 2026-08-31: that certification was underpowered, and three more seams were open

**The paragraph above overstates what was measured, and this says so rather
than deleting it.** Its artifacts are `runs/bracket-{ff1,ff2,rt1,rt2}-trace.ndjson`
— four traces, so **two invocation pairs**, one at 10x and one at realtime.
Replayed on HPC3 as 24 scheduled jobs, the fork rate on duel_lake is roughly
**one pair-observation in three**, at which rate two pairs both passing is
about a coin flip. "Separate invocations bit-exact" was a true statement
about the runs that were made and not a property of the build.

HPC3 is where it surfaced because one match per job makes **every**
comparison cross-invocation; the cluster did not introduce the defect, it
removed what had been hiding it. Three seams were still open, and each was
invisible until the one above it closed:

* **Two of the three generators were never split.** `EngineRandom.seed`
  pins the engine's generator, `Math.random()`'s and `Collections.shuffle`'s
  alike, and seeding makes a run *start* from a known state while saying
  nothing about whether the simulation's draws from it are a function of the
  seed. `Math`'s holder is JVM-global with twelve engine call sites, so the
  render path and the simulation drew from one stream. The ledger had said
  so all along: across eleven replayed seeds, every seed whose `math=` state
  agreed at frame 0 replicated over 250 samples and every seed whose state
  differed at frame 0 forked — with the engine stream diverging only *after*
  the world already had, which makes it consequence and not cause.
* **Cosmetic drawers inside the tick.** `y.a`/`y.b` decrement a per-unit
  float by the MEASURED delta and, when it drains, spend two `Math.random()`
  draws scattering a particle. Silenced, and only when not hosting — the
  containment `SyncPathTransformer` already established.
* **`aR`, the eleventh AI cadence clock.** `AiTimers` reset ten. The AI
  spends the delta two ways — `aX += f2` and `aX = f.a(aX, f2)` — and a grep
  for one form finds only half of them.

**The bracket is not at fault, and a natural-sounding diagnosis of it is
wrong.** It was tempting to conclude the bracket admits the render pass;
`e.b.a` is called from `units/d/r.java:147` inside the unit's own per-tick
update, not from rendering. The engine *interleaves cosmetic work into the
simulation update*, so no phase classifier can separate them — which is a
real limitation of routing by phase rather than by call site, and the
deleted stack walk could tell them apart precisely because it looked at the
call site.

**Where it stands.** Seed 31337, which forked 3/3 before any of this, is
bit-identical across six separate invocations under the depot's Java 8 —
world and all three draw-count streams. Cluster members now pin the frame
delta (`PINNED_DELTA_MS = 3`), which is a **new regime**: numbers from a
pinned batch are not comparable to any batch before this date. Pinning makes
the remaining in-tick cosmetic draws *deterministic*, not *absent*; the
structural fix — routing them off the sim stream per call site, patched at
class-load — is not done. Full narrative in the log, entries 2026-08-30 and
2026-08-31, one of which corrects the other.

## The standing rules

**Score on survival time, or on worth share.** The endpoint figures the scorecard reports are the worst available: final worth has a standard deviation larger than its own mean. Computed from the per-sample trace instead, the mean share of worth held against the strongest rival has a coefficient of variation of **0.066**, and survival time **0.098** — the endpoint figure beside them is **0.67**.

**Pair arms across the same seeds.** Comparing per seed removes the map and the opponents from the comparison, which is most of what the spread is.

**Twelve runs an arm.** That gives a standard error near 87 samples on survival time, so a difference of about 250 samples is detectable. Detecting a change in the *verdict* rate from 25% to 50% would need roughly 58 matches an arm.

**Do not run one-match-per-arm screens.** A twelve-way screen of army compositions would report about three survivals whichever compositions it held. Twelve are written up and parked for exactly this reason.

[^1]: `runs/sweeps/noise/` and `runs/sweeps/noise-seeded/`, twelve results each from `sweeps/noise.txt` — one job line repeated twelve times under distinct labels, since results are filed by label.
[^2]: `.decompiled/com/corrodinggames/rts/game/a/a.java:1713,1737,1761`; `game/a/o.java:96-97,166-167` — `o.w()` returns a random point on a disc and `a.java:1575` hands it to a worker as a destination; `game/units/y.java:4811-4837`.
[^3]: `.decompiled/com/corrodinggames/rts/java/u.java:210-212` stores the delta, `:637` scales it, `:710` passes it to the simulation, `:714` resets it. See [[harness-parallel-matches]] for why the frame cap is not the lever it looks like.

## The seating is part of the specification, and it silently was not

The match setup queues the engine's own GUI script, and the loader reads
`numberOfAIs` off the OPEN document with a Java fallback of **four**, capped
by the map's spawn count -- not by the count its name advertises.[^4] Every
"(2p)"-named skirmish map in the shipped roster except duel_lake carries
four spawns, so the entire first cross-map arc silently played 1v3 while
its notes said 1v1: the scorecard's `players 4 -> ...` line was the only
witness (log 2026-08-05).

The fix writes the requested count onto the live document between the open
and the load, through `setValueById` -- the engine's own script-callable
setter for exactly the attribute the loader reads. Editing the `.rml` file
on disk was tried in the original match-setup work and does nothing, and
the decompile now explains why rather than merely recording it: the file
carries no `value` attribute for the element; only the live document does.
Verified live: lake_2p, opponents=1, `players 2 -> 2`
(`runs/seat-probe.out`).

**The rule this buys:** the `players N -> ...` line is part of every
scorecard read, always. A verdict whose seating was never checked is a
verdict about an unknown experiment.

The rule promptly earned its keep against a second, stranger-looking case:
worker clones seated three players with the override verifiably reading
two -- "a value nobody wrote", dir-dependent, override-independent. Run to
ground on 2026-08-06, it was never a seating mechanism at all: six maps
added to the pinned copy after the clones were made (the true-1v1 set,
lake_2p among them) had never reached them, the engine failed the map load
**with an alert nothing read**, and fell back to its boot sandbox -- a
3-to-5-player FFA whose seat count varies with the background-map
rotation. Every scorecard of the xmap-2* batches and duel-lake_2p reads
`players 4 -> N` with a sandbox opening (worth 37,800, army 12 at s0):
**that whole cross-map family never played its named maps and its
conclusions are void.** `prepare_clone` now re-syncs the pinned copy's
maps into every reused clone, loudly (`_sync_maps`,
`runs/seat-probe5.log` is the repro). The reader-side tells beyond the
players line: `owned at compile` in the run log, and the fresh-start worth
-- a real duel opens at 3,500 ([[policy-exact-timing]]).

The agent-side half landed 2026-08-06 (`WrongWorldGuard`): the match
watcher's liveness predicate now carries a map term -- no world but the
requested one can receive the setup or open the channel, however a load
fails -- the engine's own automated-testing switch (`l.aT`) is armed so a
failing load crashes at its origin with the map named in the stack trace,
and a requested world that never arrives halts the JVM at 60s (exit 70,
under the harness's 90s port wait so the agent's diagnosis wins the log).
Verified both ways: a missing map dies loudly twice over, and the real
duel plays through untouched. Building it also settled how the old latch
ever worked, from artifacts rather than inference: **every pre-guard run
latched on the menu world** -- the latch sits ~5 log lines after "match
starting" in poisoned and healthy batches alike, the AiTimers line says
"10 team(s)", and the first map scan reads the menu's 10 pools -- because
the menu background is a running mission demo that passes a
player-and-units predicate, and the start script executes a full frame
after the runnable that queues it. The whole setup ran against the menu;
the load then proceeded under the hold and swapped the world beneath the
open channel. It worked anyway for two measured reasons: difficulty
survived because the reflective write lands before `open()` and the
document round-trips current settings back through `loadConfigCommon`
(frame-0 income ratios read exactly 1.78/1.39 across every batch,
poisoned included), and the compile stayed clean only when play.py's
sandbox-swap wait held the planner past the swap -- the poisoning was
never an agent-side race lost, it was the planner compiling before a swap
the agent had no part in signalling. The gated latch replaces both lucky
mechanisms with construction, and makes the documented reseed semantics
true for the first time: the match now starts from exactly the seed,
after the load, instead of from whatever the load left of it.

[^4]: `.decompiled/com/corrodinggames/librocket/scripts/Root.java:626-645`
    (`loadConfigCommon`: the element reads and the fallback), `:204-211`
    (`setValueById`).
