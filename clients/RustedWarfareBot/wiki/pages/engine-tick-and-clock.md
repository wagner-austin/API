---
title: "Engine Tick Method and Clock"
tags: [engine, tick, clock, obfuscation, reverse-engineering, discovery]
related:
  - "[[engine-name-oracle]]"
  - "[[agent-render-callback-noop]]"
  - "[[runtime-split-java-agent-python-brain]]"
  - "[[engine-entity-model]]"
  - "[[multiplayer-portability-invariants]]"
source_paths:
  - "runs/decompiled/com/corrodinggames/rts/gameFramework/f.java:117"
  - "wiki/sources/m3-discovery/gameengine-tick-method.txt:84"
  - "wiki/sources/m3-discovery/gameengine-tick-method.txt:87"
  - "wiki/sources/m3-discovery/gameengine-tick-method.txt:74"
  - "wiki/sources/m3-discovery/engine-snapshots.log:531"
  - "wiki/sources/m3-discovery/engine-snapshots.log:1091"
  - "wiki/sources/m4-entities/entity-count-loop.txt:11"
  - "wiki/sources/m4-commands/engine-tick-decompiled.txt:6"
  - "wiki/sources/m4-commands/engine-tick-decompiled.txt:17"
  - "agent/src/rwbot/agent/EngineHandle.java"
  - "agent/src/rwbot/agent/Snapshot.java"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [engine-internals, bot-architecture]
---

# Engine Tick Method and Clock

The simulation advances in `com.corrodinggames.rts.game.i.a(float)`, which increments `bx` — the engine's own **frame** counter — exactly once per call, and accumulates a derived millisecond value in `by` beside it.[^1][^3][^8] This resolves the first of the three prerequisites [[engine-name-oracle]] left open — the oracle names subsystems, not methods.

## The method

`game.i` is the class the boot log names as the live `gameEngine`, and `a(float)` takes a delta time.[^1] Its body contains a literal field increment — `getfield bx` / `iconst_1` / `iadd` / `putfield bx` — so `bx` counts invocations exactly.[^1] Immediately above, `by` is stored from an `f2i` conversion, making it a derived millisecond value rather than an independent counter.[^3]

## What the engine calls them

The engine names these fields itself, and its vocabulary is not the one this page first used. A debug line inside the same method prints `"updateAllGame1: deltaSpeed:" + f2 + " frame:" + this.bx + " network.currentStepRate:" + this.bX.c()` — so `bx` is the engine's **frame** counter, and the lockstep step rate is a separate quantity reached through the network engine.[^8]

The distinction matters for the planner rather than for pedantry. Decimating against `bx` decimates against local frames; the quantity that is agreed between peers is the network step rate, and that is the one a multiplayer-legal decision cadence has to key off ([[multiplayer-portability-invariants]]). `bx` remains the right thing to read for "has the simulation advanced", because it is incremented exactly once per update call.[^8]

`by` is likewise more specific than a plain millisecond counter: it accumulates `f2 * 16.666666f`, which is milliseconds per frame at a 60 Hz baseline scaled by the current delta.[^8] That it measures out at 1 kHz of wall-clock is a consequence of that formula, not an independent clock.

## The counters, measured

Three snapshots of the live engine ten seconds apart give the rates directly.[^4][^5] `by` advanced 9,994 then 9,999 over two ten-second intervals — a millisecond clock.[^4][^5] `bx` advanced 2,993 then 2,998 across the same intervals, which is 299.8 per second.[^4][^5]

Rate alone could not distinguish a simulation tick from a rendered frame, and the render loop is frame-rate limited in exactly this range ([[agent-render-callback-noop]]). The bytecode settles it: `bx` is incremented inside the engine's own update method, not the container's render path, so ~300 Hz is the rate at which `a(float)` is invoked.[^1] Whether the container calls it once per frame or steps it several times per frame to a fixed timestep is not established here.

## Where the clock is not incremented

Five other methods write `bx`, and none of them advance it. `gameFramework.ba.h()`, `j.ad.w()` and `j.ad.a(j.au)` all read the pair into locals, call the state (de)serialiser `y.a(k,…)`, and write the saved values back — they preserve the clock across a state load rather than move it.[^6] `j.ad.aD()` stores literal zero into both, which is the new-game reset.[^6] Mistaking any of these for the tick would put the agent's clock read on a path that fires only on load.

## The frame counter is part of the randomness

The engine's synced random — the one lockstep multiplayer requires, whose error string reads `notRandInt` — is not a generator at all but an arithmetic hash, and `bx` is one of its inputs. `gameFramework.f.a(min, max, salt)` mixes the seed field `l.bJ` with the caller's salt and then folds `bx` in four ways — `n6 += n4 * (l2.bx * 13131313); n6 += l2.bx * 1313131313 + l2.bx % 10`.[^9]

The consequence took five acceptance iterations to isolate: two runs whose menus lasted a different number of boot frames arrive at the match with different `bx`, and every synced draw then differs however thoroughly the seeded generators were pinned ([[policy-determinism]]). The agent therefore zeroes `bx` and `by` on the match's first live tick — the same reset `j.ad.aD()` performs on the load path this start path skips.

## Consequence for the agent

`gameFramework.l.B()` returns the engine singleton and its whole body is `return al;`, so reading state through it cannot advance or mutate the simulation — which is what makes a probe thread safe.[^7] With `bx` identified, a decimated planner has a real tick basis to decimate against rather than wall-clock guessing ([[runtime-split-java-agent-python-brain]]).

## The entity list

Resolved, and not where this page first guessed: the master list is the static `com.corrodinggames.rts.game.units.am.bE`, and `units.al` is the **tree** class rather than the unit class ([[engine-entity-model]]).[^2] The collections reached from the engine that hold `al` elements are therefore mostly trees, which is why their sizes never reconciled against the unit count.

[^1]: `wiki/sources/m3-discovery/gameengine-tick-method.txt:84` — `158: getfield #456 // Field bx:I`, followed by `iconst_1`, `iadd` at `:86`, and `putfield #456 // Field bx:I` at `:87`, inside `public strictfp void a(float);` declared at `:1`. Disassembled from the pinned `.game/game-lib.jar` with the bundled `javap`.
[^2]: `wiki/sources/m4-entities/entity-count-loop.txt:11` — the census loop iterating `com.corrodinggames.rts.game.units.am.bE`, with the `instanceof al` tree branch at `:12`. See [[engine-entity-model]] for the full derivation.
[^3]: `wiki/sources/m3-discovery/gameengine-tick-method.txt:74` — `138: putfield #457 // Field by:I`, preceded by the `f2i` conversion at offset 137.
[^4]: `wiki/sources/m3-discovery/engine-snapshots.log:531` — `bx = 1918` and `by = 6461` at the `t=10s` snapshot headed at `:400`; the `t=20s` snapshot at `:680` reports `bx = 4911`, `by = 16455` at `:811`–`:812`.
[^5]: `wiki/sources/m3-discovery/engine-snapshots.log:1091` — `bx = 7909` and `by = 26454` at the `t=30s` snapshot headed at `:960`.
[^6]: `javap -p -c -cp .game/game-lib.jar` over every class in the jar [synthesis] — a scan for `putfield` of `bx` returns exactly six methods. `game.i.a(float)` is the increment; `gameFramework.ba.h()`, `j.ad.w()` and `j.ad.a(j.au)` each restore a saved pair after `y.a(Lj/k;ZZZ)Z`; `j.ad.aD()` writes `iconst_0` to both; `gameFramework.l.<init>` initialises them. The `.game/` tree is untracked by design, so the command is the reproduction path rather than an archived artifact.
[^7]: `agent/src/rwbot/agent/EngineHandle.java` — the reflective accessor and the pinned-build failure contract; `javap` of `gameFramework.l.B()` shows a two-instruction body, `getstatic al` then `areturn`.
[^8]: `wiki/sources/m4-commands/engine-tick-decompiled.txt:6` — the debug line naming `this.bx` as `frame:` and `this.bX.c()` as `network.currentStepRate:`, with `++this.bx;` at `:17` and `this.by = (int)((float)this.by + f2 * 16.666666f);` at `:14`.
[^9]: `runs/decompiled/com/corrodinggames/rts/gameFramework/f.java:117`–`135` — `public static final strictfp int a(int n2, int n3, int n4)`, reading `l2.bJ` at `:126` and folding `l2.bx` at `:129`–`:130`, with the `notRandInt` range check at `:135`.
