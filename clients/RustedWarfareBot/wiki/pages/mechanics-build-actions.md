---
title: "Build Actions: Two Families, Two Verbs, Five Gates"
tags: [mechanics, actions, production, orders, diagnostics]
related:
  - "[[building-structures]]"
  - "[[issuing-orders]]"
  - "[[mechanics-build-tree]]"
  - "[[wire-contract-ndjson]]"
  - "[[policy-loop]]"
source_paths:
  - "runs/decompiled/com/corrodinggames/rts/game/units/a/s.java:194"
  - "runs/decompiled/com/corrodinggames/rts/game/units/a/l.java"
  - "runs/decompiled/com/corrodinggames/rts/game/units/d/k.java:433"
  - "runs/decompiled/com/corrodinggames/rts/game/units/d/i.java:144"
  - "runs/decompiled/com/corrodinggames/rts/game/units/custom/d/a.java:17"
  - "runs/decompiled/com/corrodinggames/rts/gameFramework/e.java:591"
  - "runs/decompiled/com/corrodinggames/rts/gameFramework/c.java:19"
  - "runs/decompiled/com/corrodinggames/rts/game/n.java:837"
  - "wiki/sources/m12-produce/produce-run.log"
  - "agent/src/rwbot/agent/BuildOptions.java"
  - "agent/src/rwbot/agent/Orders.java"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [game-mechanics, engine-internals]
---

# Build Actions: Two Families, Two Verbs, Five Gates

Placing a structure and producing a unit are the same mechanism in this engine — a list of actions hanging off a unit — and they are dispatched by two different verbs. Getting that wrong is silent in both directions, which is why this page exists.

## The two families

Every unit carries an action list. A builder's actions place buildings; a factory's produce units. They are told apart by one accessor: a placement action reports the type it would place, and a production action returns null for it and answers the engine's "makes something" predicate with true instead.[^1]

The build-waypoint path matches candidate actions on exactly that placement accessor, so a production action is invisible to it and ordering a factory that way is refused with `can not queue build`. Production dispatches on the action's own interned key instead, which is what the game's interface sends when the button is pressed.

Neither test alone finds both families, and the agent made each mistake in turn before settling on their union: filtering on the placement accessor loses every factory, filtering on the predicate loses every builder and reports the bot's own Builder as able to make nothing.[^9]

## What actually stops a production order

The order path applies five conditions, and only the first two say anything when they fail.

The command executor resolves the key to an action on each selected unit, logs `Could not find specialAction` if it cannot, then checks availability and logs `!isAvailable specialAction: … (action being skipped)` if that fails.[^2] Everything after that is silent. The unit's own apply hands off to its production queue,[^3] which runs four tests in one conjunction and returns null if any fails, without a word:[^4]

```java
if (s2.a(this.a, false) && s2.b(this.a)
        && (!w2.g() || this.a.bX.w() < this.a.bX.x())
        && w2.B().c(this.a)) {
```

Reading left to right: the action applies to the unit, it is available, the player is under the unit cap, and the cost is paid.

**The unit cap is the gate nothing else reveals.** `w()` and `x()` are the player's unit count and the game's configured cap; the engine names the count itself, printing `unitCountExcludingBuildingsIncludingQueued` when its cached figure disagrees with a recount.[^5] Because the count includes queued units, a factory with a full queue is at the cap before any of it has rolled out — and the refusal that follows is indistinguishable from an order that never arrived.

**The fifth condition is not a question.** `B().c(unit)` is the engine's check-then-charge helper: it tests affordability and, on success, deducts the cost in the same call.[^6] A diagnostic can read the first four gates and must not touch the fifth, so an order that passes every gate the agent can read may still be refused at the till. That is a real gap and it is stated rather than papered over.

The same trap sits one argument away in the first gate. `a(unit, false)` routes affordability through a pure read; `a(unit, true)` routes it through the charging helper.[^7] The agent pins the argument to false at every call site for that reason, and the pin is documented where the name is declared rather than at the call.

## Why the diagnostic runs on every order

The obvious economy — compute the gates only when something fails — cannot be written, because there is no observable failure to hang it on. The engine's two complaints go through a logger whose rate limiter is a static counter that is never reset: it prints four messages for the lifetime of the process and then returns early forever.[^8] So on a long run the engine is silent about both of the failures it nominally reports, and silent by construction about the other three.

What the agent does instead is read the four readable gates before dispatching, log them beside the action key, and **refuse to dispatch when one is closed** rather than sending a command the engine will drop.[^10] A closed gate is logged as an error naming which one; it is not thrown, because being at the unit cap or short of credits is an ordinary state of a game in progress and clears on its own. A missing action still throws, because that one cannot.

This is not a second opinion about the engine's rules. It is the same reads the engine is about to make, made a moment earlier where the answer can be written down.

## What the agent puts on the wire

Every sample carries what each owned unit can make, with a `placed` flag marking which verb applies, so the planner never infers the verb from a type's speed or name.[^11] Entities carry `complete` and `queued` alongside, because a building joins the roster the moment construction *starts*: a factory with an id, a position and a full option list can still be a shell, and a shell accepts a production order into a queue it never advances.

Measured end to end: the factory appeared at t=4.0s with `complete=false`, finished at t=21s, took the order, spent credits 4440 → 4289, and delivered the tank at t=29.5s.[^13] The dispatch itself is archived, naming the production class and the interned key.[^12]

Queue depth is what makes production legible in flight. The building reports `queued=1` for the whole of production and zero on the sample the unit appears, and the 45-sample timing for a Scout reproduces exactly across runs — which retired an earlier price-derived window entirely.[^14] Credits are no substitute: through a production the queue reports as active they read 3520 → 3588 → 3655 → 3723, *rising*, because income outpaced the drain.[^14]

## What this does not establish

The unit-cap gate is derived from the engine's own conjunction and its own log string, but it has not been *observed* closing — no run so far has approached the cap. When one does, the refusal will now name it instead of vanishing, which is the point; until then the gate is read correctly rather than proven.

Nothing here measures what happens when the fifth gate refuses an order the first four passed. That would need a run engineered to sit exactly at the boundary between the two affordability checks, which differ in which credit predicate they call.

[^1]: `runs/decompiled/com/corrodinggames/rts/game/units/a/s.java` — `y()` returns the placed type and defaults to null on the base; `g()` is abstract. `a/l.java` is the production family: it overrides `g()` to return true, and builds its key as `"u_" + as2.v()` in its constructor.
[^2]: `runs/decompiled/com/corrodinggames/rts/gameFramework/e.java:591` and `:595` — the two `c.a(...)` calls inside the `s.c(this.k)` branch of `e.k()`, the second guarded by `if (!((s)object2).b((am)object4))`.
[^3]: `runs/decompiled/com/corrodinggames/rts/game/units/d/i.java:144` — `public void a(s s2, boolean bl2) { this.z.a(s2, bl2, null, null); }`. The base `am.a(s, boolean, PointF, am)` forwards to the two-argument form, so the point a produce command carries is dropped for factories, which need none.
[^4]: `runs/decompiled/com/corrodinggames/rts/game/units/d/k.java:433` — the conjunction quoted above, inside `a(s, boolean, PointF, am)` at `:429`. The method returns null when it falls through.
[^5]: `runs/decompiled/com/corrodinggames/rts/game/n.java:837` and `:841` — `w()` returns `T.b` and `x()` returns `T.a`. The name comes from the engine's own consistency warning at `:715`, `"unitCountExcludingBuildingsIncludingQueued: " + this.T.b + "!=" + s2.b`; `T.a` is assigned from the game-wide setting `l.bB` at `:1862`.
[^6]: `runs/decompiled/com/corrodinggames/rts/game/units/custom/d/a.java:17` — `public boolean c(am am2) { if (this.b(am2)) { this.a(am2); return true; } return false; }`, where `a(am)` is the abstract consume. Its sibling `c(am, double)` at `:25` has the same shape.
[^7]: `runs/decompiled/com/corrodinggames/rts/game/units/a/s.java:194` — `if (bl2) { return this.B().c(am2, this.Q()); } return this.B().b(am2);`. The false branch reaches the pure affordability read at `custom/d/b.java:321`; the true branch reaches the charging helper.
[^8]: `runs/decompiled/com/corrodinggames/rts/gameFramework/c.java:19` — `public static void a(String string2) { if (++e == 5) { l.e("(Rate Limiting...)"); } if (e >= 5) { return; } l.e(string2); }`. `e` is a static int with no reset anywhere in the class.
[^9]: `agent/src/rwbot/agent/BuildOptions.java` — `ownedOptions`, whose filter is the union of the two tests, with the two failed single-test versions recorded in its comment.
[^10]: `agent/src/rwbot/agent/Orders.java` — `produce`, and `BuildOptions.gatesOf` / `BuildOptions.Gates.closed` for the readings and the refusal message.
[^11]: `agent/src/rwbot/agent/StateStream.java` — `optionRecord` and the `options` count on the frame record; decoded by `src/rw_bot/wire/state.py` into `BuildOption` ([[wire-contract-ndjson]]).
[^12]: `wiki/sources/m12-produce/produce-run.log:436` — `[rw-agent] produce: scout via action 'u_scout' on com.corrodinggames.rts.game.units.a.l [applies=true available=true locked=false]`. The three-gate summary is the format this log was captured under; a run captured now would carry the fourth.
[^13]: [synthesis] — a live probe over the agent channel: order a `landFactory`, watch the roster for it, poll `complete` and `queued` each sample at 0.25s, order a `c_tank` on the sample it first reports complete, and stop when the tank joins the roster. Recorded rather than archived because the procedure is the record and the run needs a live game; the figures are one execution of it.
[^14]: `wiki/sources/m12-produce/produce-timing.txt` — the third run, `RESULT produce_samples=45 appeared=['scout']` with `queued={213: 1, …}` at every step, against the same 45 samples in the first run. The rising credit column is in the same block.
