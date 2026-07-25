---
title: Issuing Orders
tags: [engine, commands, dispatch, agent, threading]
related:
  - "[[engine-entity-model]]"
  - "[[engine-tick-and-clock]]"
  - "[[multiplayer-portability-invariants]]"
  - "[[runtime-split-java-agent-python-brain]]"
source_paths:
  - "wiki/sources/m5-order/controller-create-and-enqueue.txt:1"
  - "wiki/sources/m5-order/controller-create-and-enqueue.txt:8"
  - "wiki/sources/m5-order/command-setters.txt:1"
  - "wiki/sources/m5-order/builtin-ai-order-idiom.txt:3"
  - "wiki/sources/m5-order/scriptengine-update.txt:2"
  - "wiki/sources/m5-order/order-accepted-unit-moved.txt:9"
  - "wiki/sources/m5-order/order-accepted-building-did-not-move.txt:4"
  - "wiki/sources/m5-order/controller-delegate.txt:2"
  - "agent/src/rwbot/agent/Orders.java"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [engine-internals, bot-architecture]
---

# Issuing Orders

The bot can now play. A unit was ordered across the map through the engine's own command queue and pathfound to the destination, which closes the last of the three prerequisites [[engine-name-oracle]] named.

## The path

Three calls, in this order:[^1]

```java
e cmd = engine.cf.a(team);   // construct and enqueue
cmd.a(unit);                 // add a subject
cmd.a(x, y);                 // set a destination
```

Nothing dispatches the command afterwards. `cf.a(team)` enqueues on construction, and the tick drains the queue — `cf.c()` is called from the simulation tick ([[engine-tick-and-clock]]).

`a(n)` is a one-line delegate: its whole body is `return this.b(n2);`.[^9] The construction, timestamping and enqueue described below all live in `b(n)`, which is why the citations name that method rather than the one the code block calls.

This is the built-in AI's own idiom, not a reconstruction of one: the engine's AI issues orders with exactly this sequence.[^2] Using the same route is what makes the bot's input the same class of thing a player's input is.

## Two details that matter for multiplayer

The command is stamped with the millisecond clock at construction — `e2.d = l2.by`, the same `by` counter measured at 1 kHz.[^3] Commands are timestamped, which is what a lockstep relay needs and what the latency invariant assumes ([[multiplayer-portability-invariants]]).

The enqueue also forks on network role: one branch calls a server-side check before adding, the other adds directly.[^4] The client/server distinction is real in the enqueue path rather than something layered on later, which is the strongest evidence yet that dispatch through this queue is the multiplayer-legal route.

## Threading

Commands land in a plain `ArrayList` that the tick drains, so writing from a probe thread would race the simulation. The engine already answers this: `ScriptEngine.addRunnableToQueue` appends under a lock, and `ScriptEngine.update` — which marks itself the main script thread on first entry — runs the queued work.[^5] The agent posts every engine touch through it, reads included, because a position sampled mid-tick can be torn just as easily as a write can corrupt.

## What "it worked" required proving

The first order was issued without error and did nothing. The subject was `units.d.e`, whose drawables are `base` — the Command Center. It was ordered 240 units east and sat still through three samples.[^6]

That is the failure the probe was shaped to catch. An order that is accepted, queued, and silently dropped is indistinguishable from a successful one unless position is sampled afterwards, so the sequence is sample, order, sample again at increasing offsets.[^6] Had the check been "no exception thrown", the wrong conclusion would have been recorded as a success — the same shape as the two wrong entity identifications in [[engine-entity-model]].

Re-targeted at the Builder (`units.e.b`, drawables `builder`) the same code moved it:[^7]

| sample | position |
|---|---|
| ordered from | (4250.0, 2610.0) |
| target | (4550.0, 2610.0) |
| t+2s | (4335.7, 2610.0) |
| t+5s | (4474.1, 2628.8) |
| t+10s | (4547.5, 2611.8) |

The y excursion at t+5 and its correction by t+10 is pathfinding around an obstacle, which also rules out the reading that a field was written directly.[^7] It arrived 2.5 world units from the destination and stopped.

`e.a(float, float)` is therefore a move-to-point order, settled by observation rather than by reading the obfuscated target-kind enum, whose constants are single letters.[^7] Observation outranks static reading here by the citation hierarchy `SCHEMA.md` sets, and it was also simply cheaper.

## Selection is not the agent's job

The agent publishes the owned-entity roster and dispatches against an index into it. It holds no mobility predicate, because deciding which unit is worth moving — or can move at all — is planner work, and a guess embedded in the dispatch layer is precisely the decision logic the agent must not carry ([[runtime-split-java-agent-python-brain]]).

The roster is short at skirmish start: the Command Center, one Builder, and an entity parked at (-1000, -1000).[^7] That third one is unexplained and is a lead, not a finding.

## Drift

Every name above is obfuscated and moves between releases. `Orders.verifyBindings` resolves all of them — seven classes, four fields and five method signatures — against the jar with no game running, and `make check` fails on any that moved.[^8] After a game update the failure names the whole broken surface at once rather than the first item.

[^1]: `agent/src/rwbot/agent/Orders.java` — `moveTo` performs exactly these three reflective calls, resolving `a(n)` on the controller and `a(y)` / `a(float,float)` on the command by parameter type.
[^2]: `wiki/sources/m5-order/builtin-ai-order-idiom.txt:3` — `com.corrodinggames.rts.gameFramework.e e2 = l2.cf.a(this);` followed by `e2.a(y2);` and `e2.a(c2);` in the engine's AI base class.
[^3]: `wiki/sources/m5-order/controller-create-and-enqueue.txt:8` — `e2.d = l2.by;` inside `public e b(n n2)`, which is at `:1`.
[^4]: `wiki/sources/m5-order/controller-create-and-enqueue.txt:1` — `b(n)` branches on `l2.bX.B`, calling `e2.l()` and adding to one list on the false branch and adding to another on the true branch.
[^5]: `wiki/sources/m5-order/scriptengine-update.txt:2` — `if (!mainScriptThreadMarked) { mainScriptThreadMarked = true; isMainScriptThread.set(true); }`, with the drain under `synchronized (arrayList)` at `:8`.
[^6]: `wiki/sources/m5-order/order-accepted-building-did-not-move.txt:4` — three samples at t+2s, t+5s and t+10s all reporting `(4250.0, 2550.0)` after an order to `(4490.0, 2550.0)` recorded at `:2`.
[^7]: `wiki/sources/m5-order/order-accepted-unit-moved.txt:9` — `t+2s ... at (4335.7466, 2610.003)`, with t+5s at `:10` and t+10s at `:11`; the roster of three owned entities is at `:2`–`:4`.
[^8]: `agent/src/rwbot/agent/Orders.java` — `verifyBindings` calls `checkClass` seven times (entity, team, orderable, command, controller, scripts, tree), `checkField` four times (entity list, owner, x, y) and `checkMethod` five times (controller `a(team)`, command `a(orderable)`, command `a(float,float)`, `getInstance`, `addRunnableToQueue`). It returns one message per unresolved name and is asserted empty by `SelfTest.checkOrderBindings`, which `make check` runs via `agent-selftest`.
[^9]: `wiki/sources/m5-order/controller-delegate.txt:2` — `return this.b(n2);`, the entire body of `public e a(n n2)` on `gameFramework.c`.
