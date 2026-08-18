---
title: Multiplayer Portability Invariants
tags: [multiplayer, lockstep, architecture, contracts, dispatch]
related:
  - "[[harness-nodisplay]]"
  - "[[engine-name-oracle]]"
  - "[[runtime-split-java-agent-python-brain]]"
  - "[[issuing-orders]]"
  - "[[perception-visibility]]"
source_paths:
  - "wiki/sources/m0-probe/nodisplay-boot.log:183"
  - "wiki/sources/m5-order/controller-create-and-enqueue.txt:1"
  - "wiki/sources/m5-order/controller-create-and-enqueue.txt:8"
  - ".game/preferences.ini"
source_git_blobs:
  "wiki/sources/m0-probe/nodisplay-boot.log": "c41e035d12ee89b66455389031be9fef55cf0b44"
  "wiki/sources/m5-order/controller-create-and-enqueue.txt": "18815aa9d9c842a0c3b601c8e5715aefadf2ffa5"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture, multiplayer]
---

# Multiplayer Portability Invariants

The bot targets single-player skirmishes against the built-in AI, which the engine supports directly through a difficulty setting and a single-player unit cap.[^1] Multiplayer is not a goal, but it is a plausible future one, and four design choices decide whether that future costs a weekend or a rewrite.

## Why single-player is not itself the risk

Rusted Warfare is a lockstep RTS: the network carries player *commands*, and every client runs its own copy of the simulation. The engine constructs a `CommandController` during init, before any map is loaded and regardless of game mode.[^2] Single-player is the same command path with zero peers.

This page carried `medium` confidence while the model was inferred from the engine's shape and a networking configuration that looked like a command relay rather than state replication — one configurable TCP port, UDP off by default.[^3] It named its own falsification test: read `CommandController`'s dispatch path. That has since been done, and the dispatch path confirms both halves of the model ([[issuing-orders]]).

Every command is stamped with the millisecond clock at construction — `e2.d = l2.by` — which is what a relay needs in order to schedule execution consistently across peers.[^7] And the enqueue forks on network role: one branch runs a server-side check before adding, the other adds directly.[^8] A single-player engine with no notion of peers would need neither. Confidence is now `high` for the model, and the four invariants below are unchanged, because they were written to be correct either way.

## The four invariants

**1. Every action is a queued command, never a state mutation.** An agent inside the JVM can call a unit's position setter directly, and in single-player that works perfectly. In multiplayer it desyncs every peer immediately, because no other client saw a command that would produce that state. All orders therefore go through the game's own command queue — the same `CommandController` the engine builds at boot.[^2]

This is not a new rule. It is the sibling TankpitBot project's "executor is pure dispatch", adopted here before the incident rather than after one: there, executor-side validators silently vetoed planner decisions and produced a 26-second self-rejection deadlock.[^4]

**2. No order is assumed to take effect this tick.** Lockstep stamps a command to a future tick so every client executes it simultaneously — and the stamp is now observed rather than assumed.[^7] A planner written against zero latency works in single-player and rots invisibly the moment latency exists.[^5]

The defence is to make the assumption impossible to form: inject a synthetic command delay in single-player from the first day, so "issued" and "observed to have taken effect" are separate states in the planner's model from the start. The decision loop already runs decimated rather than per-tick, so the delay costs nothing structurally ([[runtime-split-java-agent-python-brain]]).

**3. Perception is filtered to what the player could legitimately see.** An in-process agent can read the whole simulation, including enemy units under fog — the engine precalculates and maintains per-team fog as part of map load.[^6] Against the built-in AI nobody is harmed, but every heuristic quietly grows a dependency on omniscience, and by the time fog matters the planner has been trained against a world model it cannot have.

The visibility filter therefore has exactly one owner, with the omniscient view available only as a distinct oracle channel used for scoring — measuring how accurate the bot's beliefs were, never feeding them. This mirrors the single-owner rule the sibling project reached for walkability, where a second downstream owner produced silent rejection loops.[^4] Rusted Warfare raises the stakes: ground, air, water and hover are four movement layers, which is four chances to grow a second owner.

This one stopped being theoretical. The state stream carries every visible entity rather than only the player's, so ownership is now a per-entity fact the planner must consult; without it an opponent's structure advances the bot's own build plan ([[perception-visibility]], [[policy-loop]]).

**4. Tick gating is an accelerator, not the baseline.** Blocking the simulation until the bot has decided turns a realtime RTS into `step()` semantics and lets an evaluation run go as fast as the CPU allows. It is also impossible with real opponents, so the planner needs a realtime path with a time budget and gating must be a strategy the harness selects rather than an assumption baked into the tick loop ([[runtime-split-java-agent-python-brain]]).

## What multiplayer would still cost

These invariants make the bot's *decision and action* layers portable; they do not deliver multiplayer. Joining a lobby without a human, handling the connection lifecycle, and satisfying whatever handshake the protocol expects all remain outstanding. Menu-driving is no longer among them: `-sandbox` reaches a live skirmish with no human, and `Root.hostStart(boolean)` sits in the same unobfuscated script surface that provided it ([[harness-nodisplay]]).

There is also a non-technical boundary. Self-hosted play against the built-in AI, or against people who know they are playing a bot, is one thing; running a bot on public relay servers against strangers is a social and terms-of-service question. It stays out of scope until explicitly asked for ([[harness-nodisplay]]).

[^1]: `.game/preferences.ini:2` — `aiDifficulty:0`; `teamUnitCapSinglePlayer:1000` at `:116` and `teamUnitCapHostedGame:250` at `:115` are distinct keys, showing single-player and hosted play as separately configured modes of the same engine.
[^2]: `wiki/sources/m0-probe/nodisplay-boot.log:183` — "--Now loading:CommandController", emitted during engine init; the menu map load does not begin until `:261` ("--- Loading map ---").
[^3]: `.game/preferences.ini`, `[settings]` — `networkPort:5123` with `udpInMultiplayer:false` [synthesis]. A single configurable port with UDP off by default is consistent with a TCP command relay; it does not by itself prove the lockstep model.
[^4]: `clients/TankpitBot/wiki/pages/executor-rejection-loops.md` in the api monorepo, §"Resolution 2026-07-21 — the class is CLOSED: executor is pure dispatch", and §"Symptom" for the 2026-07-06 20:47:31 deadlock; the single-owner terrain rule is at `clients/TankpitBot/wiki/pages/terrain-composition.md`.
[^5]: `wiki/sources/m5-order/controller-create-and-enqueue.txt:1` [synthesis] — follows from the lockstep model, which the dispatch path now confirms; see [^7] and [^8].
[^7]: `wiki/sources/m5-order/controller-create-and-enqueue.txt:8` — `e2.d = l2.by;` inside `public e b(n n2)`, stamping every command with the engine's millisecond clock at construction.
[^8]: `wiki/sources/m5-order/controller-create-and-enqueue.txt:1` — `b(n)` branches on `l2.bX.B`, calling the server-side check `e2.l()` and adding to one list on one arm, adding directly to another list on the other.
[^6]: `wiki/sources/m0-probe/nodisplay-boot.log:265` — "Setting up team fog.." during map load, preceded by "--Now loading:Precalculating map fog" at `:187`.
