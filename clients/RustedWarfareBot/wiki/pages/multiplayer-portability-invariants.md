---
title: Multiplayer Portability Invariants
tags: [multiplayer, lockstep, architecture, contracts, dispatch]
related:
  - "[[harness-nodisplay]]"
  - "[[engine-name-oracle]]"
  - "[[runtime-split-java-agent-python-brain]]"
source_paths:
  - "wiki/sources/m0-probe/nodisplay-boot.log:183"
  - ".game/preferences.ini"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: medium
hubs: [bot-architecture, multiplayer]
---

# Multiplayer Portability Invariants

The bot targets single-player skirmishes against the built-in AI, which the engine supports directly through a difficulty setting and a single-player unit cap.[^1] Multiplayer is not a goal, but it is a plausible future one, and four design choices decide whether that future costs a weekend or a rewrite.

## Why single-player is not itself the risk

Rusted Warfare is a lockstep RTS: the network carries player *commands*, and every client runs its own copy of the simulation. The engine constructs a `CommandController` during init, before any map is loaded and regardless of game mode.[^2] Single-player is the same command path with zero peers.

Confidence here is `medium`, not `high`. The lockstep model is inferred from the engine's structure and from a networking configuration shaped like a command relay rather than state replication — one configurable TCP port, UDP off by default.[^3] It has not been confirmed by reading `CommandController` or observing a networked session, which is the test that would settle it. The invariants below cost nothing if the model is wrong, but the model is not yet verified.

## The four invariants

**1. Every action is a queued command, never a state mutation.** An agent inside the JVM can call a unit's position setter directly, and in single-player that works perfectly. In multiplayer it desyncs every peer immediately, because no other client saw a command that would produce that state. All orders therefore go through the game's own command queue — the same `CommandController` the engine builds at boot.[^2]

This is not a new rule. It is the sibling TankpitBot project's "executor is pure dispatch", adopted here before the incident rather than after one: there, executor-side validators silently vetoed planner decisions and produced a 26-second self-rejection deadlock.[^4]

**2. No order is assumed to take effect this tick.** Lockstep stamps a command to a future tick so every client executes it simultaneously. A planner written against zero latency works in single-player and rots invisibly the moment latency exists.[^5]

The defence is to make the assumption impossible to form: inject a synthetic command delay in single-player from the first day, so "issued" and "observed to have taken effect" are separate states in the planner's model from the start. The decision loop already runs decimated rather than per-tick, so the delay costs nothing structurally ([[runtime-split-java-agent-python-brain]]).

**3. Perception is filtered to what the player could legitimately see.** An in-process agent can read the whole simulation, including enemy units under fog — the engine precalculates and maintains per-team fog as part of map load.[^6] Against the built-in AI nobody is harmed, but every heuristic quietly grows a dependency on omniscience, and by the time fog matters the planner has been trained against a world model it cannot have.

The visibility filter therefore has exactly one owner, with the omniscient view available only as a distinct oracle channel used for scoring — measuring how accurate the bot's beliefs were, never feeding them. This mirrors the single-owner rule the sibling project reached for walkability, where a second downstream owner produced silent rejection loops.[^4] Rusted Warfare raises the stakes: ground, air, water and hover are four movement layers, which is four chances to grow a second owner.

**4. Tick gating is an accelerator, not the baseline.** Blocking the simulation until the bot has decided turns a realtime RTS into `step()` semantics and lets an evaluation run go as fast as the CPU allows. It is also impossible with real opponents, so the planner needs a realtime path with a time budget and gating must be a strategy the harness selects rather than an assumption baked into the tick loop ([[runtime-split-java-agent-python-brain]]).

## What multiplayer would still cost

These invariants make the bot's *decision and action* layers portable; they do not deliver multiplayer. Joining a lobby without a human, handling the connection lifecycle, and satisfying whatever handshake the protocol expects all remain outstanding — and the menu-driving problem is unsolved even for single-player today ([[harness-nodisplay]]).

There is also a non-technical boundary. Self-hosted play against the built-in AI, or against people who know they are playing a bot, is one thing; running a bot on public relay servers against strangers is a social and terms-of-service question. It stays out of scope until explicitly asked for ([[harness-nodisplay]]).

[^1]: `.game/preferences.ini:2` — `aiDifficulty:0`; `teamUnitCapSinglePlayer:1000` at `:116` and `teamUnitCapHostedGame:250` at `:115` are distinct keys, showing single-player and hosted play as separately configured modes of the same engine.
[^2]: `wiki/sources/m0-probe/nodisplay-boot.log:183` — "--Now loading:CommandController", emitted during engine init; the menu map load does not begin until `:261` ("--- Loading map ---").
[^3]: `.game/preferences.ini`, `[settings]` — `networkPort:5123` with `udpInMultiplayer:false` [synthesis]. A single configurable port with UDP off by default is consistent with a TCP command relay; it does not by itself prove the lockstep model.
[^4]: `clients/TankpitBot/wiki/pages/executor-rejection-loops.md` in the api monorepo, §"Resolution 2026-07-21 — the class is CLOSED: executor is pure dispatch", and §"Symptom" for the 2026-07-06 20:47:31 deadlock; the single-owner terrain rule is at `clients/TankpitBot/wiki/pages/terrain-composition.md`.
[^5]: [synthesis] — follows from the lockstep model asserted above; carries the same `medium` confidence and is falsified by the same test (reading `CommandController`'s dispatch path).
[^6]: `wiki/sources/m0-probe/nodisplay-boot.log:265` — "Setting up team fog.." during map load, preceded by "--Now loading:Precalculating map fog" at `:187`.
