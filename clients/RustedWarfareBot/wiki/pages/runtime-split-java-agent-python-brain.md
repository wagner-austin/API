---
title: "Runtime Split — Java Agent, Python Brain"
tags: [architecture, decision, agent, ipc, language]
related:
  - "[[multiplayer-portability-invariants]]"
  - "[[harness-nodisplay]]"
  - "[[engine-name-oracle]]"
source_paths:
  - ".game/fallback64.bat"
  - ".game/preferences.ini"
  - "wiki/sources/m0-probe/nodisplay-boot.log:27"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [bot-architecture]
---

# Runtime Split — Java Agent, Python Brain

The bot is two processes: a small Java agent inside the game's JVM, and a Python planner outside it, connected by newline-delimited JSON over loopback.[^1] This page records why, and the constraints that follow ([[multiplayer-portability-invariants]]).

## The agent is Java because it has no choice

The game runs on a bundled OpenJDK 13, launched as `com.corrodinggames.rts.java.Main` over `game-lib.jar` plus `libs/*`.[^1] Anything reading live simulation objects must live in that JVM, so the agent is a javaagent and targets Java 11 bytecode for headroom under the bundled runtime.[^1]

Plain Java rather than Kotlin: the agent loads into the game's own classloader beside obfuscated 1.15 classes, so every additional runtime dependency is a conflict surface, and its job is small — hook the tick, serialise state, dispatch orders. It carries no decision logic, per the pure-dispatch invariant in [[multiplayer-portability-invariants]].

## The brain is Python because the discipline already exists

Both sibling clients in this monorepo are Poetry packages under a written standard — strict typing with no `Any`, 100% statement and branch coverage, `_test_hooks` dependency injection, no mocks — and the shared `libs/` they consume are Python.[^2] Adopting that standard costs nothing; re-deriving it in another language costs the standard itself, which is the part that took two bots to learn (see [[multiplayer-portability-invariants]] for one rule that arrived the expensive way).

TypeScript was the serious alternative and lost narrowly. Its type system is genuinely better than mypy's, and it would collapse the wire contract into one definition shared with a future MCP surface. Against that: it cannot consume the Python `libs/`, and grid-shaped work — threat maps and distance fields over a 170×170 tile board — is numpy's home ground, which is C speed rather than interpreted.[^3] Runtime speed was not the deciding factor in either direction, for the reason in the next section.

## Decisions are decimated, so IPC cost is not the ceiling

The planner decides at roughly 4–10 Hz while the agent free-runs the simulation between decisions. An RTS does not need per-tick decisions; that is a twitch-game requirement. This is what keeps a cross-process brain viable and what keeps tick-gated batch evaluation fast enough to be worth running — gating happens at decision boundaries, not simulation ticks, which is the fourth invariant in [[multiplayer-portability-invariants]].

The wire format is newline-delimited JSON, chosen for replayability before performance: a JSONL stream is itself a corpus, so a captured session replays through the planner offline with no game running.[^6] msgpack is the fallback if profiling ever demands it.

## Per-tick reaction goes through standing orders, not the agent

Some behaviour genuinely needs to respond every tick, and it must not live in the agent — decision logic there would violate pure dispatch and desync any future multiplayer session ([[multiplayer-portability-invariants]]).

The engine already solves this. Attack-move, patrol, guard and rally-point are per-tick behaviours the game executes on the player's behalf, each bound to its own key action.[^4] The planner selects a standing order; the engine runs it at tick rate. This is the same mechanism a human player uses, so it is multiplayer-legal by construction.

## Why both processes stay on the host

The game needs an OpenGL context even headless — Slick opens a 10×10 display and creates framebuffer objects during boot.[^5] On Windows that is free. Containerising it would mean Xvfb plus the pinned game tree inside an image, so the bot process stays on the desktop next to `.game/` regardless of where its source lives ([[harness-nodisplay]]).

## What this decision does not lock in

The brain is replaceable. It speaks a documented line protocol to the agent, so swapping languages is a rewrite of one process against a fixed contract rather than an architecture change — the split is itself the hedge against the language call being wrong ([[multiplayer-portability-invariants]]).

One future carve-out is named deliberately: if the planner grows a search kernel — rollouts, MCTS, or a custom pathfinder across the four movement layers — that is a compiled hot loop callable from Python, not a reason to rewrite the planner. Revisit on a profile, not on principle ([[harness-nodisplay]]).

[^1]: `.game/fallback64.bat` — `jvm64\bin\java -Xmx1000M … -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main`; `.game/jvm64/bin/java.exe -version` reports "openjdk version 13 2019-09-17".
[^2]: `clients/README.md` in the api monorepo — "Each client is a standalone Poetry package with strict typing, 100% test coverage, and event-driven architecture", listing DiscordBot and TankpitBot; shared libraries `monorepo-guards`, `platform-core`, `platform-discord`, `platform-workers`.
[^3]: `wiki/sources/m0-probe/nodisplay-boot.log:264` [synthesis] — "Map size: 170, 170" for the menu map, establishing the order of magnitude for grid-shaped planner state; larger skirmish maps exceed it.
[^4]: `.game/preferences.ini`, `[keys]` section — `attack_move`, `patrol`, `guard_unit`, `stop`, `action_set_rally` are each bound key actions, confirming they are engine-side standing behaviours rather than one-shot commands.
[^5]: `wiki/sources/m0-probe/nodisplay-boot.log:27` — "INFO:Starting display 10x10", following the skipped display-mode call at `:20`.
[^6]: `clients/TankpitBot/README.md` in the api monorepo, §"Replay Bot" — `scripts.replay_bot` "loads a captured WebSocket session and replays it offline through the protocol decoders and AI planner tick-by-tick … without a live browser"; the same shape applies here with the agent stream in place of the WebSocket capture.
