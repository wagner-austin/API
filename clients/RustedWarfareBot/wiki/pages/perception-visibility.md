---
title: "Perception: Visible Entities, Economy and Health"
tags: [perception, wire, fog, legitimacy, planner]
related:
  - "[[wire-contract-ndjson]]"
  - "[[engine-entity-model]]"
  - "[[multiplayer-portability-invariants]]"
  - "[[mechanics-unit-catalogue]]"
  - "[[mechanics-resource-pools]]"
source_paths:
  - "wiki/sources/m9-perception/perception-widened.txt:9"
  - "wiki/sources/m9-perception/perception-widened.txt:11"
  - "wiki/sources/m9-perception/perception-widened.txt:14"
  - "wiki/sources/m9-perception/perception-widened.txt:15"
  - "wiki/sources/m9-perception/perception-widened.txt:5"
  - "wiki/sources/m11-pools/pool-build-run.log:402"
  - "agent/src/rwbot/agent/Perception.java"
source_git_blobs:
  "wiki/sources/m9-perception/perception-widened.txt": "8cc28b29812e235e24cecd7835059404a41970fc"
  "wiki/sources/m11-pools/pool-build-run.log": "d661b6813fdcc17b1cdc08da7fc390fe22ce67b6"
  "agent/src/rwbot/agent/Perception.java": "78629ad554596a6fd28fbd245c37a57a6b1e4743"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: medium
hubs: [bot-architecture, engine-internals]
---

# Perception: Visible Entities, Economy and Health

The world stream carried only the player's own units, their positions and their identities. A planner reading that could not tell whether it could afford anything, could not see an opponent, and could not distinguish a healthy unit from a dying one. It now carries enemies, credits and hit points.[^1][^3][^4]

## Asking the engine what is visible, not reading the map

The obvious implementation is wrong. `am.bE` is the master entity list and holds every unit on the map, so enumerating it directly would hand the bot perfect information and stop it playing the game a human plays ([[multiplayer-portability-invariants]]).

The engine already has the right function. `am.d(n)` takes a player and fog-tests the entity's own cell against *that player's* fog grid, returning false when the cell reads as hidden; own units short-circuit before the test.[^6] Perception calls it per entity rather than reasoning about fog itself, so whatever the engine would show a human in a given mode is exactly what the bot sees. That is a legitimacy property obtained by delegation rather than by discipline.

Its name is checked by the drift guard alongside the field names.[^6] Losing that binding would not break a build — it would silently make the bot omniscient, which is worse than a crash because nothing would look wrong.

## What a sample now carries

A frame record reports the visible-entity count and the player's credit balance; each entity adds its owning team, whether it is ours, and current and maximum hit points.[^1][^4][^5] Credits are floored from the engine's `double` to an `int`, which is the affordability-safe direction: it can under-report what is available but never over-report it.

Perception went from 3 entities to 15 on the same map and moment.[^1] Across three samples the visible count rises 15 → 18 → 19 as opposing AI players build, and credits rise 4,121 → 4,283 → 4,445.[^1][^2] Both are live values rather than the 4,000 the engine initialises: the stream is reporting a world in motion, which is the minimum a planner needs to react to anything.

## What this does not establish

**Fog filtering is not demonstrated, and now the reason is known rather than guessed.** The map sets up team fog[^7], yet all four opposing teams are visible in every sample.[^1] This page previously offered "`-sandbox` does not apply fog to the local player" as a plausible reading and said plainly that it was a reading rather than a finding. It has since been settled by asking the engine directly: the agent reads the map's fog-enabled flag and the player's fog grid and reports what it finds, and on this map the answer is **fog disabled**.[^8]

That is worth having as a permanent signal rather than a one-off check, which is why it is logged on every map scan. Both visibility tests — the entity one and the tile one that resource pools use ([[mechanics-resource-pools]]) — short out silently when there is no fog, and silence is the problem: a run in which everything happened to be visible is otherwise indistinguishable from a run in which the filter was working, and only one of those tells you the bot is playing fairly.

The consequence is bounded and unchanged. The *mechanism* is legitimate by construction, because it is the engine's own per-player test in both cases. The *behaviour under fog* remains untested, so the claim "the bot sees only what a player sees" is supported by how the code is written and by the flag it consults, not by an observation of anything being hidden. Establishing that still needs a mode where the local player has a fog grid, and a capture showing an entity present in `am.bE` and absent from the stream.

Confidence on this page is therefore still `medium`, and it is the visibility claim specifically that holds it there — every other claim here is observed in a capture.[^1] What changed is that the gap is now measured rather than suspected. The invariant it serves is stated in [[multiplayer-portability-invariants]].

[^1]: `wiki/sources/m9-perception/perception-widened.txt:9` — `{"kind":"frame","frame":1348,"clock_ms":4555,"visible":15,"credits":4121}`, the first of three samples.
[^2]: `wiki/sources/m9-perception/perception-widened.txt:11` — the third sample at `visible:19, credits:4445`, with the second at `:10` reading `visible:18, credits:4283`.
[^3]: `wiki/sources/m9-perception/perception-widened.txt:14` — an owned entity carrying `"team":0,"mine":true,"hp":4000.0,"max_hp":4000.0`.
[^4]: `wiki/sources/m9-perception/perception-widened.txt:15` — an opposing entity at `"team":5,"mine":false`, at `(410.0, 990.0)` against our base at `(4250.0, 2550.0)`.
[^5]: `agent/src/rwbot/agent/StateStream.java` — the record shapes; hit points come from `Orders.healthOf`, which reads the pair the engine itself divides for a health fraction (`cu / cv`).
[^6]: `agent/src/rwbot/agent/Perception.java` — `visibleEntities` calls the pinned `am.d(n)` per entity; the name lives in `EngineNames.VISIBLE_TO` and `EngineNames.verifyBindings` checks that method against the jar so a moved name fails at `make check`.
[^7]: `wiki/sources/m9-perception/perception-widened.txt:5` — `Setting up team fog..` from the engine log of the same run. The line is emitted during map load whether or not the map then has fog, which is why it was never evidence either way.
[^8]: `wiki/sources/m11-pools/pool-build-run.log:402` — `[rw-agent] fog: DISABLED on this map -- every visibility test passes`, emitted beside the map scan at `:401`. Produced by `MapTiles.describeFog`, which reads the map's fog-enabled flag and the player's fog grid through the same pinned-name machinery as everything else.
