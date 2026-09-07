---
title: Command Channel
tags: [architecture, wire, dispatch, threading, planner]
related:
  - "[[issuing-orders]]"
  - "[[building-structures]]"
  - "[[runtime-split-java-agent-python-brain]]"
  - "[[engine-entity-model]]"
source_paths:
  - "wiki/sources/m7-channel/planner-drove-a-live-game.txt:4"
  - "wiki/sources/m7-channel/planner-drove-a-live-game.txt:9"
  - "wiki/sources/m7-channel/world-sample-with-ids.ndjson:2"
  - "wiki/sources/m7-channel/scriptengine-drain.txt:2"
  - "agent/src/rwbot/agent/CommandChannel.java"
  - "agent/src/rwbot/agent/CommandRecord.java"
  - "src/rw_bot/control/channel.py"
  - "src/rw_bot/wire/command.py"
source_git_blobs:
  "wiki/sources/m7-channel/planner-drove-a-live-game.txt": "c588f77122b962a0dfb8800700f80958a128f08a"
  "wiki/sources/m7-channel/world-sample-with-ids.ndjson": "7c45f231f3ef47323cc6cf5d24bb1f5bc64d4b98"
  "wiki/sources/m7-channel/scriptengine-drain.txt": "f05e845cb2dc38e6556f36c16e965c5ee5f97838"
  "agent/src/rwbot/agent/CommandChannel.java": "8709bdbace0ebacaee2ef6908714c78e1b78ab96"
  "agent/src/rwbot/agent/CommandRecord.java": "bd4fc2f5ee3af50a3ba83cfe1370e18f82ec72ce"
  "src/rw_bot/control/channel.py": "e5c4521155fea78be4e037ca2e9f631d3a8b8443"
  "src/rw_bot/wire/command.py": "21f360c4728171fa52373d892483317f7c95b7c9"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture, engine-internals]
---

# Command Channel

Orders now originate in Python. A planner connects to a running game, reads real world state, chooses a subject from it, and issues an order the engine executes — closing the loop that [[issuing-orders]] and [[building-structures]] each proved one half of.

## Shape

One loopback socket carries both directions as newline-delimited JSON: world samples out, orders in. The agent listens and the planner connects, because the game is the long-lived process — a planner can attach, exit and reattach without disturbing the match.[^1]

Both directions are flat objects of scalar values. That constraint exists for the consumer: the Python reader runs under `disallow_any_expr`, where `json.loads` is unusable because its return is `Any` and suppressions are banned, so a format parseable strictly into typed fields is what lets the reader stay fully typed ([[runtime-split-java-agent-python-brain]]).

## Units are addressed by engine identity

The sample records both a roster index and an id, and only the id is a handle.[^2] Index is enumeration order: it renumbers the moment anything is built or dies, which in this game is constantly. The id is the engine's own object identity, assigned once at construction behind an "ID for GameObject is already set" guard and used by the engine for network identity ([[engine-entity-model]]).

Records also carry the readable type name, which is what makes selection possible at all — the planner picks a builder by asking for `"builder"`, not by trusting that position 1 is one.[^2]

## Backpressure is the design constraint

Samples are produced on the game thread and written by a separate thread through a bounded queue that drops its oldest entry when full.[^3] A planner that stops reading must never be able to stall the simulation, and blocking a socket write on the game thread is exactly how that would happen. Dropping stale world state is the right loss, because the next sample supersedes it.

Inbound orders take the mirror path: they arrive on the reader thread and are dispatched through the engine's own runnable queue, which appends under a lock and runs the work on the thread that marks itself the main script thread.[^4] Nothing in the channel touches the simulation directly.

## Rejection is loud in one direction and survivable in the other

A malformed order is reported with the offending line and the connection continues. That is not softening a failure: the planner is a separate process, and one bad line is its bug rather than grounds for dropping a live match. What must not happen is silence — an order that is accepted and quietly discarded looks exactly like an order that did nothing, and that confusion has already cost this project two runs ([[building-structures]]).

The encoder makes most of those lines unrepresentable rather than merely unlikely: a move that carries a build type is rejected before it is sent, as is a blank type name, a non-finite coordinate, or a type name containing a character the flat format cannot carry.[^5] The agent's parser rejects the same cases independently, so neither side is trusting the other.

## The run that closed the loop

A planner connected to a live sandbox and read one sample: three owned entities with ids, type names and positions.[^6] It selected the builder by type name, computed a destination from that builder's own coordinates, and sent a build order. Three samples later the roster gained a `landFactory`.[^7]

Nothing in that sequence was a constant. The subject came from the sample, the destination came from the subject, and the roster change is the engine's answer.[^7] The probe's tests hold that property directly: the same roster in a different order must yield the same choice, and a builder standing somewhere else must move the destination with it.[^5]

## What this does not make it

A planner that issues one order is not a player. There is no goal, no scoring, and no loop that reconsiders — the probe orders once and then watches.[^7] What exists now is the substrate a policy would run on: perceive, decide, act, observe the result, all in Python and all against a live match ([[runtime-split-java-agent-python-brain]]).

[^1]: `agent/src/rwbot/agent/CommandChannel.java:9` — a `ServerSocket` bound to the loopback address, accepting one planner at a time, with a reader thread inbound and a writer thread outbound.
[^2]: `wiki/sources/m7-channel/world-sample-with-ids.ndjson:2` — `{"kind":"entity","frame":854,"index":0,"id":213,"type":"commandCenter",...}`, with the builder at `:3` and the off-map placeholder at `:4`.
[^3]: `agent/src/rwbot/agent/CommandChannel.java:169` — `offer` drops the oldest entry and reports the drop when the outbox is full; `SelfTest.checkChannelBackpressure` asserts the queue stays bounded at its depth rather than trusting the comment.
[^4]: `wiki/sources/m7-channel/scriptengine-drain.txt:2` — `if (!mainScriptThreadMarked) { mainScriptThreadMarked = true; isMainScriptThread.set(true); }`, with the queue drained under `synchronized (arrayList)` at `:8`.
[^5]: `src/rw_bot/wire/command.py` — `move_order` and `build_order` validate before construction; `encode_build` rejects a type name containing a quote, backslash or newline. `agent/src/rwbot/agent/CommandRecord.java` rejects the same shapes on arrival, including a move carrying a build type.
[^6]: `wiki/sources/m7-channel/planner-drove-a-live-game.txt:4` — `id=214 builder at (4250.0, 2610.0)`, within the sample opened at `:1`.
[^7]: `wiki/sources/m7-channel/planner-drove-a-live-game.txt:9` — `frame 8013: ['builder', 'commandCenter', 'editorOrBuilder', 'landFactory']`, against the order issued at `:5`.
