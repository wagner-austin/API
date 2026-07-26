---
title: "Wire Contract — NDJSON World Stream"
tags: [wire, agent, planner, ndjson, typing, contract]
related:
  - "[[runtime-split-java-agent-python-brain]]"
  - "[[engine-entity-model]]"
  - "[[engine-tick-and-clock]]"
  - "[[issuing-orders]]"
  - "[[perception-visibility]]"
  - "[[mechanics-resource-pools]]"
source_paths:
  - "wiki/sources/m6-wire/world-sample.ndjson:1"
  - "wiki/sources/m6-wire/world-sample.ndjson:2"
  - "wiki/sources/m6-wire/world-sample.ndjson:20"
  - "wiki/sources/m6-wire/world-sample.ndjson:66"
  - "wiki/sources/m6-wire/world-sample.ndjson:131"
  - "agent/src/rwbot/agent/StateStream.java"
  - "src/rw_bot/control/channel.py"
  - "src/rw_bot/wire/ndjson.py"
  - "src/rw_bot/wire/state.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [bot-architecture]
---

# Wire Contract — NDJSON World Stream

The agent publishes what it sees as newline-delimited JSON and the planner decodes it into typed state. This is the first half of the two-process split to exist as running code on both sides ([[runtime-split-java-agent-python-brain]]).

## The format

Records are discriminated by `kind`. A `frame` record opens a sample and declares how many records of each following kind it carries; each `entity` record is one visible unit or building, carrying the engine id an order is dispatched against ([[issuing-orders]]), and each `pool` record is one visible resource pool ([[mechanics-resource-pools]]).[^1][^2][^7]

```
{"kind":"frame","frame":3569,"clock_ms":11972,"visible":18,"pools":46,"credits":4319}
{"kind":"entity","frame":3569,"index":0,"id":207,"type":"commandCenter","class":"…units.d.e","x":410.0,"y":990.0,"team":5,"mine":false,"hp":4000.0,"max_hp":4000.0}
{"kind":"pool","frame":3569,"index":0,"tile_x":115,"tile_y":6,"x":2310.0,"y":130.0}
```

`frame` and `clock_ms` are the engine's own counters, read from the same fields measured at ~300 Hz and 1 kHz ([[engine-tick-and-clock]]). Fields are limited to what has been verified against the engine rather than everything reachable ([[engine-entity-model]]).

The counts have grown twice, and each time a count rather than a terminator. `owned` became `visible` when the stream widened past the player's own units ([[perception-visibility]]), and `pools` joined it when terrain that no entity list can express had to reach the planner. A count in the opening record is what lets the reader know a sample is complete before parsing all of it, which is the property the channel frames on.[^8]

## Why every line is flat

**Every record is a flat object of scalars — no nesting, no arrays, no null.** That is a constraint on the producer, and it exists because of a constraint on the consumer.[^5]

The planner is type-checked under `disallow_any_expr`, and `json.loads` returns `Any`. Every expression touching that value is an error, `isinstance` narrowing does not rescue it because the offending expression is the call itself, and suppressions are banned by the project's standards. The standard library is therefore unusable for reading this stream.[^5]

That leaves writing a reader, and the cost of a reader is set by the grammar it must accept. A flat scalar object is small enough to parse strictly in one pass; general JSON is not. Constraining the format on the Java side is what buys a fully typed consumer on the Python side — the two decisions are one decision.[^4][^5]

## What the reader refuses

Nothing is coerced and nothing is repaired. `"4250"` stays a string, a duplicate key is an error rather than a last-one-wins merge, and a nested object is rejected as a contract break rather than half-read.[^5] Two records on one line fail on the trailing content, which is the corruption newline-delimiting exists to prevent.[^5]

One check is worth more than the rest: a sample carrying fewer records than it declared is rejected, separately for each kind.[^6] A truncated capture is the ordinary result of reading the file while the agent is still writing it, and a planner that acted on a world it could not fully see would make exactly the class of decision this contract exists to make safe — a half-read pool list would have the bot build on a pool it had already taken.

## Threading

The sample is rendered on the game thread and written off it. A position read mid-tick can be torn, so the read is posted to the engine's own queue; disk I/O on the simulation thread would pace the game by disk latency, so the write is not.[^4] The agent's queue enqueues without running, so the render is awaited explicitly rather than read back immediately — reading it back is what the first attempt did, and it produced an empty sample every time.[^4]

## Replay falls out of it

A JSONL stream is itself a corpus. Decoding is a pure function of the lines it is given and reads no file, so the same code serves a live tail and an archived replay with no branch between them.[^6] The tests decode the real capture archived here rather than a fixture written to match the parser, and they cross-check it: the frame and clock deltas across two samples reproduce the ~300 Hz measured independently.[^3][^6]

[^1]: `wiki/sources/m6-wire/world-sample.ndjson:1` — the opening frame record of a real capture from a live headless skirmish, declaring 18 visible entities and 46 pools.
[^2]: `wiki/sources/m6-wire/world-sample.ndjson:2` — the entity record at index 0, an opposing team's `commandCenter` carrying `"mine":false`.
[^3]: `wiki/sources/m6-wire/world-sample.ndjson:66` and `:131` — the second and third frame records, 3866 at 12965 ms and 4165 at 13961 ms; 299 frames across 996 ms is 300.2 per second.
[^7]: `wiki/sources/m6-wire/world-sample.ndjson:20` — the first pool record, tile (115, 6) at world (2310.0, 130.0).
[^8]: `src/rw_bot/control/channel.py` — `_complete_or_none` reads the opening record's counts and waits for `1 + visible + pools` lines before decoding.
[^4]: `agent/src/rwbot/agent/StateStream.java` — the producer, with the flatness constraint and its rationale in the class javadoc; the render-on-game-thread and await are in `Premain.writeSample`.
[^5]: `src/rw_bot/wire/ndjson.py` — the strict reader, with the `disallow_any_expr` constraint recorded in the module docstring and one traceable code per rejection (`RW-NDJSON-001` … `-006`).
[^6]: `src/rw_bot/wire/state.py` — `decode_samples` folds records into samples and enforces the declared count (`RW-WIRE-003`); `encode_sample` round-trips, which is what makes a decoded corpus re-emittable as a fixture.
