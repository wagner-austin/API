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
  - "wiki/sources/m6-wire/world-sample.ndjson:21"
  - "wiki/sources/m6-wire/world-sample.ndjson:67"
  - "wiki/sources/m6-wire/world-sample.ndjson:72"
  - "wiki/sources/m6-wire/world-sample.ndjson:190"
  - "wiki/sources/m6-wire/world-sample.ndjson:379"
  - "agent/src/rwbot/agent/BuildOptions.java"
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

Records are discriminated by `kind`. A `frame` record opens a sample and declares how many records of each following kind it carries; each `entity` record is one visible unit or building, carrying the engine id an order is dispatched against ([[issuing-orders]]); each `pool` record is one visible resource pool ([[mechanics-resource-pools]]); and each `option` record is one thing an owned unit can currently make.[^1][^2][^7][^9]

```
{"kind":"frame","frame":3725,"clock_ms":12488,"visible":19,"pools":46,"options":123,"credits":4333}
{"kind":"entity","frame":3725,"index":0,"id":207,"type":"commandCenter","class":"…units.d.e","x":410.0,"y":990.0,"team":5,"mine":false,"hp":4000.0,"max_hp":4000.0}
{"kind":"pool","frame":3725,"index":0,"tile_x":115,"tile_y":6,"x":2310.0,"y":130.0}
{"kind":"option","frame":3725,"index":5,"unit_id":214,"produces":"landFactory","action":1,"placed":true,"available":true}
```

`frame` and `clock_ms` are the engine's own counters, read from the same fields measured at ~300 Hz and 1 kHz ([[engine-tick-and-clock]]). Fields are limited to what has been verified against the engine rather than everything reachable ([[engine-entity-model]]).

The counts have grown three times, and each time a count rather than a terminator. `owned` became `visible` when the stream widened past the player's own units ([[perception-visibility]]); `pools` joined it when terrain that no entity list can express had to reach the planner; and `options` joined when what a unit can make turned out to be answerable only by asking the unit. A count in the opening record is what lets the reader know a sample is complete before parsing all of it, which is the property the channel frames on.[^8]

## Options are addressed by unit, not by type

An `option` names the **producing unit's engine id**, not its type, so a planner reading one needs no second table to get from "I want a landFactory" to "order unit 214".[^9] That matters because the answer is genuinely per unit rather than per type: the engine's availability predicate takes the unit as its argument.

`placed` is the engine's own distinction between the two verbs, not a guess from the produced type's speed. A structure is put at a position the planner chooses; a unit rolls out of the building that made it, so a produce order carries no coordinate at all.[^10]

The field exists because reading it wrong is silent. The predicate that looked like "makes something" is false for a builder's structure actions and true for a factory's unit actions — it means "produced without placement" — so using it as a filter drops every structure in the game and reports the bot's own Builder as able to make nothing.[^9]

## Why every line is flat

**Every record is a flat object of scalars — no nesting, no arrays, no null.** That is a constraint on the producer, and it exists because of a constraint on the consumer.[^5]

The planner is type-checked under `disallow_any_expr`, and `json.loads` returns `Any`. Every expression touching that value is an error, `isinstance` narrowing does not rescue it because the offending expression is the call itself, and suppressions are banned by the project's standards. The standard library is therefore unusable for reading this stream.[^5]

That leaves writing a reader, and the cost of a reader is set by the grammar it must accept. A flat scalar object is small enough to parse strictly in one pass; general JSON is not. Constraining the format on the Java side is what buys a fully typed consumer on the Python side — the two decisions are one decision.[^4][^5]

## What the reader refuses

Nothing is coerced and nothing is repaired. `"4250"` stays a string, a duplicate key is an error rather than a last-one-wins merge, and a nested object is rejected as a contract break rather than half-read.[^5] Two records on one line fail on the trailing content, which is the corruption newline-delimiting exists to prevent.[^5]

One check is worth more than the rest: a sample carrying fewer records than it declared is rejected, separately for each kind.[^6] A truncated capture is the ordinary result of reading the file while the agent is still writing it, and a planner that acted on a world it could not fully see would make exactly the class of decision this contract exists to make safe — a half-read pool list would have the bot build on a pool it had already taken, and a half-read option list does not look wrong at all: it looks like a unit that cannot make the thing the plan wants, which the planner would answer by declaring the plan dead.

## Threading

The sample is rendered on the game thread and written off it. A position read mid-tick can be torn, so the read is posted to the engine's own queue; disk I/O on the simulation thread would pace the game by disk latency, so the write is not.[^4] The agent's queue enqueues without running, so the render is awaited explicitly rather than read back immediately — reading it back is what the first attempt did, and it produced an empty sample every time.[^4]

## Replay falls out of it

A JSONL stream is itself a corpus. Decoding is a pure function of the lines it is given and reads no file, so the same code serves a live tail and an archived replay with no branch between them.[^6] The tests decode the real capture archived here rather than a fixture written to match the parser, and they cross-check it: the frame and clock deltas across two samples reproduce the ~300 Hz measured independently.[^3][^6]

[^1]: `wiki/sources/m6-wire/world-sample.ndjson:1` — the opening frame record of a real capture from a live headless skirmish, declaring 19 visible entities, 46 pools and 123 build options.
[^2]: `wiki/sources/m6-wire/world-sample.ndjson:2` — the entity record at index 0, an opposing team's `commandCenter` carrying `"mine":false`.
[^3]: `wiki/sources/m6-wire/world-sample.ndjson:190` and `:379` — the second and third frame records, 3646 at 12237 ms and 3945 at 13238 ms; 299 frames across 1001 ms is 298.7 per second.
[^7]: `wiki/sources/m6-wire/world-sample.ndjson:21` — the first pool record, tile (115, 6) at world (2310.0, 130.0).
[^8]: `src/rw_bot/control/channel.py` — `_complete_or_none` reads the opening record's counts and waits for `1 + visible + pools + options` lines before decoding.
[^9]: `wiki/sources/m6-wire/world-sample.ndjson:67` and `:72` — the Command Center's first option (`builder`, `"placed":false`) and the Builder's `landFactory` (`"placed":true`); `agent/src/rwbot/agent/BuildOptions.java` is the producer, and the predicate that reads as "makes something" is `a.s.g()`, false on `a.v` (build a structure) and true on `a.l` (produce a unit).
[^10]: `src/rw_bot/wire/command.py` — `encode_produce` emits `kind`, `unit_id` and `type` and no coordinate; `encode_build` carries `x` and `y`.
[^4]: `agent/src/rwbot/agent/StateStream.java` — the producer, with the flatness constraint and its rationale in the class javadoc; the render-on-game-thread and await are in `Premain.writeSample`.
[^5]: `src/rw_bot/wire/ndjson.py` — the strict reader, with the `disallow_any_expr` constraint recorded in the module docstring and one traceable code per rejection (`RW-NDJSON-001` … `-006`).
[^6]: `src/rw_bot/wire/state.py` — `decode_samples` folds records into samples and enforces the declared count (`RW-WIRE-003`); `encode_sample` round-trips, which is what makes a decoded corpus re-emittable as a fixture.
