---
title: "Wire Contract — NDJSON World Stream"
tags: [wire, agent, planner, ndjson, typing, contract]
related:
  - "[[runtime-split-java-agent-python-brain]]"
  - "[[engine-entity-model]]"
  - "[[engine-tick-and-clock]]"
  - "[[issuing-orders]]"
source_paths:
  - "wiki/sources/m6-wire/world-sample.ndjson:1"
  - "wiki/sources/m6-wire/world-sample.ndjson:2"
  - "wiki/sources/m6-wire/world-sample.ndjson:5"
  - "wiki/sources/m6-wire/world-sample.ndjson:9"
  - "agent/src/rwbot/agent/StateStream.java"
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

Records are discriminated by `kind`. A `frame` record opens a sample and declares how many entity records follow; each `entity` record carries one owned entity with the index an order is dispatched against ([[issuing-orders]]).[^1][^2]

```
{"kind":"frame","frame":1,"clock_ms":50,"owned":3}
{"kind":"entity","frame":1,"index":0,"class":"…units.d.e","x":4250.0,"y":2550.0}
```

`frame` and `clock_ms` are the engine's own counters, read from the same fields measured at ~300 Hz and 1 kHz ([[engine-tick-and-clock]]). Fields are limited to what has been verified against the engine rather than everything reachable, so the roster carries class and position and nothing speculative ([[engine-entity-model]]).

## Why every line is flat

**Every record is a flat object of scalars — no nesting, no arrays, no null.** That is a constraint on the producer, and it exists because of a constraint on the consumer.[^5]

The planner is type-checked under `disallow_any_expr`, and `json.loads` returns `Any`. Every expression touching that value is an error, `isinstance` narrowing does not rescue it because the offending expression is the call itself, and suppressions are banned by the project's standards. The standard library is therefore unusable for reading this stream.[^5]

That leaves writing a reader, and the cost of a reader is set by the grammar it must accept. A flat scalar object is small enough to parse strictly in one pass; general JSON is not. Constraining the format on the Java side is what buys a fully typed consumer on the Python side — the two decisions are one decision.[^4][^5]

## What the reader refuses

Nothing is coerced and nothing is repaired. `"4250"` stays a string, a duplicate key is an error rather than a last-one-wins merge, and a nested object is rejected as a contract break rather than half-read.[^5] Two records on one line fail on the trailing content, which is the corruption newline-delimiting exists to prevent.[^5]

One check is worth more than the rest: a sample whose entity count disagrees with its declared `owned` is rejected.[^6] A truncated capture is the ordinary result of reading the file while the agent is still writing it, and a planner that acted on a roster it could not fully see would make exactly the class of decision this contract exists to make safe.

## Threading

The sample is rendered on the game thread and written off it. A position read mid-tick can be torn, so the read is posted to the engine's own queue; disk I/O on the simulation thread would pace the game by disk latency, so the write is not.[^4] The agent's queue enqueues without running, so the render is awaited explicitly rather than read back immediately — reading it back is what the first attempt did, and it produced an empty sample every time.[^4]

## Replay falls out of it

A JSONL stream is itself a corpus. Decoding is a pure function of the lines it is given and reads no file, so the same code serves a live tail and an archived replay with no branch between them.[^6] The tests decode the real capture archived here rather than a fixture written to match the parser, and they cross-check it: the frame and clock deltas across two samples reproduce the ~300 Hz measured independently.[^3][^6]

[^1]: `wiki/sources/m6-wire/world-sample.ndjson:1` — `{"kind":"frame","frame":1,"clock_ms":50,"owned":3}`, the first record of a real capture from a live headless skirmish.
[^2]: `wiki/sources/m6-wire/world-sample.ndjson:2` — the entity record for roster index 0, `com.corrodinggames.rts.game.units.d.e` at `(4250.0, 2550.0)`.
[^3]: `wiki/sources/m6-wire/world-sample.ndjson:5` and `:9` — frames 1597 at 5388 ms and 3397 at 11388 ms; 1,800 frames across 6,000 ms is 300.0 per second.
[^4]: `agent/src/rwbot/agent/StateStream.java` — the producer, with the flatness constraint and its rationale in the class javadoc; the render-on-game-thread and await are in `Premain.writeSample`.
[^5]: `src/rw_bot/wire/ndjson.py` — the strict reader, with the `disallow_any_expr` constraint recorded in the module docstring and one traceable code per rejection (`RW-NDJSON-001` … `-006`).
[^6]: `src/rw_bot/wire/state.py` — `decode_samples` folds records into samples and enforces the declared count (`RW-WIRE-003`); `encode_sample` round-trips, which is what makes a decoded corpus re-emittable as a fixture.
