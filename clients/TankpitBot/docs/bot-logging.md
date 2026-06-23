# Bot Logging

`make bot` now has one runtime logging policy and one canonical output layout.

The on-screen terminal remains readable for an operator, while the same run is
persisted to stable text and JSONL artifacts documented in
[`docs/run-artifacts.md`](run-artifacts.md).

## Canonical Inspection Paths

After any bot run, inspect:

```text
runs/bot/latest.log
runs/bot/latest.events.jsonl
```

`latest.log` is the human timeline. `latest.events.jsonl` is the structured
record stream for automated inspection.

## Runtime Channels

High-signal bot output is emitted through a small fixed channel set:

- `STATE`: bot state machine transitions
- `SYNC`: waits, stalls, and action lifecycle updates
- `AI`: target choice, scoring, search, pathing, and reasoning
- `WIRE`: commands the bot actually sent
- `WORLD`: authoritative world-state changes and rendered viewport snapshots

Example human log lines:

```text
STATE: IDLE -> TELEPORTING
SYNC: waiting for teleport to (121,137)
AI: COLLECT_FUEL score=900 target=(0,0) cmd=radar equip=dual,homing,radar reason=forage_radar
WIRE: Sent: teleport(121,137)
WORLD: Fuel: 499 -> 376 (-123)
```

## What `make bot` Hides

`make bot` is intentionally not a packet-trace mode.

It suppresses noisy wire/decode chatter such as:

- raw per-message `[RECEIVED] ...` lines
- frame-by-frame protocol noise
- transport counters that are useful for sniffing but not for gameplay review

Use `make sniff` when you need exhaustive protocol output instead of behavior
analysis.

## Resource Freshness Model

The bot now distinguishes between:

- current viewport cache from `0x5A`
- remembered resources from earlier in the same run
- radar-confirmed current viewport truth

Planning rules:

- the visible viewport is modeled as `16x16`
- radar coverage is modeled separately as an `18x18` envelope around that
  visible viewport
- the bot does not trust visible fuel/equipment containers in the current
  viewport until radar has confirmed that viewport
- repeated radar in the same already-confirmed viewport is skipped
- after teleporting to a new viewport, cached resources there are treated as
  unconfirmed until radar refreshes that viewport

Since the 2026-06-21 scan refactor every radar dispatch (whether fuel or
equipment recovery owns the tick) is logged with `reason=forage_radar`. The
single forager owns scanning regardless of mode; the behaviour-mode label
(`COLLECT_FUEL` vs `COLLECT_EQUIPMENT`) on the same log line distinguishes
the caller, and `AIStateDict.local_scan_tiles` (not the server-side viewport
flag) is the gate that prevents respamming.

## Exploration Fallback

When `map_open` and radar are both cooling down, the bot now tries an ordered
set of executable viewport exploration targets before reopening the map.

Planning rules:

- `edge_for_fuel` and `edge_for_enemies` are selected from multiple
  edge-aligned candidates, not one hard-coded tile
- exploration teleport is rejected when it would burn fuel below the search
  reserve floor
- only after exploration options are exhausted does the planner fall through to
  a broader hunt fallback

This matters when reviewing loops in `latest.log` or `latest.events.jsonl`:
repeated `map_open` lines in the same viewport now indicate a broader planning
problem, not a single blocked edge coordinate.
