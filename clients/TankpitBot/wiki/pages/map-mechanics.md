---
title: Map Open/Close Mechanics
tags: [map, protocol, teleport]
related:
  - "[[teleport-mechanics]]"
  - "[[map-data-decode]]"
source_paths:
  - "runs/bot"
fact_checked: "2026-06-12"
confidence: high
hubs: [game-mechanics]
---

# Map Open/Close Mechanics

## Opening

`CMD_MAP_OPEN` (0x6c) is one-way. The server returns map data once. Sending it again against an already-open map is a server-side no-op.[^1]

## Closing

**There is no wire byte that closes the map.** Pressing `m` (or `f`) in the browser toggles `activeGame.map.h` purely client-side — no WebSocket traffic generated. The server only learns the map closed when it auto-closes on `CMD_MAP_TELEPORT` (0x74).[^1]

To close programmatically: `cdp.send("Input.dispatchKeyEvent", ...)` simulating the `m` key. See `Bot.close_map()`.[^1]

**Do NOT:**
- Send `CMD_MAP_OPEN` to close the map (does nothing)[^1]
- Invent a `CMD_MAP_CLOSE` constant (no such command exists)[^1]
- Adjacent bytes `0x6d`, `0x6e`, `0x6f` tested — server ignores them all[^2]

## Live signal

`capture_page_client_snapshot(cdp)["map_visible"]` reads `activeGame.map.h`. That is the authoritative "is the overlay showing" signal.[^1]

## Position staleness

Map blob provides positions at the moment it's opened. Enemy positions go stale as tanks move. Must re-open map periodically for fresh positions.[^3]

## Cache behavior

MAP_DATA is server-cached — byte-identical across all map opens in one session even while containers are consumed. Drifts a few dots between sessions. Treat as atlas, not live feed.[^4]

[^1]: Discovery probe of 2026-06-12; captures under `runs/bot/bot-20260612-*.capture_session.json` (9 sessions carry that date, counted 2026-08-06). Tested open, close and toggle; a CDP keypress was the only mechanism that closed the map. The state this drove is `snapshot["map_visible"]`, read by the executor's teleport short-circuit at `src/tankpit_bot/bot/executor.py:298`.
[^2]: Same 2026-06-12 probe session set. Bytes `0x6d`/`0x6e`/`0x6f` (ASCII `m`/`n`/`o`) were sent and none moved `map_visible`. Note these are NOT the map command: opening the map is `CMD_MAP_OPEN = 108` (`0x6c`), bound in `src/tankpit_bot/protocol/commands.py` and machine-checked by the claim block on [[client-commands]] — the three bytes probed here are the ones immediately after it, which is why they were tried.
[^3]: user (Austin), 2026-04-20 — "map blob only provides positions at the moment it's opened". The blob is `0x4C MapData`, named at `src/tankpit_bot/sniffer/constants.py:41` and classified `("map_data", "FULL")` at `:145`; that its payload is a session-constant snapshot rather than a live feed is stated at `src/tankpit_bot/bot/ai/context.py:78`, which caches the fuel-dot atlas "session-constant like ``terrain``".
[^4]: Sessions of 2026-06-11 under `runs/bot/bot-20260611-*.capture_session.json` — MAP_DATA bytes identical across 32 map opens within one 240 s session. **Re-counted 2026-08-06: 22 capture sessions carry that date, not the 15 stated here**; the original subset was not recorded. [[map-data-decode]] [^2] carries the same figure and the same correction.
