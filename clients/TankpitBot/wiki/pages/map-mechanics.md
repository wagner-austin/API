---
title: Map Open/Close Mechanics
tags: [map, protocol, teleport]
related:
  - "[[teleport-mechanics]]"
  - "[[map-data-decode]]"
source_paths:
  - see footnotes
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

[^1]: discovery probe 2026-06-12 — tested open, close, toggle; CDP keypress is the only close mechanism; commit history documents probe
[^2]: discovery probe 2026-06-12 — bytes 0x6d, 0x6e, 0x6f sent; no effect on map_visible
[^3]: user (Austin), 2026-04-20 — "map blob only provides positions at the moment it's opened"
[^4]: 15 runs 2026-06-11 — MAP_DATA bytes identical across 32 map opens in 240s session
