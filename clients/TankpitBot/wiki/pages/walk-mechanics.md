---
title: Walk Mechanics
tags: [game, movement, physics, timing]
related:
  - "[[game-economy]]"
  - "[[teleport-mechanics]]"
  - "[[viewport-frame]]"
  - "[[mine-mechanics]]"
  - "[[physics-module-roadmap]]"
source_paths:
  - runs/sniff/sniff-20260721-212348.capture_session.json (manual walk-timing + mine session)
  - bot archive probes 2026-07-21 (200 single-echo episodes; 1755 consecutive-echo pairs)
fact_checked: "2026-07-21"
confidence: high
verified: 2026-07-21 (manual capture + two full-archive probes agree)
hubs: [game-mechanics]
---

# Walk Mechanics

Server-side movement is **instantaneous**. A `CMD_MOVE` click is processed at
the next server tick, and at that single tick the server:

1. pathfinds the route ([[teleport-mechanics]] documents the tick; the
   quadrant-keyed deterministic pathfinder is logged in `log.md` 2026-07-21),
2. emits the `0x47` echo carrying the full route,
3. moves the tank to the destination,
4. bills 1 fuel per routed tile for the whole path, and
5. resolves destination-tile pickups —

all in the same wire flush. The walking you see on screen is a client-side
animation; the server has already teleported you.

## Evidence

- **Manual capture** `sniff-20260721-212348`: every single-click walk shows
  echo + full billing + pickup + refill at one timestamp. Example, t+63.81:
  fuel 595→587 (−8, the full 8-tile path), `0x47` echo `sswwwwww`, and the
  pickup at the destination `(3,220)` — one tick. A 12-tile walk commanded at
  t+179.71 resolved its destination pickup at t+179.92, bounding any internal
  per-tile latency below ~17 ms/tile.
- **Bot archive, billing probe**: of 200 single-echo exact walk episodes,
  **200 carry the full cost in the echo window itself**; 0 spread across
  later windows.
- **Bot archive, geometry probe**: of 1755 consecutive own-echo pairs ≥2 s
  apart, 1072 start exactly at the previous echo's destination and **0 start
  at an interior position of the previous path**. (663 start elsewhere —
  interleaved teleports; 20 start at the previous START, consistent with
  teleport-returns to the same tile, not partial walks.)

## Correction to the old model

The earlier belief that "a walk drains fuel tile by tile across several sync
windows" (encoded in `validate_walk_cost`'s docstring and the fuel book's
early designs) was **wrong**. The gradual-looking drain in bot sessions came
from the bot issuing many separate move commands over time, each billed
instantly at its own tick. The multi-echo tile overcounts (2026-07-21 probe)
came from echoed routes that never executed (position unchanged at next
echo), not from partially-walked paths.

## The client animation

The client animates at a fixed rate and **blocks map/radar/mine keys during
the animation**; a key spammed mid-walk registers at the first tick after the
animation ends. Three repeated 23-tile manual walks bounded the animation at
**≤181 ms/tile** (tick quantization leaves the lower bound open at ~87
ms/tile). The exact rate is cosmetic — it is NOT server physics and no longer
blocks the simulator.

## Implications

- **Humans are input-locked while animating**: after a long click, a human
  cannot radar, open the map, or lay mines until the animation finishes
  (~0.1–0.18 s per tile). A long move is a window of enemy unresponsiveness.
- **The bot is not animation-gated**: it writes commands to the socket
  directly. Effective bot movement is any pathable destination in ONE tick at
  1 fuel/tile — cheaper than a 30-fuel teleport for routes under 30 tiles
  (23-tile routes wire-confirmed; longer untested). The server pathfinder
  routes around enemy mines and terrain on its own.
- **Simulator**: model movement as instant relocation at the processing tick
  with per-tile billing; no walk-speed constant exists server-side.
