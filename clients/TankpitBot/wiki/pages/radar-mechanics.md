---
title: Radar Mechanics
tags: [radar, scanning, equipment]
related:
  - "[[viewport-frame]]"
  - "[[equipment-system]]"
  - "[[fuel-system]]"
source_paths:
  - "tpclient.js"
  - "runs/bot"
  - "runs/sniff"
fact_checked: "2026-07-06"
confidence: high
hubs: [game-mechanics]
---

# Radar Mechanics

## Two scan types, one wire command

`CMD_RADAR` (0x66) is the only radar wire command. The server decides which type based on inventory:[^1]

- **Extra radar** (extras > 0, enabled): scans the **entire visible viewport** regardless of rank. Client tip text: "Extra radar scans the entire visible screen." Reveals at chebyshev distance 3-12.[^1]
- **Built-in radar** (extras = 0): chebyshev radius **2 + floor(rank/3)** — rank-scaled, NOT a fixed 5x5.[^2][^14]

| Ranks | Radius | Footprint | Verified |
|---|---|---|---|
| recruit(0), private(1), corporal(2) | 2 | 5x5 | private: ~120 built-in scans, zero reveals beyond distance 2 |
| sergeant(3), lieutenant(4), captain(5) | 3 | 7x7 | sergeant: (128,120)→(128,123); lieutenant: (111,129)→(111,126) |
| major(6), colonel(7), general(8) | 4 | 9x9 | major: (234,5)→(238,5); colonel: (165,125)→(165,129) |

The step boundaries are pinned by the sergeant and major measurements (a rival fit with steps at lieutenant/colonel predicted sergeant=2 and major=3; both falsified 2026-07-06). This resolves the official guide's "higher rank tanks have a larger radar" claim with exact numbers. **Code implication**: `REGULAR_RADAR_RADIUS = 2` in `state/viewport_geometry.py` is only correct for ranks 0-2 and must become rank-derived (rank is on the wire in `self_state`).

## Both scans are clamped to viewport bounds

Neither radar reveals tiles outside the visible viewport. At a viewport edge or corner the built-in radar reveals only the intersection of `(tank±radius)` with the viewport bounds -- a rank 0-2 tank pressed into the top-left of its viewport sees a ~3x3 instead of a 5x5.[^7] Coverage tracking that pretends a free scan always reveals 25 tiles over-claims and lets the forager skip ground that was never actually scanned.

The implication for the bot: track scanned tiles by the actual revealed region (intersection of scan footprint with viewport bounds), not by a fixed-size block centered on the tank.

## What radar reveals (and what it does NOT)

Radar reveals **fuel containers, equipment containers, and mines** -- entities that are hidden by default on spawn and become visible only after a scan.[^8] Both radar types reveal: extra (paid) covers the full viewport in one shot; built-in (free) reveals tiles within chebyshev radius 2+floor(rank/3) of the tank, clipped to viewport bounds.

Radar does **NOT** reveal enemies. Enemy tanks are always visible to the bot when they enter the viewport via the normal wire stream (0x3D MovementResponse, 0x28 TankEntry, etc.). Firing radar to "search for enemies" is a category error -- any radar dispatched while in HUNT mode is wasted unless it is acquiring nearby mines / containers around a combat tile.

This deletes a class of bot mistakes: HUNT mode must use map-open or viewport-edge walking to find enemies, never radar.

## The radar response is a delta sync (reveals + corrections, unchanged omitted)

The wire response to a radar scan (0x4F, JS handler `ch` -- a batch of per-tile cache/overlay writes) is a **delta sync of the scanned area**, not an append-only reveal list:[^12][^13]

- **Newly revealed hidden entities** arrive as cache entries (fuel volume / equipment) and overlay entries (mine team 0-3).
- **Corrections** arrive as explicit removals: a cache entry with value 0 means "this tile is now empty" (247 of 2093 cache entries across the 199-session corpus, scan 2026-07-03), and an overlay entry >= 8 (255 canonical) means "no mine here". The client applies every entry as a raw tile write; the rendered dot and the mouse-hover fuel value both read the same per-tile cache slot.
- **Unchanged already-visible entities are NOT re-sent** (live run 2026-07-01 20:20:10: the teleport landing's 0x5A registered 7 visible containers; the scan-on-landing extra radar's response listed only the 2 hidden ones).

Implication for state tracking: the radar response must never be treated as the complete container set for the viewport, but its explicit entries ARE authoritative -- including the removals (`update_container_from_radar` deletes on volume 0; overlay clears route to `remove_mine`). The omission-prune (`reconcile_radar_viewport_resources`) is scoped to radar-sourced registry entries only; visible-layer entries are owned by 0x5A/0x43. Before the 2026-07-01 fix the whole-envelope reconcile deleted every visible container on each scan -- the bot would land amid 7 containers, radar, and instantly forget 5 of them (the "picked up only 2 of 7" bug the user observed live). The earlier "lists ONLY newly revealed" wording (2026-07-01) was too strong -- it missed the correction entries the corpus proves the server sends.

## Walking does NOT reveal containers

**Walking is not a reveal action.** Stepping onto a tile that holds a hidden fuel or equipment container does NOT make the container appear -- the bot does not learn about a container by walking on it.[^10] Only radar reveals. This matters because the natural intuition ("explore the viewport on foot to discover what's there") is wrong: a tank can spend its entire fuel budget walking every viewport tile and never see a single container that wasn't already radar-revealed.

What walking IS useful for is **repositioning the tank so the next free radar covers fresh tiles**. When extras = 0, the free radar reveals (2·radius+1)² tiles at most (25 at ranks 0-2) (fewer at viewport edges -- see [Both scans are clamped to viewport bounds](#both-scans-are-clamped-to-viewport-bounds)). A second free radar from the same tile reveals nothing new. The bot's foraging loop when out of extras is:

1. Free radar (radius square around current position, clipped to viewport).
2. Walk roughly one footprint-width (2·radius+1 tiles) in some direction -- no overlap, no gaps.
3. Free radar from the new position.
4. Repeat until a viewport tile that may hold equipment surfaces -- pick it up, which (eventually) lifts an extra radar -- or until the viewport is fully covered, at which point teleport to a fresh sector.

Walking costs **1 fuel per tile**.[^11] So a single free-radar cycle (walk 5 + radar ~10) burns ~15 fuel for up to 25 fresh tiles. A paid extra radar (also ~10 fuel) covers the whole 16x16 viewport in one shot -- which is why the policy is "always use extra radar when you have one." See [Policy: always use extra radar](#policy-always-use-extra-radar).

## Viewport shifting

In the current game configuration viewport shifting is **OFF**. The viewport never moves when the tank walks; the only way to change which 16x16 region the bot can see is to teleport. Once the bot has scanned every tile inside the current viewport the only forward action is a teleport to a new viewport.[^9]

Each scan with extras available auto-consumes one (count decrements by 1). The bot cannot choose per-scan.[^3]

## Policy: always use extra radar

Never ration extras via the toggle. The viewport sweep is always worth one extra. Keep stock UP through reliable equipment collection. A proposed "radar floor" (disable extras when low) was explicitly rejected.[^4]

## Death spiral at 0 extras

At 0 extras: 25-tile reveal vs 324-tile. Equipment discovery collapses, refill stalls. Three consecutive runs at 0 gained duals/homings but zero radars.[^5]

## Radar for fuel is not waste

One viewport sweep reveals ~10 containers. Live data: 32 pickups vs 9 dot hops in one run. Dots are ~40% fresh and one-at-a-time. Spending a radar to surface ten containers is high-value.[^6]

## Equipment refill

Extra radars come from equipment containers — plus ONE measured
exception (corpus-cracked 2026-07-22): a kill scored while your
extra-radar count is ZERO grants a silent mercy bundle including
+1–2 radar (deterministic, 5/5 vs 0/254; see [[equipment-system]]).
The pre-sweep "no other source" claim was falsified by that bundle —
it held for every non-kill path.[^5]

## Machine-checked claims

Binding for the built-in radar radius formula ([[physics-module-roadmap]]
Phase 1; probes chosen at the measured step boundaries[^14]). Verified
by the `physics_claims` guard stage on every `make check`.

```json claims
{
  "claims": [
    {
      "id": "free-radar-radius",
      "code": "tankpit_bot.physics.capacity:free_radar_radius",
      "formula": "chebyshev radius 2 + rank // 3",
      "probes": [
        {"args": [0], "expect": 2},
        {"args": [2], "expect": 2},
        {"args": [3], "expect": 3},
        {"args": [5], "expect": 3},
        {"args": [6], "expect": 4},
        {"args": [8], "expect": 4}
      ]
    }
  ]
}
```

[^1]: user (Austin), 2026-06-12 — extra radar = full viewport; client tip text confirms
[^2]: ~120 built-in scans across captures 2026-06-12 — zero reveals beyond chebyshev 2; hits at 1 and 2 only; was an unmeasured 7x7 assumption from April
[^3]: run 20260611-062453 — extra count series 10→9→...→3, one consumed per scan
[^4]: user (Austin), 2026-06-12 — "ALWAYS use extra radar — never ration via toggle"
[^5]: three consecutive runs at extras=0 — gained duals/homings but zero radars; verified equipment is sole radar source
[^6]: run 20260613-064xxx — single fuel scan led to ~10 "Picked up container" events; 32 pickups vs 9 dot hops all run
[^7]: user (Austin), 2026-06-21 — "it doesnt scan outside the viewport... only ever need to be able to accurate track the current tiles within the viewport"; viewport-edge intersection rule
[^8]: user (Austin), 2026-06-21 — "radar doesnt detect enemies. it is strictly and only for finding equipment and fuel and mines. those are hidden by default on spawn and revealed by radar"
[^9]: user (Austin), 2026-06-21 — "we have viewport shifting off. so the viewport will never move. the only way is to teleport"
[^10]: user (Austin), 2026-06-22 — "walking over contianers doesnt reveal them. only rsdar reveals euqipment snd fuel"
[^11]: user (Austin), 2026-06-22 — "wlak does consume 1 fuel per tile btw"
[^12]: live run 2026-07-01 20:20:10 — landing 0x5A registered 7 visible containers, scan-on-landing radar listed only the 2 hidden ones; entity_alignment samples tick 5 vs tick 6
[^13]: corpus scan 2026-07-03 (199 sessions, 1817 0x4F bodies): 2093 cache entries — 247 removals (value 0), 1074 equipment (0xFFFF), 772 fuel; 545 overlay entries all team values 0/1/3; 0 top-level (untunneled) 0x4F. JS ch handler tpclient.pretty.js:4800-4813.
[^14]: user (Austin), 2026-07-06 — four manual axial measurements on own tanks: lieutenant (111,129)→(111,126)=3; colonel (165,125)→(165,129)=4; sergeant (128,120)→(128,123)=3; major (234,5)→(238,5)=4. Sergeant/major chosen specifically to discriminate the two candidate step formulas.
