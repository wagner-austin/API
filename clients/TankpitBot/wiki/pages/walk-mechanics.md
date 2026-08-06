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
  - "runs/sniff/sniff-20260721-212348.capture_session.json"
  - "src/tankpit_bot/validate/archive.py"
source_git_blobs:
  "src/tankpit_bot/validate/archive.py": "67932554437cedff2d01b820d227dc5033c7edb5"
fact_checked: "2026-07-21"
confidence: high
verified: 2026-07-21 (manual capture + two full-archive probes agree)
hubs: [game-mechanics]
---

# Walk Mechanics

Server-side movement is **instantaneous**.[^1] A `CMD_MOVE` click is processed at
the next server tick, and at that single tick the server:

1. pathfinds the route ([[teleport-mechanics]] documents the tick; the
   quadrant-keyed deterministic pathfinder is logged in `log.md` 2026-07-21),
2. emits the `0x47` echo carrying the full route,
3. moves the tank to the destination,
4. bills 1 fuel per routed tile for the whole path, and
5. resolves destination-tile pickups —

all in the same wire flush. The walking you see on screen is a client-side
animation; the server has already teleported you.[^1]

## Evidence

- **Manual capture** `sniff-20260721-212348`:[^1] every single-click walk shows
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
echo), not from partially-walked paths.[^2]

## The client animation

The client animates at a fixed rate and **blocks map/radar/mine keys during
the animation**; a key spammed mid-walk registers at the first tick after the
animation ends. Three repeated 23-tile manual walks bounded the animation at
**≤181 ms/tile** (tick quantization leaves the lower bound open at ~87
ms/tile). The exact rate is cosmetic — it is NOT server physics and no longer
blocks the simulator.[^3]

## Implications

- **Humans are input-locked while animating**:[^3] after a long click, a human
  cannot radar, open the map, or lay mines until the animation finishes
  (~0.1–0.18 s per tile). A long move is a window of enemy unresponsiveness.
- **The bot is not animation-gated**: it writes commands to the socket
  directly. Effective bot movement is any pathable destination in ONE tick at
  1 fuel/tile — cheaper than a 30-fuel teleport for routes under 30 tiles
  (23-tile routes wire-confirmed; longer untested). The server pathfinder
  routes around terrain and around **VISIBLE** enemy mines only — mines
  that are still hidden (unrevealed by radar, [[radar-mechanics]]) are
  NOT routed around: the walk proceeds into the field and the
  walk-over law applies (detonate the one mine, 45 fuel, movement
  arrested, [[mine-mechanics]]). The earlier unqualified wording
  ("routes around enemy mines and terrain on its own") was an
  inference from the walk-timing session and is corrected here.[^4]
- **Recovery rule after an unexpected arrest**: an arrest mid-walk
  means the route crossed a mine we could not see. Fire radar to
  reveal the field, THEN re-issue the movement — the re-issued walk
  is now pathed around the revealed mines. Re-walking without the
  reveal re-enters the same blind field.[^4]
- **Mine visibility is TEAM-scoped**:[^5] a teammate's scan reveals
  the mines for the whole color; a freshly planted mine is invisible
  to enemies — even one sharing the planter's viewport — until
  someone on their team radars; own-team mines are always visible
  (and walkable, [[mine-mechanics]]). "Visible" in the routing rule
  above therefore means "revealed to the mover's TEAM".
- **Simulator**: model movement as instant relocation at the processing tick
  with per-tile billing; no walk-speed constant exists server-side.

## The cant_go partial-walk law (measured 2026-08-04)

Code 1 ("You can't go there!") is NOT a refusal — it is the receipt
of a walk the server accepted and could not finish. Exact-window
measure over the 12 live code-1s of runs bot-20260803-180918 and
bot-20260802-205105 (`analysis_scripts/mine_cant_go_choreography.py`,
pairing each logged rejection with the capture's 0x47 echoes):[^6]

- **11 of 12 carried a self 0x47 walk echo in the receipt window** —
  the server accepts the command, walks what it can, stops, reports.
  Two distinct producers were identified once field01's terrain and
  the ferry move log were laid over the echoes: (a) **blocker stops**
  — 18:12:35's 14-step route toward (16,21) stopped at (16,24) with
  Belton's body on (16,23); (b) **surface-transition truncations** —
  the four cluster-A collects and both 18:40 commands were issued
  while the bot RODE a ferry (afloat on (59,28) / (57,13) water),
  and each echo is the ferry law's own truncation (the one-step
  disembark, the water-route stop) closed with code 1 because the
  click was not reached ([[ferry-mechanics]] "The unfinished-command
  close").
- **The zero-tile pure refusal exists** (1 of 12, 20:58:45): the
  first step was already blocked — bare 0x52, no echo, no movement.
- Echo and 0x52 land in the same processing window (−39 ms/+99 ms in
  the second-granularity-aligned pairs).
- The walked prefix is billed normally (the walk-cost validator
  prices ALL 0x47 windows against fuel and stays green).

The simulator encodes this in `sim/movement.py`: primary plan avoids
tanks + team-revealed enemy mines + block obstacles; a severed
corridor falls back to a terrain-only plan executed step-by-step,
stopping BEFORE the first tank/revealed-mine/block contact
(`cant_go` with the walked prefix; the 0x47 echo precedes the 0x52
in the same batch), stepping ONTO a hidden enemy mine (the walk-over
detonation-arrest — no code 1), and refusing outright with a bare
0x52 only when static terrain itself severs the corridor.

[^1]: runs/sniff/sniff-20260721-212348.capture_session.json (user-piloted walk-timing session; frontmatter-pinned) — instantaneous echo+billing+pickup at one timestamp, all examples above re-derivable from the file
[^2]: 2026-07-21 archive probes (billing: 200/200 single-echo episodes carry the full cost in the echo window; geometry: 1755 consecutive echo pairs, 0 interior-position starts) — method + numbers in wiki log 2026-07-21; the billing law is re-derived on every `make audit` (walk-cost validator, `src/tankpit_bot/validate/archive.py`)
[^3]: same capture, three repeated 23-tile manual walks with user narration (input-lock observed live; timing bounds from the wire timestamps)
[^4]: user (Austin) 2026-08-04, verbatim: "it only auto paths you around visible mines" — supplied while resolving the F6 `cant_go` residual ([[flag-triage-20260729]]), where the unqualified version of this claim had been used to argue that revealing mines could not change routing. Consistent with [[radar-mechanics]] (mines hidden by default, revealed only by scan) and with the walk-over arrest law in [[mine-mechanics]], which would be unreachable if the server detoured around every mine.
[^5]: user (Austin) 2026-08-04, near-verbatim: "if someone on our team or same tank color scanned the mines previously, they're visible. but if any new mines are planted — even if we're on the same viewport as the planting tank — we cannot see them, unless the mines are for our team of course, or if someone on our team radars." Supplied mid-build while the sim's reveal tracking was being written per-tank; the correction moved it to per-team (`SimWorldDict.revealed_mine_keys_by_team`).
[^6]: analysis_scripts/mine_cant_go_choreography.py over runs/bot/bot-20260803-180918 and runs/bot/bot-20260802-205105 — pairs each event-log code-1 line (both log wordings) with the capture's decoded 0x47 self echoes (start + nsew path, final tile recomputed from start+path), the 0x52 receipts, and every other tank's last wire-stated position; full per-window output in wiki log 2026-08-04. Supersedes the nearest-sample "9 of 10 moved" measure recorded earlier in [[flag-triage-20260729]] — that measure's granularity blurred neighboring walks into the answer; the exact-window result is 11 of 12 with the paired echo.
