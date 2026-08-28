---
title: Teleport Mechanics
tags: [teleport, movement, fuel]
related:
  - "[[viewport-frame]]"
  - "[[fuel-system]]"
  - "[[map-mechanics]]"
source_paths:
  - "runs/bot"
  - "src/tankpit_bot/physics/costs.py"
source_git_blobs:
  "src/tankpit_bot/physics/costs.py": "cfd4ecc3b7dca858aebb7fdc7e4b5c9f93d77f62"
fact_checked: "2026-06-12"
confidence: high
hubs: [game-mechanics]
---

# Teleport Mechanics

Teleport is the primary mobility model for search and hunting. Never propose replacing teleports with walking — if fuel economy looks bad, fix fuel acquisition or hop targets. Walking is only for short in-viewport combat closes.[^1]

## Placement

**Click directly on the target** — enemy tank, container, or tile. The server handles placement:[^2]
- If the tile is open, you land exactly there
- If a **tank** occupies the tile, the server places you adjacent (typically cardinal)
- If **ENEMY mines** occupy the tile, displaced to nearest open tile — own-color mines never displace (archive 2026-08-06: 1,227 enemy vs 2 friendly displacements, 20 clean exact landings on friendly mines; [[mine-mechanics]] § team scope)
- If **terrain** (rocks or water) at the target, displaced to nearest open tile

**The refusal law, mined 2026-08-21 (CORRECTING the ejection model
this paragraph briefly carried):** beyond ring-1, NO ejection exists.
All **137** archived chebyshev->=2 "displacements" landed the tank
**exactly at its own origin** (137/137), uncharged (no 6x-distance
debit follows; the -10s in the receipts are the repair radars) — and
the archive holds 8,718 landed vs 4 rejected teleports, so the server
answers a fully ring-blocked hop with a silent **confirm-at-origin**,
never a 0x52. What the bot logged as ``TELEPORT_DISPLACED`` at
distance >= 2 was always a REFUSAL: requested tile + whole ring-1
blocked, tank never moved. For four months that receipt fed nothing,
which let the identical hop re-certify against mine-blind beliefs
forever (the 08-05 session in the attainability docstring ran **534
refusals at one tile in 43 minutes**; the 2026-08-21 marooning ran
the same loop in escape and harvest form). Now a chebyshev->=2
refusal
writes ``ws.landing_refusals`` — requested tile + timestamp — and the
composed decision terrain refuses LANDINGS in the requested tile's
ring-1 (exactly the zone one refusal proves; walking unaffected) for
a 30 s TTL, so the existing unservable/clearance laws finally receive
the input they were starved of. A fresh refusal also forces the
landing repair radar regardless of coverage ([[radar-mechanics]] §
the s9-2 correction), and the analyzer flags any destination refused
>= 3 times as a **displacement orbit** (the third liveness flavor:
successful-looking actions, no progress — a refusal resolves
``landed_inexact``, so repetition hides from every failure counter).
The sim's physics always matched this law ("beyond-ring-1
displacement does not exist"); its WIRE SHAPE did not — it answered
sealed hops with 0x52 CANT_GO, corrected 2026-08-21 to the measured
confirm-at-origin and pinned by the ring-refusal seam scenario
(``tests/sim/test_landing_refusal_seam.py``: one hop on the wire,
refusal ingested, zero repeats).

**Displacement tombstones (operator ruling 2026-08-27, "if we get
displaced once then that should be enough info... why re-attempt
unless we cleared mines").** Ring-1 one-tile displacements were
routine-and-unrecorded until session bot-20260827-170800 re-aimed
the same tile four times (its issue report's displacement orbits).
The refined law: a one-tile displacement whose requested tile holds
**no known tank** proves an invisible occupant (hidden mine) on
exactly that tile — ``ws.mark_displacement_tombstone`` records it
(single tile, no ring: one displacement proves one tile) and
``hostile_landing_keys`` serves it to the composed decision terrain
with the same 30 s TTL as refusals. Aims at a tank's own body stay
exempt and unrecorded: combat closes displace by one legitimately on
every approach, and tombstoning the enemy's tile would poison the
kill approach itself. Pinned by
``tests/sniffer/test_world_state_dispatch_teleport.py`` (tombstone +
exemption).

**Do NOT compute adjacent tiles client-side.** The server is authoritative for placement. Teleporting to an enemy's exact coordinates is correct — the server places you adjacent. This is how human players play.[^8]

## Fuel cost

`cost = floor(6 * euclidean_distance)`. Cheapest: 6 fuel (1 tile). Diagonal = 8 fuel. Below ~8 fuel no teleport is affordable.[^3]

The distance is measured to the **actual landing tile**, not the clicked target: when the server displaces the landing (occupied tile, mines, terrain — see Placement above), the fuel charge matches `floor(6 × euclid(start, landing))` exactly. 624 drift hops confirm this; the planner's target-based estimate is off by only a few fuel on drifted hops, so no code change is warranted.[^3]

An unaffordable hop answers 0x52 code 8 ("Insufficient fuel" — 0.3%
of 7,364 live teleports). The affordability law lives in
``physics/supervisor.py`` (2026-08-03): the sim refuses with it at
the resolved landing; the bot predicts with it from the requested
target, discounted by the ring-1 displacement slack, and suppresses
only provably-dead dispatches.[^code8]

```json claims
{
  "claims": [
    {
      "id": "teleport-refusal",
      "code": "tankpit_bot.physics.supervisor:teleport_refusal",
      "law": "A teleport whose cost floor(6 x euclid) to the resolved landing tile exceeds the tank's fuel is refused 0x52 code 8; a cost exactly equal to fuel spends the tank dry and lands (sim router law, differ-verified against the 7,364-teleport live catalogue)."
    },
    {
      "id": "teleport-ring1-cost-slack",
      "code": "tankpit_bot.physics.supervisor:TELEPORT_RING1_COST_SLACK",
      "value": 9
    }
  ]
}
```

## Map requirement

The map must be open to teleport. `CMD_MAP_OPEN` (0x6c) opens it; teleport auto-closes it via `CMD_MAP_TELEPORT` (0x74). See [[map-mechanics]].[^4]

## Landing auto-pickup

Teleporting onto a container tile picks it up on landing.[^5]

## Timing

Map open → teleport → fire can all happen in one burst with no waits. The tank lands immediately and can fire on the next server tick.[^6]

## Server tick and queue

Server tick rate is 2000ms. Commands sent faster are queued by the server. Consecutive shots are ~2040ms apart — the server's actual shot cooldown. The server, not the bot, owns timing.[^7]

[^1]: user (Austin), 2026-06-11 — "no walking are you stupid lol"; teleport is the mobility primitive, walking only for short in-viewport closes. Encoded as the pricing every hop is gated on: `teleport_fuel_cost_to` at `src/tankpit_bot/bot/ai/context.py:451`, and the combat close that spends it, `teleport_to_target` at `src/tankpit_bot/bot/ai/combat_close.py:48`. Walking survives only as the in-viewport pickup path — see [[walk-mechanics]].
[^2]: User (Austin) statement of 2026-06-16, verbatim: "you get moved off if there are mines, or if there is terrain in the way. or if there is water there, you get teleported to the nearest open space". No transcript exists, but the displacement law it states is what the terrain layer encodes today, and the code says so in its own words: `src/tankpit_bot/terrain.py:151-165` explains that the static minimap "carries no mines and no tank bodies -- the only blockers a landing is exempt from", which is exactly why an aimed landing is displaced rather than refused. As of 2026-08-06 the composed view distinguishes aim from outcome with a third predicate, `is_landing_attainable` (`bot/ai/ferry.py:187`) — see [[terrain-composition]], which carries the full three-question table.
[^3]: Systematic validation 2026-07-20: every teleport dispatch in every `runs/bot/*.events.jsonl` paired with its wire fuel delta (pre-hop `Self:` position fix → post-hop `Fuel: A -> B` line; windows with intervening move/radar/shoot/pickup dispatches or a fuel-level mismatch excluded). Post-2026-06-24 era (after the fuel double-count fix): 248/248 pairs exact, costs spanning 6–654. All-era: 2,335/2,538 exact (1,711 on target coords + 624 on actual-landing coords); all 203 residuals live in pre-fix runs with broken fuel tracking. Supersedes the earlier undocumented "verified across multiple runs" assertion.
[^4]: discovery probe 2026-06-12 — map open/close wire behavior; see [[map-mechanics]] for full details. The message this probe characterised is `0x4C MapData`, named at `src/tankpit_bot/sniffer/constants.py:41` and classified `("map_data", "FULL")` at `:145`; the session-constant fuel-dot atlas it carries is described at `src/tankpit_bot/bot/ai/context.py:78`.
[^5]: Fuel-dot probe of 2026-06-11; artifacts at repo root, `fuel_probe.json` + `fuel_probe.capture_session.json`. Six nearest dots, all holding fuel; the sixth was auto-picked on landing rather than measured, taking fuel 639→1100. That is why [[map-data-decode]] [^3] lists only five volumes (762/807/880/1042/1189) for six dots — the discrepancy is the auto-pick, not a miscount.
[^6]: user (Austin), 2026-04-20 — protocol command timing; confirmed no waits needed between map/teleport/fire. The only pacing the bot imposes is its own cooldowns, not inter-command waits: `map_open_cooldown_ms` and `scan_cooldown_ms`, both 5000, at `src/tankpit_bot/bot/ai/types.py:113` and `:117`.
[^7]: bot-20260614-142159.capture_session.json — server response latency 56ms-2002ms; 2000ms responses = server HOLDING queued command until cooldown elapses
[^8]: user (Austin), 2026-06-16 — "I teleport to the same exact position as the enemy tank. so the game puts me adjacent. you don't have to click on map to teleport right below them"; confirmed by official How To Play: "Open the map, then click on it to teleport". Code truth is `choose_combat_landing_tile` at `src/tankpit_bot/bot/ai/combat_landing.py:80`, whose docstring states the rule verbatim — "teleports directly to their coordinates: the server handles displacement … This is how human players teleport: click on the enemy, let the server place you" — and records the corollary that the question asked is `is_landing_legal`, never `is_passable`.

## Displacement preference order (measured 2026-07-21)

User-piloted probe sniff-20260721-200527, 11 wire-verified
displacements (sent teleport target vs 0x3D landing fix):[^9]

- **The server tries the target's neighbors in a fixed ABSOLUTE
  order: EAST first, then NORTH, then WEST** — independent of
  approach direction (the mine at (17,63) was approached from the
  northwest, due north, and due west; all three landed at (18,63),
  its east neighbor; a solo mine and two rock targets also landed
  east).
- **A tile occupied by your own tank counts as blocked**: teleporting
  at (29,68) while standing on its east neighbor (30,68) landed
  north at (29,67); repeating from (29,67) landed east again.
- **West observed only when east and north were both rock**
  ((61,53) inside the rock mass → landed (60,53)).
- ~~South never isolated; search depth beyond ring 1 unobserved.
  Both remain open questions.~~ **CLOSED by the 2026-07-22 corpus
  sweep** (2,861 sent teleports paired with their 0x3D landing fixes
  across all 246 sessions, rejections excluded): 2,020 landed exact
  and 841 displaced — cardinal ring-1 dominates with **E 448 ≫ N 89 >
  S 31 ≈ W 28**. SOUTH IS REAL (31 samples), so the full cardinal
  set E→N→W→S stands with the E-then-N ordering probe-verified and
  the relative order of the last two unresolvable from frequencies
  alone (blockage geometry confounds). **The search also extends
  BEYOND ring 1**: ~24 % of displaced landings are ring-2 or
  diagonal offsets ((-2,-2) 30, (-2,-3) 25, (1,1) 23, (2,0) 17,
  (1,2) 16, (2,2) 11, ...), so a fully blocked ring 1 widens the
  search rather than rejecting. The sim currently models ring-1
  E→N→W→S then cant_go — the wider search is a DOCUMENTED
  simplification ([[physics-module-roadmap]]).
- Legality contract (user, same day): displacement applies to ENEMY
  mines and terrain; you can land on and walk over your OWN or
  ally-colored mines, and you cannot teleport onto enemy mines.
- Bonus re-confirmation: every probe hop's fuel delta matched
  floor(6 x euclid) to the ACTUAL landing (e.g. (12,56)->(18,63):
  55 fuel, wire 1100->1045).

[^9]: runs/sniff/sniff-20260721-200527.capture_session.json — user-piloted displacement probe (11 hops); corpus extension 2026-07-22: 2,861 sent-teleport/0x3D pairs across all 246 archive sessions
[^code8]: Archive-measured over 7,364 live teleport windows; the 0x52 refusal vocabulary this code belongs to is the canonical table on [[decode-coverage]], bound as a `members` claim so an omitted code fails the `physics_claims` guard (`scripts/physics_claims.py:305`). Teleport pricing is `teleport_fuel_cost_to` at `src/tankpit_bot/bot/ai/context.py:451`. Verified present 2026-08-07.
