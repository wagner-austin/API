---
title: Larder Plan — Harvesting Verified Containers the Bot Already Remembers
tags: [bot, roadmap, collect, economy, plan]
related:
  - "[[fuel-system]]"
  - "[[equipment-system]]"
  - "[[bot-behavior-contract]]"
  - "[[client-commands]]"
  - "[[teleport-mechanics]]"
  - "[[game-economy]]"
source_paths:
  - "src/tankpit_bot/state"
  - "src/tankpit_bot/bot/ai"
source_git_blobs:
  "src/tankpit_bot/state": "01f57c7928f05025a5ca0c14ab82ec4ff320031b"
  "src/tankpit_bot/bot/ai": "0a51db229264f0bc93945d41f007c54f5ead8b0d"
fact_checked: "2026-07-31"
confidence: high
hubs: [architecture]
---

# Larder Plan — Harvesting Verified Containers the Bot Already Remembers

**Status: IMPLEMENTED, live-proven** (gate cleared, built, and
first live run all 2026-07-27; §Implementation). The user attempted
a version of this feature once before and removed it; the
post-mortem of that attempt is a standing design input still
wanted.[^1]

## The observation

The bot radar-verifies rich fuel containers (>500) and accessible
equipment containers it does not need at that moment, then forages
fresh viewports later as if that knowledge did not exist. Measured
cost of ignorance: the 2026-07-26 slow 10-kill run spent 7.70 forage
viewports per kill; the deciding resource (weapons per pickup) is
luck-distributed across unknown ground while known stock sits in
memory.[^2]

## Memory: pure reuse, zero new state

`world["containers"]` IS the larder. It already carries per-tile
volume (radar-verified), position, freshness timestamp, source, and
the `failed_pickups` blacklist; it is maintained by radar writes,
0x5A/0x43 tile updates (including partial-drain `remaining_volume`),
pickup consumption, and code-4 purges — and it dies with the session.
**User ruling: session-only, NO inter-session persistence.** The
feature adds only a query + scorer over this registry.[^3]

Two independent clocks, deliberately not unified: the scan-coverage
TTL (`FORAGE_COVERAGE_TTL_MS` = 180 s) answers "is this ground worth
a radar"; container belief answers "is that stock still there" and
expires only on hard evidence. A container scanned 10 minutes ago in
long-expired coverage is still harvestable — on the practice map,
bots are blind ([[enemy-bot-behavior]]) so belief decay is near zero.

## Selection: highest-and-nearest, never errands

Per tick that COLLECT owns with a deficit, score every candidate:
`min(volume, current_deficit) / teleport_cost` for fuel (a 900 at 25
tiles beats a 300 at 10 when down 700, loses when down 200);
expected weapons yield / cost for equipment. Argmax wins; re-scored
every tick so the plan never goes stale. **User ruling: no
fixed-container errands** — the score picks, not a queue.[^1]

## Placement: a cascade priority inside COLLECT — not a mode, not an interrupt

1. In-viewport walk pickups (unchanged, still first — walking is free).
2. **Larder harvest (new)** — chain hops through scored candidates.
3. Discovery (dot hops, scans, equipment search) — only when the
   larder is empty or unprofitable; radar is spent only when
   knowledge is exhausted.[^4]

HUNT is untouched: locks, chases, break thresholds unchanged[^7].

The under-fire teleport-refuel (PvP doctrine, [[bot-behavior-contract]])
has since LANDED as the predicted second client of the same query:
`collect_mode.py:328` calls the same `_larder_harvest` on the
under-fire path and takes the hop only when `_hop_escapes_attacker`
confirms it clears the attacker's envelope — so the escape and the
refuel are one command instead of two. It sits after the walk-pickup
attempt (skipped outright when movement rejections say the walk rungs
are dead) and before the generic escape hop.[^4]

## Harvest mechanics (all measured laws)

* **Fuel: land and it is picked.** The server auto-picks a fuel
  container when a teleport lands ON it or cardinally adjacent —
  no command at all (user law 2026-07-27; corpus check 62/82
  landings, [[fuel-system]]). Confirmation is free: the announced
  0x44/0x2E credit books in the fuel ledger and 0x43 updates or
  removes the registry entry. A landing with no gain is the
  disappointment signal (failed-pickup treatment).
* **Hop cost:** the map-open precondition stands (same-tick
  map_open + teleport is silently dropped), so a fuel stop is ~2
  ticks (map_open, teleport) — pickup and verification cost zero.
  Current forage stop: ~4 ticks (map, teleport, scan, pickup).
* **Equipment: +1 tick, land ON it.** Teleports aimed at a container
  tile land exactly on it (5/6 aimed landings, runs -225643 and
  -230858) and `pickup_equipment` from the tank's OWN tile credits
  (3/3, §Probe gate) — the SAME wire action the human long-press
  dispatches (client `bb` handler: >300 ms hold → action 5 fuel /
  6 equipment, [[client-commands]]). Equipment tiles are walkable
  and never auto-pick (user law 2026-07-27). Caveat: a pickup with
  every slot at cap is rejected with the 0x52 code-7 "Inventory
  full" receipt ([[equipment-system]]) — harmless here, since the
  larder harvests equipment only against a deficit.
* **No scan on larder hops** (user-confirmed reading of the
  zero-overlap ruling): the entry is already verified; the landing's
  free 0x5A exposure intel is taken, hidden tiles stay hidden, and a
  partially-scanned landing viewport (e.g. 4 unscanned tiles) is
  ignored — nearly-covered ground fails the untouched test anyway
  and recycles via the 180 s TTL.

## Probe gate — ANSWERED YES (2026-07-27)

The server honors `pickup_equipment` targeting the tank's OWN tile,
as the user predicted. `LarderProbe` (`action_lab/larder_probe.py`,
`make larder-probe`) teleported onto verified equipment containers
and credited the own-tile pickup **3/3 with zero 0x52 errors** (run
`larder-20260727-230858`). Two earlier iterations taught the probe
its controls: water-sitting shore containers can never host the
trial (run -224933 — every walk-on rejected, candidates now filtered
to passable tiles), and a fully-capped tank rejects EVERY pickup
with code 7 "Inventory full" (run -225643 — the one radar-spent slot
let an identical adjacent pickup credit, isolating the cap as the
cause), so each attempt now burns one extra radar first for
guaranteed headroom.[^5]

## Implementation (2026-07-27)

`bot/ai/larder.py` holds the fuel scorer (``select_fuel_larder_hop``:
physics-only gates -- legal landing, reserve, net-positive gain --
then argmax, re-run every tick). COLLECT's cascade gained step 5
``_larder_harvest`` (equipment hop first, then fuel), moved AHEAD of
forage/discovery; both hops hold a resource lock on their container
(the landing tick dispatches the pickup directly, own-tile for
equipment) and set ``suppress_landing_scan`` so the landing latches
without a radar. Non-larder landings keep the unconditional
2026-07-03 scan.[^6]

First live proof (3-minute run `bot-20260727-234645`): 2 kills,
25/25 hits, 0 rejections, exit ``session_complete`` at full stock;
forage viewports 2.00/kill (vs 3.10 best / 7.70 worst pre-larder);
6 radars total against 26 pickups and 8 teleports; equipment hops
landed ON their containers and the own-tile pickups credited;
"larder landing: latching without radar" on every harvest hop.[^6]

## Expected payoff

Fuel restocks stop paying the discovery tax: known-stock stops at
half the wire cost of forage stops, radar reserved for genuinely
unknown ground, and the weapons-per-pickup variance that decided the
803 s vs 1,187 s 10-kill pair shrinks because verified equipment
gets harvested instead of rediscovered.[^2]

[^1]: User rulings, design session 2026-07-27 (wiki/log.md entries of that date): session-only memory, highest-and-nearest scoring, no-scan larder hops, probe-first for the equipment landing; the removed first attempt is the user's own report.
[^2]: Forage economics: `tankpit-forage-economy` on runs bot-20260726-094309 vs bot-20260726-145124 (7.70 vs 3.10 viewports/kill, 2.14 vs 3.34 weapons/pickup); auto-pick corpus check 62/82 landings across runs bot-20260727-214102 + -211712.
[^3]: Registry mechanics: `state/container_mutations.py` (radar upsert, 0x43 remaining-volume update/removal, code-4 purge path), `failed_pickups` blacklist consumed by `bot/ai/equipment_search.py`; coverage clock `state/scan_coverage.py` `FORAGE_COVERAGE_TTL_MS = 180000`.
[^4]: Cascade order: `bot/ai/collect_mode.py` (in-viewport pickup steps precede hop steps); hunt-gate/lock independence per [[bot-behavior-contract]] §resume and never-drop rows; under-fire refuel doctrine from the 2026-07-27 guest post-mortem (wiki/log.md).
[^6]: Code: `src/tankpit_bot/bot/ai/larder.py` (scorer + FuelLarderSelectionDict), `collect_mode.py` (`_larder_harvest`, `_hop_toward_fuel_larder`, suppress-aware `_scan_on_landing_decision`), `types.py` `suppress_landing_scan` field; run artifacts `runs/bot/bot-20260727-234645.*`; forage-economy output recorded in wiki/log.md (2026-07-27 larder implementation entry) against baselines bot-20260726-145124 (3.10) and bot-20260726-094309 (7.70).
[^5]: Probe artifacts on disk: `runs/probe/larder-20260727-224933.json` (water-sitting candidates, 0 trials), `-225643.json` (own-tile 0/2 + adjacent control 1, all-capped inventory, code-7 receipts in the paired `.log`), `-230858.json` (own-tile 3/3, no errors) with matching `.capture_session.json` wire evidence; historical own-tile sample capture 2026-06-21 16:54:26 ([[combat-chase-bug]] footnote 6) superseded by the deliberate trials; long-press decode tpclient.js `bb` handler ([[client-commands]] Long-press pickup gesture); code-7 string table [[decode-coverage]] §Supervisor error codes.
[^7]: [synthesis] — the larder plan is a cascade priority inside COLLECT, so it edits the `collect_*` modules only. The HUNT family (`src/tankpit_bot/bot/ai/hunt_mode.py`, `hunt_acquire.py`, `hunt_lock.py`, `hunt_relay.py`) carries lock, chase, and break-threshold logic and is not in this change's scope. Verified 2026-07-31 that all four modules exist and own that behaviour; this footnote records scope, not a measured no-op.
