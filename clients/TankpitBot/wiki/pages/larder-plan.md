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
  "src/tankpit_bot/state": "474b28f74ce32e4b0409d6694cfe8a1757c1b525"
  "src/tankpit_bot/bot/ai": "0c634f50801e90ed253a379a7ffb014e9f41e606"
fact_checked: "2026-07-27"
confidence: high
hubs: [architecture]
---

# Larder Plan — Harvesting Verified Containers the Bot Already Remembers

**Status: PLANNED, not implemented** (user-directed design session
2026-07-27). One live probe gates the build (§Probe gate). The user
attempted a version of this feature once before and removed it; the
post-mortem of that attempt is a standing design input still wanted.[^1]

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

HUNT is untouched: locks, chases, break thresholds unchanged. The
future under-fire teleport-refuel (PvP doctrine, [[bot-behavior-contract]])
is a second client of the same query — nearest known container
covering the deficit is a zero-command emergency refuel.[^4]

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
* **Equipment: +1 tick.** Land adjacent (on-tile pending the probe)
  and send `pickup_equipment` — which is the SAME wire action the
  human long-press dispatches (client `bb` handler: >300 ms hold →
  action 5 fuel / 6 equipment, [[client-commands]]). The
  equipment_gain event confirms contents.
* **No scan on larder hops** (user-confirmed reading of the
  zero-overlap ruling): the entry is already verified; the landing's
  free 0x5A exposure intel is taken, hidden tiles stay hidden, and a
  partially-scanned landing viewport (e.g. 4 unscanned tiles) is
  ignored — nearly-covered ground fails the untouched test anyway
  and recycles via the 180 s TTL.

## Probe gate (before any bot-loop code)

Does the server honor `pickup_equipment` targeting the tank's OWN
tile? The single 2026-06-21 sample failed silently; the user believes
on-tile works. Probe: teleport onto a verified equipment container,
try the pickup at own-tile and at-adjacent, read the wire. The answer
sets the equipment landing rule above.[^5]

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
[^5]: Own-tile equipment sample: capture 2026-06-21 16:54:26 ([[combat-chase-bug]] footnote 6 -- server placed the tank ON the container, pickup_equipment returned no container_consumed); long-press decode tpclient.js `bb` handler ([[client-commands]] Long-press pickup gesture).
