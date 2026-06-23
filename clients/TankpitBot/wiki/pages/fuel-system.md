---
title: Fuel System
tags: [fuel, containers, map-data]
related: [[teleport-mechanics]], [[radar-mechanics]], [[map-data-decode]]
sources: [see footnotes]
fact_checked: 2026-06-22
confidence: high
---

# Fuel System

## Thresholds (current, as of 2026-06-14)

- `fuel_low_threshold`: 300 (was 500; lowered to afford combat teleports)[^1]
- `fuel_critical_threshold`: 300 (matches low)[^1]
- `hunt_min_fuel`: 100 — operating reserve for radar + teleport[^1]
- Only collect containers with volume >= 500 (smaller not worth the action cost)[^2]

## Fuel recovery cascade

When `fuel < fuel_low_threshold` the bot enters `RECOVER_FUEL`. The
owner runs the same Strict → Sense → Hop cascade the equipment owner
uses (see [[bot-behavior-contract#3.4]]):

1. **Strict** — pick up the best reachable fuel container visible in
   the current viewport. Opportunistically grab adjacent equipment
   or pickups en route. Continue a locked target if one is held and
   no markedly closer candidate beats it.
2. **Sense** — fire a radar to reveal the current viewport when there
   are still unscanned tiles (paid radar covers the full viewport;
   free radar covers a 5×5 around the tank, clipped to viewport
   bounds). When radar is unaffordable, walk toward an unscanned
   tile so the next free radar covers fresh ground.
3. **Hop** — teleport to a fresh viewport via the ring-patrol search
   hop. When no hop is affordable the owner raises loudly rather
   than idle silently.

There is no fuel-dot atlas: the bot does not consult a map-wide
fuel-container list. The MAP_DATA blob still carries the RLE fuel-dot
section, but the protocol decoder skips past those bytes (the
sniffer/state layer does not surface them).[^3]

## Marooning hazard

A teleport can land the tank on a one-tile island in a lake (run
20260612-131003: dropped at 87 fuel with no walkable exit).
Marooning has never been recoverable: if fuel is below the cheapest
teleport cost there is no escape, dot or otherwise. A "marooned
escape" used to exist that bypassed the operating-reserve gate to
fire a fuel-dot teleport. The reserve-vetoed band it covered was
narrow -- fuel high enough for the teleport but below the hunt
reserve -- and the actual stranded case (fuel below any teleport
cost) was always fatal anyway. The escape was removed with the rest
of the fuel-dot system 2026-06-22; the RECOVER_FUEL owner now raises
`ValueError` loudly when Strict / Sense / Hop all decline.[^4]

[^1]: AIConfigDict in bot/ai/types.py — thresholds lowered from 500→300 in Phase 3d (2026-06-14)
[^2]: user (Austin), 2026-06-11 — "only collect fuel containers with volume >= 500"
[^3]: Phase A/B/C of the fuel-dot strip (2026-06-22): planner, state, protocol decoder all stopped surfacing dot coordinates. RLE byte count is still parsed for length validation so the decoder advances cleanly into the MAP_DATA tank-entries section.
[^4]: Run 131003 2026-06-12 — marooned at 87 fuel on one-tile island (actually a ferry; see [[ferry-mechanics]]). Reserve-bypass escape removed 2026-06-22.
