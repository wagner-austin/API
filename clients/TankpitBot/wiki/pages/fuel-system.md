---
title: Fuel System
tags: [fuel, containers, map-data]
related: [[teleport-mechanics]], [[radar-mechanics]], [[map-data-decode]]
sources: [see footnotes]
fact_checked: 2026-06-23
confidence: high
---

# Fuel System

## Thresholds (current, as of 2026-06-23)

- `fuel_low_threshold`: 300 — single fuel threshold. Below this the
  bot enters `RECOVER_FUEL`. Also the reserve a combat teleport must
  leave behind; engaging below it would flip priority to
  `COLLECT_FUEL` the next tick.[^1]
- `fuel_full_threshold`: 1100 — `RECOVER_FUEL` exits when fuel
  reaches this level.
- `hunt_min_fuel`: 100 — operating reserve for search/recovery
  teleport hops.
- The historical `fuel_critical_threshold` was collapsed into
  `fuel_low_threshold` 2026-06-22. The two-tier "polite low vs.
  emergency critical" distinction was dead because both thresholds
  had drifted to the same value (300). One threshold now governs
  RECOVER_FUEL entry, combat-teleport reserve, and
  collect-during-fuel-mode predicates.[^5]
- Only collect containers with volume >= 500 (smaller not worth the
  action cost).[^2]

## Fuel data flow (single source of truth)

`self_state["fuel"]` is updated **only** from the wire's absolute-fuel
messages, in `sniffer/world_state_dispatch._dispatch_resource_update`:

- `0x2E TankStatusSync` — periodic fuel cadence.
- `0x44 FuelGain` — fuel pickup completion (and free-pickup events).
- `0x64 FuelDeposit` — depot returns.

All three flow through
`sniffer/world_state_containers.update_world_state_from_fuel_total`
→ `state.mutations.set_self_fuel`, which writes the absolute fuel and
emits the `"Fuel: A -> B (+delta)"` world-log line.

`state/container_mutations.pickup_container` is **strictly registry
maintenance** — it removes the picked-up container (or shrinks its
volume on partial pickup), but does NOT modify `self_state["fuel"]`.
A prior version computed `transferred = prior_volume -
remaining_volume` and added it to `self_state["fuel"]`; combined with
the wire's already-correct absolute fuel, that double-counted every
fuel pickup. Live observation 2026-06-23: a 438-volume container
yielded the correct 633 fuel in the worldstate log but the bot's
`ctx.fuel` reported 1071 (633 + 438) for the next decision tick. The
local fuel-delta branch was removed 2026-06-23; the wire is the
single source of truth.[^6]

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
[^5]: AIConfigDict 2026-06-22 — `fuel_critical_threshold` field removed; consumers (`should_enter_recover_fuel`, `minimum_recovery_fuel_volume`, opportunistic-equipment gate in `_plan_fuel_recovery`) collapsed to use `fuel_low_threshold` directly. The unused `try_collect_critical_fuel` / `try_collect_fuel` non-owner helpers were deleted at the same time; the `_plan_fuel_recovery` wrapper was inlined into `decide_recover_fuel_mode`.
[^6]: Live observation 2026-06-23 00:35:57 in `runs/bot/latest.log`: worldstate logged `Fuel: 195 -> 633 (+438)` from the 0x44 FuelGain, then the next AI decision read `ctx.fuel = 1071` (633 + 438). The 438 ghost was the container's volume being added a second time by `pickup_container`'s local fuel-delta branch on top of the wire's already-correct absolute fuel. Removed in `state/container_mutations.pickup_container` 2026-06-23; the function now only mutates the container registry.
