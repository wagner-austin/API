---
title: Fuel System
tags: [fuel, containers, map-data]
related: [[teleport-mechanics]], [[radar-mechanics]], [[map-data-decode]]
sources: [see footnotes]
fact_checked: 2026-06-23
confidence: high
---

# Fuel System

## Thresholds (current, as of 2026-06-24)

- `fuel_low_threshold`: 300 — single fuel threshold. Below this the
  bot enters `COLLECT`. Also the reserve a combat teleport must
  leave behind; engaging below it would flip priority to `COLLECT`
  the next tick.[^1]
- `fuel_full_threshold`: 1100 — `COLLECT` releases (along with the
  combat-reserve gate) when fuel reaches this level.
- `hunt_min_fuel`: 100 — operating reserve for search/recovery
  teleport hops.
- The historical `fuel_critical_threshold` was collapsed into
  `fuel_low_threshold` 2026-06-22. The two-tier "polite low vs.
  emergency critical" distinction was dead because both thresholds
  had drifted to the same value (300). One threshold now governs
  COLLECT entry, combat-teleport reserve, and the in-cascade
  fuel-pickup predicate.[^5]
- COLLECT drains every viewport fuel container (`minimum_volume=1`):
  the old "skip volumes < 500" floor was dropped 2026-06-23 after a
  live run left 20-fuel partials in viewport while burning ~200 fuel
  per hop to find a fresh one.[^2]

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

When `fuel <= fuel_low_threshold` the bot enters the unified `COLLECT`
mode (the historical `RECOVER_FUEL` / `RECOVER_EQUIPMENT` split was
collapsed 2026-06-24). The owner runs a single cascade per tick (see
[[bot-behavior-contract#3.4]]):

1. **Lock continuation** — continue a held equipment or fuel target
   from a previous tick when it is still executable and no markedly
   closer candidate beats it.
2. **Equipment pickup** — pick up the best walk-reachable equipment
   in the current viewport. A container only counts as actionable
   when a walk path to it exists inside the current viewport; the
   teleport-to-container fallback was removed 2026-06-26 because the
   server rejects pickups it cannot route to, leaving the container
   on the session-permanent blacklist after one failure.
3. **Fuel pickup** — pick up the best walk-reachable fuel in the
   current viewport when below the learned capacity (skipped at cap
   because "Tank full" is a wasted dispatch). Equipment ranks ahead
   of fuel per the user's gameplay loop.
4. **Sense** — fire a radar to reveal the current viewport when there
   are still unscanned tiles (paid radar covers the full viewport;
   free radar covers a 5×5 around the tank, clipped to viewport
   bounds). When radar is unaffordable, walk toward an unscanned
   tile so the next free radar covers fresh ground.
5. **Hop** — teleport to the cleanest fresh viewport nearby.
   Candidates are the eight compass neighbors at one and two
   viewport-widths (16 candidates). A candidate qualifies when its
   landing tile is passable, the teleport is fuel-affordable, and the
   destination viewport is unscanned. Qualifiers are ranked by the
   **walkable fraction** of the landing viewport from the static
   terrain map (mostly-"." viewports, matching the recorded human
   restock policy — see [[gameplay-loop]]); ties keep the cheapest
   hop because candidates are iterated cheapest-first (16 cardinal,
   16 diagonal, 32 cardinal, 32 diagonal). When no candidate
   qualifies the owner ends the session with exit reason
   ``out_of_fuel`` (``SessionExitError``, 2026-07-02; previously an
   uncaught ``ValueError`` crash) rather than idle silently.
   (Hop picker rewritten 2026-07-01; previously
   first-qualifying-direction with no destination-quality signal.)

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
of the fuel-dot system 2026-06-22; when lock / pickup / sense / hop
all decline the COLLECT owner ends the session with exit reason
`out_of_fuel` (`SessionExitError`, 2026-07-02 — previously an uncaught
`ValueError` crash).[^4]

[^1]: AIConfigDict in bot/ai/types.py — thresholds lowered from 500→300 in Phase 3d (2026-06-14)
[^2]: user (Austin), 2026-06-11 — "only collect fuel containers with volume >= 500"
[^3]: Phase A/B/C of the fuel-dot strip (2026-06-22): planner, state, protocol decoder all stopped surfacing dot coordinates. RLE byte count is still parsed for length validation so the decoder advances cleanly into the MAP_DATA tank-entries section.
[^4]: Run 131003 2026-06-12 — marooned at 87 fuel on one-tile island (actually a ferry; see [[ferry-mechanics]]). Reserve-bypass escape removed 2026-06-22.
[^5]: AIConfigDict 2026-06-22 — `fuel_critical_threshold` field removed; the COLLECT entry predicate (`should_enter_collect`) and the in-cascade fuel-pickup branch now consume `fuel_low_threshold` directly. The unused `try_collect_critical_fuel` / `try_collect_fuel` non-owner helpers and the `_plan_fuel_recovery` wrapper were deleted at the same time. The fuel-mode and equipment-mode owners themselves were then merged into one `decide_collect_mode` 2026-06-24.
[^6]: Live observation 2026-06-23 00:35:57 in `runs/bot/latest.log`: worldstate logged `Fuel: 195 -> 633 (+438)` from the 0x44 FuelGain, then the next AI decision read `ctx.fuel = 1071` (633 + 438). The 438 ghost was the container's volume being added a second time by `pickup_container`'s local fuel-delta branch on top of the wire's already-correct absolute fuel. Removed in `state/container_mutations.pickup_container` 2026-06-23; the function now only mutates the container registry.
