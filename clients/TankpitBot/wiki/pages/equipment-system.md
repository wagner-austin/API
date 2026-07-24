---
title: Equipment System
tags: [equipment, containers, inventory]
related:
  - "[[radar-mechanics]]"
  - "[[fuel-system]]"
source_paths:
  - "runs/bot"
  - "runs/sniff"
  - "src/tankpit_bot/sim/equipment.py"
fact_checked: "2026-06-21"
confidence: high
hubs: [game-mechanics]
---

# Equipment System

## Priority

- Any inventory item < 10 = must collect equipment ASAP[^1]
- Equipment is CRITICAL — 0 dual shots or 0 radars means severely handicapped[^1]
- Equipment takes priority over fuel top-ups when items are depleted[^1]
- Don't constantly top up fuel; only collect when actually needed[^2]

## Equipment types

Containers hold dual shots, homing shots, and extra radars. In viewport entities: `entity_id == -1` (0xFFFF) = equipment container; `entity_id > 0` = fuel container (entity_id ~ fuel volume).[^3]

## Pickup mechanic (ARCHIVE-MINED 2026-07-22 — random slot among deficient, typed stack rolls)

The 2026-06-21 user contract described the pickup as "deterministic,
fills the slot you are most behind on". Corpus mining falsified the
determinism: pairing every ``0x67 EquipmentGain`` with its immediately
following ``0x49`` snapshot (``pre = post - gained`` is exact — all
1,154 gains across 246 sessions are followed by their snapshot in the
very next frame) shows:

- **Exactly ONE slot gains per pickup** (the 1,149 ``show_message=True``
  container pickups; see the radar-zero exception below).
- **Hard cap 25** — zero grants pushed a count past it; gains clip at
  the cap (every gain of 1-4 on a weapon slot was a cap-clip).
- **True stack rolls, visible when below the deficit**: dual and
  homing roll **5-9** (uncapped distribution ≈ uniform); radar rolls
  **2-4**. Radar really is the smallest stack — the user's
  "radars are the least frequent" (2026-06-16) survives as amounts,
  not selection odds.
- **Slot choice is RANDOM among deficient slots**, not neediest-first:
  128 grants chose homing while dual was needier, 37 the reverse, and
  89 chose radar while a weapon slot was short. The "you get what you
  need" feel comes from the common state where only one slot is
  deficient. Armor gains when deficient too (2 samples); missile was
  never observed deficient.
- armor=25, dual=25, missile=25, homing=25, radar=25 → server returns
  **"Inventory full"** (0x52 error 7), no pickup, container stays.[^6]

**The radar-zero KILL REWARD (cracked 2026-07-22)**: the archive's 5
``show_message=False`` multi-slot 0x67s are all same-frame with an
own-kill 0x41 — and the trigger is deterministic: **a kill scored
while the killer's extra-radar count is ZERO grants a silent mercy
bundle**. Corpus proof: 5/5 radar-zero kills granted, 0/254 kills at
radar > 0 granted, zero exceptions. Measured amounts: dual +1..4,
homing exactly +1, radar +1..2 — and the bundle may OVERFILL past
the 25 cap (one sample landed dual at 26). Tactical meaning: a
radar-blind kill self-rescues the pursuit loop with one fresh scan.
The sim implements the deterministic medians (+2 dual, +1 homing,
+1 radar; ``SimServer._maybe_emit_kill_mercy_bundle``).

Consequence for the bot unchanged: **a pickup is never wasted unless
every slot is at cap** — the server always picks a deficient slot.

**Container-consumed signal**: no wire message announces the pickup
consumed the container — 0x67 always travels alone (1,154/1,154),
then its 0x49. The client learns a container is GONE by re-clicking
it: the server answers 0x52 error 4 ("empty container") and the bot
deletes the belief (``tick_loop_actions``, code=4 path). The sim
implements exactly this pair ([[physics-module-roadmap]]).

## "Inventory full" wire signal

Server returns 0x52 CommandResult error code **7 (`SUPERVISOR_ERROR_INVENTORY_FULL`)** when the bot dispatches ``pickup_equipment`` while every inventory slot is at 25. The error code is defined in `protocol/constants.py:147` and is in `_ACTION_BLOCKING_COMMAND_ERRORS` (`bot/tick_loop_actions.py:44`) as of 2026-06-21, so today the bot:[^7]

1. dispatches the pickup
2. server rejects with code 7 on 0x52
3. `_clear_command_error` consumes the reject and clears the in-flight action
4. the container's `failed_pickups` counter bumps, surfacing it to the blacklist heuristic
5. next tick replans with a clean slate -- no 10 s stall

Empirical evidence: capture 20260620-190728 / 20260620-190830 delivered `error_code=7` over the wire after pickup dispatches at full inventory (see `runs/sniff/latest.events.jsonl`), with cross-confirming `[GAME:EQUIPMENT] Inventory full` strings in `runs/sniff/latest.log`. Regression guarded by `tests/bot/test_tick_loop_coverage.py::test_command_error_clears_collect_on_inventory_full`. See [[bot-behavior-contract]] §3.4.

## Radar refill

Extra radars come ONLY from equipment containers. No other source. This makes equipment collection the lifeline — at 0 extras the bot is nearly blind. See [[radar-mechanics]].[^4]

## Container blacklisting

The bot picks containers walk-only: `_is_actionable_with_terrain` (`bot/ai/equipment_search.py`) accepts a container only when `is_collection_reachable_in_viewport` finds a walk path inside the current viewport, so unreachable containers are skipped at selection time and never reach a dispatch. When a dispatched pickup fails anyway (server `error_code=1 CANT_GO` from a race condition or partial-pickup `error_code=5 TANK_FULL`), the container's `failed_pickups` counter bumps via `increment_container_failed_pickups` and `is_container_pursuable` excludes it for the remainder of the session.[^5]

The pre-2026-06-26 blacklist was driven by `find_teleport_landing_tile()` returning None on the teleport-to-container path; that path was removed when the user contract collapsed pickups to walk-only. Containers without a walk path now drop out of `find_best_fuel` / `find_nearest_equipment` directly, with no blacklist round-trip.

[^1]: user (Austin), 2026-06-11 — "any inventory item < 10 = must collect equipment ASAP"; user was frustrated bot kept prioritizing fuel over equipment at 0 duals/0 radars
[^2]: user (Austin), 2026-06-11 — "don't constantly top up fuel — only collect when actually needed"
[^3]: viewport entity decode in state/viewport_entities.py — entity_id field mapping
[^4]: three consecutive runs at extras=0 — gained duals/homings but zero radars; see [[radar-mechanics]]
[^5]: run 20260610 — container (91,65) attempted 3 times with "no passable landing tile" each time; 30s TTL was the cause
[^6]: user (Austin), 2026-06-21 — "you get inventory items of whatever you need. the equipment isnt determined until pick up. if you have 24 homing shots and full everything else, you'll get 1 homing shot... if you have 25 everything you'll get 'inventory full' and the pickup will fail". The contents-decided-at-pickup and inventory-full parts are archive-confirmed; the always-the-neediest-slot part is falsified by the 2026-07-22 corpus mining above (random among deficient slots).
[^7]: protocol/constants.py:147 defines `SUPERVISOR_ERROR_INVENTORY_FULL = 7`. bot/tick_loop_actions.py:44 `_ACTION_BLOCKING_COMMAND_ERRORS` now includes codes 0, 1, 4, 5, 7, 8 (code 7 added 2026-06-21). Two `error_code=7` events recorded in `runs/sniff/latest.events.jsonl` at 19:07:28 / 19:08:30 plus parallel `[GAME:EQUIPMENT] Inventory full` in `runs/sniff/latest.log`.
