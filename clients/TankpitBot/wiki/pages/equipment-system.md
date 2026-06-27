---
title: Equipment System
tags: [equipment, containers, inventory]
related: [[radar-mechanics]], [[fuel-system]]
sources: [see footnotes]
fact_checked: 2026-06-21
confidence: high
---

# Equipment System

## Priority

- Any inventory item < 10 = must collect equipment ASAP[^1]
- Equipment is CRITICAL — 0 dual shots or 0 radars means severely handicapped[^1]
- Equipment takes priority over fuel top-ups when items are depleted[^1]
- Don't constantly top up fuel; only collect when actually needed[^2]

## Equipment types

Containers hold dual shots, homing shots, and extra radars. In viewport entities: `entity_id == -1` (0xFFFF) = equipment container; `entity_id > 0` = fuel container (entity_id ~ fuel volume).[^3]

## Pickup mechanic (deterministic, NOT random)

A single ``pickup_equipment`` on an equipment container fills the slot you are currently most behind on -- always, deterministically. The amount delivered varies by item type (dual shots arrive in larger stacks than radars / homing). Examples:[^6]

- armor=25, dual=22, missile=25, homing=23, radar=12 → pickup gives 3 dual shots (dual was furthest from cap).
- armor=25, dual=25, missile=25, homing=25, radar=0 → pickup gives a stack of extra radars.
- armor=25, dual=25, missile=25, homing=25, radar=25 → server returns **"Inventory full"**, no pickup, container stays.

Consequence for the bot: **a pickup is never wasted unless every slot is at cap**. Don't pre-filter containers based on "we have enough of X" -- the server picks the right slot, including the one you happen to need.

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
[^6]: user (Austin), 2026-06-21 — "you get inventory items of whatever you need. the equipment isnt determined until pick up. if you have 24 homing shots and full everything else, you'll get 1 homing shot... if you have 25 everything you'll get 'inventory full' and the pickup will fail"
[^7]: protocol/constants.py:147 defines `SUPERVISOR_ERROR_INVENTORY_FULL = 7`. bot/tick_loop_actions.py:44 `_ACTION_BLOCKING_COMMAND_ERRORS` now includes codes 0, 1, 4, 5, 7, 8 (code 7 added 2026-06-21). Two `error_code=7` events recorded in `runs/sniff/latest.events.jsonl` at 19:07:28 / 19:08:30 plus parallel `[GAME:EQUIPMENT] Inventory full` in `runs/sniff/latest.log`.
