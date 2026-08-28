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
source_git_blobs:
  "src/tankpit_bot/sim/equipment.py": "b8b334fef24b91465d7142edc966f2b9ea4dd398"
fact_checked: "2026-08-28"
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

The 2026-06-21 user contract[^6] described the pickup as "deterministic,
fills the slot you are most behind on". Corpus mining falsified the
determinism: pairing every ``0x67 EquipmentGain`` with its immediately
following ``0x49`` snapshot (``pre = post - gained`` is exact — all
1,154 gains across 246 sessions are followed by their snapshot in the
very next frame) shows:[^8]

- **Exactly ONE slot gains per pickup** (the 1,149 ``show_message=True``
  container pickups; see the radar-zero exception below).
- **Cap is RANK-DERIVED, not a flat 25** — `20 + 5 * rank`
  (`physics/capacity.py:103` `inventory_capacity`, sourced from the
  official rules table: recruit 20, +5 per rank). Zero grants pushed a
  count past the cap; gains clip at it (every gain of 1-4 on a weapon
  slot was a cap-clip). **Corrected 2026-08-06:** this row previously
  read "Hard cap 25" as an absolute. 25 is `inventory_capacity(1)` —
  the Private value — and the corpus it was measured from was
  all-private, so the rank term was invisible. Recruit cap 20 is
  confirmed by the `bot-20260725-211120` promotion crossing, where slot
  counts crossed 20 only after the promoting kill. Note the cap moves
  at the promoting KILL while the wire rank field stays stale, so the
  true current rank is what governs.
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
- every slot at the rank cap (armor/dual/missile/homing/radar all 25 in
  the observed all-Private corpus) → server returns **"Inventory full"**
  (0x52 error 7), no pickup, container stays.[^6]

**The radar-zero KILL REWARD (cracked 2026-07-22)**: the archive's 5
``show_message=False`` multi-slot 0x67s are all same-frame with an
own-kill 0x41 — and the trigger is deterministic: **a kill scored
while the killer's extra-radar count is ZERO grants a silent mercy
bundle**. Corpus proof: 5/5 radar-zero kills granted, 0/254 kills at
radar > 0 granted, zero exceptions.[^8] Measured amounts: dual +1..4,
homing exactly +1, radar +1..2 — and the bundle may OVERFILL past
the 25 cap (one sample landed dual at 26). Tactical meaning: a
radar-blind kill self-rescues the pursuit loop with one fresh scan.
The sim implements the deterministic medians (+2 dual, +1 homing,
+1 radar; ``SimServer._maybe_emit_kill_mercy_bundle``,
[[physics-module-roadmap]]).

Consequence for the bot unchanged: **a pickup is never wasted unless
every slot is at cap** — the server always picks a deficient slot.[^8]

**Container-consumed signal**: no wire message announces the pickup
consumed the container — 0x67 always travels alone (1,154/1,154),
then its 0x49. The client learns a container is GONE by re-clicking
it: the server answers 0x52 error 4 ("empty container") and the bot
deletes the belief (``tick_loop_actions``, code=4 path). The sim
implements exactly this pair ([[physics-module-roadmap]]).

**Own-tile pickup law (live-proven 2026-07-27)**: the server honors
``pickup_equipment`` targeting the tank's OWN tile — the larder
probe (`make larder-probe`) landed teleports exactly on verified
equipment containers (5/6 aimed landings) and credited the own-tile
pickup 3/3 with zero 0x52 errors (run `larder-20260727-230858`).
Equipment container tiles are walkable and never auto-pick — unlike
fuel, the pickup command is always required (user law 2026-07-27,
[[fuel-system]] for the fuel auto-pick contrast). Adjacent pickup
re-confirmed the same day (run -225643 attempt 1). This unlocks the
[[larder-plan]] harvest atom: teleport ON the container, one pickup
command, done.[^10]

## Death halves every inventory slot, rounding up — and the one mine death wiped it all (WIRE-VERIFIED 2026-08-28)

Found by the first `make corpus-audit` sweep, confirmed by the user
("a death causes you to lose half inventory or so"),[^14] then pinned
frame-exact by decoding the raw 0x49 inventory snapshots bracketing
every self 0x41 in the four death-run captures:[^13]

| capture | death | last 0x49 before (armor,dual,missile,homing,radar) | first 0x49 after | kill type |
|---|---|---|---|---|
| desert 18:22 | frame 2609 | 45, 9, 45, 37, 24 | 23, 4, 23, 19, 12 | tank (719) |
| desert 18:22 | frame 3248 | 40, 33, 40, 39, 38 | 20, 15, 20, 18, 19 | tank (719) |
| desert 18:22 | frame 3840 | 35, 35, 35, 35, 29 | **0, 0, 0, 0, 0** | **MINE** (team 3) |
| artax 08:48 | frame 4699 | 25, 25, 25, 25, 24 | 13, 10, 13, 13, 12 | tank (569) |
| 08-26 00:39 | frame 8114 | 25, 18, 25, 25, 9 | 13, 4, 13, 11, 5 | tank (709) |
| 08-03 18:09 | frame 4453 | 25, 17, 25, 25, 11 | 13, 7, 13, 12, 6 | tank (2678) |

The law, from the slots shooting cannot consume (armor, missile,
radar — a dual+homing loadout never spends them): **a tank-kill death
sets every slot to ceil(n/2)** — 45→23, 40→20, 25→13 (x6), 24→12
(x2), 38→19, 9→5, 11→6, zero exceptions across five deaths. The
dual/homing columns agree once shots served between the last snapshot
and the death are allowed. The mine death is DIFFERENT: everything
went to a genuine zero — rebuilt afterwards purely through 0x67
pickup gains (0 → +5 homing → +9 armor → ...), so it was no respawn
re-sync race. One sample cannot yet separate "mine kills wipe
everything" from "a third death in one session wipes everything";
the next mine death on a first-death session decides it.

Consequences: the ammo book and the corpus audit's radar expectation
now MODEL the death penalty (`record_ammo_death`: ceil-halve on a
tank kill, zero on the mine sentinel, applied at the self 0x41 —
2026-08-28), so death-runs no longer burn false ammo divergences or
drift flags; dying costs rank AND half the restock — and possibly ALL
of it to a mine; and a victim we kill respawns half-stocked, so
immediate re-pressure after a kill is disproportionately favorable.

[^13]: Frame-exact decode 2026-08-28: `decode_session_frames` + `decode_message` over `runs/bot/desert/bot-20260826-182204`, `runs/bot/artax/bot-20260826-084859`, `runs/bot/bot-20260826-003928`, `runs/bot/bot-20260803-180918` capture sessions — for each 0x41 naming the session's own tank (ids 716/601/601/1301), the nearest 0x49 on each side, counts read per `decode_inventory` (`byte & 127`). Mine attribution via the 0x41 killer sentinel (`killer_id_raw >= 65530`, residual = mine team; see [[deactivation-format]]). Corpus context: `tankpit-corpus-audit` over 436 runs put drift flags ONLY on death-runs (-5, -5, -3, -60) plus two minor unexplained +2/+7 on 08-13.
[^14]: user (Austin), 2026-08-28 -- "a death causes you to lose half inventory or so", confirming the corpus-audit inference the same day; the frame decode then fixed "or so" to ceil(n/2) for tank kills.

## Equipment spawns cluster at persistent HOTSPOTS (FULL-CORPUS MINED 2026-08-28)

The operator's hypothesis ("its probably hotspot based") holds on
both maps. Full-corpus mining of own-viewport equipment sightings
(`entity_alignment_sample` beliefs, `is_fuel: false`, 273 field01
runs + 12 field05 runs):[^15]

- **field01 (practice)**: 5,193 distinct tiles ever seen, but
  persistence is wildly concentrated: (123,123) appears in **101 of
  273 runs**, (128,126) in 99 — and seven of the ten most persistent
  tiles sit in ONE ~12x12 patch around (123-134, 123-134), the map
  center. Secondary hotspots: (180,47), (114,152)/(112,154),
  (101,203), (112,190), (139,133).
- **field05 (main)**: even at 12 runs, tiles recur in 7-8 of them,
  in crisp clusters: (59-65, 13-16), (50-58, 166-170), (176,212).

CORRECTION (same day): the first two-run comparison read "field05
5% tile stability" — that was COVERAGE BIAS (the two runs roamed
different areas), not spawn churn. Run-count persistence is the
honest measure, and by it both fields carry stable hotspots; the
long once-seen tail is spawn scatter plus roaming bias. Atlas data:
`runs/analysis/equipment-atlas-20260828.json` (per-field tile
persistence counts). Design consequence unchanged and strengthened:
foraging = teleport the hotspot circuit; blind paid sweeps are
strictly worse than the lookup (the 2026-08-28 validation run spent
9 extras on sweeps that found zero while the atlas held 5k mapped
tiles).

[^15]: Full-corpus miner 2026-08-28 over every `runs/bot` events artifact (1-in-20 belief-snapshot sampling, own-viewport sources only — `source != "world_state"` excludes fleet imports). Per-field distinct tiles / seen-in-2+-runs / top persistence lists archived in `runs/analysis/equipment-atlas-20260828.json`. The earlier 4-run pilot (382-tile field01 day-over-day set, 90% recurrence) stands as the freshness measure; the run-count table is the stability measure.

## "Inventory full" wire signal

Server returns 0x52 CommandResult error code **7 (`SUPERVISOR_ERROR_INVENTORY_FULL`)** when the bot dispatches ``pickup_equipment`` while every inventory slot is at the rank cap (`20 + 5 * rank`; 25 at Private, which is where this was observed). The error code is defined in `protocol/constants.py:147` and is in `_ACTION_BLOCKING_COMMAND_ERRORS` (`bot/tick_loop_actions.py:44`) as of 2026-06-21, so today the bot:[^7]

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

The pre-2026-06-26 blacklist was driven by `find_teleport_landing_tile()` returning None on the teleport-to-container path; that path was removed when the user contract collapsed pickups to walk-only. Containers without a walk path now drop out of `find_best_fuel` / `find_nearest_equipment` directly, with no blacklist round-trip.[^9]

[^1]: user (Austin), 2026-06-11 — "any inventory item < 10 = must collect equipment ASAP"; user was frustrated bot kept prioritizing fuel over equipment at 0 duals/0 radars. The ordering that ruling forced is the COLLECT cascade in `decide_collect_mode` at `src/tankpit_bot/bot/ai/collect_mode.py:199`, where equipment pickup precedes fuel pickup; the readiness floor that keeps HUNT from starting under-stocked is `combat_radar_min(rank)` at `src/tankpit_bot/bot/ai/mode_gates.py:229`, applied at `:288`.
[^2]: user (Austin), 2026-06-11 — "don't constantly top up fuel — only collect when actually needed". Encoded as the single COLLECT entry predicate `should_enter_collect` at `src/tankpit_bot/bot/ai/mode_gates.py:16`, gated on `fuel_low_threshold` (200, `src/tankpit_bot/bot/ai/types.py:110`) rather than on any "top up when convenient" rule.
[^3]: viewport entity decode in state/viewport_entities.py — entity_id field mapping
[^4]: three consecutive runs at extras=0 — gained duals/homings but zero radars; see [[radar-mechanics]]. The death-spiral this describes is why the last extra radar is rationed by reveal value rather than spent freely: `RADAR_RESERVE_REVEAL_FLOOR_TILES = 128` at `src/tankpit_bot/bot/ai/context.py:392`, whose docstring states "once it is gone, discovery collapses to the built-in radius-2 scan and restock stalls".
[^5]: run 20260610 — container (91,65) attempted 3 times with "no passable landing tile" each time; 30s TTL was the cause
[^6]: user (Austin), 2026-06-21 — "you get inventory items of whatever you need. the equipment isnt determined until pick up. if you have 24 homing shots and full everything else, you'll get 1 homing shot... if you have 25 everything you'll get 'inventory full' and the pickup will fail". The contents-decided-at-pickup and inventory-full parts are archive-confirmed; the always-the-neediest-slot part is falsified by the 2026-07-22 corpus mining above (random among deficient slots).
[^7]: protocol/constants.py:147 defines `SUPERVISOR_ERROR_INVENTORY_FULL = 7`. bot/tick_loop_actions.py:44 `_ACTION_BLOCKING_COMMAND_ERRORS` now includes codes 0, 1, 4, 5, 7, 8 (code 7 added 2026-06-21). Two `error_code=7` events recorded in `runs/sniff/latest.events.jsonl` at 19:07:28 / 19:08:30 plus parallel `[GAME:EQUIPMENT] Inventory full` in `runs/sniff/latest.log`.
[^8]: corpus sweep 2026-07-22 (wiki log): 1,154 0x67→next-frame-0x49 pairs across 246 archive sessions; standing re-derivation on every `make shadow` (grant-invariants 1,149/1,149 + kill-mercy-bundle 283/283 laws, `src/tankpit_bot/validate/shadow_laws.py`)
[^9]: user contract 2026-06-26 (walk-only pickups) — selection sites `bot/ai/equipment_search.py`; the removed teleport-to-container path predates the current `_is_actionable_with_terrain` gate
[^10]: larder probe artifacts on disk: `runs/probe/larder-20260727-230858.json` (own-tile 3/3, no 0x52 errors) and `-225643.json` (adjacent credit at attempt 1, code-7 receipts once capped) with paired `.log` + `.capture_session.json`; walkable/no-auto-pick is the user's law (Austin, 2026-07-27 — "you csn walk onto an ewuipment contsiner. theyre notnobstcles. it justxdoesnt pifkcup automstically"); probe source `src/tankpit_bot/action_lab/larder_probe.py`
