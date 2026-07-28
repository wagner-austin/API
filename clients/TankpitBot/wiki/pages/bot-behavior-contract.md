---
title: Bot Behavior Contract
tags: [bot, contract, verification]
related:
  - "[[client-state-machine]]"
  - "[[decode-coverage]]"
  - "[[tank-freshness-model]]"
  - "[[gameplay-loop]]"
source_paths:
  - "src/tankpit_bot/bot"
  - "src/tankpit_bot/sniffer"
  - "runs/bot/latest.events.jsonl"
source_git_blobs:
  "src/tankpit_bot/bot": "668435779bed6f88c76f2fd46f61a4bc6e9952a6"
  "src/tankpit_bot/sniffer": "7075bc983a306e326444c4111bdab64a5bc5d4e0"
fact_checked: "2026-07-25"
confidence: high
verified: 2026-06-20 (anchored to specific code paths + integration tests)
hubs: [architecture]
---

# Bot Behavior Contract

The single source of truth for *what the bot must do* and *how each behavior is verified*. When proposing a fix, consult this page first to know what else might break.[^1]

This contract complements [[client-state-machine]] (the JS client's state machine) and [[tank-freshness-model]] (our wire-presence semantic). The state machine describes the *server protocol*; this page describes *our bot's obligations*.

Format: each row in each section has **MUST / MUST NOT / Verified by**. "Verified by" names the smoke assertion and integration test that locks the behavior in. If a behavior is not yet verified, that's flagged explicitly — every claim here must eventually point to a test. The test idioms those rows name (and the DI rules they follow) are in [[testing-patterns]].

## 1. Lifecycle

### 1.1 Startup

| Aspect | Contract |
|---|---|
| MUST | Transition `INITIALIZING → WAITING_FOR_POSITION → IDLE` within `action_stall_timeout_ms` (10 s default). |
| MUST | Establish `self_state` (tank id, position, team) before leaving `WAITING_FOR_POSITION`. |
| MUST NOT | Enter HUNT or any decision mode while `self_state` is None. |
| Verified by | `smoke[1]` (login completes); `tests/integration/test_bot_login.py` (TBD). |

### 1.2 Session-end (graceful)

| Aspect | Contract |
|---|---|
| MUST | When `TANKPIT_BOT_SESSION_SECONDS` elapses, flush `runs/bot/latest.events.jsonl`, write `latest.summary.txt`, append to `runs/bot/_index.tsv`. |
| MUST | Exit cleanly (return code 0) even mid-action. |
| MUST | **Self-directed exits (2026-07-02):** decision owners end the session by raising `SessionExitError(reason, detail)`; `run_tick_loop` converts it to the same graceful shutdown as a tick-budget exit with `exit_reason` set to the request's reason. Current reasons: `no_viable_targets` (HUNT: fresh map, nothing affordable/engageable), `out_of_fuel` (COLLECT: no lock, no pickup, no forage, no affordable hop — previously an uncaught `ValueError` crash), `no_productive_collect`, `deactivated`, and `session_complete` (the wind-down exit, below). |
| MUST | **Session wind-down (2026-07-26):** two triggers raise `ai_state["wind_down"]` — the final 60 s of a bounded run (sessions > 120 s only — short diagnostic runs keep the full loop), and the kill target (`TANKPIT_BOT_SESSION_KILLS`: the Nth session kill, the natural clean-exit boundary — user follow-up: "maybe if we put it for kills instead of time based?"). Winding down, the mode selector FINISHES a live locked fight first (never abandon a target — the 2026-07-25 ruling; break thresholds still protect), opens no new engagements, collects to full, and exits `session_complete` the moment the tank is fully stocked (or immediately when nothing collectable remains). User request (verbatim): "we cant have it like run and then collect and exit cleanly? instead of the program killing it on 10 min mid action". Ending stocked is what makes the NEXT session open combat-ready — run bot-20260726-002554's t+30 s first kill ran on the prior session's leftover stock. Live proof: run bot-20260726-004729 exited `session_complete` at 273 s, fully stocked at 1100, 27 s before the clock. The hard tick budget stays as the backstop. |
| MUST NOT | Leave `latest.*` symlinks pointing at a prior run's data. |
| Verified by | `tests/integration/test_session_shutdown.py` (TBD); `runs/bot/_index.tsv` row appended (Tier 3.3, handed off); `SessionExitError` handling in `tests/bot/test_tick_loop_coverage.py`. |

### 1.3 Session-end (interrupted: SIGINT / SIGTERM / crash)

| Aspect | Contract |
|---|---|
| MUST | Flush events JSONL before exit. |
| MUST | Write `latest.summary.txt` with `exit_reason=interrupted` (or `crashed`). |
| MUST | Append to `_index.tsv` so the run is discoverable. |
| MUST NOT | Lose events emitted in the final tick. |
| Verified by | `tests/integration/test_signal_handler.py` (TBD; Tier 3.4, handed off). |

## 2. Perception (world state)

### 2.1 Tank tracking

| Aspect | Contract |
|---|---|
| MUST | Every tank in `world_state["tanks"]` carries the three freshness timestamps (`timestamp_ms`, `last_wire_seen_ms`, `last_position_update_ms`) plus a `liveness` field. See [[tank-freshness-model]]. |
| MUST | `timestamp_ms` advances on every observation (wire or map). |
| MUST | `last_wire_seen_ms` advances ONLY on `is_wire_sourced=True` observations. |
| MUST | `last_position_update_ms` advances ONLY on `is_wire_sourced=True` observations that also carry `position`. |
| MUST | `liveness` is one of `alive` / `deactivated`. 0x41 Deactivation → `deactivated`; corpse-direction wire (`direction >= 32`) → `deactivated`; wire-sourced position update with alive direction → `alive`. |
| MUST | 0x58 TankRemove is a NO-OP at the registry level. 0x58 is *tracking removal*, not a kill — verified 2026-06-20 (orange-5 got 5 TankRemove events across 2 actual kills). The earlier behaviour deleted the entry from `tanks`; that caused the bot to abandon pursuit of locked targets that merely teleported out of viewport (live capture 2026-06-22 — bot fired 1 homing then dropped the lock). Keeping the entry lets `find_locked_target_pursuit` keep firing toward the cached coords until 0x41 Deactivation arrives or `timestamp_ms` goes stale. |
| MUST NOT | Treat 0x58 as a death signal. Use `0x41 Deactivation` (it flips `liveness="deactivated"` and routes through the kill cooldown). |
| Verified by | `tests/world_state/test_mutations.py::TestRemoveTank::test_keeps_tank_in_registry` (0x58 is no-op); `tests/world_state/test_mutations.py::TestDeactivateTank` (0x41 sets liveness); `tests/bot/ai/test_hunt_mode.py::test_hunt_engage_fires_homing_when_locked_target_left_viewport` (pursuit fires while target is out of viewport). |

### 2.2 Position correctness

| Aspect | Contract |
|---|---|
| MUST | Tank positions reflect the latest authoritative source: MovementResponse (0x3D) > MapData (0x4C) > TankEntry (0x28) > TankInfo (0x21 — no position) > viewport overlay. |
| MUST | MapData updates positions for ALL tanks regardless of liveness. (A `deactivated` tank's position can still be informational; a `removed` tank doesn't exist in the registry, so this is moot.) |
| MUST NOT | Believe a `(0, 0)` position as live — that's the unsynced-tank sentinel (`analyze_threats` filters tanks at the origin). |
| Verified by | `tests/sniffer/test_world_state_dispatch_tank.py::TestDispatchMapData::test_map_data_lifts_tank_positions`. |

### 2.3 MapData processing

| Aspect | Contract |
|---|---|
| MUST | Mark `world_service.mark_map_data_processed()` after applying MapData tank observations. This is what `_clear_completed_map_open` polls. |
| MUST | Apply MapData position updates for every listed tank (no skip for any liveness state). MapData is authoritative for "where this tank is right now". |
| MUST | Store the decoded fuel-dot atlas on `WorldService.map_fuel_dots` (overwritten every map open — the atlas is server-cached per session). Consumed by the COLLECT dot hop and the HUNT dot relay (2026-07-03). |
| Verified by | `tests/sniffer/test_world_state_dispatch_tank.py::TestDispatchMapData::test_map_data_marks_action_complete` (locked in 2026-06-20); `test_map_data_lifts_tank_positions`. |

### 2.4 Container/mine registries (per-tile truth)

| Aspect | Contract |
|---|---|
| MUST | **0x5A reset-then-apply (2026-07-03):** a landing viewport patch is a complete visible-layer statement for its 18x18 grid — the skip-walk covers every tile, so a tile it does not enumerate has nothing on it. Visible-layer (non-radar-sourced) container and mine entries on silent tiles inside the patch bounds are removed. Mirrors the JS client, which wipes the tile grid and rebuilds from the patch alone. Radar-sourced entries are spared (owned by the radar omission-prune; reveal persistence across teleports is unverified). Before this, a container remembered from a previous visit survived re-entry after someone consumed it — a ghost pickup target the 0x5A had already disclaimed. |
| MUST | **Radar response = delta sync (2026-07-03):** apply every 0x4F entry as a per-tile write — cache value 0 removes the container at the tile (`update_container_from_radar`), overlay >= 8 removes the mine (`remove_mine`), never a phantom team-255 mine. Corpus proof: 247 of 2093 cache entries across 199 sessions were removals. Unchanged visible entities are not re-sent, so the response is never the complete viewport set. See [[radar-mechanics]]. |
| Verified by | `tests/sniffer/test_world_state_dispatch_viewport.py::TestViewportPatchSweep`; `tests/protocol/test_radar.py::TestDecodeRadarScanResult::test_decodes_container_removal_entry`, `test_decodes_overlay_clear_as_mine_clear`; `tests/integration/test_tier2_lifecycle_signals.py::TestRadarScanReturnsToIdle`. |

## 3. Decision-making

### 3.1 Mode selection

| Aspect | Contract |
|---|---|
| MUST | Mode selection is deterministic given (world_state, self_state, config). No randomness. |
| MUST | **Hunt only when full (user contract 2026-07-25):** HUNT entry requires `fuel >= fuel_capacity(rank)` AND duals+homings at `inventory_capacity(rank)` AND extra radars >= cap-5, every bar rank-derived ("it should never hunt if it is not full on everything except -5 max radar"; "just determine max fuel based on the tank rank"). Nothing overrides this: the 2026-07-13 cardinal-adjacent override was deleted 2026-07-25 after it produced the practice-room fight-to-death (see §5). A held HUNT releases only on a COLLECT trigger, so spending ammo mid-fight does not thrash ownership. |
| MUST | COLLECT entry uses `should_enter_collect`, which fires when ANY of: (1) `fuel <= fuel_low_threshold` (interrupts even an active combat target); (2) **Weapon emergency** -- any reserve below its *break* threshold (dual / homing < 4 or radar < 5) -- interrupts even with an active combat target; (3) **Between kills** -- `combat_target_id == -1` AND anything short of the full-tank bar above. The unified COLLECT mode replaced the historical `RECOVER_FUEL` + `RECOVER_EQUIPMENT` split 2026-06-24; the fixed resume thresholds (25/25/20) were replaced by the rank caps 2026-07-25. |
| MUST NOT | Enter HUNT while `self_state` is None (lifecycle 1.1). |
| Verified by | `tests/bot/ai/test_mode_controller.py`; `tests/bot/ai/test_strategy_coverage.py::TestHuntOnlyWhenFull`. |

### 3.2 HUNT acquisition

| Aspect | Contract |
|---|---|
| MUST | Use `analyze_threats` to score candidate enemies. Sort by distance, then finish-priority, then freshness. |
| MUST | Filter out: self, allies, unsynced `(0,0)`, `liveness != "alive"` (catches both direct 0x41 deactivations and corpse-direction wire arrivals via `apply_tank_observation`), stale `timestamp_ms` older than `WIRE_PRESENCE_TTL_MS`. |
| MUST | **Affordability gate (2026-07-02):** a map-known candidate is viable only when `fuel >= teleport_cost(candidate) + engagement_fuel_budget + fuel_low_threshold`. The bot never picks a fight it cannot pay for end-to-end. With the 1100 fuel cap and a ~450 kill budget this caps engagement range at ~58 tiles — matching the recorded human maximum of 60. Rejection reason `unaffordable` appears in the `acquisition_candidates` diagnostic. (Live counterexample: run 2026-07-01 20:45 spent 505 fuel reaching the nearest enemy and hit the fuel-low interrupt 8 shots in.) |
| MUST | **Resume-to-target (2026-07-25, supersedes the 2026-07-02 stale-lock release):** a lock reaching ACQUIRE with its target off-viewport (resumed after a mode interrupt) is PURSUED, not dropped — the restock cycle does not abandon the target ("why wouldnt we refuel and restock and then go back to the target"; damage persists, so the sortie cycle wins even 3v1). Teleport back on a trustworthy position; map-refresh a stale one; release only when the target is gone (dead / not in the registry). An unaffordable return (teleport + kill budget + fuel-low reserve exceeds fuel, the 2026-07-02 engagement gate) REFUELS WITH THE LOCK HELD and resumes once fundable (refuel-then-resume, user ruling 2026-07-27 -- the old release at this gate lost run 183703's red-1 to a fresh distance race). Never fire from stand-off range on resume — the server rejects shot aims outside the viewport. |
| MUST NOT | **Enable the armor slot (user ruling 2026-07-27: "no armor. its an advanced item you dont know how to use").** The tank may CARRY shields (25 rode through run 183703) but the enable set stays `dual,homing,radar`. No code may add `armor` to the equip set; revisit only on an explicit new ruling. |
| MUST | **Never drop a live target (user ruling 2026-07-26, implemented 2026-07-27):** the post-reroute-window consumption-miss (the target escaped past the measured 12.92 s wall, [[shoot-event-format]]) HOLDS the lock and opens the map (`target_chase` diagnostic); the ACQUIRE pursuit machinery chases the refreshed position. The lock releases ONLY when the registry says dead or vanished (`target_departed`, reason `gone_from_registry`); an unaffordable return defers via refuel-then-resume with the lock held (2026-07-27, commit a10fbcec), with the nothing-collectible block fallback as the terminator. Priced by the 2026-07-26 census: the former release cost ~1 escaped kill per 10-kill run (7 departures / 3 permanent escapes across 4 runs; orange-9 absorbed 13 hits and won the distance lottery). First live firing (run bot-20260727-083526): chase on respawned orange-8 → killed 51 s later; run ended 10 kills, 0 escapes, 0 releases. |
| MUST | Open the map (`map_open` command) when no candidate is found and the last snapshot is stale (older than `map_open_cooldown_ms`). |
| MUST | **Dot relay toward unaffordable enemies (2026-07-03):** when the fresh map has an enemy that fails ONLY the affordability gate, teleport to the fuel dot that best closes distance to it — strictly closer than the current tile (monotone, terminates), passable landing, and leaving `fuel_low_threshold` behind so a dry dot cannot strand the bot below the COLLECT reserve. Landing auto-pickup refuels the leg; acquisition re-runs next map. This is the user's own play: "yellow dot teleporting while en route to the opponent." |
| MUST | **Exit on no viable targets (2026-07-02, refined 2026-07-03 and 2026-07-19):** when the map snapshot is FRESH (within `map_open_cooldown_ms`), no enemy passes the gates, AND no relay dot makes affordable progress, the bot first tries **refuel-in-place** — hop to the best fresh fuel dot in ANY direction (`hunt_refuel`, via the COLLECT restock picker) to fund a future engagement instead of approaching it. Only when the tank is at `fuel_capacity(rank)` or no fresh dot qualifies does it raise `SessionExitError("no_viable_targets", ...)`. Rationale (run 2026-07-19 14:49): rejoined at fuel 653 with an enemy 26.6 tiles away, 622 usable dots, and only water-locked dots strictly closer — the strict-progress-only relay starved the bot amid plenty; the deficit was fuel, not distance. |
| MUST | After `map_open`, wait for the authoritative completion signal `map_data_processed` (set by `_dispatch_map_data` via `ws.mark_map_data_processed()`). |
| MUST NOT | Re-issue `map_open` while a prior `map_open` is in flight and within `action_stall_timeout_ms`. |
| MUST NOT | Fire `make_radar_command()` to "search for enemies". Radar does not reveal enemies (see [[radar-mechanics]]); enemy discovery is map-open + viewport-edge walking only. |
| MUST NOT | Walk as a travel mechanism (user ruling, verbatim, 2026-07-25: *"walking is too slow... we teleport for a reason. we walk for equipment and fuel pickups in the same viewport. but no we're not walking across the map or to enemies"*). Walking is for in-viewport pickups and sense-shuffles only; travel — to enemies, to dots, across the map — is teleport. The wire technically supports autoscroll-ON window-following walks ([[viewport-shift-protocol]]) at ~2 s per tile round-trip; that is a correctness fact, not a strategy. |
| Verified by | `tests/bot/ai/test_threats.py::TestFindAcquisitionTarget::test_rejects_unaffordable_enemy`, `test_picks_affordable_enemy_over_nearer_unaffordable`; `tests/bot/ai/test_hunt_mode.py::test_hunt_acquire_teleports_back_to_an_affordable_off_viewport_lock`, `test_hunt_acquire_returns_to_the_locked_target_after_a_mode_interrupt`, `test_hunt_acquire_refreshes_a_stale_locked_position_via_map`, `test_hunt_acquire_releases_stale_lock_when_target_unaffordable`, `test_hunt_acquire_exits_when_fresh_map_has_no_viable_targets`, `test_hunt_acquire_relays_via_dot_toward_unaffordable_enemy`, `test_hunt_relay_prefers_dot_nearest_the_enemy`, `test_hunt_relay_tie_breaks_on_cheaper_hop`, `test_hunt_relay_exits_when_only_dot_is_impassable`, `test_hunt_refuels_in_place_when_no_dot_makes_progress`, `test_hunt_refuel_exits_at_fuel_capacity`; `tests/bot/ai/test_enemy_search.py::test_fallback_exits_when_fresh_map_shows_no_viable_target`. |

### 3.3 Combat shoot gates (`_combat_shoot`)

| Aspect | Contract |
|---|---|
| MUST | **Hit/miss = per-shot ammo consumption (2026-07-02).** The ShootEvent `weapon` field is the server's per-shot ammo ledger: `weapon > 0` = one dual/missile/homing debited = HIT (even with `victim_id=-1` on off-viewport pursuit impacts); `weapon = 0` = free single at empty ground = MISS. `victim_id` is kill-attribution metadata only, never the hit discriminator. (Wire proof: run 2026-07-02 01:21, five `weapon=3` `victim_id=-1` pursuit homings killed orange-3 while the old classifier logged five misses.) |
| MUST | **Viewport-clamped aim (2026-07-03):** every `shoot` dispatch aims inside the visible 16x16 viewport — the server rejects any aim outside it with 0x52 code 0 ("You can't do this"; live run 2026-07-03 20:34 drew five rejections aiming at a pursuit target 5 rows below the viewport). The aim is only a hint: the wire-proven snipe fires at an in-viewport ground tile with the target's `tank_id` and the server picks homing, whose seeker tracks the real target (`weapon=3` hit from the target's vacated tile, same run). Registry truth is untouched — `_clamp_aim_into_viewport` clamps only the dispatched coordinate, and only when the viewport record contains the bot (a record that excludes the bot is stale/unestablished). Necessary because 0x3D MovementResponse broadcasts every map tank's position ~every 2 s, so pursuit registry coords track true off-viewport tiles. |
| MUST | **Rejected-shot feedback (2026-07-03):** a shot-rejecting 0x52 code (0 "can't do this", 3 "friendly fire", 8 "insufficient fuel") arriving while a shot is pending is authoritative refusal — no ShootEvent and no ammo delta will ever come. The feedback wait ends immediately (no 4 s dead window), the outcome is `rejected` (neither hit nor miss; counted separately in the scorecard via `session_reject_count`), and the planner blocks the target and replans — repeating an identical refused dispatch cannot change the answer. Non-shot codes (e.g. 7 "Inventory full") are left for the in-flight-action machinery. |
| MUST | Gate 1: `is_wire_present(target["last_wire_seen_ms"], now)` — guard against firing at wire-stale ghosts. |
| MUST | Gate 2: `is_position_fresh(target["last_position_update_ms"], now)` — guard against firing at a stale position. |
| MUST | On gate failure, call `block_combat_target_and_replan` (does NOT fire, picks a different target after cooldown). |
| MUST | On miss against a moved target, re-aim at the new position and keep the lock. |
| MUST | On miss against a stationary target, `block_combat_target_and_replan` — a consumption-miss at an unmoved registry position proves the tank is not there; repeating the shot cannot change the answer (run 2026-07-02 01:23: 25+ `weapon=0` shots looped at orange-1's stale tile before this was enforced — the rule existed in this contract but the code only re-aimed). |
| MUST NOT | Fire if either gate fails. |
| RESOLVED | Stationary practice-room bots pass these gates correctly under the 2-state liveness model. The earlier "ghost cache" concern (#75/#77/#80) was a false reading of the corpus — verified 2026-06-20 that tanks disappear from MapData immediately on kill; no server-side cache of dead tanks at kill tiles exists. |
| Verified by | `tests/bot/ai/test_hunt_feedback.py::test_miss_on_stationary_far_target_blocks_and_replans`, `test_miss_on_adjacent_stationary_target_blocks`, `test_miss_on_moved_target_reaims_and_keeps_lock`, `test_rejected_shot_blocks_target_and_replans`; `tests/bot/ai/test_hunt_mode.py::test_hunt_pursuit_aim_is_clamped_into_viewport`; `tests/bot/test_cdp.py::test_get_combat_feedback_rejected_on_command_error`, `test_get_combat_feedback_ignores_non_shot_command_error`, `test_has_pending_shot_feedback_ends_wait_on_command_error`; `tests/sniffer/test_world_state_dispatch_container.py::TestDispatchShootEvent`; `tests/integration/test_tier2_lifecycle_signals.py::TestCombatHitAdvancesDamageState`. |

### 3.4 COLLECT (unified fuel + equipment collection)

| Aspect | Contract |
|---|---|
| MUST | COLLECT runs a single cascade per tick: (1) continue a held equipment or fuel lock from a previous tick; (2) **Scan-on-landing** -- fire one radar per viewport entry, BEFORE any pickup (see next row); (3) pick up the best equipment in the current viewport; (4) pick up the best fuel in the current viewport (skipped at learned capacity); (5) **Sense** -- radar when the viewport has unscanned tiles, or walk toward an unscanned tile so the next free radar covers it; (6) **Hop** -- teleport to the best-value fuel dot, RANKED by `dots_in_landing_viewport * walkable_fraction / cost` with physics-only hard gates (passable landing, affordable, and a CLEAN landing viewport — **zero overlap with live scan coverage** (user ruling, verbatim, 2026-07-26: "when i say it should collect on clean viewports, that means zero overlap"; the 2026-07-18 gate had the polarity inverted — any unscanned tile counted as fresh — and run bot-20260725-235637 re-scanned a mean 89/256 tiles per hop, the radar-treadmill session). Coverage marks age out on the 180 s forage TTL. The 2026-07-03 100%-walkable hard filter was separately replaced 2026-07-18 after it rejected 428 of 622 dots — user ruling "more dots, more walkable area. but not a 100% rule"). Equipment ranks ahead of fuel by design: the user's gameplay loop is "pick up all equipment, then maybe the biggest fuel container, then hop". |
| MUST | **Unconditional scan-on-landing (2026-07-03):** the landing radar fires once per viewport entry via the `last_landing_scan_viewport` latch in `AIStateDict` (the viewport changes only on teleport, so origin-differs = just landed). HUNT's combat-landing scan records the same latch so a later COLLECT entry in the same viewport does not double-fire. User policy: "I usually always use radar right on landing from teleport" — the 0x5A patch is truthful for the visible layer but says nothing about hidden containers, and re-entering previously scanned ground is exactly when coverage marks are most stale. (The earlier zero-coverage gate skipped the scan whenever the 18-wide visible viewport overlapped 2 tiles of old coverage after a hop.) |
| MUST | Exit (`should_exit_collect`) holds the mode until the tank is genuinely FULL: `fuel >= fuel_capacity(rank)` AND duals+homings at `inventory_capacity(rank)` AND radars >= cap-5 (2026-07-25; the fixed `fuel_full_threshold`/resume configs are deleted — the collect pickup ceiling and the hunt-entry bar now share the same rank-derived numbers, so they can never disagree and deadlock). The break/full gap gives hysteresis -- entry at the low break, exit only at full. |
| MUST | `self_state["fuel"]` is updated **only** from the wire's absolute-fuel messages (0x44 FuelGain, 0x2E TankStatusSync, 0x64 FuelDeposit). `pickup_container` is registry-only -- it does NOT add `transferred = prior_volume - remaining_volume` locally. The local-delta branch was a double-count on top of the wire's already-correct absolute fuel; removed 2026-06-23 after live observation of a 438-volume container producing a +438 ghost. See [[fuel-system#fuel-data-flow-single-source-of-truth]]. |
| MUST | A pickup is NEVER pre-filtered as wasted -- the server picks the slot you're most behind on at pickup time (see [[equipment-system]]). The bot dispatches `pickup_equipment` whenever any equipment container is in range; only the all-25 case fails with code 7. |
| MUST | Recognise server 0x52 `SUPERVISOR_ERROR_INVENTORY_FULL` (code 7) as an action-blocking error in the collect kind's whitelist in `_COMMAND_ERROR_APPLICABILITY` (`bot/tick_loop_actions.py`). The in-flight pickup clears immediately on the wire signal instead of stalling the full 10 s timeout, and the container's `failed_pickups` counter is bumped so the blacklist takes over. Closed 2026-06-21 with the empirical guard `tests/bot/test_tick_loop_coverage.py::test_command_error_clears_collect_on_inventory_full`. |
| MUST | **Action-kind-scoped 0x52 (2026-07-06):** 0x52 error codes are dispatched only when the current in-flight action's kind whitelists them (`_COMMAND_ERROR_APPLICABILITY` per-`ActionKind` map). Codes outside the whitelist are **orphans** left by a prior action that already resolved via a different signal (`container_consumed`, `teleport_landed`, ...); they are consumed and emitted as an `orphan_command_error` diagnostic instead of being spuriously attributed to the in-flight action. Radar (`scan`) and map_open both have empty whitelists — the server never rejects either dispatch — so every 0x52 landing during their waits is by definition orphan. The dedicated `_drain_orphan_command_error` helper handles those two kinds; `_clear_command_error` handles movement kinds and shares the same orphan-drop diagnostic. Live run 2026-07-06 20:20:59 smoking-gun: a late-arriving `code=4 "Empty container"` from a collect that had already completed via `container_consumed` was misattributed as the following `map_open`'s rejection, HUNT could not acquire, and the session exited `no_viable_targets` at fuel 531 with a fully-stocked tank. |
| MUST | **Typed collect 0x52 outcomes (2026-07-19):** a collect's 0x52 resolves in the ledger as what physically happened, not a blanket `command_rejected`: code 4 → `pickup_empty` (container drained; belief removed), code 5 → `clamped_transfer` (server transferred `min(volume, headroom)`, kept the remainder — a SUCCESS; the 5-min soak filed +2472 fuel across four of these as "rejections" before the split), code 7 → `inventory_full` (authoritative all-slots-full; beliefs reconciled). Codes 0/1 remain `command_rejected`. The run audit treats `clamped_transfer` as success (never a retry-loop signal); `pickup_empty`/`inventory_full` count as failures for retry-loop detection. |
| MUST | After firing a radar, mark exactly the tiles the radar revealed in `AIStateDict.local_scan_tiles`. Free radar = intersection of `(tank ± 2)` with viewport bounds; extra radar = every tile in the viewport. The bot picks its next forage action from this map (see `bot/ai/scan_coverage.py`, refactor 2026-06-21). |
| Verified by | `tests/bot/ai/test_mode_controller.py`, `tests/bot/ai/test_collect_mode_fuel.py`, `tests/bot/ai/test_collect_mode_equipment.py`, `tests/bot/ai/test_collect_mode_integration.py`, `tests/bot/ai/test_resource_search.py::TestMakeResourceSearchHop` (dot hop), `tests/world_state/test_mutations.py::TestPickupContainer`; per-kind 0x52 scoping in `tests/bot/test_tick_loop_coverage.py::TestClearCommandError::test_scan_wait_drops_orphan_error_and_stays_pending`, `test_map_open_wait_drops_orphan_error_and_stays_pending`, `test_teleport_wait_drops_orphan_empty_container`, `test_move_wait_drops_orphan_tank_full`, `test_orphan_command_error_emits_diagnostic`, `test_scan_and_map_open_whitelists_are_empty`. |

## 4. Action execution

### 4.1 Action lifecycle

| Aspect | Contract |
|---|---|
| MUST | Every action has a START (`emit_wire`), a WAITING phase (`emit_sync` repeated), and a COMPLETION (`emit_wire_complete` with `signal=`). |
| MUST | Completion signal is one of: `map_data_processed`, `teleport_landed`, `radar_scan_complete`, `position_reached`, `container_consumed_or_reached`, `stall_timeout`, `movement_rejected`. |
| MUST | If no authoritative signal arrives within `action_stall_timeout_ms`, emit `signal=stall_timeout` and replan to IDLE. |
| MUST NOT | Leave an action "in-flight" forever — `_clear_stalled_action` must fire. |
| Verified by | `tests/bot/test_completion_events.py` (existing); `smoke[5]` (zero stalls in first 10 s). |

### 4.2 Anti-loop protection

| Aspect | Contract |
|---|---|
| MUST | `map_open_cooldown_ms` (5 s default) prevents repeated map opens. |
| MUST | `kill_cooldown_ms` (30 s default) prevents re-targeting a recently killed tank. |
| MUST | `scan_cooldown_ms` (5 s default) prevents radar thrashing. |
| MUST NOT | Stall + replan + re-issue the same action within one cooldown window (the open-close-map loop pattern). |
| Verified by | `smoke[5]`; `tests/integration/test_stall_timeout_replans_to_idle.py` (Tier 2, handed off). |

## 5. Anti-patterns (must never re-emerge)

These are the historical bug-shapes the bot has suffered. Every fix in `combat_strategy.py`, `world_state_dispatch.py`, or `tick_loop_actions.py` should be checked against this list. The two fully-diagnosed instances are documented end-to-end in [[combat-chase-bug]] and [[executor-rejection-loops]].

| Anti-pattern | What it looks like | Prevented by |
|---|---|---|
| Open-close-map loop | `WIRE: map_open` → `SYNC: waiting for map open sync` (× N) → `stall_timeout` → repeat. | `mark_map_data_processed()` must be called by `_dispatch_map_data`. Test: `test_map_data_marks_action_complete` (2026-06-20). |
| Ghost firing | Firing at a tile where the tank already left, just because MapData still lists it. | `is_wire_present` gate in `_combat_shoot`. Test: `test_ghost_wire_presence_regression.py` (currently under review per #77). |
| Stationary-target reject | Practice-room bots fail the kill gate because they emit no per-tank wire after join. | OPEN — design decision pending (#75). |
| Stale-position fire | Wire-presence fresh but position not updated → fire at old tile. | `is_position_fresh` gate in `_combat_shoot`. |
| Same-tile re-engage | Bot shoots the same tile 12 times after misses. | `block_combat_target_and_replan` cooldown + stationary-target detection. |
| Action amnesia | Action emits no completion event because the consumer of the wire signal never wires it up. | Authoritative-completion contract (§4.1) + Tier 2 integration tests for every action lifecycle. |
| Radar spam in covered viewport | Bot fires the radar every 2 s in the same spot. Diagnosed 2026-06-21 (live capture 19:46:33+): the old server-side `scanned_viewports` gate did not close for a free 5x5 scan, so the bot re-fired forever at extras=0 after a failed pickup. | Tile-aware forager (`bot/ai/forage.py::plan_forage_search`) uses `AIStateDict.local_scan_tiles` and `is_viewport_fully_covered(...)` as the gate. Each radar dispatch marks exactly the revealed tiles via `mark_scan_dispatched`. Tests: `tests/bot/ai/test_forage.py::TestForageSearch`. |
| Fight-to-death under an adjacency override | Mode selector let "enemy one tile away → HUNT" outrank the low-fuel break; in a gang-up an enemy is ALWAYS adjacent, so retreat was structurally unreachable (practice-room soak 2026-07-25: bot traded from 384 fuel to 0, dead in 21 rounds; the override itself was patch #3 on a four-patch stack — see the 2026-07-25 log post-mortem). | Override deleted; hunt-only-when-full contract in `_select_owner_mode` (§3.1). Ignoring an adjacent bot while collecting is safe — bots never initiate ([[enemy-bot-behavior]]). Tests: `tests/bot/ai/test_strategy_coverage.py::TestHuntOnlyWhenFull`. |
| Radar to find enemies | HUNT acquire dispatches a `radar` command when no target is visible. Diagnosed 2026-06-21: radar reveals only hidden entities (fuel / equipment / mines); enemies arrive through the wire stream. | `search_for_enemies` in `bot/ai/hunt_mode.py` dispatches map_open only -- the radar branch was deleted 2026-06-21. Tests: `tests/bot/ai/test_hunt_mode.py::test_hunt_search_dispatches_map_open_not_radar_during_acquire`, `tests/bot/ai/test_enemy_search.py::TestDecideMapOpen::test_fallback_opens_map_even_when_recently_opened`. |
| Edge-walk fuel burn during HUNT | Bot walked or teleported to viewport-edge tiles every tick the map was on cooldown. Diagnosed live 2026-06-22 (60 s run): 14 of 30 ticks were `edge_for_enemies`, 10 of those were terrain-blocked teleports at ~131 fuel each. Two reasons it was waste: (a) viewport shifting is OFF in this game configuration so a walk to the edge reveals no new ground, and (b) the teleport fallback aimed at a random edge tile rather than a known enemy. Walks also cost fuel (per-tile) so even the "free" edge walks burned the reserve. | `search_for_enemies` dispatches map_open unconditionally -- the cooldown-gated edge-walk branch was deleted 2026-06-22. Tests: `tests/bot/ai/test_hunt_mode.py::test_hunt_search_dispatches_map_open_not_radar_during_acquire`, `tests/bot/ai/test_enemy_search.py::TestDecideMapOpen::test_fallback_opens_map_even_when_recently_opened`. |

## 6. What is NOT in this contract (yet)

These behaviors exist in the code but lack a verified contract entry (one exception noted inline). Adding them is Tier 2 integration test work.[^2]

- Inventory restock thresholds and timing
- Patrol waypoint cycling
- Combat target switching when a higher-value enemy enters range
- Bridge-build vs obstacle-drop decision (carrying state) — correction 2026-07-23: NOT in code at all; the bot has no block state tracking or planner awareness ([[movable-blocks]] open work)
- Teleport-target selection (heuristic)
- Self-rank promotion handling (0x2B reception)

When you add tests for these, also add a contract row here. The contract grows with the test suite, under the coverage and no-mocks discipline in [[coding-standards]].

## 7. How to use this page

**Before proposing a bot-behavior fix:** read the relevant section. If the change affects a MUST/MUST NOT, list which one and what test will be updated. If the change affects an anti-pattern's prevention mechanism, confirm the test still passes.[^1]

**When `make smoke` fails:** the failing assertion maps to a section here (`smoke[N]` references). Open that section to know what the bot was supposed to be doing and which other behaviors might be entangled.[^3]

**When `make check` integration tests fail:** the test name maps to a row here. The contract row tells you the broader behavior that test guards.[^3]

**When you add a new behavior:** add a row here first (MUST / MUST NOT / Verified by), then write the test, then implement. Contract-first prevents drift ([[coding-standards]]).

## Open items tracked elsewhere

None at end of 2026-06-20. The last decoder gap (#72 13-byte 0x43) was a 3-record multi-pickup CacheUpdate; see [[decode-coverage]] for the corrected wire format and the ContainerPickup multi-record dispatch.

[^1]: project instruction file `CLAUDE.md` (repo root, on disk): "The wiki is the single source of truth for game mechanics, wire protocol, combat strategy, and architecture decisions" — this page is the bot-obligations slice of that policy.
[^2]: presence checked 2026-07-23 against the blob-pinned `src/tankpit_bot/bot` tree (frontmatter): patrol/waypoint types in `bot/ai/types.py`/`types_codecs.py`, 0x2B rank handling in `sniffer/world_state_dispatch.py` and `state/mutations.py`; the block-decision item was the exception and is corrected inline.
[^3]: standing instruments on disk: `Makefile` targets `check:` (line 88) and `smoke:` (line 102), verified 2026-07-23; the `smoke[N]` indices map to assertions in the smoke script the target runs.
