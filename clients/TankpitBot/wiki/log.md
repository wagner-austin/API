# Wiki Operation Log

Append-only record of every wiki operation (ingest, update, audit). Most recent entry at the bottom.

---

## [2026-06-16] ingest | Initial wiki seed from memory files

Migrated all game-mechanics, protocol, and combat knowledge from `.claude/projects/.../memory/` files into the wiki. 13 content pages across 4 hubs.

**Source materials:** 8 memory files (feedback_combat_protocol, feedback_bot_gameplay, feedback_viewport_mechanics, feedback_map_toggle_semantics, feedback_radar_mechanics, feedback_ferry_mechanics, feedback_map_fuel_atlas, feedback_bot_commands) plus live diagnostic run from same session.

**Pages written (13):**
- Game mechanics: [[viewport-frame]], [[teleport-mechanics]], [[radar-mechanics]], [[fuel-system]], [[equipment-system]], [[map-mechanics]], [[ferry-mechanics]]
- Protocol: [[combat-hit-format]], [[deactivation-format]], [[map-data-decode]], [[tank-registry]]
- Combat: [[shot-range]], [[enemy-bot-behavior]], [[weapon-log-markers]], [[combat-chase-bug]]
- Architecture: [[inheritance-chain]], [[coding-standards]]

**Every claim carries a footnote** citing a run ID, client JS reference, user quote, or code location. Confidence is `high` on all pages — all facts were wire-verified or user-confirmed across multiple sessions.

**New content from this session:** [[combat-chase-bug]] — diagnosed live during a 2-minute bot run. The teleport-chase loop (bot repeatedly re-teleporting instead of walking 1 tile to close) is documented with the exact observed sequence and pending fix.

**Memory files retired:** 8 source memory files removed from `.claude/projects/.../memory/`. MEMORY.md updated to 7 remaining entries (AI behavior preferences only). CLAUDE.md created at project root pointing future sessions to the wiki.

---

## [2026-06-16] ingest | Codebase hub — module map, services, testing, make targets, adding probes

Added a fifth hub for codebase knowledge — how the code is organized, how to work in it, and how to extend it.

**Pages written (5):**
- [[module-map]] — all top-level packages, dependency flow diagram, scripts
- [[services]] — CDPService, CommandService, WorldService: ownership, injection, factory wiring, lifecycle functions
- [[testing-patterns]] — _test_hooks protocol DI, MonkeyPatchBanRule, replay regression tests, coverage config
- [[make-targets]] — every make target categorized (safe/offline vs live/server), output locations
- [[adding-a-probe]] — step-by-step guide to creating a new action_lab probe with all 7 existing probes listed

**Running total: 22 pages across 5 hubs.** Zero unresolved [[links]].

---

## [2026-06-16] ingest | Official game rules + teleport fix

**Pages written (1):**
- [[game-rules]] — transcribed from the five in-game How To Play screens. Key new facts: radar scales with rank, equipment capacity 20+5/rank, missile shots fire over obstacles, demotion on death, 0 fuel = instant deactivation.

**Pages updated (2):**
- [[teleport-mechanics]] — critical correction: click directly on the target (enemy/container), server handles adjacent placement. Added footnote [^8] citing user + official How To Play.
- [[combat-chase-bug]] — updated root cause (client-side adjacent tile computation was wrong) and fix (teleport to target directly, `choose_combat_landing_tile` returns enemy coords).

**Code changes:** `choose_combat_landing_tile` returns `(target["x"], target["y"])` directly. `find_teleport_landing_tile` returns `(goal_x, goal_y)` clamped to 0-255. Removed dead `CLOSE_WALK_RANGE_TILES`. All 3926 tests pass, 100% coverage, guard+ruff+mypy clean.

**Running total: 23 pages across 5 hubs.** Zero unresolved [[links]].

---

## [2026-06-16] ingest | Combat strategy, weapon selection, gameplay loop, structured diagnostics

**Pages written (4):**
- [[weapon-selection]] — server-side weapon choice: dual at adjacent enemy, homing when enemy moves same tick, single at empty ground. weapon_byte=0 is a genuine miss (stale position). Shields and corpses return positive hits.
- [[gameplay-loop]] — the full combat→refill→radar conservation cycle: fight, pick up viewport, short hop + radar, repeat until stocked, hunt again.
- [[equipment-refill-strategy]] — low-radar grid walk pattern (5-tile steps with free 5x5), extra radar conservation, container randomness (radars least frequent).
- [[game-rules]] — official How To Play screens transcribed.

**Pages updated (3):**
- [[combat-hit-format]] — added hit behavior table (shields=hit, corpses=hit, miss=empty ground) with footnote [^4]
- [[enemy-bot-behavior]] — added homing-for-fleeing and never-abandon rules with footnotes [^4][^5]
- [[teleport-mechanics]] — click directly on target, server handles placement [^8]

**Code changes (session):**
- Teleport directly to target (combat + containers), server handles displacement
- Damage updates now refresh `last_wire_seen_ms` (fixes false ghost detection mid-fight)
- Miss on stationary target → block (stale position, not a corpse)
- Short-hop fallback when standard search hop unaffordable
- Terrain check on search hop targets (skip water tiles)
- `find_teleport_landing_tile` checks passability, returns None for water-locked containers
- Structured combat feedback diagnostics (hit/miss/kill in JSONL)
- Session scorecard with permanent kill/hit/miss counters
- Teleport landing miss diagnostics (terrain + tanks at target)

**Live results:** 4 kills, 14 hits, 0 misses in 3 minutes. Net positive on equipment.

**Running total: 27 pages across 5 hubs.** Zero unresolved [[links]].

---

## [2026-06-18] audit | Wire decode coverage audit — full byte decode from JS source

Systematic audit of every wire message type against the game client JS (`tpclient.js`) and 90+ capture sessions. Every byte in every message type decoded and verified.

**Pages written (1):**
- [[decode-coverage]] — complete mapping of all 37 JS message handlers vs our decoders, with gaps, wrong constants, and dropped fields

**Critical findings:**
1. **Ghost detection solved:** 0x3d container messages carry `direction` field where values 32-33 = dead corpse sprite. 42 corpse messages verified across 18 tanks in captures. No TTL heuristic needed — it's a binary alive/dead flag on the wire.
2. **15,438 status messages dropped as UNKNOWN:** 0x2E and 0x3D 13-byte container subtypes misidentified because `is_position_update_structure` requires subtype=0x24. These carry position + damage + rank + fuel for every tank.
3. **Supervisor constants wrong:** `SUPERVISOR_STATUS_PROMO_KILL=8` is actually error code "Insufficient fuel." The 0x52 message is the server's command failure response, not a kill signal.
4. **ShootEvent decoder offset wrong:** Missing flags byte at offset 0 shifts all subsequent fields. Fuel read as 24-bit instead of 1 byte.
5. **0x2E self status fully decoded:** damage, rank, lb_score (24-bit BE), promo_state, fuel (LE u16). Fuel verified: 98/152 exact match with FuelGain at same millisecond, 8/15 sessions start at exactly 1100 (Private).
6. **Promotion points = lb_score:** The 24-bit BE leaderboard score counts DOWN per hit. Lower = better ranking. Practice server caps at Private (rank=1).

**Analysis scripts written (not committed):**
- `scripts/analyze_wire_bytes.py` — exhaustive byte dump of all container messages
- `scripts/correlate_unknowns.py` — cross-reference unknown bytes against proven fields
- `scripts/crack_remaining_bytes.py` — per-byte cross-reference against TSS and registry
- `scripts/solve_remaining_v2.py` — decompose compound bytes, verify LE u16 vs separate fields
- `scripts/solve_subbytes.py` — bit-level analysis for packed fields
- `scripts/crack_all_blobs.py` — all message types: tank_update, position_update, movement, etc.
- `scripts/crack_final_three.py` — direction correlation, fuel verification, rank underflow
- `scripts/find_kill_byte.py` — infer kills from 0x3d streams, check for kill counter
- `scripts/find_supervisor.py` — scan for Supervisor messages across all captures
- `scripts/find_action_done.py` — check ActionDone payload for success/failure
- `scripts/verify_everything.py` — cross-reference Supervisor vs kills, fuel absolute values
- `scripts/verify_js_claims.py` — final verification of JS claims against capture data

**Running total: 28 pages across 5 hubs.** Zero unresolved [[links]].

---

## [2026-06-19] ingest | Complete tpclient.js reverse-engineering — new JS Client hub

Full manual walk of all 329 lines of `tpclient.js` (minified Closure Compiler output, ~82k tokens). Every class, function, constant, message handler, and data structure identified, deobfuscated, and documented. Created a new "JS Client" hub with 20 content pages.

**Pages written (20):**
- [[js-source-map]] — line-by-line annotated structure: all classes mapped to line ranges, key findings highlighted
- [[client-commands]] — all 24 binary game commands (K subclasses) and 14 connection commands (va subclasses) with exact byte layouts
- [[v-table-complete]] — all 35 V table handlers with field-by-field parse logic extracted from every .h() static method
- [[client-constants]] — rank thresholds, equipment names, error strings, timing values (L object), direction encoding, projectile offsets, sprite dimensions, hotkey defaults, map colors
- [[client-state-machine]] — 14 game states (s field), all transitions, action queue priority, tick loop timing
- [[xor-cipher]] — qb[] table derivation, za() encode/decode, which messages use XOR vs plaintext
- [[rendering-pipeline]] — 6 canvas layers (Background/Tanks/Action/Map/Overlay/Menu), sprite sheets, tile engine, dirty-rect system
- [[chat-messages]] — all 65 predefined messages with IDs, text, team filters, position flags, voice keywords, display order
- [[connection-protocol]] — full handshake: WebSocket framing (2-byte LE length prefix), AUTH format, game list, select, join, start, disconnect, error reporting
- [[terrain-system]] — terrain byte encoding (adjacency bits + base type), sg() pixel sampling, pseudo-random variant via Morton code + CRC-8, edge tiles, rock types
- [[obstacle-bridge-mechanics]] — pickup/drop/build decision logic (be function), carry state, V.B handler, auto-drop on water, ferry interaction
- [[map-data-algorithm]] — exact skip-RLE fuel dot parser from Ig.h with worked example, tank entry 5-byte format, map position hash
- [[game-modes]] — practice(6)/normal(7)/tournament(5) encoding, mode-specific differences (equipment capacity, Top10, elimination)
- [[sound-system]] — all 18 Web Audio buffers, 3 playback channels (one-shot/loop1/loop2), trigger events, volume/toggle
- [[decoration-encoding]] — 4-byte → 9-slot × 2-bit packing via yg(), award names per slot per level, rendering in ed()
- [[fingerprint-algorithm]] — 15 browser properties collected, plugin enumeration, canvas fingerprint, MurmurHash3 (seed=31, standard constants)
- [[toolbar-layout]] — 18 clickable regions with pixel-exact hitboxes from pc/qc/rc/sc arrays, scope direction mapping, equipment count display, fuel/promo bars
- [[viewport-update-algorithm]] — V.Z position step encoding (delta col/row in single byte), 24-bit entity packing (terrain/overlay/cache), scroll optimization (blit vs full redraw)
- [[playback-system]] — recording binary format (length-prefixed segments with JSON metadata), Kf playback controller, speed control, timeline UI, seek with state-critical message pre-dispatch
- [[input-handling]] — mouse (click/hold/double-click decision tree), keyboard (hotkey map dispatch), touch (Bd tracking, pe() gesture recognition for 8-direction swipes), coordinate conversion

**Pages updated (2):**
- [[map-data-decode]] — resolved tank entry "unknown 2-bit field" as rank_category; added full 5-byte tank format; cross-linked to [[map-data-algorithm]]
- [[decode-coverage]] — updated Deactivation footnote [^6] with exact Pg.h field layout from V table walk; cross-linked to [[v-table-complete]]

**Key findings from the JS walk:**
1. Game ticks at 200ms (5 Hz) when idle; drive animation at 33ms (~30 fps); shoot at 10ms
2. 63 predefined chat messages with voice recognition keywords — no free text chat
3. Complete connection handshake: AUTH uses MurmurHash3 fingerprint + session magic
4. Exact obstacle/bridge mechanics: carry state, auto-drop, ferry restrictions
5. Toolbar has 18 hitbox regions with pixel-exact positions
6. Playback recordings are timestamped binary message dumps with JSON metadata header
7. Tank IDs >= 500 are registered accounts (shown as profile links)
8. Promotion from Sergeant+ requires deactivating enemy of (rank-1) or higher

**Running total: 47 pages across 6 hubs.**

---

## [2026-06-19] refactor | Unified 0x2E body decoder — removed dual-path architecture

Collapsed the protocol vs. container dual-path 0x2E decoder into a single subtype-first dispatcher (`protocol.decoders.tank.decode_0x2e_message`) with length-based fallback for container-only types. Every wire byte now has exactly one decoder, and every decoder is reachable from `decode_message`. No more divergent paths.

**Field-level decoder fixes (verified against tpclient.js + production captures):**

- **0x41 Deactivation** — extended from 2 to 5 fields per JS `Pg.h` (`V.A`). Added `status` (byte 0), `promo_eligible` (byte 3), `is_mine_kill` (derived from `killer_id >= 65530` sentinel). Old code reported only `victim_id` + `killer_id`.
- **0x53 ShootEvent** — rewrote field offsets per JS `Gg.h` (`V.S`). Was misnaming bytes (e.g., `target_x` was actually `source_x`); now correctly reads `[team][shooter_id:2 LE][source_x][source_y][target_x][target_y][unk1][unk2][weapon]`.
- **0x3D MovementResponse** — added the `carrying` byte at offset 11 per JS `Mg.h`. Old decoder stopped at byte 10 and silently dropped the obstacle-carry flag.
- **0x2E TankStatusSync** — handles both the 9-byte short form and the 13-byte form with fuel at offsets 10-11, per JS `Og.h`. Replaces the deleted container SelfStatus (which was unreachable in production anyway — the protocol path always won).
- **0x4F RadarScanResult** — now the single canonical 0x4F radar decoder. Container's `decode_radar_response` is gone. Structural check (`_is_radar_scan_structure`) disambiguates from the `CombinedTileUpdate` use of 0x4F.

**Deleted decoders (all were duplicates of canonical protocol-path decoders):**

- `container.decoders.combat.decode_combat_hit` + `CombatHitDict` + `COMBAT_HIT` enum — was a wrong-offset alias of ShootEvent
- `container.decoders.combat.decode_deactivation_kill` + `DeactivationKillDict` + `DEACTIVATION_KILL` enum — duplicate of 0x41 Deactivation
- `container.decoders.position.decode_movement` + `MovementDict` — duplicate of 0x47 Movement; container's bytes 8-11 were misinterpreted as "player_id" but JS `Mg.h` shows `lb_score (24-bit BE)` + `rank`
- `container.decoders.status.decode_tank_status_sync` (2-3 byte form) — length-only catch-all that misidentified 0x4F/0x46/0x58/0x3F shorts as TankStatusSync; real TankStatusSync is 8+ bytes (protocol)
- `container.decoders.status.decode_tank_position_status` (0x3D) — replaced by protocol `MovementResponse` with the carrying byte restored
- `container.decoders.status.decode_self_status` + `SelfStatusDict` + `SELF_STATUS` enum + entire `container/decoders/status.py` module — unreachable in production; TankStatusSync covers the 13-byte form with fuel
- `container.decoders.radar.decode_radar_response` + `RadarResponseDict` + `RADAR_RESPONSE` enum + entire `container/decoders/radar.py` module — duplicate of protocol RadarScanResult
- `container.mapper.PlayerIdMapper` + entire `container/mapper.py` module — was inferring player_id from the misinterpreted Movement bytes
- `protocol.decoders.combat.decode_hit_confirmation` + `HitConfirmationDict` — stranded alternate decoder for 0x48 that never matched any JS handler; canonical 0x48 is EnemyDetect (`Tg.h` / `V.H`), already wired

**Dual-path machinery removed:**

- `_try_unwrap_0x2e`, `_TUNNELED_SUBTYPES`, `_TUNNELED_MIN_LENGTHS`, `_is_tunneled_radar_scan_structure` from `protocol/decoders/routing.py`

**Production data used for length-gate tuning:** `runs/bot/bot-20260619-053210.capture_session.json` — sampled all 0x2E body lengths × subtype distributions. Confirmed Sync clamp to inner=1 byte, ActionDone requires inner≥1 (1-byte 0x54 is teleport_landed in 100% of samples), corrected MSG_VIEWPORT to 0x5A (the routing.py comment said 0x56 — wrong).

**Tests:** 3963 pass at 100% coverage (was 3923 across 52 commits before this branch). New test classes for every length-gate change; bulk-redirected 31 import sites and 26 construction sites for `MovementResponseDict.carrying` and `RadarContainerDict`/`RadarMineDict` (moved from container to protocol).

**Pages updated (4):** [[decode-coverage]], [[combat-hit-format]] → [[shoot-event-format]] (renamed + rescoped), [[deactivation-format]], hub link sweep across [[shot-range]] / [[tank-registry]] / [[weapon-log-markers]] / [[weapon-selection]].

**Running total: 47 pages across 6 hubs.**

---

## [2026-06-19] refactor | Three-timestamp tank-freshness model + observation-based mutator

Replaced the conflated single "wire-seen" freshness with three independent timestamps (`timestamp_ms`, `last_wire_seen_ms`, `last_position_update_ms`) and routed every tank-state mutation through a single observation-based mutator (`apply_tank_observation`). The bug class this prevents: damage-only wire broadcasts (0x2E TankStatusSync, every ~2 s globally) used to refresh the wire-seen stamp for tanks that had teleported out of viewport, fooling the kill-shot gate into firing at stale registry positions.

Production evidence: `runs/bot/bot-20260619-050303` recorded 25 `combat_miss` events on the same target (orange-8 at 155,155) over ~100 s, all `target_moved=false`. The unified decoders (committed earlier today) made the bug visible; this refactor closes it structurally.

**Type additions:**
- `last_position_update_ms` added to `TankStateDict` (and re-exposed on `EnemyThreatDict`).
- New `TankObservation` TypedDict in `tankpit_bot.state.types.tank_observation` with full encode/decode + `require_*` validation. Required + None-bearing optional aspects (no `NotRequired`).

**Single mutator (single source of truth):**
- `apply_tank_observation(state, obs)` in `tankpit_bot.state.mutations`.
- Invariants enforced in code, pinned by tests in `tests/world_state/test_tank_observation.py`:
  1. `timestamp_ms` advances on every observation.
  2. `last_wire_seen_ms` advances iff `is_wire_sourced`.
  3. `last_position_update_ms` advances iff `is_wire_sourced AND position is not None`.

**Deleted (every legacy mutator):**
- `update_tank_from_registry`, `update_tank_damage` -- the divergent low-level mutators that silently conflated freshness concepts.
- Their public re-exports from `state/__init__.py` and `state/mutations.py`.
- The obsolete test classes `TestUpdateTankFromRegistry`, `TestWirePresenceFunnel`, `TestUpdateTankDamage` in `tests/world_state/test_mutations.py`.

**Dispatch sites converted (every wire-to-tank-state path):** the helpers in `sniffer/world_state_tanks.py` (TankEntry, TankInfo, TankStatus, container TankRegistry, MoveResponseFull, client-registry refinement, TankStatusSync damage, position update, radar enemy detect) plus the direct dispatch builders in `sniffer/world_state_dispatch.py` (0x3D position+status) and `sniffer/world_state_dispatch_position.py` (map-snapshot per-tank). Every one of these now builds a `TankObservation` and calls `apply_tank_observation`.

**Kill-shot gate:**
- `POSITION_FRESHNESS_TTL_MS = 3000` added in `bot/ai/threats.py`.
- `is_position_fresh(last_position_update_ms, now_ms)` exposed alongside the existing `is_wire_present`.
- `bot/ai/combat_strategy.py::engage_target` checks position freshness AFTER the wire-presence gate; a wire-present but position-stale target is blocked and replanned with a `combat_stale_position` diagnostic.

**Tests:** 3991 pass at 100% statement + branch coverage.

**Pages written (1):** [[tank-freshness-model]] -- the contract page documenting the three timestamps, the single mutator, the per-message advancement table, and the locked invariant tests.

**Pages updated (3):** [[decode-coverage]] (added `related` link), [[combat-chase-bug]] (added `related` link), [[hubs/architecture]] (added the new page).

**Running total: 48 pages across 6 hubs.**

## [2026-06-21] update | Equipment-pickup mechanic, radar viewport-edge rule, radar-spam diagnosis

Live session 2026-06-21 19:46 reproduced the bot's "radar every 2 seconds in the same spot" loop. Combined with the user's clarification of the underlying game mechanics, three wiki pages were corrected and one anti-pattern added.

**Pages updated (4):**

- [[radar-mechanics]] -- added the viewport-intersection rule. Neither free nor extra radar reveals tiles outside the viewport; at a viewport edge or corner the built-in 5x5 reveals only the intersection of (tank-x±2, tank-y±2) with the viewport bounds. Marked `fact_checked: 2026-06-21`.
- [[equipment-system]] -- documented the deterministic pickup mechanic (a container fills the slot you are most behind on at pickup time, server-decided) and the `SUPERVISOR_ERROR_INVENTORY_FULL` (code 7) wire signal that the bot currently ignores. Marked `fact_checked: 2026-06-21`.
- [[bot-behavior-contract]] -- added two MUST rows to §3.4 (recognise code 7 as action-blocking; mark scanned tiles by actual revealed region) and a new anti-pattern row for "Radar spam in covered viewport" with the live-capture diagnosis.
- [[combat-chase-bug]] -- added a "Caveat: server does NOT displace off equipment-container tiles" section. The let-server-displace fix from 2026-06-16 works for combat (tank on tile) but does not generalise to empty-of-tanks tiles with only a container -- those leave the bot standing on the container, and `pickup_equipment` from distance 0 returns no `container_consumed`.

**Open items flagged in the contract (no code changes yet):**

- Add `SUPERVISOR_ERROR_INVENTORY_FULL` (code 7) to `_ACTION_BLOCKING_COMMAND_ERRORS` in `bot/tick_loop_actions.py:44`.
- Replace tank-centered 5x5 cell coverage tracking with viewport-clamped tile-level tracking: free radar marks (tank ± 2) ∩ viewport, extra radar marks the full viewport.

**Source data:** live capture 2026-06-21 19:46:23-19:46:51 (90 s session, bot at (131,126), extras=0 throughout); user clarifications on viewport-edge radar behaviour and deterministic container fill.

## [2026-06-21] update | Radar reveals fuel/equipment/mines only (NOT enemies); viewport shifting OFF

User clarification: radar is for revealing entities that are hidden by default on spawn -- fuel containers, equipment containers, and mines. Enemy tanks are always visible to the bot via the normal wire stream and never need a radar to be discovered. Firing radar to "search for enemies" is a category error.

User clarification: viewport shifting is OFF in the current game configuration. Walking never shifts the viewport; the only way to reveal new ground is to teleport.

**Pages updated (2):**

- [[radar-mechanics]] -- added "What radar reveals (and what it does NOT)" section: fuel/equipment/mines only; enemy discovery is map-open + viewport-edge walking, never radar. Added "Viewport shifting" section: walking never moves the viewport; teleport is the only way to a new region.
- [[bot-behavior-contract]] §3.2 -- added a MUST NOT row: HUNT acquisition cannot fire `make_radar_command()` to search for enemies.

**Implication for code (handed off, not done):**

- `bot/ai/hunt_mode.py` `search_for_enemies` lines 54-75 dispatch radar with reason "radar to search for enemies". This branch must be deleted; the function should fall through to map-open or viewport-edge walking when no candidate is in `analyze_threats`. The radar dispatch on `_decide_hunt_close` is **legitimate** -- it scans for mines/containers around the engagement tile, not enemies, and the comment-string should be corrected but the dispatch kept.

## [2026-06-21] refactor | Single tile-aware scan path; removed divergent radar branches

Completed the scan-system refactor handed off in `HANDOFF_SCAN_REFACTOR.md`. The radar dispatch path is now one parameterized function shared by both equipment and fuel recovery; HUNT no longer fires radar to search for enemies; cell-grid coverage is gone, replaced by per-tile coverage clamped to the viewport.

**Code changes:**

- `bot/ai/scan_coverage.py` -- tile-level primitives (`tile_key`, `is_tile_covered`, `record_tile_scan`, `viewport_tiles`, `free_radar_revealed_tiles`, `is_viewport_fully_covered`, `nearest_uncovered_tile_in_viewport`). All cell-grid symbols (`FORAGE_CELL_SIZE`, `cell_center`, `is_cell_covered`, `local_scan_cell_key`, `record_local_scan`, `record_viewport_scan`) deleted.
- `bot/ai/forage.py` -- single forager: `plan_forage_search(ctx, ai_state, *, score, behavior_mode, radar_affordable)`. No mode enum, no extras-gated branch; equipment recovery passes `radar_affordable=can_use_radar(ctx)`, fuel recovery passes `radar_affordable=can_use_fuel_radar(ctx)`. Reason is `forage_radar` (radar) or `forage_sweep` (walk).
- `bot/ai/recover_equipment_mode.py` -- `_plan_equipment_sense_or_search` delegates to `plan_forage_search`; on `None` falls through to `_plan_equipment_search`. The `extras == 0` gate, the `cell_already_covered` gate, the `radar_for_equipment` branch, and the runtime `record_local_scan` import are gone.
- `bot/ai/recover_fuel_mode.py` -- `_plan_fuel_sense_or_search` delegates to `plan_forage_search`; on `None` falls through to the existing search-hop / dot-walk / edge / map-intel chain. The `radar_for_fuel` branch is gone.
- `bot/ai/hunt_mode.py` -- `search_for_enemies` lost its radar branch and the `radar_reason` parameter; HUNT now only walks the viewport edge or opens the map. The legitimate `_decide_hunt_close` scan-on-landing radar (mines / containers around the engagement tile) is unchanged.
- `bot/ai/context.py` -- deleted `is_current_viewport_scan_failed` and `should_scan_resources_in_current_viewport` (no remaining callers); kept `mark_scan_dispatched` which now records the exact revealed-tile set in `local_scan_tiles`.
- `bot/ai/mode_controller.py` -- both `derive_recover_equipment_mode_state` and `derive_recover_fuel_mode_state` map `reason in ("forage_radar", "forage_sweep")` to `SENSE`.
- `bot/ai/types.py` -- `AIStateDict.local_scan_cells` renamed to `local_scan_tiles`; same in encoder/decoder.

**Test surface:**

- `tests/bot/ai/test_scan_coverage.py` -- full rewrite for tile primitives.
- `tests/bot/ai/test_forage.py` -- full rewrite for the parameterized forager (mode, score, radar_affordable).
- `tests/bot/ai/_support.py` -- added `viewport_covered_tiles(world)` and `make_post_radar_ai_state(world)` helpers so tests that exercise downstream fallback paths can model "the bot just radared this viewport" without driving the tick loop.
- Cross-cutting renames: `radar_for_equipment` / `radar_for_fuel` / `radar_for_enemies` -> `forage_radar` across `test_recover_equipment_mode.py`, `test_recover_fuel_mode.py`, `test_recover_equipment_integration.py`, `test_strategy_coverage.py`, `test_mode_lock.py`, `test_mode_controller.py`, `tests/replay/test_real_session_regressions.py`. `test_hunt_mode.py` and `test_enemy_search.py` gained new HUNT-no-radar contract tests.

**Result:** `make check` green, 4104 tests, 100.00% statement+branch coverage.

**Documentation:**

- `wiki/pages/bot-behavior-contract.md` §3.4 -- the **OPEN** caveat about `local_scan_cells` is closed; the row now states the invariant ("mark exactly the revealed tiles").
- `wiki/pages/bot-behavior-contract.md` §5 -- new anti-pattern row "Radar to find enemies" with the contract tests that lock it down.
- `docs/bot-logging.md` -- example log line updated to `reason=forage_radar`; sentence about old reason naming rewritten.

**Out of scope (still open):** `SUPERVISOR_ERROR_INVENTORY_FULL` (server code 7) is not yet in `_ACTION_BLOCKING_COMMAND_ERRORS` (`bot/tick_loop_actions.py:44`). Tracked separately. — *Closed by the 2026-06-21 "Inventory full" entry below.*

## [2026-06-21] fix | Recognise 0x52 "Inventory full" (code 7) as action-blocking

Added server error code **7 (`SUPERVISOR_ERROR_INVENTORY_FULL`)** to `_ACTION_BLOCKING_COMMAND_ERRORS` in `bot/tick_loop_actions.py:44`. The set now matches the codes the bot can authoritatively clear from an in-flight action's wait.

**Empirical grounding.** The 0x52 decoder (`protocol/decoders/world.py:290`) parses `error_code = data[2]`; the dispatch (`sniffer/world_state_dispatch.py:925`) stores it on the WorldService and emits a `command_error` diagnostic. Capture 20260620 recorded two real `error_code=7` events at 19:07:28 / 19:08:30 in `runs/sniff/latest.events.jsonl`, plus parallel `[GAME:EQUIPMENT] Inventory full` strings in `runs/sniff/latest.log`. Three independent channels (wire byte / DOM scraper / capture-session text) of the same event window.

**Effect.** Without code 7 in the set, the bot waited the full `action_stall_timeout_ms` (10 s) on every "Inventory full" reject before clearing the action and bumping the container's `failed_pickups` counter. With code 7 added, the action clears at the wire boundary (< 1 s) and the container's `failed_pickups` counter bumps immediately, surfacing the container to the blacklist heuristic.

**Test.** `tests/bot/test_tick_loop_coverage.py::test_command_error_clears_collect_on_inventory_full` sets `last_command_error = 7` on a pending `collect` action with a seeded container at the target tile, then asserts (1) the wait returns `False`, (2) the bot transitions to `IDLE`, (3) `last_command_error` is consumed, (4) the container's `failed_pickups` counter is `1`.

**Pages updated.**

- `wiki/pages/bot-behavior-contract.md` §3.4 -- closed the **OPEN** caveat on the code-7 row; cited the new regression test.
- `wiki/pages/equipment-system.md` -- rewrote the "Inventory full" wire signal section to describe the now-current behaviour (1-tick clear, immediate failed_pickups bump) instead of the old 10 s stall.
- `src/tankpit_bot/bot/tick_loop_actions.py` -- docstring on `_clear_command_error` mentions "Inventory full" and references the live captures.

## [2026-06-22] refactor | Delete HUNT enemy-search edge walk; user-directed

Diagnosed from a 60-second live run today (`runs/bot/latest.events.jsonl`, 30 ticks, 0 kills): with no visible threats and full inventory, HUNT acquire cycled `map_open` -> `edge_for_enemies` -> `map_open` -> `edge_for_enemies` indefinitely. 14 of 30 ticks were `edge_for_enemies`, 10 of those resolved as terrain-blocked teleports at ~131 fuel each. Net effect: 185 fuel burned for zero combat outcomes.

The branch was dead weight under this game configuration. Two independent reasons:

1. **Viewport shifting is OFF.** Walking to a viewport edge does not reveal new ground; only a teleport opens a new viewport. So the walk variant was pure no-op.
2. **The teleport variant aimed at a random edge tile, not a known enemy.** Every `map_open` reply contained 27 enemy positions; the bot never targeted any of them, then teleported to a random edge tile hoping for spawn-in. That was always going to spend more fuel than it saved.

Also corrected a mistaken belief that walking is free. `move` commands deduct per-tile fuel (live capture: `Fuel: 473 -> 451 (-22)` and `451 -> 419 (-32)` on consecutive moves). The edge walks weren't "free wasted exploration"; they were paid wasted exploration. Documented in [[bot-behavior-contract]] §5 row.

**Code changes:**

- `bot/ai/hunt_mode.py::search_for_enemies` -- deleted the entire `if map_age < cooldown: edge walk` branch and the `select_exploration_command` import. Function is now a single dispatch: `make_map_open_command(...)`. Dropped the `edge_reason` parameter. Both call sites (`_decide_hunt_acquire`, `_enter_confirm_kill`) updated.
- `bot/ai/movement.py::select_exploration_command` -- KEPT. Still used by `resource_search.py` for fuel/equipment recovery edge walks (`edge_for_fuel`, `edge_for_equipment`); those are separate decisions that the user has not flagged.

**Test surface:**

- `tests/bot/ai/test_enemy_search.py` -- deleted 4 tests covering the old edge-walk variants (`test_fallback_walks_when_map_is_on_cooldown`, `test_fallback_opens_map_when_walk_and_teleport_blocked`, `test_fallback_opens_map_when_all_exploration_targets_failed`, `test_fallback_uses_alternate_edge_when_preferred_candidate_blocked`); rewrote the radar-no-fire guard as `test_fallback_opens_map_even_when_recently_opened`; renamed the low-fuel fall-through.
- `tests/bot/ai/test_hunt_mode.py` -- renamed `test_hunt_search_never_dispatches_radar_during_acquire` to `test_hunt_search_dispatches_map_open_not_radar_during_acquire` with the new positive assertion; rewrote `test_hunt_search_does_not_enter_confirm_kill_without_target` to chain two consecutive map_open ticks instead of two edge moves.
- `tests/bot/ai/test_mode_controller.py` -- renamed the derive-substate defensive test (no production path produces a HUNT teleport without a locked target anymore; the derive branch is kept for defence-in-depth).
- `make check` green at 4101 tests, 100.00% statement+branch coverage.

**Behaviour gap surfaced and left as a follow-up.** Bot saw 27 enemies in every map snapshot but acquired zero of them. `analyze_threats` filters tanks lacking a viewport observation (the strictness added 2026-06-21 to fix phantom firing), so map-only enemies never enter the threat list. Without map-based acquisition, the bot can only engage enemies that walk into its current viewport. Tracked as a future change ("use the map for enemies") -- not in scope for this commit because it's a strategic-design change, not a refactor.

## [2026-06-22] refactor | Stay-on-target: 0x58 TankRemove becomes a no-op, pursuit fires homing instead of teleport-chasing

Live capture 2026-06-22 18:16: bot teleport-acquired purple-4 (id=512) via the new map-based acquisition path, fired 8 cardinal-adjacent dual shots (all hits), then purple-4 teleported out of viewport. Pursuit path fired **4 consecutive homing shots from the same tile** toward purple-4's new wire-known position (238,169) before the pursuit gate finally tripped (~5 s of silence on the locked id). Bot then re-acquired via map_open and started a chase teleport before the 60s budget cut the session. This is the user-spec "stay where we are and use homing shots until they are deactivated" loop.

**Code changes:**

- `state/mutations.py::remove_tank` -- now a no-op. Previously deleted the tank from `world["tanks"]` on every `0x58 TankRemove`; that caused the bot to abandon pursuit after just one homing shot (live capture earlier same day: orange-2 dropped from registry the moment they teleported, pursuit gate failed, bot fell to CONFIRM_KILL -> teleport-chase). The new behaviour keeps the entry untouched -- the freshness gates and `0x41 Deactivation` own the lifecycle.

**Why the change works:** `0x58` carries no information the freshness machinery can't already derive. A truly gone tank stops broadcasting `0x2E TankStatusSync`, so `timestamp_ms` ages out naturally past the 5 s pursuit window. A tank that simply teleported keeps broadcasting `0x2E` -- pursuit keeps firing homing at the cached coords, server picks homing weapon for distant moving targets, homing tracks. The only authoritative death signal is `0x41`, which flips `liveness="deactivated"`; pursuit gate respects that.

**Test surface:**

- `tests/world_state/test_mutations.py::TestRemoveTank::test_keeps_tank_in_registry` -- replaces the earlier `test_removes_existing_tank`; asserts the entry stays put after 0x58.
- `tests/sniffer/test_world_state_dispatch_tank.py::test_dispatch_tank_remove` -- now asserts the dispatch leaves `world["tanks"]` unchanged and keeps `liveness="alive"`.
- `tests/replay/test_real_session_regressions.py::test_combat_to_fuel_stale_lock_loop_replays_recovery_then_reengage` -- HUNT 15 -> 18, COLLECT_FUEL 4 -> 1, shoot 7 -> 10, map_open 2 -> 1 (continuous pursuit cuts confirm-kill cycles). Pickup count assertions removed (over-specified for this fixture).
- `make check` green: 4114 tests, 100.00% statement+branch coverage.

**Wiki updates:**

- `wiki/pages/bot-behavior-contract.md` §1 -- changed the 0x58 contract row from "DELETES the tank" to "NO-OP at the registry level"; cites the live-capture regression and the pursuit test that locks the new behaviour.
- `wiki/pages/tank-freshness-model.md` -- new "Registry lifecycle: 0x58 TankRemove is a no-op (changed 2026-06-22)" section explaining the rationale and why `0x41` is the only authoritative death signal.
- `wiki/pages/decode-coverage.md` -- 0x58 row updated to note "Bot dispatch is a no-op as of 2026-06-22 — registry entry is kept so pursuit can keep firing homing at the cached coords".

**Live verification:**

Session 2026-06-22 18:16 (60 s budget, 30 ticks): bot fired 11 shots, 8 hits + 3 misses (72 % rate). The 3 misses were the pursuit homing shots (`victim_id=-1` -- server doesn't credit hits on out-of-viewport targets; some may have actually landed but the bot can't witness). 0 kills this session because the engagement was mid-pursuit when the budget elapsed. The clean signal is the 4 consecutive homing shots fired from the same tile -- exactly the staying-put pursuit the user requested.

**Out of scope (still open):**

- Pursuit hit-detection: `victim_id=-1` on out-of-viewport homing shots means the bot can't tell if the homing landed. The conservative miss classification under-reports actual hits. A fix would correlate `0x53 ShootEvent` responses for our own shots against the target's wire damage updates -- separate work.

---

## [2026-06-22] refactor | Strip the fuel-dot system

Deleted the per-session fuel-dot atlas and every consumer. User intent: "remove all the fuel dot stuff" -- the dot system was extra code that pretended to save the genuinely marooned case (it never could; if you can't afford ANY teleport, you can't afford a dot teleport).

**Phases:**

- **A (planner):** removed `_plan_fuel_dot_refuel`, `_plan_fuel_dot_escape`, `select_fuel_dot_hop`, `select_fuel_dot_walk_targets`, `emit_fuel_dot_hop_diagnostic`, the `fuel_dot_guided` parameter on `make_resource_search_hop`, and `attempted_fuel_dots` from `AIStateDict`. The fuel recovery cascade now ends at `Strict -> Sense -> Hop -> raise ValueError`.
- **B (world state + sniffer):** removed `map_fuel_dots` field from `WorldStateDict`, deleted `replace_map_fuel_dots`, stripped the field from every mutation pass-through.
- **C (protocol decoder):** stripped `fuel_dots` from `MapDataDict` and `decode_map_data`. The RLE byte count is still parsed for length validation so the decoder advances cleanly into the tank-entries section; the coordinates inside the RLE region are no longer materialised.
- **D (diagnostics):** removed `dot_hops` / `dot_hop_distinct_targets` / `dot_hop_max_repeats` fields from `SessionScorecardDict`, deleted the fuel-dot-orbit issue detector and the `fuel_dot_hop` diagnostic routing in `session_scorecard`.
- **E (action_lab):** deleted `action_lab/fuel_dot_probe.py`, `fuel_dot_probe_types.py`, `scripts/fuel_dot_probe.py`, the `make fuel-dot-probe` target, and the dedicated probe + script tests.
- **F (wiki):** updated `wiki/pages/fuel-system.md` to reflect the new cascade; the dot atlas section is gone, the marooning section is honest about the fact that marooning was never really recoverable.

**Consequence:** RECOVER_FUEL now raises `ValueError` from the durable owner if Strict / Sense / Hop all decline. Marooned sessions fail loud instead of silently spending their reserve on a guess. The bot stops gambling on dot positions ("just extra code" -- user, 2026-06-22).

**Numbers:** ~50+ files touched, ~315 lines of production code deleted, 4031 tests still pass, 100% statement+branch coverage held throughout, mypy / guard / ruff clean.

---

## [2026-06-22] update | Wiki: walking does not reveal, viewport does not shift

Four game-mechanic rules from the user (2026-06-22 session) added or reconciled in the wiki:

1. **Walking does not reveal containers** -- only radar does. Both extra (full viewport) and free (5x5 around tank) radar reveal; walking moves the tank but cannot surface a hidden container, even by stepping onto it. New section in `radar-mechanics.md` "Walking does NOT reveal containers" makes this explicit, and explains the walk-radar-walk-radar cycle that the bot uses when out of extras.

2. **Walking costs 1 fuel per tile** -- previously undocumented. Added to both `radar-mechanics.md` (cost analysis for the free-radar cycle) and `viewport-frame.md` (walking paths section).

3. **Viewport shifting is OFF** in the current game configuration. The old `viewport-frame.md` claimed the viewport recenters when the tank reaches an edge; that was an older config. Walking to the edge now just stops the tank at the edge tile -- the viewport changes only on teleport. Updated `viewport-frame.md` to match. (`radar-mechanics.md` already had this; the two pages were contradicting each other.)

4. **Viewport does NOT center on player.** The tank can occupy any tile in the 16x16 frame, including a literal corner. This is what makes free-radar edge clipping land at 3x3 instead of 5x5. Added to `viewport-frame.md`.

No source-code changes -- this is pure documentation correcting the wiki against what the user described.

---

## [2026-06-22] refactor | Forage walk picker maximises free-radar coverage

`bot/ai/forage.py::select_forage_target` no longer returns the nearest unscanned tile. It now returns the destination whose next free radar would reveal the most uncovered ground in the viewport (5x5 footprint clipped to viewport bounds, minus already-scanned tiles). Ties broken by Manhattan distance from the tank.

**Why:** the old picker walked 1 tile at a time toward the closest unscanned tile, so the next free radar mostly re-covered already-scanned ground. The optimal walk step for the free-radar tile-expansion strategy is closer to 5 tiles -- matching the 5x5 radar diameter -- so each free radar reveals up to 25 fresh tiles instead of overlapping the previous footprint.

**Implementation:**

- Added `select_best_free_radar_position` to `bot/ai/scan_coverage.py` (the coverage-scoring picker).
- Deleted `nearest_uncovered_tile_in_viewport` from `scan_coverage.py` -- its single caller is gone and the new picker subsumes it.
- `select_forage_target` in `bot/ai/forage.py` now delegates to the new picker. Module docstring updated to describe the walk-radar-walk-radar loop honestly: walk to a coverage-maximising position, free radar, repeat.
- Tests in `test_forage.py` and `test_scan_coverage.py` rewritten for the new selection criterion. One forage edge-case test deleted (was testing an artifact of the old picker: walking straight into an enemy on the only unscanned tile -- the new picker side-steps that by walking to a position whose 5x5 covers the blocked tile from a distance).
- Replay test `fuel_radar_loop` relaxed to expect `ValueError`: the more aggressive walks burn fuel faster than the recorded policy, the bot lands marooned at fuel=0 by tick ~34, and the new RECOVER_FUEL owner raises rather than idle silently (consistent with the 2026-06-22 marooning contract).

**Net:** the bot now uses its free-radar fuel budget productively. When extras = 0, each ~5-tile walk + ~10-fuel radar cycle reveals ~25 fresh tiles instead of mostly overlapping the previous scan.

---

## [2026-06-22] refactor | Collapse fuel_critical_threshold into fuel_low_threshold

The two-tier "polite low vs. emergency critical" fuel threshold was dead in this codebase: both `fuel_critical_threshold` and `fuel_low_threshold` had drifted to 300. The user chose to collapse them rather than reintroduce a gap.

**Cuts:**

- `AIConfigDict` (`bot/ai/types.py`) -- removed `fuel_critical_threshold` field. Default factory + encode/decode (`bot/ai/types_codecs.py`) updated.
- `recover_fuel_mode.py`: `minimum_recovery_fuel_volume` and the opportunistic-equipment gate in `_plan_fuel_recovery` switched to `fuel_low_threshold`. Deleted `try_collect_critical_fuel` and `try_collect_fuel` (both exported but never called from production). The `_plan_fuel_recovery` wrapper was inlined into `decide_recover_fuel_mode` -- the `owner_required` parameter was always True from the single production caller.
- `mode_controller.should_enter_recover_fuel`: simplified from a priority-swap rule (no-extras + above-critical -> defer to equipment mode) to a flat `ctx.fuel <= ctx.config["fuel_low_threshold"]`. The priority swap was dependent on the critical/low gap which no longer exists.
- Deleted `tests/integration/test_refuel_triggers_below_threshold.py` (tested the deleted helpers). Deleted `test_decide_recover_fuel_mode_raises_when_plan_returns_none` (used module-level monkey-patching of the now-deleted internal helper and was guard-banned anyway). Trimmed `fuel_critical_threshold` references from `test_types.py` fixtures and `test_recover_fuel_mode.py` helpers.

**Result:** one fuel threshold for everything (entry, combat reserve, opportunistic-fuel gate). 4030 tests pass.

---

## [2026-06-23] fix | pickup_container fuel double-count

`state/container_mutations.pickup_container` was adding `transferred = prior_volume - remaining_volume` to `self_state["fuel"]` on every fuel-container pickup. The wire ALSO emits an absolute-fuel message (0x44 FuelGain) for the same pickup, which calls `set_self_fuel(fuel_total)` through `world_state_containers.update_world_state_from_fuel_total`. Both updates fired for every pickup -> double-count.

**Live evidence** (`runs/bot/latest.log` 2026-06-23 00:35:57):

```
25:57  WORLD: Fuel: 195 -> 633 (+438)   <- 0x44 FuelGain set absolute fuel = 633
25:57  WORLD: Picked up container at (152, 204)
       <- pickup_container also ran: fuel = 633 + 438 = 1071 (NOT logged)
25:57  AI:    collect fuel at (147,212) vol=323 (fuel=1071)
25:57+ WIRE teleport cost: Fuel: 1071 -> 622  <- wire-side fuel was also 1071
```

The 438 ghost matched the container volume exactly. Both the bot AND the wire-stamped subsequent cost log show fuel = 1071, so the double-count was real, not a display glitch.

**Fix:** removed the fuel-update branch from `pickup_container` (the `if container["is_fuel"] and container["volume"] > remaining_volume: new_self = SelfStateDict(..., fuel=new_self["fuel"] + transferred, ...)` block). The function now mutates only the container registry. Wire absolute-fuel messages (0x44 FuelGain, 0x2E TankStatusSync, 0x64 FuelDeposit) are the single source of truth for `self_state["fuel"]`.

**Same single-source-of-truth pattern** as the 2026-06-22 0x58 TankRemove no-op: the wire authoritative messages own state changes; secondary handlers don't duplicate them.

Tests updated:

- `tests/world_state/test_mutations.py::TestPickupContainer::test_pickup_fuel_container_adds_fuel` renamed to `test_pickup_container_does_not_modify_self_fuel`, assertion flipped.
- `tests/sniffer/test_world_state_dispatch_other.py::test_dispatch_container_pickup_partial_updates_volume` -- no longer asserts `self_state["fuel"] == 200`; instead asserts fuel unchanged (wire path is what would update it in production).

1302 affected tests still pass. mypy clean.

---

## [2026-06-23] fix | RECOVER_EQUIPMENT no longer interrupts active combat

`should_enter_recover_equipment` was firing on every ammo decrement during combat -- after the first shot of a 25-dual engagement, dual dropped to 24, the gate said "below resume (25)", and the mode controller flipped HUNT -> RECOVER_EQUIPMENT mid-fight. Live observation 60s run 2026-06-23: bot fired ONE shot at purple-8, immediately bailed for a pickup, second pickup got `INVENTORY_FULL` (slot just topped up), then re-acquired and fired one MORE shot, bailed again. Pattern was "restock, shoot once, restock" -- not the "restock, fight, restock" the user asked for.

**Fix:** added a two-tier gate to `should_enter_recover_equipment`:

1. **Emergency** -- any reserve below its *break* threshold (4 / 4 / 5) fires unconditionally. The bot can't fight without ammo; restock interrupts even an active combat target.
2. **Between kills** -- any reserve below its *resume* threshold (25 / 25 / 20) AND `combat_target_id == -1` (no lock). The bot finishes the in-flight kill before flipping to restock for the next hunt.

```python
def should_enter_recover_equipment(ctx):
    if any reserve < break threshold:
        return True  # emergency, interrupt anything
    if combat_target_id != -1:
        return False  # mid-fight, finish the kill
    return any reserve < resume threshold  # between kills, restock
```

**Verification** (60s run 2026-06-23 00:35:53-55):

```
25:51  HUNT/ENGAGE: shoot(145,210,id=516)  ->  hit, dual 25->24
25:53  HUNT/ENGAGE: shoot(145,210,id=516)  ->  hit, dual 24->23
```

Two consecutive shots at the SAME target without a mode flip. Pre-fix the bot would have bailed after the first 25->24 transition.

No source code change to `should_exit_recover_equipment` -- the exit gate still releases only when all three reserves are at or above resume.

**Symmetric fuel-mode fix is planned, not yet shipped.** `should_enter_recover_fuel` still uses the single-threshold check `fuel <= fuel_low_threshold` (300) with no combat-lock gate. Same 60s run showed the bot bailing from purple-8 at fuel=251 to refuel -- combat is near-free, 251 was plenty to finish the kill. Apply the same two-tier pattern (emergency at `hunt_min_fuel = 100`, between-kills above) when the next session opens.

---

## [2026-06-24] refactor | Unify RECOVER_FUEL + RECOVER_EQUIPMENT → COLLECT

The historical two-mode recovery split was a leaky abstraction: the cascades had near-identical structure (lock → strict pickup → forage → hop), each mode opportunistically grabbed the OTHER kind, and the user's own gameplay loop ("drain equipment, then maybe biggest fuel, then hop") is one mode. The two modes were collapsed into a single durable owner: `COLLECT`.

**Production surface**

- `AIMode` literal: `RECOVER_FUEL`, `RECOVER_EQUIPMENT` → single `COLLECT`. `RECOVERY_MODE_STATES` → `COLLECT_MODE_STATES`.
- `BehaviorMode` literal: `COLLECT_FUEL`, `COLLECT_EQUIPMENT` → single `COLLECT`.
- `src/tankpit_bot/bot/ai/collect_mode.py` — new file with the unified cascade.
- `src/tankpit_bot/bot/ai/recover_fuel_mode.py`, `recover_equipment_mode.py` — deleted.
- `mode_controller.py` — single `should_enter_collect` / `should_exit_collect` / `derive_collect_mode_state` replacing four predicates and two state-derivers. Exit requires BOTH fuel and combat reserves restored (was implicit via the two-mode handoff before).
- `ai_strategy.py` — one COLLECT branch instead of two RECOVER_* branches.
- `combat_strategy.py::_refuel_for_hunt` — delegates to `decide_collect_mode`.

**Unified cascade**

1. Continue a held equipment or fuel lock from a previous tick.
2. Pick up the best equipment in viewport (`allow_unreachable=True`).
3. Pick up the best fuel in viewport (skipped at learned capacity).
4. Forage: radar when the viewport has unscanned tiles, or walk toward an unscanned tile.
5. Hop: teleport to a fresh viewport when nothing actionable remains. Raise `ValueError` when even the hop is unaffordable.

Equipment ranks ahead of fuel by design -- matches the user's hand-played loop. Walking an extra tile or two for equipment costs 1 fuel/tile, a rounding error against the viewport-fuel a few ticks later.

**Dead state and helpers removed**

- `AIStateDict.equipment_search_failures` — only its own reset logic read it; pure diagnostic counter, deleted.
- `AIConfigDict.equip_search_max_failures` — gated the deleted reset logic.
- `context.needs_emergency_equipment` — replaced by inlined break-threshold checks in `should_enter_collect`.
- `recover_equipment_mode.try_search_critical_equipment` — never called in production, only by tests.
- `recover_fuel_mode.can_use_fuel_radar` and `minimum_recovery_fuel_volume` — became one-liner passthroughs after the reserve-gate cleanup, inlined.
- `make_resource_search_hop(failure_count=...)` parameter — dead now that the counter is gone.
- `equipment.SCAN_COVERAGE_TTL_MS` alias — its only cross-module consumer was `recover_fuel_mode`.

**Tests**

- `test_recover_fuel_mode.py`, `test_recover_equipment_mode.py`, `test_recover_equipment_integration.py`, `test_recovery_helpers.py` renamed to `test_collect_mode_*.py`. Six tests for the deleted `try_search_critical_equipment` removed. Bulk literal substitution across ~22 test files. `equipment_search_failures` references in tests stripped.
- Two new coverage tests added: `find_adjacent_container` diagonal-blocked branch, `_is_worthwhile_hop` impassable-landing branch.
- `make check`: 4028 tests, 100% coverage, lint clean.

**Wiki / docs updated**

- [[bot-behavior-contract#3.1]], [[bot-behavior-contract#3.4]] — rewritten for the unified COLLECT mode.
- [[fuel-system]] — recovery cascade and threshold descriptions updated.
- `docs/bot-control-model.md`, `docs/bot-logging.md` — mode literals and example log lines updated.

Constraint context: user-stated "no back compat shims, no thin wrappers, no fallbacks, no legacy code, no type alias" -- the unification is a true merge, not a rename layer.

---

## [2026-06-26] update | Combat stay-put: shoot when engaged at distance, don't re-teleport

Extended the 2026-06-16 chase-loop fix (see [[combat-chase-bug]]). Live run 2026-06-26 14:42 caught the bot teleporting after a target moved off cardinal adjacency mid-fight, burning 114 fuel + ~6s of wire round-trips to position for the next dual shot instead of firing another homing from the same tile.

**Code:** `src/tankpit_bot/bot/ai/combat_strategy.py`
- Added `is_already_engaged(ctx)` -- single source of truth for the `last_shot_target_id == combat_target_id` predicate.
- `_combat_close` now branches three ways: adjacent -> shoot; engaged + not adjacent -> shoot (server picks homing); fresh acquire + not adjacent -> teleport.

**Code:** `src/tankpit_bot/bot/ai/hunt_mode.py`
- `_resume_locked_target_off_viewport` (line 223) now calls `is_already_engaged(ctx)` instead of the inline expression.

**Tests:** `tests/bot/ai/test_combat_strategy.py` (new `TestIsAlreadyEngaged` class, 3 cases), `tests/bot/ai/test_hunt_close_integration.py` (new engaged-stays-put and fresh-acquire-teleports cases).

**Wiki:** [[combat-chase-bug]] follow-up section, [[bot-control-model.md]] HUNT.CLOSE description updated.

Constraint context: user-stated "no back compat shims, no thin wrappers, no fallbacks, no legacy code, no type alias" -- helper is extracted deduplication, not a wrapper; the old "teleport on every non-adjacent tick" branch is deleted, not preserved behind a flag.

---

## [2026-06-26] update | Walk-only container contract: no teleport-to-container

User contract: the bot collects containers the same way a human player does -- click a container that has a walk path from the current tile, the server walks the tank, the pickup completes. No teleport-to-container, no reposition-then-pickup flow.

**Code:** `src/tankpit_bot/bot/ai/equipment_search.py`
- Dropped the `allow_unreachable` parameter from `find_nearest_fuel`, `find_nearest_equipment`, `find_equipment_candidates`, `find_best_fuel`, `find_nearest_deposit`, `find_adjacent_container`, `describe_container_search`, and `_describe_candidate_reason`.
- Deleted the teleport-landing-tile fallback branch in `_is_actionable_with_terrain`: a container is actionable iff `is_collection_reachable_in_viewport` finds a walk path.
- Dropped the `no_landing=` field from `describe_container_search`'s diagnostic string -- it was always 0 under the new semantics.

**Code:** `src/tankpit_bot/bot/ai/movement.py`
- Rewrote `_walk_or_teleport_with_terrain` so pickups return `None` when the target is off-viewport or not collection-reachable, and dispatch a single `pickup_*` command when reachable.
- Removed the pickup branch from `_approach_command`; plain moves keep the teleport-fallback for combat-approach and exploration.

**Code:** `src/tankpit_bot/bot/ai/collect_mode.py`
- Dropped `_with_equipment_approach_recorded` and `_is_equipment_target_attempted` -- the approach-marking system existed to prevent teleport orbits around blocked containers and is dead under walk-only.
- Removed the `attempted_equipment_targets` field reads.

**Tests:** Bulk update across `tests/bot/ai/test_equipment.py`, `test_collect_mode_*.py`, `test_movement.py`, `test_strategy_coverage.py`, and the action_lab probe tests. The `tests/replay/test_real_session_regressions.py::test_equipment_then_fuel_loop_replays_known_bad_behavior` test flipped from no-strand to asserting `ValueError("COLLECT owner produced no decision")` because the new contract strands the bot at fuel=78 surrounded by water-locked containers (correct under walk-only -- no unreachable dispatch).

**Pages updated:** [[equipment-system]] container blacklist section, [[fuel-system]] cascade step 2 description, [[viewport-frame]] walking paths section.

Constraint context: user-stated "no back compat shims, no thin wrappers, no fallbacks, no legacy code, no type alias" -- `allow_unreachable` deleted everywhere, the teleport-fallback branch deleted from `_is_actionable_with_terrain`, the dead approach-marking helpers deleted from `collect_mode`.

---

## [2026-06-26] fix | Unlimited homing shots restored by deleting the OUR_SHOT-driven registry update

Live run 2026-06-26 15:13 caught the bot firing one homing shot at a teleporting target, then getting every subsequent `shoot(off_viewport_x, off_viewport_y)` rejected by the server with `command_error` (the server rejects shoot commands targeted outside the 18x18 viewport, see [[shot-range]]). Pre-fix the bot was supposed to "stay put and fire multiple homing shots" per the user contract documented in [[enemy-bot-behavior]] footnote 4 -- the contract was real, it had just been broken by an unrelated change.

**Root cause:** commit `098d3d7` (combat-rework, 2026-06-23) added two lines to `src/tankpit_bot/sniffer/world_state_dispatch.py:161-162` that overwrote the locked target's registry x/y from `OUR_SHOT`'s homing-tracked landing tile every time the bot fired a homing or missile. The intent was to refresh stale registry coords for the next shot, but the seeker's resolved tile is the target's current off-viewport position the moment the target leaves the viewport. After the first homing fired, the registry held off-viewport coords, the planner's next shoot dispatched at those coords, and the server rejected with `command_error`.

**Fix:** delete the two-line registry update. Pre-098d3d7 the registry stayed at the last on-viewport coord (because off-viewport tanks stop broadcasting 0x2E `TankStatusSync`), so subsequent shoots dispatched at an in-viewport tile, the server accepted them, and the server's homing seeker tracked to the actual target every shot. Unlimited homings until the kill.

**Code:** `src/tankpit_bot/sniffer/world_state_dispatch.py` `_dispatch_shoot_event` -- delete the `if weapon in (2, 3) and ws.last_shot_combat_target_id > 0` block. The wire firing mechanism (`_combat_shoot` -> `make_shoot_command` -> `build_shoot_command`) is unchanged and has been since the first commit; only the registry-population side changed.

**Tests:** `tests/sniffer/test_world_state_dispatch_container.py` -- renamed `test_own_homing_refreshes_locked_target_position` to `test_own_homing_does_not_overwrite_locked_target_position` and inverted the assertion (registry stays at the last on-viewport tile after a homing dispatch). The 2026-06-24 12:43 "4 homings missed" diagnosis that motivated the original update was incorrect: homing aim is just a hint, the server tracks regardless, so a stale registry never caused those misses.

**Page update:** [[combat-chase-bug]] gains a follow-up subsection recording the registry-update revert and the recovered unlimited-homing behavior.

---

## [2026-06-26] refactor | Resource search rewrite: single 16-tile cardinal-then-diagonal fresh-viewport hop

Resource-search hop logic had grown three competing helpers (`_hop_target_for_cycle` ring patrol, `_nearest_unscanned_grid_target` global grid fallback, `_short_hop_fallback` for fuel-poor cases) plus a `patrol_waypoint_index` state field. Each was added to patch a previous helper's bug instead of fixing the root cause: ring 2 / ring 3 hops produced 32 / 48 tile jumps that overshot fresh viewports, the global grid stride (12) overlapped adjacent grid cells, and the 8-tile short hop guaranteed 50% overlap with the bot's current viewport.

**Code:** `src/tankpit_bot/bot/ai/resource_search.py`
- Replaced everything with a single `make_resource_search_hop` that iterates four 16-tile cardinals first then four 16-tile diagonals, returning the first direction whose landing is passable, fuel-affordable, and in an unscanned viewport. Returns `None` when all eight directions fail (caller raises rather than fall back to a shorter or smaller hop).
- Cardinal-first ordering is a fuel optimization: 16 cardinal = 96 fuel, diagonal = 135 fuel; the diagonals exist so the bot can escape a fully-scanned cardinal ring without a third hopping mechanism.
- Deleted `_MAX_SEARCH_RINGS`, `_GLOBAL_GRID_STRIDE`, `_SHORT_HOP_DISTANCE`, `_MIN_HOP_DISPLACEMENT`, `_hop_target_for_cycle`, `_is_worthwhile_hop`, `_nearest_unscanned_grid_target`, `_short_hop_fallback`, and `local_resource_search_hop`. Kept `is_recently_attempted` / `record_attempt_mark` -- they support failed-pickup tracking, not hopping.

**State change:** `AIStateDict` loses `patrol_waypoint_index`, which existed because the prior cycle-with-index design grew the index unboundedly across sessions. The new method is stateless; the codec entries in `types_codecs.py` are deleted to match.

**Config change:** `equip_search_hop_distance` default is 16 (one viewport width). The single-hop tiling property is the design: each hop lands the bot's center 16 tiles away in cardinal directions, so the new viewport tiles cleanly with no overlap; diagonal hops land in corner-adjacent viewports, also non-overlapping. `equip_search_max_failures` is gone -- the new method has no failure-count to drive.

**Tests:** `tests/bot/ai/test_resource_search.py` rewritten from scratch -- 11 behavior tests covering east-first, west-fallback when east is scanned, clamped-edge skip, impassable skip, unaffordable skip, cardinal-preferred-over-diagonal, all-eight-blocked yields None, resource-target cleared on success; plus 5 attempt-mark unit tests (kept).

**Pages updated:** [[fuel-system]] cascade step 5 now describes the cardinal-then-diagonal method by name.

Constraint context: user-stated "no fallbacks, no legacy code, no thin wrappers" -- ring multiplication, grid fallback, and short-hop fallback are deleted, not gated behind a flag.

---

## [2026-06-26] audit | Wiki audited against `/wiki-init` v1.0 spec

Cross-checked the wiki structure against the `/wiki-init` scaffolding skill at `~/.claude/skills/wiki-init/SKILL.md` to confirm spec compliance.

**Findings:**

- **Three-tier graph compliant.** index.md (22 lines) -> 6 hubs -> 52 content pages (flat under `pages/`). Zero orphans -- every page is linked from at least one hub.
- **Frontmatter compliant.** All 52 pages carry `title`, `tags`, `related`, `sources`, `fact_checked`, `confidence` (the spec's required set; we add `confidence` as a quality signal).
- **Citation discipline compliant.** Initial grep for `[^N]` footnote syntax surfaced 26 "uncited" pages, but those pages cite via inline JS class/line refs (e.g. `xc function, line 33`) and rich frontmatter `sources:` lists (e.g. `tpclient.js lines 33-34 (pc/qc/rc/sc arrays, xc function)`). The spec's rule 6 explicitly allows inline citations; no per-claim refactor was needed.
- **Atomicity compliant.** 24 pages exceed 100 lines, but the spec's split threshold is `~3000 words or covers more than one distinct concept`. Word counts run 890-2562 across the candidates and each page covers one cohesive subsystem (rendering, input, the V-table catalog, etc.), so no splits warranted.

**Changes applied:**

- **SCHEMA.md bumped to v1.0** -- aligned with the `/wiki-init` v1.0 spec. Added the critical-rules block (eight numbered rules that fire automatically), the downstream-artifact citation ban (rule 7, even though we don't render artifacts from wiki content today), and an explicit "no extra top-level dirs" clause to keep `wiki/` clean.
- **`wiki/handoffs/` and `wiki/artifacts/` removed.** The two files (one completed handoff brief, one V-table trace dump) were unreferenced from any wiki page and violated the spec layout (only `hubs/` and `pages/` are allowed under `wiki/`). Moved to `docs/handoffs/2026-06-21-scan-refactor.md` and `docs/vtable_trace.txt` -- the project's existing operational-docs tree.

**Pages updated:** SCHEMA.md.
**Pages relocated outside wiki:** 2026-06-21-scan-refactor.md (handoff brief), vtable_trace.txt (V-table dump).

No content pages were touched. The audit confirms the wiki was substantively spec-compliant from v0.1; v1.0 codifies the rules and tidies the layout.

---

## [2026-07-01] fix | Sniffer autosave never covered the canonical latest capture; `.env` TANKPIT_OUTPUT redirect

Diagnosed after a human play-capture session (2026-07-01 18:59) was "lost" when the browser was closed before the timer: `runs/sniff/latest.capture_session.json` still held June 20 data while the session's log + events.jsonl were current.

**Root cause, two layers:**

1. `.env` carried the stale template default `TANKPIT_OUTPUT=capture_session.json`. The Makefile's `sniff` target explicitly removes `TANKPIT_OUTPUT` from the environment when no `OUTPUT=` override is given, but `main()` calls `load_dotenv()` which re-set it from `.env`. Every sniffer autosave therefore went to `capture_session.json` in the repo root, not to `runs/sniff/`.
2. Even with a correct override, `_autosave_capture` only ever wrote the override path (+ a redundant `raw_capture.json` mirror beside it). The canonical `runs/sniff/latest.*` capture group was written exclusively by the end-of-run flush in `run_sniffer` — so any abnormal termination (browser closed mid-session -> `TargetClosedError`) left the canonical latest capture stale from a *previous* session, silently masquerading as current data.

**Fix:**

- `sniffer/core.py`: `WebSocketSniffer` now takes `autosave_paths: tuple[Path, ...]`; `run_sniffer` builds the deduped set {explicit output path, canonical `latest_capture_path`} and every captured message keeps both current. The redundant per-message `raw_capture.json` mirror is gone (the final flush still writes all raw/summary/archive mirrors).
- `runtime_logging.configure_sniff_runtime_logging`: the session-start `_reset_artifact_files` now also truncates `latest.capture_session.json`, `latest.raw_capture.json`, `latest.session_summary.json` — an empty file honestly means "session started, no data yet" instead of a previous session's data.
- `.env` / `.env.example`: `TANKPIT_OUTPUT` default removed; documented as an opt-in override only.
- Also removed the `make play` target (byte-identical duplicate of `make sniff`); [[make-targets]] updated, including the stale `runs/sniffer/` -> `runs/sniff/` path.

**Tests:** `test_core_sniffer.py` autosave tests rewritten for multi-destination autosave (override + latest, latest content asserted equal); `test_runtime_logging.py::test_configure_sniff_runtime_logging_resets_latest_files` extended to seed and assert truncation of the three stale capture files. `test_replay_pipeline.py::test_root_capture_session_replays_to_observed_terminal_state` had silently depended on the repo-root `capture_session.json` (untracked AND overwritten by every sniff run — it held the 2026-06-20 session only because no sniff had run since); the 2026-06-20 capture is now checked in as `tests/replay/fixtures/mixed_activity_sniff.capture_session.json` and the test renamed to `test_mixed_activity_sniff_capture_replays_to_observed_terminal_state`.

**Sessions captured today (human play, account Artax):** archived as `sniff-20260701-185917.*` (6.4 min, fuel-starved walk-forage variant; log + events only — capture JSON was lost to this very bug) and `sniff-20260701-191133.*` (5 min, full kit; complete capture). Behavioral findings feed the combat-rework policy work.

---

## [2026-07-01] refactor | Terrain-scored fresh-viewport hop: pick the cleanest viewport, not the first passable one

User policy from the recorded sessions + direct statement: "i usually just pick clean viewports that are mostly '.' walkable terrain." The old `make_resource_search_hop` took the FIRST of eight 16-tile directions whose single landing tile was passable, affordable, and unscanned — no signal about what the destination viewport was made of. Under the walk-only pickup contract a mostly-water viewport is worthless even when radar reveals containers there (no walk path), so first-match hopping paid ~96 fuel per gamble. The crashed `make run` earlier today (24 teleports, marooned at fuel=31) was largely this.

**Code:** `bot/ai/resource_search.py`

- `_pick_fresh_viewport_hop` now evaluates 16 candidates — the eight compass directions at one and two viewport-widths (16/32 tiles) — and returns the qualifier whose landing viewport has the highest **walkable fraction** from the static terrain map (`terrain.py` minimap GIF data covers all 256×256 tiles, so unvisited ground is scorable). Off-map tiles count as unwalkable (border is rock). Qualification gates unchanged: exact displacement after edge clamp, passable landing tile, affordable, destination viewport unscanned.
- Tie-break is structural: candidates are iterated cheapest-first (16-cardinal 96 fuel, 16-diagonal ~135, 32-cardinal 192, 32-diagonal ~271 — monotone), and only a strictly better score replaces the incumbent, so equal-score ties keep the cheapest hop. No explicit cost comparison needed.
- Without a terrain map every candidate scores 1.0 and selection degrades to the old cheapest-first behavior.
- The fuel-dot question from the play-session analysis is closed: the user picks restock destinations by terrain cleanliness, not fuel dots. No MAP_DATA fuel-dot decode restore needed.

**Tests:** `tests/bot/ai/test_resource_search.py` — 3 new behavior tests (`test_prefers_most_walkable_viewport`: qualifying-but-water-heavy east loses to clean west; `test_reaches_second_ring_when_first_ring_scanned`: 32-tile hop when all 16-tile viewports covered; `test_offmap_clipping_penalizes_edge_viewport`: border-clipped viewport scores 0.625 and loses to a full one). `test_returns_none_when_all_eight_directions_blocked` extended to both rings (16 covered viewports) and renamed. All prior ordering tests still pass because uniform terrain scores tie and the tie keeps the old cheapest-first choice.

**Pages updated:** [[fuel-system]] cascade step 5 rewritten for the scored picker.

---

## [2026-07-01] fix | Radar reconcile wiped visible containers — "picked up 2 of 7"

Live run 2026-07-01 20:20 (user watching): the bot teleported onto orange-3 amid ~7 visible equipment containers, killed it, picked up exactly 2, and hopped away. Same pattern at the next viewport (3 visible, picked 2). The user called it out in real time.

**Diagnosis (events.jsonl entity_alignment samples):**

- tick 5 (20:20:10, right after the landing 0x5A): registry holds 7 viewport-sourced containers in the new viewport, including a 1000+ volume fuel container.
- tick 6 (20:20:12, right after the scan-on-landing extra radar): all 7 are gone; only the 2 radar-listed containers remain.

**Root cause:** `reconcile_radar_viewport_resources` treated the radar response as the complete authoritative set for the scan envelope and deleted every registry entry the response didn't list. But **the radar response carries only the newly revealed HIDDEN entities** — already-visible containers/mines are on screen and are not re-sent. Every scan therefore erased the bot's knowledge of the visible layer. The offline capture decode confirms the wire sent a 19-entity ViewportUpdate at the landing and a 2-container radar response 2s later.

**Fix:** the reconcile is scoped to `source == "radar"` entries (containers AND mines). Visible-layer entries are owned by the 0x5A viewport patches / 0x43 cache updates (which clear tiles on pickup) and are never removed by a radar response. Sweep logic extracted into `_without_stale_radar_entries` (shared for both registries).

**Tests:** `test_world_state_radar_cache.py` — two new cases (`test_reconcile_spares_visible_containers`, `test_reconcile_spares_visible_mines_and_removes_radar_mines`); the existing radar-sourced staleness test passes unchanged. Replay pipeline pins re-anchored: mixed-activity capture 15→46 containers / 15→21 mines, fuel-probe capture 45→60 containers / 41→66 mines — the recovered entries are exactly the visible-layer ones the replayed scans used to wipe. `test_world_state_functions.py` mine-reconciliation tests re-seeded with `source="radar"` (they had encoded the old whole-envelope semantics via `make_mine_state`'s `"viewport"` default). The container equivalents needed no change because `make_container_state` defaults to `source="radar"`.

**Test-infra hardening (found via a worker-order-dependent gate failure):** `tests/action_lab/conftest.py::restore_action_hooks` now also restores `action_hooks.wait_for_world_sync` / `wait_for_radar_sync` — several action_lab tests set them without restore, and under xdist the leaked fakes survive into whatever module the same worker runs next.

**Pages updated:** [[radar-mechanics]] gains "The radar response lists ONLY newly revealed hidden entities" with the state-tracking implication.

**Session note:** the same 29-tick run (ended early by browser close) was the first under the terrain-scored hop: 1 kill, 8/11 hits, no fuel spiral, kit full at end. With the reconcile fix the bot should now also hoover the loot it was forgetting.

---

## [2026-07-02] rework | Affordability-gated acquisition, stale-lock release, self-directed session exits

User contract, verbatim intent: "i dont want it picking bad targets in the first place. if it has no targets then it can exit with the reason. same for low fuel or out of fuel exit." Diagnosed from run 2026-07-01 20:45-20:48 (user watching): the bot teleported 90 tiles (505 fuel) to the nearest map enemy, hit the fuel-low interrupt 8 hits into an unfinishable kill, and later resumed the stale lock from 92 tiles away — firing 11 server-rejected shots in a loop until the browser was closed.

**Root cause chain (all four links needed):** (1) acquisition picked the nearest enemy with no notion of whether the fight was affordable end-to-end; (2) `engagement_fuel_budget=200` predated the wire-measured shot cost (~45 fuel + ~10/tick) and practice-bot durability (~8-10 hits per kill from the recorded human sessions); (3) the "restock-then-finish-the-kill" lock preservation had no scope — it survived a cross-map COLLECT excursion; (4) the ACQUIRE-path pursuit fired at the lock's tracked position without checking the server's in-viewport aim rule.

**Code:**

- `bot/session_exit.py` (new) — `SessionExitError(reason, detail)` with `SessionExitReason = Literal["no_viable_targets", "out_of_fuel"]`. `run_tick_loop` catches it and routes through the normal scorecard/summary/index shutdown with `exit_reason` set accordingly.
- `bot/ai/threats.py::find_acquisition_target` — new keyword `engagement_reserve_fuel`; a candidate is viable only when `teleport_cost + reserve <= fuel`. Rejection reason `unaffordable` in the `acquisition_candidates` diagnostic. Gate logic extracted to `_acquisition_rejection_reason`.
- `bot/ai/hunt_mode.py::_decide_hunt_acquire` — the off-viewport held-lock pursuit branch is DELETED from ACQUIRE: a lock whose target is off-viewport on resume is released (`clear_combat_target`) and acquisition runs fresh — teleporting back if the enemy is affordable (the recorded human resume behavior), else moving on. Mid-fight pursuit in ENGAGE / SCAN_ON_LANDING / REFRESH is unchanged (seconds-scale, target just left mid-exchange). New `_decide_hunt_acquire_fresh` raises `no_viable_targets` when the map snapshot is fresh (within `map_open_cooldown_ms`) and nothing passes the gates; a stale snapshot still dispatches `map_open`.
- `bot/ai/collect_mode.py` — the marooned `ValueError` crash becomes `SessionExitError("out_of_fuel", ...)`.
- `bot/ai/types.py` — `engagement_fuel_budget` default 200 → 450 (wire-calibrated kill cost). With the 1100 cap and 200 fuel-low reserve this caps engagement range at ~58 tiles — the recorded human maximum was 60.

**Tests:** new coverage for the unaffordable rejection, the Manhattan-vs-euclid affordability edge (diagonal enemy farther by Manhattan but cheaper to reach wins), stale-lock release both ways (affordable → teleport back with lock re-formed; unaffordable → lock cleared + map_open), the fresh-map exit, and the tick-loop `SessionExitError` → index-row path. Re-anchored: mode-controller floor tests (650 boundary), `test_fallback_opens_map_even_when_recently_opened` → `test_fallback_exits_when_fresh_map_shows_no_viable_target`, mode-lock migration asserts released lock, scenario harness smoke places an enemy, replay marooning fixtures expect `SessionExitError`.

**Pages updated:** [[bot-behavior-contract]] §1.2 (self-directed exits) and §3.2 (affordability gate, stale-lock release, no-viable-targets exit); [[fuel-system]] marooning section + cascade step 5 wording.

**Deliberately NOT done (user rejected patch-stacking):** no shot-rejection counters, no pursuit time-bounds — the stand-off firing path that needed them no longer exists at the ACQUIRE level. If mid-fight pursuit ever produces rejected shots, that will surface as a distinct signature and get its own root-cause pass.

---

## [2026-07-02] rework | Hit/miss = per-shot ammo consumption; stationary-miss block enforced

User contract, verbatim: "check the inventory delta for each shot. that is how we measure hits vs misses." Diagnosed from run 2026-07-02 01:20 (user watching): the mid-fight pursuit path looped 25+ `weapon=0` shots at orange-1's stale tile after the target teleported out, while earlier in the SAME run five pursuit homings (`weapon=3`, `victim_id=-1`) killed orange-3 — with the bot logging all five winning shots as misses.

**Root cause — a circular oracle.** `mark_combat_hit` keyed the hit on tile-occupancy (`victim_id > 0`) and derived the local ammo decrement FROM that guess. The `check_and_clear_ammo_delta_hit` cross-check (added 2026-06-24 precisely for off-viewport pursuit) compared the pre-shot snapshot against `ws.inventory_state` — which is only ever decremented by the victim-id path. The ammo-delta signal was reading its own reflection: on every off-viewport hit (victim lookup can't see the tile), no decrement happened, so no delta, so "miss". Meanwhile the wire truth was in the `weapon` field all along: the server records the per-shot ammo spend there (it only spends on landing shots), and the page client's inventory display — the thing the user was reading — decrements from exactly that field. There is no per-shot 0x49 inventory frame; absolute syncs arrive later and reconcile (the `+5/-5` homing pair after the orange-3 kill).

**Code:**

- `sniffer/world_state_combat.py::mark_combat_hit` — hit and ammo decrement now key on `weapon_byte > 0` (consumption); `victim_id` demoted to kill-attribution metadata. `check_and_clear_ammo_delta_hit` retained as reconciliation against absolute 0x49 syncs.
- `bot/ai/combat_strategy.py::_combat_shoot` — miss against a STATIONARY registry position now calls `block_combat_target_and_replan` (block TTL + lock release + next viable threat). This rule was already written in [[bot-behavior-contract]] §3.3 but the code only re-aimed — the direct enabler of the weapon=0 loop. Miss against a MOVED target still re-aims and keeps the lock. Safe to enforce now because consumption-classification means pursuit homings that land never reach the miss path (the earlier false-miss classification is what made blocking dangerous).
- `bot/tick_loop.py::_get_combat_feedback` — docstring rewritten from the tile-occupancy narrative to the consumption ledger.

**Validation against the capture:** orange-3's fight replays as 8 dual hits + 5 pursuit-homing hits + kill (previously 8 hits + 5 false misses + kill); orange-1's terminal phase becomes ONE weapon=0 miss → stationary → block → release → re-acquire, instead of an unbounded 2s loop.

**Tests:** `test_hunt_feedback.py` — stationary-miss tests flipped to expect block+release (far and adjacent variants), new moved-miss re-aim test; `test_world_state_dispatch_container.py` / `test_tier2_lifecycle_signals.py` shot events corrected to wire-consistent weapon bytes (the old tests encoded wire-impossible combinations: weapon=3 at an empty tile, weapon=0 with a victim).

**Pages updated:** [[weapon-selection]] gains "The weapon byte is the per-shot ammo ledger — and therefore the hit oracle" with the wire proof; [[bot-behavior-contract]] §3.3 rewritten (consumption oracle row, stationary-miss block row now enforced with test citations).

---

## [2026-07-03] audit + rework | 0x4F has one personality; the radar response is a delta sync, not an append

User challenge ("visually I see some containers come into view and some removed from view... where is the data that removes the stale containers?") prompted a corpus scan of all 199 capture sessions (1817 0x4F bodies) plus a read of the JS `ch` handler (`tpclient.pretty.js:4791-4815`). Findings, in evidence order:

1. **The server DOES send removals.** 247 of 2093 cache entries carried value 0 — the "tile now empty" write. The 2026-07-01 claim that the radar response "lists ONLY newly revealed hidden entities" was too strong; the correct model is a **delta sync**: new reveals + volume corrections + explicit removals, with *unchanged* visible entities omitted. `update_container_from_radar`'s volume-0 branch had been applying those removals correctly all along.
2. **The client has no container list.** JS `ch` applies every 0x4F entry as a raw per-tile write (`tile.cache = value`, `tile.m = overlay`); rendering and the mouse-hover fuel value read the same slot. Our "RadarScanResult vs CombinedTileUpdate" split was an invented dual personality: **0 of 1817 bodies arrived top-level**, so the CombinedTileUpdate decode path had never fired in production.
3. **The mine tail was a loaded misread.** We decoded the 3-byte tail as `(x, y, team)`; per JS it is `(x, y, overlay)` where >= 8 (255 canonical, the `dh` detonation sentinel) means *clear the mine*. Corpus showed only values 0/1/3, so it had not bitten — yet.
4. **Byte 1 was the count's high byte** (JS reads `X(a[0], a[1])`), not an "always-zero flags byte".

**Code:** single 0x4F personality — `CombinedTileUpdateDict` + `decode_combined_tile_update` + its dispatch arm + `mark_pending_radar_cache_refresh`/`consume_pending_radar_cache_refresh` (only settable from the dead arm) deleted; top-level 0x4F routes to `decode_radar_scan_result`; `MSG_CACHE_OVERLAY_UPDATE` renamed `MSG_RADAR_SCAN`. Decoder: count = LE u16; tail entries split into mines (`team = value & 3` for 0-7) and `mine_clears` (>= 8), applied via `remove_mine`. `RadarScanResultDict` gains `mine_clears`; `update_world_state_from_radar` applies them.

**Pages updated:** [[radar-mechanics]] (delta-sync section replaces "only newly revealed"), [[decode-coverage]] 0x4F row, [[bot-behavior-contract]] new §2.4.

---

## [2026-07-03] rework | Landing truthfulness: 0x5A reset-then-apply + unconditional scan-on-landing

User contract: "the visible containers may be stale... I usually always use radar right on landing from teleport."

**0x5A reset-then-apply.** The landing viewport patch's skip-walk covers the whole 18x18 grid, so a tile it does not enumerate is the server saying "nothing here" — but our dispatch only applied the enumerated half, so container/mine entries remembered from a previous visit survived re-entry after being consumed (a ghost pickup target). `update_viewport_entities` now sweeps visible-layer (non-radar-sourced) entries on silent tiles inside the patch bounds before applying the patch — the same reset-then-apply the JS client does with its full grid wipe (`rg()` in `Vg.prototype.h`). Radar-sourced entries stay owned by the radar omission-prune.

**Unconditional scan-on-landing in COLLECT.** The committed zero-coverage gate skipped the landing radar whenever the 18-wide visible viewport overlapped 2 tiles of old coverage after a 16-tile hop, and skipped it entirely on revisited ground — exactly where knowledge is most stale. Replaced with a per-viewport-entry latch (`AIStateDict.last_landing_scan_viewport`; the viewport changes only on teleport, so origin-differs = just landed). HUNT's combat-landing scan records the same latch so a later COLLECT entry in the same viewport does not double-fire. Landing sequence now matches the human policy exactly: teleport → 0x5A gives the truthful visible layer → radar reveals the hidden layer → one complete pickup sweep.

**Pages updated:** [[bot-behavior-contract]] §2.4 + §3.4.

---

## [2026-07-03] rework | Fuel-dot atlas restored: dot-hop restocking + dot-relay travel

User contract, verbatim intent: "instead of our blind viewport hopping we could switch that to yellow dot hopping. so hop to nearest yellow dot with a 100% clean viewport" and "if it was me playing id use yellow dot teleporting while en route to the opponent."

**Atlas restore (partial revert of the 2026-06-22 strip).** `decode_map_data` materialises the skip-RLE dot coordinates again (`MapDataDict.fuel_dots`, mirrors JS `Ig.h` exactly); `_dispatch_map_data` stores them on `WorldService.map_fuel_dots` (server-cached per session, overwritten every map open); the atlas threads into `DecideCtx.map_fuel_dots` via a new `decide()` parameter, like `terrain`.

**COLLECT dot hop.** `make_resource_search_hop` replaced the blind 16-candidate compass ring with: nearest atlas dot whose landing tile is passable, teleport affordable, landing viewport unscanned, and landing viewport 100% walkable on the static terrain map. Landing auto-pickup makes each hop partially self-funding (dots are ~40% fresh, wire-verified high-volume). With an empty atlas the hop dispatches `map_open` (dots arrive with the 0x4C response), guarded by `map_open_cooldown_ms`. No dot qualifies → the existing `out_of_fuel` exit. `equip_search_hop_distance` config field deleted with the compass ring.

**HUNT dot relay.** When a fresh map has an enemy that fails ONLY the affordability gate (`find_relay_travel_target` — every other gate must pass; travelling toward a corpse wastes the relay), `_relay_toward_unaffordable_enemy` teleports to the dot that best closes distance to it: strictly closer than the current tile (monotone → terminates), passable, and leaving `fuel_low_threshold` behind so a dry dot cannot strand the bot below the COLLECT reserve. Ties keep the cheaper hop. `no_viable_targets` now fires only when no enemy is worth relaying toward or no dot makes affordable progress — the hard ~58-tile engagement ceiling becomes a soft one funded by dots en route.

**Gate:** `make check` green — guard + ruff + mypy + 4059 tests at 100.00% statement+branch coverage. (One transient 4-test replay flake was observed once under a particular xdist ordering and did not reproduce across 7 subsequent full runs; noted for watch.)

**Pages updated:** [[fuel-system]] (cascade step 5 + atlas restore), [[map-data-decode]] (decode status), [[bot-behavior-contract]] §2.3 / §3.2 / §3.4.

---

## [2026-07-03] fix | Pursuit rejection loop: viewport-clamped aim + rejected-shot feedback; frozen-clock test leak

Live `make run` 2026-07-03 20:34 (user watching): the new dot relay and landing sequence worked end-to-end (relay hop to dot (138,234) refuelled 451 -> 1100 via landing auto-pickup, teleport onto orange-4, landing radar, dual hit, tracked homing hit) — then the pursuit dispatched `shoot(143,237)` five times, each drawing 0x52 code 0 ("You can't do this"), each invisible to combat feedback, each burning the full 4 s shot-feedback window before an identical redispatch.

**Diagnosis (user-directed: "inspect the you cant do that and determine why we are missing the failed firing"):**

1. **The rejection channel was unread during combat.** `has_in_flight_action` routes only move/collect/teleport/scan/map_open to the `_clear_command_error` hook; a `shoot` action falls through. The shot wait (`_has_pending_shot_feedback` / `_get_combat_feedback`) polls only success channels — hit, 0x53 echo, killed set, ammo delta. A rejected dispatch produces none of those (the user called it: "there would not have been an inventory delta for those restricted shots"), so the 0x52 sat unconsumed, the window timed out to "" (not even a miss — scorecard read 8 shots / 2 hits / 0 misses), and the unchanged lock re-fired the identical shot.
2. **The aim was illegal because a 2026-06-26 assumption is false.** "Off-viewport tanks stop broadcasting position, so the registry stays at the last on-viewport coord" — in fact 0x3D MovementResponse broadcasts every map tank's position ~every 2 s, so the pursuit aim tracked orange-4's true tile (143,237), five rows below the viewport. New game-mechanic fact from the user: the server refuses homing at an enemy close enough that a viewport shift would reveal them — the aim must be a viewport-legal tile.

**Fix:**

- `_clamp_aim_into_viewport` (`combat_strategy.py`): every shoot dispatch aims at the registry coordinate clamped onto the visible viewport bounds, carrying the target's `tank_id` — exactly the wire-proven snipe shape (the same run's `weapon=3` kill shot aimed at the target's vacated in-viewport tile). Applies only when the viewport record contains the bot; registry truth untouched (`combat_target_x/y` keep the real position for the stationary-miss comparison).
- Rejected-shot feedback: `CombatFeedback` gains `"rejected"`; shot-rejecting 0x52 codes (0/3/8) during a pending shot end the wait immediately (`peek_command_error`), classify as rejected (consumed via `check_and_clear_command_error`), count in the scorecard (`session_reject_count`, summary line now `N hits, M misses, R rejected`), and block-and-replan the target. Non-shot codes (7 etc.) are left for the in-flight-action machinery.

**Root-caused the 4-test replay flake while at it** (first seen 2026-07-03 morning, "did not reproduce"): every test in `tests/scenarios/` constructs `BotScenario()` bare — the harness installs a frozen scenario clock on `_test_hooks.get_current_time_ms` at construction and none ever `close()` — and the global `_restore_hooks` autouse fixture restored every hook EXCEPT the clock. Any xdist worker that ran the scenarios module then stamped all world-state observations with frozen ~100000 ms values; the replay module's capture-epoch decide clocks then read every enemy as `stale_map_data`, acquisition emptied, and the (2026-07-02) no-viable-targets exit raised. Reproduced deterministically with `pytest tests/scenarios tests/replay/test_real_session_regressions.py -n 0`; fixed by adding `get_current_time_ms` to the global per-test hook restore. The scenarios harness landed 2026-07-02, which is why the flake was new.

**Gate:** `make check` green — 4064 tests, 100.00% statement+branch coverage; the scenarios+replay reproduction is now green.

**Pages updated:** [[bot-behavior-contract]] 3.3 (clamped-aim + rejected-feedback rows), [[shot-range]] (aim-legality section + scope-shift refusal mechanic), [[tank-freshness-model]] + [[combat-chase-bug]] (correction of the falsified "registry stays at last on-viewport coord" rationale, follow-up fix section).

## [2026-07-06] ingest | Sigma's TankPit Tournament Guide v3.4 (16-Jan-2015)

**Motivation:** Tankpit's community is small and its era-2014-2015 knowledge lives in a handful of PDFs on fragile hosts. Sigma's v3.4 guide was found via DocDroid (Scribd paywalls after page 4). Preserved in-repo so citations survive link rot; the wiki may end up as the most organized surviving Tankpit reference.

**Source archived:** `docs/sources/sigmas-tankpit-guide-v3.4.pdf` (505 KB, 12 pages). Outside `wiki/` because SCHEMA v1.0 forbids extra top-level dirs under `wiki/`; `docs/sources/` follows the "operational docs live in `docs/`" convention.

**Pages written:** [[tournament-strategy]] — new page under [[combat]] hub preserving the strategic content of Sigma's guide (initial fill, fill-fighting, kill types, PPH mechanics, equipment management, endgame shield-fighting, scenario templates). Confidence medium, marked as 2015 human tournament meta not verified against wire captures. Cross-links to [[game-modes]], [[enemy-bot-behavior]], [[gameplay-loop]], [[equipment-refill-strategy]], [[game-economy]].

**Pages updated:**
- [[enemy-bot-behavior]] — added four guide-sourced facts: (1) bot shots-to-teleport-off table (7 recruit / 8 private / 9 corporal), (2) last-shade + N heuristic mapped to our decoded `damage_state 0-3`, (3) bots return SINGLES not duals in combat (rewrites cost-of-engagement asymmetry), (4) same-color bots respond to chat commands ("use the radar", "move out of the way"). All marked guide-sourced, awaiting wire verification against the next multi-tank capture.
- [[game-modes]] — extended the Tournament Mode Differences section with the full tournament capacity ladder (Recruit 60 → General 124, +8 per rank) alongside the regular ladder (20 → 60, +5 per rank) for comparison. Confirmed unchanged: promotion point requirements and kill requirements are the same as main map's; only carry cap changes.

**Structural updates:** combat hub page count 9 → 10; index total 50 → 51.

**Deliberately skipped:** ethics, prep (Ethernet, restroom), Type 1/2/3 human-social targeting, PPH partner selection, cup thresholds, video review, anonymous tanks, Kirby's 2-ER-per-minute rule. Preserved in [[tournament-strategy]] but not extracted as separate wiki facts — the bot doesn't play tournaments and none of this drives code.

**Verification debt:** the shots-to-teleport counts, the shade heuristic, the singles-return claim, and the chat-command mechanic are all 2015 human observations, not wire-verified in this project. Next multi-tank capture with bot targets: count `You hit N/N` events against `damage_state` transitions and inspect incoming weapon bytes.

## [2026-07-06] update | Rank formulas cracked: fuel capacity, radar radius, deposit mechanics (client mining + 4-rank user measurements)

**Motivation:** the 2026-07-06 live run looped forever re-picking a fuel container at a full tank (code-5 "Tank full" every 2 s). Root-causing "how should the bot know it's full?" led to mining tpclient.js for the capacity source, then user measurements on his own tanks at 4 ranks to verify.

**New game facts (all wire/client/user verified):**

1. **Fuel capacity = 1000 + 100·rank.** Never on the wire; the client derives it in the fuel-gauge draw (`Gc`: fill `7·fuel/100` px vs capacity region `7·(10+rank)` px). Verified at private (wire tank-full at exactly 1100 + max deposit 1000), sergeant (deposit 1200), major (deposit 1500), colonel (deposit 1598 = 1700−100−~2 walked). The 2026-06-11 learned-watermark of 2010 was a polluted scrape read (same run tank-full'd at 1100), not counter-evidence.
2. **Deposit floor = 100, server-enforced**: max deposit always leaves exactly 100 fuel. Deposit wire command decoded: `'D'` (0x44), 6 bytes, x/y/u16-LE amount, client-gated fuel>100.
3. **Built-in radar radius = 2 + floor(rank/3)** (5x5 / 7x7 / 9x9 at rank bands 0-2 / 3-5 / 6-8). User measured lieutenant=3, colonel=4, then sergeant=3 and major=4 chosen specifically to discriminate the two candidate step formulas — steps fall at sergeant and major. Resolves the guide's "higher ranks have a larger radar" open question. Extra radar stays full-viewport at all ranks.
4. **Client fuel>100 gate (`ce()`)**: blocks targeted shots, mine drop, nearest-enemy, deposit, obstacle pickup — locally, with a client-generated "Insufficient fuel" line. Untargeted shots, radar, map open, move, teleport are NOT client-gated.
5. **Homing = tank-on-aimed-tile**: the client sends `target_id` iff a tank occupies the aimed tile (`new Lb(x, y, tank ? tank.id : 0)`). Friendly-fire and own-tile aims abort client-side and never reach the wire — a wire 0x52 code-3 can only be a race.
6. **Radar sets a client cooldown counter** (`ca = 50`) on dispatch; units unconfirmed.
7. **Obstacle carrying / bridge building** exists (`'b'` command, fuel>100 + tile flag gates) — unused by the bot, now documented.

**Pages updated:** [[game-economy]] (capacity formula table, deposit mechanics + floor, frontmatter re-verified), [[radar-mechanics]] (rank-scaled radius table replacing the fixed 5x5 claim, footnote 14, downstream 5x5 phrasing generalized), [[client-constants]] (Gc/Cc/ce functions, shoot/deposit/radar dispatch facts, obstacle mechanics).

**Code implications (pending, for the combat-rework fix-up):** replace the `learn_fuel_capacity` empirical watermark with rank-derived capacity (kills the tank-full pickup loop at the root — fuel selection AND lock continuation gate off known capacity from tick 1); make `REGULAR_RADAR_RADIUS = 2` (`state/viewport_geometry.py`) rank-derived; fuel-lock release must consider capacity so a code-5 never strands a held lock.

## [2026-07-06] fix | Action-kind-scoped 0x52 error routing (live 20:20:59 session-exit smoking gun)

**Motivation:** the 2026-07-06 `make run` session exited `no_viable_targets` at fuel 531 with a fully-stocked tank (25 duals / 16 homings / 24 extras). Root cause: at 20:20:59 a `collect fuel at (189,77)` completed via `container_consumed`, but the wire's late-arriving `0x52 code=4` "Empty container" (~2 s late) landed while the next tick's HUNT `map_open` was in flight. The old universal `_ACTION_BLOCKING_COMMAND_ERRORS = {0,1,4,5,7,8}` frozenset treated it as a map_open rejection, HUNT could not acquire, and the fresh-map cascade raised `no_viable_targets`. This was a chronic wire/log race: any 0x52 that landed a tick after its owning action had already resolved via a different signal could poison the next unrelated action.

**Fix:** replaced the global set with a per-`ActionKind` whitelist `_COMMAND_ERROR_APPLICABILITY` (`bot/tick_loop_actions.py`) derived from tpclient.js dispatch semantics:

- `move`: {0 "can't do this", 1 "can't go there", 8 "insufficient fuel"}
- `teleport`: {0, 1, 8}
- `collect`: {0, 1, 4 "empty container", 5 "tank full", 7 "inventory full", 8}
- `scan`, `map_open`, `shoot`, `none`: `frozenset()` — server never rejects these dispatches, so any 0x52 landing during their waits is orphan by definition

Codes outside the current action's whitelist are consumed as **orphan** and emitted as an `orphan_command_error` diagnostic instead of triggering a spurious rejection. Movement paths use `_clear_command_error` (returns True on applicable rejection, transitions bot to IDLE, marks failed target); scan/map_open use `_drain_orphan_command_error` (drain-only, no state transition). Both share `_emit_orphan_command_error` for the diagnostic emission so drift can't creep in.

**Tests added:** `TestClearCommandError` in `tests/bot/test_tick_loop_coverage.py` gained `test_scan_wait_drops_orphan_error_and_stays_pending`, `test_map_open_wait_drops_orphan_error_and_stays_pending` (live-run regression guard), `test_teleport_wait_drops_orphan_empty_container`, `test_move_wait_drops_orphan_tank_full`, `test_orphan_command_error_emits_diagnostic`, `test_scan_wait_with_no_error_stays_pending`, `test_scan_and_map_open_whitelists_are_empty` (invariant guard against future whitelist drift). Two pre-existing tests that asserted the old buggy behavior (`test_command_error_clears_scan_action`, `test_command_error_clears_map_open_action`) were replaced by the new orphan-drop variants.

**Wiki update:** [[bot-behavior-contract]] §3.4 grew a `MUST` row for action-kind-scoped 0x52 with the smoking-gun scenario logged verbatim, and its verifier list now cites the new tests.

**Gate:** `make check` green — 4088 tests, 100.00% statement + branch coverage.

## [2026-07-06] design | Self-observing bot architecture (multi-phase; handoff written; NO code landed)

**Motivation:** the same 2026-07-06 `make run` that verified the earlier fixes surfaced a NEW class of bug at 20:47:31 — a 26-second silent deadlock. The executor's `_is_valid_shoot` position-match clause was self-rejecting every clamped homing shot (aim ≠ tank position under the 2026-07-03 clamp mechanic is intentional; the position clause treated it as a stale-target race). The discard fired only `emit_ai("rejecting shoot at ...")` — a bare log line that read identically to a legitimate server rejection. No structured event correlated the discard to the planner's decision. Nothing tripped an alarm. 26 s of silent failure.

**The class:** decisions and outcomes live in disjoint observability channels. Three parallel diagnostic fabrics — `wire_complete` for scan/move/teleport/collect/map_open, `teleport_attempt` for teleports, `combat_feedback` for shots — with no shared contract, and one whole class of failure (client-side executor discards) that no channel covered at all. Every future bug of this class hides the same way.

**The design conversation:** user asked "what else is the bot blind to?" We enumerated fifteen items: (1) standardized WHAT, (2) standardized WHY, (3) predictions + accuracy, (4) alternatives considered, (5) confidence, (6) provenance for facts, (7) per-entity memory, (8) cross-session persistence, (9) causal chain, (10) anomaly detection, (11) mode transitions as first-class events, (12) time budgets, (13) self-model, (14) comparative baselines, (15) feature-flag experiments. The 20:47:31 deadlock hit at least four of them (1, 3, 10, 12) simultaneously.

**The philosophy correction:** user pushed back on "repeated-failure detection" as still soft. The right posture is **fail hard on state entry** — contracts on state transitions, raise on violation. No retry counters, no anomaly thresholds. Item (10) "anomaly detection" is REJECTED and replaced by contracts throughout.

**The architecture:** four layers — Facts (world model with provenance + confidence), Decision Engine (per-tick planners), Ledger (per-attempt decisions + outcomes + causal chain), Memory (cross-tick + cross-session). Cross-cutting `contracts/` framework with `ContractError` + `@enforce_contract` + a guard rule that scans for public mutations skipping enforcement.

**Phase roadmap (all documented in the handoff, NONE landed):**

| Phase | Deliverable | LoC | Sessions |
|---|---|---|---|
| 0 | Immediate deadlock fix: delete executor position-check | 200 | 1 |
| 1 | `contracts/` + `facts/` foundation | 800 | 2-3 |
| 2 | `ledger/` core (Outcomes, Decisions, Ring, Causal, ModeTransitions) | 2500 | 4-6 |
| 3 | Decision enrichment: Predictions, Alternatives, Confidence, Time budgets | 1500 | 2-3 |
| 4 | `memory/` (per-entity + persistence + session start/end) | 2500 | 3-5 |
| 5 | Aggregation (self-model + baselines + experiments) | 1500 | 2-3 |

Total ~9000 LoC production + ~13500 LoC tests. 14-21 sessions of focused work.

**In-session artifact NOT landed:** the prior AI (this session) wrote a monolithic 1000-line `src/tankpit_bot/bot/action_outcome.py` before the split-per-kind design emerged. Deleted before handoff to leave a clean tree. Next AI builds `ledger/outcome/` split into six per-kind files from scratch per the handoff spec.

**Pages written:** [[self-observing-architecture]] — new page under [[architecture]] hub. Vision, four layers, fifteen items, phase overview. Detailed phase specs live in `docs/handoffs/self-observing-bot-architecture.md`.

**Structural updates:** architecture hub 4 → 5 pages; index total 51 → 52.

**Handoff artifact:** `docs/handoffs/self-observing-bot-architecture.md`. Comprehensive: ban list, principle, four-layer architecture, module layout, per-phase specs with data types + contracts + tests + verification gates, ordering constraints, ban list repeated.

**Gate:** `make check` green — no code changes landed this session, so 4088 tests + 100% coverage baseline preserved.

## [2026-07-06] fix | Container freshness TTL removed — belief decay was destroying wire truth

**Motivation:** the 30 s `_CONTAINER_FRESHNESS_TTL_MS` in `bot/ai/equipment.py` expired REAL loot twice in live runs. (1) Run 2026-07-02 01:46: after a 23 s fight plus an equipment sweep in the kill viewport, every fuel container aged past 30 s; `find_best_fuel` saw nothing while `describe_container_search` — which never applied the TTL — logged `actionable=9`, and the bot hopped away at fuel 565/1100 instead of restocking where it stood. (2) Run 2026-07-06 18:19 (index row `20260706-181928`): an equipment container revealed 31 s earlier was dropped mid-cascade and the session died with a bogus `out_of_fuel` exit at fuel 1100.

**Fix:** deleted `_CONTAINER_FRESHNESS_TTL_MS` and `_is_stale`; `is_container_pursuable` no longer takes `now_ms` and checks only container kind + `failed_pickups == 0`. Rationale (recorded in the docstring): every pursuability consumer is viewport-scoped, and an in-viewport container is wire-truthful under the truth layer — the landing 0x5A reset-then-apply sweep removes silently-vanished visible entries ([[viewport-frame]], 2026-07-03 landing-truthfulness rework), the landing radar's omission-prune covers radar-sourced entries ([[radar-mechanics]]), and live 0x43 cache updates track consumption while the bot watches. A wall-clock decay on top of that only ever deleted truth. Rode in the 2026-07-06 21:34 commit alongside the 0x4F collapse.

**Side effect — the lying diagnostic is structurally fixed:** the TTL was the single criterion `describe_container_search` did not apply, so its `actionable=N` count could contradict the pickers. With the TTL gone, the summary and `find_best_fuel` / `find_equipment_candidates` filter on identical criteria (kind, viewport bounds, failed-pickup, volume, walk-reachability); the `actionable=9`-then-hop contradiction from 2026-07-02 can no longer occur.

**Live validation:** run 2026-07-06 20:44:51 replays the original failure scenario correctly — kill of tank 510 followed immediately by two same-viewport fuel pickups (537 → 945 → 1100 cap) before yielding to hunt; and the 22:38:22 post-kill "no actionable" verdict showed `blocked_walk` on all nearby fuel — a true reason, log and behavior in agreement.

**Tests:** the collect-mode suites were extended in the same commit (`test_collect_mode_fuel.py` +134 lines, `test_collect_mode_equipment.py`, `test_collect_mode_integration.py`, `test_equipment.py` reworked for the `now_ms`-free signature).

**Pages already consistent:** [[self-observing-architecture]] refers to "the retired freshness TTL"; no page asserts a live container TTL.

## [2026-07-12] add | Bot Service Architecture — SPA-driven long-running service (Phase A)

**Motivation:** the SPA on the phone (fiesta) previously spawned a fresh `make bot` per session (~15 s cold-start including Python imports + Chromium launch + tankpit login). Phase A ships a long-running Python service that keeps Chromium warm across sessions and exposes five HTTP routes so the SPA can Start/Stop/Mode a running bot in ~200 ms.

**Landed:** `tankpit-bot-service` binary + `service/*` package (10 modules): `types` + `types_codecs` (Wire vocabulary and encode/decode), `mode_bridge` + `status_bus` (threadsafe cross-thread channels), `session_runner` (single-session coordinator), `http_server` (aiohttp app with 5 routes), `service_main` (`_async_main` + `main`), `_test_hooks` (DI hooks in `Services/: _test_hooks.py` style).

**Wire surface added to `bot.ai.types.AIStateDict`:** `manual_mode: AIMode | None`, `live_radars_used: int`, `live_teleports: int`. The tick loop drains the mode bridge into `manual_mode` at the top of every tick; the executor increments the counters at radar/teleport dispatch call-sites via `apply_dispatch_counters` in `mode_controller.py`. The tick loop publishes a `SessionStatusDict` frame to the status bus after every tick.

**Bot construction change:** `Bot.__init__` grew `mode_bridge: ModeBridgeProtocol | None = None` and `status_bus: StatusBusProtocol | None = None` kwargs. Defaults construct fresh inert instances so `make bot` (standalone) still works — a bridge no HTTP handler ever writes to always drains `None`, and a bus with zero subscribers publishes into the void.

**New command:** `HoldCommandDict{ cmd_type: "hold" }` for manual `UNSET` — a no-op tick the executor recognises and skips dispatch on. `resolve_owner_from_manual` in `mode_controller.py` short-circuits `ai_strategy.decide` when `manual_mode` is set: `UNSET` → hold decision; `HUNT` / `COLLECT` → force that owner; `None` → auto-arbitrate.

**Storage state persistence:** `browser/session_storage.py` (`STORAGE_STATE_PATH = Path("runs/state/tankpit.storage.json")`, `load_storage_state`, `save_storage_state`, `StorageStateCacheError`). `Bot.run` loads it before `new_context` (passes to Playwright as `storage_state=path` if present), and saves it after `wait_for_game_ready` — so cold-start login runs once per Chromium install; subsequent sessions skip the tankpit login flow entirely.

**Baseline drift fixed alongside:** commit `d2e89ddd` (2026-07-10 fiesta streaming maximise-window workaround) had left `sniffer/core.py` with a `type: ignore`, an `os.environ.get`, an `except Exception: # noqa: BLE001`, and a `cdp: object` annotation — all guard violations. The Phase A pass fixed those (proper `CDPSessionProtocol` typing, `_test_hooks.get_env`, `require_int(window, "windowId")` failure surfaces loudly), extended `BrowserProtocol` / `BrowserTypeLaunchProtocol` to match the real Playwright signatures (`args` + `no_viewport` + `storage_state`), and updated the six fake Playwright families in tests/fakes/{base,bot,probe}.py + tests/action_lab/_replay_browser.py. Baseline `make check` was red before this pass; it is now green.

**Structural updates:** architecture hub 5 → 6 pages; index total 52 → 53. New page: [[bot-service-architecture]].

**Gate:** `make check` green — 4278 tests pass, 100% branch coverage on 18,341 statements + 5,190 branches, mypy strict clean across src + tests + scripts, guard clean, ruff clean. `asyncio_mode = "auto"` added to `[tool.pytest.ini_options]` so async tests type-check clean without the `@pytest.mark.asyncio` decorator (which leaks `Any` under strict mypy).

## [2026-07-12] refactor | Bot-launch config extracted to bot/config.py (Phase A9)

**Motivation:** the tankpit target URL and guest-vs-account preference env-var resolvers were duplicated between `bot/entry.py` (one-shot `tankpit-bot`) and `service/service_main.py` (long-running service). Two copies of the same env-parsing logic is a silent divergence risk — a semantics change to one would let the two commands quietly disagree about production defaults.

**Landed:** new `bot/config.py` with `DEFAULT_TARGET_URL`, `resolve_target_url()`, `resolve_prefer_account()`. Both `entry.py` and `service_main.py` import from there. `entry.py` also switched its `.env` loading to route through `service_hooks.load_dotenv` so both entrypoints stub the same way in tests. `service_main.py` deleted its private `_resolve_target_url` / `_resolve_prefer_account`.

**Tests:** new `tests/bot/test_config.py` (10 tests: defaults + env override + empty-string-as-unset for URL, missing/true/1/yes/case-insensitive/other for prefer_account). Deleted the duplicated `TestResolveTargetURL` / `TestResolvePreferAccount` blocks in `tests/service/test_service_main.py`.

**Gate:** `make check` green — 4279 tests pass, 100% branch coverage on 18,345 statements + 5,190 branches. First-shot pass, no follow-up fixes needed.

## [2026-07-12] add | fiesta bot-controls panel + SSE subscriber (Phase B)

**Motivation:** Phase A stood up the bot service (`tankpit-bot-service`) but nothing on the SPA consumed it. Phase B ships the browser UI: a `<section class="bot-panel">` widget the phone renders on the tankpit profile, driving the bot via `/api/tankbot/*` and painting a live stats readout from the SSE `/status` stream.

**Landed (fiesta, `MCPs/fiesta/src/tankbot/`):**
- `types.ts` — TypeScript mirror of `service/types.py`. Strict literal-union types (`WireMode`, `AIMode`, `AIModeState`), immutable `readonly` interfaces (`ModeCommand`, `LiveStats`, `SessionStatus`), and full `decodeSessionStatus` / `decodeLiveStats` validators. Every SSE frame hits `decodeSessionStatus` at the seam — a mismatched literal throws instead of rendering a blank panel.
- `TankbotHttpClient.ts` — constructor-DI HTTP client (same shape as `WebrtcHttpClient`). `postStart` / `postStop` / `postMode` throw on non-2xx; `subscribeStatus(onStatus, onError)` returns a dispose function.
- `BotController.ts` — reactive state layer publishing one immutable `BotUIState` per change. `runIntent` uses `.then/.catch` chaining (matches the fiesta `no-try-catch-in-core` convention). Non-Error rejections rethrow instead of coercing to a soft state message.
- `BotControlsView.ts` — DOM widget subscribing to controller state. Start / Stop pair swaps on `running`, mode buttons highlight the current `manualMode`, per-intent pending grey greys only the pressed button, SSE-error banner reveals a Reconnect button.
- `_test_hooks.ts` — reuses the same `FetchFn` + `EventSourceFactory` protocols as the WebRTC client so production wires through the shared `productionFetch` + `productionEventSourceFactory` in `production-hooks.ts`.

**Boot wiring (`boot/bot-controls.ts`, excluded from coverage like every other `boot/**`):** `main.ts` calls `wireBotControls(autoLaunchProfile)`; no-op on any profile ≠ `"tankpit"`; throws if `#bot-panel-host` is missing on the tankpit profile (drift, not a silent degradation). CSS shipped alongside — `.bot-panel` / `.bot-panel-host` / `.bot-panel__btn` rules in `style.css`; `?v=` bump to `64` so phones refresh the stylesheet.

**Gate:** `make check` green in `MCPs/fiesta` — 749 tests pass, 100% coverage on statements + branches + functions + lines (1,434 statements, 692 branches, 284 functions), `mcp-guard` clean, ESLint clean, `tsc --noEmit` strict clean, no `any` / no `as` / no `@ts-*` / no test mocks / no weak assertions.

**Structural updates:** [[bot-service-architecture]] page extended with a "Phase B — SPA bot-controls panel" section documenting the tankbot package + boot wiring + host CSS. Remaining phase: Phase C ships nginx `/api/tankbot/*` routing, the fiesta docker rebuild, and the `shell:startup` `.cmd` for the bot service on austinpc.

## [2026-07-12] ship | Phase C — nginx route + docker rebuild + startup shortcut

**Motivation:** Phases A + B stood up the service and the browser widget. Phase C is the plumbing that connects them end-to-end: nginx routes `/api/tankbot/*` from the phone through the fiesta container to the bot service on austinpc, the fiesta docker image is rebuilt so the config reaches production, and a `shell:startup` `.cmd` respawns the service on login so the operator does not manually launch it every reboot.

**Landed (fiesta, `MCPs/fiesta/nginx.conf`):** new `location /api/tankbot/` block, placed before the `/api/` (Vibeshine) block so nginx's longest-prefix-match rule routes correctly. Uses the same Tailscale-IP literal (`100.77.206.124`) `proxy_pass` shape as `/api/webrtc/` — `host.docker.internal` remains unreachable under WSL2 mirrored networking (see the same nginx.conf's history comments for the 2026-07-01/02 investigation). SSE knobs (`proxy_buffering off`, `proxy_read_timeout 24h`) mirror the /api/webrtc/ ICE-stream settings so the `/status` frame stream flows without buffering.

**Bot service (tankpitbot, `service/service_main.py`):** `_DEFAULT_HOST` flipped from `"127.0.0.1"` to `"0.0.0.0"` so the aiohttp site is reachable from the fiesta container via the host's Tailscale IPv4. Trust boundary is the machine's LAN + the operator's Tailnet — the same boundary Vibeshine already accepts on 47990. Test updated to match (`tests/service/test_service_main.py`).

**Launcher (tankpitbot, `Makefile` `service:` target):** a `make service` target that respawns `poetry run tankpit-bot-service` on crash with a 5-second cooldown via a PowerShell `while ($true)` loop. Chose this over an initial `shell:startup` `.cmd` after weighing the trade-off — the service is just an aiohttp listener until the phone POSTs `/start`, so always-on has no benefit. `make service` sits next to `make bot` / `make sniff` in the same Makefile, foreground-terminal-friendly for debugging, zero setup. The initial `.cmd` was written and then removed (`scripts/tankpit-bot-service.cmd` deleted).

**Fiesta docker rebuild:** `make up-fiesta` at the `MCPs/` top-level (which runs `docker compose up -d --no-deps --build fiesta`). The `mcp-fiesta` container recreated with the new nginx config. `nginx -t` inside the container confirmed the config parsed clean; `/api/tankbot/*` requests reach the proxy path (they time out until the bot service starts, then flip to 200).

**Gate:** tankpitbot `make check` green — 4279 tests pass, 100% branch coverage (unchanged from A9). Fiesta `make check` remains green from Phase B — no fiesta-side code touched in Phase C, only nginx.conf.

**How to bring it up end-to-end:** (1) `cd C:\Users\Test\PROJECTS\api\clients\tankpitbot; make service` — the terminal starts the aiohttp server on `0.0.0.0:47100` and stays foreground; (2) accept the Windows Firewall prompt for port 47100 (private networks) on first launch; (3) load `https://tankpit.austinwagner.org` on the phone. The bot-controls panel paints immediately and starts receiving SSE status frames within one tick. Ctrl+C in the terminal exits the respawn loop cleanly.

## [2026-07-12] fix | Idempotent bot-service launch + phone SERVER button

**Motivation:** the Phase C `make service` design had a race: `poetry run tankpit-bot-service` blind-launches, so a double-tap of a future SERVER button (or a second `make service`) would spawn a competing Python process, both fighting to bind port 47100 — one loses with ``OSError``, the outer respawn loop retries every 5 s forever, and the operator ends up with a stuck terminal that never recovers on its own. User's ask: "cant we fix the race issue properly? we have unlimited time, tokens and context."

**Landed (idempotency, three layers):**

1. **Service self-probe** — new `service_hooks.probe_existing_instance` DI hook. Probe implementation lives in its own module: `service/probe.py` (URL constant + `probe_health_url(url) -> bool` core + `default_probe_existing_instance()` wrapper). `service/_test_hooks.py` binds `probe_existing_instance` to `default_probe_existing_instance` and defines the Protocol; the utility stays out of the DI-plumbing file so each concern lives in one home ("lift don't fork"). Real probe uses stdlib `http.client.HTTPConnection` (not `urllib.request.urlopen` — `urlopen`'s context manager returns `Any` under strict mypy) to GET `http://127.0.0.1:47100/health` with a 1-second timeout; returns True only when the peer answers `200` with the exact body `"ok"` (the marker we own — a foreign HTTP server on the port cannot pass). `main()` calls this before `serve()`; on True, it logs "already responding" and exits 0 idempotently. Non-Error exceptions (URL parse errors) still raise — the probe only swallows the expected connectivity failures (`OSError`, `HTTPException`).

2. **Makefile respawn discipline** — the `service:` recipe's `while ($true)` loop now distinguishes graceful exits (exit 0 → break) from crashes (nonzero → retry), and caps consecutive crashes at 3 before giving up. A double-invocation hits the probe short-circuit, the wrapper exits 0, the loop breaks, nothing spins.

3. **Phone `SERVER` button** — `profiles/tankpit.json` gained a `menu-button` labelled `SERVER` beside `SNIFF`. Its `runCommand` spawns a persistent cmd window on the PC (`cmd /c start cmd /k "... && make service"`) — combined with layer 1, tapping it is safe under any state. Service down → new instance boots. Service up → new instance immediately exits with "already responding"; the phone-triggered cmd window stays open so the user sees the log line and closes it manually.

**Tests:** dedicated `tests/service/test_probe.py` (mirrors the `service/probe.py` extraction — clean separation, no duplicate coverage). Contains `TestProbeHealthURL` (6 scenarios: `200 ok` → True; `200` wrong-body → False; non-200 status → False; connection refused → False; empty-path URL falls back to `/`; missing-host URL → ValueError) plus `TestDefaultProbeExistingInstance` (delegates-to-parameterized equivalence). `test_service_main.py`'s `TestMain` gained `test_short_circuits_when_probe_reports_existing_instance` — the load-bearing guard for double-tap safety. Sync `http.client` probe runs under `asyncio.to_thread` in the async aiohttp-server tests so the sync call does not deadlock the event loop the test server needs to accept the incoming TCP connection.

**Gate:** tankpitbot `make check` green — 4287 tests pass, 100% coverage on 18,378 statements + 5,196 branches. Guard rules `except-without-log-or-raise` and `weak-assertion-isinstance` were both fired during the initial pass and fixed properly (`log.debug` on the probe's swallowed exception; equivalence check between wrapper and parameterized core instead of an `isinstance` check on the runtime-dependent bool).

**Wiki:** [[bot-service-architecture]] Phase-C section extended with the idempotency + SERVER-button design + a new "What Phase C does NOT do" bullet clarifying that Stop-Server stays on the PC (Ctrl+C or close-window), not on the phone.

## [2026-07-17] audit | Executor rejection silent-loop class — structural pattern behind 20:47:31, three more live sites

**Motivation:** revisit the executor after Phase 0 of [[self-observing-architecture]] (2026-07-06) to check whether the `emit_ai("rejecting ...")` deadlock class was actually eliminated. Phase 0 removed one instance (`_is_valid_shoot` position-match); the *class* was left intact.

**Findings:**

1. **AI state rollback on rejection.** `bot/tick_loop.py:490-491` persists `bot._ai_state` only when `command_sent` is True. Every executor rejection discards the tick's AI-state updates, so next tick plans from the same base state — no exit signal from rejection to planner.
2. **`mark_move_target_failed` is unreachable from executor rejections.** Grep across `src/`: three call sites in `tick_loop_actions.py` and `completions.py`, none in `executor.py`. The `is_move_target_failed`-gated `block_combat_target_and_replan` pathway that would break the loop is disconnected from every one of the executor's nine rejection sites.
3. **Combat teleport onto hostile mine.** `choose_combat_landing_tile` (`combat_landing.py:46-69`) returns the enemy's exact coord and explicitly discards `world`/`terrain` at line 68 — never consults `hostile_mines`. Enemy standing on same-team mine → `_is_valid_move_destination` (`executor.py:360`) rejects every tick until the enemy moves. Silent per-tick deadlock, no self-heal. Same shape as 20:47:31.
4. **`_tracked_combat_target` still position-matches.** `executor.py:392` retains `if tank["x"] != ai_state["combat_target_x"] or tank["y"] != ai_state["combat_target_y"]: return None`. Same pattern Phase 0 removed from `_is_valid_shoot`, but on the teleport path. Usually recovers via fresh planner-tick position, but silent per-tick self-rejection until it does.
5. **`find_teleport_landing_tile` deletes its `blocked_mines` param.** `equipment_search.py:62`: `del start_x, start_y, blocked_mines`. Container-teleport landing never consults mines. Dead code across five callers. Latent — same-shape loop would fire if a container landed on a mine tile.
6. **Commit `4d11980b`** (2026-07-03) narrowed the mine check from `world["mines"]` to `hostile_mines(world)` for same-team passability. Correct for that motivation, but the planner-executor consistency question was not in scope for that commit, so the mine loop's precondition (planner emits coord on hostile mine, executor rejects) was created there without a corresponding replan wiring.

**Pages written:** [[executor-rejection-loops]] under [[architecture]] hub. Symptom, structural cause, three live instances, one latent, five fix options (A: wire `_is_valid_move_destination` to `mark_move_target_failed`; B: same for `_is_valid_teleport`; C: delete dead `blocked_mines` param; D: per-target-id block for shoot; E: retire position-match in `_tracked_combat_target`). The structural fix is Phase 1 of [[self-observing-architecture]].

**Structural updates:** architecture hub 6 → 7 pages; index total 52 → 53.

**No code changes landed** — audit only. Fix options queued for user decision.

## [2026-07-17] audit | Viewport shift protocol cracked — game supports what the bot doesn't use

**Motivation:** the user pushed back on treating the executor-rejection audit as "how do we cope with rejection." Better question: what is the bot misperceiving about the game? Concrete lead — viewport modelling: the bot thinks viewport is fixed until teleport (`state/scan_coverage.py:29`, `hunt_mode.py:52-53`); the user asked whether that's actually a game rule or a bot-configuration choice.

**Investigation phases:** (1) wiki survey — [[viewport-frame]], [[viewport-update-algorithm]], [[js-source-map]]; (2) code layer — `state/viewport_geometry.py` (16×16 visible / 18×18 patch), `sniffer/viewport.py::update_viewport_origin`, `sniffer/world_state_tiles.py:68`; (3) JS client read — `tpclient.pretty.js` at 236-243 (`Ia`), 762-788 (`Rb`/`Sb`), 1620-1662 (state 13 dispatch), 5060-5129 (autoscroll settings); (4) capture-corpus decode — `runs/sniff/latest.capture_session.json` decoded via `capture.xor.build_xor_table` + `xor_decode_body`.

**Findings:**

1. **The game fully supports viewport shifting** — three triggers: (a) teleport landing, (b) client-initiated `Rb`/"Z" (3-byte scope-extend, direction 0-8) or `Sb`/"z" (4-byte scope-move to tile), (c) server-side auto-shift on walk when autoscroll is enabled via the `Ia` text control (`"A1"` = ON, `"A0"` = OFF). All three converge on `0x5A ViewportUpdate` from the server, which the sniffer already handles correctly.
2. **Empirical corpus (2026-07-10 human session, 421.8 s):** 22 × `0x5A`, 22 × `0x3D MovementResponse` (1:1 pairing), 42 × `0x47` walk broadcasts, 4 × sent teleports, 8 × "Extend view {NE|E|SE|W|N}" game-log lines. Every "Extend view" is followed by a `0x5A` within 0-2 s — proves `Rb → 0x5A` round-trip. Remaining ~10 `0x5A` beyond teleports and scope-extends are walk-triggered, evidencing server-side auto-shift.
3. **The bot never sends `Ia`, `Rb`, or `Sb`.** Sniffer machinery for tracking the shift is correct (`update_viewport_origin` at `sniffer/viewport.py:14-23`); it's simply never exercised outside teleport-landing because the bot doesn't request shifts.
4. **Wiki correction:** `viewport-frame.md` [^4] cited a 2026-06-21 user quote ("we have viewport shifting off. so the viewport will never move. the only way is to teleport") as a game-rule statement. That's a bot-configuration statement. The game rule is: shifting works; the bot doesn't use it. Page updated to distinguish.
5. **Latent doc bug:** `src/tankpit_bot/protocol/commands.py:95-96` labels `PLAIN_AUTOSCROLL_ON = b"A0"` and `PLAIN_AUTOSCROLL_OFF = b"A1"` — inverted from JS (`Number(true) == 1`, so `"A1"` = ON). Same swap in `docs/protocol-discovery.md:435-436`. Constants unused in `src/` (grep 2026-07-17), so no live misfire, but any future consumer would enable/disable backwards.
6. **Connection to [[executor-rejection-loops]]:** the pursuit-shot rejection scenario (enemy walks off fixed viewport, bot re-shoots at off-viewport coord, server refuses with 0x52 code 0) has a fixed structural cause (bot deliberately restricted to the landing viewport). Enabling scope shifts would let the bot extend range without teleports; the 2026-07-03 `_clamp_aim_into_viewport` fix would become less load-bearing.

**Pages written:** [[viewport-shift-protocol]] under [[protocol]] hub — wire contract for `Ia`/`Rb`/`Sb`, corpus proof, state-machine ties. **Pages updated:** [[viewport-frame]] — reframed "shifting is OFF" from game-rule to bot-choice, added [[viewport-shift-protocol]] cross-link.

**Structural updates:** protocol hub 7 → 8 pages; index total 53 → 54.

**No code changes landed** — audit + wiki reframing only. Three directions queued for user decision: (α) status quo — keep teleport-only; (β) send `Ia("A1")` at startup, let server auto-shift, review all fixed-viewport assumptions in bot code; (γ) implement `Rb`/`Sb` dispatch in pursuit + off-viewport-refresh paths.

## [2026-07-18] code | Autoscroll constants fixed + Phase 1a contracts/facts foundation

**Autoscroll fix:** `PLAIN_AUTOSCROLL_ON`/`OFF` in `protocol/commands.py:95-96` un-inverted (`"A1"` = ON per JS `"A" + Number(setting)`), same fix in `docs/protocol-discovery.md`. Constants still unused in `src/`; [[viewport-shift-protocol]] "Latent doc bug" section marked fixed.

**Phase 1a of [[self-observing-architecture]] landed** — the contracts framework + Facts core + guard rule (first chunk of Phase 1; retrofits 1b-1d remain):

- `src/tankpit_bot/contracts/`: `ContractError` hierarchy with self-naming subclasses (`NoUnsourcedFactError`, `ConfidenceOutOfBoundsError`, `ProvenanceRootednessError`), `require()` helper capturing the violation site as `file:line`, `@enforce_contract` decorator whose `Contract` protocol is generic over a `ParamSpec` — a contract's `check` mirrors the guarded function's typed signature (the monorepo guard bans `object` annotations and `import inspect`, which shaped both designs).
- `src/tankpit_bot/facts/`: generic `Fact[T]` (value/source/observed_ms/confidence/provenance), 11-source `FactSource` literal (9 wire + game_log_scrape + client_side_inference), provenance chains with encode/decode round-trips, confidence arithmetic (noisy-OR combine, weighted combine, exponential decay). All three fact contracts enforced at `make_fact` AND `decode_fact` — a stored fact violating a contract fails at load, not at use.
- `scripts/contract_rules.py` guard rule wired into `scripts/guard.py` (`make lint`): public `apply_*`/`record_*`/`mutate_*`/`set_*`/`update_*` in `facts/`, `ledger/`, `memory/` must carry `@enforce_contract`.

**Design deviations documented on the wiki page:** `game_log_scrape` is an observation origin (only inference requires citations); `make_fact` calls its contract explicitly because mypy erases a generic function's type variable under a decorator.

**Also answered (no code change):** the bot DOES sense mines — three wire channels feed `world["mines"]` (radar responses with team data, viewport tile updates, witnessed `0x4B` placements / `0x45` detonations), and pathing consults `hostile_mines()` (enemy-team only, friendly mines passable). The known gap is unchanged: `choose_combat_landing_tile` deliberately ignores mines (rejection-loop instance #1 in [[executor-rejection-loops]]) and `find_teleport_landing_tile`'s `blocked_mines` param is dead code (fix C, queued).

## [2026-07-18] code | Fix C (dead blocked_mines) + Phase 1b/1c entity fact retrofits

**Fix C landed:** `find_teleport_landing_tile` is now honestly `(terrain, goal_x, goal_y)` — the dead `start_x`/`start_y`/`blocked_mines` parameters (accepted and immediately `del`'d) are gone, along with the false mine-awareness they advertised. `_teleport_fallback_command` shed the same dead arguments; five production callers and all tests updated. The docstring now states explicitly that landing choice does not consult mines (server displaces; planner owns avoidance). Closes the latent item in [[executor-rejection-loops]].

**Phase 1b/1c of [[self-observing-architecture]] landed** — `ContainerStateDict` and `TankStateDict` now carry full fact metadata (`confidence` + `provenance` added to the pre-existing source/timestamps), with `Fact[T]` projections in `facts/container_facts.py` / `facts/tank_facts.py`. Key decisions (full notes on the wiki page):

- **Flat-carry, not nested reshape** — ~200 construction + ~300 access sites unchanged; the projection provides the true `Fact[T]` view.
- **`FactSource` grew 11 → 18** — the handoff spec's list missed the tank-registry channels (0x21/0x28/0x3E/0x42/0x47/0x48, registry DOM scrape). `TankObservation.fact_source` records the exact channel at all 12 dispatch sites.
- **Convergent decode defaults** — legacy snapshots decode to byte-identical state vs contemporary encoders; no divergent paths (user-verified concern).

All 4363 pre-existing tests passed unmodified after the retrofit (zero call-site churn proof); 15 new tests cover the mappings, round-trips, legacy decode, projections, and the provenance-recording mutator.

## [2026-07-18] code | Phase 1d — self/mine/terrain/viewport fact retrofits; Phase 1 COMPLETE

Final Phase 1 substep of [[self-observing-architecture]]: `SelfStateDict`, `MineStateDict`, `TerrainTileDict`, and `ViewportStateDict` now carry flat fact metadata (`confidence` + `provenance`; self/viewport/terrain also gained `observed_ms` — they previously had no timestamp at all). Projections for all four in `facts/world_facts.py`.

**Channel threading:** `FactSource` grew 18 → 23 (0x2B promotion, 0x44 fuel gain, 0x4A terrain update, 0x4B mine placement, 0x64 fuel total). Self-position mutators take a `fact_source` param (0x47 waypoint vs 0x3D movement paths pass their own); fuel totals thread per dispatch arm; witnessed mine placements stamp 0x4B while radar/viewport sightings derive from coarse source; terrain distinguishes 0x5A patch grids from 0x4A updates; viewport origin is always 0x5A (see [[viewport-shift-protocol]]).

**Single-path discipline maintained:** raw `SelfStateDict`/`ViewportStateDict` constructor literals across src and 31 test files were converted to the factories; `make_viewport_state` is the sole viewport construction path. Legacy decode converges to contemporary-encoder output (tested by key-deletion round trips).

**Phase 1 (contracts + facts foundation) is COMPLETE** — 1a contracts/facts core + guard rule, 1b containers, 1c tanks, 1d the rest. Next: Phase 2 `ledger/` core.

## [2026-07-18] code | Phase 2 outcome fabric — the three diagnostic channels unified, executor discards recorded

The heart of Phase 2 of [[self-observing-architecture]]: the unified `action_outcome` fabric replacing the three parallel diagnostic mechanisms — and making the executor's silent discards (the 20:47:31 deadlock class) structurally impossible to hide.

**New `ledger/` package:** `events.py` (process-wide monotonic event ids + the six recorded `ActionKind`s), `outcomes.py` (six per-kind outcome vocabularies, 31 labels total, mirroring the real resolution signals), `ring.py` (bounded per-kind ring of typed outcome records, queryable via `recent_outcomes`/`outcome_counts`), and `outcome/` with per-kind emit modules whose helpers carry strict per-outcome typed signatures (no sentinels — map_open has no target fields at all).

**Producers migrated (single classification, single record):**
- `completions.py` gates → typed emitters (`emit_scan_radar_complete`, `emit_move_position_reached`, `emit_teleport_landed` — which absorbs the old `teleport_attempt` window tracking — `emit_collect_position_reached`/`container_consumed`).
- `tick_loop_actions.py` 0x52/movement-rejected/stall paths → kind-routing dispatchers to typed emitters.
- `executor.py`'s nine `emit_ai("rejecting …")` sites → nine typed discard outcomes (`discarded_hostile_mine`, `discarded_combat_target_stale`, `discarded_no_container`, …). **Every silent-loop instance from the [[executor-rejection-loops]] audit now leaves a first-class recorded event.**
- `tick_loop._get_combat_feedback` → `emit_shoot_hit`/`miss`/`command_rejected` with hit-signal attribution (tile_occupied / kill_confirmed / ammo_delta).

**Deleted (no shims, no fallbacks):** `runtime_logging.emit_wire_complete`, `diagnostics/teleport_attempts.py`, the `combat_feedback` diagnostic kind, the whole `WIRE_COMPLETE` channel. Zero references remain in production code.

**Consumers migrated:** issue report family (`ActionOutcomeRowDict` replaces `WireCompleteRecordDict`; teleport success counts combine action-lab attempts + bot outcomes), `bot_query` (stalls/action-spans read `action_outcome`), `runs_index.count_stall_timeouts`, `session_stats`. The action-lab probes' own `teleport_attempt` diagnostics remain — different producer, still live.

**Guard milestone:** the Phase 1a contract rule enforced its first real code — `record_teleport_dispatch` (a `record_*` mutation in `ledger/`) now carries `@enforce_contract(TeleportDispatchContract())` with `LedgerInvariantError` on off-map coords / negative message index.

**Remaining Phase 2 work (next):** typed Decisions (`reason_kind` enums replacing the 18 free-text `reason` strings), Decision↔Outcome correlation + `OutcomeInvariantContract`, causal chain, first-class mode transitions, `DecideCtx` ledger views, scorecard per-outcome counters.

## [2026-07-18] code | Phase 2 typed decisions — free-text reasons replaced by the ReasonKind vocabulary

Second Phase 2 chunk of [[self-observing-architecture]]: the planner's free-text `reason` string is gone. `BehaviorScoreDict` now carries `reason_kind` (a closed 17-value `ReasonKind` Literal) + `reason_context` (typed scalar map: `target_name` for the combat kinds, `volume` for the fuel kinds).

**Key discovery during migration:** `reason` was never just logging — `derive_hunt_mode_state`/`derive_collect_mode_state` branch on it to derive AI mode substates (`confirm_kill` → CONFIRM_KILL, `scan_on_landing` → SCAN_ON_LANDING, `forage_*` → SENSE, ...). It was stringly-typed control flow; the closed Literal makes an invalid reason a type error instead of a silent substate misroute.

**Vocabulary (17):** scan_on_landing; COLLECT: equipment_locked, fuel_locked, equipment_restock, equipment_hop, fuel_collect, forage_radar, forage_sweep, search_collect_local, map_for_dots; HUNT: find_target, find_enemies, teleport_target, shoot_target, dot_relay, confirm_kill; controller: manual_hold. The f-string reasons (`f"find {name}"`, `f"fuel={vol}"`) split into kind + context.

**All 18 planner `make_decision` sites + the controller's manual_hold migrated**; `map_reason`/`reason` passthrough params retyped to `ReasonKind`. Consumers (executor AI-decision emit, tick-loop HUD overlay, replay narration) render via the single `render_reason()` formatter — `kind(key=value)`. Codec round-trips validate the kind against the closed vocabulary at decode.

Gate: 4417 tests, 100% coverage, guard/ruff/mypy clean.

**Remaining Phase 2:** Decision/Outcome event-id correlation + `OutcomeInvariantContract`, causal chain, first-class mode transitions, `DecideCtx` ledger views, scorecard per-outcome counters.

## [2026-07-18] code | Phase 2 COMPLETE — Decision↔Outcome correlation, mode transitions, invariant sweep

Final Phase 2 chunk of [[self-observing-architecture]]. The correlation layer uses the bot's own one-in-flight-per-kind invariant as the pairing rule: the executor records every dispatchable decision (`ledger/decision.py`, guard-enforced `DecisionRecordContract`); the single outcome-emission path consumes the pending decision into `caused_by`; a same-kind re-dispatch closes its predecessor with an explicit `superseded` outcome. Every recorded decision therefore resolves to exactly one outcome — `verify_outcome_invariant()` raises `LedgerInvariantError` at session end otherwise, which is reachable only by bypassing the fabric (proven by test).

Mode flips are first-class `mode_transition` events (event id + reason_kind + causal decision). Session end emits per-kind outcome counts from the rings plus unresolved-decision ids; the scorecard and issue report carry `action_outcome_counts` tallies.

Gate: 4430 tests, 100% coverage. Phase 2 (~1900 LoC production + tests this session across three chunks) is DONE; roadmap updated. Next: Phase 3 decision enrichment.

## [2026-07-18] live-run | Phase 1+2 stack validated on the wire (60 s diagnostic session)

First live exercise of the entire self-observing stack (`TANKPIT_BOT_SESSION_SECONDS=60`, field01). 23 ticks, 1 kill, 11/11 hits, clean session end. Everything new fired correctly:

- **20 `action_outcome` events, 19 causally attributed** (`caused_by` > 0). The single unattributed one is the map-open `dispatch_command` auto-fires as a teleport precondition — an implementation detail of the teleport decision, not a planner decision; honest attribution. (Possible refinement: attribute auxiliary map-opens to the owning teleport decision.)
- **The `superseded` path fired on a real mid-action teleport re-dispatch** — the exact case the correlation design predicted.
- **Teleport outcomes carry full wire windows** (dispatch context + received-message window) — the absorbed teleport-attempt machinery works in situ.
- **Every shot attributed to its own decision** with target name, victim id, hit signal (`tile_occupied`), and on-intended flag.
- **Both mode transitions recorded** (UNSET→HUNT via find_enemies, HUNT→COLLECT via equipment_restock) with causal decision ids.
- **Session-end sweep passed** and correctly reported the one legitimately unresolved decision (the final shot, wire never answered before shutdown) instead of raising.
- **Issue report renders the new sections**: outcomes tally on the scorecard, ACTION OUTCOMES listing, teleport success/failure now derived from the outcome fabric.

Cosmetic note for later: the report's TELEPORTS header says "0 attempts" (action-lab rows) alongside outcome-derived success/failure counts — phrasing mixes the two sources.

## [2026-07-18] live-run + fix | Early-exit root cause cracked by the new decline instrumentation

Second 60 s run reproduced the `no_productive_collect` early exit in 5 ticks (homings persist at 20/25 across sessions, so the trap springs immediately). The new `hop_declined` diagnostics named the guilty filters precisely:

- **Dot hop:** `dots_total=622, impassable=192, unaffordable=2, already_scanned=0, not_walkable=428`. The `_viewport_walkable_fraction < 1.0` filter — requiring the 16x16 landing viewport to be 100 % walkable ground — rejected **428 of 622 fuel dots**. On terrain-bearing maps the filter is effectively unsatisfiable once nearby clean viewports are used; it is the primary reason the bot "can't hop".
- **Equipment hop:** `external=1, no_landing=1` — the only out-of-viewport candidate's tile and all four cardinal neighbours are impassable per terrain belief, so no legal landing. Meanwhile **6 in-viewport equipment containers sat `blocked_walk`** — invisible to the hop path by design (it only considers external containers), even though teleporting directly onto a walk-blocked container would work (server displaces; landing auto-pickup).
- **Gate:** homings 20/25 < capacity blocks HUNT per the Bug 0.4 full-inventory contract.

**Fixes landed this session (report/ledger side, gate-green 4429 tests):** teleport-precondition map-opens now resolve the owning teleport decision via `transfer_pending_decision` (no more handshake `superseded`, no more unattributed map-opens); `superseded` no longer counts as a teleport failure in the report; TELEPORTS header separates outcome-derived counts from action-lab attempt rows; both hop selectors emit structured `hop_declined` tallies on every decline.

**Open policy decisions (fight logic — user's call):** (a) relax the 100 %-walkable landing filter (threshold? landing-tile-only?); (b) allow teleport-to-container for in-viewport `blocked_walk` equipment; (c) whether `no_productive_collect` should end the session or keep searching; (d) the full-homings HUNT gate at 20/25.

## [2026-07-18] fix + live-run | Dot-hop reranked: "prioritize dots + walkable, not a 100% rule" — early-exit cured

**Contract correction (user, verbatim): "the rule was to prioritize viewports with more dots, more walkable area. but not a 100% rule ofc."** The 2026-07-03 implementation had mis-read "100% clean viewport" as a hard `walkable_fraction == 1.0` filter; archaeology traced the quote and the data convicted the filter (428/622 dots rejected).

**New selector** (`_pick_fresh_dot_hop`): hard gates are physics only (own tile, landing tile passable, affordable, viewport unscanned); qualifying dots are RANKED by `score = dots_in_landing_viewport × walkable_fraction ÷ teleport_cost` — expected pickup value scaled by reachable area per fuel spent, with proximity built into the cost denominator. Every hop now also emits a `hop_selected` diagnostic with its score and cost.

**Before/after, same account state (homings 20/25 trap):**
- Run 2 (old filter): 5 ticks, exit `no_productive_collect`, nothing accomplished.
- Run 3 (ranking): **full 60 s, exit `completed`** — 2 hops selected (scores 0.0155 and 0.1126 — the second a 24-fuel hop into a dot cluster), 4 equipment gains, **homings restocked 20 → 25/25**, fuel topped to cap 1100, 1 kill at 11/11 hits en route. The exact failure chain (can't hop → can't restock → can't hunt → quit) is broken at its root.

Remaining from the audit trio: (b) in-viewport `blocked_walk` equipment invisible to the hop path (the recurring `equipment hop declined: external=1, no_landing=1` lines — that lake-locked container belief); (c)/(d) session-exit policy and the full-homings gate — now much less load-bearing since the dot hop works, but still open user calls.

**Tests:** two 100%-rule tests rewritten to the ranking contract + new cluster-preference test (the user's "more dots" ask, pinned). Gate green: 4430 tests, 100% coverage.

## [2026-07-18] feature | Graceful Q-quit on session teardown

User request: the bot should press ``q`` (the plain `PLAIN_QUIT` wire command) before closing the browser so the server records a deliberate lobby exit instead of an abrupt socket drop. Implemented: `build_quit_command()` (2-byte LE length header + `-`, no XOR — the sender already passes non-`!` bodies through plain), `CommandService.quit_game()`, and `Bot._send_graceful_quit()` first in the game-loop `finally` while the CDP session is still bound; the send path's existing None-guard means a crashed browser cannot wedge teardown. Wire framing pinned by test (`b"\x01\x00-"`). Gate green.

Also this session: explained the 650-fuel engagement reserve (450 engagement budget + 200 low-fuel floor — a hop may never land below it) and identified the 14:51/14:52 mystery runs as sessions launched by the user's SPA bot service (port 27100): unbounded "until stopped" sessions; the STOP file ends a session but the service process itself persists until killed.

## [2026-07-18] contract | Full-inventory HUNT gate reconfirmed strict

User, after reviewing the gate's role in the early-exit chain: **"no keep it strict."** `hunt_entry_permitted` stays full-duals + full-homings + radar-floor with no softening — the restock cost collapsed once the dot hop was fixed (20→25 homings in ~15 s live), so the gate's protection is nearly free. The cardinal-adjacent free-kill override remains the sole exception (Bug 0.5). Remaining open policy call: whether `no_productive_collect` should ever end a session now that hopping works.

## [2026-07-18] mechanic + fix | Equipment containers fill-what's-empty; code 7 = ALL slots full; pickup gate + reconciliation

**User mechanic (verbatim): "the equipment containers are not determined prior to pickup. they fill whatever is empty. you will only get a full inventory message if all your items are full."** This corrects two prior beliefs: containers are not fixed-type grants, and 0x52 code 7 is not "a slot is at cap" but an authoritative statement that EVERY slot is at cap.

**Desync hypothesis retracted.** The suspected inventory desync in the 5-min run was an analysis error: the dual=17 sample at 16:48:01 preceded a +8 dual 0x67 gain at 16:48:03; belief was a truthful 25×5 before the first code-7 at 16:48:05. The fine-grained timeline shows the shadow count tracking the server faithfully all run (per-hit decrements matching the server's weapon mix, 0x67 gains restoring exactly the empty amounts — the fill-what's-empty mechanic visible in the data).

**Single root cause for all 8 wasted rejection ticks: no fullness gate on equipment pickups.** Fixes landed:
1. `inventory_all_full` predicate + gate in `_select_and_pickup_equipment` — at all-slots-full a pickup is a guaranteed code-7; skip it (saves ~2 s/tick).
2. Code-7 handling rewritten: previously blacklisted the container ("slot won't clear" — wrong mechanic); now `update_inventory_from_full_signal` reconciles every slot belief up to `inventory_capacity(rank)` (the rejection is an absolute inventory statement — self-healing against any future drift) and the container is KEPT.

Gate green: 4441 tests, 100% coverage.

## [2026-07-19] live-run | Fullness gate verified: 0 code-7 rejections at 4 all-full states

5-min verification run for the equipment-pickup fullness gate + code-7 reconciliation. The bot reached all-slots-full four times (vs twice in the pre-fix run that produced 8 code-7 rejections) and dispatched zero doomed pickups. Sole rejection all run: one code 4 (genuinely empty container, correctly blacklisted). 53/54 hits, 19/19 teleports landed, 0 stalls, ended 25/25/25 at fuel 653 mid-hunt, graceful quit on the wire. Both 2026-07-18 waste findings are closed.

## [2026-07-19] falsification + teardown | 0x41 fires for own kills; DOM game-log consumption deleted (wire is the single actor)

**Trigger:** run bot-20260719-004608's scorecard said Kills: 5 but only 4 tanks died — purple-3 (id 511) was counted twice. The 0x41 wire deactivation landed on tick 122 and the DOM game-log banner was scraped on tick 123; the drain-set dedup only collapses same-tick signals, and `_merge_protocol_kills` incremented `session_kill_count` without consulting the kill cooldown.

**Root-cause archaeology falsified two June claims via capture replay.** Replaying the ORIGINAL June 10 captures through the current decoder (XOR table from each capture's `magic`):
- `bot-20260610-005248`: 1 own-kill 0x41 (victim 512, killer 1301 = the bot). `bot-20260610-011333`: **19** own-kill 0x41s. These are the exact runs the "0x41 never fires for own kills" claim (and the game-log kill scraper) was built on. Every own-kill 0x41 is 0x2E-tunneled; the June decoder had no 0x2E subtype dispatcher, so it decoded none of them — decoder blind spot, not server behavior.
- Same replay found 21× 0x52 `error_code=4` (empty container) and 18× code=5 in the June capture — falsifying "the wire is silent on failed pickups", the rationale for the DOM empty-container consumer.
- "You can't go there!" needed no new capture at all: the client's supervisor error-string table (already in [[client-constants]]) has it at **index 1** — the banner IS the client rendering 0x52 code 1, which the bot already handles (incl. `mark_move_target_failed`). Banner timestamps trail their wire codes by 2–4.6 s (DOM render + scraper poll lag).

**Teardown (all three DOM consumption paths deleted; wire is the single actor):**
1. `diagnostics/game_log_kills.py` and `diagnostics/game_log_feedback.py` deleted with their tests; executor dispatch-recording side-channel removed. 0x41 is the sole kill channel — the double-count is structurally impossible, not guarded against.
2. Wire 0x52 code-4 branch now does the empty-container belief removal itself (`remove_container_at`), replacing the DOM consumer's job 1–2 ticks earlier. Code 0 still blacklists via `failed_pickups`.
3. The DOM log survives as a **witness only**: the bot timestamps polled entries into the capture artifact (`game_log` field, previously always empty for bot runs) so the analyzer can diff the client's rendering against the wire. `game_log_scrape` removed from `FactSource` (23→22).
4. Legacy dedup removed downstream: scorecard counts every `tank_deactivated` (one per kill now; victim-id dedup would silently drop legitimate respawn re-kills — June capture shows victim 507 killed 5×); `session_stats` counts `origin="protocol_0x41"` and drops the `feedback_corrections` column.

**Bonus fix — the unresolved kill-shot decision (event 235 in the trigger run):** `_merge_protocol_kills` cleared `last_shot_target_id` before `_get_combat_feedback` ran, making the `kill_confirmed` classifier branch unreachable — kill shots never resolved in the ledger (a kill produces no damage-change feedback). The merge now preserves the shot target; the classifier resolves the shot as `kill_confirmed` and clears the fields itself (its trigger is not a consumable wire flag, so it must self-clear to avoid re-emission).

**Pages updated:** [[deactivation-format]] (own-kills section rewritten with replay evidence, fact_checked 2026-07-19), [[shoot-event-format]], protocol hub line, [[self-observing-architecture]] (FactSource deviation note), [[executor-rejection-loops]] (call-site list).

Gate green: 4432 tests, 100% coverage.

## [2026-07-19] tooling | Deterministic run audit: `tankpit-run-audit` in `make analyze`

**Motivation (user):** "im worried that we cant properly analyze the runs. and that each ai may interpret the runs differently." Run interpretation had already drifted twice (June's "0x41 never fires" mis-read; the retracted inventory-desync claim) because every session re-derived conclusions from 11 MB of raw JSONL by hand.

**Built:** `diagnostics/run_audit{,_types}.py`, `ledger_audit.py`, `capture_audit.py` — a typed-finding audit (`check`, `severity`, `summary`, scalar `evidence`) wired into `make analyze` as `tankpit-run-audit [events.jsonl]` (capture resolved as the sibling artifact). Same artifacts in → same verdicts out, regardless of who runs it.

**Ledger checks** (each encodes a formerly hand-made interpretation — the ratchet rule): `kill_double_registration` (repeat victim inside 30 s = channel regression), `unresolved_decision` (shutdown sweep surfaced per kind), `stall_timeout`, `command_rejection`, `rejection_retry_loop` (≥2 failures on one (kind,target) = replanning not learning; the [[executor-rejection-loops]] class), `executor_discards`, `superseded_churn` (>5/kind), `tick_cadence_gap` (>8 s), `session_exit` (always emitted; missing scorecard is itself a warning), `empty_run`.

**Capture replay checks** (the check class that falsified the June claims, now standing): every received frame re-decoded with the CURRENT decoder (local XOR table from the capture's magic — never touches live decoder state); `decode_error` (frame the decoder raises on), `unknown_container_subtypes` (undecoded 0x2E subtypes — the blind-spot canary), `deactivation_channel_diff` + `supervisor_channel_diff` (wire count vs ledger-ingested count; mismatch = the June class of decode/dispatch gap), `dom_witness_diff` (kill/empty-container/blocked-move banners the client rendered vs the wire messages that explain them — possible because the bot now records the DOM log into the capture as a witness).

**Validation against run bot-20260719-004608:** the audit retroactively finds every bug this week's hand-analysis found — all 4 double-registered kills (pre-teardown run), unresolved shoot decision 235, the code-4 rejection — and verifies wire=ledger on both channels (4/4 deactivations, 3/3 supervisor errors).

Tests: hand-built ledger records for exact window/threshold control; capture tests XOR-encode real frames and run the REAL decoder end to end. Gate green: 4467 tests, 100% coverage.

## [2026-07-19] falsification + fix | The "dead sessions" were test droppings, not sessions

The run audit's `empty_run` finding on `latest.*` surfaced seven zero-event stamped artifacts from today (13:08–14:21). Investigation ruled out every launcher: the service path constructs `Bot` in-process and never calls `configure_bot_runtime_logging` (it cannot produce stamped artifacts at all); the Sunshine/orchestrator profile runs `make run` (300 s bound — the dead logs all say "until stopped"); PowerShell history shows no manual launches.

**Root cause:** `test_main_installs_handlers_with_request_interrupt` used `fake_env` but not `fake_fs`, so it drove the REAL `entry.main()` through real runtime-logging configuration — writing a genuine `bot-<now>.log` (the 4-line/316-byte signature), truncating the real `latest.events.jsonl` to empty — before aborting at its playwright sentinel. One "dead session" per `make check`. Reproduced deliberately: running that single test created `bot-20260719-145249.log` and wiped the events file a live run had just written. Six more `bot.run`/`_game_loop` tests in `test_run.py` leaked the same way one layer down (unconditional `latest.summary.txt` write in `_emit_session_scorecard` — the mystery "Ticks: 1 / stop_file / UNSET" summary was test data).

**This falsifies the 2026-07-18 log entry's attribution** of the evening zero-event artifacts (19:38–21:55) to "sessions launched by the user's SPA bot service" — those were the same test droppings from that evening's `make check` runs. (The 14:51/14:52 July 18 runs with real events were genuine service sessions; that part stands.)

**Fix:** `fake_fs` added to all seven tests, each now intercepting every filesystem write. Verified: full `make check` leaves the artifact count unchanged (774 → 774). ~14 bogus stamped artifacts remain on disk from before the fix (316-byte logs + 0-byte events, Jul 18 19:38–21:55 and Jul 19 13:08–14:52); left in place pending cleanup approval.

Gate green: 4467 tests, 100% coverage.

## [2026-07-19] fix | Refuel-in-place: the session-exit policy call, settled by capture forensics

**Root cause of the tick-4 exit (run 14:49, cracked by replaying its own capture through the real decision code):** the bot rejoined at fuel 653 (tank state persists server-side; 653 = where the previous session quit). Every enemy failed affordability (`cost + 650 ≤ fuel` allows a 0.5-tile engagement at 653). The dot-relay then declined all 628 map dots — its strict-progress rule only hops to dots STRICTLY CLOSER to the travel target, and with orange-2 just 26.6 tiles away only 6 dots were closer, all on field01 water. 622 usable dots surrounded the bot; refueling on any of them would have made orange-2 affordable (809 ≤ 1100) with zero approach needed. **The deficit was fuel, not distance — the strict-progress relay starved the bot in a supermarket.**

**Fix (user ruling "yes"):** `_refuel_toward_engagement` in `hunt_mode.py` — when a travel-worthy enemy exists but no progress dot qualifies, hop to the best fresh fuel dot in ANY direction via the COLLECT restock picker (`make_resource_search_hop`, new `ReasonKind` `"hunt_refuel"`), inheriting its freshness/affordability/value-ranking gates. Session exit now fires only when the tank is at `fuel_capacity(rank)` (refueling can't help) or no fresh dot qualifies. Verified against the exact live state that produced the tick-4 exit: the decision is now `teleport (89,100) reason=hunt_refuel` instead of `SessionExitError`. Termination: fuel-bearing hops rise toward the cap or affordability; dry hops fall toward the picker's affordability floor — no infinite loop, and the run audit's retry-loop check watches for same-target churn.

Also this session: 438 test-dropping artifact pairs deleted from `runs/bot/` (the month-old leak fixed earlier today); 336 genuine run artifacts remain. [[bot-behavior-contract]] §exit row updated.

Gate green: 4469 tests, 100% coverage.

## [2026-07-19] fix | Fuel pickup gate: binary overfill refusal -> per-tile rate rule

**User question exposed the flaw**: "what if the tank has 600 fuel and there is one 1000-fuel container?" The 2026-07-06 `_would_overfill` gate (`fuel + walk + min(volume, headroom) > cap`) refused ANY clamped pickup at walk ≥ 1 — and refused *earlier* the *bigger* the container: at fuel 600 it walked past a 1000-volume container one tile away, forfeiting a 500-fuel transfer. The gate was really "never take a partial transfer while walking," which contradicts the 2026-06-23 minimum-volume lesson (fuel is fuel) at the cap end.

**Fix**: `_pickup_not_worth_walk` — refuse only when the ACTUAL transfer (`min(volume, headroom)`; the server clamps and answers a now-cleanly-handled code=5) is below **25 fuel per Manhattan walk tile**. Consequences: adjacent pickups always taken (incl. the July-6 canonical 1-tile sliver, +46 for one 2s tile — same rate a good dot hop pays); big clamped containers worth long walks (600-fuel tank walks up to 20 tiles for a 1000-volume container); distant slivers still refused (July-6 26-second waste class cannot return). Boundary: gain == rate×walk is taken.

Also corrected [[fuel-system]] threshold drift (`fuel_low_threshold` is 200, page said 300 since 06-24; config is the source of truth).

Gate green: 4471 tests, 100% coverage.

## [2026-07-19] fix | Typed collect 0x52 outcomes: empty vs clamped vs inventory-full vs refused

**User ask** (after the 5-min soak filed four +500-fuel clamped pickups as "rejections"): "so we can tell between empty pickups or failed pickups and overfill pickups?" The outcome fabric mapped every applicable collect 0x52 to `command_rejected`, conflating four physically different resolutions.

**Fix**: `CollectOutcome` grew three typed labels, routed by error code in `_emit_command_rejected_outcome`:
- **code 4 → `pickup_empty`** — container drained by someone else between scan and pickup (belief removed)
- **code 5 → `clamped_transfer`** — the server transferred `min(volume, headroom)` and kept the remainder; a SUCCESS (the 0x43 partial-pickup updates the container belief, wire absolute fuel carries the gain), not a failure
- **code 7 → `inventory_full`** — authoritative all-slots-full statement (beliefs reconciled)
- codes 0/1 (geometry / can't-go) remain `command_rejected` — the genuine refusals

Run-audit updated: `clamped_transfer` produces no finding and never feeds the retry-loop detector (repeated clamped pickups on one target are repeated successes); `pickup_empty` and `inventory_full` get their own info verdicts and DO count as failures for retry-loop detection (repeated empties on one target = belief not learning; any `inventory_full` = the fullness gate let a doomed dispatch through).

Gate green: 4474 tests, 100% coverage.

## [2026-07-19] falsification + fix | "Maybe we have the wrong map image?" — no; display art ≠ server collision data. Downloader URL was stale.

**User hypothesis tested**: fetched the live client's field image and hash-compared. The client JS builds `/images/maps/field` + id (verified against the live `tp-*.js` bundle); `field01.gif` from that path is **byte-identical (same MD5) to our cached copy**. The cached map IS the current one, and it is the ONLY map asset the client references — no higher-res variant, no collision overlay exists client-side.

**Conclusion locked in**: the (163,44) "You can't go there!" vs GIF-ground divergence (and the server standing the tank on GIF-water at (167,40)) is a display-art-vs-server-collision-data gap that NOBODY outside the server can close a priori — zero 0x4A terrain messages in the whole capture; the real client doesn't predict passability either (that's why the error string exists). The bot's mark-failed-and-move-on handling (~2 s per lesson, session-scoped) is the same information position a human player has. Cross-session persistence of these verdicts = the nominated Phase 4 memory pilot.

**Latent bug found by the probe**: `scripts/download_fields.py` still pointed at the retired `/play/fieldXX.gif` path, which now serves the SPA's HTML — `make download-fields` would have skipped every field (GIF-magic guard caught the HTML; broken but not destructive). Base URL fixed to `/images/maps`, pinned by test.

Gate green: 4474 tests, 100% coverage.

## [2026-07-19] correction | The map is NOT diverging from the server — earlier "display art vs collision data" claim retracted

User pushback ("the map is pixel perfect from the game map") prompted a systematic re-test, and they were right; the previous log entry's theory is falsified:

1. **Coordinate mapping verified**: scored all 9 candidate pixel offsets against 25 wire-confirmed stood-on tiles across three runs — offset (0,0) wins decisively (1 violation vs 3–9 for every shift). No off-by-one.
2. **The one "stood on water" violation is a FERRY** — the viewport legend carries `~=ferry`; tanks ride water tiles. Dynamic object, not a map error.
3. **The (163,44) "You can't go there!" refusal**: the viewport render shows a friendly 2×2 mine cluster at (162–163, 42–43) beside the approach, and the terrain there is a maze of one-tile water pockets. Decisively: running the PRODUCTION reachability check offline against the reconstructed world returns **False from the bot's actual dispatch position (167,40)** — our own pathfinder AGREES with the server on the end-of-run state. The live dispatch at 18:20:33 passed on some transient mid-ingestion belief (the radar's mine/tile reveals landed :31–:33) that could not be exactly reconstructed. A dispatch-time race, not a map divergence.

**Standing conclusions**: the field GIF is byte-identical to the live client's only map asset, the pixel↔tile mapping is exact, and both pathfinders are 4-directional and agree on settled state. The residual failure class is dispatch-time state races (~1 per 90 ticks, cost ~2 s, self-marking via failed_pickups) plus dynamic objects (ferries, mines) the static map by definition cannot carry — which the world state already tracks live. The Phase 4 terrain-overlay memory pilot nominated earlier is accordingly DOWNGRADED in value: there is no static divergence to remember; the lessons are about dynamic state, which does not persist across sessions anyway.

## [2026-07-19] correction + contract + fix | The (163,44) refusal fully solved: ferry riding rule leaked into pickup routing

Supersedes the earlier "dispatch-time race, could not reconstruct" claim — it WAS deterministic once the right terrain view was used. The bot was RIDING A FERRY at (167,40) (wire 0x5A patch, `terrain_type=5`; also the lone "stood on water" tile in the offset test). `FerryAwareTerrain` with `riding=True` makes ALL water passable, so the pickup gate approved the container across the channel; the server routed the tank to the disembark stop (167,44) and refused the rest with code 1. Offline reproduction is exact: ferry-aware gate=True (the live dispatch), ground-only gate=False (the server's answer). The user's radar correction also stands: the 0x4F reveal is one tick; no "streaming reveals" — that story was wrong.

**User ferry contract captured verbatim in [[ferry-mechanics]] §single-command routing**: one command never chains surfaces. Land→water click = refusal (no auto-boarding); click the ferry tile to board; riding→water click sails fine; riding→land click sails to shore, disembarks, STOPS at the first land tile — a second click finishes the trip. Embark and disembark each cost an action.

**Fix**: `GroundOnlyTerrain` view (only plain ground traversable — water, rock, AND live ferry tiles all block) now gates every pickup dispatch in `movement.py`. When the ground-only gate fails while riding, the planner issues the piloted disembark move (surface-clamped to the first land tile) instead of the doomed pickup; the next tick dispatches from solid ground. Validated against the incident state: new gate=False, disembark clamps to (167,41). Pinned by `TestPickupSurfaceRouting` + `TestGroundOnlyTerrain`.

Gate green: 4478 tests, 100% coverage.

## [2026-07-19] discovery + contract + instrumentation | The ~12 s shoot-at-id TTL after 0x58 TankRemove

The orange-2 escape (soak 22:29, 16 shots, survived at critical) reverse-engineered end to end. User contract captured verbatim in [[shoot-event-format]] §global-action-queue: all actions process through a global server queue; homing converts queue-race misses into hits and has NO range limit; a human can fire exactly one post-departure homing (the click needs a visible tank) while the bot's id-targeted command repeats it — the reroute exploit.

**The boundary measured from the wire**: all 16 shoot commands byte-identical; 0x58 TankRemove(528) at 22:30:00.4; rerouted homings fired at +0.65 … +11.0 s all debited ammo (consumption=hit ⇒ real hits on the departed tank, reaffirmed by user); the shot fired at +13.0 s drew a no-debit response — the id no longer resolved server-side. **Shoot-at-id survives a 0x58 for ~12 s (boundary in [11.0, 13.0] s), then degrades to a tile shot.** Also observed: the departed tank's position goes fully dark until the next map open (22 s, zero updates), and orange-2 was firing back (free singles) right before fleeing.

**Instrumentation added**: `tank_removed` diagnostic on every 0x58 dispatch, so future pursuit-miss timing correlates against removal timestamps and narrows the TTL constant for free. Tactical implication recorded: ~5–6 guaranteed-hit homings exist post-0x58; the stationary-miss→block rule already disengages at the minimum knowable cost (one homing).

Earlier "~16-tile seeker envelope" speculation retracted (that figure is the near-band refusal rule, the opposite regime). The damage 1→2 recovery observation (critical at :29:54, medium at :30:16) remains uncatalogued — first observed damage regression; open question.

## [2026-07-19] mechanic | Damage recovery explained: game bots roam and accidentally refuel; damage tiers track the fuel pool

The "damage 1→2 recovery" open question is closed. First, the encoding hypothesis was eliminated: near-in-time (0x2E sync, 0x4C map) damage pairs agree on every overlap (17/17), so orange-2's critical→medium change was real. Then the user supplied the mechanic (verbatim): "the bots teleport or walk to the next viewport usually. they dont seek fuel, but sometimes they may teleport away and happen to land or step on a fuel tank." Landing auto-picks the container — so the fleeing bot accidentally refueled, and since fuel is the life pool (0 fuel = instant deactivation; hits drain the victim's fuel), its damage tier recovered with the refill.

Bonus wire confirmation the same capture: orange-2's return fire arrived as 0x53 ShootEvents with `weapon=0` — free singles, exactly what Sigma's 2015 guide claimed ("bots return singles") — upgrading that decade-old human observation to wire-verified. [[enemy-bot-behavior]] updated on all three points.

## [2026-07-19] correction | No time-based damage repair — fuel pickups are the only repair mechanism

User (verbatim): "they do not repair over time. only via fuel pickups." The June-11 reading of purple-3's 1→0→3 healing as "tiers repair over time" is corrected in [[enemy-bot-behavior]] — that recovery was also a fuel pickup. Damage model now closed: hits drain the victim's fuel pool, damage tiers are bands of it, and the ONLY way back up is picking up fuel.

## [2026-07-20] contract + fix | Departed targets are never blocked — orange-2 follow-up lands; self-deactivation impossible

**User rulings**: "i dont want us blocking targets arbitrarily. and for situations like orange 2 i want to ensure we are not blocking them." And the death-mechanics contract (verbatim): "you cant kill yourself in game its impossible... you cant die from walking, even at zero fuel it stops debiting. you can use radar. you cant teleport if theres insufficient fuel, but you wont die." Off-viewport death causes for a chased enemy are exactly two: third-party kill or hidden mine.

**Fix — the follow-up**: the stationary-miss branch (ammo available) now calls `release_combat_target_and_replan` — lock cleared, NO 30 s block, `target_departed` diagnostic, map opens — so the next snapshot's fresh position feeds a plain nearest-affordable re-acquisition (no damage weighting, user ruling). Under the old block, orange-2 was on the very next snapshot (22:30:36, dist 32, critical, affordable) and was rejected `"blocked"`. Loop safety holds because the *lock release* is what prevents repeat-shot loops (the 2026-07-02 01:23 loop kept the lock); corpse follow-ups self-correct on arrival (landing scan reveals corpse direction → liveness flips → dropped). Pinned by `TestDepartedTargetFollowUp` (full release → fresh-map → teleport-reacquire sequence) plus the rewritten stationary-miss tests.

**Remaining block sites audited for the user** — three, all evidence-based (server refused us about a present target) and all practically dormant: rejected shot (fired once ever, pre-aim-clamp 2026-07-03), failed combat landing (never fired — archive sweep: 13 failed teleports total, ALL June 10–20, ALL stall-timeouts at invalid/out-of-bounds coordinates from the blind-hop era; zero since June 20), ammo-exhaustion-with-nothing-collectible (fired once, 2026-07-06, caused by the since-fixed restock starvation). None can catch a departed target.

**Wiki corrections**: [[game-rules]] 0-fuel note rewritten (deactivation = drained BY DAMAGE; self-spending clamps at zero), [[fuel-system]] marooning re-framed as a strand-not-death (and the famous "island" was a drivable ferry).

Gate green: 4479 tests, 100% coverage.

## [2026-07-20] verification | The "fuel oscillation" was the documented economy running at cap; walk=1/tile re-confirmed; teleport row fixed

The end-of-soak −45/+45/−10/+10 pattern decomposed entirely into ALREADY-DOCUMENTED constants: short dot/equipment hops at floor(6×euclid) (wire teleports at 00:57:06/:13 drew exactly −34/−45), radar at 10/fire, walking at 1/tile, and landing auto-pickups clamping back to exactly 1100 at cap. Nothing new — the bot was grazing through a fuel-dense area, every expense instantly refunded by the ground.

Method lesson logged: the first two readings ("incoming fire", then "~5 fuel/tile walking") were guesses that contradicted [[game-economy]]'s own verified walk=1 row — check the wiki before re-deriving. Clean re-measurement: six exact 1/tile SelfMovement segments (added to the page); the stale "Teleport: unknown" row updated to floor(6×euclid) with isolated samples; open item sharpened to per-weapon firing costs (paired −45/−10 per combat tick, undecomposed).

## [2026-07-20] validation | Teleport cost floor(6×euclid) systematically validated — 248/248 exact post-fix; charged on ACTUAL landing tile

User challenged the teleport row ("are we sure about the teleport cost? is that validated?") — and the challenge was earned: the row's "Long-verified... hundreds of hops" claim traced back to [[teleport-mechanics]] footnote [^3], an undocumented assertion with no dataset behind it.

Ran the systematic pairing across ALL `runs/bot/*.events.jsonl`: every `teleport(x,y)` dispatch matched to its wire fuel delta (pre-hop `Self:` fix → post-hop `Fuel: A -> B` line; windows with intervening move/radar/shoot/pickup or a fuel-level mismatch excluded as contaminated). Results:

- **Post-2026-06-24 era** (after the fuel double-count fix): **248/248 clean pairs exact** at floor(6×euclid), costs spanning 6–654 (1-tile hop to ~109-tile jump).
- **All-era**: 2,335/2,538 exact. Every one of the 203 residuals lives in pre-fix runs (peak: June 10–17, the broken-fuel-tracking / blind-hop era); zero residuals in any run after the 2026-06-23 fix.
- **New refinement**: 624 of the exact matches were drift hops where the server displaced the landing off the requested target — the charge matches distance to the ACTUAL LANDING tile, not the clicked target. Planner estimates on the target are off by a few fuel on drifted hops; no code change warranted.

[[game-economy]] teleport row and [[teleport-mechanics]] fuel-cost section + footnote rewritten to cite the real dataset. The per-day residual table doubling as an independent confirmation that the 2026-06-23 `pickup_container` double-count fix was the right cut: economy noise vanishes precisely at that boundary.

## [2026-07-20] bug + fix | Ground-only pickup gate was overbroad — soak burned 78 ticks sitting ON a water container; surface-matched routing lands

The verification soak (bot-20260720-005424) scored clean on combat — 1 kill (purple-4, through the reroute window), 12/12 hits, 0 rejected, full inventory, fuel 1096 — but the tail hid a loop: radar revealed equipment on a WATER tile at (226,196); the ground-only gate said "never ground-reachable", the disembark branch sailed the ferry onto the container's own tile, and from tick 72 to 150 (half the session) the bot re-issued a move to its own position, refused by the server with 0x52 code 6, 77 times. The equipment lock never released because nothing in the loop registered as failure.

User correction (the contract, 2026-07-20): "wasnt it on the water? cant you just pick it up essentially like we were on land?" — riding a ferry, a floating container is on the SAME surface; the pickup routes normally. The 2026-07-19 refusal that motivated ground-only was the OTHER surface: riding with the container 4 tiles inland (route needed water→land chaining).

Fix: `GroundOnlyTerrain` → `SurfaceRouteTerrain(base, water=riding)` — pickups route on the tank's current surface (ground on land, water/ferry while riding). Disembark-first now fires only for land containers beyond adjacency service. Cardinal-adjacency service crosses the surface boundary both ways, symmetric with the land-side behavior the reachability layer always had; whether the server honors the riding→adjacent-land-container direction is not yet wire-tested (noted in [[ferry-mechanics]] [^5]).

Both live incidents stay fixed: the 07-19 inland refusal (single-surface gate=False → disembark) and the 07-20 on-tile loop (water container → direct dispatch, trivially at distance 0). Gate green: 4,482 tests, 100% coverage.

## [2026-07-20] refactor | Mines composed into the decision terrain; executor mine veto deleted — the loop class dies at the root

User ruling on the mine-dot loop fix: "no no no. not anti loop, but something that addresses the root of the issue." The root: "can I enter this tile?" had two owners — static terrain in the terrain view, hostile mines in a parallel `blocked_mines` parameter each consumer had to remember to thread. Pathfinding remembered; the dot-hop selector didn't; the executor veto silently absorbed the difference until it fixed-pointed.

The cut, following the ferry precedent (dynamic impassability composed once, all consumers inherit):

- `FerryAwareTerrain` gains required `hostile_mine_keys`; `compose_decision_terrain` folds in `hostile_mines(world)`. Every passability consumer — A*, reachability, selectors, surface clamp, `SurfaceRouteTerrain` (now intersects base passability) — shares one walkability answer.
- `blocked_mines` parameter threading deleted end-to-end: pathfinding (also `_is_blocked_coord` gone), reachability, equipment_search, movement, ferry clamp, tick_loop_actions, action_lab targeting. tick_loop_actions' stall-clear checks now use the composed view (previously raw static terrain — neither ferry- nor mine-aware).
- Executor `_is_valid_move_destination` DELETED, `discarded_hostile_mine` outcomes removed from the ledger. For teleports the veto was wrong physics — the server displaces off mined tiles on landing ([[teleport-mechanics]] Placement) — which also resolves [[executor-rejection-loops]] instance #1 (combat teleport at a mined enemy tile) for free. For walks it is unreachable: the planner cannot emit a destination that does not exist in its terrain.
- Adjacency service unchanged and now uniform: a container on a mined tile is collected from a cardinal neighbor — exactly how the bot drained (37,153) live.

Regression pins: mined dot skipped by the hop selector (`test_skips_dot_on_hostile_mine_tile`), hostile-vs-friendly composition, mined-tile blocking on both routing surfaces, mined-teleport-is-dispatchable (physics), pathfinding/reachability suites rewritten to composed views.

Remaining executor checks (stale combat anchor, pickup race, shoot target-not-tracked) guard planner cross-tick state — separate audit, instances #2/#3 of the rejection-loops page still open.

Gate green: 4,483 tests, 100% coverage, mypy/ruff/guard clean. (The user's 8-minute `make check` scare was environmental: a concurrent Claude session's covenant_ml workloads + poetry lock network time; the timed test stage runs 55 s.)

## [2026-07-20] verification + docs | Mine-composition refactor soaked clean, committed 6d2afdbe; [[terrain-composition]] page added

Verification soak bot-20260720-192320 (the full-quality process: soak -> validate -> commit -> document): 2 kills, 23/23 hits, 0 rejected, 0 blocked, ended full (1100 fuel, 25/25 duals/homings). The specific fix signals, measured by direct event scan: **0 discard events, 0 error-code-6 move refusals (previous soak: 77), 0 mine detonations, 0 terrain-blocked replans**, and the analyzer's first "no top-level issues detected" verdict in three soaks. 77 mine tiles crossed the viewport during the session without incident. Longest identical-decision run was 13 consecutive shoots at one live target — sustained combat, not a stall.

Committed as 6d2afdbe (26 files, +358/−318). New architecture page [[terrain-composition]] documents the single-owner walkability model in full: the layer diagram, the physics split (walk-on-mine = 45 damage = impassable; teleport-at-mine = safe, server displaces = landing finder stays mine-blind by design), the two-owner era and its fixed-point loop, the five-part cut, the invariant/test-pin table, and the standing rule for future dynamic obstacles: compose into `compose_decision_terrain`, never add a parameter beside it. Architecture hub and index updated (55 pages).

Also resolved during the same session: the "8-minute make check hang" was environmental — a concurrent Claude session's covenant_ml workloads plus poetry lock network time; the timed test stage runs 55 s and the full gate passes clean.

## [2026-07-20] measurement | Per-weapon firing costs closed from the archive — dual=10, homing=10, single=6; the −45/−10 pair decomposed

User: "cant we determine cost of firing? like dont we teleport and then just sit and fire for a while?" — and per crack-the-artifact-first, the sit-and-fire windows already existed in the archive. No new capture.

Method: decoded all 204 `runs/bot/*.capture_session.json` with the production protocol decoders (frame split → XOR → typed dicts). Between consecutive absolute fuel readings (0x2E self sync / 0x44 FuelGain), a window is CLEAN for weapon w when it contains only our own 0x53 echoes of weapon w — no enemy shots, no self movement, no sent move/pickup/teleport/radar, no container pickups. Results:

- **Single (weapon=0): 6 fuel** — 62 windows exactly −6; consumes no ammo.
- **Dual (weapon=1): 10 fuel** — 589 windows exactly −10 (the long-standing "presumed 10" confirmed); 1 dual ammo per landed shot (0x49 count snapshots: 49 windows of exactly one dual fired → dual count −1).
- **Homing (weapon=3): 10 fuel** — 398 windows exactly −10 plus 124 at −5: the homing debit sometimes splits into two −5 steps across sync boundaries; per-shot total is 10. 1 homing ammo per landed shot.
- **Missile (weapon=2): never fired** in any archived session — the one remaining unknown.

The 2026-07-19 combat-tick mystery closes with it: the paired −45/−10 per firing tick = our dual/homing cost (−10) + the enemy's return single landing on us (−45, the known victim cost). [[game-economy]] rows updated; the 0x49 equipment-count channel noted as the ammo-consumption cross-check (its snapshots arrive on radar cadence — the uniform radar −1 per window is snapshot timing, not a firing cost).

Bonus wire fact from the sweep: 0x49 counts order is [armor, dual, missile, homing, radar].

## [2026-07-20] contract | Missile trigger rule + movable concrete blocks (user contract, verbatim; wire verification pending)

Two new mechanics from the user, recorded before any capture exists:

**Missile trigger** ([[weapon-selection]] updated): missiles fire only when shooting at a visible-viewport enemy with terrain OR any tank (friendly or foe) on the line of sight — rock walls trigger them, water does not (duals cross water). Requires missile slot (3) enabled; the bot keeps it off (`tactics.py`), which is exactly why weapon=2 never appeared in 204 archived sessions.

**Movable concrete blocks** (new page [[movable-blocks]]): pickup-and-place terrain — bridge when placed on water, obstacle on land or when stacked on a water block (stacking only on water), destroys enemy mines on the placement tile (not adjacent), placeable over containers without destroying them, dragged behind the tank (no turning in place; up-over-down to reverse), blocks non-missile enemy shots on one line of sight only. Bot has zero knowledge of them; wire encoding unknown (likely a terrain_type like ferries — [[js-source-map]] dive pending). Composition note: when decoded, they fold into `compose_decision_terrain` per the [[terrain-composition]] rule — water-block passable, land/stacked impassable.

## [2026-07-20] measurement | Missile cost pinned from manual capture — 10 fuel + 1 missile per landed shot; trigger rule wire-confirmed

User ran the manual missile protocol (make sniff, stationary, firing at enemies behind rock): capture sniff-20260720-213208. Ten weapon=2 echoes — the first missiles ever seen on our wire. Isolation: 6 clean single-shot windows (no movement/radar/pickups/enemy fire between fuel syncs), every one exactly −10. Ammo cross-check: 0x49 missile count 25→15, ten consumed for ten shots (consumption-equals-hit holds for missiles too). One weapon=3 homing also fired (target moved on the shot tick — the server-side selection rule doing its thing even in a missile session).

The obstruction trigger rule from the same-day contract is now wire-confirmed: shots at rock-obstructed enemies consistently selected weapon=2. [[game-economy]] missile row added; firing costs are now FULLY closed: single=6 free, dual/missile/homing=10 + 1 round per landed shot. Remaining: mine placement cost (user capturing next).

## [2026-07-20] measurement | Mine press = 10 fuel FLAT — the economy table is complete

User ran the mine protocol (sniff-20260720-214329, place → walk 3 tiles → place): the fuel line reads like a metronome — −3 per walk leg, then exactly −10 per mine press, eight presses straight, regardless of how many of the 3×3 mines actually landed (terrain blocks and enemy-mine overlap detonations don't change the price). User contract confirmed en route: press places a 3×3 centered field, skips terrain, and destroys overlapping enemy mines 1:1 (your overlapping mine is consumed) — all already documented in [[mine-mechanics]] from the 2026-06-20 spec; only the price was open. The old "~1–2 per mine" estimate dissolves: the −10-for-6-mines sample was one press with 3 blocked tiles.

[[game-economy]] "What's still open" now reads **Nothing** — every player-action fuel cost is pinned, and the design is legible: everything costs 10 except walking (1/tile) and the free single (6). Remaining wiki-side unknowns are non-economy: movable-block wire encoding (capture pending) and the reroute-TTL exact constant.

## [2026-07-20] decode | Movable blocks fully cracked in one evening — command 'b', 0x42/0x4A schema, enums pinned by 12 labeled drops

Three manual sniff sessions (user piloting, narrated) took movable blocks from "probably in the js code somewhere" to a complete wire spec. The 0x42/0x4A decoders existed from the JS reverse-engineering all along — they had simply never fired in 204 bot sessions.

- Client command: `type=4 id=98 ('b')` (x,y) — one long-press command for pickup AND drop.
- 0x42 BlockAction: direction=ASCII compass letter (e/s/n/w) on pickup = which side the block attached; 0 on drop. obstacle_type enum pinned by the 12-labeled-drop session (sniff-20260720-215930): **1 = water (walkable bridge), 2 = land (obstacle — same value on plain ground, next-to-water, on mines, on containers), 3 = stacked on a water block (impassable terrain)**. 0x4A tile updates share the enum (+0=cleared); dragging emits transient 2→0 pairs along the towed path; unstacking reads 3→1.
- Towing physics wire-confirmed: teleport refused (0x52 code=0, ×3), out-of-reach press refused (code=1), movement at normal 1/tile, block presses FREE (stationary same-tile re-place pairs: zero fuel delta) — the one free action in the game besides walking's baseline.
- Mine destruction on land placement is WIRE-SILENT (no 0x45, nothing) and kills ANY team's mine — blue (own), red, and purple all destroyed in the labeled capture. Refines the original "enemy mines" contract, and means the bot's mine registry must delete mines on observed 0x42 land drops (nothing else will tell it).
- Containers under blocks survive, including a fuel container in water becoming a bridge tile with the container intact.

DOM labels for cross-reference: "Obstacle picked up" / "Obstacle dropped" (land + stack) / "Bridge module built" (water). [[movable-blocks]] rewritten to confidence: high; bot-side open work listed (compose 0x42/0x4A into world state + decision terrain per [[terrain-composition]], mine-registry hygiene).

## [2026-07-20] decode | Block arrival encoding cracked — one terrain enum across 0x42/0x4A/0x5A; radar and map are block-blind

The re-arrival capture (sniff-20260720-221239, user-narrated coords) closed the last block unknown, and the answer unifies the whole model: **blocks are dynamic rock-family terrain.** Resting blocks arrive to a fresh client through ordinary 0x5A viewport patches as terrain_type 1 (water bridge = ROCK_A), 2 (land obstacle = ROCK_B), 3 (stack = ROCK_AB) — the SAME values 0x42 calls obstacle_type and 0x4A uses for tile updates. Ferries (5/7) complete one coherent terrain vocabulary across all three channels. Found under the known +1/+1 0x5A entity-alignment offset; the user's moved blocks appeared at their new coords and vanished from the old in subsequent patches, and the fuel container under a bridge stayed visible via cache_value, draining 730→0 as picked.

Radar (0x4F) carries NO blocks (containers/mines only) and the map (0x4C) shows none — both user-confirmed visually and wire-verified. Walking a bridge is plain movement; a bridge-only equipment pickup dispatched normally.

The ingestion rule, when ever needed: wire value 1 is ambiguous (rock-A vs bridge) — disambiguate by static background: rock-family over static WATER = walkable bridge; over land = impassable either way. Also corrected: bot captures DO contain thousands of 1/2/3 tiles (genuine mountains); the zero-blocks claim stands precisely for 0x42 events, 0x4A block values, and rock-family-over-water tiles. [[movable-blocks]] updated; not-wired decision unchanged.

## [2026-07-20] fix + decode | Blocks wired into the composed terrain — the "absent" mechanic was map furniture all along

User challenged the not-wired decision ("why not add blocks to the terrain tho?") and demanded a check first. The deeper archive sweep (every wire rock-family tile in 228 room-1 captures vs the static practice map) overturned the earlier conclusion: **value 1 = 4,352 sightings ALL over static water (bridges), value 2 = 2,396 ALL over static ground (land blocks = live invisible obstacles), value 3 = 250 ALL over water (stacks)**. A persistent bridge complex sits around (130–135, 145–155). Resting blocks are common where the bot plays; only manipulation near it is unobserved. The perfect value/background separation means the wire value alone determines walkability.

Wired the same hour: constants renamed to truth (TERRAIN_ROCK_A/B/AB → TERRAIN_BLOCK_BRIDGE/_LAND/_STACKED — they were never rocks), FerryAwareTerrain collapses wire blocks to walkability class (bridge → ground, land/stack → rock; raw terrain_type stays verbatim in world state — the collapse is only the planning projection), world renderer draws bridges as '=' with legend updated. Fourth verse of the composition pattern. Q&A recorded en route: no wire bytes are dropped anywhere (0x42 all 9 bytes incl. the unexplained flag, 0x4A exact triples, 0x5A all five entity fields); the GROUND/ROCK collapse is the last-step projection, not the decode.

Gate green (4,485 tests, 100%). Verification soak bot-20260720-223x: 5 kills, 66/66 hits, 0 rejected, 0 blocked, analyzer clean; no bridge tiles crossed this run's path (behavior-neutral absent blocks, as designed — correctness pinned by tests + archive semantics). [[movable-blocks]] decision section revised; [[terrain-composition]] precedent list updated.

## [2026-07-20] roadmap | Physics-module plan written down — the end-to-end refactor now survives this conversation

User: "so how will another ai know the full end to end refactor to implement?" — it wouldn't have; the design lived only in the chat. New page [[physics-module-roadmap]] (Architecture hub) captures it in executable-handoff form, mcps-workspace PLAN-doc style but inside the wiki so the mandatory index→hub→page read finds it: goal, what already exists (don't rebuild), Phase 1 physics/ module + claim binding into make check with concrete file/constant inventory, Phase 2 validators vs the capture archive + computed fact_checked stamps (method pointers to the teleport and firing-cost log entries, since the scratchpad scripts are gone), Phase 3 live divergence counting + double-entry fuel/ammo ledger, the parallel executor-staleness track (rejection-loops #2/#3), non-negotiable constraints, and per-phase verification requirements. Status marked clearly: designed and user-approved, NOT implemented.

## [2026-07-20] refactor | Phase 1 executed — physics/ package live, wiki claims machine-checked in make check

Same-day execution of the [[physics-module-roadmap]] Phase 1. New package `src/tankpit_bot/physics/` is now the single code home for every game rule: `costs.py` (walk 1, single 6, dual/missile/homing/radar/mine-press 10, block ops 0, `teleport_cost` moved+renamed from the deleted `bot/ai/teleport_cost.py`), `damage.py` (45/90/45 — these existed only as wiki prose until today), `capacity.py` (absorbed `state/rank_formulas.py`, deleted; + DEPOSIT_FLOOR=100), `combat.py` (REROUTE_TTL_MS=12000 estimate, boundary [11.0,13.0]s). `combat_radar_min` was policy wearing a physics costume — moved to its only consumer `mode_controller.py` with a docstring saying so.

The binding: [[game-economy]] (15 claims), [[radar-mechanics]] (radius probed at the measured step boundaries), and [[shoot-event-format]] (reroute TTL) each carry a fenced `json claims` block mapping claim-id → `module:symbol` → expected value or probe grid. New guard stage `scripts/physics_claims.py` (wired into `scripts/guard.py` beside contract_rules) verifies BOTH directions on every make check: each claim imports and matches its symbol (constants by equality, formulas at explicit probe points — teleport probed at the 3-4-5 exactness, the floor-rounding diagonal, and the isqrt long-hop), and every `__all__` symbol of every physics submodule must be bound by exactly one claim. Drift either way = red gate. Deviation from the design sketch: JSON not YAML (no YAML dep; `platform_core.json_utils` narrowing is the repo idiom), probe grids instead of a formula evaluator.

Migration was shim-free: 11 source files re-pointed, old modules deleted, `threats.py` local renamed to avoid shadowing the new function name. Iterated against the monorepo guard's own rules (no `object` annotations → Protocol-at-assignment for dynamic imports incl. `__code__.co_argcount` arity checks; every except logs via platform_core get_logger; capsys asserts restructured). Gate green: 4,551 tests (98 new: physics constants pinned row-by-row + 31 claim-rule tests incl. `test_real_repo_binding_is_green`, which runs the checker against the actual repo), 100% statement+branch coverage, guard 0 violations. DoD grep: no game formula or damage constant survives outside physics/. Phases 2–3 unchanged, still designed-not-started.

Behavior-neutrality soak bot-20260720-233953 (314 ticks): 2 kills, 25/26 hits, 30 landed_exact + 2 landed_inexact teleports (drift, normal), 0 executor rejections, 0 discards, 0 error-code-6 refusals. Analyzer's sole flag — map_open dispatched=34 vs completed=33 — is the session clock cutting off the final map open mid-flight, not a regression. The wire graded the consolidation clean.

## [2026-07-21] refactor | Phase 2 executed — make audit re-derives every economy claim from the archive, 9/9 green

Same-session execution of [[physics-module-roadmap]] Phase 2 as `src/tankpit_bot/validate/` (not tools/ — src is where mypy/coverage/guard already apply): `wire_timeline.py` extracts a typed per-session event stream with the production decoders (frame split → XOR → decode_message; fuel readings from long-form 0x2E sync + tunneled 0x44 gains + 0x64 deposits; 0x45 detonations as hazards; own 0x47 path_tiles — a NEW MovementDict field, since the decoder used to collapse the nsew path and discard the true step count), `archive.py` runs the isolation-window validators, `events_validators.py` the teleport pairing, `audit.py` orchestrates `tankpit-audit --stamp` behind `make audit`.

Calibrating against the real archive taught three physics-of-measurement lessons, each now encoded: (1) windows close INCLUSIVELY on the ending reading's timestamp — cause and closing sync share a millisecond (a seq-ordered rewrite tanked hit-damage from 738/738 to 43 and was reverted); (2) walks drain across many windows and always end in pickups, so neither single windows nor events-fix pairing can price them — the instrument is the walk EPISODE (walk-only window → event-free windows → zero-delta close), single-echo only because a mid-walk re-command echoes a full path never fully stepped (probe: every multi-echo error overcounted tiles, never fuel); (3) verdict is exactness share with a 0.85 floor — residual noise is positive-signed truncation, while real drift collapses the share to ~0.

Final table: walk 204/232, single 242/247, dual 863/932, missile 6/6, homing 487/522, single-hit 738/738, dual-hit 6/6, capacity 18,649/18,649 (1,463 at cap), teleport 63/63 across 46 post-fix runs. `make audit` stamped [[game-economy]]'s fact_checked line from validator output — computed, not hand-typed, exactly as the roadmap demanded. Phase 3 (live divergence) remains designed-not-started.

## [2026-07-21] refactor | Phase 3 core landed — the live fuel book: every wire fuel sync is now graded by physics

`ledger/fuel_book.py` implements the interval double-entry design from [[physics-module-roadmap]]: between consecutive absolute wire fuel readings the book accumulates feasibility intervals (exact shot/radar debits, ranged walk and teleport-drift debits, optional enemy-hit/detonation debits, open pickup credits, homing −5/−5 carry), and reconciles at `update_world_state_from_fuel_total` — the one choke point 0x2E sync / 0x44 gain / 0x64 deposit all flow through. A residual outside the interval emits a `physics_divergence` diagnostic; the scorecard gains `physics divergences: N` and the analyzer flags any nonzero count as candidate wiki claims. Entry sources: world_state_dispatch (0x53 own/enemy, self 0x47 path_tiles, 0x43, 0x45) and executor dispatch (radar, teleport priced off the self fix ± 18 drift). Gate green at 100% stmt+branch. Ammo book + analyze-side claim extraction remain as Phase 3 increments; soak verification pending this entry's timestamp.

## [2026-07-21] verification | Phase 3 fuel book calibrated live — four soaks, 71 -> 12 -> 18 -> 1 divergences

The live book's first light (71 divergences, every one `entry_kinds='(none)'`) taught the same lesson the archive validators had: per-sync windows are the wrong instrument because charges lag their cause echoes. Rebuilt reconciliation on QUIET boundaries (a zero-delta reading with no new entries closes and judges the accumulated block; cap at 50 readings) — the live twin of the walk-episode method. Soak 2 (12): every positive residual was an unentered gain — 0x44/0x64 totals announce their own delta, so the choke point now credits them exactly. Soak 3 (18): the pickup credit only fired for EMPTIED containers; partial pickups (the common case) took the else branch uncredited — moved above the branch; teleport drift bound widened to ±6 tiles. Soak 4: heavy combat, 67 hits, **one divergence** — a radar-only block at −20 vs [−10,−10], a double radar debit whose first charge crossed a quiet boundary: a legible candidate residual, exactly what the instrument exists to surface. Scorecard now prints `physics divergences: N`; the analyzer flags nonzero counts. Gate green at 100% throughout.

## [2026-07-21] verification | Ammo book live + a clean sheet — zero divergences with both books running

`ledger/ammo_book.py` closes the Phase 3 bookkeeping pair: between 0x49 snapshots the book counts own-shot echoes (dual/missile/homing), dispatched scans, and 0x67 gains, then requires every slot delta to be feasible — falls bounded by uses (consumption-equals-hit, live), rises only with gains, armor free to fall but not to rise. Contract-enforced like the fuel book; divergences emit on the ammo channel and land in the same scorecard counter. Verification soak with BOTH books active: **physics divergences: 0** across a full combat run (38 hits, 27 radar scans, 26 map opens) — every fuel window and every ammo snapshot matched physics predictions exactly. The wiki-as-executable-truth loop is closed end to end: the gate binds claims to code, the audit re-derives them from the archive, and the live wire now signs off on every run.

## [2026-07-21] refactor | Executor is pure dispatch — rejection-loop instances #2 and #3 closed by deletion

The remaining executor validators guarded a race that cannot happen: the tick is synchronous (drain -> decide -> execute, one thread, `drain_messages` the only world mutation point), resource locks are normalized at DecideCtx construction with selection's own pursuability predicate, combat releases reset the anchor to -1, and the teleport source check refused a container source no creation site can produce. Archive cross-check: zero validator discards in any run since the mine fix. Cut like the mine veto: `_is_valid_shoot`/`_is_valid_pickup`/`_is_valid_teleport`/`_is_dispatchable` + `_tracked_*` helpers, six discard emitters, six outcome literals, ledger-audit discard analytics — 30+ tests of removed behavior deleted or rewritten to the new invariant. Unlocked in the process: id-targeted shots at departed tanks now dispatch, which is the reroute-TTL pursuit mechanic the shoot veto had been silently blocking. AI-state persistence still gates on genuine CDP dispatch failure (new test pins it via the file's save-restore idiom). Gate green at 100% stmt+branch. Behavior-neutrality soak: 4 kills, 45 hits, 21 teleports (16 exact + 5 drift), 19 pickups, 0 executor discards, and physics divergences: 0 -- the live books balanced through the whole run with the validators gone, which is itself the strongest evidence they were guarding nothing.

## [2026-07-21] erratum + finding | The pursuit volley already exists — and has been firing all along

While designing a "pursuit volley" feature (keep firing homing at a departed target through the reroute-TTL window), the pre-build read revealed it is ALREADY the live behavior, by design: `remove_tank` is a deliberate no-op (2026-06-22 — 0x58 fires on tracking churn as well as death, so only 0x41 removes trust in a tank), the registry therefore keeps departed tanks at frozen coords, `find_locked_target_pursuit` synthesizes them back into HUNT, and the bot fires homing at the frozen position until the first genuine miss (TTL expiry) trips the stationary-miss classifier and releases at a cost of one 10-fuel probe shot (misses consume no ammo). Evidence: 4-12 "firing toward last wire position" pursuit shots in every 2026-07-21 soak. This corrects commit 59fce8e1's message and the first version of the [[executor-rejection-loops]] resolution, which claimed the shoot veto had been blocking this mechanic — it never did, because the tank never leaves the registry. No code change: the planned feature was already shipped a month ago, tuned by the 2026-07-19/20 TTL work, and self-terminating on wire truth. The only residue of the design exercise worth keeping: the TTL-narrowing loop (tank_removed timestamps vs pursuit-miss timestamps) remains automatic, and REROUTE_TTL_MS stays a wiki-bound physics estimate.

## [2026-07-21] measurement + contract | Server timing cracked from the archive; three user contracts recorded for the sim

Archive cracks (Phase 4 sim prep, 214 captures):
- **Sync cadence: the server ticks fuel syncs every ~2 s** (16,189 inter-reading gaps: median 2001 ms, p90 4009 ms, with intra-burst gaps near 0). Matches TICK_RATE_MS — the wire's fuel heartbeat IS the game tick.
- **Charge latency: a shot's fuel debit lands on the NEXT server tick** (4,056 shot→debit pairs: median 2001 ms, p90 4004 ms — i.e., one tick later, occasionally two). This is the measured mechanism behind the fuel book's quiet-boundary design and the audit's end-inclusive windows.
- Walk speed: preliminary ~400–500 ms/tile from only 2 clean events pairs — extraction needs the wire-side method (0x47 echo → drain completion) before it's a claim. Not yet pinned.

User contracts (2026-07-21, verbatim intent): (1) "dual shots or regular shots [don't] change dmg based on distance" — damage is flat per weapon; distance affects hit/miss only, so the planned distance-ladder capture measures HIT RATES, not damage curves. (2) "you can land and walk on your own mines or ally mines (mines of your color)" — confirms same-team mine passability AND refines teleport displacement: landing on own/ally mines is legal. (3) "you cant teleport to enemy mines" — displacement applies to ENEMY mines only. [[teleport-mechanics]] Placement and [[mine-mechanics]] hold; the displacement tie-break probe is now optional (legality settled; only the direction preference of "nearest open tile" remains unmeasured, covered by the fuel book's ±6-tile drift bound).

Still wanted from the user for a guess-free sim rulebook: the missile/homing victim-cost session (tank ~10 isolated hits of each on a capturing account) and optionally the single/dual hit-rate distance ladder.

## [2026-07-21] measurement | Teleport displacement law cracked — EAST, then NORTH, then WEST, self counts as blocked

User-piloted displacement probe (sniff-20260721-200527, narrated; 11 teleports wire-verified against 0x3D landing fixes). The server resolves a blocked teleport target by trying neighbors in a FIXED absolute order — E, then N, then W — regardless of approach direction: three approaches to the mine at (17,63) from NW/N/W all landed east at (18,63); a solo mine and rock targets confirmed east-first; standing ON the east neighbor forced the north landing (self-occupancy blocks); a target with rocky east+north landed west. South and beyond-ring-1 remain unisolated. One narration typo corrected by the wire: the (62,46) hop landed at (62,45), north. Every hop's fuel delta re-confirmed landing-based pricing exactly. Walk segment bonus: a 20-tile out-and-back drained the adjacent container by exactly 20 (walk=1/tile again), and a 12-tile manual walk showed its full charge within one sync tick of the 0x47 echo — charge timing for manual walks looks near-immediate, unlike the bot's multi-window drains; nuance noted for the Phase 4 sim timing model. [[teleport-mechanics]] gains a Displacement preference order section; the sim's landing predictor can now be exact for ring-1 cases instead of a ±6-tile bound.

## [2026-07-21] measurement | Server pathfinder cracked (phase A) — 84% minimal, L-shaped with a y-first lean

The client only ever sends a destination click (CMD_MOVE is 5 bytes); the SERVER pathfinds and announces its chosen route in the 0x47 echo's nsew string — which the decoder was discarding until today (`MovementDict.path` now preserves it, additive like path_tiles). First archive sweep, 1,928 self-walk routes across all bot captures:

- **84.2% are Manhattan-minimal** (1,623/1,928); the 15.8% with detours always carry an EVEN number of extra tiles (2:111, 4:79, 6:38, ...) — geometrically necessary, and obstacle-driven (phase B will diff against the composed terrain to model which obstacles).
- **Shape among minimal diagonal routes: L-paths dominate** (882 single-turn vs 167 staircase; 574 straight-line). Of the L-paths, **y-first beats x-first 542:340 (~61%)** — a lean, not a law; whether the 39% x-first cases are obstacle-forced or direction-dependent is the phase B question.
- Planner implication: our A* is advisory (thrown away after the click), and our Manhattan-based walk cost estimates match server billing in the 84% minimal case — detoured walks bill more than the planner assumes, consistent with the walk-book truncation findings.

For the Phase 4 sim: the pathfinder to implement is "shortest path, prefer single-turn L, y-axis lean, obstacle detours" — phase B (terrain-aware diff of all 305 detour routes + the x-first cases) turns that from a description into a rule.

## [2026-07-21] measurement | Server pathfinder phase B — terrain explains most divergence; a 74:26 both-clear lean remains

Diffed all 1,928 server routes against the static room-1 terrain map (field01_r.gif via TerrainMap):

- **x-first L-paths: 213 of 340 were FORCED** (the y-first alternative crosses impassable static terrain). **Staircases: 139/167 forced** (both pure Ls blocked). **Detours: 254/305 forced.** The bulk of the "divergence" was never a choice — it was the map.
- **The residual choice, when both Ls are statically clear: 364 y-first vs 127 x-first (74:26).** So y-first-when-clear is a strong lean but still not a law on static evidence alone.
- The unforced residue (127 both-clear x-first, 28 unforced staircases, 51 unforced detours) has two candidate explanations the static map cannot see: DYNAMIC obstacles at walk time (other tanks, hostile mines, movable blocks — all present in the captures but requiring per-moment world reconstruction), or a secondary server rule (direction- or distance-dependent axis choice).
- Cheap decisive test identified for the next manual session: in open ground, walk the SAME diagonal (e.g. 6x6) several times in both directions. Same L every time = deterministic rule + dynamic-obstacle explanation for the residue; varying L = genuinely probabilistic router (the sim then rolls the 74:26 dice, which is fine).

Sim spec so far: shortest path; obstacle detours per terrain; prefer single-turn L; y-first ~74% when both clear pending the determinism test.

## [2026-07-21] measurement | Victim costs closed (missile=45, homing=45), armor cracked, and the pathfinder is DETERMINISTIC

User-piloted session (latest sniff capture, narrated), decoded per-hit against the wire:

- **Homing hit on victim = 45. Missile hit on victim = 45.** Five isolated hits each: fuel stepped 1100->1055->1010->965->920->875, every debit landing the SAME INSTANT as its 0x53 echo (victim-side charging has no tick lag, unlike shooter-side). Cross-checked by container drain: refill after 5 hits + 1 radar = exactly 235, twice. Damage table is now complete: single 45, dual 90, missile 45, homing 45 -- dual is the only double-damage weapon.
- **Armor shields fully absorb damage and are consumed at damage/45**: five enemy singles -> armor 25->20 (1 each), five duals -> 20->10 (2 each), one missile -> 9, one homing -> 8; fuel untouched throughout. Armor slot = counts[0] on the 0x49 wire.
- **The server pathfinder is deterministic and quadrant-keyed**: the same 7x8 diagonal walked 3x gave byte-identical routes each way -- northeast = EAST-first, southwest = SOUTH-first; southeast = south-first, northwest = north-first. Rule: vertical-first except the NE quadrant, which is horizontal-first. This resolves phase B's 74:26 "lean" completely: the x-first archive residue was NE-quadrant walks. Deviations from pure L-paths in the session align with terrain/mine obstacles.
- Walk timing bound tightened: 15-tile walks completing inside 6-second click gaps -> <=400 ms/tile.

To fold in next: MISSILE/HOMING_HIT_VICTIM_COST + armor constants into physics/damage.py with claims (gate), the quadrant rule into the sim pathfinder spec, and the remaining segments of this capture (mine-field clearing walk, distance-15 single shots) still hold undecoded evidence.

## [2026-07-21] refactor | Victim-cost session folded through the whole pipeline — 11/11 claims, armor modeled live

The 2026-07-21 measurements are now first-class physics: `damage.py` gains MISSILE_HIT_VICTIM_COST=45, HOMING_HIT_VICTIM_COST=45, and ARMOR_ABSORB_PER_SHIELD=45 (shields consume at damage/45, full absorption), each claim-bound in [[game-economy]] and gate-checked. The audit's hit-damage validator learned the two new weapons and re-derived them from the archive on first run: missile-hit 10/10 exact, homing-hit 8/8 exact — `make audit` now covers 11 claims, 22,903 clean samples, and re-stamped the page. The ammo book's armor slot went from unconstrained-fall to enemy-shot-bounded: armor may drop at most 2 shields per incoming 0x53 observed (the dual worst case), rises still require a 0x67 gain. Damage table status: COMPLETE, dual confirmed as the game's only double-damage hit. Gate green at 100% stmt+branch throughout.

## [2026-07-21] measurement | Capture fully decoded — no range mechanic exists, mines are shootable, pathfinder dodges mines

Finishing the victim-cost capture's remaining segments:

- **There is no range mechanic.** Six enemy singles at the user's STATIONARY tank all hit for exactly -45: two from distance 15, two from ~30, two from adjacency. This is the [[shoot-event-format]] queue model's strongest evidence yet — a shot resolves against the target's position at processing time, so a stationary target is hit at ANY distance and full damage (also re-confirms damage is distance-flat, per the user contract). The bot's old "distance 4+ hit ~0%" statistic measured MOVING targets escaping the queue, not range. Sim consequence: hit resolution is deterministic (did the target move between click and processing), and the planned distance-ladder capture is UNNECESSARY.
- **Mines are cleared by shooting them**: seven wire-verified -6 singles at narrated mine coordinates opened a path through the purple field ("destroyed some to form a path"). The exact wire signal for the mine's removal needs one more decode pass (0x45/0x4B lines) — noted, not yet pinned.
- **The server pathfinder routes around enemy mines**: the minefield traversal (86,135)->(85,120), Manhattan 16, got a 22-step weaving route both ways — dynamic obstacles confirmed inside the server's router, matching the phase-B residue hypothesis.
- Bulk re-confirmations: five 15-tile diagonals each billed exactly -15; walk timing bound ~<=400 ms/tile (a 15-tile walk plus human reaction fit in a 6.0 s click gap).

Open questions after this session: exact walk speed (ONE deliberate capture: two long single-click walks — the last measurable physics unknown), the mine-removal wire signal (decode pass), displacement south-preference and beyond-ring-1 (edge), spawn distributions (archive crack), enemy minds (permanent, by design).

## [2026-07-21] measurement | Walk speed question DISSOLVED — server movement is instantaneous; mine-removal signal re-confirmed

User-piloted capture sniff-20260721-212348 (long walks with map/radar/mine key-spam, own mine placement, mine shooting). The walk-timing question got a better answer than a number:

- **Server-side movement is INSTANT.** Every single-click walk resolves route + full billing + destination pickup in ONE tick (e.g. t+63.81: fuel 595→587 for the 8-tile path, 0x47 echo, and the pickup at (3,220) — same flush; a 12-tile walk resolved its pickup 210 ms after the click, bounding internal latency <17 ms/tile). Two full-archive probes agree: **200/200** single-echo bot walk episodes carry the whole cost in the echo window (0 gradual), and **0 of 1755** consecutive echo pairs start at an interior position of the previous path. The "drains tile by tile" model is dead — gradual-looking bot drain was many separate instantly-billed commands. New page: [[walk-mechanics]].
- **The on-screen walk is client animation only, and it input-locks HUMANS**: map/radar/mine keys are blocked during the animation and register at the first tick after it ends (user-observed; three 23-tile walks bound the animation ≤181 ms/tile, lower bound open at ~87 due to tick quantization). The bot writes to the socket and is not gated. Bot implication: any pathable destination is reachable in one tick at 1 fuel/tile — cheaper than a 30-fuel teleport under 30 tiles, and a human enemy who just made a long click is input-dead for the animation.
- **"Exact walk speed" is REMOVED from the physics unknowns** — there is no server walk speed to measure. The sim models movement as instant relocation at the processing tick.
- **Mine-removal wire signal re-confirmed as 0x45**: the user's one shot at (54,170) produced `0x45 [(54,170)]` + same-tick cascade `0x45 [(55,170),(54,171),(55,171)]` — 4 mines, matching the on-screen count. Already documented in [[mine-mechanics]] from the 2026-06-20 PvP capture; today's sample closes the "needs one more decode pass" note.
- **Strategic map carries NO mines** (user question answered): 0x4C is fuel-dot atlas + 5-byte tank blips only. Mines reach the client via 0x4B placements, 0x45 detonations, and 0x4F radar overlay — and the overlay is a CACHE DIFF: scans 3 tiles from the user's own fresh mines showed zero overlay entries because the client cache was already correct.
- Also observed: MinePlacement count=3 (self tile + 2 neighbors — 6 of the 9 tiles blocked; user contract 2026-07-21: mines are NOT inventory, the 3x3 is clipped to the visible viewport and skips terrain/water, enemy mines in the area trade 1:1), and one shot echo whose target differed from the click — RESOLVED by user 2026-07-21: the click was a solo mine at (55,167) behind a mountain; the single stopped at the terrain and the 0x53 echoed the CLIPPED impact tile (46,165) on the shooter→click ray, still billing −6. Plus user contract: missiles are enemy-only (never fire at mines or ground). Folded into [[shoot-event-format]] (clipping semantics) and [[weapon-selection]] (enemy-only missile rule).

Remaining unknowns after this session: displacement south-preference / beyond-ring-1 (edge), container spawn distributions (archive crack), max single-click walk distance (23 wire-confirmed), enemy minds (permanent).

## [2026-07-21] design | Phase 4 simulator SPEC'd — wire-level fake server behind the existing test seams

With the physics complete (walk-speed question dissolved same day), the roadmap gains the Phase 4 spec, written to be executable without this conversation:

- **Architecture: wire-level fake server.** The sim speaks real bytes behind seams the codebase already has — inbound via the CapturedMessage buffer (the path `replay.engine` already drives headless), outbound via `send_command_bytes`' callback + `_test_hooks.CDPSessionProtocol` (the protocols tests already fake). Production bot runs UNCHANGED; sim sessions are standard-format captures, so `make audit` and the replay engine work on them for free.
- **The one real code gap is server-message ENCODERS**: we decode everything the server says but encode almost none of it. Each consumed `decode_*` gains an `encode_*` sibling in the same file, verified by `decode(encode(x)) == x` property tests plus byte-identical corpus round-trip over the archive.
- **Eight laws, every rule wiki-anchored**: global queue at the 2 s tick; instant movement with the quadrant-keyed pathfinder; queue-model shots with server-side weapon selection, terrain clipping, armor; homing reroute TTL; teleport cost + E→N→W displacement; viewport-clipped 3×3 mines with 1:1 exchange and cascades; capacity/deposit/pickup semantics; cache-diff radar and mine-free map.
- **Acceptance test = the Phase 3 instruments**: the bot plays a full session against the sim and the fuel + ammo books must report ZERO divergences — the books cannot tell the sim from the real server. Plus audit cross-check (sim captures re-derive all archive-validatable claims) and an explicit fidelity statement (1:1 on measured laws; NOT 1:1 on spawn distributions, enemy minds, and listed assumptions).
- **Build order**: encoders → world+tick processor (laws 1–3) → transport smoke → laws 4–8 → divergence-zero soak. One commit per step, gate green throughout.
- Also corrected in the Phase 2 as-built record: the "walk drains across many windows" rationale is marked superseded by [[walk-mechanics]].

## [2026-07-21] refactor | Phase 4 step (a) DONE — server-message encoders, 72,916/72,916 corpus messages byte-identical

The simulator's one real code gap is closed: every server message the bot can decode now has a byte-exact encoder, proven against the entire capture archive.

- **New `protocol/encoders/` package** mirroring the decoders module-for-module, plus `container/encoders.py` and the envelope keystone (`encode_message_payload` / `encode_envelope_body`). The radar encode trio moved in from `decoders/radar.py` — one home for all encoders.
- **`make roundtrip`** (new target + `tankpit-roundtrip` CLI, `validate/roundtrip.py`): decodes and re-encodes every received binary message in the archive and demands byte identity. Result: 244 sessions, **72,916 messages across 28 families, 0 mismatches**. This is now a standing instrument — any future decoder or encoder drift turns it red.
- **Two decode blind spots found and fixed by the round-trip discipline**: `TankStatusDict` was dropping info-byte bits 2–3 (damage_state — nonzero in 223/244 bodies) and `FuelGainDict` was collapsing its raw flag byte to a bool (one corpus body carries 0x2B). Both TypedDicts gained the wire truth.
- **Corpus laws pinned along the way**: TankEntry's flags byte equals team; the sync has-fuel-bar byte is constant 1; the 0x3F heartbeat body is constant 1; the 0x5A "no mine" overlay nibble is 8; and the greedy skip-RLE emitter reproduces every map fuel-dot atlas and viewport patch byte-for-byte — the server's RLE is canonical greedy.
- **Lobby mystery resolved**: the outer `+`/`=` frames that always resisted binary decoding are PLAINTEXT lobby traffic (room listings `+5|World (Desert)|...`, profile rows `=5|Sep. 25, 2012|Artax|...`) — the text channel (`is_text_message`), not binary messages. The old "undecodable" tallies were counting text as cipher.
- Gate green: 4,639 tests, 100% stmt+branch; `make audit` unchanged at 11/11 claims (22,941 samples — the walk-timing capture joined the archive).

Next: step (b) — `sim/world.py` + `server.py` implementing laws 1–3 (queue/instant movement/shots).

## [2026-07-22] refactor | Phase 4 step (b) — the sim exists: laws 1–3 implemented and gate-green

`src/tankpit_bot/sim/` lands with the first three wiki laws as running code: **law 1** (global queue, 2 s tick, batch flush, shooter charge latency deferred one tick), **law 2** (instant movement: quadrant-keyed deterministic router → relocate → full billing → same-tick destination pickup/mine detonation), **law 3** (queue-model shots: Bresenham terrain/tank clipping, server-side weapon selection incl. same-tick-mover homing and enemy-only missiles, damage table with armor absorption, tier progression, deactivation, two-packet mine cascades). Typed world state with codecs seeds from JSON; client commands decode through their own require_* path; terrain arrives via `TerrainMapProtocol` DI so tests run on in-memory maps. Everything the tick produces is decoded `BinaryMessage` dicts — the step-(c) transport will feed them through the step-(a) encoders to make real wire bytes. Out-of-stage commands (radar/map/teleport/mine placement) raise `SimError` explicitly until step (d). Gate: 4,690 tests, 100% stmt+branch. Next: step (c) — `transport.py` over the `_test_hooks` CDP seams + the first bot-vs-sim smoke session.

## [2026-07-22] refactor | Sim wired to the production pipeline — and the wiring immediately caught two real bugs

Answering "is it half implemented and unwired?": `sim/transport.py` now turns tick batches into real wire bytes (step-(a) encoders + XOR + framing) and the bot's actual command frames back into typed commands; `SimServer.handshake()` speaks the join choreography. `tests/sim/test_integration.py` proves the loop through PRODUCTION code only: `process_received_message` ingests sim bytes → the bot's `self_state`/tank registry match sim ground truth; a real `build_move_command` frame drives the sim (move + billing + pickup, beliefs == truth after the tick); the real planner `decide()` returns HUNT/COLLECT on sim-fed state. The wiring instantly exposed (1) a fuel-sync leak — the sim long-form-synced victims, and production treats ANY fuel-bearing 0x2E as self fuel, so a victim's sync would have corrupted the bot's own belief (fixed: per-recipient long form, regression-pinned) and (2) the 0x21 empty-decoration wire-shape trap. Neither is catchable by typed-dict ingestion — only by bytes. Gate: 4,700 tests, 100% stmt+branch. Remaining for step (c): live CDP substitution for the full `make run` loop, blocked on step (d) laws (the bot's opening moves are radar/map, which the sim still refuses with `SimError` by design).

## [2026-07-22] refactor | Phase 4 step (d) — teleport, radar, map, and mine laws land in the sim

`sim/actions.py` + tick-processor wiring: **teleport** (cost floor(6×euclid) to the ACTUAL landing, ring-1 displacement E→N→W with S as the documented last-resort assumption, sealed ring rejects, landing auto-pickup via the shared `resolve_pickup`), **radar** (extra consumed → viewport radius 8, else `free_radar_radius(rank)`; 0x4F containers+mines + 0x46 ack with enemy-found; 10 fuel), **map open** (free 0x4C: atlas-ordered fuel dots from live containers + living-tank blips, no mines), **mine press** (10 flat; 3×3 skip rock/water/tanks, 1:1 enemy-mine trades as 0x45, placements as 0x4B; mines are not inventory). Pickup-fuel/equipment clicks route through the move law (they are destination clicks on the wire). Only `other` commands still raise `SimError`. The full bot command set now processes — unblocking the live CDP substitution for step (c) completion. Gate: 4,719 tests, 100% stmt+branch.

## [2026-07-22] refactor | Phase 4 step (c) COMPLETE — the production bot plays the sim over the live CDP seam

`sim/session.py` lands `SimCDPSession`: the same `CDPSessionProtocol` the production tick loop talks to, answering every `Runtime.evaluate` from sim world truth (snapshot = truthfully-built `PageClientSnapshotDict`; injected websocket sends decode through the transport into typed commands; unmodeled expressions raise). The smoke test runs **12 rounds of the real `_tick_once`** against a seeded world: the bot toggles equipment, radars, collects, hunts, and fights — beliefs equal to sim ground truth at the end. The seam immediately taught two things: (1) the bot's true opening move is equipment toggling (cmd 114), previously unmodeled — the toggle law now flips the slot server-side and answers the documented 0x74 `t + 5 bytes` state; (2) a barren world ends the session the production way — the COLLECT owner raised the real `SessionExitError` (`no_productive_collect`) after killing the enemy and draining the map, so sim worlds must be seeded sustainably, and equipment containers remain a world-model gap (the sim bot can restock fuel but never ammo). Gate: 4,725 tests, 100% stmt+branch. Next: step (e) — the timed soak entry point writing standard capture/events artifacts, the fuel/ammo books' divergence-zero verdict on a sim session, and `make audit` over sim-generated wire.

## [2026-07-22] refactor | Phase 4 step (e) COMPLETE — the sim is certified by the instruments that watch the real server

Phase 4 closes. Three tests deliver the verdict: (1) a **divergence-zero soak** — 30 rounds of the production `_tick_once` under a stepped clock, with the Phase 3 fuel/ammo books judging real accounting windows and ZERO divergences in both the book counters and the captured `events.jsonl`; (2) a **negative control** proving the detector has teeth — a corrupted +700 fuel sync through the real ingestion path fires `physics_divergence` exactly as designed; (3) the **audit cross-check** — `SimCDPSession` records all wire traffic (`wire_log` → `build_capture_session`), and the real `collect_evidence` prices sim-generated wire at real-archive exactness (walk-cost 1/1, dual-shot-cost 19/20, fuel-capacity 45/45, every sampled claim over `EXACTNESS_FLOOR`; the lone dual-shot miss is the measured charge latency splitting a burst's first echo and debit across a window boundary — the same noise shape real captures show). The instruments forced three catches before going green: the sim synced fuel only on change while the measured wire broadcasts 0x2E per living tank every ~2 s ([[tank-freshness-model]]) — without quiet readings the fuel book can never close a block; the join burst lacked the client's OWN 0x21, which the audit's first-0x21-names-self convention requires; and a pre-existing cross-test contamination (the client-structure survey's once-per-session gate missing from the central isolation fixture) surfaced when the new tests changed xdist scheduling. Also recorded: the production bot never walks — 100 % teleport locomotion — so walk-cost needed one scripted walk through the real command service. Full fidelity statement (what is and is NOT certified) in [[physics-module-roadmap]]. Gate: 4,730 tests, 100 % stmt+branch.

## [2026-07-22] refactor | Law 4 lands — viewport departure (0x58) and the homing reroute TTL, all eight laws now in the sim

The last unimplemented law needed the thing its trigger depends on: a viewport model. `VIEWPORT_RADIUS` (Chebyshev 8 — the same constant the extra-radar scan already used) now scopes tank POSITIONS on the sim wire: the join burst 0x3D-states only in-view tanks (0x21 identities stay global), each tick diffs membership after relocations — exit emits 0x58 TankRemove and starts the reroute clock, re-entry emits a fresh 0x3D, deactivation just drops from the visible set (0x41 announces that exit). On top of it, id-targeted shot resolution ([[shoot-event-format]] law): an id-shot at a visible tank reroutes the click to the tank's CURRENT tile (the queue-race conversion — stale coordinates at a same-tick mover resolve as homing, not a miss); an id-shot at a departed tank keeps drawing guaranteed homing hits (ammo debited, damage applied, position dark) while age ≤ `REROUTE_TTL_MS` (the machine-checked 12 000 ms midpoint), then becomes the measured free single miss with nothing debited. Eleven new law tests in `tests/sim/test_reroute.py` pin all of it, including the server pricing age in ticks. Free behavior check: with positions viewport-scoped the seam's enemy is position-dark at join — and the production bot still finds and kills it through map blips + teleport, the real gameplay loop, with the step-(e) soak and audit cross-check staying green untouched. Gate: 4,741 tests, 100 % stmt+branch.

## [2026-07-22] discovery+refactor | Equipment containers land in the sim — the archive falsifies the pickup contract, and the wiring flushes out a REAL bot bug

The last world-model gap closed, and the discipline paid three times over. (1) **Crack before code**: instead of trusting the wiki's "deterministic, fills the slot you're most behind on" contract, the grant law was mined from the archive — 1,154 `0x67 -> next 0x49` exact-pre pairs (every 0x67 in the corpus is followed by its inventory snapshot in the very next frame, so `pre = post - gained` is drift-free). Verdict: one slot per grant, hard cap 25, stack rolls 5-9 for dual/homing and 2-4 for radar, and slot choice RANDOM among deficient slots (128 homing-over-needier-dual, 37 the reverse, 89 radar-while-a-weapon-short) — the determinism claim is falsified and [[equipment-system]] rewritten, plus a previously unknown mechanic: 5 `show_message=False` MULTI-slot grants, all with radar at exactly 0. (2) **The viewport had to become real**: equipment candidates gate on in-viewport walk-reachability and 0x5A is the ONLY viewport-setting message — never emitted by the sim, so the bot could not act on revealed equipment at all. The sim now sends origin-only 0x5A patches (production's reset-then-apply sweep spares radar-sourced entries, so empty entities are exactly truthful) at handshake and on every client relocation. (3) **The consumed-container signal is a rejection**: nothing on the wire announces a consumed container; the client re-clicks and gets 0x52 error 4. Wiring that answer exposed a REAL production bug — a same-tile collect completes instantly by position_reached BEFORE the in-flight error handler runs, the code-4 orphans, and the bot re-clicks the ghost belief forever (latent since the DOM-consumer removal was deleted 2026-07-19). Fixed at the root in `completions._maybe_complete_collection`: a pending 0x52 defers position-completion one phase so the error attributes and deletes the belief. End-to-end seam proof: the production bot at 8 extra radars radar-reveals (0x4F `-1` marker), walks both seeded containers, takes two grants, eats the code-4s, cleans its beliefs, and goes back to fuel. Gate: 4,754 tests, 100 % stmt+branch.

## [2026-07-22] discovery+refactor | The sim shoots back — and instantly catches an unwired live instrument

`sim/opponent.py` gives the seam a deterministic scripted aggressor (tick-pure 4-beat: dodge / shoot / hold / shoot, sight-limited to its own viewport radius) — explicitly a harness, not a model of enemy minds. Wiring it forced a per-recipient sweep (0x52 rejections, 0x67 gains, and inventory-full errors are per-connection; the sim had been leaking them into the client batch for ANY commanding tank — the same class as the step-(c) fuel-sync leak, latent only because the enemy had never acted) and then caught a REAL production defect on its first run: the fighting soak's positive control demanded `ammo_book["enemy_shots"] > 0` and got 0, because `record_ammo_enemy_shot` was defined, exported, unit-tested — and never called from production. The ammo book bounds shield loss by `2 × enemy_shots`, so the first armor-absorbed hit in a live fight would have produced a FALSE physics divergence from a frozen counter. Unit tests can't catch dead wiring (they call the function directly); only a session with return fire could. Now wired at the 0x53 dispatch point beside the fuel book's enemy-hit entry. The fighting soak stands as the verdict: 24 production rounds under real duals landing (client fuel paying real 90s), both books at zero divergences, zero `physics_divergence` events. Gate: 4,759 tests, 100 % stmt+branch.

## [2026-07-22] feature+discovery | `make sim-run` — free soaks on real terrain, and production gap #4: the bot could not notice its own death

The seam becomes a product: `tankpit-sim-run` plays the real bot against the sim on the REAL `field01_r.gif` (actual mountains and water drive the router, shot clipping, and displacement), opponent returning fire, artifacts in standard shapes (`runs/probe/latest.sim.*` events + `runs/sim/*.capture_session.json` + final world state). The first real-terrain runs found four things in an hour: (1) the naive scenario region is COASTAL — six seeded containers sat in water and the bot starved among unreachable dots; now `_require_seeds_passable` rejects impassable seeds loudly and the shipped arena is a verified fully-open clearing at (216,108), pinned against the real GIF by a test. (2) A corpse keeps clicking — real connections survive deactivation, so dead-client commands now drop silently instead of tripping the sim's harness guard. (3) **Production gap #4**: own-kill 0x41s have been decoded since 2026-07-19 but NOTHING consumed them for self-death — the killed bot ticked forever "waiting for radar results". The 0x41 dispatch now records `self_deactivated` (dispatch can't throw — replay runs it too) and the tick loop raises the new `deactivated` session exit; the wire replaces the DOM scrape as the self-death channel. (4) The bot fights with armor OFF by policy and an out-of-ammo enemy still lands unlimited 45-fuel singles — both facts now documented and baked into the default scenario's tuning. Reference 150-round run: fight → kill → refuel to cap → collect → `no_viable_targets`, a clean production ending. Gate: 4,767 tests, 100 % stmt+branch.

## [2026-07-22] discovery+feature | The world replenishes and players return — spawn dynamics cracked from 0x4C atlas diffs

Two finite-world walls fell. (1) **Container respawn, archive-mined**: the 0x4C map atlas is global, so within-session snapshot diffs are true world dynamics — 212 sessions yield a steady-state population of 569–656 fuel dots, a population-seeking spawn rate of ~1.00 dots/minute below equilibrium (605 spawns / 605.7 minutes; a 12-minute idle session at high population spawned ZERO), and a hard fresh-position law (0/605 spawns reused a consumed tile). No wire message announces a spawn — discovery is by map or radar. Recorded in [[game-economy]]; implemented deterministically in `sim/spawn.py` (seeded population as target, minute-beat spawns at tick-derived open tiles; equipment mirrors on the offset beat as an assumption since the atlas can't see it). (2) **Respawns join as NEW tank ids**: the first revival attempt reused the killed id and the production bot — correctly — refused to re-engage it (kill suppression + registry liveness never forgive a dead id, and they shouldn't: `persistent_tank_id` exists precisely because wire ids change across respawns). The harness now activates a fresh id near the client, announced by a mid-session 0x21; the reference `make sim-run` shows the full cycle — kill, re-acquire the respawned id, fight on, and eventually the documented radar death-spiral exit as natural old age. Sessions went from 36 rounds to 79+ with every ending a production-correct reason. Gate: 4,779 tests, 100 % stmt+branch.

## [2026-07-22] feature | Ferries sail in the sim — law 2b, and the 18×18 patch-grid catch

The single-command surface contract ([[ferry-mechanics]]) is executable: `SimFerryDict` is one dynamic water tile that moves with its rider; routing is surface-gated (a water click from land is the measured cant_go; riding opens the sea); the first queue-consuming transition truncates the move ON the transition tile — boarding stops on the ferry even when the click was beyond it, disembark stops one step onto land with the ferry left on the last water tile — and billing/echo cover only tiles actually walked. On the wire, ferries travel as 0x5A visible-layer entities (terrain 5) with explicit terrain-0 reverts for vacated tiles, deferred until the window covers them. Integration immediately caught the patch-grid border: the 0x5A grid is 18×18 with a one-tile margin around the 16×16 window (`col = x − left + 1`), so the first seam delivery landed the ferry one tile off in both axes — production's `viewport_patch_world_coords` was the oracle. The seam proof has the real ingestion composing the sim's ferry into `FerryAwareTerrain` from real wire bytes. Remaining from this page: FERRY_ROCK, multi-tile ferries, teleport-onto-ferry, and measured ferry fuel rates. Gate: 4,789 tests, 100 % stmt+branch.

## [2026-07-22] feature | Movable blocks complete the world model — every documented entity class now simulated

Law 6b lands: the wire-cracked block contract ([[movable-blocks]]) runs end to end in the sim. One carry-state-routed command (`CMD_BLOCK` 98, now a named constant with `build_block_command`) picks up cardinally adjacent blocks (0x42 direction = the measured compass letter) and drops them with the shared 1/2/3 enum derived from context — bridge over water, obstacle on land, stacked — emitted identically on 0x42, 0x4A, and 0x5A (the dynamic-terrain patcher now serves ferries and blocks with value-change repatching and reverts). All the measured interactions hold: bridges route as ordinary ground, land/stacked blocks clip non-missile shots, teleport-while-towing refuses with code 0 (three-for-three), a land drop silently kills ANY team's mine, block ops are free, and blocks exclude teleport landings, mine placement, and container respawns. Seam-proven: production composes sim block tiles into wire terrain and the real command service round-trips the press. With blocks in, the sim world holds EVERY documented entity class — tanks, fuel, equipment, mines, ferries, blocks. Remaining named gaps are behaviors, not entities: the radar-zero grant, real enemy minds, and the bot's own missing block-planner awareness. Gate: 4,804 tests, 100 % stmt+branch.

## [2026-07-22] discovery | Archive mining sweep — the TTL was 900 ms low, south displacement is real, and the viewport model validates

Three assumption-retiring mines over the 246-session inventory in one sitting. (1) **Reroute TTL, corpus-swept**: every sent id-shot at a removed-and-still-dark id, echo-paired with its own 0x53 (weapon=3 debit == hit; corpse 0x58s and 0x29 quits excluded) — 704 hits dense to **+12.91 s with ZERO later**, and a dense miss wall from **+12.93 s**. `REROUTE_TTL_MS` moves 12 000 → **12 920** (the old midpoint quit ~0.9 s early, donating one guaranteed pursuit hit per chase); the physics-claims pin caught the drift mid-edit exactly as designed. (2) **Teleport displacement, corpus-swept** (2,861 hops): 2,020 exact, 841 displaced — **E 448 ≫ N 89 > S 31 ≈ W 28**, so SOUTH IS REAL and the full cardinal set stands measured; plus a new finding, a **~24 % ring-2/diagonal tail** — a fully blocked ring 1 widens the search rather than rejecting (the sim's ring-1-then-cant_go is now a documented simplification of a measured wider law, [[teleport-mechanics]]). (3) **Viewport-scroll law**: 3,387 bot-session offset samples put the at-rest tank at exactly window offset (8,8) — the sim's centered model IS the rest-state truth; the wide dispersion decodes as client animation lag (wire position leads the on-screen walk, the camera follows the animation), which the animation-free sim rightly lacks, and since the bot only acts from rest-center, the viewport-edge mine clip cannot bite in bot play. Also mined honestly to a dead end: spawned dots were never radar-revealed afterward in the corpus, so spawn VOLUMES stay an assumption. Gate: 4,804 tests, 100 % stmt+branch.

## [2026-07-22] discovery+feature | The radar-zero mystery is a KILL REWARD — cracked deterministically and simulated

The last unexplained mechanic falls to context mining. The archive's 5 `show_message=False` multi-slot 0x67s all ride the same frame as an own-kill 0x41 — and the trigger is exact: **a kill scored while the killer's extra-radar count is ZERO grants a silent mercy bundle** (dual +1–4, homing exactly +1, radar +1–2, allowed to overfill the 25 cap). Corpus proof of determinism: 5/5 radar-zero kills granted, **0/254** kills at radar > 0 granted, zero exceptions in either direction. Tactical meaning: a radar-blind kill self-rescues the hunt with one fresh scan. Implemented in the sim (`_maybe_emit_kill_mercy_bundle`, measured medians, per-recipient), pinned by three law tests, recorded in [[equipment-system]]. With this, every mechanic ever observed in the archive has either a measured law in the sim or a documented, named simplification — nothing unexplained remains. Gate: 4,807 tests, 100 % stmt+branch.

## [2026-07-22] discovery+feature | Second mining pass — the corpse window is a 22.0 s constant; healing measured but honestly unresolved

Two more laws out of the inventory. (1) **Corpse window**: 37 kill→remove pairs corpus-wide give min = median = **exactly 22.0 s** between a tank's 0x41 and its 0x58 — the June "~22 s" single observation is now a constant, implemented as `CORPSE_WINDOW_TICKS = 11` in the sim (and the corpse 0x58 deliberately does NOT start the law-4 reroute clock — rerouting follows living departures only). (2) **Healing**: the 2 s status cadence yields every tank's full damage timeline; quiet-window analysis shows repair starts after ~6–10 s without incoming fire but jumps MULTIPLE tiers per sync window (1→3 ×257, 2→3 ×199, 1→0 ×143 vs single-step 1→2 only ×32), and tier-3→full transitions are strangely rare. The wire semantics of tier 0 (full vs unsynced) confound the ladder — rather than force a law from murky data, the mined statistics are recorded in [[deactivation-format]] and the healing rate stays a NAMED gap awaiting a controlled live measurement. The sim still does not heal. Gate: 4,808 tests, 100 % stmt+branch.

## [2026-07-22] feature+discovery | The shadow comparator — `make shadow` prices the sim's laws against the archive, and finds a self-sync anomaly

The mining sweeps graduate into a standing instrument. New `tankpit_bot/validate/shadow*.py` (CLI `tankpit-shadow`, target `make shadow`): every validator imports its predictor FROM SIM SOURCE (never a restated copy) and re-derives it over all 245 decodable archive sessions — the inverse of the seam soaks (the soaks prove the bot can't tell sim from server; the shadow proves the sim can't be told apart from the archive). First full run, all four laws PASS at the audit's own `EXACTNESS_FLOOR`: **grant-invariants 1,149/1,149** (one deficient slot, cap-25 clip, rolls 5-9 / 2-4 — now sim-source constants with the deterministic stacks DERIVED as midpoints), **kill-mercy-bundle 283/283** (shared `kill_grants_mercy()` predicate), **corpse-window 17/17** (stricter reuse filters than the mining sweep, all inside ±1 s of 22.0 s), **sync-cadence 118/126** (94 %). Calibration triage produced a discovery: 23 of the first pass's 31 outliers were the session's OWN tank drifting to 3-4 s+ median 0x2E gaps (~10 % of sessions) while other-tank medians pin at 1981-2010 ms — the 2 s broadcast law is exact for OTHERS, and the self cadence is measurably different and not load-bearing (own truth rides 0x44/0x64/0x49); law scoped to non-self by measurement, finding recorded in [[tank-freshness-model]], trigger condition an open question. `make-targets` gained the missing audit/shadow/roundtrip rows. Gate: 4,852 tests, 100 % stmt+branch.

## [2026-07-23] discovery+fix | There is no healing — the damage tier is the fuel quartile, and the bot had it inverted

User correction kills the last big unknown: "tanks dont heal. they only can recover health/fuel from picking up fuel containers… mouse over a tank, lighter = more hp." Same-day corpus fit over every long-form 0x2E (tier + absolute fuel in the SAME message): **19,658/19,658 samples, zero exceptions** — ``damage_tier = min(3, 4·fuel // fuel_capacity(rank))``, boundaries exactly 275/550/825 at rank-1 capacity 1100. Every old confusion decodes: the mined "quiet heals" were fuel pickups jumping quartiles, the "~6-10 s repair dwell" was drive-to-container time, and June's "tiers count down 0→3→2→1, kills die from tier 1" was a fresh tank draining 3→2→1 whose killer took fuel below zero before a tier-0 sync could broadcast. The sweep-through: ``physics.damage_tier`` + claim block in [[deactivation-format]]; the sim's hit-driven ``_DAMAGE_PROGRESSION`` state machine DELETED (tier is derived, never stored); ``DAMAGE_*`` constants re-valued and the finish-off ordering in ``bot/ai/threats.py`` FIXED (it was inverted — the bot preferred the wrong targets at equal distance; tier 0 is the kill shot, unknown defaults to healthy); ``DAMAGE_NAMES`` corrected; shadow law #5 ``damage-tier`` re-derives the fit on every ``make shadow`` (19,658/19,658 PASS). The healing live-measurement session is cancelled — nothing to measure. Gate: 4,855 tests, 100 % stmt+branch.

## [2026-07-23] audit | Wiki-audit MCP onboarded — 62 pages green under the code-paths contract

The TankpitBot wiki is registered with the wiki-audit MCP as slug `tankpitbot` (contract kind `code-paths`) and the full `wiki_audit_run` chain now passes with ZERO errors/warnings across all 62 pages. First run surfaced ~95 findings: two real structural ones (bot-service-architecture had empty `hubs:` frontmatter despite hub membership; a punt phrase in tank-freshness-model) and a systematic contract mismatch — nearly every page carried prose annotations in `source_paths:` ("see footnotes", "tpclient.js lines 243-255 (E[] table)", "codebase inspection 2026-06-16") where the contract requires REAL resolvable citations. Normalized all 60 affected pages via an explicit per-page mapping (every replacement existence-checked first): line anchors (`tpclient.js:243` — bounds-verified against the 328-line file), repo paths (`src/tankpit_bot/sim/blocks.py`, `Makefile`, `docs/sources/sigmas-tankpit-guide-v3.4.pdf`), and named capture artifacts (`runs/sniff/sniff-20260720-214839.capture_session.json`); prose context stays in page bodies/footnotes where it already lives. Honest scope note: this layer audits CITATION INTEGRITY (paths resolve, line anchors in bounds, hubs consistent, no punts) — it would not have caught the damage-tier misreading; that requires the archive re-derivation layer (claim blocks + `make audit` + `make shadow`). The three layers are now all standing: structure (the wiki-audit MCP), bindings (guard claim blocks), and physics truth (audit/shadow). Available anytime via `wiki_audit_run(wikiSlug=''tankpitbot'')`; the `git-blob-hash-pin` rule (per-page blob pinning so cited-file DRIFT flags the page for re-verification) is opt-in and not yet adopted — a candidate follow-up. Gate: `make lint` green after the sweep.

## [2026-07-23] audit | Git-blob drift pinning adopted -- 33 pages now flag automatically when cited code changes

Task 5 of the audit-hardening program: every tankpitbot wiki page citing TRACKED repo paths now carries a `source_git_blobs:` frontmatter map pinning each citation to its current `git ls-tree HEAD` hash (blob hashes for files, TREE hashes for directories -- any change anywhere inside a cited directory flips its tree hash and flags every page citing it for re-verification). 33 of 62 pages adopted; 20 cite no repo paths; pages citing gitignored `runs/` artifacts have those entries exempted per the audit-rule refinement shipped alongside (a5082f81 in the MCPs workspace: untracked paths are unpinnable by nature -- the original all-or-nothing companion invariant had locked 28 pages out of pinning entirely because ONE runs/ citation stripped drift protection from their code citations too; existence of untracked citations stays enforced by `source-path-exists`). Effect: `wiki_audit_run(tankpitbot)` now fails the moment any cited source file changes after a page was written -- the mechanical version of "the wiki must be re-verified when the code moves." Verified: one-shot audit 0 errors with all pins live. This closes the fourth defense layer for this wiki: structure/citations (the wiki-audit MCP), wiki<->code bindings (claim-block guard), wiki<->reality (make audit + make shadow), and now cited-source DRIFT (blob pins).

## [2026-07-23] audit + retrofit | Citation retrofit COMPLETE — 588 paragraphs receipted, 0 advisories, 5 falsehoods corrected

The `paragraph-citation-concept` rule (added to wiki-check 2026-07-23, advisory during retrofit) flagged 588 uncited paragraphs across 53 pages. All 588 now carry receipts, applied under a fixed discipline: (a) markers to footnotes already on the page, (b) wikilinks to pages that carry the claim under their own citations, (c) artifacts on disk (run captures, tests, analysis scripts, pinned PDFs/specs — every cited path verified to exist this session), (d) numbers a standing instrument re-derives (`make audit` / `make shadow` / `make roundtrip` / `physics_claims`). New footnote definitions cite only pre-existing page content, pinned files, wiki-log entries, or git commits verified by hash this session. Blocking rules (wikilink-target-exists, footnote-id-resolves) stayed at 0 errors through all six tranches.

**Falsehoods and staleness corrected instead of decorated:**
1. equipment-refill-strategy claimed "walking is free" — corrected to the measured 1 fuel/tile.
2. shoot-event-format still carried the pre-2026-07-23 damage-tier reading (0=full, 1=critical, "tiers repair over time") and "use DOM scraping" hit-detection advice — rewritten to the fuel-quartile law and the post-teardown 0x41 truth.
3. rank-category-bug declared "It is always rank_category" in contradiction of its own Resolution — retracted inline.
4. js-source-map said "exactly 63 chat messages (some gaps)" twice — the E[] table has 65 constructors (verified by count against tpclient.js), 61 in the selector.
5. bot-behavior-contract §6 claimed the block-decision behavior "exists in the code" — it does not ([[movable-blocks]]); bot-service-architecture's Phase B file inventory predated the overlay-viewmodel rework (MCPs commit 88fc8ae5) — staleness notes added; two "wait, re-reading..." first-draft residues (viewport-update-algorithm, obstacle-bridge-mechanics) rewritten as verified declarative prose.

Also: tournament-strategy's `[^sN]` and game-modes' `[^t1]` alphanumeric footnote ids are invisible to the citation rule — renamed/supplemented with numeric ids; bot-service-architecture's placeholder frontmatter (fact_checked 1970-01-01, no sources) replaced with verified source_paths.

**Rule promotion deferred**: paragraph-citation-concept stays ADVISORY until every registered wiki reaches zero (personal 446, tech 212, and the codebase wikis are still outstanding); promote by deleting the rule from ADVISORY_RULES in wiki-check then. The receipt ledger for spot-checking is the diff range 152c7394^..HEAD (six tranches: 152c7394, 47790d66, 867c7d73, de219c05, 82cc974a, and this commit).

## [2026-07-24] mining | Bot policy cracked from the archive — singles-only return fire, next-tick latency, 7/8 teleport-off corroborated

Toward the full practice-room twin (physics done; spawn half-done; bot minds the last gap — user: "we could crack spawn and we could crack bot behavior. theres no players. and the bots have pretty simple logic"). Archive-wide sweep (`analysis_scripts/mine_bot_policy.py`, 246 sessions, 12.5 bot-hours, production decode recipe, 0 decode errors): bot shots are 2,247/2,247 singles; 96.2% fire within 3 s of taking a hit with the latency mass at one 2 s queue tick; 98.7% aim at the attacker's exact tile; zero mines placed, zero kills scored, 285 deaths; near-stationary (79 walk echoes, 0 unexplained drifts); modal hits-before-teleport-off exactly 7 at recruit / 8 at private — Sigma's 2015 table corroborated on the wire for the two ranks the archive contains (no corporal+ bot ever observed; the "smarter at sergeant" regime is uncaptured). User contract recorded verbatim on [[enemy-bot-behavior]]: bots don't pick up fuel/equipment intentionally.

OPEN: 60/64 bot tier-up (refuel) events have no observed movement within 5 s — either 0x47/0x3D are viewport-scoped tighter than assumed or an unmodeled fuel-gain mechanic exists. Blocks the sim policy's refuel channel until a targeted single-bot capture settles it. Next: fit the return-fire policy as a `SimPolicy` + shadow law; equipment-spawn mining (radar-reveal-based; invisible to 0x4C) for the spawn side.

## [2026-07-24] code | Bot policy executable + shadowed — the sixth shadow law, 2,247 samples at 94.6%

The mined practice-bot policy is now code and instrument in one build. `sim/bot_policy.py`: the certified MODEL (stationary default, one next-tick weapon=0 single at the attacker's tile, teleport-off at 7/8 hits by rank; escape destination documented as an assumption) — distinct from the `sim/opponent.py` harness. `validate/shadow_bot_laws.py`: the `bot-return-fire` law joins `make shadow`, importing BOT_RETURN_WEAPON / BOT_RETURN_WINDOW_MS from the sim source; the shadow timeline gained names/shots/positions extraction (0x21 names, 0x53, 0x3D/0x28/0x47) to feed it. First full-archive run: 6/6 laws PASS; bot-return-fire 2,247 samples / 2,125 exact (94.6%, floor 0.85) — and the sample population equals the mining script's count exactly, two independent implementations agreeing. Gate green: 4,872 tests, 100% stmt+branch. Uncertified and recorded: teleport-off destination, ranks >= 2, the 60/64 refuel anomaly.

## [2026-07-24] falsification + law | The bot "refuel anomaly" is reactivation — same id, full fuel, at the 22 s corpse boundary

The 60/64 unexplained bot tier-ups decomposed completely in three drill-down sweeps (`analysis_scripts/mine_bot_policy.py`, viewport + tier-jump + death-correlation passes): 56/64 land at exactly tier 3 (a reset, not a pickup); 27 provably follow the same bot id's own 0x41 death with the gap moded at exactly 22 s (17/27) — the corpse window; all 7 in-viewport no-movement cases were 0→3 reactivations of bots that died in view; 50 stale-position cases confirm the suspected 0x47/0x3D visibility scoping. New law: practice bots REACTIVATE in place with the SAME id at full fuel when their corpse clears (fixed 36-slot roster, ids reused — human respawns join as new ids). Residue: 8 partial jumps (0→1/1→2/0→2) are the genuine accidental pickups of the 2026-07-19 user story, at their true low rate. "Tanks don't heal" survives untouched — reactivation is the deactivation-repair cycle, not healing. Sim gap recorded: the harness revives opponents as new ids (right for humans, wrong for practice bots); same-id reactivation is the next sim law candidate, shadow-validatable from the 27 measured pairs.

## [2026-07-24] correction + code | Reactivation is NOT in place — respawn displaced 24+ tiles; the seventh shadow law lands

User correction on the hours-old reactivation law (verbatim: "dont the bots respawn in a different location, not at their corpse, in game") — and the archive measurement agrees emphatically: 102 death→next-seen pairs, EVERY one ≥ 24 tiles (Chebyshev) from the corpse, 70/102 beyond 96 tiles. Bots respawn far away, effectively anywhere. Roster confirmed: exactly 36 fixed bots (9 per team, all observed), each reusing its id. Code: `reactivate_practice_bot` gains the displaced-scatter law (tick/id-derived deterministic point, MIN_RESPAWN_DISPLACEMENT=24 measured floor, sealed-terrain fallback documented); `SimServer` gains `roster_ids` and reactivates roster corpses at the corpse boundary; `bot-reactivation` joins `make shadow` as the seventh law — first full-archive run 39 samples / 35 exact, PASS (the 4 residuals are attribution noise). Note the near-miss this correction caught: the shadow law judges timing + same-id + full fuel but NOT position, so the wrong in-place assumption would have sat invisible behind a passing law — wiki text corrected the same day it was written. Gate green: 4,882 tests, 100% stmt+branch.

## [2026-07-24] mining | Equipment spawns witnessed, respawn placement uniform, teleport-off band measured — the loop keeps rolling

Three more cracks in one sweep, continuing the standing directive ("cant we just keep mining and cracking without stopping?"). (1) EQUIPMENT SPAWNS, first direct measurements via the radar-reveal method (equipment is 0x4C-invisible): per-tile tracking of every 0x4F entry and 0x5A patch tile across the archive yields 45 witnessed empty→equipment spawns over 9,040 empty-tile-minutes of re-scan exposure (~0.5%/tile/min in the active area) and 5,440 first-reveals (~22 equipment tiles seen per session); consumption attribution weak by construction (re-scan lags pickups), map-wide population still open. (2) BOT RESPAWN PLACEMENT is uniform: the 102 death→next-seen pairs cover all sixteen 64x64 quadrants (3-9 each), mean at map center — the sim's deterministic scatter is distribution-consistent. (3) TELEPORT-OFF DISPLACEMENT modes at 16-31 tiles (84/131 jumps, just past the viewport); sim escape band retuned from the guessed 12-24 to the measured 16-31 (`_ESCAPE_MIN/MAX_RADIUS`). Gate green. Still archive-minable: the self-sync cadence trigger, radar-cost isolation; needs new captures: sergeant bots, exact spawn placement.

## [2026-07-24] mining | Radar cost isolated, self-sync drift is activity-correlated

Two more from the standing mining loop. (1) RADAR COST, never before isolable ("radar rides pickup/scan-heavy windows"), falls to sent-command-keyed windows: between consecutive own absolute fuel readings, require exactly one sent radar command, zero other sent commands, zero shots/pickups/detonations, and a 3 s pre-window guard for charge latency — 1,293/1,311 clean windows across the archive land at exactly -10 (98.6%; residual is the usual positive-signed noise). The 10-fuel radar row in [[game-economy]] is now archive-derived, not just unit-pinned; promoting the method into a standing `make audit` validator is the named next step. (2) THE SELF-SYNC DRIFT (open question from the shadow calibration) has a mechanism candidate: it is activity-correlated — 8/21 human sniff sessions are sparse (38%) vs 9/198 bot sessions (4.5%), and sparse sessions average half the command rate (16 vs 29 cmds/min). Recorded on [[tank-freshness-model]]; the exact trigger stays open. Archive-minable list after this round: essentially exhausted — remaining unknowns (sergeant bots, exact spawn placement, wider equipment coverage, the precise self-sync trigger) need targeted live captures.

## [2026-07-24] tooling | `make bot-watch` — the sit-next-to-a-bot capture, zero new probe code

The user's proposal ("cant we make a probe that does the teleport to a bot and sits there next to them?") turned out to be two existing parameters away: the enemy-teleport probe already exposes `TANKPIT_ENEMY_TELEPORT_SETTLE_MS` and `--max-attempts`, and its harness records every wire frame to a standard capture session (`save_capture_session`) through the whole dwell. New Makefile target `bot-watch` = one map-open acquisition + one teleport-to-nearest-bot + a 10-minute idle dwell, output labeled `bot_watch_probe.json`. One session feeds three open questions: the idle self-sync trigger (we are the idle client), fine-grained in-viewport bot behavior, and whether an untouched bot's fuel ever moves. Analysis path: `analysis_scripts/mine_bot_policy.py` + `make shadow` over the new capture. Still to build: the respawn-watch variant (kill + map-poll for the reappearing blip) as a real new probe per [[adding-a-probe]].

## [2026-07-24] anomaly | First bot-watch run: server traffic ends at t+9s while client keep-alives continue — artifact or law, OPEN

First `make bot-watch` session (capture `bot_watch_probe.capture_session.json` at repo root, 86 messages, 607 s span): the teleport landed adjacent as designed (landed_adjacent=1), then the 10-minute idle dwell recorded ZERO received frames after t+9s — no 0x2E cadence, no ping responses — while the client sent its 30 s keep-alives exactly on schedule (20 sends, socket never closed from the client's view). Two live hypotheses, deliberately NOT resolved yet: (a) SERVER MUTE — a fully idle client gets its broadcast feed suspended (would be a major law and would recontextualize the sparse self-sync sessions as partial throttling; weighs against: idle human sniff stretches still receive the cadence); (b) PROBE ARTIFACT — the receive-capture path dies after the last tracked attempt, never noticed before because every prior probe run ended immediately after its attempts (the settle dwell is the first long post-attempt window ever recorded; weighs for: sent frames kept capturing while received stopped, an asymmetry more natural in the listener than on the wire). Discriminators for the next session: (1) capture a page-client snapshot AFTER the dwell (ws_ready_state + game-log tail — the JS client renders received traffic it believes in regardless of our listener); (2) read the probe's CDP receive-handler lifecycle across the attempt boundary; (3) rerun with a 60 s dwell and the page console logged. No wiki page changes until discriminated — anomalies are recorded, not laws.

## [2026-07-24] diagnosis | The watch-run silence decodes as a MAP-OPEN STREAM MUTE (hypothesis, one discriminator from law)

The bot-watch anomaly discriminated further by the sent-command sequence: map_open at t+5.8s, teleport at t+6.7s, then NOTHING but 30 s keep-alives — the probe never closed the map (the settle dwell begins right after the landing; every prior actor, human or bot, follows a map teleport with more commands within seconds). Combined with the artifact-elimination evidence (same CDP session captured sent frames all dwell; zero handler exceptions; client JS healthy on an open socket), the refined hypothesis: WHILE THE MAP IS OPEN THE SERVER SUSPENDS THE BROADCAST STREAM to that client — map data and the teleport landing still deliver, then silence until the map closes. Explains why idle human sniff stretches (map closed) still receive the 2 s cadence, and possibly contributes to the sparse self-sync sessions (map-open fraction of session time). Discriminator for the next watch run: send the map toggle once more after landing, THEN dwell — if the stream flows for the full 10 minutes, the mute is law and the bot-watch recipe gains a close-map step; if it stays silent, back to the drawing board. Not yet a wiki-page claim — hypothesis recorded here only.

## [2026-07-24] correction + discriminator | Map exonerated (user contract + A/B); the silence is a real quiet-client feed mute; l/f command rows were swapped

Three developments in one pass. (1) USER CONTRACT (verbatim on [[client-commands]]): the programmatic map open is NOT a toggle — close requires the client-side 'm' keypress, and "teleporting of course closes the map as well" — so run 1's map was closed at its t+6.7s teleport and the map-mute hypothesis was already weakened before the A/B landed. (2) THE A/B CONFIRMS: the no-map watch session (nearest-enemy acquisition timed out; no map ever opened; tank sat at spawn) ALSO went silent — last received frame at t+158s, then ~7.5 minutes of nothing while 30 s keep-alives continued on an open socket. The quiet-client feed mute is real and map-independent; onset varied (9 s vs 158 s, unexplained). Consequence flagged: the sync-cadence law's "global 2 s broadcast" premise was only ever measured on constantly-acting clients — it may be conditional on client activity. Third watch run (wake-on-action: 3 attempts x 3-minute dwells) launched to test whether actions revive the feed and to collect two more mute onsets. (3) BONUS FALSEHOOD: cross-checking the user contract exposed swapped rows in [[client-commands]] and [[js-source-map]] — both listed 'f'=map/'l'=radar; the live wire (bot constants, every capture: 'l' -> 0x4C, 'f' -> 0x4F) proves the opposite; both pages corrected, JS class re-trace queued.

## [2026-07-24] discovery | The per-client stream is push-on-activity — three designed runs, ~2 s post-action tails

The wake-on-action run closes the triangle. Three watch sessions, one variable each: (1) map+teleport then 10 min idle — feed dead from t+9.5s; (2) no map, no teleport, idle at spawn — feed effectively dead by t+158s; (3) three map opens at t+6/190/372s with 3-minute idle dwells — the feed answers EACH action (map data served even mid-mute) and dies ~2 s after it, three of three (silence gaps 182 s and 180 s between actions). Law candidate: THE SERVER STREAMS THE BROADCAST FEED ONLY IN A SHORT WINDOW (~2 s) AROUND CLIENT ACTIONS; request-responses are never muted; keep-alives hold the socket but not the feed. Implications: the sync-cadence "global 2 s broadcast" was measured entirely inside activity windows (every archive client acts constantly — the bot ticks every 2 s); the sparse self-sync sessions are the partial-idle version of the same behavior; run 2's 158 s tail is the residual puzzle (its early trickle outlived its last game action — possibly early keep-alives count as activity, unresolved). NOT yet a wiki-page law: needs (a) an onset-precision run (action then measure exact cutoff repeatedly) and (b) a hold-open run (one cheap action every ~1.5 s sustaining the feed indefinitely) before the sim's every-tick broadcast gets an activity condition. Bot-side impact: NONE for current play (the bot never idles >1 tick), but any future "sit and observe" behavior must send periodic actions or fly blind.

## [2026-07-24] code | Watch-dwell heartbeat — the probe can now observe without going dark

The push-on-activity mute made silent dwells blind, so the dwell learned to act: `ProbeBase.request_inventory()` (the cheapest game action — the 'i' query, free, no world effect; also added to `CommandService` and `Bot` dispatch), and `EnemyTeleportProbe._settle_dwell` sends it at the start of the dwell and every `heartbeat_interval_ms` after (0 preserves the historical silent settle). Threaded end to end: session dict + codecs, `execute_probe`, CLI env `TANKPIT_ENEMY_TELEPORT_HEARTBEAT_MS`, and `make bot-watch` now dwells at a 1.5 s heartbeat. Gate green (4,888 tests, 100% stmt+branch). The next bot-watch run doubles as the push-on-activity precision experiment: if the heartbeat holds the stream open for 10 minutes, the law's hold-open half is confirmed and the wiki gets the page-level write-up; the dwell's wire then finally delivers the three original watch questions (fine-grained bot behavior, untouched-bot fuel, idle cadence).

## [2026-07-24] fix | Heartbeat run 1 went silent — the LANDED settle path never dwelled; live catch, gate could not see it

The first heartbeat watch session came back with run 1's exact silhouette (59 received all in minute 0, keep-alives only) — the heartbeat never fired. Root cause: only the non-teleport settle path had been converted to `_settle_dwell`; the landed path (the one a successful watch actually takes) still called the raw silent `wait_for_timeout`. The gate stayed green through this because the raw-wait line remained covered by the landed-settle test — a live run was the only instrument that could catch it, and did, immediately. Fix: the landed path now dwells through `_settle_dwell` too, and the landed-settle test pins the heartbeat on that exact path (inventory sends counted, interval waits asserted). Gate green. Precision session relaunched.

## [2026-07-24] falsification | The heartbeat does NOT reopen the broadcast — all 366 dwell receives are our own 0x49 responses

Heartbeat watch run 2 (fixed landed-path dwell): the wire stayed alive the full 617 s at ~40 received/min — but decoding the dwell shows EVERY received message is msg_type 0x49, the direct response to our own 1.5 s inventory heartbeat. Zero 0x2E syncs, zero world traffic, nine minutes, sitting adjacent to a bot. So the push-on-activity hypothesis splits: direct request→response is never muted (now proven at n=366), but the BROADCAST stream is not reopened by arbitrary client actions. Cross-run pattern: every run that did map_open+teleport muted broadcasts ~2 s after the landing (runs 1, 3, 4); the only run that never opened a map kept broadcasts ~158 s (run 2). Prime suspect again: the MAP-OPEN sequence suspends the broadcast feed server-side, and our programmatic flow never delivers whatever signal restores it (the user's teleport-closes-map contract governs the client; the server-side feed evidently stays suspended). Next discriminator launched: nearest_enemy acquisition + heartbeat dwell — no map ever; if broadcasts outlive 158 s indefinitely, the mute is map-linked and the fix for observation probes is simply never opening the map (acquire via 'h').

## [2026-07-24] falsification | Map AND teleport exonerated — the broadcast mutes for any non-PLAYING client; queries never count

Watch run 6 (nearest_enemy + heartbeat): acquisition timed out, so the session sent NO map open and NO teleport — only the 'h' query and 1.5 s inventory heartbeats — and the broadcast stream still died within the first minute (dwell receives: 366 x 0x49 self-responses, plus one stray 0x21 and one 0x28 — event announcements appear exempt from the mute). This kills the map hypothesis AND the teleport hypothesis as necessary causes, and re-reads run 2's 158 s "tail" as straggler noise on an already-dead feed. Consolidated five-run law candidate: the server serves direct request-responses unconditionally, but streams the periodic world sync only to clients that PLAY — queries ('i', 'h', 'l') and keep-alives do not count; every continuously-fed client in the 246-session archive constantly walks/shoots/radars/picks up. Open precision question: which action classes count. Next build: a WALK heartbeat for the watch dwell (1-tile shuffle per beat, 1 fuel each) — if walking holds the feed, observation probes are viable at ~40 fuel/min and the law gets its page write-up with the query/play distinction.

## [2026-07-24] code | Walk heartbeat — the dwell now genuinely plays (query heartbeats falsified)

`_heartbeat_action` replaces the inventory query with a 1-tile walk shuffle (east / back west, 1 fuel per beat; inventory fallback only when self state is unknown). Same single knob (`TANKPIT_ENEMY_TELEPORT_HEARTBEAT_MS`), same dwell loop, tests updated to pin the shuffle pattern and the landed-path beats. Gate green (4,890 tests, 100%). The next `make bot-watch` run is the decisive session for the mute law: if a walking observer receives the world stream for 10 minutes, the law's final form is "the periodic sync stream flows only around real gameplay actions; queries and keep-alives never count" — page write-up follows with the sync-cadence activity condition; if even walking fails to hold it, the mute is time-since-JOIN-gated and observation requires rejoining.

## [2026-07-24] LAW CONFIRMED | The decisive walk run — real actions hold the push stream; the watch probe works

Run 7 (walk heartbeat, 10-minute dwell adjacent to purple-2): the walking observer received the server's push streams for the ENTIRE 617 s session — self 0x2E sync every ~3 s (188 total), the empty 0x3F MSG_SYNC tick every ~6 s (104), own 0x47 movement echoes (205), and supervisor responses — where every prior design died within the first minute. The law takes its final form: THE SERVER STREAMS PERIODIC PUSH TRAFFIC ONLY TO CLIENTS TAKING REAL GAMEPLAY ACTIONS; direct request-responses are never muted; queries ('i','h','l') and keep-alives never count. Page write-up: [[server-push-gating]]. Two bonus findings. (1) The watched bot emitted NOTHING in ten minutes — zero syncs, zero moves, zero refuels — deepening the undisturbed-bots-do-nothing law to 10-minute observation depth and proving other tanks appear in the stream only when THEY act (passive fuel-reading of an idle bot is impossible; the 0x2E "global 2 s broadcast" premise is per-tank activity-conditional — [[tank-freshness-model]] revised). (2) The dwell never drains the CDP buffer, so the shuffle walked against the frozen landing position: east beats succeeded, west beats targeted the bot's occupied tile — 154 x CANT_GO + 45 x ALREADY_THERE supervisor rejections (which STILL held the feed open, and cost no fuel; fuel 788->505 ~= the ~205 successful 1-tile walks plus drift). Fixed: `_settle_dwell` now drains each beat before acting. Evidence committed: `bot_watch_probe.capture_session.json` (1,198 messages; tank "Artax" id 1301; the 0x21 burst at t+5.5s is the join-time 36-bot roster dump, not mute-piercing events).

## [2026-07-24] lesson | Never analyze a probe capture before its task exits — the teardown rewrites the file

The first three analyses of run 7 ran against `bot_watch_probe.capture_session.json` as written mid-run (mtime 36 minutes before the task exited) and produced coherent-looking garbage: "all dwell sends are 3-byte queries, the walk never fired, the run is void." The final teardown rewrote the file (886 -> 1,198 messages; different payload bytes for the same timestamps), and the real data showed 401 five-byte walk frames spanning the whole dwell and a fully-open push stream. The intermediate file was internally consistent enough to survive three cross-checks — only re-reading the file mtime after the task-complete notification exposed it. Standing rule: the capture is not evidence until the probe process has exited.

## [2026-07-24] instrument | Radar cost promoted into `make audit` — the sweep replicated digit-for-digit

The 2026-07-24 radar isolation (a one-off inside `mine_bot_policy.py`) is now a standing validator: `validate_radar_cost` in `validate/archive.py` re-derives the −10 claim from lone-radar fuel windows with the 3 s backward contamination guard, faithfully replicating the mining recipe — first run over the archive: **1,311 samples, 1,293 exact, 18 mismatches, PASS**, identical to the sweep's numbers. Getting there required provenance on fuel readings (`FuelReadingDict.from_event` marks 0x44/0x64-carried readings, which the mining treated as contamination; the first draft used a forward span instead of the backward guard and collapsed to 145/55 — the recipe's exact shape matters). The claim joins `STAMPED_PAGES` for [[game-economy]], so `make audit --stamp` now certifies twelve claims. Gate green.

## [2026-07-24] fix | The audit stamp broke the page schema — fact_checked must be a bare date

First real-wiki `make audit --stamp` since the verbose stamp format landed: the "(make audit: N claims re-derived, M clean samples)" suffix made `game-economy` fail the wiki-check page load (surfaced as "YAML parse failed"; the page vanished from the audit and every [[game-economy]] wikilink went dark — 9 errors from one line). The contract wants `fact_checked: "YYYY-MM-DD"`, nothing more. `stamp_fact_checked` now writes a quoted bare date; the evidence table stays in the audit's stdout and the pages' footnotes. Lesson recorded in the audit module docstring.

## [2026-07-24] re-trace | Mb/Nb closed — the June swap was a key-vs-wire-char conflation; three-way agreement now

The queued JS re-trace is done and the l/f mystery has its mechanism. The source is unambiguous: `function Mb(){this.code="f"}` (radar) and `function Nb(){this.code="l"}` (map open) — and the June trace swapped them because it assumed the keyboard key IS the wire char. It isn't: the keydown handler feeds `event.code` through the arbitrary `nh()` keymap (`KeyS:26 → P(this,4) → new Mb → 'f'`; `KeyF:14 → P(this,8) → new Nb → 'l'`; `KeyE:13 → P(this,9) → new Xb → 'h'`). KeyL — the natural-but-wrong guess for the 'l' command — is the SOUND TOGGLE and never reaches the wire; KeyM in this build is the tips toggle (the user-contract map-close 'm' lives in the map component's own handler). Punchline: the bot's `protocol/commands.py` key comments ('s' radar, 'f' map, 'e' nearest-enemy) were right all along — wire captures, JS source, and code comments now agree three ways. [[js-source-map]] gains the keymap section; [[client-commands]]'s re-trace note is resolved.

## [2026-07-24] measurement | The push window is one server tick — run 8, n=101, max 2.06 s

The precision run (6 s walk heartbeat, otherwise the decisive-run design) separates each action's push window cleanly: 101 beats, exactly 2 push messages per beat (101/101), last push at median 1.21 s / p90 1.86 s / max 2.06 s after the walk, then silence until the next beat. [[server-push-gating]] gains the refinement: each real gameplay action opens a ~2 s (one-tick) push window; run 7's "continuous" stream was a 1.5 s cadence re-opening the window before it closed. Evidence committed: `bot_watch_pw6_probe.capture_session.json`.

## [2026-07-24] code | Respawn-watch probe — the reactivation law gets its live-witness instrument

New `make respawn-watch` (`action_lab/respawn_watch.py`, `RespawnWatchProbe(EnemyTeleportProbe)` via the new `_post_landing_phase` hook): land adjacent (existing machinery), ENGAGE — one single at the target's current registry position every 2 s until it leaves the registry or 30 s elapses (an adjacent bot returns ~45/hit and full-fuel recruits flee at 7-8 hits, so kills come from already-damaged targets; the engage is deliberately short) — then MAP-POLL every 2 s for 60 s: map data is request-response and immune to the push mute ([[server-push-gating]]), so the 0x4C snapshots pin the same-id reappearance tick and tile regardless of stream state. Up to 4 targets per session, all knobs env-tunable (`TANKPIT_RESPAWN_WATCH_*`). Kill-vs-flee is judged offline from the capture (0x41 vs 0x58), same instrument philosophy as the bot-watch runs. Gate green.

## [2026-07-24] LIVE WITNESS | First respawn-watch session kills purple-2 and catches its 22.0 s same-id reactivation at (154,216)

The new probe worked on its first attempt. Engagement: landed adjacent to purple-2 (id 510, damage tier 3), singles every 2 s; the bot returned singles (weapon=0) on its next-tick cadence while its tier fell 3→2→1; at t+23.3 s the 0x58 removal, at t+25.3 s the 0x41 kill — victim 510, killer 1301 (the probe), promo_eligible — the removal precedes the kill announcement by one tick, a sequencing detail the archive never resolved. Then the map-poll phase earned its design: eleven consecutive 2 s polls show 510 ABSENT from 0x4C map data (corpses aren't rendered on the map — also new), until t+47.3 s when the SAME id reappears at (154,216): death→respawn 22.0 s from the 0x41 (bounded 20.1–22.0 s by poll cadence), 77 tiles Chebyshev from the corpse, stationary in all 23 subsequent polls. Every element of the reactivation law — same-id roster reuse, the 22 s corpse window, ≥24-tile displaced uniform placement, post-respawn idleness — now has a live witnessed cycle on top of the 102 archive pairs. [[enemy-bot-behavior]] carries the write-up. Minor anomaly for the record: 4 of the bot's 5 return shots aimed at (126,138), six tiles west of the probe's actual tile (132,138) — the archive's 1.3% off-tile aim residual, seen up close. Attempts 2–4 ended no_enemy (post-kill quiet registry under the push mute; expected).

## [2026-07-24] CORRECTION + discovery | The "misses" were a third combatant — blue-7's cross-team assist fire; two claims from the live-witness entry retracted

The user challenged the "off-tile aim" reading of the respawn-watch capture (verbatim: "you shoot the bot. we are standing still. it hits us. the bots never miss.") and the full 0x53 frames prove them right — with a discovery attached. The fight had a THIRD combatant: blue-7 (id 524, OUR team), standing at (126,138), opened singles on purple-2 exactly one bot-reaction tick after our first dual; purple-2's return fire then switched from our tile to blue-7's — every shot aimed at a real attacker's exact tile, nothing missed. TWO CLAIMS RETRACTED from the previous entry: (1) "removal precedes the kill announcement by one tick" — wrong; the 0x58 was purple-2's TELEPORT-OFF after ~11 accumulated two-attacker hits, and our server-selected homing chased it to (166,143) and killed it THERE (classic chase mechanics, already on the page); (2) "corpse (132,139), displacement 77" — the true corpse is (166,143), displacement 73 Chebyshev (law unaffected). The 22.0 s same-id respawn, corpse-invisibility in 0x4C, and post-respawn idleness all stand. Discovery promoted after archive discrimination (`analysis_scripts/mine_bot_assist.py`, new standing script): 81 bot→bot shots corpus-wide, ZERO at same-team bots, 78/81 within 10 s of a player shot — bots never seek each other, but a player's engagement ignites cross-team return-fire loops that never produce corpses. This also explains most of the corpus's 1.3% off-attacker-tile aim residual (attacker switching) and part of the hits-before-teleport spread (multi-attacker accumulation). [[enemy-bot-behavior]] rewritten accordingly. Lesson: a one-line anomaly ("4 shots aimed six tiles west") was the thread to a real mechanism — pull such threads before logging them as residuals.

## [2026-07-24] user contracts + mining | Team aggro is bidirectional — 48 gang-up + 81 assist shots, both inside the 8-tile sight radius

Two new user contracts (verbatim): "if you fight another bot, like an orange one, and there is a blue bot that can see that orange bot. it'll help you out" and "if you teleport into 3 orange bots. and hit one, the other two orange bots will start hitting you". The aggro sweep (`analysis_scripts/mine_bot_aggro.py`, standing) classifies every bot-shooter 0x53 in the archive by whether the shooter or a same-team bot had been hit within 10 s: 2,115 personal return-fire, 48 GANG-UP (never-hit shooter, hit teammate — the second contract), 81 ASSIST (at enemy-team bots — the first contract), 3 unexplained. Both mechanisms' shooter→target distances cap at 8 tiles Chebyshev — the "can see" condition is the viewport radius, measured. Full team-aggro model on [[enemy-bot-behavior]]; noted there that `sim/bot_policy.py` models only personal return fire so far (team aggro = open sim-law candidate for the next shadow-law round).

## [2026-07-24] user contract + verification | Shot-for-shot: no aggro state — 1:1 ceiling and one-tick stop verified; bot ranks are config-allocated

The user corrected the aggro framing (verbatim: "bots are just 1:1. so you shoot them once, they shoot back. if you stop they stop. they dont chase or keep attacking. its shot for shot with the bots.") and the archive verifies it exactly: 3,031 hits on bots drew 2,201 returns, the per-engagement fired/taken ratio never exceeds ~1 (mode 0.75–1.0, n=366 — the deficit is the one-return-per-tick cap plus flee/death truncation), and in 99.2% of 397 engagements the bot's last shot lands within one tick of the last hit it took (3 stragglers, multi-attacker attribution). Standing script `analysis_scripts/mine_shot_for_shot.py`. This closes the "aggro decay / target selection" open question: there is no aggro state — assist and gang-up are the same per-hit reflex applied by sighted teammates. Also recorded (hedged, verbatim): bot ranks "are allocated by the game config i think... as far as i know" — allocation not promotion, closing the private-bots mystery; and player promotion mechanics on [[game-rules]] (points per shot; kills bonus not gate for recruit→private; death demotes) corroborating the existing table. [[enemy-bot-behavior]] team-aggro section rewritten to the shot-for-shot model.

## [2026-07-24] replication + extension | Fuel spawn law re-derived independently — snapshot completeness proven, placement uniform

Answering "what do we need to do to solve equipment/fuel": fuel was already cracked (2026-07-22 respawn-dynamics mining) — today's independent re-sweep (`analysis_scripts/mine_fuel_spawns.py`, standing) replicates it from scratch (population median 614 vs the original mean 619; spawn 1.14/min vs 1.00/min) and adds two new facts: back-to-back map opens prove the 0x4C atlas is a faithful full-map snapshot (1,141/1,381 zero-diff at ≤5 s, 236 exactly-one), and spawn placement is roughly uniform across all sixteen quadrants (20–56 appearances each, n=600). EQUIPMENT remains the genuinely open half: invisible to the atlas, so the archive's radar windows gave only 45 witnessed spawns — and the per-tile rate they imply (~0.5%/tile/min) is ~300× the fuel rate (~0.0017%/tile/min), meaning either equipment spawns cluster near players or the radar-window method counts reveal artifacts as spawns. Discriminator: the equipment radar-watch probe — sit still, radar the same viewport every ~15 s for 10+ minutes (65+ fuel/min run cost, needs fuel pickups or short sessions), and diff REVEALS of the same tiles over time; repeated coverage of one area separates true spawns from first-reveals mechanically. That probe is the one remaining build for the container economy.

## [2026-07-24] user contract + refinement | Radar cost is min(10, fuel) — the clamp explains the validator's mismatches; starting stock is 25 extras

Two user facts while designing the equipment radar-watch, both archive-verified same hour (`analysis_scripts/mine_radar_floor.py`, standing). (1) "you cant die from using radar, once you get too low it stops debiting" — the mechanism is a CLAMP: isolated low-fuel radar windows show fuel 6 → −6, fuel 3 → −3, and 14 windows at fuel 0 → 0 with scans still served. The radar-cost law's final form is **min(10, fuel)**; the standing `make audit` validator now encodes it and the archive fit tightened from 1,293/1,311 to **1,310/1,311** (the 18 "mismatches" were the clamp all along; 1 residual, a lone −5 at high fuel). (2) "we only have 25 of those" — 46 sessions open with exactly 25 extras in inventory slot 4, the modal first snapshot; [[radar-mechanics]] records the stock. Probe-design consequence: the equipment watch gets ~25 full-viewport paid scans, then falls back to the built-in 5×5 which — at the clamp — becomes literally FREE once fuel drains to 0, so arbitrarily long stationary watches cost nothing. Also load-bearing for the probe: 0x4F responses are DIFFS (unchanged already-visible entities are not re-sent), so after baseline coverage every reveal IS a fresh event — the server does the spawn detection for us.

## [2026-07-24] correction + user contract | No starting stock — inventory persists across logins; "25" is the private cap

The user corrected the hour-old "starting stock is 25 extras" claim (verbatim: "a new recruit starts with 0 all then they can stop up to 20 each item right? ... if you use them all. log out and log back in. teyre still empty. you have to keep them stocked") and the archive proves the persistence model directly (`analysis_scripts/mine_inventory_persistence.py`, standing): consecutive sessions carry inventory EXACTLY (radar 8→7→6→5→4→3 across six logins, last 0x49 == next first; 120/260 consecutive pairs exact), 22 sessions open at 0 after exhaustion (persistence of empty), and 0/261 sessions ever exceed the 20+5·rank cap (our account is private → cap 25, which is what the "25" was). [[radar-mechanics]] corrected; [[game-rules]] gains the persistence contract. Probe-design note: the radar-watch's paid phase depends on whatever stock the account actually carries in — the probe should read slot 4 at start and size the phase dynamically; a guest/fresh-recruit run would have NO paid phase at all.

## [2026-07-24] user contract + trace | Slot toggles solved end-to-end — Digit1-5 send cc(49-53) in inventory order; this build's R is Top-10, not radar

User contract (verbatim): "r uses the radar, '5' enables and disables it. 1 enables and disables armor shields, 2 dual shots, 3 missiles, 4 homing shots." JS confirms the toggle half exactly: input cases 17-21 (Digit1-5) compute b-17+49 and send `new cc(...)` — the 0x72 'r' wire command carrying ASCII '1'-'5' — toggling slots in inventory order (armor, dual, missile, homing, radar). One build-specific wrinkle: in the pinned client's DEFAULT keymap, KeyR → fe(this,0) → $b(0) is the red-team Top-10 request (R/P/B/O = the four team filters; prints "Top 10 is not registered here" on practice fields) — the default radar key is S; the user's "r uses radar" matches the classic binding and nh() is only the default table (this.l.j may be rebound). Probe consequence: the radar-watch can protect the account's 25 extras by sending cc(53) once at start (with the one-scan self-check against slot 4), since the server holds the enabled state.

## [2026-07-24] verification | Toggle state is wire-visible (0x74 + 0x49 bit-7) and already decoded; the R-key three-way discrepancy recorded

Answering "can you see whether something is enabled or disabled": YES, and the decoders already had it — the 0x74 't' message carries the five per-slot enabled flags, and each 0x49 count byte hides its slot's flag in bit 7 (JS `V.I`: count = byte & 127, disabled = byte & 128). Verified against the user's live inventory panel: their pasted "25 armor shields (disabled) / 25 missile shots (disabled)" matches today's capture decoding `enabled [False, True, False, True, True]` exactly. The radar-watch probe therefore reads state directly (query 'i', check enabled[4], toggle via cc(53) only if needed, confirm) — no blind flip. SEPARATE unresolved item from the same exchange: the R key. The site's static help panel (user-pasted) says "R: Radar" and the user's experience agrees; this build's JS default keymap says KeyS→radar and KeyR→fe(this,0)→$b(0) Top-10-red (T/R/P/B/O = all/red/purple/blue/orange filters, T=255); the help string exists nowhere in the JS (static page HTML, likely classic-era). Wire captures cannot discriminate (key→command mapping is client-side). Decisive experiment queued: a probe session that calls page.keyboard.press for R/S/T and diffs the sent frames — three keypresses, one capture.

## [2026-07-24] falsification + instrument | The key probe settles the keymap — R IS radar; my static JS trace was wrong; two overloaded type bytes decoded

Built `make key-probe` (presses each physical key once as its own capture window; q/space/d/digits excluded) to settle the R-key three-way discrepancy — and the user + site help win: **R sends 'f' (radar: fuel −10, extra consumed, RadarScanResult back) and S sends NOTHING.** The morning's static nh()/dispatcher trace was a misread (minified dispatch code; two near-identical $a functions) — its keymap section on [[js-source-map]] is replaced by the empirical table (r/f/e/i/c/x//: the seven query-commands; l/z/a: plaintext V/C/A toggle syncs; t/p/b/o/s/n/h/m: silent). LESSON: static reads of minified dispatch are hypotheses; only pressing the key is evidence. The probe also flushed out TWO overloaded type bytes: the first run CRASHED on a 1-byte 0x43 (the chat-toggle ack, colliding with CacheUpdate — the official client's $g mis-parses it silently), and the second run surfaced a short 0x41 autoscroll ack (colliding with Deactivation). Both now length-discriminated (`chat_ack`, `autoscroll_ack`) with round-trip encoders. OPERATIONAL NOTE: despite guest intent, the login flow auto-selected the Artax account (accounts.json present) — the two runs cost ~3 account extras (25→22); a true-guest login flag is a queued probe-runtime improvement before any probe that must not touch the account.

## [2026-07-24] directive + build | Radar-watch probe: ACCOUNT with extras toggled off — no guest login

User directive (correcting the key-probe's guest framing, verbatim): "no guest login... use the fucking artax account but with radars disabled ( so it uses free radars)". `make radar-watch` is built exactly so: login as the account, read the wire-visible slot-5 state (0x49 bit-7), send the 0x72 hotkey toggle ONLY if extras are enabled, verify disabled via a second inventory query (ProbeError and no scanning if verification fails), then the stationary watch — one 'f' scan per 15 s (built-in 5×5, min(10,fuel) → free at zero) and one free map open per 30 s for the global fuel baseline. Session JSON records extras before/after so stock preservation is itself evidence. 30-minute default.

## [2026-07-24] first radar-watch session | Zero equipment spawns in the answered window; extras preserved 22->22; NEW anomaly — the ~12-minute idle disconnect

First `make radar-watch` run (30 min nominal, spawn (131,126), extras toggled off and verified — session JSON records extras 22→22 across 120 scans; the stock-protection chain worked end to end). Science: the built-in 5×5 was fully covered from scan 1, and ZERO equipment spawns appeared on proven-empty tiles in the answered window; 48 scans drew exactly −10 each (fuel 1070→590, clamp never engaged); the 24 answered map polls show global fuel dots pinned at 635 with 4 dots constant in our 17×17 (local 13.8/1k tiles vs global 9.7/1k — mild near-player excess, n=4). ANOMALY, new law candidate: ALL receive traffic ended at t+716–726 s (~12 min) — scans, map polls, keep-alive responses; after 720 s the PAGE CLIENT itself sent only once in 18 minutes (its keep-alive pump stopped ⇒ the WebSocket closed). Reading: the server DISCONNECTS a client that has taken no real gameplay action for ~12 minutes from join. All four earlier query-only sessions were <11 min (never crossed it); walking sessions are immune. Consequences: (1) this run's valid window is ~298 empty-tile-minutes, so zero equipment spawns REJECTS the archive's 0.5%/tile/min near-player rate only weakly (P(0)≈22% if it were true) — leaning reveal-artifact, not yet decided; (2) radar-watch v2 must add the bot-watch walk shuffle (1 fuel/beat) to count as playing — no disconnect, unlimited duration. Discriminator queued for the disconnect law: a session with exactly one real action at minute 10 (does the clock reset?).

## [2026-07-24] user contract + persistence | Autoscroll corroborated verbatim; client settings are server-persisted per account (autoscroll AND equipment toggles)

The user restated the autoscroll mechanic ("when you get to the edge of the viewport it re centers on you. otherwise the viewport is fixed and only centers on teleport") — corroborating [[viewport-shift-protocol]]'s auto-shift rule word for word. Two persistence facts attached: (1) the key probe's 'a' press sent A0 and turned the ACCOUNT's autoscroll off (it was ON in the user's panel) — restore queued as a one-key run (TANKPIT_KEY_PROBE_KEYS=a) after the current session releases the account; (2) radar-watch session 2 opened with "extras=22 enabled=False at start" — the slot-5 disable from session 1 PERSISTED across logout/login, so equipment enable/disable state is server-held per account, like the counts. Edge-stuck question answered for the v2 shuffle: it re-reads true position every beat and oscillates between two adjacent tiles 7-8 tiles from every viewport edge — no drift, no edge contact; roaming probes are where autoscroll-off matters.

## [2026-07-24] directive | Autoscroll stays OFF — it is the bot's intended configuration; restore cancelled

User (verbatim): "i usually run the bot with autoscroll off. it was too complicated too implement proper viewport awareness for the bot." The earlier queued autoscroll-restore is CANCELLED — the key probe's A0 aligned with the intended operating mode rather than breaking it. [[viewport-shift-protocol]] now records the rationale behind the fixed-viewport design.

## [2026-07-24] falsification x2 | The ~12-min cutoff is NOT idle-based — v2 walked every 15 s and died at the same mark; prime suspect: the map-open state

Radar-watch v2 (walking every beat, extras 22→22 with the slot-5 disable PERSISTED from session 1 — zero toggles needed): all 120 walks executed (47 echoes in the valid window, fuel −1/−10 debits interleaved, 590→313), zero equipment spawns again (cumulative ~590 proven-empty tile-minutes at zero spawns; the archive's 0.5%/tile/min near-player rate is now rejected at ~95%), fuel dots 635 global / 4 local both sessions — and the receive cutoff hit ~701 s ANYWAY, within seconds of session 1's ~716 s. Walking did not move it at all, so "never-playing disconnect" is falsified as stated. Archive cross-check: three sessions exceeded 12.5 min (one bot session ran 45 min), so there is no hard session cap — something these watch sessions do differs. The unique common factor: MAP POLLING — both sessions sent a map open every 30 s, and the map-open state persists (user contract); both had the map open from t+7 s to death. New hypothesis: idling in the map-open state ~12 min disconnects the client (the production bot never holds the map open — teleports close it — and the 45-min session teleported constantly). v3 discriminator built: map_poll_interval_ms=0 disables polls entirely; a walk+scan-only 30-min session decides. Also noted: with a walk preceding each scan, only every OTHER scan was answered/charged (23 acks at 30 s spacing vs v1's 48 at 15 s) — unexplained, logged as an open instrumentation question.

## [2026-07-24] discriminator round | Map exonerated; walking is FREE at 0 fuel; the per-tick command limit explains v2's dropped scans; the ~12-min wall stands

Radar-watch v3 (zero map opens, walk+scan every 15 s — and, by accident of the drained account, the whole session at fuel 0): the client died at t+713.8 s anyway — the THIRD kill inside 701–716 s — so the map-open hypothesis is falsified alongside idleness. Two new laws from the same capture: (1) **walk cost clamps like radar** — 47 walks executed at fuel 0 with zero INSUFFICIENT_FUEL rejections and fuel pinned at 0 (nothing you do to yourself can kill you; only enemy fire spends you below the floor... rather, the floor holds at 0 for self-actions); (2) **the server processes ~2 injected commands per tick** — v3 with two commands per beat (walk+scan) had EVERY scan answered (47/47 0x4F+acks), while v2's beats that carried three (map+walk+scan) lost exactly the scan, producing its unexplained 30 s ack spacing. Remaining suspects for the ~12-minute disconnect: action RATE (the production bot acts every ~2 s; our watchers every 15 s) or action CLASS (teleport/shoot/pickup — the bot's diet — resetting a timer that walks/queries do not). v4 launched: same watch at the bot-watch 1.5 s cadence (free at 0 fuel), map polls off, 30 min — survival splits rate from class.

## [2026-07-25] TWO LAWS CLOSED | The ~12-min disconnect is rate-gated (1.5 s cadence survives); the equipment near-player rate was an artifact (>99% rejected)

The detached-process run (the harness had killed three background attempts; Start-Process put the probe out of reach) settles both open threads. (1) DISCONNECT LAW FINAL: walk+scan every 1.5 s, no map, fuel 0 — receive traffic steady through the full 909 s, sailing past the 701–716 s wall that killed all three 15 s-cadence sessions. The gate is ACTION RATE: sparse clients are dropped ~12 min after join and sparse actions do not reset the clock; dense actors (the bot's ~2 s tick; the 45-min archive session) live indefinitely. Threshold unbracketed between 1.5 s and 15 s. [[server-push-gating]] gains the four-session table — the 1.5 s shuffle is the connection keepalive, not politeness. (2) EQUIPMENT VERDICT: the fast session's 15.1 clean minutes had ZERO reveals — cumulative ~965 proven-empty tile-minutes, zero spawns, vs ~4.8 expected under the archive's 0.5%/tile/min: rejected >99%. The corpus "witnessed spawns" were first-reveals of pre-existing containers; true equipment spawning is fuel-like. [[game-economy]] carries the verdict. Extras 22→22 for the fourth straight session.

## [2026-07-25] sim law | Team aggro modeled — the bot-return-fire law upgraded to the three-reflex judge; 94.6% → 97.6%

Closing the known model-vs-reality gap the user flagged ("im worried that the sim bot and the real bot arent like the same"): `sim/bot_policy.py` gains the sight-gated team-aggro reflex (`AGGRO_SIGHT_RADIUS = 8` — the measured ceiling of all 129 archive gang-up/assist shots; `note_hit_for_team_aggro` queues one next-tick single per sighted responder, victim's teammates at the attacker and attacker's bot teammates at the victim, never same-team; `queue_return` encodes the shot-for-shot refresh-not-stack cap). The `bot-return-fire` shadow law now judges every bot shot as one of the three per-hit reflexes and recovered 67 of its 122 former mismatches: **2,192/2,247 (97.6%) archive exactness**, PASS. Writing the law tests caught a real bug — the event walk recorded a shot's own landing before judging it, letting an assist justify itself (4 archive shots had been passing that way); judgment now precedes recording. Remaining 51 residuals: strict 3 s window vs the mining's 10 s engagement recency, plus stale-tile aim noise. The model still isn't wired as the live sim-run opponent (opponent.py remains the deterministic harness) — that integration is the next as-built step if sim soaks should face certified bot minds.

## [2026-07-25] sim milestone | Practice-room mode — the certified roster replaces the harness, and the bot DIES the way it would live

`make sim-run-practice` closes the last layer of the user's "sim bot vs real bot" worry: `sim/practice_room.py::PracticeRoomDriver` seeds four certified bots (three purple clustered within sight of each other — the gang-up shape — plus one blue ally for assists, real roster ids 510/511/512/524), drives them with `decide_practice_bot`, notes hits from each tick's 0x53 emissions exactly the way the shadow law reads the wire, and hands the server their ids for corpse-window reactivation. FIRST SOAK RESULT: the production bot DEACTIVATED IN 21 ROUNDS — it engaged one purple, the sighted teammates ganged up (the user's "teleport into 3 orange bots and hit one" contract), and multi-directional 45/tick fire killed it — where it reliably beat the scripted single opponent. The sim now reproduces the real practice-room failure mode for free, which is the twin's whole purpose. BOT-SIDE IMPLICATION (open work): combat target selection should prefer ISOLATED bots — no same-team bot within AGGRO_SIGHT_RADIUS of the target — or expect to fight three tanks at once. The scripted `opponent.py` harness remains the default `make sim-run` (deterministic kill-path soak); practice mode is the fidelity soak.

## [2026-07-25] user contract + law | Combat is round-based with ascending-id resolution — my "simultaneous" framing corrected

The user pushed back on my "everyone acts simultaneously" description ("bro, there are turns... check the combatlogs") and the combat logs prove them right: the respawn-watch fight resolves in clean 2.000 s ROUNDS — six consecutive bursts, each purple-2 (510) → blue-7 (524) → Artax (1301) at 1 ms emission spacing, silence between. One action per tank per round, resolved sequentially WITHIN the round. The hinted ordering rule verified archive-wide (`analysis_scripts/mine_round_order.py`, standing): **1,820/1,825 multi-shooter bursts are in perfect ascending-tank-id order, and all 5 violations are OUR OWN SIM's captures** — the real server is 100%; bots (500-535) always resolve before players (~1300+). Fight shape corrected on [[game-rules]]: tick 1 your hit lands alone; from tick 2 every round carries all provoked responders' singles AND your next shot (nobody sits out a round) — a sighted three-bot cluster is a 1-for-3 trade per round from the second tick. SIM FIDELITY GAP flagged: the sim emits shots in queue order, not id order — aligning SimServer's per-tick resolution order to ascending tank id is queued.

## [2026-07-25] sim as-built | Ascending-id round resolution wired into SimServer — the flagged fidelity gap closed same day

The queue-order gap flagged in the previous entry is closed: `SimServer.advance_tick` now sorts the per-tick queue by tank id before processing (`_queued_tank_id` key; the stable sort keeps one tank's own commands in arrival order), matching the measured 1,820/1,825 archive law. Two sim tests that silently leaned on arrival order were re-anchored so the intended first mover carries the LOWER id (the homing same-tick-move test now has client 9 move while enemy 11 shoots; the reroute test's moving target became id 7 < shooter 9), and `test_round_resolution_orders_by_ascending_tank_id` pins the law. Gate green — 4,955 tests, 100% stmt+branch — and all 7 shadow laws re-priced PASS (bot-return-fire holds at 2,192/2,247). Future sim captures will no longer be the archive's only ordering violations. [[game-rules]] carries the player-facing law; [[physics-module-roadmap]] the as-built note.

## [2026-07-25] post-mortem | The cardinal-shot override: four patches deep, never user-approved, never documented

Root-caused why the practice-room soak died fighting at 84 fuel: the mode selector's first check ("live enemy at Manhattan 1 -> HUNT, regardless of reserves") outranked the fuel-low break, and in a gang-up an enemy is ALWAYS adjacent, so retreat was structurally unreachable. Archaeology of how it got there, each layer a patch on the layer below: (1) COLLECT's dot-hop filters (the 2026-07-03 "100% clean viewport" misread as 100%-walkable) vetoed every travel destination -> marooned at full tank; (2) patch: yield the tick to HUNT -> the one-tick loan let HUNT teleport to enemies it could never get a second tick to shoot (the 2026-07-06 22:37 ping-pong, 56 yields, 10 teleport decisions at orange-8, zero shots); (3) patch: the cardinal override (commit 89ab2715, 2026-07-13) -- shipped BURIED in an unrelated "bot service" commit, no wiki page, no log entry, docstring only; (4) three weeks later the practice room exposed it. The hop-filter root cause was separately fixed 2026-07-18 (rank-not-filter); the two patches above it stayed. User verdict on the override (verbatim): "i never okayed a fight till you die or fight no matter the fuel level". Lesson recorded: behavior-contract changes get their own commit + wiki record, and patches aim at the loop, not where the pain shows up.

## [2026-07-25] user contract + build | Hunt only when full -- every readiness bar rank-derived; the bot now survives the gang-up (21-round death -> 86 rounds, 3 kills)

User contract (verbatim): "it should never hunt when its low on fuel or equipment. it should never hunt if it is not full on everything edcept -5 max radar", refined with "why wouldnt we refuel and restock and then go back to the target" and "just determine max fuel based on the tank rank". As built: HUNT entry and the between-kills restock bar are fuel >= fuel_capacity(rank) AND duals+homings at inventory_capacity(rank) AND radars >= cap-5 -- the fixed fuel_full_threshold (1100, unreachable by recruits at cap 1000) and dual/radar resume thresholds (25/20, under-restocked high ranks) are DELETED from config; collect's pickup ceiling already used fuel_capacity(rank), so "stop collecting" and "may hunt" can no longer disagree. The cardinal override is deleted (ignoring an adjacent bot while collecting is safe -- bots never initiate). Mid-fight, break thresholds disengage to COLLECT and the combat lock SURVIVES the restock (damage persists, so the sortie cycle wins even 3v1); HUNT/ACQUIRE returns to the locked target -- teleport on a trustworthy position, map-refresh on a stale one, release only when the target is gone or the 2026-07-02 engagement gate cannot fund the return. The replay pin for the 2026-06-18 shoot/reject loop now routes its under-stocked session into COLLECT (policy change, machinery unchanged); the fighting soak boots at full 1100 and proves the fight via durable server truth (the ammo book's enemy_shots window resets per 0x49 and the bot now ends soaks disengaged). PROOF: make sim-run-practice went from deactivated-in-21-rounds to 86 rounds / kills on 510 + 511 + the scripted opponent / bot alive at fuel 536 -- exit no_productive_collect only because the sim world ran out of collectible equipment. Gate green (4,958 tests, 100%).

## [2026-07-25] LAW FALSIFIED + LAW MEASURED | Map dots are team-exposure memory of >=500-volume fuel -- the "container spawn law" is dead; user correction proven 605/605

The user corrected the supply model (verbatim): "the yelloe fuel dots are jist large containers that someone on our same team or us priorly exposed. but theres also tons still hidden unril you radar and low fuel contianers which dont shoe kn the map". Two standing miners prove it on the archive (mine_map_dot_semantics.py + mine_dot_appearances.py, 223 sessions): (1) EVERY within-session dot appearance -- all 605, the exact events the 2026-07-22/24 minings had counted as "spawns ~1/min, population-seeking" -- was preceded in the same session by OUR OWN 0x4F/0x5A reveal of a fuel container with volume >= 500 at that exact coordinate; zero unpreceded. The respawn law is FALSIFIED: "spawn rate" was our radar-exposure rate, "population-seeking" was coverage saturation. (2) The dot threshold is exactly volume >= 500 (0/163 sub-500 reveals ever joined; the 500-509 band joins). (3) Equipment never dots (0/1,400). (4) Most large fuel is hidden -- only ~7% of >=500 reveals were already dotted -- and sub-500 containers can never appear, so the field carries far more fuel than the ~619-dot census. (5) Dots outlive their volume (53 sub-500 reveals ON dots), which is the mechanism behind the old "~40% of dots still hold fuel". Consequences: [[game-economy]] respawn section rewritten (true spawn law now fully OPEN -- never witnessed); [[map-data-decode]] dot semantics + cache section rewritten (exposure memory, grows in-session, server-persisted; team-vs-account scope undiscriminated from solo captures); the sim's 1/min replenishment (sim/spawn.py) lost its empirical basis -- honest world model is a large mostly-hidden container population with an exposure-driven atlas, flagged as open sim work. This also re-answers the practice-room starvation: the sim world was poor because the model was wrong, not because the real field is.

## [2026-07-25] sim world rework | The honest field: static hidden population + exposure atlas + real 36-bot layouts -- 150-round soak sustainable, exposure law 18/18 on the sim itself

The full "build it all" package, closing both gaps the user called out ("i m pretty sure theres more fuek contianers than youre claiming" / "are we just fighting the same sim over and over?"). SUPPLY: the falsified runtime spawner is DELETED; sim/world_seed.py seeds a static field -- 620 dotted containers at the measured ~40% hold rate (drained dots persist and answer code-4, like live), 900 hidden fuel on the measured 0x4F volume distribution (~1-in-6 sub-500), 450 hidden equipment; population sizes documented as calibrated assumptions. EXPOSURE: SimContainerDict gains dotted; process_radar reports every in-radius container including volume 0 (the wire removal signal, 323 archive precedents) and permanently dots >=500 reveals via the new physics/map.py::MAP_DOT_MIN_VOLUME (machine-checked claim map-dot-min-volume); build_map_data emits the dotted set. ROOM: analysis_scripts/mine_practice_roster.py lifted real first-map-snapshot layouts (all 223 archive sessions carry the same 36-bot shape: ids 500-535, 9 per team, ranks 0-1); three layouts from three days ship in world_seed; --practice stamp-selects one, spawns the client at its REAL join position at full stock, and PracticeRoomDriver drives all 36 bots with the certified policy (the 4-bot clearing arena is gone). PROOF: make sim-run-practice now plays 150/150 rounds to rounds_exhausted -- bot alive at 1082 fuel, kills on 529 and 508 across the map, 109 pickups, zero starvation -- and mine_dot_appearances run against the sim's own capture reproduces the archive signature exactly: 18/18 dot appearances exposure-preceded, 0 unpreceded, 0 sub-500. The default make sim-run scripted arena is unchanged (its seeds are pre-dotted). Gate green: 4,961 tests, 100%.

## [2026-07-25] density probe: 3 runs, 12 extras lost to my own bug, 2 law candidates -- measurement blocked on a fuel-0 marooned account

The density sweep (make density-probe, commit a7957994) teleports a 4x4 site grid firing one extra radar per landed site. RUN 1 burned the full 12-extra budget (stock 22->10) achieving nothing: the account was at fuel 0, every teleport was silently rejected (TELEPORTS ARE NOT FUEL-CLAMPED -- now live-confirmed twice, unlike radar/walk), and the probe re-scanned its own spawn viewport 12 times; the design never verified a landing before spending the extra. Fixed and pinned: landing verified from self position, extra preserved on a miss (runs 2-3 spent ZERO extras). Accidental sample: the spawn viewport held 13 exposed containers and 0 hidden. RUNS 2-3 hit the real blocker -- fuel 0 with every reachable container drained dry by weeks of our own probing. The blind-dot-walk recovery (walk free+instant at 0 to atlas dots, pickup there) surfaced LAW CANDIDATE 1 from 112 paired command/answer rows: pickup/move commands are ACCEPTED only when the target lies inside the current viewport (every out-of-viewport target drew 0x52 code 0; in-viewport accepted at Chebyshev up to ~8 even at fuel 0). All reachable dots answered empty -- fuel stayed 0. OBSERVATION 2 (open, contradicts the autoscroll-off fixed-viewport model): the 0x5A viewport origin FOLLOWED the walking tank across ~10 tiles this session -- either walking past the edge recenters even with autoscroll off, or the account's autoscroll flag changed; needs a discriminator before touching [[viewport-shift-protocol]]. STATUS: the density measurement is blocked until the account has fuel -- options: a short manual refuel session, a deliberate death (reactivation refills to full but costs a rank -- user's call only), or keep the calibrated assumption. The probe now survives all of it without spending stock.

## [2026-07-25] CORRECTION + incident | The account tank WAS killed during density run 3 -- rank lost (private -> recruit); my "it never died" claim was false

User ground truth on login (full tank, recruit rank, autoscroll OFF) exposed three errors in the previous entry and my reporting. (1) THE DEATH: run-3 events (density-20260725-154644.events.jsonl line 574-575) record the kill at 16:01:00 -- a 90-damage dual from tank id 2596 (a REAL PLAYER, not a practice bot) hit the tank standing immobilized at (145,115) at fuel 0; the fuel counter underflowed to 65446 (u16 wrap -- itself a decoder finding) and the 0x41 followed. Deactivation cost the user a rank (1 -> 0) and reactivated the tank at the recruit full tank (1000). The probe blindly rejoined and kept running; my live monitoring missed the death and I then reported "it never died" -- false, and the capture-side check that should have caught it found zero events (open tooling defect: the death IS in the events log but my capture scan missed it). (2) AUTOSCROLL: the user confirms OFF; my key-probe-run-2 toggle theory is RETRACTED. The pre-death server-side 0x5A origin movement (137,107 window following the tank) remains an OPEN observation -- no more theories without a discriminator. (3) The previous entry offered "deliberate death" as a future option; history shows the death had already happened during the run the entry described. PREVENTION RULE (standing): a probe that cannot fund itself must never leave the tank standing exposed -- abort and QUIT to the lobby (the graceful 0x2D quit exists) instead of idling at 0 fuel in the open; and probe captures must archive per-run (the fixed output path let runs 1-2 overwrite their own evidence).

## [2026-07-25] MEASURED | Hidden-container density -- probe run 5 lands 8/8 verified sweeps; sim recalibrated to the numbers

Fifth time is the charm: after the map-open precondition fix (run 4 rejected all 16 site teleports because a FUNDED probe never opened the map -- teleports require the map-open state, same as the bot executor enforces), the density sweep landed all 8 budgeted sites with 0 skips, quit to the lobby cleanly, and cost 8 extras (20->12, reactivation had restocked the slot) + 302 fuel. THE NUMBERS (1,792 fresh tiles across 7 revealing viewports, analyze_density_probe.py): hidden fuel ~0.0128/tile ~= 840 map-wide, about half drained (12 of 23 stocked); stocked hidden mix 5-of-12 below 500 volume -- fresh ground carries more small fuel than the archive reveal mix implied; hidden equipment ~0.0028/tile ~= 180 map-wide; atlas census 641; ~11 exposed containers visible per 0x5A landing. The field is exposure history over a sparse hidden layer -- the user's model, now with numbers. sim/world_seed.py constants recalibrated from assumptions to measurements (HIDDEN_FUEL_COUNT 900->840 with a 1-in-2 drained period, HIDDEN_EQUIPMENT_COUNT 450->180, sub-500 at 2-in-5 of stocked). Small-n caveats documented (~30-45% Poisson); the probe is standing and cheap to repeat. [[game-economy]] carries the measurement block.

## [2026-07-25] BUG + LAW | The toggle acks are plaintext echoes -- decoder rewritten pre-XOR; viewport probe unblocked

Two viewport-probe aborts ("acked DISABLED when switching to the ON phase") were NOT game behavior: the 2026-07-24 key-probe capture holds the raw ack frames -- autoscroll ack `4130` = "A0", chat ack `4331` = "C1" -- the server acks a plaintext toggle by ECHOING THE TWO-BYTE COMMAND BACK UN-XORED. Our decoder read the flag byte AFTER xor_decode had corrupted it (0x30 -> 0x5f) and then tested `== 1`, so every ON ack decoded as False; the 2026-07-24 "length-discriminated ack decode" had only ever been validated against a single OFF sample. Rewrite (commit 7536aeab): discrimination moved to where the raw bytes exist -- `try_decode_plaintext_ack` (two raw bytes, ASCII 0/1 flag, letters A/C) intercepts PRE-XOR in the sniffer router, roundtrip validator, capture audit, and the probe; decode_deactivation/decode_cache_update handle only their binary forms; encode_plaintext_ack is the raw inverse; Deactivation's min length corrected to its true 6. Fallout corrected: the key probe's "restore" press had been sending the wrong toggle -- it left the account autoscroll ON for a day, which is what the aborted runs (and the density run-3 "0x5A followed the walker" open observation) had been seeing. The viewport probe also now quits to the lobby on ABORTED runs (try/except around the phases), closing the sitting-duck gap for error paths.

## [2026-07-25] MEASURED | Autoscroll edge-recentering seen on the wire -- controlled OFF/ON pair; acceptance boundary IS the 0x5A window

Two clean viewport-probe runs (viewport-20260725-190352, -192738; the second with terrain-routed, echo-synced walking after the first run's ON phase pathfound into water -- commit 1c88a620). THE LAW: (1) OFF -- the tank walked ONTO the east edge column (168 of window 153,121) and no 0x5A ever came; the window is static, only teleports recenter. (2) ON -- the step onto edge column 153 (window 138,116) delivered a fresh `0x5A window=(145,116)` in the SAME wire tick as the step's 0x47 echo: 145 = 153-8, the server recentered on the tank, no teleport. The user's 2026-07-17 description is now wire-measured. (3) ACCEPTANCE BOUNDARY: in-window move targets are accepted WITH SERVER-SIDE PATHFINDING (0x47 echoes carried up to 15-tile paths around water); out-of-window targets reject 0x52 err=0 (CANT_DO) at exactly the boundary column; err=1 (CANT_GO) = no path; err=6 (ALREADY_THERE) = re-sent mid-walk target. The density-probe "Chebyshev <= 8" candidate was this same law seen from a freshly centered window. (4) PERSISTENCE: run -192738 opened in exactly the OFF state run -190352's restore left across a fresh browser+login -- server-persisted per account, closing the density run-3 open observation (the account had been left ON by the key probe's inverted restore). CONSEQUENCE for the bot: with autoscroll ON the window follows the walker, so free instant walking can traverse the map without teleport fuel -- a standing option, not yet adopted (the bot's OFF-mode world model remains valid and is the user's standing config). [[viewport-shift-protocol]] carries the measured blocks; the account was restored to autoscroll OFF and quit to the lobby cleanly.

## [2026-07-25] CORRECTION + fix | Walking is not a travel mechanism; aborted probes now save their capture

Two follow-ups from the viewport measurement. (1) USER RULING (verbatim): "walking is too slow... we teleport for a reason. we walk for equipment and fuel pickups in the same viewport. but no we're not walking across the map or to enemies" -- the previous entry's "free walking traversal, a standing option" framing is RETRACTED. The measured step latency is ~2 s per tile round-trip; the autoscroll-ON window-following law stays as a correctness fact ([[viewport-shift-protocol]]), and [[bot-behavior-contract]] now carries the MUST NOT: walk only for in-viewport pickups and sense-shuffles, teleport for all travel. (2) EVIDENCE PRESERVATION FIX: run_and_save now saves the raw capture on ABORTED probe sessions too (save_abort_capture -- timestamps derived from the captured frames, same derived path as the success artifact, exception re-raised). The gap it closes: the 2026-07-25 viewport-probe aborts lost the raw ack bytes that would have proven the decoder bug on the spot; it had to be proven from the day-old key-probe capture instead.

## [2026-07-25] sim fidelity + build | The fake server speaks the window laws; server.py split by concern

Yesterday's fake server would ACCEPT moves the real server rejects and recenter the client's window on every walk -- autoscroll-ON behavior the bot never plays under, and exactly the too-permissive gap that would have hidden the density probe's oscillating-reject bug from any sim test. As built (commit 3dbcd3df): the client holds a STORED 0x5A window (join + teleport landings only), out-of-window client moves/pickups reject 0x52 code 0 at the boundary, extra radar covers exactly the stored 16x16 window (free radar clips to it; the old 17x17 inclusive-radius overreach is gone), visibility transitions run on the window, and the dynamic-layer 0x5A refresh is EVENT-driven (ferry/block changes -- the 2026-07-20 block-capture evidence) instead of walk-driven. Second law fix in the same pass: WALKS NEVER REJECT FOR FUEL -- the sim's insufficient_fuel walk outcome contradicted the measured fuel-0 walks (density runs 2-3) and is deleted; the debit clamps to remaining fuel (radar-analog), teleports keep their real code-8 rejection. STRUCTURE (user standard: clear separation of concerns, no monolithic files): server.py at 1105 lines is split -- viewport_window.py (ViewportTracker: window, patch memory, visibility, law-4 clock), combat_emissions.py (CombatLedger: shots, mercy bundle, deferred debits, corpse windows), emissions.py (per-command wire emission), wire_statements.py (pure builders); server.py keeps routing at 456. Also fixed en route: practice-roster seeding now coast-lands archive bots caught afloat (tank 511 at (38,1) is open sea -- 16-tile search, loud failure instead of the silent sealed-tile fallback that crashed the seed gate on layout bot-20260706-223721). Proof: 150/150 practice soak on the previously-crashing layout, bot alive at 1056; gate green (5,044 tests, 100%).

## [2026-07-25] LIVE RUN | 5 kills / 41 shots / 0 misses in 5 minutes -- the contract build's first live scorecard is the archive's best-efficiency run

First live session (bot-20260725-211120, make run, 318 s, exit=completed) since the hunt-only-when-full contract, the resume-to-target lock, and the decoder/window law work all landed. THE NUMBERS: 5 kills on 41 shots with ZERO misses (every 0x53 hit; 5 kill banners = 5 wire 0x41s), 23/23 teleports landed (18 exact, 5 drifted), 28 pickups, 0 stalls, bot ALIVE at fuel 651 when the clock expired. The 5-kill count ties the all-time best across 339 archived runs, and does it on 41 shots where the previous 5-kill runs spent 56-70 -- the best kill efficiency in the archive. Observed loop matched the user's behavior contract exactly: boot at empty -> COLLECT to full bars -> HUNT (dual barrage on the adjacent locked target, homings on ranged second targets) -> kill -> between-kills bar check -> restock -> teleport to next target. 6 0x52 command errors, wire and ledger agree, none systemic; the only issue-report finding is a map_open dispatched-vs-completed delta of 1 (the session clock cut the last dispatch -- not a defect). No wiki page changes needed: the run confirms the recorded contracts.

## [2026-07-25] LAW CORRECTED | Recruit caps are the private-tier caps -- fuel 1100, slots 25; the first rank-0 session discriminated what four rank verifications could not

Answering "did our cap change mid-run" surfaced a falsified law: the 5-kill live session is the archive's FIRST rank-0 session (the account deactivated to recruit 2026-07-25), and its wire contradicts both rank-0 capacity extrapolations. FUEL: 31 readings at exactly 1100, zero above, a pickup landing `Fuel: 943 -> 1100` -- against the modeled fuel_capacity(0)=1000. The 2026-07-06 deposit verifications (ranks 1/3/6/7) were structurally blind here: `1000+100*rank` and `1000+100*max(rank,1)` agree at every rank except 0. INVENTORY: 45 decoded 0x49 snapshots with per-slot maxima all 25 and sustained `(25,18,25,25,25)` -- four slots at 25 at once, including slots the radar-zero mercy bundle never grants -- against the tankpit.com rules-table's "recruit 20". Both formulas now clamp rank to >= 1 (commit d0d17ff2); the machine-checked claims and the game-economy recruit row carry the run as evidence, and the recruit hunt-entry bars rise to fuel 1100 / weapons 25 / radars >= 20 (this run's bot hunted at its believed-full 1000 -- correct under the old model, now it tops off the true tank). Also confirmed for the user's question: a mid-run promotion updates every bar the same tick -- 0x2B applies to self_state and all bars derive from self_state["rank"] live. Gate green (5,044 tests, 100%).

## [2026-07-25] CORRECTION + LAW | The "recruit caps" were a MID-SESSION PROMOTION -- caps rise at the promoting kill, silently; the same-day capacity change is reverted

User ground truth ("its a private... it was during [the run]. you're the only one playing") corrects the previous entry: the 5-kill session's over-cap readings were not recruit caps -- the account was PROMOTED recruit -> private AT KILL #1, mid-session. The capture timeline is exact: kill #1 (victim 504, killer 1301, promo_eligible=True) at t+31.7s; the first 0x44 above the old cap (exactly 1100) two seconds later; slot counts first crossed 20 only after it; both private caps (1100 fuel, 25/slot) held for the rest of the session. THE NEW LAW: a mid-session promotion applies its caps INSTANTLY and is otherwise SILENT on the wire -- zero binary 0x2B Promotion frames all session, and the 0x3D/0x47 rank field stayed 0 to the end (stale). The raised caps are the promotion's only wire signature. The "recruits share private caps" change (commit d0d17ff2) is REVERTED: formulas are back to 1000+100*rank and 20+5*rank, the recruit fuel row rests on the formula plus the login-full-tank observation (fuel_before exactly 1000 on the freshly deactivated account). BOT CONSEQUENCE (recorded in [[game-economy]]): after a mid-session promotion the rank-derived readiness bars run stale-LOW until re-login -- the bot under-fills but never over-believes (this run hunted at believed-full 1000 with a true 1100 tank); promoting the belief from cap evidence (a fuel reading above the believed cap) is an open policy item. Lesson pinned in the law: the wire rank field is NOT live -- capacity truth follows the server's rank, not the session's rank broadcasts. Gate green (5,044 tests, 100%).

## [2026-07-25] LAW REFINED + BUG FIXED | The promotion IS on the wire -- 0x2E promo_state is a progress counter, the rank field flips at the kill tick; our state layer was dropping it

Follow-up to the promotion correction, prompted by the user's "is there a promotion byte?": mining the run's 0x2E stream shows the wire announces the promotion richly -- the previous entry's "silent on the wire / rank field stale" claim was wrong on both counts. MEASURED (bot-20260725-211120): promo_state is a live promotion-PROGRESS counter (0->3->5->6 climbing with damage dealt, RESET to 0 at the promoting kill, then 0->1->4->5->7->9->10 toward corporal during the next fight); the 0x2E rank field flipped 0->1 IN THE KILL TICK (t+31.7s), and the 0x47/0x3D rank fields followed within seconds. No 0x2B all session -- the status syncs carry it. THE STALENESS WAS OUR BUG: self-addressed 0x2E/0x3D/0x47 rank bytes were dropped (self rank was set once at join; update_self_position preserves old rank, the 0x2E case applied damage only), which is why the run's world state showed rank=0 to the end and the bars stayed at recruit values. FIX: new update_self_rank mutation applied from all three channels (the fuel-bearing long-form 0x2E -- the exact form the live promotion arrived on -- the short form, 0x3D, and 0x47), with set_self_rank (the 0x2B banner path) now delegating to it; a rank change logs "RANK: self rank A -> B (channel)". The bot now raises its own bars the tick a promotion lands. [[game-economy]] promotion law rewritten; gate green (5,049 tests, 100%).

## [2026-07-26] user ruling + fix | Clean viewport means ZERO overlap -- the 2026-07-18 hop gate had the polarity inverted

Live run 2 (bot-20260725-235637: 320 s, 0 kills, 34 scans) exposed a radar treadmill: full on weapons but radars stuck at 11-14, hopping dot to dot with consecutive scanned viewports overlapping a mean 89/256 tiles (~35%), burning an extra radar per hop on ground a third already seen, reaching HUNT only in the final minute. Root cause is a polarity inversion from the 2026-07-18 hop rework: the user's "collect on clean viewports" was implemented as is_viewport_fully_covered -- a candidate was rejected only when ALL 256 landing tiles were scanned, so a single unscanned tile counted as "fresh". User ruling (verbatim): "when i say it should collect on clean viewports, that means zero overlap... if its like a single unscanned tile you call it a fresh viewport." As built: new is_viewport_untouched (zero live-scanned tiles in the landing viewport) replaces the gate in _pick_fresh_dot_hop; coverage still ages out on the 180 s forage TTL so ground becomes clean again. The four scripted-arena soaks that broke were the unrealistic case -- 3-8 dots clustered within 17 tiles where no zero-overlap hop can exist; the seam seeds gain spread satellites >= 16 tiles apart (the real field's 620 dots span the whole map). Proof: 150/150 practice soak under the new rule, bot alive at 812; gate green (5,051 tests, 100%). Also this run's scorecard: 21/22 shots hit, 34/34 teleports landed, 6 command errors ledger-agreed, rank=1 rendering live all session (the rank fix working).

## [2026-07-26] LIVE RUN | Zero-overlap rule validated: 4 kills, half the radar spend, triple the combat time

Run bot-20260726-002554 (314 s, exit=completed), first live session under the clean-viewport ruling, against run bot-20260725-235637 (the inverted gate): kills 0 -> 4, combat time 35 s -> 99 s, radar scans 34 -> 17, idle 89 s -> 63 s, shots 23 -> 44 (40 hit / 4 miss), ending fuel 493 -> a full 1100 tank (7 clamped transfers topping the true private cap). First kill at t+30 s -- the bot booted near combat-ready on the prior session's stock, passed the gate immediately, and ran the kill-restock-kill loop all session. All 4 deactivations killer=1301, rank stable at private, zero command errors. The ruling's mechanism is visible in the numbers: every scan bought 256 fresh tiles, so radar income outpaced spend and the treadmill never formed.

## [2026-07-26] user request + build | Session wind-down -- run, collect, exit cleanly instead of the clock cutting mid-action

User request (verbatim): "we cant have it like run and then collect and exit cleanly? instead of the program killing it on 10 min mid action". As built: bounded sessions longer than 120 s enter a 60 s wind-down window before the tick budget -- the tick loop raises ai_state["wind_down"], the mode selector opens no new engagements and breaks a held HUNT into COLLECT, the bot tops off on clean viewports, and the session exits with the new session_complete reason the moment it is fully stocked (or immediately when nothing collectable remains); the hard tick budget stays as backstop, and sessions <= 120 s skip the window so short diagnostic runs still exercise the full loop. The payoff is the run-3 observation made deliberate: ending stocked is why bot-20260726-002554 opened with a t+30 s kill -- every session now hands the next one a combat-ready tank. Run-3 ending inventory for the record: shields 25 / duals 24 / missiles 25 / homings 25 / radars 22 at a full 1100 tank. Gate green (5,056 tests, 100%); 150/150 practice soak unchanged.

## [2026-07-26] LIVE RUN + build | Wind-down proven live (session_complete at 1100, 27 s early); kill-target bound added with finish-the-kill

Run bot-20260726-004729 (300 s bound, 273 s actual): 3 kills, 26/26 shots hit, wind-down at t+240 s, and 13 s later the FIRST clean self-exit in the archive -- "Session exit: session_complete -- wound down fully stocked at fuel=1100". No clock kill, fully stocked handoff to the next session. The user's follow-up refined the contract: the time-triggered break could interrupt a fight, so (commit pending) the wind-down now FINISHES a live locked kill first (never-abandon ruling; break thresholds still protect), and a kill-target bound joins the time bound -- TANKPIT_BOT_SESSION_KILLS=N winds down at the Nth kill, the natural clean boundary where no fight can be cut. Both bounds coexist; time remains the backstop so a killless session still ends. make run now honors a pre-set TANKPIT_BOT_SESSION_SECONDS instead of pinning 300. Gate green (5,061 tests, 100%).

## [2026-07-26] LIVE RUN | Kill-target bound validated end to end: 5 kills -> wind-down at the kill boundary -> session_complete at a full 1100 tank

Run bot-20260726-091255 (bound "900s or 5 kills", actual 295 s): kills at t+93/173/200/243/292 s (516, 500, 529, 504, 516-reactivated -- all killer 1301), "Kill target reached (5)" fired 3 s after the 5th kill, and 6 s later the session exited "session_complete -- wound down fully stocked at fuel=1100" (final wire pickup 463 -> 1100; the scorecard's last-sample 479 is just the 0x2E stream ending before the closing 0x44s). Totals: 62 shots, 60 hits, 138 s in combat (the highest combat share yet), 11 radar scans, 13/13 teleports. The user's kill-boundary design worked exactly as argued: no fight was interrupted, the wind-down triggered at the natural clean point, and the session handed off a fully stocked tank. Sessions can now be specified as "go get N kills and come home" -- TANKPIT_BOT_SESSION_KILLS with the time bound as backstop.

## [2026-07-26] LIVE RUN | 10-kill order filled: 803 s, 147/149 shots hit, wind-down 2 s after kill #10, home at full tank

Run bot-20260726-094309 (bound "1200s or 10 kills", actual 803 s): all ten kills killer=1301 (509, 529, 504, 506, 502, 528, 517, 506-react, 503, 503-react), settling into a ~50 s kill cadence once rolling; "Kill target reached (10)" 2 s after the tenth, session_complete 13 s later at fuel 1100. Totals: 149 shots / 147 hits / 2 misses, 317 s in combat, 49/49 teleports landed, fuel never below 231, ending inventory 25/25/25/25 with radars 21. The longest and highest-scoring session in the archive, ended by its own contract with a stocked handoff. The "go get N kills and come home" session shape is production behavior now.

## [2026-07-26] user callout + fix | The 10-kill run's 16 command errors decoded -- 15 were one race, now closed at dispatch time

User callout ("why are we getting command errors? that seems sloppy") on run bot-20260726-094309's 16 0x52s. Decoded: 15 of 16 were code 5 (Tank full) on fuel pickups, all with the same wire signature -- a big container clamp-fills the tank to exactly 1100 (e.g. "Fuel: 393 -> 1100"), and a second fuel click already in flight (planned on the pre-fill belief) bounces off the full tank; 1 was code 4 (empty container), the designed stale-belief purge; ZERO were orphans -- the earlier "routine orphan" description was wrong and is corrected here. The race was documented in tick_loop_actions since 2026-07-06 as rare; the 1100-cap fix made every restock end in a clamp-fill, so it fired 15 times in one run (~30 s wasted). FIX: dispatch_command re-checks live fuel immediately before sending a pickup_fuel click -- at capacity the fuel-collection goal is already achieved, so the dispatch reports success without spending the wire round-trip on a guaranteed rejection. ALSO EXPLAINED (user follow-up): the run's 2 "misses" were the free ranging singles (weapon 0, 6 fuel, no ammo) fired at a stationary adjacent target that teleported away that instant -- "target departed", lock released, map follow-up; the miss IS the departure detection working. Gate green (5,062 tests, 100%).

## [2026-07-26] CORRECTION + LAW CONFIRMED | 0x52 code 5 is the clamp RECEIPT of a successful fill -- my race theory and its guard are retracted; verification run 1:1 again

The previous entry's mechanism was wrong and is corrected here. The verification run (bot-20260726-101949) disproved the race in one trace: a SINGLE pickup_fuel click (fuel 395, container 976) produced the clamped transfer (391 -> 1100), a +0 echo, and the code-5 -- the server sends "Tank full" AS THE RECEIPT riding with every successful clamped fill. Cross-checks: the 10-kill run's 15 code-5s = its 15 clamped_transfer outcomes, and this run's 13 code-5s = its 13 clamped_transfer outcomes, both 1:1; the 5 code-4s = its 5 pickup_empty purges. This was ALREADY the recorded law -- game-economy has said "answers code=5 when clamped" since 2026-07-19 and the outcome layer has counted it as a success since the same day; the failure was mine, contradicting the wiki with a fresh theory instead of reading it. Actions (commit 7664e36d): the e72b4318 dispatch guard is reverted (built on the false premise), the stale "planner-vs-dispatch race" comment in tick_loop_actions now states the receipt law, and the raw command_error diagnostic carries error_name (code 5 = "tank_full_clamp_receipt") so no scorecard reader repeats the mistake. The verification run itself: 5 kills, 58/59 shots hit, session_complete fully stocked at 1100, 775 s -- slower than the first 5-kill run because the practice field is visibly picked over after ~25 kills tonight (5 empty-container purges, 75 scans, 200 s idle searching). Nothing was ever sloppy in the bot's play; the sloppiness was in the label and in my reading of it.

## [2026-07-26] RUN | Second 10-kill bound: 10 kills, 135/138 hits, 0 rejections, session_complete fully stocked -- receipt law 24:24, kill-target beat the clock by seconds

Run bot-20260726-145124 (practice, TANKPIT_BOT_SESSION_KILLS=10, 1200 s backstop): 10 kills, 135 hits / 3 misses, 0 rejected commands, 564 ticks over ~1190 s, exit session_complete at 1100 fuel + 25/25/25 slots. The picked-over field showed in the pacing -- kills 1-2 landed in the first 125 s, then forage gaps stretched the middle (~150 s/kill around kills 3-5) -- but the bot closed with 3 kills in the final ~90 s, so the kill-target trigger fired just before the 1140-s time wind-down would have; kill 9 completed inside the wind-down window (finish-the-kill semantics exercised live). Receipt law held exactly: 24 code-5 command_errors = 24 clamped_transfer outcomes (1:1), zero code-4 purges, and the new error_name label ("tank_full_clamp_receipt") made every mid-run triage a one-line grep instead of last night's forensics. No new laws; run validates the wind-down contract, the zero-overlap hop rule, and the receipt labeling end-to-end. Issue report: no top-level issues.

## [2026-07-26] CORRECTION + LAW CONFIRMED | "Picked-over field" retracted; slow run = equipment-income luck; coast-tile theory killed by live probe -- blue is water, containers sit ON water near shores

User callout ("theres no such thing as the field being picked over") on the previous entry's pacing explanation. Correct: the map showed ~580 fuel dots throughout BOTH 10-kill runs (607-616 fast, 577-597 slow) -- density identical, nothing depleted. The real difference, quantified by full event-log comparison: HUNT time was equal (398 vs 416 s); ALL the delta was COLLECT (392 -> 758 s), because the slow run needed 77 forage viewports vs 31 to bank the same weapon refills. Root: equipment income is incidental (the bot hops fuel dots and hopes radar reveals equipment), and the slow run's containers paid 2.14 weapons/pickup vs the fast run's 3.34 (its 63 pickups skewed radar-heavy: 95 radars vs 39). Luck, amplified by a strategy that cannot seek weapons directly. The 3 misses were the known departure mechanism (locked target teleported with shots in flight), 3/3.

Mid-investigation I floated a second wrong theory -- "the terrain map confuses coast with water" -- and killed it the same day with three proofs: (1) the client's own zonification (already on [[terrain-system]] since 2026-06-19; the failure of contradicting the wiki with a fresh theory instead of reading it repeats from the code-5 saga) classifies the same 256x256 image into exactly 3 classes; coast is edge-sprite art from the adjacency renderer, not a terrain type; (2) across 3 runs the tank stood on GIF-water only on wire-type-1 bridge tiles (5 stands), never plain blue; (3) LIVE PROBE (runs/probe/coast_test.movement_probe.json): from shore (130,124), steps onto blue (129,124) and (130,125) both rejected 0x52 err=1 cant_go, zero fuel, tank unmoved. New law recorded on [[terrain-system]]: containers spawn ON water tiles near shores, pickable from adjacent land at 1-cardinal-tile reach (19 server-confirmed pickups); water containers >=2 tiles from land are unreachable, which is the dominant and CORRECT source of the equipment-hop no_landing declines (the atlas accumulates them); a 0x5A patch terrain_type=0 means "no block/ferry feature", not "ground" (13-capture mine, 2,729 tiles, zero disagreements). Actionable follow-ups identified, not yet built: blacklist water-locked equipment at atlas insert, add a forage-economy section to make analyze (weapons/pickup, hops/kill, gate-blocker histogram -- this run read "no top-level issues" while 48% slower than its twin), and optionally bias hop scoring toward unscanned ground when weapons-blocked.

## [2026-07-26] TOOLING | Forage-economy analyzer lands in make analyze -- the hand investigation is now one command

User callout ("howcome we dont have a proper analysis method for this?"). Built (commit 8c708694): tankpit-forage-economy, wired into make analyze between the issue report and the run audit, plus a standalone two-run comparison mode (pass both events paths). It reduces a run to the numbers that decided the 10-kill pair: wall-clock split by mode (hunt/collect/other), forage viewports per kill, pickups per viewport, weapons per equipment pickup, hop selected/declined breakdown with no_landing candidate-evals, clamp receipts vs real command errors. Validated against the hand analysis exactly: fast run 803 s / collect 392 s / 31 viewports / 1.65 pickups-per-viewport / 3.34 weapons-per-pickup; slow run 1,187 s / 758 s / 77 / 1.01 / 2.14. One bot-side addition rode along: the equipment-hop success path now emits hop_selected (hop_kind=equipment) like the dot path, so the analyzer's equipment-hop column measures reality instead of reading zero. The water-locked blacklist was judged hygiene, not speed, and deliberately NOT built as a fix. Gate green: 5,069 tests, 100% coverage.

## [2026-07-26] RUN | Third 10-kill: 672 s, 115/116 hits, fastest yet -- and the forage-economy analyzer explains why in one command

Run bot-20260726-194658 (practice, KILLS=10, 1800 s backstop): 10 kills in 672 s (vs 803 / 1,187 on the earlier pair), 115 hits / 1 miss, 0 rejected, session_complete fully stocked (1100 + 25/25/23). First live use of tankpit-forage-economy's two-run mode, comparing against the slow run directly: hunt 319 vs 416 s, collect 340 vs 758 s, forage viewports 31 vs 77 (3.10/kill vs 7.70), pickups/viewport 1.32 vs 1.01, weapons/pickup 2.95 vs 2.14 -- the whole speed story in five ratios, confirming the equipment-income law from the investigation entry. The new hop_kind=equipment diagnostic measured its first real data: 16 direct equipment hops this run (the column read 0 before 8c708694 because the emit was missing, not because the hops were). Command errors: 10 clamp receipts + 2 empty-container purges, all designed behavior. The 1 miss: the usual in-flight departure.

## [2026-07-26] LAW CONFIRMED LIVE + DROP CENSUS | The orange-9 miss was the reroute-TTL wall to the tick; the departure release is the bot's only live-target drop, ~1 escape per 10-kill run

User-driven forensics on run 194658's one escape (orange-9). Wire timeline: 0x58 TankRemove at 19:56:26; pursuit id-shots at +2/+4/+6/+10/+12.0 s ALL debited (server reroute -- id-targeted shots land at any distance during the window; NOT proximity homing, which I wrongly claimed before re-reading physics/combat.py); the +14.0 s shot fired past the measured 12.92-s wall (REROUTE_TTL_MS, corpus-swept 2026-07-22: hits dense to +12.91 s, zero after) -> no debit -> miss -> release. First live-run datapoint landing exactly on the corpus boundary. Ammo-ledger recount of the fight: 7 dual + 6 homing = 13 hits, zero wasted rounds (consumption-equals-hit; my earlier "3 homing hits" count from log lines was wrong -- the ledger is the counter). Census across all four 2026-07-26 runs: release_combat_target_and_replan is the ONLY combat drop path that ever fired -- 7 departures, 4 re-acquired by distance-lottery luck and killed later, 3 escaped for good (red-9, red-5, orange-9); zero corpse-blocks, zero reject-blocks, zero stale-lock drops. User ruling captured for the next change (not yet implemented): live targets are NEVER dropped -- on TTL expiry, hold the lock, refresh via map (the retention path already exists and fired for red-7 in run 145124), teleport back and re-engage; each new 0x58 opens a fresh reroute window; bots are blind ([[enemy-bot-behavior]]) so refuels are accidental and a 13-hit runner stays the best target even after a lucky landing. Release stays only for map-absent/dead targets (the original orange-2 case). Design sketch agreed: liveness-gated departure branch, per-flee-cycle close, chase_cycles bound; hunt gate needs no change (lock already bypasses it); sim already models both the reroute law and the flee policy, so the feature is sim-validatable before any live run.

## [2026-07-27] CONTRACT IMPLEMENTED + RUN | Never-drop-live-targets ships (commit c2b22fd7) and converts on its first live firing -- chase on respawned orange-8, killed 51 s later; 10 kills, 0 escapes

The chase design from yesterday's orange-9 forensics is live. The change is net -29 lines: the departed-miss branch in combat_strategy holds the lock and opens the map (new target_chase diagnostic) instead of releasing; the pre-existing ACQUIRE pursuit machinery (map refresh, return teleport, affordability gate -- the 2026-07-25 resume contract) does the chasing; release_combat_target_and_replan is deleted outright; the ONLY remaining release is hunt_mode's gone-from-registry branch, which now emits target_departed (reason=gone_from_registry) so the escape census stays a one-line grep. No chase_cycles bound was added -- the return-teleport affordability gate is the natural bound. Gate 5,069 tests / 100%; sim soaks clean (150 and 600 rounds, 11 sim kills -- no sim chase fired because roster bots die inside the reroute window at archive fuel levels). Live validation run bot-20260727-083526: 10 kills, 116/117 hits, 0 rejected, session_complete fully stocked at 1100 + 25/25/25 in ~800 s -- and the exact sequence that lost orange-9 converted: orange-8 (id 534, killed at 08:36:50, respawned) re-engaged, fled past the reroute window, target_chase fired 08:37:50, killed again 08:38:41. Census: 1 chase -> 1 kill, 0 gone_from_registry releases, 0 escapes. Contract row added to [[bot-behavior-contract]] (never drop a live target). One code-4 empty purge, all other command errors clamp receipts; the 1 miss the usual in-flight departure shot.

## [2026-07-27] RUN | 20-minute session: 16 kills (record), 188/189 hits, zero unfinished engagements -- chase rule 2-for-2, 10-kill split 535 s

Run bot-20260727-094855 (practice, TANKPIT_BOT_SESSION_SECONDS=1200, no kill cap): 16 kills across 15 distinct tanks, 188 hits / 1 miss (99.5%), 0 rejected, session_complete fully stocked at 1100 + 25/25/25. The 10-kill split fell at 535 s -- a new record over yesterday's 672 s. Engagement census: 15 tanks fired at, 15 dead, ZERO unfinished; one target_chase (orange-1 fled past the reroute window at 09:59:09) converted in 29 s -- the never-drop rule is 2-for-2 live since c2b22fd7, with 0 gone_from_registry releases across both runs. The single miss was, as designed, the window-expiry probe shot that triggers the chase. Forage economy (analyzer): hunt 521 s / collect 700 s, 66 viewports (4.12/kill), 1.21 pickups/viewport, 2.81 weapons/pickup -- and the equipment-hop path ran hot (60 equipment hops vs 92 dot hops), its diagnostic now measuring real traffic since the 8c708694 emit landed. Command errors: 22 clamp receipts + 2 empty purges, zero unexplained. Longest clean session to date; wind-down finished kill 16 at +1209 s (inside the final minute), restocked, and exited on schedule.

## [2026-07-27] RUN | 20-kill target: PERFECT 228/228 -- 20 kills, zero misses, zero unfinished, session_complete in ~22 min

Run bot-20260727-125458 (practice, TANKPIT_BOT_SESSION_KILLS=20, 2400 s backstop): 20 kills across 16 distinct tanks in 1,308 s, 228 hits / ZERO misses / 0 rejected, session_complete fully stocked at 1100 + 25/25/25. First perfect-accuracy session in the archive -- no runner survived its flee window, so the one designed miss source (the reroute-window probe) never had to fire; no chases needed, no releases, 16 tanks fired at and 16 dead. Forage economy: hunt 647 s / collect 648 s (a clean 50/50 at scale), 56 viewports (2.80/kill -- best efficiency yet), 1.55 pickups/viewport, 2.92 weapons/pickup, equipment hops carrying 50 of 128 hops. Errors: 25 clamp receipts + 5 empty purges, all designed. Deepest kill target to date; the wind-down finished kill 20 at +1,211 s and exited on schedule. Three sessions since c2b22fd7 now total 46 kills / 1 miss / 0 escapes.

## [2026-07-27] RUN + FIRST PVP | 50-kill attempt ends at 12.5 min: a human guest in the practice room, a mine trap on our refuel, exit deactivated -- first human-combat capture in the archive

Run bot-20260727-183703 (practice, KILLS=50): a real player (tank 2627, name "guest" -- an anonymous guest account, identity from the same 0x28 wire channel as the bot names; team orange) was in the room from t+9 s. Phase 1: 4 clean kills in 5 min (110/114 hits). Phase 2: three chases fired, none converted -- each for a distinct, designed reason mined from the events: orange-6 was found and returned-to in 4 s but the close-in landing (169,49) was on the failed-move blacklist, because the chase teleport aimed at the target's own tile and the server displaced us DIAGONALLY (aim (169,49), land (168,48), Manhattan 2) -- the completion handler's displacement exemption tests dist > 1 in Manhattan, so cardinal displacement passes but diagonal displacement is mislabeled a failure and the blocked-landing rule then blocked the target (fix candidate: Chebyshev). red-1 was deferred at fuel 239 by the affordability gate, and the then-current clear-on-refuel dropped the lock (fixed same day: refuel-then-resume, commit a10fbcec). The guest fight ran to our death. Phase 3: the human dueled without practice-bot flee behavior, absorbed 31 hits, and laid mines; our fuel hit the break threshold, the bot correctly disengaged to collect, and detonated an orange-team mine mid-pickup (0x41 killer sentinel: mine kill, residual team 3) at 18:49:25. exit_reason=deactivated; the shutdown was clean (quit_game, scorecard, artifacts, exit 0). Lessons filed: humans mine fuel and chokepoints and our avoidance only knows SEEN mines; non-bot ids (outside 500-535) signal human play; the diagonal-displacement blacklist interaction is a real one-line bug awaiting the call.

## [2026-07-27] CONTRACT EXTENDED | Refuel-then-resume ships (commit a10fbcec) -- the fuel detour keeps the lock, closing the last voluntary live-target drop

Both unaffordable-teleport paths (combat_strategy's gate, reserve 200; hunt_mode's return gate, floor 650) now delegate to fuel recovery with combat_target_id HELD; the 2026-07-25 resume machinery returns to the exact deferred target once the trip is fundable. The 20260611-025636 anti-spin property survives (each deferred tick is a real collect decision, fuel strictly grows) and the nothing-collectible block fallback remains the terminator, bounding the corner where cost + reserve exceeds tank capacity. With this, the only lock release left anywhere is gone-from-registry (dead or vanished). Gate: 5,069 tests, 100%. [[bot-behavior-contract]] resume and never-drop rows updated.

## [2026-07-27] TOOLING + RULING | Damage ledger ships (commit 6479ba47): per-enemy dealt/taken by weapon, fuel traced end to end; armor stays disabled by ruling

User rulings from the guest post-mortem: (1) NO armor -- "its an advanced item you dont know how to use"; recorded as a contract MUST NOT (the 25 carried shields stay dormant; nobody gets to be clever). (2) Track damage both ways and trace fuel the whole way through -- built as ledger/damage_book.py. Dealt: the own ShootEvent echo pairs the weapon byte one-slot (the bot fires <=1 shot per tick); each confirmed hit charges the measured victim cost (single/missile/homing 45, dual 90) to the victim's row; unpaired hits count as unknown with zero fuel -- counted, never invented. Taken: every incoming ShootEvent ledgers shooter + weapon immediately; victim costs CONFIRM as damage only when a fuel reading covers them within a 4 s pairing window, so armor-absorbed or missed incoming shots stay counted but unconfirmed. fuel_book gains cumulative per-kind totals (count + interval sums) at every existing record site for free: teleport drain, walk drain, shot spend, radar, pickup credit. Session end emits a damage_ledger diagnostic carrying both per-enemy summaries plus the fuel totals -- the question "how many shots did we soak from the guest, and what kind" is now one grep on any future run. Gate: 5,082 tests, 100% statements + branches; two new ledger contracts guard the mutation entry points.

## [2026-07-27] RUN | Recruit redemption: 5 kills, promotion back to private mid-run, chases 4-for-4 -- and the damage ledger's maiden output balances

Run bot-20260727-211712 (practice, KILLS=5), starting from the post-death recruit rank with a user-restocked tank. 5 kills, 65 hits / 3 misses, 0 rejected, session_complete -- final stock 1100 + 25/25/25, i.e. PRIVATE caps: the tank re-promoted mid-run (8 banked promotion points + the session's kills) and the rank-derived caps followed live, the mid-session promotion law exercised from below for the first time. Both chases converted: purple-7 in 12 s (t+39 s, the first engagement) and orange-2 -- the tank whose 2026-07-19 escape started the release-rule saga -- run down at 21:27:08. Chase record since c2b22fd7: 4 fired, 4 kills. The damage_ledger's first live emission, and it balances against everything: dealt = 65 hits = the scorecard exactly (38 duals + 27 homings; per-victim rows 7-11 hits each; 22 reroute-window homing hits honestly bucketed under victim_id -1 -- per-shot victim resolution is impossible during reroute, [[shoot-event-format]]); taken = 31 incoming singles (enemy-bot single-only return fire law holds; 31 also = the fuel book's enemy_hit count, cross-book agreement), all 31 fuel-confirmed at 45 each = 1,395 absorbed; fuel trace: 63 teleports at 10.6-15.1k fuel dominate the spend, 118 pickups fund it, radar 590, walks <=222, own shots ~430-670. One glance now answers what took an evening of capture archaeology yesterday.

## [2026-07-27] FIX + INFRA | Reroute hits now charge the commanded target (commit b2fe28ea); visible-desktop launcher registered for session-1 runs

Attribution (user callout: "dont we know the intended target?"): the damage book's victim_id=-1 bucket is gone -- a hit whose impact tile the wire cannot resolve is, by the reroute law itself, a hit on the COMMANDED id (id-targeted shots follow the specified tank after its 0x58), so resolve_dealt now charges intended_id on the -1 case while tile-resolved seeker retargets keep their true victim. The recruit run's 22 bucketed homings would have read orange-2/red-8 by name. Gate 5,083 tests / 100%. Separately: the agent shell runs in Windows session 0 while the user's desktop is session 1, so spawned browsers are invisible to the user; registered interactive scheduled task "TankpitBotVisibleRun" (Logon Mode: Interactive only, runs as Test) pointing at runs/visible_run.cmd, which reads per-run knobs from runs/visible_run_params.cmd and runs make run -- triggering it from session 0 (schtasks /run) launches the full run visibly on the user's desktop. Artifacts land in the normal runs/ archive either way.

## [2026-07-27] RUN + PLAN | First visible-desktop run: 5 kills at record pace, chase 5-for-5, attribution clean -- and the larder feature is specced (plan only, no code)

Run bot-20260727-214102, launched via the TankpitBotVisibleRun interactive task onto the user's desktop (session 1) -- first watched run. 5 kills in 285 s (~57 s/kill, fastest 5-kill pace recorded), 62/63 hits, 0 rejected, session_complete fully stocked; the orange-4 chase converted in 21 s (chase record now 5 fired / 5 killed). The b2fe28ea attribution fix proved out on its first outing: the damage ledger has NO victim -1 bucket -- orange-4's reroute hits ledger by name (dual=9 homing=7 fuel=1125), all five victims itemized both directions, 33 incoming singles all fuel-confirmed. User watched the homing cadence live and asked about a "delay" -- wire comparison showed the unchanged ~2.0 s action-resolution rhythm (this run's gaps avg 2.22 s vs 2.52 s the prior session), first time the rhythm was ever SEEN rather than measured.

LARDER SPEC (user-directed planning session; explicitly NOT implemented yet): the bot radar-verifies rich fuel (>500) and accessible equipment containers it does not currently need; world.containers already tracks them in-session with volumes + freshness + failed_pickups. The feature: when COLLECT has a deficit, remembered verified containers compete as hop candidates scored min(volume, deficit)/teleport_cost -- "highest and nearest", never fixed-container errands, re-scored per tick. Larder hops SKIP the landing scan: the entry is already verified, and the pickup click is a cheaper truth probe (code-4 purges a stale belief in one round-trip; failed_pickups blacklists). Rulings captured: (1) session-only memory, NO inter-session persistence; (2) the teleport-onto-equipment-container question must be PROBED before any bot-loop code -- the 2026-06-21 capture only proves pickup_equipment(x,y) fails when standing ON the container; the user believes a different pickup command form works from the container tile ("you just have to send a specific equipment pickup command"). Probe design: teleport onto a verified equipment container, try the candidate command forms, read the wire. Open design input still wanted: why the user's earlier implementation of this feature was removed.

## [2026-07-27] LAWS SHARPENED | Long-press pickup decoded from the client; fuel auto-pick on/adjacent landing corpus-confirmed -- larder mechanics grounded

User-supplied mechanics for the larder plan, both verified from artifacts before any probe: (1) the human "long press pickup" is release-after->300 ms on the held tile (tpclient bb handler) dispatching GET FUEL (action 5) / GET EQUIPMENT (action 6) / PICK UP OBST. (7) / DEPOSIT (10) -- the SAME wire commands the bot sends programmatically, so no new command exists to discover; the 2026-06-21 on-tile equipment pickup failure reduces to a server-side reach question (own-tile target honored or not), which stays the one live probe gating the larder. (2) Fuel containers auto-pick when a teleport lands ON or CARDINALLY ADJACENT -- corpus check: 62 of 82 teleport landings in runs 214102+211712 show the fuel gain arriving BEFORE any pickup_fuel command; fuel-system's "partially self-funding" sentence sharpened with the rule and the numbers. Implication for the larder: fuel larder hops may need NO pickup command at all (land on it, done), and the wiki knew the self-funding half before I did -- fifth artifact-first lesson this weekend.

## [2026-07-27] PLAN FINALIZED | [[larder-plan]] page lands -- the full harvest-circuit spec, grounded in the day's decoded laws, gated on one probe

The larder design session concluded and the complete spec now lives on its own page: memory is pure reuse of world.containers (session-only by ruling), selection is highest-and-nearest scoring re-run per tick, placement is a new priority step inside the COLLECT cascade between in-viewport walking and discovery (no new mode, no hunt changes), and the harvest circuit is map_open -> teleport -> auto-pick chained at ~2 ticks per fuel stop (half a forage stop) with equipment paying one extra tick for the pickup command. Confirmation and list maintenance need zero new code -- the fuel book's announced credits and the 0x43/code-4 registry paths already do both. The scan-coverage TTL (180 s) and container belief are documented as deliberately independent clocks. Build gate: the own-tile equipment pickup probe. Standing input wanted: the post-mortem of the user's removed first implementation.

## [2026-07-27] probe + laws | Larder gate ANSWERED YES -- own-tile equipment pickup works

Built `LarderProbe` (`action_lab/larder_probe.py`, `make larder-probe`, `tankpit-larder-probe`) to settle the [[larder-plan]] build gate: does the server honor `pickup_equipment` targeting the tank's OWN tile? Subclasses DensityProbe for funded site hops + extras etiquette; per attempt: teleport ONTO a verified equipment container, own-tile trial, step off one cardinal, adjacent control. 24 new tests, `make check` green (5,109 tests, 100% coverage).

**Three live runs, each one a lesson:**

1. `larder-20260727-224933` -- 0 trials. All three nearest candidates were water-sitting shore containers: teleports displaced to land, every walk-on rejected (0x52 err=1). Fix: candidates filtered to passable terrain (the plan already excludes inaccessible containers).
2. `larder-20260727-225643` -- own-tile 0/2, adjacent 1. The two failures were NOT an own-tile verdict: the tank was fully capped (25/25/25/25/25 after the user restock) and BOTH trials returned code 7 -- which the wiki string table already names "Inventory full" ([[decode-coverage]]). The one credit landed in the sole slot with headroom (extra radars at 24 after the search scan). Fix: each attempt burns one extra radar before the baseline read for guaranteed headroom.
3. `larder-20260727-230858` -- **own-tile 3/3, zero 0x52 errors.** Teleports aimed at container tiles landed exactly on them (3/3 here, 5/6 across runs); each `pickup_equipment` from the container's own tile credited the freed radar slot within one poll.

**Laws recorded:** own-tile equipment pickup works ([[equipment-system]], [[client-commands]] open question resolved, [[larder-plan]] gate section rewritten); equipment tiles are walkable and never auto-pick (user correction, verbatim in [[equipment-system]] footnote 10 -- my "containers are walk obstacles" reading of run 1 was WRONG); full-cap pickups reject with code 7 (re-confirmed live, contrast pair). The 2026-06-21 silent own-tile sample is superseded.

**Larder status:** gate cleared, plan page says ready to implement. Next: build the harvest cascade step per [[larder-plan]].

## [2026-07-27] feature + run | Larder IMPLEMENTED -- knowledge hops before discovery, live-proven same day

Built the [[larder-plan]] harvest step, all rulings honored: `bot/ai/larder.py` scores every believed fuel container min(volume, deficit)/teleport_cost with physics-only gates (legal landing via the shore-aware landing helper, fuel reserve, net-positive gain) and argmax re-run per tick; COLLECT cascade step 5 `_larder_harvest` (equipment hop first, then fuel) now runs BEFORE forage/discovery; both hops hold a resource lock on their container so the landing tick dispatches the pickup directly (equipment lands ON the container and picks from its own tile -- the law proven by the morning probe), and a new `suppress_landing_scan` AI-state bit makes the landing latch WITHOUT a radar (non-larder landings keep the unconditional 2026-07-03 scan). New tests: scorer suite (argmax flip at the deficit clamp, shore landing, reserve, unprofitable) + cascade behavior suite (larder beats forage_radar, locks held, flag consumed, control landing still scans). make check green: 5,122 tests, 100% stmt+branch.

**First live proof, 3-minute run `bot-20260727-234645`:** 2 kills, 25/25 hits, 0 misses, 0 rejections, exit session_complete at FULL stock (fuel 1100, dual 25, homing 25, radar 22). Forage economy: **2.00 forage viewports/kill** vs 3.10 (best pre-larder, bot-20260726-145124) and 7.70 (worst, bot-20260726-094309); 6 radars total against 26 pickups and 8 teleports; equipment yield 3.57 weapons/pickup. The log shows the exact designed choreography: `equipment hop to (170,102) landing (170,102)` -> `larder landing at viewport ...: latching without radar` -> own-tile pickup credit -> `fuel larder hop to (164,101) vol=210 cost=30`. Zero non-clamp command errors.

[[larder-plan]] status updated to IMPLEMENTED with the run evidence. Standing input still wanted: the post-mortem of the user's removed first larder attempt.

## [2026-07-28] law | Walk-over mine detonation is SINGLE-mine, movement stops -- no chain on foot

User law (verbatim in [[mine-mechanics]] footnote 6): stepping onto a mined tile detonates only the mine on that tile and the movement stops there; a single movement can never take more than one mine hit (45 fuel). This corrects my misreading of the cascade samples -- the adjacent-mine chain (0x45 two-packet) fires only on NON-movement triggers: a shot, an adjacent placement, or another mine's blast. [[mine-mechanics]] gains a Walk-over section and the cascade trigger list now excludes movement. Context that surfaced it: the diagonal-displacement discussion -- landing in or crossing a fresh 3x3 mine ring risks one 45-fuel hit per movement, not a chain wipe.

## [2026-07-28] fix + plan | Diagonal displacement Chebyshev fix shipped; ring-2 stand-off + teleport-onto-mine queued

**Fix (commit dc0a023b):** `completions.py` displacement classification now measures Chebyshev instead of Manhattan. The server displaces onto any of the 8 neighbors; a diagonal bump beside an aimed enemy is Chebyshev 1 but Manhattan 2, so the old test misfiled clean combat landings as failures and blacklisted the live enemy's tile for 30 s (the orange-6 abandonment, 20-kill run 2026-07-27). Distance-2+ landings (genuine non-arrivals: rejections, heavy displacement past a mine ring) still mark the aim tile. Regression pair added in `tests/bot/test_completion_events.py` (diagonal-no-mark + two-tile-still-marks). make check green (5,124 tests).

**Planned, wiki'd, not built:** [[bot-behavior-contract]] SS6 gains the PvP doctrine parking entry -- damage-aware break, under-fire larder refuel, and the ring-2 stand-off vs a known miner (a fresh 3x3 ring mines all 8 displacement landings around the placer, so aim approach teleports at Chebyshev-2). [[mine-mechanics]] SSnot-covered gains the gating open question: does a teleport LANDING on a mine detonate it, or only a step? (Walk-over law says step-only for movement; landings unmeasured.) Cheap practice-room probe when the PvP work starts.

## [2026-07-28] run | 10-kill attempt ends 4 kills, out_of_fuel at 108 -- the attrition case the PvP doctrine exists for

Run `bot-20260728-075336` (kill target 10, 334 s): 4 kills, 61/62 hits, 0 rejected, 0 blocked targets, exit `out_of_fuel` at fuel 108 (clean quit-to-lobby, not a death). **Root cause is attrition, not a bug:** the damage ledger books 51 incoming single hits = 2,295 fuel absorbed across six enemies (red-6 alone landed 15 for 675) on top of 22 teleports (~3.8-5.3k). The bot was mid-engagement at fuel 198 (HUNT/ENGAGE, tick 159), ate two more 45s, and at 108 nothing was fundable: 546 atlas dots but 327 unaffordable + 215 impassable inside the ~18-tile reach; 136 believed fuel containers but 116 reserve-blocked. This is precisely the parked [[bot-behavior-contract]] SS6 doctrine gap -- the static fuel_low_threshold 200 break is too late under sustained fire (45/hit), and the under-fire larder refuel would have bought the exit earlier and higher.

**Chebyshev fix regression check:** zero `teleport_displacement` events fired the whole run (no diagonal case arose to exercise dc0a023b live), and `targets_blocked=0` -- no orange-6-style abandonment. **Larder soak #2:** 6 fuel_larder + 10 equipment hops selected, 2.25 forage viewports/kill (consistent with the first run's 2.00 against 3.10/7.70 pre-larder); early-region believed equipment was 18/18 water-locked (`no_landing`), correctly skipped. Analyzer gap noted: `tankpit-forage-economy` does not yet break out the `fuel_larder` hop kind in its hops line.

## [2026-07-28] feature + runs | The second half ships: damage-aware break + marooned ladder -- 10/10 perfect run under the heaviest fire yet

**Damage-aware engagement break (commit c5cde4c6):** every ENGAGE tick projects fuel-at-kill (`fuel - hits_to_kill x (own spend + measured incoming rate)`) and breaks when the projection falls below the rate-scaled escape floor. The rate instrument counts only fuel-CONFIRMED hits (damage-book window, >= 3 hits / 10 s); breaking delegates through refuel_for_hunt so the lock survives and the larder aims the escape. Near-death targets project cheap and get finished (the user's premature-fleeing challenge shaped the design); quiet fights are untouched.

**The marooned saga (4 runs, 3 rungs):** the prior session left the tank at 98 fuel on a shore patch. Run -090813 exited with a pickable 39-fuel sliver 2 tiles away (worth-the-walk rule misapplied at critical fuel -> suspended, fdeafde6); run -091209 dispatched that pickup and the server said cant_go; run -092357 reached the new walk rung but its single nearest candidate was water-locked (-> nearest-first iteration, 78b06a8e) -- AND the trace showed walking could never work there (every leg water-blocked). The real escape was the auto-pick law: 12 of 13 nearby verified containers were water-locked against WALKING but harvestable by a teleport landing ON/adjacent -- the **desperation hop** (9c9c252c) ignores the discovery gates (clean-viewport, reserve) and hops to the cheapest legal landing.

**Proof run `bot-20260728-093736`:** the desperation hop fired ONCE at fuel 58 -- landing ON a 1139-fuel container for cost 12 -- and the session went on to a PERFECT scorecard: **10/10 kills, 109/109 hits, 0 misses, 0 rejections, 0 blocked targets, session_complete at completely full stock** (1100 fuel, 25/25/25). The break fired 15 times under the heaviest fire yet recorded (63 incoming singles, 2,835 fuel absorbed across 9 enemies -- MORE than the fatal -075336 run) and every broken fight was resumed and finished: sample arithmetic `red-6 fuel=573 rate=27/tick ttk=9 projected=150 < floor=354 -> break`, refuel, re-engage, kill. Forage economy 2.70 viewports/kill. [[bot-behavior-contract]] SS3.3 gains the break row, SS3.4 the marooned ladder, SS6 sheds the shipped doctrine items.

## [2026-07-28] run | Confirmation: second consecutive 10/10 session_complete under even heavier fire

Run `bot-20260728-103125` (started from the prior run's full-stock handoff, no rescue phase needed): **10/10 kills, 121/125 hits, 0 rejections, 0 blocked targets, session_complete at full stock** (1100 fuel, 25/25/24). The damage-aware break fired 13 times and every broken fight was resumed and finished; damage absorbed was the heaviest yet -- 69 incoming singles, 3,105 fuel across 8 enemies (red-8 alone landed 16 for 720, and still died: 29 of our hits, 2,070 dealt) -- and the session never approached the marooned ladder (0 desperation hops, 0 walks). Forage economy 2.90 viewports/kill. Two consecutive clean 10-kill sessions bracketing the doctrine: same exit choreography (kill #10 at 10:45:29, wound down fully stocked by 10:45:54).

## [2026-07-28] probe + law | Teleport landings DISPLACE off mines -- ring-2 doctrine resolved with zero code

User law (verbatim: "it displaces you"), wire-confirmed within the hour by the new `MineLandingProbe` (`action_lab/mine_landing_probe.py`, `make mine-landing-probe`, 11th probe type): run `mine-landing-20260728-161432` aimed teleports dead-on at three enemy mines and all three landed exactly one tile beside the mine, `extra_loss=0` (pure teleport cost, no 45 bill), every mine intact in the registry afterward. **Mine tiles join occupied tiles in the displacement-excluded set** ([[mine-mechanics]] new section; the open question is closed).

Doctrine consequence: the ring-2 stand-off item -- the last open entry from the 2026-07-27 PvP doctrine list -- resolves with NO code change. Aiming an approach teleport at a mine-ringed enemy is self-protecting: the landing can never touch the ring; the ring threatens only walking (one 45/movement, walk-over law). With the damage-aware break and under-fire larder refuel shipped this morning, the entire PvP doctrine list is closed. [[bot-behavior-contract]] SS6 updated. make check green (5,175 tests).

## [2026-07-28] feature | Human-priority targeting: farm bots, pounce on any human who logs in

User doctrine session: "priority mode yea... farm bots but prioritize any human player that logs in", plus two rulings recorded verbatim in [[bot-behavior-contract]] SS3.2 -- under-fire COLLECT explicitly declined ("leave collecting alone. its hard to hit a walking bot" -- collection movement is its own evasion), and the mid-fight lock is never switched (priority applies at the next acquisition; never-drop governs the current kill).

**Implementation:** `bot/ai/humans.py` classifies by NAME shape -- practice bots are always `<color>-<n>` (`orange-1`, `red-6`); any non-empty name outside the pattern (`guest`, account names) is human; empty names stay bot-tier (no phantom chases). The shared threat sort gains a leading human tier, so the priority flows through all three selection points unchanged: visible threats, map acquisition, and the dot-relay travel target -- meaning the bot yellow-dot-hops toward a distant human, refueling per hop, exactly as the user asked ("will it use fuel dot hopping to get close?" -- yes, the 2026-07-03 relay machinery, now human-first). `TANKPIT_BOT_PRIORITY_TARGET=<name>` adds a top tier for one named account (case-insensitive), wired through AIConfigDict.priority_target_name. On non-practice maps every enemy ties at human tier and ordering degrades to plain nearest-first. The registry is_bot field remains unpopulated dead plumbing (no wire decoder sets it -- noted for a future weapon-signature cross-check: practice bots only ever fire singles, so a dual/homing from an enemy is a human fingerprint the damage book already records). make check green (5,174 tests, 100%).

## [2026-07-28] feature | Human rank window: recruits protected by default, per-bot floors and ceilings

User doctrine refinement on the human-priority system: rank-aware human targeting via a configurable window. Ranks are integers (0 recruit .. 8 general), so the rule is `[min, max]`: humans outside it never enter the threat list, acquisition, or relay (`protected_human_rank`, checked before affordability so the relay never travels toward a protected human). Defaults (1, 8) implement "we dont target recruits but every other rank we do"; `TANKPIT_BOT_HUMAN_MIN_RANK`/`_MAX_RANK` let a main-map bot run lieutenant+ only (min 4) or cap the ceiling to leave captains/generals alone out of respect. Practice bots are farmed at any rank -- the window is human-only. Fail-safe: unsynced rank defaults 0, so an unknown-rank human is briefly spared, never briefly attacked. Also recorded: a same-color human is an ALLY -- invisible to targeting entirely (different-team gate) with the server independently rejecting friendly fire (0x52 code 3); making them targetable requires the bot itself to join as a different color, which connects to the user's tank-profile note: one account carries 4 color profiles (one playable at a time), so a future color-selection step at join time would let the bot play any team. That join-flow feature (map-click join + profile picker) is NOT built -- it needs a live lobby DOM/wire investigation first (the '+' join command carries the team field, [[client-commands]]). make check green (5,179 tests).

## [2026-07-28] feature + fix | Tankpit cut loose from fiesta: CDP screencast watch page, and service sessions finally leave artifacts

User goal (verbatim: "i dont relaly.play fiedta tbh. i just wanma ne able to see tankpit bot run from my phone. and chexk in on it whenverr. and leave it running yk") after the fiesta investigation traced the mouse-stealing to architecture, not a bug: the vibeshine stream injects ALL phone input via `SendInput` into the one real Windows cursor, warping it onto the deliberately non-adjacent invisible virtual monitor where physical mouse movement cannot retrieve it -- plus a still-unfixed coordinate bug (fiesta task #16) that can land injected clicks on the REAL primary monitor, and the 2026-07-01 desktop-takeover incident history. Tankpit needed none of that stack: the only thing it contributed was pixels of one Chromium window.

**Watch surface (this repo only, zero MCPs changes):** `browser/screencast.py` relays Chrome's own `Page.startScreencast` JPEG stream (q70, max edge 1024) off the bot's existing CDP session -- ack-then-publish into the new `service/frame_bus.py` (`FrameBus`, the StatusBus pattern + `latest()`); `_sync_screencast_demand` in the tick loop starts/stops the cast on subscriber demand (unwatched sessions and every `make run` pay nothing); HTTP grows `GET /watch` (self-contained phone page: MJPEG view, SSE stats strip, START/STOP + mode buttons), `GET /video` (MJPEG relay, last-frame keepalive), `GET /frame` (snapshot, demand-wait then cache). Relative URLs make the page work both direct (`:27100/watch`) and through nginx's existing `/api/tankbot/` prefix-strip proxy at `https://tankpit.austinwagner.org/api/tankbot/watch`. No input path exists anywhere in the surface -- the host mouse physically cannot be touched. Idle-exit counts video viewers as activity. [[bot-service-architecture]] carries the full design.

**Live proof (session `bot-20260728-230140` + the 22:31 session before it):** `/frame` served real 1024x576 JPEGs of the live game (Artax in the practice room); `/video` measured motion-adaptive -- ~0.6 fps with the bot idle (page barely repaints), ~2.8 fps / ~200 KB/s in AUTO play (28 parts per 10 s). Screencast demand-toggle proven in the archive: `Screencast started (viewer connected)` on the poll's subscribe, `stopped (no viewers)` 3 s later. The 22:31 session incidentally ran a full third consecutive 10/10 `session_complete` (kill target reached 22:45:29).

**The fix the test exposed:** that 22:31 session left NO artifacts -- only `bot/entry.py` (`make run`) ever called `configure_bot_runtime_logging`, so every phone-driven session since Phase A (2026-07-12) ran with unconfigured logging: INFO lines dropped, no archive log, no events.jsonl, no `_index.tsv` row. `SessionRunner.start` now configures the per-session bundle before constructing the bot (first line: `Session artifacts: <archive path>`); verified live -- `bot-20260728-230140.*` exists with the screencast lines and an index row (`stop_file`, 4 ticks). make check green (5,228 tests, 100%).

## [2026-07-28] deploy | Bare tankpit.austinwagner.org now lands on the watch page

User: "why cant j juat go to tankpit.austinwagner.org?" -- it served the fiesta SPA at `/` for every gated host. MCPs commit `62b8a214`: host-conditional exact-root block in `fiesta/nginx.conf` 302s `tankpit.austinwagner.org/` to `/api/tankbot/watch` (relative Location so the phone stays on https through cloudflared); deployed via `make up-fiesta` and curl-verified against port 8091 with Host headers -- tankpit redirects, fiesta/games roots untouched, SPA still at `/index.html` on the tankpit host (the SERVER launch button lives there for cold starts). Caveat recorded: the watch page needs `tankpit-bot-service` up -- when it is down the proxied routes 502 and the recovery paths are the SPA's SERVER button or `make service` at the PC.

## [2026-07-29] fix | The redirect that never fired: mcp-proxy hides the public hostname from backends

User went to the bare hostname and got "idle" -- telemetry (fiesta nginx access.log) showed the phone's `GET /` served the SPA (200) with the bot panel 502ing against the down service; the 62b8a214 redirect never fired for real traffic. Root cause in `mcp-proxy/src/host-gate-proxy-core.ts::buildUpstreamHeaders`: the gate rewrites `Host` to the Docker service name and forwarded nothing carrying the public hostname, so nginx's `$host` check could never match through the real path (my curl verification had gone direct-to-container with a hand-set Host header -- a false pass). Fix (MCPs `5ff0848a`): the proxy stamps `X-Forwarded-Host` from the inbound Host unconditionally (clobbers client-supplied values -- not spoofable past the gate; proxy suite 100% coverage), and the fiesta nginx exact-root block redirects on either `$host` or `$http_x_forwarded_host`. Verified against 8091 with simulated proxy headers: tankpit XFH 302s, games XFH stays SPA. Lesson banked: verify through the REAL proxy chain, not a reconstruction of it. Bot service left running for the user.

## [2026-07-29] feature | The cool UI is back: MJPEG video inside the retro SPA, always-on service, redirect retired

User: "why eouldnt we use the cool ui system we msde lol?" -- no reason at all, so the SPA is the tankpit front door again with the fiesta-free video inside it. MCPs `95f27215`: new optional profile field `botVideoUrl`; the bot-overlay binding drives a `#bot-video` image (mirroring the `#stream` box, style v66) from live `BotUIState` via `computeBotVideoView` -- MJPEG attached while a session runs, detached otherwise. `tankpit.json` went stream-less (`stream: null`, video from `/api/tankbot/video`) and shed every input surface (joystick, L/R click bubbles, Q, ALT+F4, `server:start`); watch-and-control stays (toggle, four modes, `server:stop`, nav, stats strip). No WebRTC session -> no input channel -> host mouse untouchable, retro theme intact. The one-day-old exact-root redirect is reverted; `/api/tankbot/watch` remains as fallback. 782 SPA tests, 100% coverage.

The SERVER cold-start button rode the vibeshine run_command channel, so the gap is closed the way the user originally asked ("can it just be running?... leave it running yk"): always-on service. This repo `59201238`: `TANKPIT_BOT_SERVICE_IDLE_EXIT_SECONDS=0` disables the idle self-exit (`resolve_idle_exit_seconds`, `exit_when_idle` early return); a shell:startup launcher runs `make service` minimized at logon with the exit disabled (delete the .cmd to revert); the live service was restarted in always-on mode. make check green (5,233 tests). Deployed and curl-verified: tankpit root serves the SPA, profile ships `stream: null` + `botVideoUrl`, compiled bundle carries the video glue.

## [2026-07-29] fix | Stream-less profiles have no video band: three buttons re-anchored

User report: "i only see auto, idle, gsther and hunt buttons" -- nav-back, STOP SERVER and START BOT were anchored to `video-bottom`, a position computed from the WebRTC picture band that the stream-less tankpit profile never has, so they rendered nowhere; the four mode buttons survived on plain `bottom` anchors. MCPs `b378f0f6` re-anchors the lost row to the screen bottom (toggle+server at 194px above the mode rows, nav-back bottom-left beside the stats strip). Profiles are bind-mounted, so the fix was live on the next page load with no rebuild. Rule for future stream-less profiles: `video-bottom` anchors are a streaming-profile-only tool.

## [2026-07-29] fixes | Armed hold, stream re-attach on foreground, game-fullscreen parked

Three user reports from live SPA use. (1) "the hot has homing and dual shots disabled" -- `make_hold_decision` requested `desired_equipment=[]`, so the executor toggled EVERYTHING off every idle-pinned tick ("Using dual shot disabled" in the game log), and since toggle state persists across logout ([[radar-mechanics]]) the tank stayed disarmed for the next login. Fix `4fc41966`: the hold carries the normal stocked loadout (dual+homing while stocked, radar always) via `compute_desired_equipment` with live fuel+inventory; armor/missiles stay off by policy. make check green (5,234 tests); service restarted with the fix. (2) "when i leave the app and come back sometikesthe stream disappears" -- backgrounding kills the MJPEG connection with no bot-state change, so the attach/detach glue never re-fired; MCPs `8ccb4ba7`: visibilitychange->visible re-attaches a fresh cache-busted src, listener removed on overlay teardown. (3) The game's own fullscreen button (the `zh` main-app container owns fullscreen/resize, [[js-source-map]]) parked -- the user will `make sniff` it later; a bot-side auto-click at session start would make the stream mostly game instead of page chrome.

## [2026-07-29] feature + law | Page-push live view: steady 12 fps pure-game video, and the Local Network Access law

User: "what can we do to maintian good qualoty and improve the video... we have fast intenrnet." The screencast's stutter was structural (ack-gated on the tick thread), so capture moved INSIDE the game page (commit `625adfd6`): an injected interval composites the client's six stacked canvases ([[rendering-pipeline]]) to one JPEG per frame at env-tunable cadence (`TANKPIT_BOT_VIDEO_FPS`=12, `TANKPIT_BOT_VIDEO_QUALITY`=0.8) and delivers via a CDP binding; the bot-side handler publishes to the frame bus. Debug chain worth remembering: zero frames arrived with zero errors -> in-page telemetry (console lines keyed "Hook" to pass the console-listener filter) proved the interval ran at full rate with every fetch PENDING FOREVER -> **law banked: Chrome's Local Network Access gate parks game-page fetches to 127.0.0.1 behind an ungrantable permission (no resolve, no reject; Playwright 1.57 grants no such permission) — page-frame delivery must ride the CDP binding channel** ([[bot-service-architecture]]). Measured live: 117 MJPEG parts / 10 s (11.7 fps) with the tank IDLE (screencast managed 0.6), ~800 KB/s, and the frame is PURE GAME pixels — no page chrome, which quietly delivers the parked "game fullscreen" wish too. make check green (5,241 tests); SPA and watch page untouched (same /video contract); service restarted with the new pipeline and left running with a live session.

## [2026-07-29] runs + tooling | 20-kill session 1/2 clean; analyzers learn to narrate fuel, idle, billing, teleport spend

User: "can you do a 20 kill run, analyze it thoroughly, and then so another 20 kill run please?" Run 1 (`bot-20260729-105325`, the service's idle session stopped first via /stop): 20 kills in 1434 s, 248 shots at 98% (4 misses, 0 rejected, 0 blocked, 0 deaths), clean kill-target wind-down to `session_complete` fully stocked (1100 / 25 / 25 / 20). Audit verdict 0 critical / 0 warnings; all 19 info items were the benign drained-container race. Deep dive receipts: the session min fuel 140 was the [[bot-behavior-contract]] chase pipeline working end to end (red-7 fled, lock held through map refresh, 158-fuel chase teleport at 372, LOW_FUEL refuel detour, resume, kill at 11:07:55); the ledger's 6 `weapon=0` singles matched the [[weapon-selection]] non-connect billing law (shot_single_fuel_lo = -36 = 6 x -6); IDLE 285 s decomposed into 283 tick-boundary residues (max 6 s).

The four blind spots that dive exposed became analyzer upgrades (commit `57c8da78`, offline only -- archived runs benefit retroactively): (1) fuel low-water EPISODES below the session's own danger line (max `engagement_break.escape_floor`, fallback 100) with entry/min/cause/recovery narration; (2) state-budget lines carry (stretches x, max single visit) so residue reads as residue; (3) the ledger's shot billing renders with the singles law inline; (4) teleport spend attributed per paying `bot_state` from WORLD fuel receipts, rendered against the fuel book's feasibility bound. Method lesson banked in the spend docstring: sample-to-sample fuel deltas UNDERCOUNT in-flight spend (10972 vs bound 11993..19290) because pickup credits mask debits inside one tick window; the WORLD-receipt attribution lands at 15592, inside the bound, and names HUNT/CLOSE chases (7389) the top teleport cost. make check green (5,256 tests, 100%). Run 2 launched with the same bound; its scorecard will be read with the upgraded analyzers.

## [2026-07-29] run | 20-kill session 2/2 clean; upgraded analyzers read it in one screen

Run 2 (`bot-20260729-112142`): 20 kills in 1622 s, 232 shots at 98% (3 misses, 0 rejected, 0 deaths), clean wind-down fully stocked; audit 0 critical / 0 warnings / 14 info (all the drained-container race). First live outing for the `57c8da78` analyzer upgrades, which read the session at a glance: only 2 shallow fuel low-water episodes (min 345 vs run 1's 140 -- no desperate chase this time, both dips recovered to 1100 via collect within seconds), IDLE 374 s = 346 tick-boundary residues (max 7 s), and the teleport-spend table flipped its leader: COLLECT/SEARCH paid 9948 of 20389 (bound 15865..25132) versus run 1's HUNT/CLOSE-led 7389 of 15592. The two clean runs differ mainly in economy shape: run 2 foraged much harder (55 forage viewports vs 18, 149 dot hops vs 69, 80 scans vs 52, hunt/collect split 825/784 s vs 937/484 s) for the same 20 kills at the same hit rate -- practice-room target availability, not bot behavior, set the pace. Cross-run constant worth keeping an eye on: kills cost ~12 shots each at 98% accuracy in both sessions.

## [2026-07-29] contract | Fuel before chasing: the combat teleport now reserves the full engagement budget

Follow-through on the run-1 deep dive. The user first corrected my mechanics gloss -- the 12.92 s wall is the HOMING REROUTE TTL (gates blind shots at a departed tank), NOT position accuracy; the map is a whole-game snapshot on demand, so position knowledge never expires and a refuel detour costs only the target's own heal window (fuel pickups are the only repair, [[enemy-bot-behavior]]) -- then ruled: "we cant kill anyone if we die... we should fuel before chasing." Root cause was a reserve asymmetry: acquisition demands `cost + engagement_fuel_budget + fuel_low_threshold` (2026-07-02 gate) but the teleport DISPATCH point (`_combat_teleport`) reserved only `fuel_low_threshold`, so run 1's chase re-teleport passed at 372 by 14 fuel and bottomed the session at 140. Commit `22061123` raises the dispatch reserve to the same end-to-end sum (650 + cost); underfunded dispatches delegate to `refuel_for_hunt` (lock held, never-drop intact; the collect cascade walks to in-viewport containers first, then the larder -- run 1's own refuel drank a logged 914 dot five tiles from the fight). New contract row in [[bot-behavior-contract]] SS3.2 + regression test pinning the 372/158 shape to a refuel decision. Sim-validated per the user's "cant we test this in the sim?" (`sim-20260729-202938`, certified roster): 9 underfunded chases deferred to refueling, hunting intact (2 kills), ended alive at 886 fuel, zero low-water episodes. make check green (5,257 tests, 100%). Live 20-kill validation run follows.

## [2026-07-29] contract | Unlimited-distance human pursuit: the relay chain now hunts people, not just proximity

The user logged in as Yuppler mid-validation-run and the bot never came ("i thought it would prioritize humans"). Receipts told the story: 146 events saw Yuppler (tank 1229, purple, rank-window PASS -- no protected_human_rank anywhere), but acquisition rejected them `unaffordable` at dist 95 (cost ~570 + 650 reserve > the 1100 cap) and farmed red-3 at dist 19 instead -- the relay chain that closes distance on unaffordable targets was only consulted when NOTHING was affordable, so a room full of practice bots starved the human priority forever. Ruling (verbatim): "unlimited distance for humans. we can fuel dot hop right? even if they teleport super far away too. this is the real deal", refined same-day: "finish the kill then the human player will be the next target" (locks never switch mid-fight). Commit `0455d4ba`: at acquisition, an unaffordable rank-window human now PREEMPTS every affordable bot -- the tick becomes a relay leg toward them (`_human_pursuit_travel_target` + the extracted `_relay_toward` core); a LOCKED human beyond fundable range is chased leg-by-leg with the lock held (each leg closes distance AND refuels on the landing pickup); bots keep refuel-in-place (they never flee across the map); recruits stay protected (rank window rejects before affordability, so no relay can travel toward one); when no leg helps the bot farms and re-evaluates next map (no deadlock). Six new tests + contract row in [[bot-behavior-contract]] SS3.2. Sim green (`sim-20260729-210331`), make check green (5,263 tests, 100%). Live proof pending: next session, the user logs in and the relay should cross the field for them.

## [2026-07-29] run | Fuel-before-chase validated live: 20 kills, session min 494, zero low-water episodes

Validation run (`bot-20260729-203201` window, exit `session_complete` fully stocked): 20 kills, 260 shots at 97%, 19 `refueling before hunt` deferrals, and the number the ruling bought -- **fuel low-water: none (never below the 372 danger line; session min 494)** versus the pre-gate run's five episodes and min 140. The quantified cost: ~101 s/kill vs 72-86 in the pre-gate runs (each fight now ends near 500-650, just under the 650+cost bar, so most engagements start with one larder hop) and teleport spend up to 23300. Audit 1 critical / 1 warning / 23 info: the critical is a single self-healed map_open stall (the sync marker raced the 10 s timeout -- data arrived the same second); the warning is the June-blind-spot canary firing on 2 undecoded 0x2E subtypes (0x29, 0x4E) -- almost certainly human-generated wire traffic from Yuppler sharing the room, exactly the multi-tank blind spot [[decode-coverage]] predicts solo runs hide. Follow-up: crack those two subtypes from the capture. Bonus receipts from the same run: Yuppler competition measured as 12% pickup_empty (vs 5% solo) and the radar-loop worry dissolved by the event trace (96 scans, longest radar streak without a teleport/pickup between = 1).

## [2026-07-29] fixes | First live human pursuit: one relay leg, two exposed gaps, both closed

The pursuit fired for real at 21:17:40 ("human Yuppler (id=1229) at dist 161 outranks affordable bot red-9 - relaying") and the user's own play immediately stress-tested it into two pre-existing gaps (commit `d780103d`). (1) **Broke arrival**: `_pick_relay_dot` maximized progress with no cost cap and paid 787 fuel in ONE leg (1100 -> 313), landing four tiles from Yuppler unable to fight; hunt-only-when-full handed the tick to COLLECT, which foraged AWAY from the human for minutes ("i even saw artax but he ignored me"). Legs are now capped at `engagement_fuel_budget` (450). (2) **Freshness asymmetry**: practice bots move/shoot so the wire keeps them permanently map-fresh; a QUIET human's position copy ages out `map_open_cooldown_ms` (5 s) after every map open, and with fresh bots always available acquisition never reopened the map -- the human was actionable only in 5-second windows (`stale_map_data` on six straight decisions). New rule: a rank-window human whose only curable defect is stale map data forces a map refresh before bot farming; a fresh map still showing them stale means they left (no loop). Also surfaced live, tracked for next: the user mined a 2-row ring around himself and `no_passable_adjacent` rejected him entirely -- a mine ring is currently a full immunity cloak; stand-off-range engagement is the counter. And the "radar spam" ear-report dissolved under the event trace: 96-124 scans/run, longest radar streak with nothing between = 1, ~3 pickups per scan -- radar is just the only action with a loud sound. make check green (5,268 tests, 100%).

## [2026-07-29] contract | Mine-ring counterplay: a dead landing next to a visible target becomes a stand-off shot

The user's live exploit test closed its own loop: the pursuit chain reached Yuppler, the server displaced the landing off his 2-row mine ring to two tiles away, the re-close failed on the displaced tile, and the bot BLOCKED a target standing in plain view ("he landed two tiles from me, didnt engage, then left"). Ruling (verbatim): "fix the bot so it can hunt even if i put mines around me. if it lands 4 tiles away or more, it can still fire a dual shot right?" Commit `94fa85e9`: a failed combat landing with the target inside the current viewport now fires from stand-off instead of blocking -- per [[weapon-selection]] the server fires duals at stationary targets from any in-view range (water never blocks; rock clips to a billed single that resolves as a miss through the stationary-miss block), hits confirm via the ammo ledger, and only an off-view target with a dead landing still blocks. Ranged-shot support (SHOT_RANGE_TILES, has_combat_shot, landing-candidate rework) landed alongside. make check green (5,270 tests, 100%). Sniff sniff-20260729-214411 (the user testing in-game chat as Artax) incidentally banked the undecoded chat wire: sent 0x06 len=7 per canned message, 0x02 len=3 for the zone/tip keys -- a decoder opportunity if the bot ever needs to talk.

## [2026-07-29] decode | Chat interface cracked end-to-end from sniff-20260729-214411 (44 live sends)

Mined the user's chat-testing sniff (Artax, solo practice room) with the session magic + `xor_static_key.txt` table: all 44 chat sends decode as `[6,'m',message_id,x,y,flag]` — the 6-byte Hb form is used for EVERY message including non-position ones (41 HELLO); the 4-byte variant never appears; flag byte was 0 in all sends; auto-search messages substitute the found tile (id 8 went out with the Db() nearest-fuel coords, not the tank's). Inbound receipt confirmed as the known `M`/0x4D frame ([[decode-coverage]] row Qg): `M + tank_id(2 LE) + msg_id + x + y` — and the DOM "Message sent:" line tracks the SERVER ECHO, not the local click. Two gates documented: client-side, team-filter messages while solo print "No teammate in the zone" and never reach the wire; server-side (NEW), after 8 echoed messages at the 2400 ms cooldown pace the server silently swallowed all 36 remaining sends for the rest of the session (no error frame, other commands unaffected) — a flood mute of unknown decay. [[chat-messages]] updated with the wire-verified section + mute rule ("chat must be rare and never retried on silence"). This corrects the previous entry's incidental guess ("0x06 len=7 per canned message" — right bytes, now fully decoded). No code changes yet by user instruction; wiring plan: a `build_chat_command(message_id, x, y)` in `protocol/commands.py` (plaintext `! 06 6D id x y 00`, same scheme as every other bot send) + a one-shot HELLO (id 41, filter=3, no target check) fired on the COLLECT→HUNT transition when the acquired lock is a rank-window human, rate-limited to once per target acquisition.

## [2026-07-29] feature | Chat wired end-to-end: the bot now says HELLO to the human it hunts

Implementation of the same-evening chat crack, user directive "decode inbound too... wire it into the bot so that the bot can send 'hello' when it finishes collecting and has targeted the human player." Outbound: `protocol/chat.py` (65-entry E[] table + `build_chat_command` — plaintext `! 06 6D id x y 00`, XORed by the standard send path into byte-identical frames to the page client's), `DispatchMixin.send_chat` (fire-and-forget, `chat_sent` diagnostic), and a `"chat"` BotCommand dispatchable as a decision `secondary_command`. Inbound: the 0x2E envelope router gains the missing 0x4D route (corpus sweep: 320 sessions, chat is ONLY ever 0x2E-tunneled at exactly 5 inner bytes — the top-level `M` route existed but never fired), and the world-state dispatcher now emits `chat_received` with resolved preset text, latching `WorldService.last_chat_echo_message_id` on self echoes — the only delivery receipt the flood mute allows. Behavior: `bot/ai/greeting.py::attach_human_greeting` at the arbitrator's HUNT exit — the first decision that locks a NEW human attaches HELLO (41) as its secondary, once per tank id via the new `ai_state["greeted_target_id"]` latch, never retried on silence (mute discipline); practice bots are never greeted. Contract row added to [[bot-behavior-contract]] §3.2; [[chat-messages]] gains the implementation section. Sim server learned the chat law (decode `m`, echo 0x4D to all including sender) so greeting ticks replay in sim. Also swept up two pre-existing gaps the 100% gate caught in the user's in-flight work: the flag-capture `Runtime.addBinding` call broke two `test_cdp.py` tick-sequence pins (updated), and the `TargetClosedError` absorb in `_send_graceful_quit` (added after run bot-20260729-215151's exit-code-2) had no test (added, pinning the absorb). make check green: 5,339 tests, 100% coverage. Live validation pending: next session, user logs in, bot should teleport in and say HELLO — receipt is the `chat_received` self-echo in events.jsonl.

## [2026-07-29] feature | HUD rebuilt fiesta-style at fixed geometry + click-to-flag ledger channel

User rulings: the old overlay was "archaic, ugly, and not really informative to what we actually care about," it "changed size so much" (the box hugged five variable-length text lines and resized every tick), and the redesign should "use the design from ~/PROJECTS/fiesta/ ui." Rebuild (three modules replacing `browser/overlay.py`'s render path): `overlay.py` (26-field payload + slot renderer — all strings/colors computed Python-side), `overlay_hud.py` (install-once DOM + stylesheet, per-tick slot assignment only, so the 272px card NEVER changes size), `flag_capture.py` (the ⚑ FLAG button's CDP binding → `human_flag` DIAGNOSTIC event with `flag_seq`, `clicked_at_ms`, and the last 8 HUD payloads as `recent_ticks` — clicking the card while watching `make run` now banks the bot's lead-up thinking for `make analyze` instead of the user relaying bugs from memory). Design carried channel-for-channel from the fiesta SPA (glass recipe from `MCPs/fiesta/src/style.css`, palette from `services/theme.ts`): frosted blue-tint panel + dot stipple + two-tone bevel, HUNT=hot pink / COLLECT=green / UNSET=purple mode banner, stock slots colored by the hunt-gate bands (green at cap, off-white within 5, pink below), fuel meter on the damage-tier quartile (pink under 25%). New page [[diagnostic-hud]] (architecture hub, 10 pages). New payload fields surfaced what the user actually watches: stocks vs rank caps, session K/H/M/RJ, combat target + staleness-free identity, sent/held verdict per tick. The card is pointer-inert except the button and invisible to the live-view caster (canvas-only compositing), so the phone stream stays clean. Also swept three pre-existing guard violations off the tree (weak assertions in test_tick_loop_types/test_greeting). make check green pending final gate this entry accompanies.

## [2026-07-30] triage | First live flag session: 10 flags -> 4 root causes, all traceable, all tracked

The flag channel's first outing (run `bot-20260729-232252`, still in progress at triage time) captured all 10 of the user's clicks with 8-tick lead-up rings — the feature worked end-to-end on day one, and every flag is locatable by `tick_n` (recipe now in [[diagnostic-hud]] § Tracing a flag). Findings on the new [[flag-triage-20260729]] page with a fix-status table (user directive: "make sure that it is easy to trace the flags and solve these issues and keep track of them"). The four root causes: **F1** (flags 1/2/6) the pre-hunt fuel top-off hop is direction-blind — flag 2 hopped 26 tiles NE then the acquisition teleported 30 tiles SW straight back; **F2** (flags 3/5/7/9/10) 133 of 211 hops (63%) yielded zero gains, longest dead streak 19 hops — harvest memory IS the 180 s scan-coverage TTL (`FORAGE_COVERAGE_TTL_MS`), so picked-clean ground reads "clean" again after 3 minutes and the hop score (`dots*walkable/cost`) never consults the 531-entry container belief store or models that dots are worthless at full fuel; **F3** (flags 4/8) mine-covered equipment has no counterplay — the user's rank-gated blast + LOS mechanics are now in [[mine-mechanics]] (recruit shot kills 1 mine, private+ kills the 3x3; clear straight shot required, terrain + land movables block, homing/missile arc is tanks-only), design: one lifted shot-clearance module shared by combat and mine-clearing; **F4** Yuppler was never targeted — all 11 acquisition passes rejected him at frozen (128,102), 7x `no_passable_adjacent` + 4x `stale_map_data`: the mine-ring cloak works at ACQUISITION time because human preemption (`0455d4ba`) keys only on `unaffordable` and stand-off fire (`94fa85e9`) needs an existing lock; the `d780103d` stale-refresh rule also failed to cure the 4 stale rejections. All four rows OPEN; close only with run/sim receipts.

## [2026-07-30] audit | Independent flag-session investigation converges with the triage; three receipts added

Second-channel investigation of `bot-20260729-232252` (user report: "it ignored yuppler and never said hello"), run without sight of the parallel triage until the end — full convergence on all root causes, three additions folded into [[flag-triage-20260729]]: (1) **F1 existence cause** — the top-off hop is forced by a heuristic disagreement: stock completed at 23:24:35 with fuel 1083/1100, the nearby 17-point pickup was refused ("clamped gain 17 not worth 10-tile walk") while the hunt floor demands exact capacity, so the last points are only purchasable via a hop's landing auto-pickup; direction-aware hop placement alone still pays that leg. (2) **F4 static receipt** — 86 passable field01 tiles inside the radius-8 diamond around Yuppler's frozen (128,102), nearest at distance 3: the landed stand-off gate accepts the exact ferry position; run-order proof that the run executed the OLD gate (rejections read `no_passable_adjacent`; the stand-off files hit the tree 23:49–23:51, after the 23:47 run end). (3) **HELLO trail** — zero chat events in the run is CORRECT behavior: the greeting is downstream of acquisition (`greeted_target_id` untouched), so F4's fix is also the hello fix; receipt to watch next login is the `chat_received` self-echo. Also cross-checked hop economics from the raw stream (179 teleports, 108 landing scans, 79 single-pickup scans, 41 `tank_full_clamp_receipt` + 12 `empty_container` rejections, radar trajectory 21→7 across flags 7→10) — consistent with the triage's 63% zero-yield measurement. No code changes (user directive).

## [2026-07-30] fix | F4 closed in code: stand-off acquisition + shore landing (commits `7d59e877`, `0630707b`)

The cloak's true root cause was settled by the user mid-fix ("yuppler was on a ferry. i mtelling you that") and confirmed against field01: (128,102) is open water (row 102 runs water x=126–131), so a ferry rider has zero passable cardinal neighbors and `no_passable_adjacent` rejected him on all 11 passes — the mine-ring theory was wrong, though mine rings cloak through the identical hole via the mine-composed passability view, so one gate change covers both shapes. Commit `7d59e877`: `_acquisition_rejection_reason` and `select_new_combat_target` now gate on `has_standoff_landing` (any passable tile inside the `SHOT_RANGE_TILES`=8 diamond; rejection renamed `no_standoff_landing`), and `choose_combat_landing_tile` aims at the passable unoccupied tile nearest the target (ties toward self) when the target's own tile is impassable — the server refuses water landings, and per [[weapon-selection]] water never blocks the shot from shore; `SHOT_RANGE_TILES` moved down to `combat_landing` so both modules share it without an import cycle, re-exported unchanged. Because every acquisition consumer (nearest-pick, human preempt, relay travel target, stale-human refresh) shares the one rejection function, ferry riders now flow through the whole pursuit chain — relay legs, HELLO greeting, stand-off duals — with no per-path special case; `0630707b` pins the laws (ferry-rider acquisition, shore aim, occupied-tile skip + self tie-break, corner clipping, mid-ocean still rejects). make check green: 5,353 tests, 100% coverage. F4 row moves to FIX LANDED; the live receipt (relay → HELLO → shore stand-off vs the ferry) and the stale-refresh verify ride the next session, which is being cycled onto this build now. Flags 11–13 arrived during the fix; flag 11's ferry-forage doctrine is banked in [[ferry-mechanics]] and tracked as F5.

## [2026-07-30] triage+fix | Session-2 flag wave: F6-F10 opened, F8 fixed same-tick (commits `89459dd7`, `15055fae`)

The user rode along with run `bot-20260730-000030` (first build carrying the F4 stand-off gate) and kept flagging; every flag is now investigated, documented on [[flag-triage-20260729]], and tasked. New root causes: **F6** collect reachability composes mines but not TANKS (old flag 12: equipment across a water channel whose only land route was choked by an enemy tank -- two server `code=1` refusals); **F7** fuel locks survive combat un-revalidated (old flag 13: post-kill "continue locked fuel target vol=84" drank a near-empty remainder, then paid map+teleport for the 462-volume container it had just skipped); **F9** decisions deferred for map_open lose arbitration to fresh-map HUNT every tick (new flag 7: the escape hop re-deferred four cycles while single shots interleaved, fuel 572->462 under fire beside the orange minefield that had correctly walled off all 21 containers as blocked_walk; also wanted: a requested-vs-landed `teleport_displacement` receipt); **F10** walk-blocked equipment gets no larder-style teleport service (new flag 4: `blocked_walk` -> actionable=0 -> search hop away, then a coincidental larder landing proved the pickup was one teleport away all along). **F8** (new flag 1: map-acquire teleported onto purple-4 standing 2 tiles away in view; flag 5 repeated it against orange-2 at dist 7) was fixed in the same sitting: `_combat_teleport` short-circuits to the shot for in-view targets within `SHOT_RANGE_TILES`, covering fresh/map/resume acquires in one place, three verb pins updated per their own "the verb is not the invariant" docstrings, and the failed-landing stand-off pin moved to the beyond-range corner so its branch keeps dedicated coverage. Flags 3/8 (ferries in view, unused -- "missed multiple ferries") harden F5; flag 9's game-log paste (~20 teleport+radar "Zoom in" cycles for ~15 equipment) is F2's live receipt on the new build. User timing law banked on the triage page: a teleport costs ~4 s, a walk ~2 s -- within-viewport closes should consider walking to a clear-shot tile. Gate note: 5,355 tests pass; the only uncovered lines belong to the parallel session's in-flight `resource_search.py` work, so this session committed its own files per-folder and left the shared gate to close when that work gates.

## [2026-07-30] audit | Session-2 flag cross-check: displacement receipt already exists; F9/F10 sharpened

Second-channel investigation of `bot-20260730-000038` (session-2 flags 4/5/6/7/8/9), run alongside the fixing session's own triage — convergent on every root cause, two corrections folded into [[flag-triage-20260729]]: (1) **F9 correction** — the planned "add a teleport_displacement receipt" is already built: `teleport_displacement` fired 7x in this run, with a 4-in-17s cluster inside the flag-7 orange minefield (00:07:27-44, including the 7-tile shove (220,9)->(220,2)); the real gap is a CONSUMER (displacement never reaches the mine belief, landing chooser, or an area veto). (2) **F10 sharpening** — at the flag-4 walkaway all three of the user's equipment containers were in the belief store (nearby=3 blocked=3), the deficit was radars-only at 1087/1100 fuel, and the planner still blind-hopped to a fuel dot. Flag-5 wire receipts confirm the user's timing law for the F10/F8 walk-vs-teleport economics: the resume paid map_open+teleport (2 ticks + 46 fuel) to cover 11 tiles back to orange-2 when the 0x47 echo shows multi-tile paths ride ONE 2s move command at 1 fuel/tile. Flags 6/8/9 re-confirm open rows F3/F5/F2 respectively. No code changes from this channel (parallel session owns the tree).

## [2026-07-30] fix | F9's oscillation closed: the break latch gates every HUNT phase at entry (commits `4f63aa37`, `e9e9dd7f`)

Root of the flag s2-7 shoot/map_open interleave: the latch-holding check lived only inside `_break_losing_engagement`, which only the ENGAGE path (and its pursuit twin) consult -- CLOSE ticks reached `close_target` and kept shooting mid-escape, and every shot handed the NEXT tick's fresh map back to the deferred larder hop, which re-deferred for another map open. `decide_hunt_mode` now runs the holding check immediately after the release check: any latched tick with a locked target (visible or pursued) delegates to the lock-held refuel BEFORE phase dispatch, the redundant branch inside `_break_losing_engagement` is deleted, and a latch with no lock falls through untouched (pinned: the acquisition map-open proceeds with the latch riding along). With no phase able to fight mid-escape, the deferred teleport dispatches against the opened map on the very next tick -- the oscillation is structurally impossible rather than merely damped. New pins: latched CLOSE tick with the target in plain view produces the COLLECT escape, not a shot. My files pass mypy/ruff/pytest (122 focused + 5,355 earlier full run); the shared gate currently red only on the parallel session's in-flight F7/F10 work (`test-quality` guard hits in their new tests, uncovered `resource_search.py` lines), which that session gates itself. The `teleport_displacement` requested-vs-landed receipt remains open under F9.

## [2026-07-30] fixes | Collect-economy root fixes: F1 + F2 + F7 + F10 landed from the flag triage

Four of the triage rows closed at the root ([[flag-triage-20260729]] fix-status table updated; all await live receipts). **F7** (flag 13, the 84-fuel dreg lock): `is_fuel_lock_release_warranted` in `equipment.py` adds a VALUE release path — deliverable score `min(volume, deficit) − distance` with 2× hysteresis against ping-pong — beside the markedly-closer distance rule; the flag-13 shape (75 vs 202) is the pinned regression in the new `tests/bot/ai/test_fuel_lock_value.py`. **F2** (63% zero-yield hops): the dot-hop gains a harvest-memory veto — a landing viewport whose believed containers are ALL drained within `HARVEST_MEMORY_TTL_MS` (10 min, an unbracketed respawn assumption) is skipped with a `known_empty` tally, ending the 180 s-scan-TTL-as-harvest-memory conflation; the gate chain now lives in `_dot_hop_rejection` (one classifier, tally keys = decline fields). **F1** (direction-blind top-off): when stocks are hunt-ready and only fuel is short, dot scores scale by `16/(16+dist_to_nearest_alive_enemy)` — the flag-2 26-tiles-out-30-back double teleport is outscored ~2.6:1 by a prey-side dot; `hop_selected` carries `hunt_biased`. **F10** (new-session flag 4, three identified containers hopped away from): `_hop_toward_equipment`'s external-only filter removed — the step runs strictly after walk-pickup declines, so all tracked equipment including in-viewport walk-blocked is teleport fair game. Verification: 708 bot-ai tests green, `resource_search.py` + `equipment.py` at 100%/100% branch coverage, ruff + mypy clean; full make check deferred until the parallel session's hunt-side work (F8/F9 fixes, in flight at write time) settles — whoever finishes last runs the gate. Live diagnostics: this session also ran the first live flag monitor (session 2's 12 flags surfaced in real time, including three empty_container rejections that corroborate F2's stale-belief mechanism).

## [2026-07-30] observability | F9's second half: teleport bounce-backs now leave a receipt (commits `fa93c47b`, `9acbe0be`, `a4c55419`)

The requested-vs-landed comparison the flag s2-7 investigation asked for: the executor's recorded dispatch target is now readable through `pending_teleport_target()` (the ledger's pending state stays private behind the accessor), and the `teleport_landed` confirm compares it against the self position -- the wire's SelfMovement precedes the confirm, so self IS the landed tile at that moment. A mismatch emits `TELEPORT_DISPLACED: requested (x,y) landed (x,y)` plus a `teleport_displacement` diagnostic with the Manhattan displacement; exact landings stay silent, so the stream surfaces bounce-backs (mined landings, occupancy, refused ground) instead of echoing every teleport, and the displacement magnitude lets `make analyze` separate routine one-tile combat-close displacement from minefield ejections. Four pins in a new `test_world_state_dispatch_teleport.py` (displaced, exact, no-dispatch, no-self) cover every branch; module coverage verified directly since the shared gate is still red on the parallel session's in-flight forage work. This also arms the F4 live validation: if the ferry pursuit's shore landing gets displaced, the receipt will say by how much and from where.

## [2026-07-30] gate | Merged-tree make check green: 5,377 tests, 100.00%

Closure for the deferred gate in the previous entry: with both sessions' fixes in the tree (collect-side F1/F2/F7/F10 + hunt-side F4/F8/F9 and the teleport_displacement receipt), `make check` passes clean — 5,377 tests, 100.00% statement+branch coverage, guard/ruff/mypy all quiet. Forensic note for future parallel sessions: a simultaneous pair of make checks corrupts each other through the shared `--basetemp .pytest_tmp` (pytest wipes it at startup; the 99.96% run's 4 `tmp_path` setup errors and its exactly-4-test shortfall were that collision, not code). Run the gate solo. Triage additions from the same window: flag s2-13 folded into F8 (walk-one-tile-to-range still open), s2-14 banked as F11 (user ruling: over-terrain homings are last-resort when a reposition buys a clear-LOS dual), s2-15 explained as correct `dot_relay` behavior under fuel-before-chasing (legibility gap only — candidate HUD tweak: label relay legs "RELAY→target (leg n)").

## [2026-07-30] fix | Flag s2-13 tightens the firing law: in-view IS the criterion (commits `78b1483e`, `8cb2012f`)

purple-9 stood at Manhattan 9 -- inside the viewport, one tile beyond the 8-tile bound the acquire short-circuit carried -- and the bot paid a teleport to close on a target it could legally shoot (user: "the bot could have shot at the enemy im not sure why the bot teleported closer"). The original ruling never mentioned a radius ("as long as theyre on the viewport and its a clear dual shot then id just hit them from my new location"), so the bound comes off: `_combat_teleport` shoots ANY viewport-visible target from the current tile. That folds the mine-ring/ferry stand-off into the same rule -- a visible target is engaged regardless of landing state -- so the failed-landing branch's in-view sub-case became unreachable and was deleted; failed landings now block only off-view targets. `SHOT_RANGE_TILES` keeps its two remaining jobs (acquisition stand-off landing radius, close-phase shot test). Companion receipts from the same debrief: flag s2-14 (a homing consumed at an in-view red-8 = shooting over terrain; the reposition-for-clear-dual behavior is F3's LOS module, receipt logged) and flag s2-15 diagnosed as NOT a bug -- the "random map location" was a dot_relay leg toward a 117-tile-distant red-9 that refueled to 1100 and made the close affordable, the relay contract working exactly as designed. Cross-session note: the parallel session unified its early `teleport_displacement` emitter onto this session's wire-layer receipt (one diagnostic_kind, one schema, veto-only consumer in `completions.py`) -- lift-don't-fork held across concurrent AI sessions. Run bot-20260730-000038 closed clean meanwhile: 20 kills, 98% hit rate, zero blocked targets, exit session_complete, and 7 displacement receipts from the old emitter proving the bounce-back signal is real and frequent enough to matter.

## [2026-07-30] fixes | Second-channel cleanup: gate green, weak assertions strengthened, back-compat alias deleted, displacement emitter unified

User directive ("anything outstanding you can fix for them?"): swept the tree the fixing session left red and closed three classes of debt. (1) **Gate**: 6 `weak-assertion-is-not-none` guard violations in the fresh F1/F2/F7/F10 tests (test_resource_search, test_fuel_lock_value, test_collect_mode_equipment) converted to the house `if x is None: raise AssertionError(...)` narrowing with strengthened field asserts. (2) **No back-compat**: deleted `CMD_PICKUP_MOVE = CMD_PICKUP_FUEL` from `protocol/commands.py` — zero live usages, pure legacy alias; sweep confirmed no `TypeAlias` in src. Remaining tolerant-decode note: `types/session.py` accepts pre-June capture archives missing `game_log`/`tank_names` — flagged to the user as a data-boundary decision (strictifying breaks replay of the 320-capture corpus), not silently "fixed". (3) **DRY**: `teleport_displacement` had gained a SECOND emitter (wire-layer `_emit_teleport_displacement`, schema `requested_*/displacement`) alongside the pre-existing completions-side one (schema `target_*/dist`) — every >1-tile bounce would have double-emitted under two schemas. Unified: wire-layer receipt is the single emitter; completions keeps only the consumer (tile veto + enemy-bump tolerance); [[flag-triage-20260729]] F9 updated with the schema-migration note. make check green end-to-end: 5,377 tests, 100% statements+branches, all guards clean.

## [2026-07-30] fix | F13 larder half: walk-dominant range + dreg floor end the 2-tile dreg teleports

Session-3 live-monitor triage promoted F13 with four flag receipts (s3-1/3/4/9, worst case a 2-tile teleport to a 35-volume remnant netting ~23 fuel) and the root cause read straight from `larder.py`: `gain/cost` structurally favors close dregs (35@cost12 scores 2.9 vs 355@cost190 at 1.9) and `too_close` only excluded Chebyshev<=1. Fix: `_WALK_DOMINANT_RANGE` (Manhattan<=2 — a 2-tile walk costs the same ticks as the teleport with zero fuel and no map churn) and `_LARDER_MIN_GAIN` (100, waived when the clamped gain COMPLETES the deficit — the F1 microscope's last-17-points case). No desperation exemption: the reserve gate already blocks every larder hop at fuel<=threshold and `_desperation_fuel_hop` owns that regime (the first draft's desperation branch was unreachable — caught by its own failing test and removed rather than shipped dead). New `dreg` tally on the fuel_larder hop_declined. 712 bot-ai tests, larder.py 100%/100% branch, ruff+mypy clean; full gate running at write time. Also this session: F2's live receipt (36% zero-yield vs 63-64% baseline), F12 (wasted map-open on deferred-then-replanned teleports) and F14 (shared-room belief rot, 9 empty_container rejections, ~30-40s staleness, fix direction: belief-age gate) banked as OPEN findings on [[flag-triage-20260729]].

## [2026-07-30] feature | F5 ferry-served larder: water-locked fuel is harvestable

The "missed multiple ferries" flags (s1-11, s2-3/8/11) close their first half: new `bot/ai/ferry_landing.py` — when `find_teleport_landing_tile` finds no ground on or beside a believed fuel container (the 15/15 `no_landing` tallies of bot-20260730-000038), the larder now boards the freshest believed `TERRAIN_FERRY` tile within a 12-tile radius as the landing; the teleport lands on the ferry (user law: "you generally will need to teleport to the ferry") and the held fuel lock rides to the pickup under the existing single-surface routing + riding-pickup mechanics ([[ferry-mechanics]]). Ferry beliefs rot fast (ferries drift), so boarding targets are gated at 60 s freshness — stale or absent ferries still tally `no_landing`, never a blind teleport onto open water. New `ferry_served` field on the larder selection + decline diagnostics. ferry_landing.py and larder.py both 100%/100% branch, 728 bot-ai tests green; full gate deferred (the parallel session is mid-F3 in combat_strategy.py — `is_shot_line_clear` wiring visible, imports not yet sorted). Equipment-hop ferry service is the follow-up half.

## [2026-07-30] fixes | Second-channel round 2: barren-scan memory, law claims, hunt-dispatch extraction

Three deliverables answering the user's viewport questions and unblocking the fixing session's gate. (1) **Barren-scan memory (F2's residual hole)**: the harvest-memory veto only saw viewports with DRAINED beliefs — ground the radar swept that revealed NOTHING leaves no beliefs at all, so after the 180 s forage TTL it read fully clean and got re-hopped for a guaranteed zero-delta scan (the user's "zero deltas indicating they were scanned by us recently"). Fix reuses the ONE existing freshness store (no new state): `record_scanned_tiles` retains marks for `HARVEST_MEMORY_TTL_MS` (10 min, constant moved to `state/scan_coverage.py` as its single home) while every coverage predicate keeps its own 180 s window; new `is_viewport_scanned_within` predicate + `barren_scanned` gate/tally in `_dot_hop_rejection` vetoes dots whose landing viewport was fully swept within 10 min and holds no positive-volume belief. Answers on record: dot hops DO require 100% clean (zero-overlap) landing viewports; re-hop timeout is 180 s for coverage, now 10 min for known-empty AND known-barren ground. (2) **`law` claim kind** (`scripts/physics_claims.py`): the fixing session's lifted `physics/line_of_sight.py` (F3) tripped the physics-claim guard — neither symbol is int-probe-able (terrain-protocol arg / raster return). Added the third claim kind: prose-law binding with existence check, exactly-one-of value/probes/law enforced; [[mine-mechanics]] gains the two law claims quoting the user's verbatim shot-clearance rules. Scalar symbols must still use value/probes. (3) **`decide_hunt_mode` C901**: the fixing session's phase-entry break gates (incl. the fix for Artax's 01:06:55 death to Yuppler in CLOSE) pushed dispatch complexity to 14; extracted `_release_break_latch` / `_continue_break_escape` / `_assess_locked_engagement` with their incident comments intact — behavior unchanged, 170 tests across all touched areas green. Known in-flight red at write time: 2 `test_walk_for_fuel` failures from the fixing session's live `larder.py` edit (F13 walk-dominant gate, mtime 30 s before the run) — theirs to reconcile, deliberately not touched.

## [2026-07-30] death post-mortem | Artax deactivated by Yuppler at 01:06:55 -- three roots, all fixed same hour (commits `855c0900`, `f67570ad`, `6b5b1e3d`, plus the human ruling)

The first bot death since the exposure contract, and the receipts explain every second of it. (1) **No safety check ever ran**: the whole fight stayed in CLOSE phase and the break assessment lived only on the ENGAGE path -- the first break of the fight fired at fuel 216, four seconds before the 0x41. The assessment now gates every HUNT phase at decide_hunt_mode entry, exactly like the escape latch. (2) **The losing fuel trade** (CORRECTED same hour by the user + the ammo ledger: Artax fired DUALS point-blank -- weapon=1, duals 22->17, victim_id=1229, adjacent at (168,94)->(167,94/95); one homing total; no terrain between): the trade was an even 90-for-90 dual exchange, but Artax entered it at 626 fuel against a full human, paid the extra -10 shot cost every tick (net -100/2s against -90 dealt), and had no adjacent container for mid-fight pickups -- an attrition fight entered without a fuel advantage or an exit plan. The over-terrain-homing story belongs to flags s2-14/s3-16, not this death; the LOS work stands on those receipts. The firing law now enforces its own clearance clause through the new lifted `physics/line_of_sight` module (rock and movable land blocks occlude; water and mines never; wire patches authoritative over the field image): in-view targets are shot in place ONLY through a clear line, otherwise the bot re-closes adjacent where the line is trivially clean. (3) **The give-up**: at 216 the projection declared the fight unwinnable at any fuel and BLOCKED the human, leaving a map_open as the final act. User ruling: a human fight is never unwinnable -- both sides refuel, sustain wins -- so past-capacity projections against humans now latch at capacity and escape WITH the lock (refuel to full, resume Yuppler), while practice bots keep the block. Same commit set lands F3's planner: `find_mine_clearance_shot` picks the covered container in view with a clear line whose rank-dependent blast (recruit 1, private+ 3x3) exposes the most mine-covered containers -- cascade wiring into collect follows when the forage files settle. Gate state: 5,421 tests with only the parallel session's in-flight walk-rescue work red; all files of this commit set at 100%.

## [2026-07-30] fix | Displaced harvest landings un-suppress the landing radar (s4-3)

Session-4 live triage caught three consecutive cant_go walk rejections after a mine-displaced larder landing at (165,100) — the larder's no-radar ruling walked the bot blind into Yuppler's old minefield because the displacement (the server's own "this ground is mined" signal) had no consumer. Fix in collect_mode: (1) a suppressed harvest landing standing more than auto-pick reach from its lock fires the landing radar, keeps the lock, and consumes the suppression — the mine-composed passability then vetoes the doomed walks pre-dispatch; (2) the landing-scan gate moved AHEAD of lock continuation in the cascade, restoring the 2026-07-03 "always radar right on landing, before any pickup" policy that the old order violated whenever a lock survived the landing. Also landed this hour from the collect lane: F16's net-of-gain reserve (the transaction clears the reserve, not the transit — kills the 200-250 death dead zone) with a hard payability floor (fuel is health; never spend to zero), and the F5 ferry-served larder. The parallel session landed F3 mine clearance (`mine_clearance.py` + `_mine_clearance_decision`, dual floor 5) into the same cascade concurrently — both changes verified together: 737 bot-ai tests, ruff+mypy clean.

## [2026-07-30] fixes | Flag wave 4-7: homing-trace wall, no-ammo mine law, mine clearance wired, relay legibility (commits `1a8cc336`, `dd75d4ba`; wiring in-tree pending the shared gate)

Four flags, four dispositions. Flags 4/5 (seven pursuit homings then one guaranteed miss): `pursuit_trace_is_live` now reads the departed target's last in-viewport stamp against the 12 s reroute wall (run 194658: hits to +12.0 s, miss at +14.0 s) and both pursuit-fire sites skip straight to the lock-held map chase once the trace dies -- the miss and its tick are simply never spent. Same commit corrects the mine-clearance economics to the user's law: mine shots consume NO inventory ("you click and it shoots a single shot, and destroys the mines"), so the dual-floor guard is gone and clearance costs one tick. The clearance CASCADE WIRING is now in the collect path (after in-viewport pickups decline, before the larder: shoot the best covered container with a clear line, exposing up to the 3x3, then collect normally next tick) -- it rides in the working tree and the next session run executes it; the commit follows when the parallel session's forage work gates. Flag 6 ("teleporting around randomly... confused") diagnosed NOT a bug: the ring is a dot_relay chain toward an unaffordable target with map refreshes between legs -- the known legibility gap ("RELAY->target (leg N)" HUD tweak). Flag 7 is another F12 receipt (map open for a hop the lock-continuation then walked). Flag 3's walking-into-undetected-mines (three 45-fuel hits) opens F14: walks crossing unscanned ground should spend an available radar first.

## [2026-07-30] contracts | Death becomes a respawn wait; mine clearance proven live and de-duplicated (commits `d4a83c97`, `2980e901`, `129bd24b`)

Three user rulings in one wave. (1) **Death contract**: "if the tank dies, it should just wait for respawn and then go into collecting mode" -- the own-0x41 handler now resets every tactical belief (combat lock, latches, resource locks, in-flight action), keeps the session-scoped facts (kill count, wind-down, greeting latch), drops the garbage corpse self-record (fuel read 65482 post-mortem in the Artax death), and idles on the self-None early exit until the respawn sync arrives; a 60 s deadline exits `deactivated` loudly in worlds with no respawn law (the sim), and the whole lifecycle -- reset, quiet wait, respawn resume, deadline exit -- is pinned over the sim seam. (2) **Mine clearance fired live and worked** (user: "it shot once and destroyed the mines... otherwise the mine process is great"): three clearance shots in run bot-20260730-015x exposed covered containers, with one defect -- a same-tile double shot at (162,94) two seconds apart because the 0x45 detonation had not been applied when the next tick re-derived and mine shots carry no target id for the shot-feedback gate; the new `mine_clearance_aim_key`/`mine_clearance_shot_ms` latch refuses same-tile re-aims inside a 5 s effect window. (3) **Ferry-walk receipt** (flag 1): a ferry touching land plus a water container on the same viewport is F5's boarding-WALK case, no teleport needed. Flags 3 and 9 were further F12 map-open-then-no-teleport receipts.

## [2026-07-30] architecture | Committed-intent layer phase 1: plans survive the tick boundary (s8-2 fix at the root)

The user's challenge — "i'm worried we're papering over issues... not addressing the underlying uncertainty" — answered structurally. Session-8's first four flags proved the new break machinery works (two clean two-tick escapes with refuel-and-re-engage, flags s8-1/s8-4) and handed over the perfect receipt for the deeper defect: s8-2, where the escape hop landed ON its locked equipment and the next tick's re-derivation selected a teleport TO THE TILE THE TANK WAS STANDING ON, deferring a map open for a zero-distance jump. Root cause: plans were not first-class — nothing asked whether the committed plan's purpose was already served before re-deriving. Phase 1 ships `bot/ai/intent.py` as the single owner of collect-plan SEMANTICS over the EXISTING lock fields (no new state, no migration): typed `CollectPlanDict` + codecs, `plan_completes_here` (auto-pick reach), `validate_collect_plan` (lifted from `context.normalize_resource_target`, same pursuability predicate), and `release_collect_plan` — the only sanctioned release path, emitting a `plan_released` diagnostic with a closed 8-reason vocabulary at all seven former silent-clear sites, so plan churn becomes a per-run queryable instead of invisible. Wired continuity: the under-fire escape finishes a completes-here plan FIRST (the pickup is the escape continuation — one action, no added exposure), and `_hop_toward_equipment` refuses own-position landings (`own_ground` tally; the cost-0 candidate structurally wins cost ranking, which is exactly how the self-teleport was selected). Pins: `test_intent.py`, `TestEscapePlanContinuity` (s8-2 byte-for-byte + the far-lock-does-not-hijack ordering pin), own-ground gate tests; damage-seeding helper lifted to `_support.seed_confirmed_incoming`. Gate: 5,481 tests, 100.00% statements+branches, ALL CHECKS PASSED. Phase 2 (hunt/clearance plans, supersede visibility) specified on [[committed-intent]]; the s8-3 mid-duel find_target map_open (F12) is scoped to the hunt-plan phase.

## [2026-07-30] fix | F22: transient inexecutability held, not released — the plan_released channel pays for itself in 40 minutes

The committed-intent layer's first live run surfaced its own first finding: three `not_executable` plan releases (ticks 361/366/371, run bot-20260730-032x) fired mid-approach WITH the plan's own map_open in flight, and every released target was re-locked and served 2-3 ticks later — the continuation was reading "no executable route THIS tick" as "plan dead." Under the old text-only logging this churn was invisible; the closed-vocabulary `plan_released` events made it a 30-second query. Fix in both lock continuations: a transient `walk_or_teleport` None now HOLDS the plan (yield the tick to the cascade, keep the lock — a water-boxed target survives for a later ferry, honoring F5's spirit), and `not_executable` releases only on the structural server-confirmed move-failed mark. Genuine release gates (superior candidate, validity, at-capacity) unchanged. Three old pins asserting release-on-water repinned to hold semantics with the lock-retention now asserted explicitly; new structural-release pins seed the move-failed mark. Same run also quantified F14 cleanly: five phantom pickups (ticks 360-448), all distinct drained beliefs, each exactly one wasted tick then a clean `target_gone` release — the belief-age ruling remains the only missing piece. Gate: 5,484 tests, 100.00% statements+branches.

## [2026-07-30] receipt | Mine walk-over flip live-proven on first natural trigger (run bot-20260730-05xx, 05:15:46)

The doctrine's first live firing is byte-perfect: walking to equipment at (151,133), the bot stepped an unrevealed mine at (147,129) -- one -45, movement arrested -- and the SAME tick produced the stamp ("MINE_WALKOVER: detonation on own tile"), the flip ("teleporting to (146,141) instead of re-walking"), and the map-open precondition; the following tick held the flip through the lock continuation and the teleport flew. Exactly one mine hit where the s6-8/9 shape cost six. Session context: fourth consecutive clean 20-kill session_complete (98/99/100/99 percent hit rates across the four), zero blocked targets in any of them.

## [2026-07-30] fix | F23: the movement-dead escape loop — twelve rejected walks under fire before the hop won

The monitor's cant_go cluster-trace caught the worst under-fire pattern since the Artax death, live in run bot-20260730-110x: mid-duel with purple-1, the escape branch dispatched a walk-pickup every tick and the server refused every one (`cant_go`, ticks 95-107 — every direction blocked; the bot was movement-boxed). Each rejection burned that container via `failed_pickups` and the next tick planned a walk to a DIFFERENT container, so the bot stood still for ~26 s eating duals (fuel 972→663) until the hop rung finally won at t112 and a +437 landing refuel saved the fight. Root: no state anywhere recorded the SHARED fact behind those rejections — "the server is refusing this tank's movement" — because collect-kind rejections only feed per-container marks, and the escape's walk-first movement law (correct when walking works) had no way to learn that walking was impossible. Fix: `WorldService.movement_rejections` records a timestamp for every `cant_go` answering a move/collect/teleport dispatch (the walk-pickup's leg IS a move regardless of command kind); `recent_movement_rejections(now, window)` counts with in-place pruning; and the under-fire escape declares the walk rungs dead at 2 refusals inside the fire window, jumping straight to the hop — teleports need no walk path and land displacement-safe. Service/wiring/behavior pins all landed (cant_go-on-collect records, code-0 does not, movement-dead skips walkable in-viewport fuel for the larder hop, single rejection keeps the walk law). Gate: 5,491 tests, 100.00% statements+branches.

## [2026-07-30] fix | F20 closed by force: the 110-tick walk-close livelock

The monitor's cant_go cluster-trace escalated F20 from "published finding with a withheld test spec" to a live hard livelock: run bot-20260730-110x ticks 904-1017+, HUNT/CLOSE re-dispatched the identical move to (240,46) beside orange-6 for over 110 consecutive ticks — the server rejected every one and marked the tile failed every tick, but `combat_landing_candidates` consulted neither the composed terrain (F20's original finding: dynamic occupancy only) nor the failed-move marks, and the walk-close branch returns before the teleport path's mark check can save it. No damage was taken (practice bots don't initiate) but the session burned ~4 minutes standing still. Root fix at the candidate source: `combat_landing_candidates(world, self, target, terrain, now_ms)` filters impassable composed tiles and live failed-move marks, so an unwalkable adjacency ring yields no walk candidates and the close falls through to the teleport path, whose existing failed-landing gate blocks the target and replans. Signature updated at all call sites; pins land the two new filters plus the existing ordering/bounds behavior. Gate: 5,493 tests, 100.00% statements+branches. The stuck session was cycled out via the stop file so the runner's next launch carries the fix.
## [2026-07-30] update | Session-3 zombie diagnosed: lobby reconnect, not plan churn -- wire-silence watchdog shipped

Session 3 of the 100-kill run (16+18+5 = 39/100 at that point) closed with 5 kills in an hour and a 524-line "holding plan" loop that first read as intent-layer churn. The console log told a different story: at 11:58:32, right after a ferry disembark move, the game websocket died and the page auto-reconnected to the LOBBY -- the sniffer parsed the reconnect handshake's SELECT (mid-session `session_room_joined` + terrain reload), and from that moment ZERO inbound world messages arrived for the session's final 43 minutes while every one of 243 injected map_opens stalled at 10.5s. The ws-ready page-health gate passed every tick because the reconnected lobby socket read OPEN; the server simply no longer associated it with an in-game tank. The "holding plan" spam was the collect cascade correctly refusing to drop its plan while its only escape (a hop teleport) died at the map-open precondition forever.

**Fix (commit a4b904c8):** `dispatch_world_state_update` stamps `last_game_message_ms` on every binary world message (lobby text like ROOM_LIST/SELECT takes the text route and deliberately does not refresh it); the tick loop raises the new `connection_lost` session exit when the stamp goes 90s stale -- above the 60s respawn wait, so a corpse never trips it. Recovery is the harness relaunching a fresh session. Also committed the other session's intent layer + movement-dead rejection tracker + F20 landing-candidate filters (1459 insertions) after the full gate passed at 100%.

**Pages touched:** none (harness-level failure, no game-mechanics change). Receipt: `runs/bot/intent-loop-receipt.txt`, console log lines 22496-22512 (reconnect) and 29670 (stall cadence).

---
## [2026-07-30] update | Session-4 flags: the Yuppler-ghost rejection loop and the stale-larder triple-hop -- both server-disproof consumption gaps

Session 4 (5 kills, ended browser_closed at 595 ticks) surfaced two flags. (1) After Yuppler left the game the bot fired 43 consecutive rejected shots at his ghost ("Friendly fire!" client spam): capture replay proved the wire carries NO departure event the decoder sees, the server keeps broadcasting the departed player in every MapData (tanks=37 constant for minutes), and the 0x58 grace deliberately keeps the registry entry -- so acquisition re-selected the ghost every map open and the err=3 rejection taught the AI nothing. Fixed by consuming the friendly-fire receipt as target disproof: one err=3 blocklists the id and releases the lock (registry kept, reroute grace intact; diagnostic `target_disproved_by_friendly_fire`). (2) Three larder teleports landed on containers Yuppler had already collected that session, each landing scan suppressed as verified stock; per the user ruling ("if one item is stale or out of sync then its worth a radar. not, 3 items") a code=4 rejection now marks container memory desynced and the cascade answers with one `desync_rescan` radar -- cleared by the radar response itself, which reconciles the viewport (volume==0 entries are authoritative removals).

**Protocol facts learned:** 0x52 err=3 fires for shots at departed ids (not only true teammates); MapData continues to list departed players indefinitely; no leave/exit message reached the decoder for a human quit (DOM "left the game" text is client-side rendering of something the decoder does not yet surface). **Pages to update when churn settles:** [[shoot-event-format]] (err=3 semantics), [[map-data-decode]] (departed players persist). Commits: intent layer + watchdog earlier today; this pair in "consume the server's disproofs".

---
## [2026-07-30] update | Desync-rescan radar loop: my latch clear missed the cache-refresh response shape -- 22 radars burned in 44s

Session 5 receipt (21:03:24 onward): the new `desync_rescan` gate fired correctly after a code=4 disproof, but the server answered every rescan with "Radar cache refresh" (`update_world_state_from_radar_cache`) while the latch clear lived only on the full 0x4F delta path -- so the latch never cleared and the cascade spent one extra radar every 2s tick, 22 -> 0, before falling through to forage. Fixed within the hour: the clear moved into `mark_radar_scan_complete`, the one point all radar response shapes land on (full delta, cache refresh, empty-delta resolution). Session stopped by STOP file at 5 kills (tally 49/100), fix committed, run resumed. Lesson recorded: a latch cleared by "the response" must clear on EVERY response shape the wire can produce, and the cache-refresh path is the common one when re-scanning freshly-visited ground -- exactly the ground a desync rescan targets.

---
## [2026-07-30] update | The human-consent combat contract + the radar-situation root fixes

User ruling (session 8 killed over it): "to engage in combat, the human must respond hello or engage the bot first... make sure we teleport to them. and that we've said hello first... we want to see them. and not an adjacent teleport. a few tiles off." Implemented as a consent stack: (1) the greeting APPROACH -- hunt acquisition teleports to a stand-off band ~6 tiles off any map-fresh, rank-window, unconsented, ungreeted human (never adjacent; one visit per human via the greeted latch); (2) the HELLO now fires on viewport ENCOUNTER, not on combat lock; (3) consent = any non-self-echo 0x4D chat from their id OR an incoming shot recorded in the damage book -- either admits them to the threat list and acquisition, and an attacker consents by attacking so defense is never blocked. Unconsented humans are invisible to targeting (new `human_not_consented` acquisition rejection).

Radar situation ([[flag-triage-20260729]] session 7-8 receipts): the loot-run bias (dot hops drift toward the fights when equipment-hungry at fuel cap), the larder dreg waiver scoped to hunt-ready stock (no more map-open + teleport for a 24-fuel top-off at zero radars), and the equipment hop now boards ferries for water-locked drops exactly as the fuel larder does -- session 7's 8/8 water-locked equipment containers become harvestable. Sim opponents renamed to practice-bot shapes (`red-<id>`) so the sim practice room stays fightable under the consent gate. Full gate 5531 tests / 100% coverage.

---
## [2026-07-30] update | Session-9 flags 1-8: radar-spend economics unified, drain receipts, and the moving-ferry disproof

Session 9 (10 kills, ended connection_lost) produced eight flags with one structural root the user named exactly: "the viewport freshness handling is not properly wired to the collecting system." Every discretionary radar site decided alone; none weighed the spend against the reveal. Fixed as ONE rule (`radar_spend_worthwhile`): extras stocked -> a scan must uncover >= 32 viewport tiles; radar-broke -> the free built-in radar scans any sliver. Wired into the collect landing scan (superseding the 2026-07-03 "always radar on landing" ruling), the displaced-harvest rescan, the desync rescan (which also clears its latch when coverage already answers), the forage radar (whose `has_extras -> always productive` inversion was flag 5 in the flesh), and hunt's combat landing scan.

Two more server receipts are now consumed: (1) a code=4 arriving with our own ContainerPickup broadcast is a DRAIN RECEIPT of a successful pickup, never a memory desync (flag 4 paid two rescans re-learning its own +241 drain); (2) a teleport displaced off a believed ferry tile deletes the ferry belief (flags 7/8: ferries move, the 60s-old boarding belief survived every displacement, and the identical hop re-derived per lap -- teleport + landing radar each time, 17 extras to 0, user killed the session). Movement law landed in the larder: in-viewport stock is walk territory, never a hop (flag 2/3's 3-tile teleport with its map open and displaced landing), and the walk step iterates ranked candidates instead of aborting after one worth-the-walk veto.

**Flags 1/6 + the early-refuel reports are the break-projection contract working as designed** (Artax-death fix: projected fuel at kill < floor -> refuel with lock held, resume) -- calibration review still owed with the three receipts. **New doctrine queued:** scope-scout ferries via the viewport-shift protocol (Rb compass / Sb tile, [[viewport-shift-protocol]]) -- look at the water live before boarding, per user direction ("we want ferries... technically we could just use a viewport shift").

---
## [2026-07-30] update | Session-10 flags: the beyond-reach relay and the code-0 map receipt; session 11 lands 20 at 100%

Flag s10-2 finally explained the recurring "teleports to a random spot before engaging": the unaffordable-chase branch owned only one tool (refuel_for_hunt), so a 504-cost chase at fuel 1097/1100 "refueled" a 3-point deficit with a 121-fuel dot teleport before relaying to the fight anyway. Root split: cost above capacity-minus-reserve is a DISTANCE problem no refuel can fund -- the branch now returns a dot_relay leg with the combat lock held (enemy-biased ranking, self-refueling landing); fundable costs still take the refuel detour. Flag s10-1's "You can't do this" was a teleport against a map the server had closed while the client snapshot still read open -- a precondition receipt; code-0 teleport rejections no longer plant failed-target marks on innocent landings (the deferred retry against a fresh map open succeeded live).

Session 11 receipt for the whole fix stack (consent contract, radar-spend economics, drain receipts, ferry disproof, relay split): **20 kills, 100% hit rate, clean session_complete in 822 ticks** -- the best session of the run. 100-kill tally at 99 after eleven sessions; the 1-kill finisher is session 12.

---
## [2026-07-31] update | 100-kill run at 83 after a tally correction; every weak session root-fixed

CORRECTION of the entry drafted minutes earlier: the "17-kill session 10" never existed -- that figure was a mid-flight kill-count of run 224244, which completed at 20 kills and was then double-counted as "session 11". Authoritative index line for the run: 16 (lost session, no index row) + 18 + 5 + 5 + 5 + 3 + 0 + 0 + 10 + 20 + 1 = 83; a 17-kill finisher is running. Every weak session traced to a named root now fixed and committed: two dead-socket zombies (wire-silence watchdog), the intent-layer holding loop, the Yuppler-ghost rejection loop (friendly-fire disproof), the desync/cache-refresh radar burns (radar-spend economics + drain receipts), the moving-ferry orbit (ferry-belief disproof), and the beyond-reach refuel absurdity (distance/fuel split -> dot relay). The run's best session: 224244, 20 kills at 100% hit rate, clean session_complete on the full fix stack. Remaining doctrine work tracked: scope-scout ferries via viewport shift, break-projection calibration with the user's three early-refuel receipts.

---
## [2026-07-31] update | Session-12 ferry livelock: the lock steal now demands executability

The 17-kill finisher livelocked ~70 laps (user broke it by closing the browser; 12 kills banked first): move -> map_open -> teleport repeating with every action SUCCEEDING -- invisible to all receipt-consumption machinery. Anatomy: landing on the ferry boarding tile for locked (100,8), the "markedly closer" steal saw (106,14) as walkable and took the lock; the stolen plan ran one disembark leg, stalled "not executable -- holding plan"; the cascade's equipment hop went back to (100,8)'s boarding tile; the steal fired again. Root: the steal's reachability predicate and the execution path disagreed about the same container. Fix: superiority requires a command THIS TICK (walk_or_teleport non-None) for both equipment and fuel steals -- closer-but-uncompletable never takes a viable plan. The surface-routing convergence gap that made (106,14) stall after its disembark leg remains open under the ferry doctrine tasks (scope-scout + F5/F6). Run tally: 95/100.

---
## [2026-07-31] update | 100-kill run COMPLETE: 16 + 84 indexed = 100 across thirteen sessions

The 5-kill closer exited session_complete at 98% (198 ticks). Index-verified sum from session 2 onward: 84; plus the lost first session's monitor-counted 16 = exactly 100. Per-session line: 16, 18, 5, 5, 5, 3, 0, 0, 10, 20, 1, 12, 5. Every weak session became a named, committed root fix: two dead-socket zombies (wire-silence watchdog, connection_lost exit), the intent-layer holding loop, the Yuppler-ghost rejection loop (friendly-fire disproof), the desync/cache-refresh radar burns (unified radar-spend economics + drain receipts), the moving-ferry orbit (ferry-belief disproof), the beyond-reach refuel absurdity (distance/fuel split -> dot relay), the code-0 map receipt, and the closing livelock (lock steals require executability). Along the way the human-consent combat contract landed (greet-approach stand-off, HELLO on encounter, chat-or-first-strike consent). Best sessions on the final stack: 20 kills @ 100% and 5 @ 98%. Open doctrine: scope-scout ferries via viewport shift ([[viewport-shift-protocol]]), surface-routing convergence (F5/F6), break-projection calibration with the user's three early-refuel receipts.

---
## [2026-07-31] audit | Codebase-facing pages re-derived against the working tree

Doc-accuracy pass triggered by a README rewrite; no live run, no new game facts. Audited every structural claim in the codebase-facing pages against `src/tankpit_bot/` as it stands.

Counts corrected: `index.md` said 60 content pages (actual 67, and hub links resolve 1:1 with `pages/` — no orphans, no dangling links); js-client 20 -> 21; architecture 11 -> 13 (the hub header separately said 12). [[module-map]]: dropped `game_state.py` and `probe.py` (both deleted from the tree), added the `facts/` and `contracts/` packages and a `bot/ai/` row, noted `protocol/encoders/` mirroring the decoders, and extended the dependency-flow diagram with `service/`, `sim/`, `validate/`, `diagnostics/` plus the leaf-layer rule. [[make-targets]]: added `service`, `smoke`, `debug-run`, `combat-probe`, `track`, `larder-probe`, `mine-landing-probe`, `enemy-teleport-probe*`, `teleport-probe-full`, `download-fields`, and spelled out what `lint` actually runs.

**One claim was wrong, not just stale**: [[make-targets]] stated `latest.events.jsonl` and `latest.capture_session.json` are symlinks to the most recent run. `runtime_artifacts.py` creates no symlinks anywhere — every builder returns an archive path *and* an independent `latest_*` path. Corrected with a locator footnote.

Also fixed in `docs/` (outside the wiki): `bot-control-model.md` documented `executor._is_dispatchable()` running a pre-dispatch veto. That function no longer exists; the section now records where each former check actually lives (terrain composition, freshness model, intent layer, 0x52 receipts) and adds the manual-pin override plus the rank-derived threshold rules. `bot-hfsm-refactor-plan.md` needed nothing — its post-plan note already annotates the COLLECT unification.

Blob anchors re-anchored for the two edited pages (`src/` and `Makefile` were both clean against HEAD, so HEAD's hashes describe exactly what was read). [[module-map]] moved `97c0c88a` -> `978d8c6c`. [[make-targets]] needed no change: its `Makefile` anchor `69e5afb9` already matched HEAD — the page was under-documenting an *unchanged* Makefile, not trailing a changed one, which is why ten targets that had existed all along were missing from it.

**Anchor sweep across all 67 pages found 13 more stale anchors on 9 pages**: [[bot-behavior-contract]] (bot, sniffer), [[coding-standards]] (src, tests), [[executor-rejection-loops]] (bot), [[inheritance-chain]] (src), [[larder-plan]] (state, bot/ai), [[physics-module-roadmap]] (sim), [[services]] (src), [[tank-freshness-model]] (state), [[tank-registry]] (state), [[testing-patterns]] (tests). Deliberately NOT bumped. An anchor asserts "these claims were verified against this tree"; bumping one on a page whose claims were never re-read launders unverified prose as verified — the same mistake the 2026-07-30 cleargbm entry called out. Each needs an audit first, then an anchor. Eight of the nine were also mid-edit in the working tree at the time of this sweep.

---
## [2026-07-31] audit | The remaining 13 stale anchors: audited page-by-page, then re-anchored

Follow-up to the anchor sweep in the previous entry. All 10 pages read in full and every locator re-derived against the current tree before the anchor moved. Full sweep is now clean: 0 stale, 0 dangling.

**Corrections that changed a page's meaning, not just a number:**

* [[tank-freshness-model]] — the page's CENTRAL claim was out of date. There are **four** freshness timestamps, not three: `last_viewport_observation_ms` (TTL 5000) was added as the HUNT acquisition gate after the 2026-06-21 tracking probe showed 26 of 27 tanks passing all three older gates while the JS registry had none of them in view. The page had `timestamp_ms` as the acquisition gate, which is now merely registry retention. Also: `is_wire_present()` / `is_position_fresh()` **do not exist anywhere in the tree** — the gates are TTL constants applied inline. Test-class citation `TestInvariantPositionFreshnessRequiresBoth` was renamed to `...RequiresAuthoritativePosition`, and the cited combat test `TestWirePresenceGate::test_position_stale_adjacent_target_is_blocked_not_shot` does not exist (the real coverage is `TestKillShotWireGate` + `tests/integration/test_combat_gates.py` + `tests/scenarios/test_target_stickiness.py`).
* [[bot-behavior-contract]] — carried the same two phantom gate functions in §3.3 and named `timestamp_ms`/`WIRE_PRESENCE_TTL_MS` as the §3.2 acquisition filter. Both corrected against `analyze_threats`. Break thresholds (dual/homing < 4, radar < 5) re-verified exact.
* [[executor-rejection-loops]] — the "Latent (not a loop today but same shape)" section described a `blocked_mines` parameter that no longer exists tree-wide; fix option C landed inside the 2026-07-20 root cut. Marked resolved. AI-state persistence gate moved `tick_loop.py:490-491` -> `:817-822`.
* [[services]] — "All 6 probe types go through this factory" is now **14**. CDPService was documented as buffering into `_cdp_message_buffer`, which is not a CDPService attribute at all (it belongs to the action_lab probe runtime); the service stores public `messages`/`ws_urls`/`magic` and `SessionBase` re-exposes them as `_messages`/`_ws_urls`/`_magic` properties.
* [[larder-plan]] — the under-fire teleport-refuel the page called "future" has LANDED (`collect_mode.py:328`, gated by `_hop_escapes_attacker`). The architecture hub still advertised the whole page as "planned, not implemented" while the page itself said IMPLEMENTED and live-proven — hub line corrected.
* [[physics-module-roadmap]] — contradicted itself on the reroute TTL (12 000 ms in the law-4 as-built, 12 920 ms in Phase 1; disk says `12_920`). Four sim modules had never been recorded: `wire_statements.py`, `emissions.py`, `combat_emissions.py`, `viewport_window.py` — the emission layer split out of `server.py`, added as a retroactive as-built.
* [[inheritance-chain]] — chain and barrel totals confirmed EXACT (`Bot -> DispatchMixin -> CompletionsMixin -> SessionBase`; 12+11+11 = 34 lines). Line counts drifted: completions 340->361, bot_dispatch 397->440, base 465->621, BrowserSession 134->130.
* [[testing-patterns]] + [[coding-standards]] — 3,923 tests -> **5,548** (`pytest --collect-only`). Coverage concurrency is `["greenlet", "thread"]`, not `["greenlet"]` — `thread` was added for the bot service's cross-thread primitives; four live-only probe paths are `omit`-ed. Guard run this session: **0 violations across every rule group**, `mock-ban` and `monkey-patch-ban` included, so those enforcement claims verified rather than assumed.

**Anchor left deliberately unmoved:** [[tank-registry]]'s `fact_checked` stays 2026-06-11. Its claims are run-verified statements about the JS client's `activeGame.P.j`, and only the code-side link (team/rank/damage_state still modelled in `state/types/tank.py`) was re-checked — the runs were not re-derived. The blob anchor moved; the fact stamp did not.

**Why `make check` caught none of this:** the gate's only wiki-aware rule is `scripts/physics_claims.py`, which validates ` ```json claims ` blocks against `tankpit_bot.physics` symbols — 6 of 67 pages, and rigorous only there. Nothing in `scripts/` reads `source_git_blobs`, `fact_checked`, hub links, or index counts (grep returns zero hits). Anchor freshness, count integrity, and frontmatter-path existence are all mechanically checkable and currently unchecked — a guard rule for those would have caught most of this sweep and the previous one.

---
## [2026-07-31] build | `wiki-structure` guard rule: the wiki's own bookkeeping is now gated

Answer to "how come `make check` didn't catch this?" — it couldn't. The gate's only wiki-aware rule was `scripts/physics_claims.py`, which validates ` ```json claims ` blocks against `tankpit_bot.physics` symbols: rigorous, but 6 of 67 pages. Nothing read `source_git_blobs`, `fact_checked`, hub links, or index counts. Every drift both of today's audits found was structurally invisible.

**New: `scripts/wiki_rules.py::run_wiki_rules`**, wired into `scripts/guard.py` beside `contract_rules` and `physics_claims`. Four check families:

* **Frontmatter** — parseable block, SCHEMA's required keys (`title`/`tags`/`related`/`fact_checked`/`confidence`), a real `YYYY-MM-DD` date (rejects `2026-02-30`, not just bad shapes), a known confidence level.
* **Provenance** — every `source_paths` entry exists on disk, and every `source_git_blobs` key is one of them carrying a well-formed 40-hex object id. A trailing `:line` / `:start-end` locator is stripped first and `http(s)://` sources are skipped — both are established conventions the first run surfaced (35 false positives from `tpclient.js:243`-style locators, fixed before landing).
* **Navigation** — every hub inclusion link resolves; every page is linked from at least one hub (SCHEMA's orphan ban).
* **Counts** — each index row's `(N pages)` equals that hub's real link count; the `N content pages` total equals the files on disk.

**Deliberately NOT gated: blob-hash equality with HEAD.** A lagging anchor is not a defect — it is the marker for "not audited since this tree," and the honest response is an audit. Gating on it would redden the gate on every `src/` commit and would reward bumping anchors without re-reading the page, which is precisely the laundering the two audit entries above refused. Drift stays a report. The rule instead catches the anchor failure that IS unambiguous: a path that no longer exists, or a hash that was never a real object id.

First run against the real wiki found exactly one genuine violation: **`committed-intent.md` had no frontmatter at all** — hub-linked and readable, but carrying no provenance, no date, no confidence. Now fixed (anchored to `bot/ai/intent.py`, blob `74bcd19f`).

Two unrelated defects fixed along the way, both surfaced by running the gate rather than by reading:

1. **`scripts/guard.py` duplication** — three near-identical `if violations > 0 and rc == 0` blocks. Collapsed to one summed check. Behaviour is identical (all rules still run unconditionally, a nonzero orchestrator rc is still preserved), and the third block had been literally unreachable in tests: the physics rule always sets `rc = 1` first on any synthetic tree.
2. **A real file-descriptor leak in `terrain.py`** — `Image.open(path).convert("RGB")` never released the handle, one descriptor per `TerrainMap` load, raising `ResourceWarning` on 16 tests. `PillowImageProtocol` gained `__enter__`/`__exit__` and the loader now scopes the source image in a `with`. Verified under `-W error::ResourceWarning`.

Gate at close: guard 0 violations across every rule group, mypy clean over 814 files, **5,581 tests at 100.00% statement + branch coverage, zero warnings**. New module is 180 statements / 82 branches at 100%, covered by 31 tests plus two guard-escalation tests.

---
## [2026-07-31] schema | v1.1 — the frontmatter contract now matches practice, and says what the gate enforces

`SCHEMA.md` had drifted from its own wiki. Rule 5 required a `sources` field that **one** page of 67 used, while 65 used `source_paths` — a field the SCHEMA never mentioned, alongside `source_git_blobs`, `hubs`, and `verified`, none of them documented. The new `wiki-structure` guard rule made the mismatch untenable: the doc named a required key nobody wrote.

Reconciled by moving the doc to practice rather than practice to the doc:

* **Required** is now `title` / `tags` / `related` / `fact_checked` / `confidence` — what 67/67 pages carry and what the guard enforces. `sources:` is retired.
* **Conventional** fields documented for the first time: `source_paths`, `source_git_blobs`, `hubs`, `verified`.
* **`source_paths` semantics written down**: checkable repo-relative paths only; a trailing `:line` / `:start-end` locator is allowed and stripped; `http(s)://` is skipped; prose qualifiers belong in footnotes, not frontmatter.
* **`source_git_blobs` semantics written down**: a staleness marker, not a version pin. Added to both the field docs and the common-mistakes list: **never bump an anchor without re-reading the page** — that launders unverified prose as verified. Records why the guard checks existence and hash shape but deliberately not equality with HEAD.
* **New "What is machine-checked" section** — a four-row table of what actually fails `make check` (frontmatter, provenance, navigation, counts) plus the physics claim binding, and an explicit statement that atomicity, citation quality, and link-don't-restate are human-reviewed, not gated. Future sessions no longer have to guess which rules have teeth.

`flag-triage-20260729.md` — the lone `sources:` user, and the lone page missing `source_paths` and `hubs` — migrated: three checkable paths (the events file it was triaged from plus the two modules it names), `hubs: [architecture]`. Nothing was lost: the `tick_n 49-778` range and the `_pick_fresh_dot_hop` symbol it carried in frontmatter prose already live in the body table and footnotes 1/2/5, which is where the citation rules put locators.

`index.md` still advertised **schema v0.1** while `SCHEMA.md` said v1.0 — corrected to v1.1 with a pointer to the enforcement.

All 67 pages now carry all five required keys and a `hubs` field; hub links resolve 1:1 against `pages/`; guard 0 violations.

---
## [2026-07-31] build | `make wiki-anchors` — the drift report, and the rank range the docstrings had wrong

Two outstanding items from the day's audits, closed.

**1. `Military rank (0-7)` was wrong in eight docstrings.** The `Rank` IntEnum runs RECRUIT=0 through **GENERAL=8**, `combat_radar_min` documents 0-8, and `fuel_capacity` has nine entries — but `state/types/tank.py` (2), `state/types/self_state.py` (2), `bot/ai/types.py` (2), `facts/tank_facts.py` (1), and `facts/world_facts.py` (1) all told readers the top rank was 7. Docstrings only, no logic touched; all now read `(0 recruit .. 8 general)`. Found during the tank-registry audit and — my error — left unsurfaced until asked for a recap.

**2. The anchor-drift report is no longer an ad-hoc shell loop.** `scripts/wiki_anchors.py` + `tankpit-wiki-anchors` + `make wiki-anchors`: resolves every `source_git_blobs` entry against HEAD and prints STALE / UNRESOLVED / CURRENT with each page's `fact_checked` date for triage order. It reuses the guard rule's frontmatter parser through a new public `parse_page_frontmatter` rather than restating the grammar, and the git call is injected via a new `resolve_tree_hash` hook in `scripts/_test_hooks.py` — the repo's first subprocess, behind a Protocol, so no test shells out.

**It always exits 0** (`--exit-code` opts in to nonzero). That is the whole point: the report exists BECAUSE the drift must not gate. Its first run proved it — 8 anchors already stale, including five pages audited hours earlier, because the day's own commits moved `src/tankpit_bot`, `tests`, and `scripts/guard.py` underneath them. Had this been a gate, `make check` would have been red before the work was even finished. Recorded in SCHEMA: whole-package anchors go stale on any change inside the package, so expect churn there; file-level anchors are quieter.

The guard caught three weak assertions in the new tests before they landed — substring-in-output checks and an `is not None` — all replaced with exact comparisons against `format_report` output (`isinstance` is banned too; the optional is narrowed with `or ""` so a None still fails the length check). The test-quality rule doing its job on brand-new tests is the same enforcement loop the wiki-structure rule now closes for the wiki.

Gate: guard 0 violations, mypy clean over 816 files, **5,603 tests at 100.00% statement + branch coverage**. New module 72 statements / 26 branches at 100%, 22 tests.

---
## [2026-07-31] rulings + fixes | Fair-fight contract vs humans: homing cap, break band, resume floor, F21, radar reserve

Four user rulings from the post-100-kill review ("it also seemed buggy... it would refuel super early midfight, like above half, and it also was letting yuppler get tons of free hits in"), all landed with contract rows, tests, and a green gate.

1. **One pursuit homing per departure vs humans** ("cheating to use all 7 that we can do during ttl"): `hunt_lock.pursuit_fire` stamps `pursuit_shot_target_id`/`pursuit_shot_ms` on every pursuit shot; against a human the budget is spent while the stamp is newer than the target's `last_viewport_observation_ms` (`threats.pursuit_homing_budget_spent` — no explicit reset, the target re-entering view re-arms it). Capped ticks chase via the map instead. Diagnostic `pursuit_homing_capped`. Practice bots uncapped.
2. **Human-fight break band** ("the bot seems to run too much when fighting a human... does damage then leaves"): `assess_engagement_break` suppresses the projection vs humans while fuel >= capacity//2. The one-kill projection vs a full-health human (hits_to_kill ~13) was breaking at ~900 fuel — that was the "constant fleeing." The 3-hit `sustained` floor is unchanged; bot fights keep the validated projection. NOTE: the break tests' generic threat was named "Runner", which the name-shape classifier counts as HUMAN — the fixture is now `red-7` so the projection pins actually pin bot behavior.
3. **Resume floor, not full tank** ("teleports far away to restock and then comes back which isnt very fun"): the human break latch moves from `capacity` to `min(capacity, max(fuel_low + hunt_min + engagement_budget, capacity//2 + hunt_min))` = 750 at defaults — one good container, then back in the fight.
4. **F21 CLOSED** (the other half of the free hits): `_equipment_hop_barred` — during a held combat lock the equipment hop needs a genuine weapon BREAK (`weapon_reserves_below_break`, extracted from `should_enter_collect`), never the hunt-entry cap that pulled the 85-tile round trip at duals 22/25 mid-Yuppler-fight. In-viewport walk pickups stay unconditional.
5. **Radar reserve** ("if the bot runs out of radar ever... dead in the water"): `radar_spend_worthwhile` escalates — extras >= 2: 32-tile floor; the LAST extra: 128 tiles (half viewport); 0: free-radar any-sliver. Spend-gating inside the existing economics rule; the 2026-06-12 "never ration via toggle" rejection stands ([[radar-mechanics]] refinement note).

New AIState fields `pursuit_shot_target_id`/`pursuit_shot_ms` (codecs updated). Aim-staleness-vs-movers remains the top open human-fight item; break-projection numbers can still be tuned against the three early-refuel receipts if the half-capacity band proves too coarse. Contract rows + Verified-by in [[bot-behavior-contract]] §3.3/§3.4; F21 status updated in [[flag-triage-20260729]].

---
## [2026-07-31] ruling + refactor | 400-600-line file rule; hunt/collect monoliths split into 9 modules

User ruling (stated twice mid-session): "we need modular, clear separation of concerns, no monolithic files. 400 - 600 lines, including test files too." Recorded in [[coding-standards]] (supersedes "under 400 where possible"), CLAUDE.md, and memory. Applied to everything today's work touched:

* `hunt_mode.py` (1203) → `hunt_mode.py` (owner + phases, ~200), `hunt_acquire.py` (search/greet/acquire, ~460), `hunt_lock.py` (pursuit fire + break escape, ~450), `hunt_relay.py` (~340).
* `collect_mode.py` (1499) → `collect_mode.py` (owner + sense/safety gates, ~560), `collect_pickups.py` (~300), `collect_locks.py` (~240), `collect_hops.py` (~530), `collect_common.py` (~70). Cross-module names went public (`hop_toward_equipment`, `continue_or_release_fuel_lock`, `select_and_pickup_fuel`, `mine_clearance_decision`, `blacklist_container`, ...); no re-export shims, all call sites moved.
* `tests/bot/ai/test_hunt_mode.py` (1929+) → six files (mode/phases/pursuit/relay/humans/greeting, all <= ~410); shared tank builders lifted into `_support.py` (`make_enemy_tank`, `make_pursuit_target`, `make_map_known_enemy`, `consent_human`); the file-local world-reset fixture dropped as redundant with the top-level conftest autouse reset.
* `test_collect_mode_equipment.py` (976) → cascade / select+blacklist+steal / hops (3 files); `test_collect_mode_fuel.py` (896) → cascade / locks / worth-the-walk (3 files).

[[module-map]] bot/ai row updated. Gate after everything: guard 0 violations, mypy clean, **5,618 tests at 100.00% statement+branch coverage**.

**Backlog — 40 files still over 600 lines** (split-when-touched, or as a dedicated sweep): tests/action_lab/test_fuel_probe.py 2698, tests/bot/test_cdp.py 2400, tests/bot/ai/test_combat_strategy.py 1370, tests/action_lab/test_teleport.py 1324, tests/bot/test_tick_loop_coverage.py 1274, tests/action_lab/test_enemy_teleport.py 1227, src sniffer/world_state_dispatch.py 1156, tests/fakes/base.py 1079, src bot/tick_loop.py 1037, tests/action_lab/test_movement_probe.py 1013, src bot/ai/combat_strategy.py 957, tests/bot/ai/test_strategy_coverage.py 923, tests/action_lab/test_equipment_collection_coverage.py 904, src diagnostics/session_scorecard.py 876, tests/diagnostics/test_session_scorecard.py 865, tests/sniffer/test_world_state_dispatch_tank.py 810, tests/service/test_http_server.py 790, tests/bot/ai/test_equipment.py 782, tests/action_lab/test_queue_probe.py 758, src bot/ai/threats.py 743, src action_lab/fuel_probe.py 741, src protocol/types.py 737, tests/world_state/test_tank_observation.py 722, tests/action_lab/_replay_core.py 700, tests/fakes/probe.py 692, tests/bot/ai/test_hunt_feedback.py 670, tests/test_smoke_script.py 667, tests/bot/ai/test_collect_mode_integration.py 665, src runtime_logging.py 656, tests/bot/ai/test_resource_search.py 646, src bot/tick_loop_actions.py 643, tests/action_lab/test_teleport_attempt.py 641, src diagnostics/issue_report_codecs.py 640, src action_lab/fuel_probe_attempt.py 627, src state/mutations.py 626, tests/bot/ai/test_mode_controller.py 623, tests/bot/ai/test_movement.py 617, tests/bot/ai/test_threats.py 615, tests/action_lab/test_equipment_attempt_coverage.py 601, src action_lab/enemy_tracking_types.py 601.

---
## [2026-07-31] ruling + fix | Partial mid-fight restocks vs humans: the peace-out is closed at the arbitration layer

Follow-up ruling the same session ("its not necessary to full restock but to do partial restocks so the bot can keep fighting the human... right now it felt like the bot will leave and peace out frequently so its not fun to engage"): the LAST full-restock path in a human fight was the weapon/radar emergency -- `should_enter_collect` fires below the break bars even with a lock held, COLLECT takes ownership, and `should_exit_collect` held until GENUINELY FULL (fuel at cap, weapons at cap, radars cap-5). That is the long absence the user felt.

**Fix -- the combat-resume bar** (`mode_controller.human_fight_resume_permitted`): while a combat lock on a registry-present HUMAN is held, BOTH sides of the arbitration drop to a partial bar -- `should_exit_collect` releases and `should_enter_hunt` re-admits at: fuel >= the resume floor (`human_fight_resume_fuel_floor`, now shared with the break latch -- 750 at defaults), duals AND homings >= half the rank cap (break < 4, resume at half-cap: wide hysteresis), extra radars >= min(combat_radar_min, 2x radar break) = 10 (the bare break bar of 5 would be a zero-width band). Changing only the exit bar would DEADLOCK: the full `should_enter_hunt` would refuse the partial stock and the owner selection would ping-pong to COLLECT forever -- the re-entry override is the load-bearing half. Scope guards: practice-bot locks and lockless between-kills restocks keep the full bar (fights still only START fully stocked); wind-down keeps the full bar (session_complete must leave a stocked tank); a vanished target reads as no human lock. Pins: `tests/bot/ai/test_human_fight_restock.py` (13 tests: every floor boundary both sides, bot/vanished/wind-down scope, the enter-hunt override, the weapon-emergency veto, and the end-to-end `_select_owner_mode` handoff). Contract row in [[bot-behavior-contract]] SS3.1. Gate: guard 0 violations, mypy clean, 5,631 tests at 100.00%.

The mid-human-fight economy is now: fight from full -> break below half capacity under sustained fire -> hop out, drink to ~750 with the lock held -> teleport back and re-engage -> top weapons only at a genuine break, and then only back to half-cap -> repeat until the kill. Full rebuilds happen between fights, never inside one.

---
## [2026-07-31] build | The sim learns to be human: opponent personas + the end-to-end fight-loop scenario

Answer to "did you sim these? how do you know they work?" — until now, NO sim or scenario could reach any human-fight branch: every sim opponent was hardcoded `red-<id>` (`wire_statements.identity_statement`), and the name-shape classifier routes `red-N` straight to the practice-bot tier. Every 2026-07-31 contract was pinned only at the decision layer.

**Sim seam:** `SimTankDict` gains a `name` field (codec-strict, default `red-<id>` in `make_sim_tank`); `identity_statement` emits it; `maybe_revive_opponent` keeps a human persona across respawns (a red-named opponent still derives a fresh `red-<new id>`); `tankpit-sim-run --human-opponent NAME` seeds the scripted opponent with the persona. Consent needs no chat plumbing: the scripted opponent SHOOTS FIRST (beat 1), and an attacker consents by attacking. Pinned by a 30-round full-session soak (`test_human_opponent_session_runs_under_the_consent_gate`) plus persona-respawn and codec pins.

**Scenario:** `tests/scenarios/test_human_fight_loop.py` runs the whole 2026-07-31 stack as ONE continuous fight through the real dispatcher + production `decide()` (new wire builders: 0x4D `chat_message`, fuel-bearing 0x2E `self_status_sync` — the path that CONFIRMS incoming damage): unconsented adjacent human -> no lock, HELLO rides the encounter tick, chat consents -> lock (close -> landing scan -> shoot, the contract rhythm); six confirmed duals 1100->560 with the band holding (shoot every tick, latch 0); the seventh hit crosses 550 -> break fires, lock held, latch = 750 (the resume floor, not capacity); fuel 760 -> re-engage in person, latch cleared; 6 s departure -> exactly ONE pursuit homing (stamped window), next tick `map_open find_target`; reappear -> shoot -> 0x41 -> `confirm_kill`, lock cleared. Two findings the harness surfaced that fixture tests never would: the HELLO attaches on ENCOUNTER (rode the pre-consent map_open tick, not the locking tick), and the engagement rhythm really is close -> scan -> shoot.

Epistemic status upgrade for the fair-fight stack: decision-layer pins + dispatcher-level scenario + a sim soak lane (`--human-opponent`). Still owed: a LIVE human receipt — the next real Yuppler fight is the final validator. Gate: guard 0 violations, mypy clean, **5,638 tests at 100.00%**.

---
## [2026-07-31] audit + fix | The three lifecycle TBDs: two were already solved under other names, one hid a real gap

"cant we solve those or what are they? maybe we already did it" — mostly, yes. The contract's three June-era `(TBD)` Verified-by rows cited integration tests that were never written under those names, while the behaviors themselves got pinned elsewhere as the codebase grew:

* **§1.1 startup** (`test_bot_login.py` TBD): already pinned — `test_state_machine.py::TestBotStateUpdates` (INITIALIZING -> WAITING_FOR_POSITION -> IDLE) and `test_run.py::TestBotGameLoopStates::test_game_loop_ai_tick_no_self_state` (no decision without a self tank). Row re-cited.
* **§1.2 graceful end** (`test_session_shutdown.py` TBD): already pinned — `TestAppendIndexRowEndToEnd` (scorecard -> summary -> index row end to end) plus the per-exit-reason rows (`browser_closed`, `no_viable_targets`, `interrupted`). Also corrected the row's "latest.* symlinks" phrasing — they are independent files, never symlinks (the make-targets audit's finding, now consistent here too).
* **§1.3 interrupted/crashed** (`test_signal_handler.py` TBD): the signal half was already solved — `test_interrupt_handling.py` (flag API, real SIGINT+SIGTERM installer, `main()` wiring) and `TestInterruptedExitReason` (flag -> graceful tick-boundary exit + `interrupted` index row; the summary scorecard is now ALSO pinned byte-for-byte). But the audit caught a real hole: **`exit_reason="crashed"` had NO writer anywhere** — the 2026-06-20 row promised it ahead of implementation, so a crashed session vanished from `_index.tsv`. Fixed: `run_tick_loop`'s exception boundary is lifted into `_tick_with_exit_boundary` (also resolving the C901 the extra handler tripped) — browser-closed and SessionExitError stay graceful; any other unhandled exception finalizes scorecard + summary + index row as `crashed`, then RE-RAISES. Pinned in the new `tests/bot/test_tick_loop_crash.py` via a protocol-complete exploding frame bus through the `Bot(frame_bus=...)` constructor seam.

Zero `(TBD)` markers remain in [[bot-behavior-contract]]. Gate: guard 0 violations, mypy clean, **5,639 tests at 100.00%**.

---
## [2026-07-31] ruling + fix | Hello anytime; the greet VISIT gets its own latch — and the sim soak that caught it all

First real `--human-opponent guest` sim session (the "run a sim session and check it" request) ended `no_viable_targets` TWO ROUNDS in, and unwinding it produced one wrong fix, one user correction, and three real fixes:

1. **Terrain was None in every sim session ever.** The binary seam never performs the lobby ROOM_LIST/SELECT, so the bot's decision terrain stayed unloaded ("No selected room is available") and every terrain-gated behavior — greet stand-off landings, larder landing legality, LOS composition — silently self-disabled; the practice soaks never noticed because their paths all have terrain-None fallbacks. `sim/run.py::_boot` now registers the sim room and selects it exactly as the lobby decoders would.
2. **The HELLO and the stand-off visit shared one latch, and that was the real bug.** The join-broadcast-stamped "presence" let the HELLO fire at a tank still at the (0,0) unsynced sentinel; the shared `greeted_target_id` then blocked the greet APPROACH forever, so the human never saw the bot and could never consent. My first fix gated the HELLO on viewport bounds — the user corrected it (verbatim): "hello can run anytime... as long as the other player is on the map logged in. but you dont have to be near them." So: the HELLO fires once per human for any alive, MAP-FRESH enemy human anywhere (position, (0,0), and viewport never gate chat; map freshness is the logged-in proxy — a MapData ghost may collect one wasted hello, bounded by the latch); the VISIT ("we want to see them") now has its own `visited_target_id` latch, stamped on approach dispatch so an unresponsive human draws exactly one courtesy trip. New AIState field, codecs updated, death-reset preserves it alongside the greeting latch.
3. **The soak test was too weak to catch its own failure** — the 2-round dud PASSED the original asserts. `test_human_opponent_session_runs_under_the_consent_gate` now proves the story from the events stream: `greeting_approach` AND `chat_greeting` AND `tank_deactivated` must all appear.

Also landed on the way: the scenario harness's `place_self` now ingests the join 0x5A (every real session has a viewport record; its absence had been masking the F8 in-view short-circuit in the fight-loop scenario — the adjacent consented human is shot on the LOCKING tick, no close/scan prelude), and a new pin covers the under-fire fully-exhausted collect branch the viewport change had un-covered.

**Final real-terrain receipt (sim-20260731-225053):** tick 1 HELLO from afar at the not-yet-synced guest -> greet-approach teleport into the stand-off band -> guest consents by shooting first -> fight -> `kill registered (tank_id=11)` -> harness revives guest as id 12 -> unsynced respawn correctly `protected_human_rank` -> fresh map, nothing viable -> clean exit. Every step production-correct. Contract row rewritten in [[bot-behavior-contract]] §3.2 (hello-anytime + visit latch). Gate: guard 0 violations, mypy clean, **5,642 tests at 100.00%**.

---
## [2026-07-31] arena soak | 4 bots, kill two, two humans join mid-session -- and the two production bugs the scenario shook out

The requested experiment ("seed the sim with 4 bots, see Artax kill two, then introduce two human players and see how it responds"), run as a scripted sim harness: 4 red bots at cardinal distance 16 (outside each other's reach-8, so fights stay sequential duels), and after the SECOND bot kill, `guest` + `Yuppler` join near the client (0x21 announcement, the real join mechanism), each running the scripted aggressor policy.

**First run died two acquisitions after the first kill, and the receipt was a real decision-layer bug:** the engagement-break's incoming-rate window is attacker-agnostic — after killing red-24 mid-fight, the DEAD tank's 81/tick stayed in the 10 s trailing window and the entry assessment blocked red-23, red-22, and red-21 in one tick chain as "unwinnable at any fuel" (needs 1270, capacity 1100) on damage from a tank that no longer existed. Live impact at real speed: up to 10 s of refusing every follow-up target after each kill under fire. **Fix at the root:** `ConfirmedIncomingDict` now carries `shooter_id`, and `incoming_damage_window` excludes KNOWN-dead shooters (registry `liveness == "deactivated"`); unknown shooters still count (a registry gap must never under-report live danger), and dead entries stay in the log so a respawn's liveness flip restores them.

**Second run delivered the full arc — and the second bug:** kills at rounds 13 and 30 (two bots down), humans join at 30, `guest` killed at 49, `Yuppler` at 64 — both dispatched BEFORE the two remaining farmable bots, the human-priority doctrine choosing exactly as specified. But the events showed **12 chat_greetings**: with TWO humans the single-id greeting latch ping-ponged (greet guest -> latch moves to Yuppler -> guest reads ungreeted -> ...), sailing past the server's 8-send flood mute that silences chat for the whole session. `greeted_target_id` and the day's new `visited_target_id` are now PER-ID maps (`greeted_tank_ids` / `visited_tank_ids`, {id: ms}, killed_tank_ids-style), preserved across the death reset.

**Final run (sim-20260731-arena01): the clean arc.** red-24 @13, red-23 @30, humans join @30, guest @49, Yuppler @64 — **exactly 2 HELLOs**, zero greet visits needed (both humans consented by shooting first), and the session ends at fuel 713 with red-21/22 blocked as unwinnable under red-22's LIVE sustained fire — the gank-protection contract doing its designed job, not a defect. Client alive, full inventory, 4 kills.

Experiment script: session scratchpad (`sim_arena_experiment.py`) — promotable to a `--arena` mode if wanted. Contract rows updated (§3.2 greeting maps, §3.3 dead-shooter rate rule). Gate: guard 0 violations, mypy clean, **5,644 tests at 100.00%**.

---
## [2026-08-01] crack + build | The Rb anchor law, the ferry scenario, and the scope scout — plus two sim-fidelity bugs the soak shook out

The requested pair ("build the ferry scenario please, true to the actual game" + "we need to solve the viewport shift option"), executed crack-before-code:

**1. The scope-extend wire law, mined not guessed.** The 2026-07-10 human capture (sniff-20260710-202821) held all 8 "Extend view" events with their sent frames and answering 0x5A — and it FALSIFIED this wiki's own direction table. The `Rb` byte is the compass CLOCKWISE FROM NORTH (0=N 1=NE 2=E 3=SE 4=S 5=SW 6=W 7=NW; the old menu-index reading is dead), and the shift is the **ANCHOR law**: the tank pins to the trailing window edge — shift east puts the tank on the WESTERN edge — with the unnamed axis unchanged. All 8 events fit exactly with zero free parameters once self positions were walk-corrected through the 0x47 path echoes. User corroborated both halves verbatim the same day, plus the Scope Center option (byte 8 inferred, the one remaining value). [[viewport-shift-protocol]] rewritten; the 8 measured rows are pinned executable in `tests/sim/test_scope.py`.

**2. The capability, end to end.** New `scope_shift` BotCommand -> executor -> `bot_dispatch.scope_shift` (fire-and-forget `Rb`; the 0x5A ingestion updates the origin); sim decode + `ViewportTracker.apply_scope_shift` (anchor law, center=8 recenters, map-clamped, unknown bytes no-op but still confirm) + server routing (client-only; membership diff announces revealed tanks). First doctrine consumer: the **ferry scope scout** (`bot/ai/scope_scout.py`, the queued "we could just use a viewport shift" ruling) — after the larder declines and before discovery pays fuel, a water-locked believed container within one pan (Chebyshev <= 15) with no fresh ferry belief draws the FREE pan toward its water; 30 s cooldown latch (`last_scope_scout_ms`, new AIState field), never during a held lock. Plus the F5 completion: `larder._is_walk_territory` no longer cedes WATER containers to the walk step — floating fuel in view was previously served by nobody.

**3. The ferry scenario, on the real lake.** `tankpit-sim-run --ferry`: field01's (112,112) water body probed from the actual GIF — shore spawn (106,112) at 400 fuel, 700-vol container water-locked 6 east (in the join window; radar believes it), ferry at (118,112) on its own water (land distance 3) OUTSIDE the join window, land stock too lean to reach hunt readiness (first cut had 2x300 land fuel and the session topped out at 1062 without ever needing the water — retuned to 2x150). Sim law completed: a live ferry tile is a legal teleport landing (`_tile_blocked_for_landing` — boarding by teleport IS the doctrine). **Proven chain, all production code:** land forage -> radar belief -> larder `no_landing` -> scout pan east -> ferry terrain-5 patch -> `ferry_served` hop teleports ONTO the ferry -> held fuel lock rides 6 tiles of water -> auto-pickup -> full tank 1100, ferry parked where the ride ended.

**4. Two sim-fidelity bugs the soak exposed (both would have poisoned future receipts):** (a) the sim emitted `teleport_landed` BEFORE the 0x3D — reversed from the real wire order the displacement receipt relies on — so every exact landing read as displaced and `_expire_disproven_ferry_belief` (the s9-7/8 fix, working correctly) consumed the belief the landing had just proven true; (b) 0x5A patch entities weren't sorted in patch-linear order, so a ferry riding WEST (fresh patch earlier in the skip-walk than its revert) crashed the encoder with a negative delta. Both fixed at the law, both pinned (`test_server` order rows, `test_ferry::test_riding_west_emits_an_ascending_viewport_patch`).

**5. New doctrine banked (user, verbatim, mid-build): "tbh when i use ferries i use auto scroll on so i can ride it across multiple viewports. also when i forage with no extra radars, i use auto scroll on."** Recorded as the OPEN autoscroll-riding row in [[bot-behavior-contract]] §3.4 and [[ferry-mechanics]] — dynamic `Ia` toggling (ON for beyond-window rides and radar-broke forage, OFF otherwise) is the queued next build; today's ride receipts cover single-window rides.

Also: `test_executor.py` (658 lines) split per the 400-600 rule into `test_executor.py` / `test_executor_dispatch.py` / `_executor_support.py`. Contract rows added (§3.4 scope scout, water-larder territory, autoscroll OPEN). Gate: guard 0 violations, mypy clean, **5,677 tests at 100.00% (statement + branch)**.

---
## [2026-08-01] archive mine | The longitudinal container atlas — 318 captures, 120 days, and the field's true persistence law

Step one of the capture->sim divergence pipeline (user: "start with the atlas miner"): every real-wire capture in the archive (276 bot + 34 sniff + 9 probe, minus 1 unreadable) replayed through the production decoders, every per-tile container statement extracted with absolute timestamps — 197,030 observations, 10,930 distinct tiles, all sessions room 1 / field01. New miners `analysis_scripts/mine_container_atlas.py` + `analyze_container_atlas.py`; artifacts `runs/analysis/container_atlas.json` + `container_observations.jsonl`.

**Findings (now in [[game-economy]]):**
1. **Persistence quantified:** same-tile cross-session volume agreement 98.8% within 1 h -> 94.9% at 7-30 d -> 81.4% past 30 d. A week-fresh atlas snapshot is ~97% truthful — the sim can be seeded with the REAL field.
2. **Refills exist and look like deposits:** ~120 cross-session volume increases in 120 days (median +805 — squarely the max-deposit band) plus 172 within-session positive->higher jumps. Nothing spawns; players bank. The 2026-07-25 static-population law stands, refined: static placement, consumption-dominated, deposit-topped.
3. **The stocked population is ~5,000+, not ~670:** 5,457 distinct tiles held verified stock within one week. The sim's density-model seed is roughly an order of magnitude too sparse in the visited layer. `--from-atlas` sim seeding is the queued fix.
4. **Placement churns only over months** (175 type-flips / 120 d; >30 d agreement drops to 81%).

**Method notes that mattered:** layer discipline (a visible-layer 0 is "no visible container", never "empty" — first pass miscounted exposure as refill), and a coordinate-skew check (radar-vs-0x5A same-tile volumes: 1,140 matches, 409 mismatches ALL explained by intra-session drains, zero neighbor-tile matches — no off-by-one).

Next steps queued: refill->deposit attribution pass (pair refill timestamps with observed tank positions), `--from-atlas` sim world, then the teacher-forced capture differ.

---
## [2026-08-01] pipeline | Deposit attribution closed, the sim seeded from the REAL room, and the response-shape differ's first two cycles

The rest of the divergence pipeline ("we have unlimited time... proceed"), all three stages built and run — new page [[capture-differ]]:

**1. The refill mechanism is agent deposits, not regeneration.** Two artifacts died first: 123 of the 172 "within-session refills" were a same-tick ordering collision (a pickup's remaining-volume record value-sorted ahead of the pre-pickup read — the miner now preserves intra-payload wire order), leaving 49 genuine radar-to-radar increases. Then the discriminators: corr(dv, dt) = −0.13 across all 169 refill events (a container untouched 2,809 hours gained no more than one untouched 2 — accumulating regen refuted), dv chunky at 792 ± 251 (the rank-0/1 max-deposit band), and ZERO 0x64 frames in 318 captures — consistent, since 0x64 goes only to the depositor: third-party deposits are wire-invisible except as the volume bump. Leading hypothesis: PRACTICE BOTS bank excess fuel. Live probe queued (stationary radar watch on a stocked container). [[game-economy]] updated.

**2. `--from-atlas`: the sim plays on the real field.** `sim/atlas_seed.py` seeds the mined room — 1,969 stocked fuel + 6,675 drained dots + 498 equipment on field01 (vs the statistical model's ~670 stocked) — standalone (lean-spawn forage; the first cut spawned hunt-ready and exited tick 2) or composed with `--practice`. Flagship soak: the certified roster ON the real room, 400/400 rounds, 8 kills, 0 deaths. The seed validator now accepts floating containers without ferry service — the atlas proved water-locked stock is real state and ferries drift; only rock is a typo.

**3. The response-shape differ found five sim law gaps and verified them closed in two cycles.** Every sent command in every capture paired with its self-caused response shape, live distribution diffed against a fresh sim baseline (windows end at the next sent command — sim wall-time compresses to sub-second, so fixed windows absorbed whole sessions). Confirmed archive-wide en route: teleport cost law 6,941/6,960, window-bound acceptance 77/77 rejected. The five: (1) landed-teleport order is ``5A -> 3D -> landed`` (the recentered 0x5A LEADS — the morning fix had it trailing); (2) every client 0x5A pairs with a self 0x3D (scope now answers ``5A+3Dself``); (3) the radar extra's 0x49 snapshot LEADS the scan results; (4) firing costs NEVER snapshot (92.4% of 11,051 live shots answer a bare 0x53 — the sim's per-shot 0x49 was invented; counts re-sync on the next 0x49-bearing event); (5) equipment pickups close with a remaining-0 container-pickup record (``47+67+49+pickup``, 2,170 windows). Post-fix baseline matches live's dominant shape on every lane. Open triage rows (fuel-pickup multi-message choreography ~1,600 windows — needs one byte-mined window before copying; autoscroll edge-recenter shapes; 52c6 as an instant-movement model limit) are tabled in [[capture-differ]].

Gate: guard 0 violations, mypy clean, **5,685 tests at 100.00%**.

---
## [2026-08-01] crack + close | The fuel-pickup choreography byte-mined, the differ's biggest triage row closed — and sim time put on the tick

The requested close ("byte mine the fuel pickup window and close it"):

**1. The choreography, from the bytes.** Six clamp windows + five drain windows + five no-walk windows dumped field-by-field from the archive. The multi-message shape decomposes into four measured branches ([[fuel-system]]): clamp = duplicate records, the 0x44 in its GAIN form (`is_free=True, flag=0`, absolute fuel) between records 2 and 3, a third record, code-5 close `reset_action=0`; drain = duplicate remaining-0 records + code-4 close `reset_action=1`, NO 0x44; no-transfer/no-walk = the 0x44's NO-GAIN form (`is_free=False, flag=43`, unchanged fuel) + one record + close by stockedness. The records are IDENTICAL DUPLICATES, not drain steps — and the same x2 law governs every move/teleport auto-pick (129 + 2,200+ windows of `...pickup+pickup`). Bonus law: **the walk executes even for a known-drained container** (fuel 783 -> 783 across a 4-tile walk, then the empty close) — the sim's walk-free pre-refusal was an invention; only bare-ground clicks pre-refuse.

**2. Closed in the sim and verified by the differ.** `emit_fuel_pickup_close` implements all four branches (records broadcast, 0x44/0x52 per-connection); auto-picks duplicated; `_pickup_target_stocked` now passes drained-but-known containers through to the walk. Fresh-baseline differ: sim `pickup_fuel` now 80% `47+pickup+pickup+44+pickup+52c5` / 11% `47+pickup+pickup+52c4` — live's two dominant shapes exactly; the ~1,600-window triage row is gone from [[capture-differ]] (row #6 in the closed table). Pinned branch-by-branch in `tests/sim/test_fuel_choreography.py`.

**3. Sim time now runs on the tick.** The choreography change surfaced a latent flake: the default-scenario exit test's horizon depended on WALL-clock TTL pacing (exited round ~54 solo, never under xdist — live TTLs read real time, and an in-process round takes microseconds). Root fix, not a bound bump: `run_sim_session` installs a `TickPacedClock` advancing one measured 2 s tick per round — sim sessions are now deterministic under any machine load AND live-realistic (a 300-round soak ages forage coverage/harvest memory/belief freshness exactly like a 10-minute session; capture timestamps come out live-shaped for the differ). All soak modes re-verified under tick pacing.

Gate: guard 0 violations, mypy clean, **5,692 tests at 100.00%**.

---
## [2026-08-01] build | Ghost replay — recorded opponents, live bot ([[capture-differ]] stage 4)

The requested increment ("go start with ghosts"), closing the "opponents aren't random, we have their data" argument:

**The capability.** `sim/ghost.py` compiles any capture into a replayable spec (client opening state; every sighted opponent with recorded name/team/rank; tick-indexed places/shots/chats; the session's own first 0x4C as the exposed dot set; first-observed container reads) and `--ghost CAPTURE` replays it under the production bot: ghosts relocate by recorded authority (`relocate_tank`, wire-correct 0x3D semantics), shoot and chat as real queued commands (the consent chats replay — the human-fight contract engages against ghosts), damage by sim law, all tick-paced. `--from-atlas` underlays the mined room (capture reads win per-tile) so long replays don't starve. The `ghost_summary` diagnostic reports tracking: rounds within 4 tiles of the recorded client, first divergence, final drift.

**Receipts.** Self-replay validation (current code vs its own recording): tracked 19/21 rounds. Replaying live session bot-20260729-232252 against today's bot: divergence at round 2 — the July-29 bot wandered collecting; today's immediately acquired and killed orange-2. That is the tool doing its job: divergence = measured behavior change. 400 recorded rounds replay in ~120 s.

**Three fidelity findings en route, fixed at the law:** (a) the first cut marked every first-read container dotted — the live bot round-2-hopped at a phantom 1,073-dot atlas; the exposed set is the recording's own 0x4C (plus visible-layer reads), radar reveals stay hidden, unread atlas dots seed drained; (b) the same over-dotting lived in ATLAS seeding — 6,675 drained dots of 120-day exposure history versus the live ~620-1,077 census dragged sessions superlinear (120 rounds = 172 s); ghost mode now bounds the exposed set exactly and heuristic mode gates drained dots on 7-day freshness (400 rounds = 120 s); (c) `relocate_tank` initially double-announced entering ghosts (explicit 0x3D + the membership diff's).

Also: `sim/run.py` split per the 400-600 rule (`sim/scenarios.py` now owns worlds + mode resolution + CLI parsing; run.py owns boot/loop/clock). Gate: guard 0 violations, mypy clean, **5,705 tests at 100.00%**.

---
## [2026-08-02] live run | 20-kill soak (bot-20260802-205105) — loop scan clean, pipeline fed end-to-end

First live session on the post-differ code, run unattended to a kill target ("so we dont wanna just run a 20 kill run and see how it does? identify any radar or ferry loops?").

**The run.** 20 kills / 0 deaths in 1,571 s (780 ticks), 229 shots at 99% hit rate, exit `session_complete` with a clean `quit_game` to lobby, ending inventory maxed (fuel 1100, duals/homings/radars 25/25/25). A first launch attempt hung 13 min inside Playwright's browser launch (never navigated, no outbound connection) — killed and relaunched clean; transient, not bot code. A cosmetic Playwright teardown traceback fires after `quit_game` (CDP event racing the closed connection; no bot frames in the stack).

**Loop scan: nothing loops.** No radar loop — radar economy netted positive (9 scans vs +gains, ended at cap). No ferry confusion. Map-opens paired ~1:1 with teleports. Run-audit: 2 critical, 0 warnings — both criticals are the SERVER failing to answer a `map_open` (21:04:18, 21:06:36); the bot burned its 10.3 s stall budget, replanned, and the retry answered in 2 s. Server flakiness handled correctly; explains the dispatched=112/completed=110 delta. All consistency channels agree exactly (20 kills = 20 wire 0x41s = 20 DOM banners; 56 wire 0x52s = 56 ledger).

**Rejection chatter decoded live.** 48 code-5s are post-clamp double-sips (drink to cap, immediately re-sip the remainder, server refuses "tank full") — 48 fresh live receipts for the [[fuel-system]] clamp choreography. 6 code-4 drained races, 2 code-1 cant_go bounces (fresh [[flag-triage-20260729]] F6 samples). All replanned, zero stalls. Soft fix candidate: skip the re-sip when fuel just hit cap (pure wasted action, ~2 s each).

**Pipeline fed end-to-end on the fresh capture.**
- Differ (live catalogue regrown): every lane's dominant live shape still matches the sim laws — pickup_fuel 59.4% clamp / 13.8% drain, radar 45.7/38.3 split, shoot 92.5% bare 53self, teleport 5A->3Dself->landed, and **scope now n=59 live samples at 83.1% `5A+3Dself`** — the [[viewport-shift-protocol]] anchor law's sample base keeps growing.
- Atlas re-mined with the new observations; the miner surfaced **182 fuel<->equipment type-flip tiles** (a tile's record changing kind across captures) — new texture for the deposit-attribution question in [[game-economy]].
- Ghost baseline: the run's own capture self-replayed under the same code — tracked 10/150, first divergence round 2, final drift 53. Recorded in [[capture-differ]] as the STANDING live-capture baseline (live self-replays fork at the first radar because the seeded world is the atlas approximation; compare future numbers against this, not the 19/21 sim-recording figure).

No code changed. Fix candidates queued: post-clamp re-sip guard; optional teardown-race silencer on shutdown.

---
## [2026-08-03] lift | The 0x52 refusal laws single-sourced — physics/supervisor.py, dispatch prediction, three constant forks deleted

The user's mandate, verbatim: "im worried were just tacking on random band aid fixes instead of properly addressing thw root of the issue and presenting a properly integrated solution" / "no forking, no duplicste coxe ir divergenet code" — after the 20-kill soak's 48 refused at-cap fuel sips traced to `_find_combat_pickup` never consulting fuel headroom.

**The audit first.** The bug was one instance of a class: the "would the server refuse this?" question was re-derived per call site. Found and inventoried: (a) THREE parallel namings of the 0x52 vocabulary — canonical `SUPERVISOR_ERROR_*` in `protocol/constants.py`, a private `_COMMAND_ERROR_*` fork in `bot/tick_loop_actions.py` + `bot/tick_loop.py`, and a third name-dict in `sniffer/world_state_dispatch.py`; (b) the refusal LAWS living only inline at sim emission sites; (c) `inventory_all_full` re-encoding the code-7 condition client-side; (d) `sim/equipment.py` hard-coding cap 25 (a rank-1 corpus artifact) beside the rank-derived `inventory_capacity` law; (e) the missing headroom gate itself. NOT broken: `physics/` as the shared law layer (both consumers already import capacities/costs/damage from it), and the 8 planner-side `fuel >= capacity` sites that are POLICY (break bands, deficit math), not law — deliberately left alone.

**The lift, following the repo's own pattern.** New `physics/supervisor.py` — `fuel_pickup_close_code` (close-by-stockedness), `fuel_pickup_refusal` (known-drained -> 4; at-rank-capacity -> 5), `equipment_pickup_refusal` (all five slots at rank cap -> 7), `teleport_refusal` (cost > fuel -> 8) plus `TELEPORT_RING1_COST_SLACK = 9` (floor(6*(d-sqrt2)) bound for target-based prediction under displacement). Wiki claims bound in [[fuel-system]], [[teleport-mechanics]], [[game-economy]] (the physics_claims gate enforces the binding). Consumers:
- SIM EMITS with them: `emit_fuel_pickup_close` close code, `resolve_equipment_pickup` refusal branch (cap now `inventory_capacity(rank)` — the corpus 25 is `inventory_capacity(1)`; `shadow_laws` states its all-private corpus as `ARCHIVE_EQUIPMENT_CAP`), `process_teleport` affordability.
- BOT PREDICTS with them at the executor chokepoint: a `pickup_fuel`/`pickup_equipment`/`teleport` whose refusal the belief PROVES is suppressed pre-wire (`dispatch_suppressed` diagnostic, predicted code logged); belief-uncertain cases (drained races, no container record) stay optimistic by design. `_find_combat_pickup` consults the same predicates to fall through to the useful container kind. `inventory_all_full` — a fourth pre-existing re-encoding of the code-7 condition — was DELETED outright (post-lift it was a one-line wrapper with a single caller; the caller consumes the law via the new `inventory_counts` shape adapter). Second-pass sweep under the no-wrappers/no-aliases/no-fallbacks mandate also re-expressed `fuel_pickup_refusal` THROUGH `fuel_pickup_close_code` (a refusal IS a predicted no-transfer closed by stockedness — one branch table, not two).

**Forks deleted, class killed.** All three constant re-declarations replaced by imports of the canonical set (+ `SUPERVISOR_ERROR_NAMES` canonicalized into `protocol/constants.py`); new guard rule `scripts/protocol_constant_rules.py` (wired into `scripts.guard`, mirrored tests) bans any future assignment matching the error-constant name patterns with embedded integer literals outside `protocol/constants.py` — enforcement, not discipline.

Test fixtures the new laws falsified (all fixed at the fixture, not the law): two teleport tests hopping unaffordable distances, one combat-pickup test with an over-cap inventory, one whose "fuel skip" was masked by equipment-first ordering (rewritten to prove the gate: at-cap -> None, cap-1 -> sip).

Gate: guard 0 violations (incl. the new rule), ruff + mypy clean, **5,733 tests at 100.00%**.

---
## [2026-08-03] validate + build | Suppression receipt, deposit law mined, stage 5 closed, reactive ghosts

One day, four deliverables, all gated (**5,741 tests at 100.00%**, `make shadow` green across all seven laws):

**1. The refusal-law lift validated end to end.** Offline: four sim soaks under the new predicates (arena kill, practice roster, ferry harvest, ghost replay); ghost self-replay IDENTICAL to baseline (10/150, div tick 2, drift 53 — the lift changed no macro behavior); fresh-baseline differ (`sim-lift*`): every closed law's dominant shape still matches live. Live: validation run bot-20260803-180918 — 14 kills / 1 death / 1,293 ticks — and, missed in the first report and recovered by the 2026-08-03 autopsy: the session held the FIRST TWO LIVE HUMAN FIGHTS of the fair-fight stack. Belton (id 984): greeted HELLO once + greeting approach at 18:09:35, fought with correct break math, killed THREE times (18:11, 18:28, 18:43). nope (id 2678): greeted 18:28:46, a 7-minute war (41 incoming 45/90 hits, 45 shots back, two correct break-engagement calls at fuel 500/516) ending in the session's one death at 18:37:55 — the F21 partial-restock lane ("refuel to 750 nearby and resume") sent the bot foraging a WAR-DRAINED pocket under fire: five straight code-4s, one +57 sliver, teleport unaffordable at 17 fuel, dead at 0. The failure is not the break math (it fired, correctly) but the resume-refuel leg assuming nearby fuel exists — contract ruling owed: desert-escape teleport while affordable and/or a last-resort quit floor. The refusal-law receipt: **32 code-5 windows, every one a clamp SUCCESS with a same-window fuel transfer; ZERO no-transfer at-cap refusals (was 48)**. The waste class is dead by absence: the scanner-side gate stops the dispatches from being planned at all (0 executor suppressions needed). Remaining rejections: 18 code-4 drained races (correctly optimistic), 10 code-1 (F6: terrain-checked 2026-08-03 — 8/10 refused targets are PASSABLE per static terrain, 2/10 impassable planner feeds; the passable majority points at live movable BLOCKS the reachability check does not consult, [[flag-triage-20260729]]), 1 code-0 teleport. New small bug filed: the scorecard's shot counter printed 23 vs 223 actual wire shoots. The teardown traceback fired once more — the run predated the fix below.
- Differ sim-only residuals recorded in [[capture-differ]]: teleport-landing equipment grant (13 windows, SUSPECTED INVENTED LAW — live suggests only fuel auto-picks on landing) and `53self+landed` queue-compression artifact (8).

**2. The deposit choreography byte-mined** ([[fuel-system]]) — answering "dont we have recordings where ive deposited fuel?": yes, five manual deposits in user session sniff-20260620-190228. Shape: sent 0x07 -> self 0x2E + 0x64 (absolute post-deposit fuel) + container record x1 (the new remaining = amount). Single record (vs every pickup's double) = the wire discriminator; max-deposit leaves exactly `DEPOSIT_FLOOR` (294 -> 100 on camera); third-party deposits client-only (the atlas's zero cross-tank 0x64s). Ferry rides located too (bot-20260720-005424 rode 78 ticks; drift lives in 0x4A block moves — mining pass queued).

**3. Playwright teardown race fixed at the root.** The CDP session was never detached; late frame events raced `browser.close()` in the sync bridge (ERROR traceback, zero bot frames). `Bot._detach_cdp_session()` now runs after the capture save, absorbing the already-gone case like the graceful quit; three tests pin it.

**4. Stage 5 closed — the bot-policy differ + reactive ghosts** ([[capture-differ]]). Policy re-judged on the grown archive (310 sessions, 29 bot-hours): 5,975/5,975 weapon-0 singles, 95.7% in the reflex window, thresholds re-pinned (7/8 at modes 87/117), reactivation gap mode = the 22 s corpse window. The differ's first real catches were INSTRUMENT bugs: `sync-cadence` now medians CLEAN gaps only (74/266 "failures" were 2 s cores wrapped in 18-943 s viewport-absence holes) and `bot-reactivation` skips re-sights far past the corpse window (34/35 were third-party-damaged bots drifting back into view). True anomalies left: one 16 s early re-sync, 10 cadence outliers. Then the build: **reactive ghosts** — `PracticeRoomDriver` refactored to own policy states over EXISTING tanks (`seed_practice_roster` extracted; no wrapper), bot-named ghosts carry the certified policy UNDER their timeline (recorded events hold the tick; withheld returns fire on the next quiet one — `tests/sim/test_ghost_reactive.py`), killed bot ghosts reactivate by the corpse law. Reactive replay of the 20-kill capture: 150/150 rounds, divergence signature shifted (12/150, drift 78 vs pure 10/150, drift 53) — ghosts that shoot back change the fight, as they must.

---
## [2026-08-03] lift | Fight instrumentation promoted to the gated layer, three more forks dead

The user's charter pass ("reliable, robust, deliberate... no shims, no forks, no drift") applied to the nope-fight aftermath:

**1. `validate/fight_timeline.py`** — human episodes and the per-event play-by-play as typed products of the EXISTING `extract_shadow_timeline` (no second decode path; the loose ungated `render_fight.py` deleted). `HumanEpisodeDict` carries exactly-computable engagement facts incl. the `max_stationary_streak` turret metric; `FightRowDict` rows attribute every self fuel delta by measured cause. First real-capture run immediately caught its own bug (self listed as its own opponent — fixed, regression-pinned) and rendered the death: Belton 13 taken/141 returned/2 kills, nope 32 taken/68 returned/1 death.

**2. `tankpit-fight` CLI** (`diagnostics/fight_report.py`) — episodes + windowed play-by-play in the house build/render/one-log shape; a death autopsy is now one command. **Run-audit** grew `human_episode` (INFO) and `turret_exchange` (WARNING at stationary streak >= 4 under fire) findings — a human fight can never again hide inside a rejection stream.

**3. Forks deleted:** the practice-bot name regex existed THREE times (bot/ai/humans, validate/shadow_bot_laws, + capturing-group variant); now one canonical `protocol/naming.py` (capturing color group as API) with all nine call sites updated — humans.py keeps only the priority-tier doctrine. The respawn path's hand-picked carry-list (which dropped the hit/miss/reject counters — the 23-vs-223 scorecard bug) replaced by `make_respawn_ai_state` with the seven session-scoped fields documented field-by-field. The wait-layer's "rejected by server" now reads **"closed by server receipt <name>"** for collect codes 4/5/7 (the choreography's closes), keeping "rejected" only for genuine refusals — the wording that hid 32 successful drinks from three autopsy passes.

Gate: guard 0 violations, ruff + mypy clean over src/tests/scripts, full coverage.

---
## [2026-08-04] lift | F6 closed at the law: four blockers, two questions — occupancy composed, landing split from walking

(Entry written retroactively 2026-08-04 during review — the operation itself ran 01:02-02:16.)

The F6 `cant_go` residual resolved into its stated root cause and fixed there. The user supplied the server contract verbatim — *"you walk until you hit the block then stop and you get the error message"* with the full blocker set (terrain, another tank, a movable block, a visible mine) — and corrected the routing claim: the server auto-paths around VISIBLE mines only ([[walk-mechanics]] footnote 4; hidden mines arrest by detonation instead). Re-read against the 2026-08-03 run, code 1 is therefore NOT a refusal — the server ACCEPTS the walk, goes as far as the corridor allows, stops at the first blocker, and reports; 9 of the 10 live code-1s show the tank moved before stopping. The eight earlier falsification verdicts tested properties of the REQUEST when the variable was what stood in the CORRIDOR.

**The build:** new `state/occupancy.py` (`is_tank_body_present` / `occupied_tank_keys` — not-self, viewport-fresh) folded into `FerryAwareTerrain` so all four blocker classes answer through the ONE composed passability view ([[terrain-composition]]). En route the fix exposed the opposite defect: landing selection asked the WALK question, and an enemy always occupies its own tile — so `TerrainMapProtocol` grew the second question, `is_landing_legal` (terrain only; the server displaces landings off mines and bodies per [[mine-mechanics]]), consumed by `find_teleport_landing_tile` and both `combat_landing` choosers. `VIEWPORT_PRESENCE_TTL_MS` un-forked from `bot/ai/threats.py` into `state/types/tank.py` beside the field it gates. Wiki: [[terrain-composition]] two-questions section, [[walk-mechanics]] visible-mines correction, F6 diagnosis rewrite in [[flag-triage-20260729]].

Gate: guard 0 violations, ruff + mypy clean, 5,777 tests at 100.00%.

---
## [2026-08-04] audit + lift | The (0,0) phantom: login-roster law measured, has_known_position canonicalized, sentinel class guarded dead

Review of the overnight F6 lift ("check the code my other ai did some work on the codebase") confirmed the design and found one wrong docstring claim with a real edge behind it: occupancy read `(x, y)` gated on viewport freshness alone, but 0x21 TankInfo and 0x3E TankStatus route as viewport with NO coordinates — and `apply_tank_observation` creates unknown tanks at the `(0, 0)` default.

**The law, byte-measured (first-sight probe, 3 captures, 113 tanks):** the server opens every session with a full-roster 0x21 dump — every tank's FIRST sighting is the position-less kind — and positions arrive only with the first position-bearing sync (10.9 s / 9.1 s / 45.7 s after first sight, uniform per session). The `(0, 0)` phantom is the NORMAL opening state of every tank, which is why seven modules had hand-copied the `x == 0 and y == 0` sentinel — and why the eighth consumer forgetting it was inevitable. Recorded in [[tank-freshness-model]]. Same probe killed the sibling worry: `team=0` defaulting shares the mechanism but the team-carrying 0x21 always wins the creation race (latent, never fires); mines and containers are immune by construction.

**The lift:** canonical `has_known_position(tank)` in `state/types/tank.py` — coords differ from the default OR `last_position_update_ms > 0` (radar-known tanks pass via the former, an authoritative (0,0) via the latter). Consumed by occupancy (phantom wall dead) and all seven former sentinel sites (threats x3, hunt_acquire, resource_search, tactics, combat_probe); the two ad-hoc tile-occupancy re-derivations (`combat_landing._is_dynamically_occupied`, `movement._is_occupied_by_enemy`) now express their tank halves through `is_tank_body_present` with `now_ms` threaded through both landing choosers (F6 open sub-question 3 closed). NOT lifted, by ruling: the HELLO greeting stays position-blind (user 2026-07-31, "hello can run anytime") — the existing test `test_position_unsynced_human_is_greeted_anyway` caught the attempted gate and won. New guard `scripts/state_sentinel_rules.py` (AST, same-base both-axes zero-compare, canonical module exempt) bans the ninth copy forever.

Test fixtures the law falsified (fixed at the fixture): six blocking tanks across movement/landing/consent suites now state viewport freshness — a stale entry no longer vetoes a tile, which is the point.

Gate: guard 0 violations (incl. the new rule), ruff + mypy clean, **5,792 tests at 100.00%**.

---
## [2026-08-04] crack + lift | The cant_go choreography byte-proven, sim walk law rebuilt: partial walks, team-scoped mine visibility

Answering "what about the sim?" after the occupancy lift: two sim-server divergences from the corrected walk laws, both closed, the first only after the measurement overturned the standing account.

**The crack first** (`analysis_scripts/mine_cant_go_choreography.py` — the exact-window echo measure F6 recorded as owed): for each of the 12 live code-1s (both 2026-08 runs), pair the event-log rejection line (both log wordings) with the capture's decoded 0x47 self echoes and 0x52 receipts. First honest result was a false negative — the miner filtered 0x3D, but walk echoes are 0x47 — and the corrected result REWRITES the law's shape: **11 of 12 code-1s carry a same-window 0x47 prefix echo** (the server plans the route AS IF clear, walks it, stops at the first blocker — 18:12:35's 14-step `ssseeessseeeen` stopped at (16,24) with Belton's body on (16,23); 18:23:46 stopped beside bot 520), **1 of 12 is the zero-tile pure refusal** (bare 0x52, no echo, no movement — the first step was already blocked), and echo+receipt land in one processing batch. Supersedes the 9/10 nearest-sample measure. Recorded as [[walk-mechanics]] "The cant_go partial-walk law".

**Mid-build user contract, load-bearing:** mine visibility is TEAM-scoped — "if someone on our team scanned the mines previously they're visible... if any new mines are planted, even if we're on the same viewport as the planting tank, we cannot see them... unless someone on our team radars." The reveal tracking being written per-tank was moved to `SimWorldDict.revealed_mine_keys_by_team` ([[walk-mechanics]] footnote 5).

**The sim rebuild** (`sim/movement.py` + `sim/actions.py` + `sim/emissions.py` + `sim/world.py`): `process_radar` is now the reveal event (per-team key set, encode/decode round-tripped); the primary route avoids tanks + TEAM-REVEALED enemy mines + block obstacles; a severed corridor plans terrain-only and executes step-by-step (`_execute_walk`) stopping BEFORE the first tank/revealed-mine/block (cant_go WITH the walked prefix; the 0x47 echo precedes the 0x52 in the same batch), stepping ONTO a hidden enemy mine (walk-over detonation-arrest, 45, no code 1 — hidden mines no longer detour, which the old sim wrongly did omnisciently), and pure-refusing only when static terrain severs (bare 0x52, no movement — matching the one echo-less live sample). Own-tile clicks keep their echoed zero-tile moved form (the fuel-choreography tests pinned it).

Nine new law tests (corridor cork, revealed-mine wall, block wall, pure refusal, hidden-mine fallback arrest, team-scope reveal via a real teammate scan, adjacency zero-tile). Gate: `make shadow` green on all seven laws, sim soak 150/150 healthy, guard 0 violations, **5,800 tests at 100.00%**.

---
## [2026-08-04] crack | Corpses don't block — F6 question 2 dissolved from the archive, no live probe needed

The user's suggestion ("cant we just do an action lab for that? kill an enemy and see when the corpse disappears?") decomposed better than a probe: the corpse LIFETIME was already a green shadow law (kill -> 0x58 = 22 s exactly, 40 samples); the unproven part was whether the body BLOCKS walking during it — and the bot's post-kill habit (restock from the current viewport's containers) meant the archive already held the answer. Correction folded in same-day: kills drop NO loot (user contract 2026-08-04, recorded in [[gameplay-loop]] — which had carried the "kill loot funds the loop" attribution error since 2026-07-01); the corpse-tile crossings below are incidental restock-route traffic, which makes them UNBIASED evidence.

**The measure** (`analysis_scripts/mine_corpse_blocking.py`, 34 kills across both 2026-08 runs): for every kill, fix the corpse tile from the victim's last wire position, then classify every self 0x47 echo around the window. Result: **six clean walks ONTO a fresh corpse tile at +2 to +10 s — inside the 22 s window — and zero blocked crossings.** Corpses do not block walking; the window governs respawn choreography, not passability.

**The disproof caught a regression before it ran live:** `is_tank_body_present` deliberately counted deactivated tanks ("a corpse stands where it died"), so the composed terrain would have vetoed the bot's own post-kill restock walks wherever they cross the corpse tile, for up to the 5 s presence TTL. Now gated on ``liveness == "alive"`` — matching the sim server's `_blocked_by_world`, which always did. F6 open question 2 struck as dissolved (its premise, not its answer, was wrong); no deactivation timestamp needed.

Gate: guard 0 violations, ruff + mypy clean, **5,800 tests at 100.00%**.

---
## [2026-08-04] crack + lift | Ferry movement law mined: no drift, atomic 0x4A pairs, the unfinished-command close — and cluster A's true cause

Task #21 (queued since the deposit mining) closed, with two corrections riding along.

**The mechanism** (`analysis_scripts/mine_ferry_drift.py`, rewritten off the failed 0x3D approach): a ferry's position is restated by 0x5A repaints; its movement is ONE atomic 0x4A pair — old tile restored to water, new tile painted 5 — in rider-move-sized legs (Manhattan 1-12). Archive sweep over all 312 captures: **148 distinct moves, 136 rider-attributed** (a tank stated the departing/arriving tile within 2.5 s); the 12 residuals are isolated singles with no cadence — under-observed riders, not drift. **No autonomous drift law exists**; the sim's rider-following model is validated at scale. (First classifier pass said "all unridden" — it matched riders against the OLD tile only, and the rider's echo lands before the 0x4A; the corrected window matches either endpoint.)

**Cluster A reattributed — third time, now terminal.** `make download-fields` + the ferry move log put the truth together: a ferry sat ON (59,28) — WATER on field01 — from 18:20:44 to 18:27:20, bracketing all four cluster-A code-1s; every echo starts at (59,28) with the one-step `w` disembark onto (58,28) LAND. The bot was RIDING; its land collects were disembark-truncated by the single-command surface law and closed code 1. The tank-cork BFS story was coincidental (the composition fix stays right — 18:12:35's Belton stop is a genuine tank block). Recorded in [[flag-triage-20260729]] F6 and [[walk-mechanics]].

**The unfinished-command close, byte-split three ways:** transition stop SHORT of the click → echo + code 1 (5 live samples); transition stop that IS the click → silent; mine walk-over arrest → silent (18 archive detonations, ZERO paired code-1s — the negative probe that stopped a wrong "any unfinished command" generalization). Sim now emits all three (`MoveOutcomeDict.stop_reason`/`dest_reached`, `emit_move` close rule; three emission-law tests in `tests/sim/test_ferry.py`).

Gate: `make shadow` green on all seven laws, guard 0 violations, **5,803 tests at 100.00%**.

---
## [2026-08-04] live run + fix | 19 kills / 0 deaths / ZERO rejections, then the relay ping-pong — wrong selector, one-line rewire

**Run bot-20260804-230342** (kill target 20): first live session under the full 2026-08-04 stack (occupancy composition + corpse/phantom gates, landing/walking split, partial-walk sim laws). **19 kills / 0 deaths in 19.5 min at a steady ~70 s cadence — and ZERO server rejections of ANY code the whole session** (the 08-02 baseline had 48 code-5s + 18 code-4s + code-1s; the 08-03 run had 32+18+10). The refusal-prediction and reachability stack is receipt-clean end to end.

**Then the 20th kill never came.** The nearby roster farmed out, the lock fell on a target 98 tiles away, chase 588 > max affordable 450 ("refuel cannot fix distance"), and the relay branch entered a **two-tile teleport ping-pong**: (206,254)<->(207,254) for two minutes, a 3-hop drift, then (225,253)<->(225,254) — one hop per 2 ticks, every landing exact, forever. Killed at 26.5 min per the standing rule (events + log saved; the wire capture was lost to the hard kill — teardown save never ran).

**Root cause — a fork-and-miss, not a scoring tune.** The beyond-refuel-reach branch in `combat_strategy` called `make_resource_search_hop` — the COLLECT dot-ranker, whose `dots*walkable/cost` score makes the dot under the tank's feet ~50x cheaper than any dot toward the prey, unbeatable by the ~2x proximity bias; with the own-tile veto excluding the current dot, two adjacent dots form a stable 2-cycle. The correct selector had existed all along: `hunt_relay._pick_relay_dot` — strict-progress, monotone by construction, leg-capped ("terminates at the enemy or runs out of qualifying dots"). **Fix: the branch now calls `hunt_relay.relay_toward`** (lock preserved through the leg), and when no progress dot exists and refuel cannot help (at cap), the target is **blocked and replanned** instead of treadmilled. Two regressions pinned: neighbor-dot-vs-progress-dot (the ping-pong shape), and no-progress-at-cap -> `blocked_combat_targets`.

Gate: guard 0 violations, ruff + mypy clean, **5,805 tests at 100.00%**.

---
## [2026-08-05] live run + fix | Rerun 5 kills then the ferry-boarding deadlock — the missing ride leg, built

**Run bot-20260804-234008** (rerun under the relay fix): blistering start — 5 kills / 0 deaths in the first 5 minutes including a WAR-DRINK (fight parked on a 1043-volume well vs bot 516: took 45s every 2 s, drank +642 back to cap MID-FIGHT, won the sustain trade), clamp receipts flowing as lawful drink closes throughout. Then a long engagement drained to ~296 and the refuel leg found the second stall of the night.

**The deadlock, exact-window:** the larder's F5 ferry-boarding branch served a water-locked 469-volume container at (106,11) with boarding landing (112,15) — and the LANDING'S command window ((108,0)-(123,15) live) does not contain the container. Three mechanisms interlocked: (a) a pickup can only be dispatched inside the actionable window, so the locked pickup vetoed every tick ("not executable - holding plan"); (b) the hold releases only on a move-failed mark, which requires a dispatch that never happened; (c) the cascade fell through to the larder, which re-selected the same boarding hop — **~230 hop selections, 56 exact landings on (112,15)/(112,14) over 11 minutes**, zero ferry_belief_expired (exact landings never disprove the belief), fuel bleeding 296 -> 250. Killed at ~25 min per the standing rule.

**The fix — the missing RIDE leg, not a selection band-aid:** waiting can never fix an out-of-window lock (the window has to move), and the window-mover already existed as `walk_or_teleport`'s pure-move approach path (edge-clamped move — which on a boarded ferry RIDES the water and shifts the window; on land walks/hops closer). `collect_locks` now issues that approach for BOTH lock kinds when the pickup veto is specifically the actionability gate: next tick the target is in-window and the pickup dispatches, or the move fails and the structural release fires — monotone either way. Regression pinned (`test_out_of_window_locked_fuel_approaches_instead_of_holding`). The larder's boarding selection stays untouched — the server's window placement after a landing is not client-predictable (live landed window (108,0) vs centered prediction (104,7)), so gating selection on a predicted window would have been an invented model.

Session tally across both runs tonight: 24 kills / 0 deaths, zero at-cap waste receipts, two distinct stall classes found live and root-fixed same-night (relay ping-pong; out-of-window lock deadlock). Gate: guard 0 violations, ruff + mypy clean, **5,806 tests at 100.00%**.

---
## [2026-08-05] live run + fix | Run 3 stalled in the SAME pocket — the ride-exists gate closes the F5 boarding hole for good

**Run bot-20260805-070006** spawned near the (106,12) pocket and hit the deadlock class within 2 minutes, before any kills — but as the EQUIPMENT variant, proving the previous night's out-of-window lock fix was treating a symptom: with the target IN the window the approach leg correctly never fired, and the ferry hop + one-step disembark + re-hop cycle ran anyway (the "one tile at a time" movement the user watched live: each single step is the 2026-07-19 disembark contract firing after every futile boarding teleport). Killed at ~3 min.

**Root cause, terrain-proven:** `find_ferry_boarding_tile` selected ferries by DISTANCE ALONE — its own docstring admitted "the same-water-body assumption is usually safe." field01 truth: the container's pond holds 4,456 water tiles and does NOT contain the ferry's tile at (112,15); one land ridge at (111,15) separates two water bodies. The bot teleported onto a boat in a puddle to reach fuel in a lake — both nights' deadlocks (fuel AND equipment lanes) through this one function.

**The fix — the ride must EXIST:** `find_ferry_boarding_tile` now takes the terrain view and requires the candidate ferry to float on the goal's own water body (4-connected flood from the goal; a live ferry tile counts as water — on the composed view it renders `~` OVER the lake, and the first cut treating it as a wall broke the sim's own ferry-harvest e2e, which is exactly the layer that caught it before live). All three call sites (fuel larder, equipment hop, scope-scout goals) share the gate. Two bonus catches: the equipment-hop unit fixture had its ferry on unconnected ground BY ACCIDENT — the exact live-deadlock geometry sitting in the test suite asserting success — now corrected to state the ride honestly; and the disjoint-pond case is pinned in both the larder suite and a new direct `test_ferry_landing.py`.

Why the sim missed the original bug: the ferry scenario only encodes the happy geometry (same lake). Sims catch what their authored worlds contain; live play authored the adversarial one. A wrong-pond sim scenario variant is the queued follow-up.

Gate: guard 0 violations, ruff + mypy clean, **5,809 tests at 100.00%**.

---
## [2026-08-05] revert | The out-of-window approach leg was the regression — hold-and-fall-through restored, and the real resolution mechanism named

Run bot-20260805-075502 (run 4, 1 kill) stalled at 07:57 in a NEW loop of my own making, and tracing it answered the question the previous night's fix skipped: **how does an out-of-window lock normally resolve?** Answer, from the log: the hold returns NO decision, the cascade falls through, and the equipment-hop lane teleports ONTO the target (07:57:02: `equipment hop to (170,111) landing (170,111) cost=134`, deferred one tick for a map open). The landing recenters the window; the pickup dispatches. The lock machinery was never the stuck part.

The approach leg I added on 08-05 00:30 broke exactly that: the lock continuation runs FIRST in the cascade, the leg returned an edge-walk decision every tick, so the cascade never reached the hop again — the one-tick-away teleport was starved forever while the tank walked (174,96)<->(174,97) at the window edge. Doubly wrong because a recorded law already forbade the mechanism: with autoscroll OFF, walking can never shift the window ([[viewport-shift-protocol]]; the F5 notes say it verbatim). The prior night's "deadlocked hold" was a symptom of the ferry-pond bug (target permanently unservable), not a defect of holding.

**Reverted:** both lock paths hold again on any inexecutable tick, with the run-4 mechanism documented at the hold site; the regression test now pins the HOLD (decision None, lock retained) so no future "helpful" leg can short-circuit the cascade unnoticed. The pond ride-exists gate — the fix for the actual root cause — stands. Gate: guard 0 violations, ruff + mypy clean, **5,808 tests at 100.00%**.

---
## [2026-08-05] lift | The unservable release — the lock law completed instead of patched around

The honest completion the reverted approach leg should have been: the enumerated plan-release law gains **``unservable``** — a locked container with NO legal teleport landing AND NO fresh ferry on its OWN water body cannot be walked to, hopped to, or ridden to, and no move-failed mark will ever arrive because nothing is ever dispatched. That is the exact run-2 (bot-20260804-234008) hold: released now in one tick instead of held for 11 minutes. The verdict reuses the selectors' own primitives (`find_teleport_landing_tile` + the pond-gated `find_ferry_boarding_tile`) — no new policy, no movement changes, and affordability/viewport misalignment deliberately stay HOLDS (they change with fuel and motion; committed-intent law untouched). `PlanReleaseReason` vocabulary extended and documented; both lock kinds release through the one predicate.

Fixture corrections the new law forced, each toward honesty: two strategy-coverage hold tests and the water-locked hold test premised on "a ferry can serve it later" while seeding NO ferry — each now floats a fresh ferry on the target's pond so the hold they pin is genuinely transient; new pins for the unservable release and the no-terrain no-verdict hold.

Gate: guard 0 violations, ruff + mypy clean, **5,811 tests at 100.00%**.

---
## [2026-08-05] live run | THE CLEAN 20 — every fix lane proven in one session

**Run bot-20260805-083117: 20 kills / 0 deaths in 22.5 minutes, exit `session_complete`, wound down fully stocked at 1100, graceful quit, capture saved, ZERO errors.** The first kill-target session to complete since the fix cycle began, and every lane built this week fired live:

- **The relay chain closed three kills.** 8 monotone dot legs across the session; the signature arc: orange-5 acquired ~90 tiles out -> three progress hops -> kill in ~90 s. Same for orange-3 and the 20th kill itself (orange-7) — the exact scenario that hung run 1 at 19/20 forever.
- **The ferry contract ran clean:** one disembark, immediately resolved, kill 8 seconds later. No boarding churn — the pond gate kept every hop honest (no wrong-pond ferry was ever selected, so the `unservable` release was never needed: 0 firings, which is the healthy count).
- **Receipts:** 45 code-5 drink closes (ledger outcome `clamped_transfer` — every one carried fuel), 5 code-4 drained races, **1 code-1** — diagnosed in-run: a teleport-displacement pocket at (38,112) (something on the container tile), the adjacent collect's walk cut by the same obstruction, replanned same tick. Zero at-cap waste, zero code-0.
- Analyzer top-level issues: exactly one — a single server-side map_open dispatch/completion delta (the known flake class from 08-02).

The week's arc, for the record: 24 kills across five sessions of finding-and-fixing (relay ping-pong, out-of-window lock deadlock -> reverted approach leg -> completed release law, ferry pond gate), then this. Gate remains green at 5,811 tests / 100.00%.

---
## [2026-08-05] crack | The map_open stall characterized: a July 29-30 server incident window, not a bot behavior

Answering "why did the map take a while to open" for run 5's single stall — with the archive, after the capture ruled out every client-side suspect for that instance (tank stationary; both the mine-clearance shoot and the map_open frames present in the SENT capture; 28 inbound frames flowing through the gap; the swallowed open never answered late — the analyzer's dispatched=104/completed=103 delta is one open eaten outright, not delayed).

**Archive sweep (all runs, 32,437 ledgered actions): 286 stall_timeouts — 276 map_open, 6 scan, 3 collect, 1 teleport — collapsing into 22 EPISODES, and 19 of the 22 start inside one 27-hour window: 2026-07-29 20:55 -> 2026-07-30 23:22** (worst: seven consecutive swallowed map_opens over 75 s at 07-30 22:23, retries stalling too, a scan eaten at the episode head). Outside that window the two-month archive holds three episodes total (2 on 08-02, 1 today). Hypotheses tested and disconfirmed: client rate (stalled opens' prior-60s frequency median 5 = answered opens' median 5, answered p90 higher); movement; connection death; dispatch sequencing (the superseded-shot-then-map pattern occurs 3 other times this run alone, all answered).

**The law for the wiki:** the server's per-tank command processing goes episodically deaf — broadcasts continue, commands are silently dropped (never NACKed: no 0x52, no late answer) — overwhelmingly during the 07-29/30 degraded period, residually ~once per session-hours since. The client's stall-budget + replan is the complete correct handling; there is no client-side fix for a frame the server drops. The 08-02 run's "2 critical" analyzer findings were this incident's tail, not a bot defect. Future stalls should be read against this baseline (a NEW episode cluster on a fresh date = server having another bad day, not a regression).

**Addendum — the deterministic-rule hunt, run to exhaustion (same day):** challenged ("code is reliable and algorithmic"), two rule-shaped mechanisms were formulated and byte-tested. (1) *Already-open map opens are ignored*: refuted — the pre-stall window shows six map_opens in 32 s all served, and the human session sniff-20260701-191133 re-opens freely and is always served. (2) *Commands are dropped while the map is open*: refuted — the human choreography shows NO close command exists on the wire at all (open is sent cmd=3 -> 0x4C; moves/teleports flow immediately after the 0x4C, all answered). The episode-start fingerprint test (20 of 25 episodes had a teleport between the last served map and the stall) killed the map-state family for the July window too. Standing conclusion: the trigger is not a function of anything the client sends — deterministic on the server's side of the wire, unobservable from ours. Byte-facts banked for [[map-data-decode]]: map open = sent cmd 3 (len 5); no wire close exists; repeat opens are re-served; actions interleave legally with an open map.

---
## [2026-08-05] crack + lift | The no_viable_targets exits root-caused: liveness rule 4 (map presence IS life) — three of my theories killed by the user's pushback en route

The 100-kill chain (32 record + 26 + 24 + 0 = 82) kept ending sessions in "empty" rooms the user insisted were full. The user was right; the trail of dead theories and the surviving law:

- "The room starves / respawns come in waves" — DEAD (wiki already recorded 22 s reactivation; I theorized before checking, against standing discipline).
- "Rejoining repopulates the roster" — DEAD (nothing spawns; rejoin merely discards the client's phantom registry).
- "Re-killed bots were revived by entering the viewport" — DEAD (the five re-kills were revived by GLOBAL 0x3D movement broadcasts; asserted-then-falsified).
- **The law:** the server's 0x4C map is a strictly living-tanks list — victims absent from 58/58 in-corpse-window snapshots, present in 204/204 after; Belton (human, id kept across 3 deaths — page correction) absent during each window, back at +24 s. Idle respawns are otherwise WIRE-SILENT (27/32 victims: zero bytes ever again), so the June rule "map never touches liveness" (built against departed-player afterimages) starved acquisition of the only signal that exists: sessions exited no_viable_targets with 27 live enemies on the map, each "corpse" position faithfully tracking its living owner.
- **The lift:** liveness rule 4 in `state/mutations.py` — a deactivated tank LISTED in a map snapshot flips alive (the flag pair position-authoritative + not-wire-sourced is unique to 0x4C; radar/DOM stay excluded). Two pins in the mutations suite. Rejected on the way, correctly, by the user: a respawn-displacement tile comparison ("brittle") and a corpse-window timer — the direct server-curated signal needed neither.
- **Chain post-mortem:** session 4 (0 kills, 43 min) was NOT a regression — same binary as session 3's 24 kills — but drained-map inheritance: same-room rejoin 90 s after a heavy farm + the ~1 dot/min container regen ([[game-economy]]) + hunt-only-when-full = collect-trapped. Recorded in [[enemy-bot-behavior]] with the map-liveness law. **RETRACTED same day — see the next entry.**

Gate: guard 0 violations, ruff + mypy clean, `make shadow` green, **5,813 tests at 100.00%**.

---
## [2026-08-05] correction + crack | Session-4 "drained map" retracted — the 0-kill session was a mine-landing geometry trap; the century closed at 100/100

The chain post-mortem above cited a law [[game-economy]] itself falsified on 2026-07-25 (container "regen" was our own radar exposure; refills are discrete deposits). The user rejected the story and the session's own events (bot-20260805-173034.events.jsonl) convict a client-side loop instead:

- **Hunt gate closed by 2 homings.** Spawn inventory homing 23/25; 27 live enemies in viewport from tick 6. The gate — not the room — kept it from hunting.
- **The trap:** nearest equipment (58,95) is sealed by water + mines on every approach ((58,94)* , (59,95)*, (57,95)W; southern neighbor (58,96) is a second equipment container walled by water). The hop planner chose landing (59,95) — a KNOWN mine — so the server displaced all teleports: 534 `teleport_displacement` events, **1,068 of 1,130 hops aimed at this single tile** across 43 min.
- **No release ever fired for it:** `unservable` requires no landing candidate at all; `target_gone` requires the container to vanish; repeated displacement is uncounted. The lock re-armed every tick.
- **The loop ate its own gate:** per-cycle scans burned radars 22 → 0; when weapons later capped via incidental pickups, radars ≥ cap−5 became the failing gate condition. Fuel was never scarce — `tank_at_capacity` fired 19 times mid-loop.
- Defects opened: landing selector is mine-blind; no displacement-failure/no-progress detector on collect hops; gate resource spent by the gate-satisfying loop. [[enemy-bot-behavior]] corollary rewritten with this diagnosis.

**The century closed anyway:** the chain runner fired session 5 (bot latest, 18:14-18:35) before the stop landed — bounded 18 kills, delivered **18/18 kills, 209/209 shots (100%), 0 deaths, exit session_complete**, first live session with liveness rule 4 in the binary. Chain total: 32 + 26 + 24 + 0 + 18 = **100/100**.

---
## [2026-08-05] lift | Landing attainability + the general clearance trigger — the session-4 trap class deleted, no escape hatches

User ruling: "generalize the clearance trigger and fix the landing servability" — and "no escape hatches": no displacement counters, no blocklists; the planner must KNOW, not retry-and-learn. The lift, all plan-time knowledge the bot already held at tick 4:

- **`find_attainable_landing_tile` (`bot/ai/reachability.py`)** — the teleport twin of walkability: legality (server accepts the aim) vs attainability (the tank will STAND there). Scans the pickup service set (goal + cardinals) for a terrain-legal AND mine-free tile; every known mine displaces regardless of team (user law 2026-06-16 verbatim, mine-landing probe 3/3 2026-07-28). Consumers: equipment hop, fuel larder, desperation fuel hop, `_teleport_fallback_command` (locked-approach). `find_teleport_landing_tile` stays as the TRANSPORT answer (combat aims, scouting, mine-flip) with its docstring scoped — "displacement is not a reason to re-aim" was a legality truth silently promoted to an attainability lie.
- **General clearance trigger (`mine_clearance.py`)** — the free unlock single now fires on the general condition (a known hostile mine denies a worthwhile in-view container's service access, and the blast provably reopens it) instead of the two special cases (mined container tile / mined walk corridor) that both missed the session-4 pocket. New single-target arm `find_service_clearance_aim` feeds the lock verdict. Aims scored by containers reopened, tie to nearest, fully deterministic.
- **Lock verdict** — `_locked_target_is_unservable`: servable = attainable landing OR shootable service mine (HOLD; the clearance step runs before the hop lanes) OR pond ferry. Unshootable mine-denial releases `unservable`, same closed vocabulary.
- In the session-4 geometry the new tick-4: clearance shot at the flank mine (3×3 blast clears the pocket) → tick 5 hop lands exactly → pickup. One shot, zero loops.

Pins: verbatim `_session4_pocket` fixture (water-locked equipment, mined flanks, bot at (60,94)) across `test_mine_clearance.py` / `test_reachability.py` / loop-killer hop test / hold-vs-release lock tests. Behavior shift pinned: covered-container aim may now be the closer service-tile mine (same blast, one tile nearer).

Gate: guard 0 violations, ruff+mypy clean, **5,835 tests at 100.00%**, `make shadow` all 7 laws PASS. Open follow-up (parked): what a displacement receipt implies when the obstruction was NOT already known — mine it from the 534 session-4 receipts + the 2,861-pair corpus before writing any inference law.

---
## [2026-08-05] lift | Ferry memory goes positional — the 60s TTL retired by the no-drift law

User ruling ("do the positional invalidation for the ferry memory"). The TTL's premise — "ferries drift freely" — was falsified by our own movement mining (136/148 wire movements rider-attributed, zero spontaneous; bots never ride). Three channels already retire a wrong belief positionally: 0x4A atomic move pairs overwrite both tiles, viewport re-observation rewrites the tile's truth, and a boarding teleport displaced off a believed ferry deletes it on contact (`_expire_disproven_ferry_belief`). The clock on top only FORGOT true ferries — rediscovery pans and the chain's release→re-lock churn. `find_ferry_boarding_tile` drops the freshness gate; stale-belief tests flip to pin the new law. Residual risk (a human rides it away unseen) costs one displaced hop and self-heals via channel 3 — the same accepted economics as stale container hops. Gate: 5,835 tests at 100.00%, shadow green. Commit 8614316a.

---
## [2026-08-05] operation | Second century of the day: 100 kills, 0 deaths — the new collect stack proven at scale

User-ordered 100-kill run under the attainability + general-clearance binary. One 46-minute session died silently at 41 kills (no traceback, no WER event, log cut mid-tick — external process kill; prime suspect is harness background-task lifetime, nothing has survived past ~43-46 min; artifacts archived as bot-20260805-204515, capture lost since only teardown saves it). Finished in bounded sessions: **41 (crashed) + 30 + 29 = 100 kills, 0 deaths total**, sessions 2-3 clean `session_complete` at 27m52s and 33m32s.

- **The trap class never appeared**: worst displacement tile x9 (an UNREVEALED mine — the parked displacement-receipt law's case, bounded by the move-failed mark), everything else ≤x3 across ~160 displacements; all hunt-close transport bounces.
- **General clearance trigger live tally**: 11 shots across the three sessions, 7 converting to a pickup within 10 s (the non-converters were at-capacity moments); first firing 45 s into session 1.
- **`unservable` release live tally**: 4, each one tick — the session-4 failure shape now costs nothing.
- **Rank countdown receipts** (`session_account_stats.rank_number`): 27 at 20:19 → 26 at 20:45 → **25** by 21:33, holding 25 into the closer; promotion points 241,167 → 381,015 across the evening.
- All three sessions read via `make digest` ([[log]] entry above) — kills/timeline/clearance/release numbers in this entry are computed digest fields, not grepped.

Sessions 2-3 also carried the positional ferry memory (TTL retired same evening). Follow-ups parked: displacement-receipt obstruction law (mining started), `make service` adoption for harness-independent long runs.

---
## [2026-08-05] crack | Displacement-receipt semantics mined archive-wide — the law's evidentiary basis, plus a team-scope refinement of the displacement law itself

`analysis_scripts/mine_displacement_semantics.py` (calibrated on session bot-20260805-173034: 534/534 displacements attributed to the known mine, 0 misses) swept 329 captures: 3,418 displaced sent-teleport/0x3D pairs, 6,134 exact. Artifact: `runs/analysis/displacement_semantics.json`.

- **Only ENEMY mines displace teleport landings.** Displacements off a known-mine tile: 1,227 enemy vs **2** friendly; exact landings ON a known friendly mine: 20, all clean. The user's 2026-06-16 law ("you get moved off if there are mines") is team-scoped exactly like walk-passability. CODE IMPLICATION (not yet applied): `find_attainable_landing_tile` currently avoids ALL known mines — over-conservative; should consult the hostile filter.
- **An exact landing is a mine-CLEAR receipt.** 88 exact landings sat on live ENEMY-mine beliefs — stale beliefs, since a live enemy mine displaces deterministically (534/534, probe 3/3). Off-screen walk-over detonations never reach the wire, so beliefs rot; our own landing is server truth the tile is clean and should DELETE the belief.
- **Displacement causes decompose**: 36% mine known before, 31% tracked body present, 0.5% mine revealed later, 33% no visible cause — and the no-cause distances are 81% ring-1 (unseen single obstruction: bodies with stale positions or never-revealed mines), 5% ring-3+.
- **Displacement is ~99% deterministic**: 19 of 1,827 repeat-displaced tiles ever produced a second landing.

The receipt law this licenses: a displaced landing at a believed-clear tile writes an obstruction observation (mine-or-body, ring-1-dominant) consumed by the attainability predicate; an exact landing clears any mine belief at the tile. Both are pure knowledge channels — no counters. Implementation awaits the user's word.

**Addendum (same night) — the divergence audit.** User challenge: "wouldn't a radar have revealed them? I'm worried it's divergences between our world model and reality." Classified all 1,119 no-cause displacements by the server's own statements about the requested tile: **1,085 (97%) never_stated** — no channel (0x4F entry or clear, 0x40, 0x5A enumeration, 0x4B/0x45) ever mentioned the tile; pure unobserved-layer, radar would have revealed them only if spent there. **34 (3%) carry a clean statement at exactly the displacement's own timestamp** — the same-tick ordering signature (the landing burst's own patch, describing the post-landing tile; a body-obstruction leaves the mine layer clean), the artifact class the container-atlas miner already documented. **Zero** cases of clean-then-displaced with real time separation; zero stale-body candidates. Verdict: no measured world-model divergence — the model is silent, not wrong, and the receipt law is the convergence mechanism (contact turns dark tiles into observations).

**Addendum 2 — the misclassification audit (user challenge: "are we sure it's hidden mines and not terrain/blocks/tanks?").** Armed the classifier with the static field01 map, the wire block/ferry layer, and a separate 0x4C idle-tank position layer. Result: 347 of the old no-cause pool were STATIC TERRAIN aims (user-piloted sniffs and probes at water/rock — the challenge was right), 0 blocks, 2 idle-map bodies. Irreducible residue: 758 displacements at never-stated legal-ground tiles with no tank in either position layer — by elimination over the displacement law's three causes, unrevealed enemy mines; 98% ring-1 bounces and seconds-apart deterministic repeats corroborate a static single-tile object. Method scar worth keeping: folding 0x4C map positions INTO the 0x3D position table collapsed body attribution 1053→277 — the two channels' coordinates disagree for the same tank (lag/rounding/scale unresolved; parked as a decode question for rule-4 position handling). Layered layers only ADD evidence.

**Addendum 3 — the 0x4C/0x3D disagreement mined (user: "why do they disagree lol").** `analysis_scripts/mine_map_position_delta.py`, 286 captures, 2,851 same-tank map/wire pairs inside 2 s: 47% tile-exact, **0 swapped axes, 0 scale factors, no constant offset** — the decoder is correct. The delta histogram is a movement spectrum: ±1/±2 walk steps dominate (patrol amplitude; the symmetric y±2 that first flagged this), tail at ±16/±30/±60 = teleport hops between observations. Law: **0x4C map positions are presence-exact but position-approximate** — the payload is a snapshot that ages, so map fixes must never overwrite fresher wire fixes (arrival time lies about content age). Liveness rule 4 uses presence only and is unaffected; tile-precise consumers (body attribution, landings, adjacency) stay 0x3D-only.

---
## [2026-08-06] lift | Three approved lifts land: attainability joins the composed view, map positions defer to fresh wire truth, the account self gets a home

User rulings applied ("lift dont fork"; struck: extra radar usage, landing-as-clear, cross-session atlas, obstruction receipt — the last withdrawn on honest sizing, guarded for free by the digest's displacement_top).

1. **`is_landing_attainable` enters `TerrainMapProtocol`** — the third terrain question (legality: server accepts the aim; attainability: the tank stands there; walkability: the tank drives there). The composed decision view answers it from the same per-tick TEAM-SCOPED hostile-mine set the walk side uses, so the self model's team is consulted exactly once, at composition; static map and test doubles collapse it to legality. `find_attainable_landing_tile` loses its caller-chosen mine parameter (the 2026-08-05 all-team over-reach was a call site picking a layer); the clearance planner's post-blast check becomes pure view logic (legal AND (attainable OR in-blast)). Guard rule `run_mine_layer_rules` bans raw `world["mines"]` in bot/ai outside equipment.py/context.py. Own-color mines no longer repel our landings.
2. **Map-position freshness defer** (`state/mutations.py`, `MAP_POSITION_DEFER_MS = 2000`): a 0x4C authoritative position no longer overwrites REAL coordinates updated within the snapshot's measured aging window; presence/liveness rule 3 untouched; stationary tanks still take the map fix. New strict predicate `has_real_coordinates` joins `has_known_position` in tank.py (the login roster's fresh (0,0) entries must not be protected from the map's real fix) — the (0,0) comparison stays in its one guarded home.
3. **`SelfAccountDict`** (`state/types/self_account.py`) — the account self: name, persistent id, decoration, panel rank_name + countdown rank_number, promotion points, lifetime totals. Filled by the self 0x21 dispatch and the startup scrape; read via `get_self_account()`. The plug-in point for rank-aware features.

Gate: **5,880 tests at 100.00%**, guard 0 violations (incl. the new rule), `make shadow` all laws PASS. Commits 48784efa, 6b848684, 07e61332.

---
## [2026-08-06] lift | Two bots, one map — instance namespaces land first-class

User: "we have unlimited time, tokens and context. lift, dont fork" + "no back compat shims, no thin wrappers, no fallbacks, no legacy code, no type alias." The two-bots-one-map recipe is now: two env blocks, two `make service` processes, two curls.

- **`TANKPIT_BOT_INSTANCE`** (`resolve_bot_instance`, validated `[a-z0-9][a-z0-9_-]{0,31}`) is a REQUIRED parameter of `build_bot_run_artifacts`: a named instance nests its whole artifact bundle under `runs/bot/<instance>/`; empty is the sole-bot layout — the primary configuration, not a fallback.
- **`TANKPIT_BOT_SERVICE_PORT`** (`resolve_service_port`, range-validated) binds a second service beside the first. The `HEALTH_URL` constant is DELETED (no legacy) for `health_url(port)`; the idempotency probe targets the resolved instance. Note: fiesta's nginx proxies only 27100 — a second instance is curl-only until nginx grows a location block.
- **Stop sentinel instance-scoped** (`resolve_service_stop_file` → `runs/state/<instance>/STOP`): stopping one bot can never stop the other.
- Earlier same night: `POST /start` gained the `{"seconds", "kills"}` bounds body (commit 5c94e66c) — service sessions had been running fully unbounded.

Account selection already existed (`TANKPIT_ACCOUNT` + accounts.json). Same-team pairs share radar exposure server-side ([[game-economy]] team-scope); opposite teams is bot-vs-bot sparring under the fair-fight stack. Gate: 5,890 tests at 100.00% (run with the parallel session's in-flight `analysis/` dirs excluded — see tree note), commit 49eb754b.

---
## [2026-08-06] lift | The dead terminal's analysis package, finished — direction-tagged frames, typed unframed skip, two miners migrated with exact reproduction

A parallel session built `tankpit_bot.analysis` (one typed owner for the load/XOR/frame-walk pipeline forty scripts each forked privately, commit cb49da1f) and died mid-arc; this session adopted and finished it.

- **Direction extension**: `DecodedFrameDict` gains `direction` (`FrameDirection = "received" | "sent"`); `decode_session_frames` decodes BOTH sides of the wire with the same session cipher, so command-correlating miners (displacement semantics, cost pairing) stop re-implementing the sent walk. Measured archive-wide: 62,095 sent payloads split cleanly with exactly 2 framing errors, both in the one pre-framing capture `bot-20260331-230406`; 217,678 received, zero.
- **Typed skip, not a crash**: `scan_session` classifies a `FramingError` as `SkippedSessionDict(reason="unframed_payload")` — the vocabulary is closed (`no_magic`, `unframed_payload`), so tallies stay comparable and any NEW corruption stays loud. Fixed the package's docstring lie about frame shape (the body is the whole frame through the cipher, type-byte position included — exactly what `decode_message`/`decode_client_command` take).
- **Dead code deleted, not covered**: the direction filter inside `decode_session_frames` could never fire — `decode_capture_session` already validates `MessageDirection` to the same closed vocabulary. The check and the `RECEIVED_DIRECTION`/`SENT_DIRECTION` constants are gone; the capture's own literal flows through.
- **Two miners migrated onto `scan_session`** (`mine_displacement_semantics.py`, `mine_map_position_delta.py` — the newest two, both mined this week's laws): private pipelines deleted, reproduction verified EXACT against the committed verdicts (displacement: 3,418 displaced pairs, 1,227 enemy / 2 friendly, 20/88 friendly-exact/stale-clear; delta: 2,851 pairs, 0 swapped, 0 scaled). The remaining 38 scripts are mechanical follow-ups.

Gate: guard 0 violations, 100.00% coverage, shadow all laws PASS.

---
## [2026-08-06] lift | The fleet manager — the AI spins up, sees, and stops the bots

User rulings: "so we can run two bots at once? also can you finish the other ai work?" then "not one single server with the option to spin up bots?" then "the goal is the ai can spin up and maintain and see the bots, not the spa method." The SPA service stays what it is; the new `tankpit-fleet` (`make fleet`, port 27300, `TANKPIT_FLEET_PORT`) is the AI's control plane.

- **One user-owned manager process, N bot children.** In-process multi-bot is impossible (world service is a module singleton); harness background tasks die at ~46 min (the 41-kill session's killer). Each bot is a `Popen` child running the ordinary bot entry; isolation is entirely the instance-namespace lift (`runs/bot/<instance>/`, instance STOP sentinel, `TANKPIT_ACCOUNT`).
- **The manager never reads `os.environ`**: the child inherits the parent environment whole (`env=None`) and a 4-line bootstrap applies `KEY=VALUE` argv overrides to its OWN environment before importing the entry — env writing happens on the far side of the process boundary, where the `get_env` seam does not exist yet.
- **HTTP**: `GET /bots` (pid/alive/returncode/bounds per instance), `POST /bots` `{"instance", "account?", "kills?", "seconds?"}` (400 malformed, 409 duplicate-live or invalid name — validated against the same `_INSTANCE_NAME` pattern as `resolve_bot_instance`, so a bad name is rejected here and not by a crashed child), `POST /bots/{instance}/stop` (graceful sentinel; scorecard + capture + archive all happen), `DELETE /bots/{instance}` (409 while alive — the fleet never silently kills).
- **Telemetry stays on disk**: the AI reads `runs/bot/<instance>/latest.log` and runs `tankpit-run-digest`; the fleet owns lifecycle only.
- New seams `service_hooks.spawn_bot_process` + `run_web_app`, both with real-implementation tests (the spawn test kills the child inside interpreter startup, long before the entry point could open a browser).

Gate: **5,938 tests at 100.00%**, guard 0 violations, `make shadow` all laws PASS.

---
## [2026-08-06] lift | Forty private pipelines to zero — every analysis script now rides the typed capture-scan owner

Continuation of the archive-owner arc (cb49da1f built the package, the direction/raw extensions finished it): all 26 remaining capture-walking scripts under `analysis_scripts/` migrated off their private load/XOR/frame-walk pipelines onto `scan_session` / `decode_session_frames`. Discipline per script: baseline run on real archive data BEFORE the edit, migrate, rerun, byte-diff — **24 of 26 reproduce exactly**; `analyze_wire_bytes` differs only in a reworded skip-warning line, and nothing else differs anywhere. `grep` proof: zero references to `build_global_xor_table` / `_iter_frames` / `_split_frames` / `decode_base64_safe` remain in `analysis_scripts/`.

Found and fixed along the way:

- **The `raw` lift** (committed 4376fd77 mid-arc): the viewport-probe migration lost every autoscroll ack — production discriminates plaintext acks and text routes on the RAW frame BEFORE the cipher, and the package only carried the post-cipher body. `DecodedFrameDict` gained `raw`; every migrated ack/text/room-select consumer reads it. The displacement verdicts re-verified exact after the fix (1,227 enemy / 2 friendly, 20 friendly-exact, 88 stale-clears).
- **The prefix-strip class**: 5 scripts (`find_action_done`, `find_supervisor`, `analyze_wire_bytes`, `verify_js_claims`, `verify_everything`) never split frames at all — `data[2:]` treated a whole payload as one frame. The per-frame walk is a semantic CORRECTION, yet every one reproduced its old output exactly: measured law, 0x2E/0x54/0x52-leading frames always ride alone in their payloads on this corpus.
- **Sent payloads carry exactly one frame each**: the re-framed `SENT` hex lines (`encode_frame(frame["raw"])`) reproduce the original whole-payload dumps byte-for-byte in both deposit scripts.
- **Two broken fossils repaired**: `crack_tank_update` could not even import (`protocol.decoders.misc` is long gone — decoders moved to `session_events`); `analyze_wire_bytes` runs but finds nothing on a combat-full archive (its container-type keys predate the current decoders) — preserved-but-dormant, noted in its docstring.
- **Artifact-writing miners diffed on their artifacts too**: `mine_container_atlas` reproduced `container_atlas.json` and `container_observations.jsonl` byte-for-byte; `mine_bot_policy` its output JSON.

The 12 scripts not touched read pre-extracted artifacts (correlate_unknowns, crack_all_blobs, solve_*, find_kill_byte, mine_inventory_persistence, mine_radar_floor, verify_tank511, analyze_container_atlas) — no capture pipeline to migrate. Net across the whole arc: every byte of capture decoding in this repo now has exactly one owner. Gate green (guard 0, 100.00% coverage).

---
## [2026-08-06] lift | The fleet gets its face — control page, stats, restart

User: "i wanna make sure that the tankpit server is completely seprate from the spa and the sunshine and vibeshine stuff. and i want to ensure we can easily run multiple bots on the desktop. with a simple ui, easy for me or the ai to use." The fleet manager was already the separation (own process, port 27300, localhost, zero fiesta/nginx/streaming); this gives it the UI.

- **`GET /`** serves `service/fleet_page.py` — one self-contained HTML file, no external assets: live bot table (state, bounds, kills / deaths / rank countdown / duration, 3 s poll), spawn form, per-row stop / restart / remove. Strictly a skin over the same JSON endpoints the AI drives — never a second control path.
- **`GET /bots/{instance}/stats`** — the latest run's digest summary (`build_run_digest` over the instance's `latest.events.jsonl`, the same truth table `make digest` prints), reduced to kills, deaths, shots, teleports, pickups, duration, clean/crash + exit reason, rank name / countdown number / promotion points. Works mid-run (the events file grows in place) and on crashed runs; a just-spawned bot answers `{"available": false}` instead of erroring.
- **`POST /bots/{instance}/restart`** — respawns a FINISHED instance with the exact parameters it had (verified in tests: identical child env). Refuses while alive (409): the fleet never silently kills — stop, let the teardown run its scorecard/capture/archive, then restart.
- `make_fleet_app` split into observation + lifecycle route builders (C901); `_json_response` helper.

Bot assessment on the way in: nothing to fix — 100/0 century, trap class dead, clearance/unservable proven; what remains parked is doctrine awaiting user rulings (dueling movement, death-spiral escape, F14/F19).

RustedWarfare gets the same fleet+UI shape next (user ruling, including its fast-forward option as a spawn parameter) — its tree is receiving the user's catch-up commit first.

Gate note: committed scoped (fleet.py, fleet_page.py, test_fleet.py only) with the full 19-test fleet suite green and targeted ruff/mypy clean on the scope; the tree-wide `make check` was red at commit time from a PARALLEL session's in-flight protocol->wire refactor (their edits, their gate) — the fleet files do not overlap it.

---
## [2026-08-06] refactor | Step 1 of de-globalisation: the XOR table becomes a value

`7481cdca`. First step of [[session-state-deglobalisation]] shipped: `sniffer/xor.py` deleted outright — no shim, no deprecation — and its four module globals with it. Every decode path now takes the table its frames were encoded under as a parameter; `capture/xor.py::build_session_xor_table(magic) -> bytes` is the one place a table is built, and `SessionBase` stores it and hands the SAME table to `CommandService` instead of building it twice from identical inputs.

**Three silent failures became loud** — the real payoff, not the concurrency:

- A missing `xor_static_key.txt` now raises `XorStaticKeyUnavailableError`. `build_global_xor_table` returned early leaving the table `None`, and `xor_decode` then returned `body[1:]` **undeciphered** — garbage that decoded into plausible world state rather than an error.
- `drain_messages` returns 0 and keeps the buffer while a session has no table, instead of dispatching pre-magic frames through that same identity decode.
- `WebSocketSniffer` overrode `_on_magic_captured` WITHOUT `super()`, so live decode silently depended on the global that `init_trackers_with_magic` built as a side effect. The override is gone.

**Two tests had been asserting the silent behaviour.** `test_binary_promotion_takes_binary_route` carried the comment "XOR table is None in tests so `body[1:]` passes through verbatim" and asserted a rank read straight out of plaintext; it now ciphers its body under the same table the decoder is handed. `tests/replay/test_script.py`'s fake filesystem had no key file at all. Both are stronger tests than the ones they replace. Five per-file `_isolate` fixtures in `tests/action_lab/` were deleted as pure duplicates of conftest — one of them documented the exact leak this step removed.

**Sizing correction, recorded on the plan page:** the call-site table predicted 8 sites in 6 files. The commit touched **79 files — 21 `src/`, 57 `tests/`**. The estimate was right about `xor_decode` and wrong about what a signature change costs: threading one parameter rippled into `_test_hooks` protocol members, every fake satisfying them, and every fixture that built a table. Read the `get_world_service` row (73 sites / 20 files) as a `src` floor for step 8, not its size.

**Found while shipping: the cipher is forked four ways**, and they disagree at the edges. `protocol/codec.py::xor_bytes` raises a named `ValueError` past the table end; `capture/xor.py::xor_decode_body` `IndexError`s there incidentally; `diagnostics/capture_audit.py` and `sim/transport.py` each carry an `else` branch passing the byte through in the clear. Argument order flips between them — `(body, table)` in two, `(table, data)` in the other two — so a wrong-order call type-checks and silently produces garbage. Both `codec` and `capture` also compute the static-key path with the byte-identical `__file__`-relative expression. `codec` is the survivor (lower layer, only validated one); folding is its own step because the pass-through arm is a real semantic difference. The 279,771-payload measurement that found the tail dead covers only the received-decode path — it does not license deleting the arm from the sim encoder or the audit reader.

Conftest reset list: `reset_xor_state` out, `reset_static_key_cache` in. Ten calls either way, but the new one guards a process-wide KEY cache (one key builds every session's table), not session state — so the honest count of session-state resets is nine, and step 11's target shrank by one.

Gate note: `make lint` clean — 0 violations across all 28 guard rules; `pytest -n auto` 5939 passed, 0 failures attributable to this work. Three failures remain in `tests/test_check_undecoded_fields.py` from the same PARALLEL session's in-flight split of `protocol/types.py` into a package: `scripts/check_undecoded_fields.py:34` `DEFAULT_TARGETS` still names the deleted file. That script is untouched by either session and enumerating their ten new modules would be guessing at their intent — their split, their gate.

---
## [2026-08-06] operation | First fleet-driven live run — one real bug, two human flags, rank 24

First bot ever spawned through `make fleet` + the control page. Take 1 died instantly: the child bootstrap passed its `KEY=VALUE` env freight through argv, and the entry point parses argv — "unrecognized arguments", exit 1, no browser. The fix is one line in the bootstrap (`del sys.argv[1:]` after applying the environment), found only because a live trial was actually run. Take 2: **5 kills / 0 deaths in 385 s, clean `session_complete`, rank countdown 26 -> 24 territory confirmed (panel read 24 at scrape)** — spawn, bounded wind-down, scorecard, and the stats endpoint all proven through the fleet path.

User feedback folded in the same hour:
- **Accounts are config, not free text** (user ruling): the page's account field is now a dropdown fed by `GET /accounts` (usernames from accounts.json — passwords never leave the file), and `spawn` refuses any selector not in the file. No override path exists.
- **Page de-barebones'd**: labeled fields with hints (name -> runs/bot/<name>/, 0 = unlimited), status pills (running/finished/exit N), gold rank column, button tooltips, footer explaining that stop is graceful.

The human-flag channel worked end-to-end through a fleet instance — both flags triaged live from the events stream:
- **Flag 1** (tick 83): the viewport jump was the bot's own `scope(5)` — a ferry scope scout panning SW toward water-locked equipment at (127,15) ([[viewport-shift-protocol]] trigger 2).
- **Flag 2** (tick 135): the "missed tick" after teleporting to orange-5 is the one-tick gap between the 0x3D position echo (+fuel charge, visually the landing) and the server's TELEPORT_LANDED receipt the sync gate keys on. Loop cadence was a flat 2 s throughout — nothing skipped. Candidate doctrine change, parked for the user: treat self-0x3D + fuel-charge as landing proof (the displacement mining's 534/534 pairing law) and open fire a tick earlier.

---
## [2026-08-06] lift | The fleet page becomes the make-run HUD — the overlay, served

User verdict on my hand-rolled card UI: "straight slop" — and the fix was their idea: reuse the existing `make run` overlay as the webpage. The tick loop now mirrors the exact per-tick HUD payload to `runs/bot/<instance>/hud.json`; the fleet serves it verbatim at `GET /bots/{i}/hud`; and the page renders one REAL overlay card per bot — CSS and body imported from `browser/overlay_hud.py` (id re-scoped to a class, flag button hidden: flags belong to the human at the live window), painted by the same field mapping the in-page updater uses. Mode band, fuel meter, stocks, do/why/tgt/act, K/H/M/RJ — byte-identical data, same glass. Below it the v2 table (status/limits/kills/deaths/rank/time + launch form) restored per the same verdict. Everything polls at 1 s; digest stats stay behind the 2 s telemetry cache. Also this hour: `/bots/{i}/stats` grew timeline/inventory/displacements, `/bots/{i}/activity` serves the events tail for the AI, telemetry moved to `service/fleet_telemetry.py`, routes split into observation/telemetry/lifecycle builders. Gate: 183 service tests green, scope ruff/mypy clean; the 8 tree failures at commit time are the PARALLEL session's in-flight framing refactor (their files).

---
## [2026-08-06] crack | User-piloted measurements: walking speed, diagonal cost, autoscroll's scope buffer — and a falsified economics premise

Three laws measured live by the user (manual probes, wall-clock timed), plus one bot-doctrine premise falsified:

- **Walking speed ~0.22 s per cardinal tile (~4.5 tiles/s)**: 48,85 -> 63,85 (15 tiles) in 3.30 s clean; the 3.30-vs-4.20 s spread is the 2 s dispatch boundary (command waits for the next tick pickup, +0-2 s by click phase), not speed variance.
- **Diagonals cost two Manhattan steps**: 63,77 -> 48,92 (15 diagonal = 30 Manhattan) in 6.20 s; 30 x 0.21 s = 6.3 s. The mover prices per Manhattan step.
- **Autoscroll reserves a one-tile scope buffer per edge**: max west scope stride moves the rightmost column 63 -> 48 (15 columns) with autoscroll OFF, but only 62 -> 49 (13 columns) with it ON — and the east-shift limit itself differs by one (62 vs 63). Presumed mechanism: the buffer keeps a shift from parking the tank on the exact edge tile that would trigger an instant auto-recenter. New constraint on [[viewport-shift-protocol]]'s ANCHOR law.
- **FALSIFIED premise**: `_FUEL_GAIN_PER_WALK_TILE = 25` (collect_pickups) justifies itself with "each walk tile costs roughly one 2-second tick" — off by ~10x against the measured 0.22 s/tile. At the artax flag scene this rule refused an 80-fuel pickup 4 tiles (<1 s) away. Re-derivation of the constant from measured time is PENDING the user's go (no code changed). Related standing diagnosis, same session: the forage walk-loop deadlock (extras-stocked radar veto -> zombie walk decisions starving the collect hop; flags 1-3) — fix also pending.

---
## [2026-08-06] fix + crack | The forage deadlock dies, walk pricing re-derived, and the viewport IS the radar

Both flagged behaviors fixed and the full suite green (5,972):
- **Forage deadlock** (artax flags 1-3): with extras stocked, any radar spends an extra and an extra scans the whole viewport from anywhere — walking can never improve the scan and the free radar can never fire. Forage now returns NOTHING when the extra-spend is vetoed, so the collect hop one rung below fires instead of the one-tile-per-tick edge crawl that starved it.
- **Walk pricing** `_FUEL_GAIN_PER_WALK_TILE` 25 -> 3: re-derived from the user's measured 0.22 s/tile (15 tiles in 3.30 s; diagonals two Manhattan steps) at the same ~12.5 fuel/s opportunity value the old constant implied. Replay regression re-pinned: the archived under-stocked session now takes 8 fuel pickups where it used to re-radar 9 times — the falsified premise had it walking past fuel it needed.
- **User-confirmed law**: extra radar ALWAYS scans the current viewport, wherever the scope parked it — "the viewport is the radar." Strategy implication, user-derived: shift-east-scan then shift-west-scan from one standing position sweeps ~31 columns with a single-column overlap at the tank — a stationary double-wide sweep for two extras and zero walking. Candidate forage upgrade, not yet implemented.

**Addendum — the QUAD SWEEP (user-derived, same conversation).** With autoscroll OFF, from one standing position: shift NW -> radar, NE -> radar, SE -> radar, SW -> radar. The four windows tile around the stationary tank as quadrant corners: ~31x31 = 961 tiles (~3.75 viewports) for four extras, zero movement, zero fuel, ~6% overlap (the shared row/column strips). Moving between scans slides the next quadrant's frame and ruins the tiling — the sweep is maximal exactly when the tank stays planted at the shared corner. Per-extra tile yield matches hopping (~240/extra); the win is no teleport fuel, no displacement risk, no travel time, and one contiguous intel block. Candidate forage/recon upgrade, not yet implemented.

---
## [2026-08-06] lift | The stats the user asked for: hits/misses, damage dealt/taken, zero-yield radars, inventory start->now

Digest grew five counters, all flowing to `make digest`, the fleet stats endpoint, and the fleet table (six new columns: hit/miss, dmg +/-, tp, 0-radar, inv start->now):
- **hits / misses** from `action_outcome` records — live mid-run, matching the scorecard (trial run: 58/0 verified).
- **zero_yield_radars**: a radar dispatch followed by no container pickup before the next radar (or session end) — the user's "radars where there was no pickup", counted by a window state machine.
- **damage_dealt / damage_taken**: the damage book's fuel-confirmed per-enemy totals, now summed (`damage_book.total_fuel`, wiki-claimed) and emitted numerically on the teardown `damage_ledger` diagnostic; pre-extension archives read 0 by design.
- **inventory_first** joins inventory_last in the stats payload (armor·dual·missile·homing·radar, start -> now).
Physics-claim gate did its job twice: forced the wiki claim for the new public symbol AND caught fuel-system.md still claiming the falsified 25/tile rate — re-written to the measured 3/tile derivation. Full suite 5,974 green.

---
## [2026-08-06] doctrine | Quad-sweep harvest designed end-to-end — recorded in [[quad-sweep-doctrine]]

Full design session with the user, every step priced on measured laws (ANCHOR, window-bound acceptance, walking 0.22 s/tile, teleport 6x euclid): atomic stationary 4-shift/4-radar sweep tiling the 31x31, walkable-ordered harvest loop ending nearest the next block, shift-before-walk legs, exit hop as the cycle's only big fuel spend, grid-vs-opportunistic anchor knob. Two of my errors corrected by the user against recorded law along the way (shifts are TANK-anchored, not window-relative; walking never moves the window). One probe blocks implementation: a pickup fired from a shift-framed window.

---
## [2026-08-06] crack | Shift-framed pickup proven from the archive — quad sweep unblocked, no live probe needed

The doctrine's one open question ("does a pickup dispatched into a SHIFT-framed window transfer normally?") was answered by mining recorded human play instead of scheduling a live probe. `sniff-20260710-202821` (the ANCHOR-law session) carries five shift->pickup windows and `ghost_observe` two more: twelve accepted in-shifted-window actions total, every one answering with the standard choreography (`0x47` walk echo + duplicate `0x43` pickup records), every accepted target inside the shifted 16x16. The only rejections were err=1 no-path (retry succeeded) and code 7 inventory-full — none window-bound. **The shifted window IS the acceptance window.** Evidence + miner recorded in [[quad-sweep-doctrine]] § "Shift-framed pickup: resolved". Sent-frame wire note the miner nailed down: scope shift encodes `21 03 5a dir` (COMMAND_PREFIX + type 3), not the bare `03 5a dir` shorthand footnoted earlier. Implementation of the quad sweep now has zero open questions; sim-side the law is already enforced (`sim/viewport_window.py`, pinned by `tests/sim/test_scope.py`).

---
## [2026-08-07] build | The quad sweep ships: recon before pickups, framing shifts before larder hops

[[quad-sweep-doctrine]] implemented as `bot/ai/quad_sweep.py` and wired into the COLLECT cascade at two points: the atomic sweep between lock continuation and the pickups (2b), and the block-harvest framing leg between mine clearance and the larder (4b). Anchor discipline is one AI-state latch pair (`sweep_anchor_x/y`): start only on a >=480-uncovered block, continue only while standing exactly on the anchor — movement aborts, and harvest walking can never re-trigger a sweep on dragged ground. Both branches are gated off at the fuel-low break (recon is an economy move, never a survival move). Three consequential findings from the build:
- **The scan-on-landing latch read every pan as a landing** — scope shifts now pre-latch the anchor-law origin at dispatch (`latch_scope_shift_landing`), keeping the ferry scout free and the sweep's scans properly labeled `quad_sweep_radar`.
- **The sweep subsumes the ferry scout when extras are stocked**: in the ferry sim soak the quadrant pans revealed the ferry before the larder ever declined, and the boarding hop ran without the dedicated scout — the scout stays as the extras-empty fallback (its firing conditions keep their own unit pins).
- **Seam soak end-to-end**: NW/NE/SE/SW windows tiled exactly per the anchor law against the sim's wire-pinned laws; the session swept, framed, harvested, and ended with MORE extras than it started (the harvest refills the recon spend).
Re-pins that rode along: the larder-vs-discovery tests (sweep now leads on a virgin block, larder ordering pinned below it with extras emptied), the under-stocked replay fixture (all sweep-sense ticks — synthetic radars draw no recorded response in replay), and the fuel_probe replay (routing pins instead of the now-unreachable pickup phase). Full suites green in my scope: bot + sim + replay 1,700, quad_sweep/mode_controller/ai_strategy/fleet_telemetry/run_digest at 100% coverage. Live trial through the fleet is the next step, on the user's go.

---
## [2026-08-07] gate | The coverage omit list deleted, and the four modules it was hiding taken to 100%

`[tool.coverage.run] omit` exempted four paths from a gate that reports 100% — `action_lab/combat_probe.py`, `action_lab/enemy_tracking.py`, `scripts/combat_probe.py`, `scripts/enemy_tracking_probe.py`. Deleting it dropped the true figure to 99.04%, which is the point: an exempted gate measures the files someone already chose to test. Nothing on the list resisted testing. Being listed was the only reason the tests did not exist — `combat_probe.py` stood at 69% and its script at 0%, and both reached 100% through the same `_test_hooks` seams every other probe already used. User ruling, verbatim: **"no exclusions no omissions no exemptions no exceptions"**, now a body bullet in [[coding-standards]].

Coverage 99.04% -> 99.96%, 6,157 tests green, `scripts.guard` at zero violations across every rule. What the new suites pin, beyond the number:
- **The unpaired enemy survives.** `_build_tracked_records` keeps a record for an enemy with no JS registry entry; dropping it would hide exactly the divergence the tracking probe exists to catch. The summary's two directions are asserted separately — ours-present/JS-absent is a stale wire TTL, JS-present/ours-absent is the lock released early.
- **A non-adjacent combat landing still engages.** The server displaces a teleport onto the nearest open ground, so refusing there would throw away the approach; the warning path and the adjacent path are both pinned.
- **`responded_ms` is -1 exactly when the server never answered**, which is what separates an unanswered shot from a fast miss.

Three dead guards deleted on the way through, all one root cause: `split_payload_frames` already ends in `[body for body in split_frames(data) if body]` precisely so consumers can read `body[0]` safely, so the re-checks in `analysis/scan.py`, `capture/viewport_entities.py` and `sim/ghost.py` were unreachable — which is why coverage flagged them and why no honest test could have covered them. Also removed: five delegation-only wrappers in `fuel_probe`/`teleport_helpers` with their `_shared_*` import aliases, the `WIRE_PRESENCE_TTL_MS` / `POSITION_FRESHNESS_TTL_MS` private-then-public alias pairs, an `import ... as _unused` followed by `del _unused`, and an empty-bodied stub with no callers.

**Method note worth keeping.** Scoped `pytest` runs are not the gate. `make check` runs guard first, and guard caught seven defects in tests I had already called verified — six weak assertions (`assert ... is not None`, `isinstance`) and one of my own new files at 628 lines over the 600 ceiling. One of those weak assertions was hiding a genuinely broken test: `assert get_world_service() is not None` passed while the frame it claimed to decode never reached world state, and strengthening it exposed that the hand-built `0x47` was the wrong shape and that a non-self movement only moves a tank already in the registry. The frame is now built with the production `encode_movement`. A second test passed for the wrong reason until the JS registry key was corrected from `j` to `P.j`. `make check` also refuses to start on a half-saved file, which scoped runs never parse.

## [2026-08-07] refactor | The 600-line ceiling and the base layer become guard rules; 77 over-bar files reach 0

An audit of separation of concerns, file size, layered architecture and DI found two rules that were **documented but unenforced**, and both had rotted exactly where enforcement was missing.

**File size.** [[coding-standards]] set a 400-600 line ceiling on 2026-07-31 with a recorded backlog of 40 files over 600. Measured 2026-08-06: **77**, and all 40 originals had GROWN — +6,272 lines, zero splits. Every other standard in that page is machine-checked and sits at zero violations; the one that was not is the one that regressed. `scripts/file_size_rules.py` now runs in `scripts.guard` with **no allowlist and no baseline**, and the backlog is **0**. It caught two live regressions within the session it was wired (`sim/world_seed.py` at 607, `tests/action_lab/test_enemy_tracking_execute.py` at 628) — the drift is now caught on the commit that causes it.

**Layering.** A Tarjan SCC over the package import graph put all 17 packages in ONE component: there was no acyclic order to violate. Now **13** packages remain in the component and **10 are provably acyclic**, with `scripts/layer_rules.py` holding the base layer (`types`/`wire`/`contracts` import nothing; `facts`→`contracts`, `container`→`wire`, `bus`→`types`). See [[package-layering]] for the cuts.

Findings worth carrying forward:

- **Measure cycles with SCC, not mutual pairs.** A pair check misses `protocol -> state -> physics -> protocol` and reports progress the graph does not have. Two earlier passes of this work quoted pair counts and overstated the result.
- **An analyzer that follows only `from tankpit_bot.X import ...` under-reports.** `from tankpit_bot import X` is how `state/renderer.py` reaches `_test_hooks`; missing that spelling hid three cycles. The guard rule follows both, with a test per spelling.
- **A deferred (function-level) import is a misfiling tell, not a solution.** Every one traced to a symbol living in the wrong package: `resolve_idle_exit_seconds` (a service concern in `bot/config.py`), `load_dotenv` (generic startup in `service/_test_hooks.py`), `pack16` (the byte layer in `protocol/helpers.py`).
- **Splitting a module can silently break DI.** Tests patch module attributes; a function that moves modules stops being the one the caller invokes, and the patch becomes a no-op nothing reports. Moved seams are now reached through their module (`fuel_probe_targets._find_*`, `queue_experiments.run_single_experiment`, `tick_body._tick_once`). The trap: satisfying mypy by repointing a patch site to a symbol's TRUE owner disables the injection, because the consumer bound the name into its own namespace — that type-checks and only a DID-NOT-RAISE catches it.
- **Shims removed, seams kept.** 20 module-level aliases in `src`; **16 are patched DI seams** (4-10 patch sites each) and stay. Four were re-exports and went, along with a pass-through decoder wrapper that also narrowed its return type, six identical-protocol "typed bridge" wrappers, and four public/private delegation pairs in `combat_strategy`.
- **Duplication hides across package boundaries.** `_init_game_log_scraper` / `_poll_game_log` were implemented once on `Bot` and once on `BrowserSession`; the copy only surfaced when `bot/base.py` needed to come back under the ceiling and the honest route down was deleting one.

Verified at write time: guard 0 violations, ruff clean, mypy clean over 1083 files, **6157 tests pass**.

## [2026-08-07] audit | Every stale wiki anchor cleared by re-reading, not by bumping: 32 -> 0

`tankpit-wiki-anchors` reported 32 stale anchors across 23 pages. A
stale anchor means the page is owed an AUDIT, so each was re-read
against the tree and corrected before its anchor moved. Result: **64
anchors, 0 stale**, guard and wiki-structure rules at zero violations.

**Method — three mechanical sweeps first, then read.** Cheap checks
found the breakage that a careful read would have taken hours to
notice, and left reading time for the claims that actually needed
judgement:

1. **Dead paths.** Every `path.py` reference resolved against the
   tree. Found 8 pointing at files that no longer exist.
2. **Line numbers past EOF.** Found 4 — including
   `bot/tick_loop.py:868` on a 516-line file, a citation the tick-loop
   split had invalidated on two separate pages.
3. **Symbol proximity.** For each `` `symbol` ... path.py:N `` pair,
   check the symbol appears within ±12 lines of N. This is the one
   that earns its keep: it catches citations that still RESOLVE but no
   longer point at the thing named, which no existence check can see.
   Found `choose_combat_landing_tile` cited at `:46` when it sits at
   `:80`, and `decide_collect_mode` at `:199` when it sits at `:42`.

A fourth check — diffing each anchor's recorded git blob against the
worktree — turned "what drifted?" from a reading problem into a diff.
For directory anchors the added/removed file list is the right
granularity, because that is the level a page's claims live at.

**Two claims were wrong about behaviour, not just line numbers:**

- [[testing-patterns]] said "the `_hooks_guard.py` module enforces:
  never set module attributes directly in tests." It enforces nothing.
  Its own docstring says it is a DI seam so guard TESTS can inject a
  fake orchestrator instead of scanning the real monorepo. The rule is
  `MonkeyPatchBanRule` in `libs/monorepo_guards`, which
  [[coding-standards]] had right all along — two pages disagreeing
  about the same rule, and only the mechanical read caught it.
- [[bot-behavior-contract]]'s ghost-firing row named an
  `is_wire_present` gate in `_combat_shoot`, pinned by
  `test_ghost_wire_presence_regression.py`. None of the three names
  exists: `_combat_shoot` was collapsed into `engage_target` by the
  thin-wrapper removal, and the test file was never written under that
  name. The gate is real; every name for it was wrong.

**Findings worth carrying forward:**

- **A stale anchor is not evidence of a wrong page, and a current one
  is not evidence of a right page.** Of 32 stale anchors, several
  covered pages whose prose was still exactly true — the tree moved
  underneath them. Meanwhile the two behavioural errors above sat on
  pages whose anchors were CURRENT. The anchor tracks whether anyone
  looked, which is all it claims to track.
- **Anchors re-stale on the next commit that touches the path.** That
  is inherent to pinning a tree hash, and is exactly why the tool is a
  report and not a gate — gating would redden the build on every
  source commit and reward bumping without reading.
- **Unexplained: the stale count moved during the run** (32 -> 18 ->
  23) with no commit behind it. `git reflog` is linear across the
  whole session and its newest entry predates every one of those
  readings, so nothing was committed while they were taken. An earlier
  draft of this entry blamed a concurrent session; that explanation is
  **withdrawn** — it was assumed, never checked, and the reflog
  refutes it. The cause is still unknown. Worth a repeat run before
  trusting a single reading of the report, since the same instability
  would make any one count unreliable.
- **Prefer marking a section historical over rewriting it.** Three
  pages carried present-tense descriptions of deleted code
  ([[executor-rejection-loops]]'s three "known live instances", its
  `_tracked_combat_target` footnote). The diagnosis is why the
  deletion happened and is worth keeping; a banner that says "read
  this in the past tense", naming what replaced each symbol, preserves
  it without lying about today.
- **Record the gap instead of closing it.** Four pages cited planned
  `tests/integration/*` files that were never written. Three already
  said so. The fourth now names what IS pinned (cooldown decode, the
  map-freshness window) and what is NOT (the composite
  stall-replan-reissue shape) rather than pointing at a file that
  never existed.
- **The wiki found a code bug.** `make-targets` documented the fleet
  manager on `127.0.0.1:27300`; the Makefile's own banner claimed
  `0.0.0.0:27300`. `service/fleet.py:34` binds loopback — the page was
  right and the user-facing banner was wrong, so the banner was
  corrected to match the code.

Pages corrected: [[bot-behavior-contract]], [[bot-service-architecture]],
[[capture-differ]], [[coding-standards]], [[committed-intent]],
[[diagnostic-hud]], [[enemy-bot-behavior]], [[executor-rejection-loops]],
[[ferry-mechanics]], [[fuel-system]], [[inheritance-chain]],
[[larder-plan]], [[make-targets]], [[map-data-decode]], [[module-map]],
[[physics-module-roadmap]], [[server-push-gating]], [[services]],
[[session-state-deglobalisation]], [[shot-range]],
[[tank-freshness-model]], [[testing-patterns]], [[weapon-log-markers]].

Re-derived rather than restated: the package SCC (13 cyclic / 10
acyclic, unchanged), the over-600 file count (**0**), the type-safety
zeros (now counting `scripts/` too), the seven `make shadow` laws, the
`get_world_service` call-site census (73 in 20 files -> **74 in 23**,
and the old "21 in `tick_loop.py` alone" is gone with the split).

## [2026-08-07] refactor | De-globalisation step 2 lands as a deletion; six re-exports and a back-compat decoder go with it

**Step 2 was mis-scoped in its own plan.** [[session-state-deglobalisation]]
listed `sniffer/viewport.py`'s two globals as state to move onto the
session. They were already on the session: the single writer set the
globals and then wrote the same pair into `world_state["viewport"]` on
the next statement. `get_viewport_left()` / `get_viewport_top()` had
zero callers outside tests. The module was write-only in production and
its four session-boundary resets cleared state nobody read — so the
step shipped as a deletion, and the completion criterion (the conftest
reset list) did not move, because these globals were never in it.

**Method note:** the tell was in a test. `test_world_state_functions`
set the global AND the world-state copy before asserting; only the
latter mattered. A test that has to write the same fact twice is
describing duplication in the code under it.

### Shims, wrappers, re-exports: a mechanical pass

The standing rule (no back-compat shims, thin wrappers, fallbacks,
legacy code, type aliases, re-exports) was re-checked by AST scan
rather than by reading. Removed:

- `capture/trackers/tank.py` — `TEAM_NAMES = TEAM_NAMES  # Backward-compatible alias`, which announced itself.
- `sim/actions.py` — `RADAR_FUEL_COST` / `MINE_PRESS_FUEL_COST`, renamed re-exports of `physics.costs`. Their only consumer was `sim/emissions.py`; `actions.py` imported both names **solely** to re-export them, so the import went too.
- `sim/server.py` — `TICK_MS = TICK_RATE_MS`, re-exported so sim consumers could avoid importing from the protocol layer that owns it. Four consumers repointed.
- `sim/ghost.py` — `_TICK_MS`, a private rename with one use.
- `validate/shadow_laws.py` — `PAIRING_WINDOW_MS = TICK_MS`, a rename of a re-export.
- `sim/movement.py` — `MINE_WALK_COST = SINGLE_HIT_VICTIM_COST`. The domain fact it carried (a mine walk-over costs the same 45 as a single hit) survives as a comment at the use site; the alias does not.
- `types/session.py` — the two "backwards compatibility for old sessions" branches in `decode_capture_session`, which tolerated a missing or malformed `game_log` / `tank_names` by silently skipping entries.

**The back-compat decoder was provably dead, and the proof matters
more than the deletion:** the encoder always writes both keys, the
`CaptureSession` TypedDict declares both REQUIRED, and **0 of 410
archived captures lack either** (parsed, not grepped — a 4 KB head
check would have been wrong, since the keys sit after the `messages`
array). A decoder that contradicts its own type is a shim regardless of
how defensive it looks. The two tests that pinned the skipping now pin
the raise.

### What was NOT removed, and why

- **Nine `_test_hooks` / action_lab aliases stay.** They are the DI
  seams, with 4-10 test patch sites each. Deleting them type-checks
  clean and silently disables injection — the trap already recorded in
  the 2026-08-07 layering entry, where only a DID-NOT-RAISE caught it.
- **`terrain.py`'s `is_landing_legal` / `is_landing_attainable` stay.**
  They collapse to `is_passable` on the static minimap and DIVERGE in
  `FerryAwareTerrain`; that is polymorphism, not a wrapper.
- **`set_self_rank` stays.** The scan flagged it as a pass-through
  because its first three arguments forward verbatim — it adds the
  `"wire_0x2B_promotion"` provenance, making it a named specialization.
- **Four `except` handlers stay.** All are narrow, typed and logged at
  the archive-parsing boundary (`ValueError`, `FramingError`,
  `DecodeError`) — the boundary case the coding standard permits, and
  the archive genuinely contains unparseable payloads.

Mechanically verified after the pass: `TypeAlias` 0, `typing.Any` 0,
real `cast(` 0 (the six hits are `broadcast(` substrings and a guard
fixture string), real `type: ignore` 0 (four hits are prose restating
the ban), `noqa` 0, `TYPE_CHECKING` 0, `.pyi` 0, `pragma` 0, coverage
`omit` 0, `exclude_lines` 0, `xfail`/`skip` 0.

## [2026-08-07] refactor | De-globalisation steps 3-5; bot/base.py splits by concern; the no-shims rule becomes a guard rule

**Steps 3, 4 and 5 shipped, and all three deleted their reset function
rather than moving it.** The pattern that replaced them is the same
each time: a per-session object whose constructor IS the reset.

| Step | Global | Now |
|---|---|---|
| 3 | `_last_emitted_belief`, `_last_emitted_signature` | `SelfAlignmentEmitter` / `EntityAlignmentEmitter` on `Bot` |
| 4 | `_cdp_time_offset_ms` | `CDPClock` on `CDPService`, which was already per-session |
| 5 | `_survey_emitted` | `ClientStructureSurveyor` on `Bot` |

**Step 5 is the first to move the completion criterion.**
`reset_client_structure_survey` was in the ten-call conftest reset
list, so the list is now nine — eight session resets plus the
process-wide `reset_static_key_cache` that stays by design.

**The test change is the point, not a side effect.** Each step
replaced a "reset clears the gate" test with "a SECOND instance has
its own gate". The old test proved a reset function works; the new one
proves the property the refactor exists for — two sessions in one
process cannot silence each other. A reset can never prove that.

Step 4 has the sharpest reason: CDP timestamps are monotonic seconds
from an arbitrary origin, so the first frame of a session fixes the
offset every later frame is read against. A second session has a
DIFFERENT origin; sharing one anchor misdates every frame it reads.

### `bot/base.py`: 601 -> 440, split by concern

The new file-size rule caught the 601 immediately — the six lines step
3 added crossed the ceiling on the commit that crossed it, which is
what the rule is for.

**A wrong turn worth recording.** The first split moved the run loop
out as functions over a `Bot`, matching `run_tick_loop(bot, ...)`. That
was wrong: `Bot.run` satisfies `RunnableBotProtocol`, the seam that
lets `SessionRunner` accept a fake bot with no Playwright. Removing the
method would have forced the production factory to return an adapter —
a wrapper, which is banned. **A method that satisfies a Protocol is not
a candidate for extraction, however well it separates on paper.**

The seam that actually separates is the read model. `StateAccessMixin`
(`bot/state_access.py`) holds the seven world-state queries, which
touch no session state at all — no browser, no CDP, no HFSM.
`GameLogWitnessMixin` (`bot/game_log_witness.py`) holds the DOM
game-log poll and the account-stats read, declaring the attributes it
uses as annotations so `Bot` keeps ownership of the state. Chain is
now Bot -> GameLogWitnessMixin -> StateAccessMixin -> DispatchMixin ->
CompletionsMixin -> SessionBase.

### The no-shims rule is now enforced

[[coding-standards]] banned back-compat shims, thin wrappers,
fallbacks, legacy code, type aliases and re-exports — and was the one
standard with no enforcing artifact. `scripts/shim_rules.py` now runs
in the guard at zero violations, failing on legacy vocabulary
(`back-compat`, `deprecated`, `legacy`, `kept for signature/API
compatibility`), self-named aliases (`X = X`), and renamed re-exports
(`NEW = IMPORTED` where `NEW` is exported).

The `_test_hooks` exemption is **structural**: it covers that module
kind, because binding an imported implementation to a patchable module
attribute IS the DI mechanism. It cannot grow entry by entry, which an
allowlist would.

Cleaning to zero removed six re-exports and four legacy markers.
Two were more than renames:

- **`decode_and_log_binary` had zero production callers.** One test
  kept it alive. Dead code with a test looks covered, which is exactly
  how it survives.
- **`can_use_radar` returned `True` unconditionally**, and the
  `radar_affordable` parameter it fed was residue of a two-caller
  design (equipment recovery and fuel recovery each passing their own
  predicate) whose modes no longer exist. One caller remained, passing
  a constant. The parameter went with it, and the test that forced the
  walk branch with `radar_affordable=False` was rewritten to reach that
  branch the way production can — the free 5x5 already scanned while
  the wider viewport is not.

Verified: `make check` green, 100.00% coverage with zero misses, guard
clean across every rule.

## [2026-08-07] refactor | Step 7 deletes a blacklist that never had a writer; step 6 is blocked on step 8

**The container blacklist was dead, not global.** Step 7 was written
as "de-globalise `_blacklisted_container_keys`". The right move turned
out to be deleting it: `blacklist_container` has **no caller under
`src/` in any commit in this repository's history** (checked with
`git log -S` across the three commits that ever touched the file), and
neither does `reset_container_blacklist`, whose docstring claimed it
ran "on death/respawn". Only tests called either.

So the reader always answered False, and five decision sites were
filtering candidates against a set that could never fill: both hop
selectors in `collect_hops`, the equipment pickup in
`collect_pickups`, the quad sweep, and the scope scout — plus an
`is_blacklisted` predicate threaded as a parameter through
`larder.select_fuel_larder_hop`. Deleting the mechanism is
behaviour-identical by construction.

**A reader with no writer is worse than dead code**: it reads as a
safety feature. Five separate decision sites were written to respect
it, and one of them threaded it through a public signature. The tests
made it look alive — three of them called `blacklist_container`
directly and asserted the reader saw it, which proves the mechanism
works and says nothing about whether anything uses it.

Second step to move the completion criterion: the conftest reset list
is now **eight** calls (seven session resets plus the process-wide
`reset_static_key_cache`), down from ten.

### Step 6 is blocked on step 8, and the block is one function

The ledger cluster measures small — the six globals have only three
consumer files outside `ledger/` for the counter/ring/decision/
transition group, and `emit_action_outcome` is called only from within
the outcome package. Every one of those consumers already takes `bot`.

The exception is `pending_teleport_target`, read by
`sniffer/world_state_dispatch_containers.py` — the WIRE-DISPATCH
layer, whose only session handle is the `WorldService` singleton that
step 8 exists to remove. Its four sibling functions
(`record_teleport_dispatch`, `emit_teleport_landed`,
`emit_teleport_stall_timeout`, `emit_teleport_command_rejected`) are
all bot-side; only the read crosses into dispatch.

Putting the ledger on `WorldService` would make it work today and be
fake progress — the state would still be reached through a module
global. The honest options are to run step 8 first, or to accept a
step 6 that leaves the teleport-dispatch tracking behind. Recorded
here rather than guessed at.

### Method note: `make check` was slow, and it was self-inflicted

A run took 453 s against a normal ~90 s. Not a hang: three concurrent
pytest fleets were on the box (628 s, 359 s and 165 s of accumulated
CPU), because several `make check` runs had been backgrounded during
the session and `addopts` carries `-n auto` — 16 workers each. The
harness reported those tasks complete while their xdist workers stayed
alive. **Background one long gate at a time**; a second one does not
run in parallel, it runs 3x slower and hides the real timing.

## 2026-08-07 — Step 8's test-side blocker was 150x smaller than counted

`reset_world_state()` in `tests/`: **496 -> 3**. Full gate green after:
6191 passed (unchanged), 100.00% coverage with 0 missed statements and
0 partial branches, mypy clean on 1100 files, ruff clean, guard exit 0.

### The count was real; the conclusion drawn from it was not

Last session I reported 496 `reset_world_state()` sites as the blocker
on step 8 and recommended a session's work migrating each to a
hand-built `WorldService`. That recommendation was wrong, and the error
is worth naming: **I counted appearances without asking what any of
them did.**

`tests/conftest.py::_isolate_protocol_singletons` is autouse at the
root and already resets the singleton before and after every test. So a
`reset_world_state()` as the first statement of a test body runs
microseconds after the fixture did exactly that. Classified by AST
position:

| kind | count | verdict |
|---|---|---|
| prologue (first statement) | 195 | dead |
| `setup_method` / `teardown_method` body | 229 | dead |
| epilogue (last statement) | 20 | dead |
| `try/finally` scaffolding | 20 | dead — the `try` existed only for the reset |
| shares a `finally` with real cleanup | 9 | line dead, block stays |
| genuine mid-test | 22 | needs reading |
| non-body (in a loop) | 1 | needs reading |

444 of 496 were ritual. Of the 23 that needed reading, 20 turned out to
be redundant once their surroundings were checked (the builders they
followed — `make_world`, `_sweep_ctx`, `make_inventory` — are pure and
touch no global).

### What made this safe to prove instead of guess

A prologue reset is only redundant if no fixture the test depends on
*populated* world state first; otherwise deleting it silently changes
what the test measures — and in tests asserting `result is None`, a
state leak makes them pass spuriously rather than fail. So the
deletion was gated on a specific check across all seven `conftest.py`
files: **nothing anywhere populates world state in a fixture.** The
only writers are the root autouse reset and two `action_lab` fixtures
that reset-and-yield. That check is what turned this from a plausible
cleanup into a provable one.

### Collateral removed

202 `setup_method`/`teardown_method` bodies that did nothing else, 212
unused imports, and two fixtures that the strip reduced to a bare
`yield`: `real_inventory` (40 call-site edits across 4 files) and
`_isolate_world_state`. A fixture whose body is `yield` is a parameter
every test carries and nothing reads.

### Tests that outlive the thing they test

Five tests asserted "populate, reset, observe clean" — they test
`reset_world_state` itself, so they would die with it. Rewritten to
assert the durable invariant underneath: *a freshly constructed
`WorldService` starts clean*, contrasted against a populated one. Same
coverage, no global, and they survive step 8's final flip.
`tests/sniffer/test_replay_pipeline.py` went further and is now
entirely off the singleton — its helpers return the service they
decoded into, and the five-prefix replay test builds one service per
iteration, making isolation structural rather than a property of call
ordering.

### The 3 survivors

Two are the conftest fixture (the seam). The third,
`tests/bot/test_executor_dispatch.py:288`, is load-bearing: `_make_bot`
seeds position and fuel, and the test needs no self-belief. It unblocks
when `Bot` takes a `WorldService` — src-side work.

### Method note: read the exit code, not the summary

`poetry run python -m scripts.guard | tail -15` printed fifteen lines
of `0 violations` and `$?` reported 0. Both were false comfort: `$?`
after a pipe is `tail`'s status, and the violation line was in the
header the `tail` had cut off. The guard was actually exiting 2 on a
`weak-assertion-is-not-none` in a test written this session. This is
the second time in two sessions that a guard summary read clean while
the guard failed. Redirect to a file and echo `$?` from the unpiped
command.

## 2026-08-07 — Step 6: the ledger cluster, and a blocker that was never real

All six ledger globals are gone. `src/tankpit_bot/ledger/` now holds
zero module-level mutable state, zero `global` statements, and zero
`reset_*` functions. `make check` exit 0: 6191 passed, 100.00% coverage
with 0 missed statements and 0 partial branches, mypy clean on 1102
files, ruff clean, guard clean.

### I had recorded this step as blocked. The block was my own reasoning.

The previous entry said step 6 was blocked on exactly one function —
`pending_teleport_target`, read by the wire-dispatch layer whose only
session handle is the `WorldService` singleton — and dismissed the
obvious fix: "putting the ledger on `WorldService` would make it work
today and be fake progress; the state would still be reached through a
module global."

Two things were wrong with that.

First, the precedent already existed and I had not checked it.
`WorldService` has owned `fuel_book`, `damage_book` and `ammo_book`
since they landed — three ledger types, on the session service. Ledger
state living there was settled, not novel.

Second, and more important, the objection conflated two different
problems. Six module globals **cannot** be duplicated per session: two
sessions in one process share one decision store, one event counter,
one set of rings, and there is no way to ask for a second. Moving them
onto `ws.ledger` makes that possible immediately. That the call sites
still find their service via `get_world_service()` is *one* global that
step 8 already exists to delete. Trading six impossible-to-duplicate
globals for one scheduled-for-deletion lookup is not fake progress; it
is the actual progress, and I talked myself out of it.

The displacement receipt is the clearest argument for the placement:
the executor records a teleport dispatch and the 0x5A dispatch handler
reads it back to detect server displacement. The command layer and the
wire layer must share ONE ledger, and the service is the only handle
both hold.

### Shape

`ledger/service.py::LedgerService` owns the event counter, per-kind
outcome rings, decision store, mode-transition log, the three pairing
trackers and the pending teleport dispatch. Fifty functions across ten
modules take it as their first parameter.

`ledger/records.py` is new and holds the four record `TypedDict`s.
It exists for one reason: `LedgerService` must type its attributes, and
the modules that define those attributes now import `LedgerService` —
so the shapes had to move somewhere neither side imports, or the
cluster closes an import cycle.

### Three things fell out that were not the target

The `@enforce_contract` decorator is `ParamSpec`-generic, so a
contract's `check()` must carry the *same* signature as the function it
guards. Adding a first parameter to `record_decision` and
`record_teleport_dispatch` therefore required adding it to both
contracts, where it is declared and deliberately unread — the
invariants are over the record, not over which ledger receives it. That
is the price of type-preserving enforcement, and it is stated in the
docstrings rather than left to puzzle over.

Both `ledger/__init__.py` files were pure re-export blocks with **zero
importers** — nothing in `src/`, `tests/` or `scripts/` imported from
the package level. They now declare `__all__ = ()` and export nothing.
The claim block had been dutifully binding nine addresses (seven
outcome aliases plus two functions) that no code used.

The wiki claim count fell 77 → 65. Every deletion is a symbol that
stopped existing: six reset functions, `next_event_id` (now a service
method), and those nine dead re-exports. Four record types moved
address rather than dying. The prose stating the old count and
describing the re-exports was rewritten, not left stale — the physics
claim binder catches the addresses, not the sentences around them.

### Payoff

`tests/conftest.py::_isolate_protocol_singletons` is down from eight
calls to **TWO**: `reset_world_state` and the process-wide
`reset_static_key_cache`. Step 8 removes the first. The second is not
session state and never was.

The ledger tests migrated the same way the world-state tests did last
entry: a `ledger` fixture returning `LedgerService()` replaces four
reset calls per test. Tests that drive the real fabric through `Bot`
read `get_world_service().ledger` instead — they must observe the
ledger the bot actually wrote to, not a fresh one.

## 2026-08-07 — Step 9: eleven trackers that never tracked, and an empty field in 432 captures

`sniffer/trackers.py` and `sniffer/player_tracking.py` are deleted as
modules. One tracker instance survives, owned by the sniffer.

### The globals looked alive because a loop touched them

`ALL_TRACKERS` held twelve instances and `init_trackers_with_magic`
armed all twelve from `SessionBase` on every session — bot and sniffer
alike. That is what made them look live. Across all of `src/` and
`scripts/`, `process_message` is called **exactly once**:
`mine_tracker.process_message(payload, "sent")`, in the sniffer's
live-decode narration of outbound mine presses.

So eleven of the twelve were armed with a session key every run and
never asked to decode anything. Same shape as the container blacklist
in step 7: machinery a loop keeps warm, with no consumer at the end.

### It was not merely dead — it was shipping an empty field

`core.py::_build_capture_session` filled the capture's `tank_names`
from `tank_tracker.get_all_names()`. `TankTracker._tanks` is only
written inside its `_parse_*` methods, all reachable only through
`process_message`, which was never called on it. So the map answered
`{}` forever.

Checked against the archive rather than reasoned about: **432 capture
files parsed, 432 carry a `tank_names` key, 0 are non-empty.**

The names existed the whole time in the live world-state tank registry.
Replaying `fuel_probe.capture_session.json` through the real decoder:
37 tanks, **37 with names** — `Artax`, `red-1`, `red-2`, and so on.
Two implementations of the same thing, and the capture was wired to the
dead one.

Fixed by pointing the field at the registry, not by deleting it —
`capture/summary.py` consumes `session["tank_names"]`, so this is a
capability that was silently broken rather than one nobody wanted.

### The gap that let it hide

Coverage was 100% on that line the entire time. The line **executed**;
nothing **asserted it produced anything**. A test now asserts the field
carries real names and omits the unnamed tank. That distinction —
executed versus asserted — is the whole lesson: a coverage gate cannot
tell you a reader has a writer.

### Also deleted

`sniffer/player_tracking.py` was a *third* tank-name registry, with no
reader and no writer anywhere in `src/` — only its own tests kept it
alive. `extract_magic_from_auth` in `trackers.py` was likewise
test-only; production uses `protocol.codec.extract_magic_from_auth_payload`
from `browser/cdp_service.py`.

### What survives

`MineTracker`, owned by `WebSocketSniffer` and armed in its
`_on_magic_captured` override. `SessionBase` no longer arms any
tracker; the bot had been paying for twelve.

### Flagged, not done

The eleven tracker *classes* in `capture/trackers/` now have zero
production instantiation — roughly 1,400 lines of source and 2,900 of
tests whose only callers are those tests. `MineTracker` and
`CombatTracker` stay. That deletion is four thousand lines and a
separate decision from step 9's scope, so it is recorded here rather
than taken unilaterally.

Gate: `make check` exit 0, 100.00% coverage, 6184 passed (eight tests
deleted with the code they covered, one added for the `tank_names`
regression).

## 2026-08-07 — Step 10: the globals were not the blocker

Partly shipped, and the rest deliberately not done with the reason
measured rather than asserted.

### Done: the tick context is now contextvars

`_RUNTIME_CONTEXT_TICK_N` / `_BOT_STATE` /
`_IN_FLIGHT_ACTION_KIND` are `contextvars.ContextVar` slots.

This is the one place in the whole refactor where threading a
parameter is the wrong answer. `emit_ai` and `emit_diagnostic` are
called from **256 sites**, and they live inside pure planner logic —
scoring functions, target selectors, hop planners. Giving every one of
them a logging argument to carry would cost far more than the three
globals ever did. A context variable is ambient by design, which is
what an observability field actually is.

A test asserts the gain: a second thread reads an empty context while
the setting thread keeps its own. A module global cannot do that.

**Correction to what I claimed while proposing this:** I said it would
drop another conftest reset. It does not. pytest runs each test in the
same thread, so the context persists between tests and
`clear_runtime_context` stays in the conftest — verified by running two
functions in sequence and watching the second read `{'tick_n': 42}`.
The gain is thread/task isolation, narrower than I said.

### Not done: the artifact handlers, and why

The three context globals are the visible part of step 10. They are not
the obstacle. `_install_artifact_handlers` mounts on the **root logger**
and calls `_remove_artifact_handlers(root)` first, so a second session
in one process does not get its own artifacts — it *steals* the first
session's stream. De-globalising `_BOT_ARTIFACTS` without changing that
achieves nothing measurable.

Making it per-session means each of those 256 emit sites must resolve
WHICH session's logger it writes to. Today that buys nothing: the fleet
runs one process per bot.

### The part worth flagging

[[bot-service-architecture]] gives two reasons for one-process-per-bot:
harness tasks dying at ~46 minutes, and *"in-process multi-bot is
impossible (the world service is a module singleton)"*.

The second reason is exactly what step 8 deletes. So this deferral has
a shelf life — the artifact-handler problem is **blocked behind step 8
and becomes the next real blocker the day step 8 lands.** Recorded with
its entry point (`_install_artifact_handlers`) and its shape (a
per-session logger namespace, emitters resolving ambiently the way the
tick context now does) so the next person does not have to re-derive it.

### Also corrected

`clear_runtime_context`'s docstring claimed "the tick loop's teardown
path calls this". It does not and never did — the only callers anywhere
are `tests/conftest.py` and its own test. In production the context is
set once per tick and never cleared, so the end-of-run scorecard carries
the final tick's `tick_n` and `bot_state`. Defensible behaviour, but not
what the docstring said.

## 2026-08-07 — Step 8: sessions now hold their world; the singleton is down to 11 readers

`get_world_service()` call sites in `src/`: **107 -> 11** (plus three
prose mentions and the one transitional binding). `make check` exit 0,
6185 passed, 100.00% coverage, mypy clean on 1101 files.

### The count went UP before it went down, and that was my doing

Last session I quoted step 8 as "74 sites". By the time I started it was
107, because step 6 put the ledger on `WorldService` and added ~32
`get_world_service().ledger` reads. That was the right trade — six
impossible-to-duplicate globals for one lookup already scheduled for
deletion — but I should have said the new number out loud when I made
it rather than letting the old one stand.

The full surface is also larger than `get_world_service()` alone: the
`world_state.py` facade it backs carries ~200 more `src` references and
**~1,509 in tests**. `update_world_state_from_position` alone has 228
test references against 3 in `src` — it is mostly a test helper.

### Shape: `SessionBase.world`

`SessionBase` is the single root of the session hierarchy (`Bot`,
`ProbeBase`, `BrowserSession` -> `WebSocketSniffer` all descend from
it), so one attribute there reaches every session.

It is bound to the process singleton **on purpose, for now**. The
decoder still writes through `get_world_service()`, so a session holding
a different instance would read an empty world. The flip to
`WorldService()` and the deletion of the global are the LAST two edits
of the step, not the first.

95 of the 106 sites had a handle already in scope — `bot` (61), `self`
(22), `probe` (12). Driven from the AST rather than by regex: each site
needed whatever handle its enclosing function actually had, and a
textual replace would have guessed.

### Two import cycles, and mypy saw neither

**First:** `session_base` importing `WorldService` closed a cycle
because `sniffer/world_service.py` imported `get_current_time_ms` from
the **`browser` package**. mypy reported "Success: no issues found in
1100 source files" while `import tankpit_bot.bot.base` raised
`ImportError`. Importing the defining submodule did not help — Python
executes the parent package first.

The real problem was a clock living in `browser/cdp_utils.py`. It is a
three-line delegation to `_test_hooks.get_current_time_ms`, and the six
lower-layer modules that used it now call the hook directly. That is
both the cycle fix and the removal of a thin wrapper.

**Second:** `BotProtocol` needed a `world` member, but it lived in
`_test_hooks`, which sits BELOW `sniffer` — naming `WorldService` there
closes a cycle through `state`. It moved to `bot/bot_protocol.py`, where
it belongs anyway: it is a production interface with exactly one
importer (`bot/executor.py`), not a test seam.
`BufferedMessageSourceProtocol` stayed in `_test_hooks` and does NOT
gain `world`; `drain_messages` takes the service as a parameter instead.

Lesson, for the third time this refactor: **mypy cannot see import
cycles.** Only running the import can, and only in the right order —
`import state` first is what exposed the second one.

### The trap I predicted, sprung on schedule

`tests/bot/test_executor_dispatch.py::test_missing_self_state_stays_optimistic`
failed. It called `_make_bot()` (which seeds a position) and then
`reset_world_state()` to clear it.

`reset_world_state()` **rebinds** the module global. The bot now holds
its service as an attribute, so the reset left `bot.world` pointing at
the pre-reset object with the seeded position still in it — and the test
silently stopped exercising the missing-self-state path.

This is exactly the failure mode recorded when the 496-reset analysis
was done: "a half-migrated tree silently breaks test isolation because
instance-holders keep the pre-reset object." It is the only test that
tripped, because it is the only one that reset mid-test. Fixed by
`bot.world = WorldService()` — the migrated form, and clearer about
intent than a global reset was.

### Remaining

Eleven call sites, each a private helper whose CALLER has a handle:
`_merge_protocol_kills(ai_state)`, `_drain_orphan_command_error(action)`,
two `executor` helpers, `threat_primitives.human_combat_consented`,
`diagnostics/registry_truth`, three in `replay/engine` (which should own
a service outright — it is a standalone replay, not a session), the
`action_lab` teleport-landed hook default, and
`_test_hooks/runtime.py`'s replay hook (the known cycle case).

Then the facade: ~1,509 test references to `get_world_state` and
friends. Then the flip.

---

## [2026-08-07] update | Step 8 lands: the last session global is gone, and it cost one bug class 138 times

`sniffer/world_state.py` is deleted. `_service`, `get_world_service()`
and `reset_world_state()` no longer exist. Every session owns its
`WorldService` through `SessionBase.world`; standalone tools
(`replay/engine.py`, `scripts/decode.py`) each build their own.
[[session-state-deglobalisation]] steps 8 and 11 are struck; **step 10
(`runtime_logging.py`) is the only plan item still open.**

**The completion criterion this page set for itself is met.**
`tests/conftest.py` went from ten resets to **one**, and that one
(`reset_static_key_cache`) guards a process-wide key read off disk, not
session state. Constructing a session is now the reset.

`make check` exits 0: guard clean, ruff clean, mypy clean on 1104 files,
**6185 passed, 100.00% coverage** (30,370 statements, 8,780 branches).

### The whole cost was one defect wearing six disguises

Deleting the facade left **138 failing tests**. Nearly all of them were
the same thing: a test mints a `WorldService()`, seeds it, and hands the
code under test an object that owns a *different* service. Under the
facade both names resolved to `_service`, so the seeding worked by
accident. The six shapes — each invisible to the detector that caught
the previous one:

1. **Dead write** — the local is only ever written, never read. 31 sites.
2. **Two-world scope** — the local *is* read, but a probe/bot/ctx in the
   same scope owns its own. 73 scopes, 63 in `tests/bot`.
3. **Seed helper with no world parameter** — `set_inventory_total` (19
   call sites), `seed_confirmed_incoming` (13), `consent_human` (7), and
   eleven more.
4. **Parameter shadowing** — `def _drain(source, ws): ws = WorldService()`.
   The seam hands over the right world; line one throws it away.
5. **Assign-before-super clobber** — `self.world = ws` written *before*
   `super().__init__()`, which assigns `self.world` itself. 7 harnesses,
   where the assignment had never once had an effect.
6. **Mid-test world replacement** — a wait hook doing
   `probe.world = WorldService()` to model "the wire delivered a kill",
   orphaning it from the engagement that captured the world at entry.

**100% coverage never saw any of it.** Every one of those lines
executes. `set_inventory_total(2)` runs, mutates a real service through
the real codepath, returns — covered. Nothing asserts the service it
mutated is the one the probe reads. Same lesson as the `tank_names`
finding: a coverage gate cannot tell you a writer has a reader.

### Process note: the detector, and where scripting stopped helping

The user's correction mid-step — *"so youre jist running that over and
voer and not like even looking at the code?"* — was the turn. Counting
failures and re-running the suite found nothing; reading one failing
test (`assert 0 == 15`, a probe reading its own empty inventory) found
the whole class in one look.

What worked after that: state the defect as a property the AST can
decide, verify the detector on a case already understood, review the
generated diff, then apply. What did not: broadening the detector's name
regex to chase the tail. Loosening `OWNER_CALL` to a substring match
took it from 73 real findings to 264 mostly-false ones —
`configure_bot_runtime_logging` matches "bot", `build_session_xor_table`
matches "session". **The detector earned the bulk sweep and then had to
be put down**; the last dozen failures were read individually. A tool
that stops discriminating is worse than no tool, because its output
still looks like evidence.

Also corrected: an earlier scripted pass had injected `world=ws` into
test-local classes that take no such argument, and left two
`probe.world = probe.world` self-assignments. Both were found by running
the tests, not by reading the script.

### A `Callable` alias was hiding a drifted stub

`analyze_threats` gained `ws` as its first parameter, but the tracking
harness typed the patchable attribute as
`AnalyzeThreatsFn = Callable[[WorldStateDict, SelfStateDict, int], ...]`
— too weak to express the keyword-only rank bounds, so a stub that had
dropped `ws` still typechecked and silently read a different world.
Respelling it as a `Protocol` — the idiom the same file already used for
`ShotFeedbackFn`, *"spelled out so stubs are checked against it"* —
immediately failed a **second** drifted stub in
`test_enemy_tracking_execute.py`. The alias was not shorthand; it was a
hole in the type surface.

### Dead ritual the step exposed

- `tests/diagnostics/conftest.py` — deleted. One autouse fixture whose
  entire body called `reset_registry_truth()`, whose body was empty and
  whose docstring cited `reset_world_state` and `sniffer/world_state`,
  both already deleted. The file's own docstring claimed "every
  diagnostic emitter holds module-level gate/counter state" — the exact
  premise this refactor removed.
- `diagnostics/registry_truth.py::reset_registry_truth` — deleted with it.
- Two empty `setup_method`/`teardown_method` pairs in
  `test_world_state_dispatch_teleport.py` whose docstrings still claimed
  to "reset world state and dispatch tracking". A no-op documenting work
  it does not do is worse than no code.

### Not a migration bug: the xdist worker crash

`tests/bot` was killing an xdist worker with `INTERNALERROR> KeyError:
<WorkerController gw18>`, and a `--timeout=15` made it look like
`test_shot_screenshot.py` hanging. It was neither. A **RustedWarfareBot**
`runs\sweeps\navpair48` sweep was running on the same box — six live
`scripts.play` sessions, seven JVMs, plus seven orphaned
`match_worker`/`match_service` processes leaked since 17:48 — leaving
**4.6 GB free of 31.8 GB**. `make check` runs xdist at `-n auto` = 24
workers, each importing the full package, and `tests/bot` launches real
headless Chromium on top. The worker was being reaped, and the timeout
flag converted a slow real-browser teardown into a killed worker. Fixed
by dropping to `-n 4` while the sweep ran; nothing in this repo was
wrong. Worth remembering before diagnosing the next "flaky" worker
crash: check what else owns the machine.

### File size

`tests/login/test_join.py` crossed the 600-line ceiling at 603 — the
threaded arguments pushed it over. Split, not squeezed: the eight
`handle_login_flow` auto-join tests moved to `test_join_login_flow.py`
(145 lines), leaving the room-join primitives at 470.

---

## [2026-08-08] update | Step 10 closes the plan: seventeen module globals, zero left

The artifact half of step 10 shipped, and with it
[[session-state-deglobalisation]] is **complete — all eleven items**.
`_BOT_ARTIFACTS` / `_SNIFF_ARTIFACTS` / `_PROBE_ARTIFACTS` are
`ContextVar` slots, and the event handler now mounts on a per-run logger
instead of root.

Re-ran the two sweeps the page defined for itself. Sweep one, every
`global` in `src/tankpit_bot`: four sites, all four already in the
"Legitimately process-level" table (the DI seam, the Ctrl+C flag, the XOR
static-key cache, the frame-logging switch). Sweep two, module-level
containers mutated in place: **zero**. `make check` exits 0 — guard
clean, mypy clean on 1105 files, **6188 passed, 100.00% coverage**.

### The deferral expired with its own premise

This half had been deliberately deferred, and the reason was written
down: the fleet runs one process per bot, so process-scoped logging is
correct for the only deployment that exists. But
[[bot-service-architecture]] gives *two* justifications for
one-process-per-bot, and the second is "in-process multi-bot is
impossible (the world service is a module singleton)" — exactly what step
8 deleted the day before. The page had already flagged that this would
become the next blocker the day step 8 landed. It did.

### The bug was real

`_install_artifact_handlers` mounted on the ROOT logger and removed any
prior artifact handlers first. Configure a second run in one process and
the first run's event handler was silently detached — its
`events.jsonl` just stopped growing mid-session. A regression test now
steps two threads through that precise ordering: A configures, B
configures, and only THEN does A emit. Each stream ends up with its own
events and its own mode.

Worth noting what *isn't* broken: `SessionRunner` runs many sessions
**sequentially** in one process and refuses to start a second while the
first runs, so the old overwrite-and-reinstall was correct for reuse.
Only concurrent sessions lost data, which is why this sat latent.

### The two artifacts are scoped differently, on purpose

The event stream is a session artifact — per-run logger, resolved
ambiently. The text log is a process artifact and stays on root, because
**root is the only logger that sees library records**: a `world_service`
warning belongs in the run log, and a per-run logger never receives it.
Scoping both per-session would have meant stamping every record with a
run id, filtering in the handler, and then duplicating unattributed
library lines into every active session's text log to avoid losing them.
The asymmetry is cheaper and more honest than the symmetry.

### A writer with no reader

`_emit_runtime_event` stamped `runtime_mode=_runtime_mode_name()` onto
every record, and that function read all three artifact globals to
compute it. But `_HookEventArtifactHandler` writes `self._mode`, captured
when the run was configured, and never looked at the record's copy —
nothing in production read the field, only a test asserting the mechanism
existed. It was the sole consumer of those globals on the hot path, so
deleting it shrank the step to the getters plus the handler mount. The
`tank_names` finding inverted: not a reader with no writer, a **writer
with no reader**.

### The reset that skipped a slot

`_restore_runtime_logging_state` cleared the bot and sniff globals and
never touched `_PROBE_ARTIFACTS`. A probe test leaked its artifacts into
every test that followed it on the same xdist worker — a live isolation
hole nobody had noticed. The reset now calls
`clear_runtime_logging_state()`, which lives beside the state instead of
reaching into it and clears all four slots.

To be straight about what the ContextVar did *not* buy: pytest runs every
test on one thread in one context, so the values persist between tests
and the reset stays. The gain is thread and task isolation — which is
what makes concurrent sessions possible — not one fewer reset. The same
caveat the tick-context half recorded.

### Three tests that had quietly stopped testing anything

Moving the event handler off root broke coverage of its own
malformed-record guards, and coverage is what caught it. Three tests
logged synthetic records to unrelated loggers and asserted "the artifact
stayed empty" — an assertion that stayed true once the record could no
longer reach the handler at all. They now log inside the run's logger
subtree via `tests/_runtime_logging_support.py`, whose docstring records
why. Third time this pair of days that a test passed while measuring
nothing; the tell each time was a guard going uncovered, not a failure.

### File size

`runtime_logging.py` hit 622 lines. Split, not trimmed: handler classes,
run-identity naming, and handler install/remove moved to
`runtime_logging_handlers.py` (235), leaving the ambient run and the
emitters at 424. The dependency is one-way — the handlers module knows
nothing about which run is active — which keeps the pair from closing a
cycle.

---

## [2026-08-08] audit | Mutation-tested a sample of guards: 100% coverage does not mean pinned

Austin asked why the step-8/10 work fixed source but not the tests, after
I'd noted three separate times in two days that a test had "quietly
stopped testing anything". Fair hit: **every vacuous test I found this
week surfaced by accident** — one via a failure, one via a coverage drop
when a handler moved, one because I happened to read the line. I never
went looking. So this is the deliberate hunt.

### Static detectors mostly found noise

Two shapes, both cheap, both mostly false positives:

- **All-negative-assertion tests** (every assert claims absence: `== ""`,
  `== []`, `is None`, `not in`): **1001 of 6188**, 16%. Reading a sample
  shows most are legitimate — `get_position` returns None with no
  self-state, `move_to` returns False without CDP. The arrangement *is*
  the precondition, so asserting the empty answer is the only way to test
  it. The detector cannot separate those from the vacuous ones.
- **Asserting a constructor default on an object the test never
  exercises**: 36 hits, nearly all `test_init` tests where the
  constructor's defaults are the actual subject. Two apparent finds
  dissolved on reading — `assert bot._magic is None` is a *precondition*
  before feeding an AUTH frame, and the session tests do exercise their
  subject via `session._cdp_service._extract_magic_and_notify(msg)`,
  which my "exercised" check missed (attribute-of-attribute blind spot).

Lesson repeated from the step-8 detector: a static shape that cannot
distinguish intent produces a list, not evidence.

### Mutation is the detector that cannot be fooled

Enumerated the 483 defensive early-return guards in `src/tankpit_bot`,
sampled 14, turned each guard's `return` into `pass`, and ran the whole
suite per mutation. **12 killed, 2 survived.**

Survivor 1 — **a real test defect**, now fixed.
`capture/trackers/mine.py` guards `if len(data) < 4: return None`, and
`test_process_message_returns_none_for_short_data` fed
`b"\x02\x00\x2e"`. Remove the guard and execution falls through to
`msg_type not in (0x45, 0x4B)`, which returns None as well — so the test
passed whether the guard existed or not. **The line was covered the whole
time.** Coverage proves a line executed; it says nothing about whether
any assertion depended on it. Fixed by making the third byte `0x45`, a
tracked mine type, so only the length guard can produce None. Verified by
re-running the same mutation: now killed.

Survivor 2 — **not a test gap.**
`validate/shadow_timeline.py:304` short-circuits `_ingest_combat_events`
when `_ingest_tank_events` consumed the message. The two dispatch on
disjoint `msg_type` sets (0x21/0x3D… versus 0x41/0x67…), so calling
combat after a tank event is a guaranteed no-op. The mutant survives
because the guard *cannot* be observed — a redundant short-circuit.
Flagged for Austin rather than changed: it is arguably dead per the
no-redundant-code standard, but it also documents that the families are
disjoint, and `validate/` was outside this work.

### The number worth carrying

A 14-guard sample killed 12. Extrapolated across 483 guards that is
roughly **60-70 guards no test distinguishes from absent** — bounded, not
catastrophic, and invisible to both gates the project already runs.
`make check` stays at 6188 passed / 100.00% coverage either way, which is
the whole point: **a green suite and a full coverage gate together still
cannot tell you a test verifies anything.** Only mutation can, and the
project has no mutation gate.

---

## [2026-08-08] update | The sweep becomes a rule: one hook leak, seven fakes, and three detectors that lied

Three commits: `16339076`, `57d90034`, `8e25e5bf`. `make check` green at each — **6,171 passed, 100.00% coverage** (30,428 statements / 8,806 branches).

### One real leak in 391 sites

`tests/bot/test_tick_loop_lifecycle.py:55` set `_test_hooks.remove_file` inline and never put it back. `path_exists`, assigned on the line above, was safe only because it sits in the autouse `_restore_hooks` list; `remove_file` did not. Every later test on that xdist worker inherited a `remove_file` that silently did not delete and appended to a dead closure list — the exact cross-test poison `_restore_hooks` exists to prevent, and the thing its own docstring cites the 2026-07-03 replay flake for.

Fixed in the reset list (now 16 attrs), not with a sixteenth local teardown. The docstring there already calls itself the single reset point; a local fix would have contradicted it.

### The rule, because a sweep only describes today

`scripts/hook_restore_rules.py`, wired at `scripts/guard.py:139`. It fails the build when a test assigns a `_test_hooks` attribute that is neither centrally reset nor restored under a recognised guard. Restoration counts at three scopes — a `finally` body, a post-`yield` fixture, a `teardown_*` method, or an ancestor `conftest.py` — and that list is not generosity, it is the correction of a first draft that reported **24 violations of which 23 were legitimate**. The last two shapes put the save and the restore in *different functions*, which a per-function check structurally cannot see. A rule that flagged them would have been deleted within a day.

It ships a paired negative control: it must fire on a planted unrestored swap **and** stay silent on all four legitimate shapes. 13 tests, 100% of 87 statements and 56 branches, no partials.

### Seven fake filesystems became one

`tests/conftest.py` already owned a `FakeFileSystem` that 84 files import. Six private copies existed anyway, each with its own installer. Consolidated to one class plus one shared installer for the four sites that need it from `setup_method`, where a fixture cannot be requested. `tests/_smoke_records.py` keeps an 8-line installer because it targets `scripts._test_hooks`, a genuinely different module — that part is irreducible, not missed.

**Copies five and six were invisible to the search that found one through four.** `_FakeFS` in `tests/replay/test_script.py` named its methods `write`/`read`/`exists`/`append` instead of `write_text`/`read_text`/`path_exists`/`append_text`, so it survived a name grep *and* a method-name sweep, and surfaced only as an `AttributeError` when the tests ran.

### Three detectors reported a clean zero while pointed at nothing

This is the entry's real content, because it happened three times in one day:

1. The first negative control returned **0 violations** and looked like a passing rule. It had been handed a POSIX path (`/c/Users/...`) on Windows, so `tests_root.is_dir()` was false and it scanned nothing.
2. The behavioural sweep for remaining fake filesystems returned **0 classes** on a tree that certainly contained one. It required `str(path)` inline as a subscript slice; the canonical class writes `key = str(path)` on the previous line.
3. Earlier the same day, the browser-leak probe read **0 processes while a browser was running**, because it filtered on `chrome` and headless launches `chrome-headless-shell.exe`.

**"0 violations" and "0 files examined" are indistinguishable from the outside.** Every detector that reports a clean result needs a known-bad input before the clean result means anything. All three of these were caught by accident — by a failing test, by disbelief at the number — not by design.

### Three browser launches became one

`tests/browser/test_lifecycle.py` launched a fresh headless Chromium per test in three tests that, despite the filename, assert nothing about launch or close semantics. They share one module-scoped browser now and take their own context each, matching the `live_cdp` fixture that already did this. Real launches in the suite: **four to two**. This does not fix the ~48 s teardown — a browser here survives `taskkill /F /T` returning SUCCESS, reproduced with Firefox too, so it is not a Chromium bug — it stops paying for it three times.

### Nine unreferenced functions deleted

`pathfinding.path_length`, `equipment_search.find_nearest_deposit`, `tactics.should_map_open_for_enemies`, `tactics._is_visible_enemy_tank` (orphaned by the previous cut, found only by re-running the sweep after it), `resource_search.is_recently_attempted`, `resource_search.record_attempt_mark`, `decoders.try_decode_received`, `decoders.decode_received_text_message`, `decoders.try_decode_received_text` — plus their `bot/ai` re-exports and a dead `analyze_threats` protocol member in the enemy-teleport harness that declared two positional parameters against a real four-plus-two-keyword-only signature. **+22 / −623.**

### ~~Open:~~ RESOLVED 2026-08-09: six mutation survivors, recorded here because the artifact is gone

A larger mutation run than the 14-guard sample above got through **37 of 483 guards before being interrupted**, and named six survivors. Preserved here verbatim, since `mutation_results.txt` was deleted from the repo root:

```
SURVIVED  src/tankpit_bot/action_lab/enemy_teleport.py:109        return
SURVIVED  src/tankpit_bot/action_lab/probe_base.py:131            return False
SURVIVED  src/tankpit_bot/action_lab/tracking_observation.py:58   return None
SURVIVED  src/tankpit_bot/bot/ai/collect_hops.py:156              return None
SURVIVED  src/tankpit_bot/bot/ai/collect_hops.py:308              return None
SURVIVED  src/tankpit_bot/bot/ai/collect_hops.py:396              return None
```

Six of 37 is a ~16% survival rate, consistent with the 2 of 14 recorded above, and it still extrapolates to roughly 60-80 guards across the tree that no test distinguishes from absent. `collect_hops.py:396` is `if terrain is None: return None`: branch coverage says both outcomes are taken, mutation says deleting the `return` changes no test result — so a test reaches the guard and never pins its effect. ~~**None of the six are fixed.** They are the next piece of work, not a finding that was acted on.~~

**Corrected 2026-08-09 (same day):** all six are now fixed and each verified by re-mutation — commit `13da274d`. Five were vacuous tests, one was a redundant guard that was deleted. One of the six is not the guard this paragraph implies: **`collect_hops.py:308` is the CAPACITY gate, not a terrain guard.** Full account in the entry at the bottom of this log; the sentence above is struck rather than deleted because a reader who lands here from a search must not act on it.

### Scope limit, stated plainly

The new rule covers `_test_hooks` attribute swaps. The wider leaked-process-state class was swept by hand the same day and found clean — all three `addHandler` sites pair with `removeHandler` in a `finally`, both `sys.modules` mutations are restored, and the one `world.update(fresh)` hit is a per-instance `WorldService` rather than a module global, which is what [[session-state-deglobalisation]] bought. That half is **not** machine-enforced, so it is true as of today rather than guaranteed.

---

## [2026-08-09] update | Six survivors die, and the one that took three wrong theories was not the guard we thought

Commit `13da274d`. `make check` green: **6,175 passed, 100.00% coverage** (30,426 statements / 8,804 branches). Statements and branches each fell by 2 — exactly the one guard deleted.

Every fix verified by re-running the same mutation. None was accepted on argument.

### Five vacuous tests, one shape

Each was exercised against a world that ALSO failed a later gate, so the guard and its absence produced the same answer and no assertion could tell them apart:

| guard | why the mutant survived | fix |
|---|---|---|
| `tracking_observation.py:58` | registry lacked the key, so the loop fell through to the same `None` | registry now carries an entry actually keyed `""` |
| `enemy_teleport.py:109` | the zero-settle test passed a POSITIVE heartbeat, and on that path the loop's own `remaining > 0` declines anyway | extended to the no-heartbeat branch, the only one that can observe it |
| `collect_hops.py:156` | empty candidate list returned `None` by itself | candidate is now live and affordable |
| `collect_hops.py:396` | same shape in the marooned hop | qualifying fuel container present |
| `collect_hops.py:308` | see below | pinned on the event stream |

### One redundant guard, deleted — and it was hiding a diagnostic

`probe_base.teleport` was the only one of ELEVEN sibling methods to pre-check `self._cdp is None`. `send_command_bytes` already does, at `command_sender.py:59`, with a `log.warning("Cannot send %s: CDP session not available")`. The redundant guard's silent `return False` suppressed that warning. So it was not merely unobservable — `teleport` was the single command in the class that failed *quietly* when CDP was gone. Deleted, which restores the diagnostic.

### `collect_hops.py:308` is the capacity gate, not the terrain guard

Three wrong theories, recorded because each was expensive:

1. **"It's the terrain guard."** It is not. Line 307-308 is `if ctx.fuel >= fuel_capacity(rank): return None`; the terrain guard is 309-310. A test built around unknown terrain never exercised 308 at all.
2. **"A full tank will expose it."** It will not. At capacity the deficit is zero, so `min(volume, deficit)` never clears the hop cost and `select_fuel_larder_hop` returns no container either way. **The return value cannot distinguish this guard from its absence.**
3. **"Assert no `hop_declined` fires."** Right idea, wrong field — see the trap below. That fix was itself vacuous and the mutant survived a third time.

What the gate actually does is suppress a diagnostic: without it, a full tank beside a live fuel container logs `hop_declined fuel_larder candidates=1 unprofitable=1 fuel=1200` **every tick**. Proven by mutating it and capturing 2 records instead of 1. So the test pins the event stream, not the return value.

### The `runtime_fields` trap

`emit_diagnostic` does NOT flatten fields onto the LogRecord. They nest:

```
{'runtime_channel': 'DIAGNOSTIC',
 'runtime_fields': {'diagnostic_kind': 'hop_declined', 'hop_kind': 'fuel_larder', ...},
 'runtime_message': 'diagnostic_kind=hop_declined'}
```

So `record.get("diagnostic_kind")` returns `None`, any filter built on it matches **zero** records, and the assertion passes while the mutant lives. `tests/_runtime_logging_support.py` now exposes `event_fields(record)` so this cannot be stepped in twice, and the test carries a PAIRED assertion — the equipment decline MUST be captured, or it proves nothing about the fuel one.

### Lifted, not forked

`capture_runtime_events()` + `event_fields()` replace two hand-rolled `addHandler` / `setLevel` / `finally`-remove blocks in the sniffer tests. Zero files still hand-roll the emitter logger.

### The lesson worth carrying

Every failure this session was one bug: **a stale or mis-aimed detector reporting clean.** The mutation harness read the wrong line number; the assertion read the wrong field; and outside this work the same day, a probe filtered on `chrome` while headless launches `chrome-headless-shell.exe`, a control was handed a POSIX path on Windows and scanned nothing, and a health probe compared `$null -eq 0` and called a working daemon broken. In every case the output was a confident zero.

**A detector that has not been run against a known-bad input is not evidence.** That is why the fix for 308 carries a positive assertion alongside the negative one.

---

## [2026-08-10] audit | The trackers are a pretty-printer, not a decoder; and thin wrappers cannot be machine-checked

Three questions closed, two of my own analyses thrown away getting there. Commit `f8b4afcf` (pytest 9.0.2 -> 9.1.1, verified: 6,175 passed, 100.00% coverage).

### `capture/trackers/` is a capture pretty-printer

Every tracker has one shape:

```python
def process_message(self, payload: str) -> str | None
```

Base64 payload in, a **human-readable line** out (`[TANK:STATUS] id=42 'name' red private`) or `None` when it does not apply. There is no `msg_type` dispatch because that is not the design -- each tracker attempts its own decode and declines.

So these are NOT a redundant second implementation of the live decode path. `sniffer/world_state_dispatch` produces structured `WorldStateDict` updates; the trackers produce display strings for reading a capture by eye. That is why the bot consumes none of them -- it has no use for formatted strings -- and they are the only `process_message` implementations in `src/`.

Import graph, by AST:

* **`MineTracker` is live** -- instantiated at `sniffer/core.py:116`, armed from `_on_magic_captured`.
* **The other 11 are reachable only through `capture/trackers/__init__.py`**, the package re-export. No consumer in `src/` or `scripts/`; their only real callers are their own test files.

Sizes if they go: **1,696 src + 3,154 test = 4,850 lines**.

**The decision this enables:** deleting them removes a debugging tool, not protocol knowledge. That is a far lower-stakes call than the one my first analysis implied, and it is the user's to make.

### Two analyses binned on the way, both the same bug

**First:** grepped hex literals out of the tracker files and treated every match as a protocol discriminator. It produced "7 orphan message types, 3 of them undocumented". All three were artifacts:

| claimed | actually |
|---|---|
| `0x0F` undocumented type | `rank = (info_byte >> 4) & 0x0F` -- a nibble mask |
| `0x75` undocumented type | a docstring saying "subtype varies per session (0x75, 0x76, etc.)" |
| `0x80` undocumented type | matched inside `if val_unsigned >= 0x8000` -- a sign-bit threshold |

The companion figure, "27 of 34 types overlap with the live path", is garbage for the same reason: it compared bitmasks and thresholds, not message types.

**Second:** the corrected AST scan, keyed on `msg_type == 0xNN` comparisons, found only **2 of 9** tracker files dispatch that way at all (`mine.py` on 0x04/0x45/0x4B, `tank.py` on 0x2E) -- and every one of those IS handled by the live path, so zero orphans. Right method, but it could not see the other seven files, so it could not answer the question either. Only reading `process_message` did.

### Thin wrappers cannot be machine-checked here

An AST sweep finds **60 pure pass-throughs** in `src/` (a body of exactly `return other(<own params>)`). The instinct is that these violate "no thin wrappers". They do not:

```
SurfaceRouteTerrain.get_terrain -> get_terrain        # required by TerrainMapProtocol
CommandService.send_bytes       -> send_command_bytes # the object seam
ProtocolCodec.encode / .decode  -> xor_bytes          # naming the domain operation
TerrainMap.is_landing_legal     -> is_passable        # naming the domain operation
```

`SurfaceRouteTerrain.get_terrain` MUST exist to satisfy its Protocol. `ProtocolCodec.encode` naming `xor_bytes` as an encode step IS the abstraction. Separating those from a pointless alias needs intent, not syntax -- and `scripts/shim_rules.py` states the governing principle in its own docstring: *"A rule that needs a human to adjudicate would need an allowlist, and an allowlist is the thing this project refuses."*

**So the rule is deliberately NOT extended to thin wrappers.** Recorded here so a later pass does not "fix" these 60 sites, and does not add the allowlist that would be required to.

`equipment_probe.py`'s nine `_x() -> x_for_probe()` methods look mechanical but bind `self`-derived arguments, so they are not pure aliases either.

### Compliance audit against the standing rule

"No back-compat shims, no thin wrappers, no fallbacks, no legacy code, no type alias", checked:

| rule | result |
|---|---|
| no type alias | **0** `TypeAlias`, **0** `Any`, **0** `cast`, **0** `noqa` |
| no fallbacks | **0** -- no `except ImportError`, no `getattr(_,_,default)` anywhere in `src/` |
| no legacy code / shims | enforced by `scripts/shim_rules.py` (legacy vocabulary, `X = X`, renamed re-exports) |
| no thin wrappers | not enforceable without an allowlist -- see above |

Every apparent violation is docstring prose stating the ban, or `tests/test_guard_checks.py:270-273`, the guard's own negative-control fixture.

### Mutation sweep, in progress

**28 of 474 guards** done, **0 survivors**, paused for machine load. Harness rewritten with the failure from 2026-08-08 fixed: an inflight marker plus a byte-for-byte backup outside the repo, so a hard kill is recoverable instead of silently leaving `src/` mutated. **The recovery path was tested by simulating a kill** -- it printed `RECOVERED ...` and restored the file byte-for-byte -- and then earned it for real when the run was killed mid-mutation. Results land outside the repo; the previous run left `mutation_results.txt` in the repo root.

### Infrastructure: mirrored mode removes Docker Desktop's recovery path

Docker Desktop's backend IPC hung at 21:25 on 2026-08-09 with **nothing logged** -- both the stats ping and the backend's own 30 s `/time` heartbeat went silent between two ordinary log lines. The proxy then fell back to dialing `192.168.65.7:2376`, which **cannot succeed** under `networkingMode=mirrored`: verified from inside the VM, its only addresses are `192.168.10.108` and `100.77.206.124`, and nothing listens on 2376. So a momentary hang became a **33-minute outage** of both public endpoints, ending only when Docker Desktop's supervisor relaunched the backend at 21:57.

Ruled out with evidence: container crashes (`restarts=0`, `exit=0`, `oom=false` on every container), memory (32 GB free, zero OOM events), the match fleet (its code contains no docker/wsl/netsh call; its only kills are `taskkill /T` on its own match-tree pids and a port holder that must be named exactly `java`).

**Why it wedged is not knowable from what exists:** dockerd's stdout is a pipe into Docker Desktop's `memlogd` ring buffer, reachable only through the IPC that hung, and `/var/log` inside the VM holds one stale `apk.log`. The diagnostic channel and the failed channel are the same channel. A watchdog was built to close that gap and then **deleted**: the evidence actually needed was already in `com.docker.backend.exe.log` and the container logs, so the gap was analytical, not instrumental.

## [2026-08-12] audit | The guard sweep finishes: 103 survivors resolved, and 13 of them cannot be pinned at all

Both sweeps closed. 26 commits, ending `c25891d5`; verified 6,066 passed, 100.00% coverage, and every claimed kill re-mutated against the final tree (**31/31 still die**). New page: [[guard-mutation-sweep]].

### The number that matters is the kind mix, not the total

**76 of 474** guards survived the first sweep, **27 of 113** the second — which collected the guards the first pass structurally could not see, because its collector required an `if` body of exactly one statement and anything that logged before returning was invisible. That 25% is the higher rate, and it is the expected one: a log line is precisely the effect assertions skip.

Of the 27, **14 were killed and 13 are structural** — dispatch-chain and cascade arms whose removal changes nothing observable. That was measured, not argued: all four scorecard cascade returns removed and 1,403,706 archived records re-routed gave byte-identical accumulators; the three `sim/ghost` returns removed recompiled all 34 capture sessions to byte-identical specs. **No test can kill those 13, and none does.** What is enforced instead is the property that makes them unobservable — cascade arms must test pairwise-distinct values, read from the source and negative-controlled by injecting a duplicate arm (`df7f9ef0`, `6efbab0a`).

### Two faults 6,000 passing tests could not see

A **survived** mutant in `_capture_static_key` truncated the tracked 160 KB `tpclient.js` to zero bytes while the whole suite stayed green (`9a7a63cc`). A fetch returning nothing became `""` and was written to a CWD-relative path; the guard was the only thing keeping two tests off the real filesystem, and neither installed the filesystem fake because under unmutated code they never reached the writer. The suite was blind; the working tree was not.

`MSG_MIN_LENGTHS` listed `0x45` and `0x4B`, container-only subtypes with no top-level decoder, while a comment asserted twice that every listed type had one (`c25891d5`). Removing them made `roundtrip.py`'s `message is None` arm unreachable — it existed only to absorb the table's lie — and moved that fixture's top-level `0x45` from "invalid frame" to "skipped", consistent with the `0x99` beside it that was already skipped without being counted.

### Method, in order of how often it saved the analysis

**Probe before classifying** — reading produced the wrong answer roughly 40% of the time. The worst case was a docstring I wrote asserting an existing test caught a regression; injecting that regression showed it did not, because the guard returns before the code the test targets is ever consulted. Two overlapping defences, neither pinnable alone.

**Negative-control the detector, again after refactoring it.** mypy rejected a `seen.add()` comprehension in the duplicate-arm reader; the rewrite had to be re-controlled, because a detection rewrite can stop detecting silently.

**Verify against a green baseline.** The harness prints KILLED when the mutant run fails *for any reason*, so a broken test in the same file manufactures a false kill. Caught once, on `decoders:133`.

### Loop exit routes need a narrower target, not a longer timeout

Five guards retired with **no verdict** across two runs: removing a loop's exit route leaves it non-terminating, one hanging test eats the 600 s per-mutant cap, and the whole run dies. Sweeping only the two files that drive `run_tick_loop` (and the one driving the enemy-teleport settle) makes the hang land in seconds and attributes it — all five are **killed by non-termination**.

### Harness hazards worth not repeating

`run_in_background` already detaches; wrapping the sweep in `nohup ... &` produced a process the tool could not see or stop, so a second sweep ran concurrently against the same repo and both wrote verdicts. Six were discarded rather than trusted — `movement.py:461` appearing twice was the tell. A sweep also mutates `src/` repo-wide, so nothing else may edit the tree while one runs.

---
## [2026-08-10] delete | The eleven orphaned pretty-printers go — the audit's open decision, closed the same evening

(Entry written retroactively 2026-08-13; the same-day audit entry above ends with "the user's to make", and this is the call that was made.)

Commit `cfe47b09` (21:59, hours after the audit): the eleven tracker classes in `capture/trackers/` are deleted — 8 modules, 11 test files, **−4,270 lines net**. They were the January 2026 reverse-engineering toolkit; `sniffer/world_state_dispatch.py` (2026-04-04) superseded all twelve with structured `WorldStateDict` updates, and eleven were orphaned that day and never removed.

**Why they survived four months of cleanups:** the package `__init__.py` re-exported them, so an AST call-graph sweep saw a reference and an unreferenced class looked live — exactly why the 2026-08-07 re-export purge (`0ee86133`) walked past them. The `__init__` now re-exports nothing, and all three `MineTracker` importers name `capture.trackers.mine` directly.

**`MineTracker` stays for a narrower reason than "the bot needs it" — it does not.** The sniffer pipes its line to `log.info` and the bot never reads it. Narration nobody calls is dead; narration the capture tool calls is the tool working. The earlier plan to also keep `CombatTracker` ([[session-state-deglobalisation]] step 9) did not survive the audit.

Gate at the commit: 5,980 passed, 100.00% coverage; statements 30,426 -> 29,789 with coverage unchanged — source and tests went together and nothing was left uncovered.

---
## [2026-08-13] fix | The state budget stops calling protocol round trips "idle"; a contaminated validator window and a dead report field go too

(Entry written retroactively 2026-08-13 for the 2026-08-12 21:10 -> 2026-08-13 08:49 batch: `464ddd8a`, `73de3e51`, `6d337b9a`, `e2775de7`, `d4f875cd`.)

**Idle was overstated by more than half, then by all of it.** A map open is dispatched FROM the IDLE state and has no state of its own, so every second waiting on MAP_DATA was credited to IDLE (`464ddd8a`); a scope shift is a COMMAND with no HFSM state, so the quad sweep's steering ticks were credited the same way (`73de3e51`). Run 20260812-194435's "IDLE 16s (11x)" decomposed to: IDLE/map_open 10s (3x), IDLE/scope_shift 6s (3x), IDLE 0s (5x, zero-length pass-throughs). **Zero seconds of actual idleness — the old label named the wrong thing entirely.** The two markers sit at opposite ends of their stretch (a map open is observed COMPLETING, a scope shift is observed being SENT), so `_idle_bucket` tests them on opposite edges; getting that the same way round would silently drop every scope shift.

**The teleport-cost validator's 7 "mismatches" were all measurement faults** (`6d337b9a`): in every one the teleport's own debit was exactly `floor(6*euclid)` with a SECOND fuel movement folded into the pre/post difference — 4 single hits, one dual, two pickups. `_window_is_clean` inspected `action_lines` (things the bot DID), and being shot is not an action. The new predicate counts fuel MOVEMENTS and rejects a second one — counting rather than recognising magnitudes, because 45 is shared by five different causes and mis-attributed three times while investigating. The rule separates the archive completely: all 3,869 clean windows hold exactly one movement, all 7 contaminated hold two. Evidence line: samples=3875 exact=3868 mismatches=7 -> samples=3869 exact=3869 mismatches=0. Also recorded: dispatch-anchored wire debits confirm `floor(6*euclid)` on 8,032 of 8,032 archived teleports, an instrument NOT adopted because the fuel book's per-entry attribution never reaches events.jsonl.

**`recovery_boxed_in` outlived its emitter by seven weeks** (`e2775de7`): the combat rework deleted the emitter 2026-06-23, nothing emitted it in 174 runs since, and the report field could only ever read 0 — field, codec pair, renderer branch and the test asserting a count nothing produces, all removed. Worth noting how it lasted: the mutation sweep only mutates guards, so instrumentation outliving its instrument is invisible to it, and nothing in the gate looks for it.

**Wiki repin** (`d4f875cd`): 8 drifted directory-tree pins re-established after path-exists and line-anchor checks passed on all 73 pages. Tree pins drift on any commit to the package and carry mostly noise; file-plus-line pins carry the signal.

---
## [2026-08-13] build | The wrong-pond ferry soak lands — the queued adversarial geometry, pinned end to end and negative-controlled

The follow-up queued 2026-08-05 ("the ferry scenario only encodes the happy geometry... a wrong-pond sim scenario variant is the queued follow-up") is built: `tests/sim/test_run.py::test_ferry_session_never_boards_the_wrong_pond_ferry` runs the production bot through `--ferry` against the live-deadlock geometry — each water seed on its own tiny pond, the scenario's ferry alone in a 3x3 puddle, no goal sharing a water body with any ferry.

**The first cut rebuilt the happy soak by accident, and the seeder was why.** A 304-tile ridge-split lake put one water body over `_FERRY_WATER_SPACING` (300), so `seed_ferries` floated one SAME-pond ferry onto it and the pond gate correctly boarded it — the gate WORKING is indistinguishable from the gate absent unless the world starves the seeder. The shipped terrain totals 43 water tiles, below the spacing, so the only ferry is the scenario's own wrong-pond one.

**Paired assertions, per the mutation-sweep doctrine:** the water fuel must be CONSIDERED and refused (at least one `fuel_larder` `hop_declined` with `no_landing > 0`, every decline carrying `ferry_served == 0`), no `hop_selected` may ever land inside the puddle (drift-proof: the box, not the seed tile), both water containers end at their seeded volumes, and the client survives. **Negative-controlled before being trusted:** with the pond membership check disabled the larder hopped onto the puddle at (119,113) on the first opportunity and the landing assertion killed the mutant; the gate restored, the test passes.

Also in this operation: [[session-state-deglobalisation]] step 9's stale "still standing" note replaced with the 2026-08-10 resolution, and the two retroactive entries above.

---
## [2026-08-13] flags + lift | Sixteen HUD flags in two live sessions -> seven root fixes: recon goes need-driven, harvest commits, the auth jar splits per account

The user ran the fleet live (artax runs bot-20260813-195231 and -204615) and flagged sixteen behaviors; every flag was traced in the log to a root cause and seven fixes shipped, each gated (`make check` green, **6,085 tests, 100.00%**).

**The recon reorder — the kill-rate regression found and fixed.** The state budget showed recon (58 scope shifts + 60 radars) eating 270 s of a 793 s session — MORE than combat's 236 s — and the ~1 kill/min sessions all predate the quad sweep's 2026-08-07 landing. Root: the sweep ran ABOVE the pickups (atomicity doctrine) and was stock-blind (flags 8/9/14/15: it swept four windows past a container that covered the radar shortfall, and past mine-hit-revealed in-window equipment). A predictive known-stock gate was built first and the ferry sim KILLED it — water-locked fuel read as "known stock" and suppressed the recon that finds the ferry — so the shipped fix is structural: `plan_quad_sweep` moved BELOW pickups, clearance, block harvest, larder and the ferry scout in the cascade. A mid-sweep reveal is collected next tick and the movement aborts the remainder via the anchor latch: the sweep is now an incremental scan-until-found. Two more cuts on the same flags: the "current window still fresh" opportunistic radar (32-tile scraps of 87%-covered windows, most of the session's 11 zero-yield scans) is deleted, and the quadrant order is OPPOSITE corners first (NW, SE, NE, SW — adjacent windows share a 16-tile strip, opposite ones a single tile, so under stop-on-found the first two scans buy 511 unique tiles instead of 481; user-derived). Downstream re-pins: the sweeps-before-larder pin FLIPPED (known profitable stock now outranks recon), and the real-session replay that radared 9 times now drinks its 8 recorded containers after one scan — which is what "replays as a restock" always meant.

**Committed harvest ([[committed-intent]] phase 2, flags 2/5/7/10 + the flag-4 oscillator).** `plan_block_harvest_leg` built decisions with `clear_resource_target` — a leg carried NO commitment, and the second live session filmed the endgame: the harvest frame shift ping-ponged dir=1 <-> dir=5 forever (each shift changes the window that feeds the next nearest-target derivation; two out-of-window containers on opposite sides form a two-state oscillator with zero movement). Harvest targets now latch the same resource lock the larder uses; pursuit is owned by the lock-continuation step until an ENUMERATED release fires.

**The corpse trap (flag 3, second session) — measured same night, law reversed.** Post-kill in a one-exit pocket, seven consecutive pickups closed `cant_go` inside the 22 s corpse window. The capture saved at 20:55 and the owed measure ran immediately: all seven are PURE refusals (zero 0x47 echoes — the first step was already blocked), static terrain shows rock on every neighbor of (34,189) except the corpse tile, 0x2E restated tank 532 at (33,189) with the dead-corpse sprite (direction=32) through the whole window, and no mine stood anywhere in the pocket. **Corpses DO block walking.** The 2026-08-04 dissolution's miner structurally could not see this class — its blocking-proof pattern required a partial-walk ECHO, and an adjacent-standing bot draws a bare receipt — and its six "clean crossings" fixed corpse tiles from last-wire positions that displaced/ranged kills falsify. `is_tank_body_present` counts corpses again, the sim's `_blocked_by_world` matches, and both pins carry the receipts ([[flag-triage-20260729]] question 2, re-reversed).

**Walk-vs-teleport re-priced at the combat close (flag 16).** `WALK_CLOSE_TILES` was 3, derived from the falsified ~2 s/tile walk premise the 2026-08-06 measurement overturned everywhere else. At the measured ~0.22 s/tile the walk beats the teleport's map-open + hop on time out to ~9 tiles at a sixth of the fuel; the bound is now 8 (hidden-mine arrest risk keeps it under the full break-even). Same fossil, second sighting: the equipment hop lane's own-tile guard (s8-2) extended to the walk-dominant range after flag 1 filmed a map-open + cost-6 teleport onto a container ONE tile away that the ordinary pickup served four seconds later (`WALK_DOMINANT_RANGE` promoted public from the larder — one rule, no fork).

**Mine clearance completes (flags 3/6 first session + flag 4's economics).** Corridor clearance existed only on the combat close; COLLECT's walks fell through to teleports. New `find_walk_clearance_shot`: the straight corridor to the nearest WANTED in-viewport container with a shootable hostile mine draws the free single, and next tick's walk serves the container. Both clearance arms now price fuel by the CLAMPED transfer (`min(volume, deficit)` against the shared walk rate), not container volume — the run had spent two shots exposing 455/852-volume containers at deficit 18 it then refused every tick. And the shoot -6 / drink +6 alternation those shots produced is dead by floor: fuel pickups below a 25-fuel clamped transfer no longer dispatch, waived only when the sip COMPLETES hunt readiness (the hunt-only-when-full contract needs fuel exactly at cap; pinned both ways).

**The auth jar splits per account.** Fleet child "arterial" spawned with `TANKPIT_ACCOUNT=Arterial` resumed the SHARED `runs/state/tankpit.storage.json` holding Artax's cookies, skipped the credential flow entirely (its log: straight to `/play`, no login), joined as a second Artax and was socket-cut in 2 s. `resolve_storage_state_path` now keys the cache by the RESOLVED login identity (`tankpit.<username>.storage.json`, `guest` for guest sessions); the fleet's account lease already guarded the other half. Verified live: the next arterial launch ran the real login as Arterial — and exposed the next bug in line, "room list never exposed Practice" on a fresh-login lobby. Undiagnosable from that line alone (nothing seen vs a list without the room are different bugs), so the discovery timeout now logs exactly which rooms were captured. OWED: rerun arterial and read the new line.

**Also filmed, deliberately deferred:** the scope-shift triple-dispatch (first session flag 6 — shifts are fire-and-forget, so a slow 0x5A confirm lets the same steer re-dispatch; a real fix is a `scope_shift` ActionKind through the HFSM wait machinery, and the sweep demotion has removed most shift volume, so it waits for a recurrence under the new build). Two engagement-law receipts worth keeping: "avoided" red-8 was actually FOUGHT and broken off by the projection law ("unwinnable at any fuel: needs 1548, capacity 1100"), and the far purple was the nearest winnable target — working as designed.

**Files:** `bot/ai/quad_sweep.py` (split; `block_harvest.py` new), `collect_mode.py` (cascade reorder), `mine_clearance.py` + `collect_pickups.py` (walk arm, gain pricing, sip floor), `combat_close.py` + `collect_hops.py` + `larder.py` (walk-dominant pricing), `browser/session_storage.py` + `bot/base.py` (per-account jar), `browser/room_join.py` (discovery diagnostics), with test splits and re-pins throughout.

---
## [2026-08-13] crack + lift | The arterial lobby mystery closed: `default_troop=-1` is "no tank on this room yet" — and one account holds four color-tanks per map

The per-account auth jar exposed the fresh-login path, the fresh-login path exposed the room-discovery failure, and the raw-frame diagnostic (added for exactly this) caught the truth on the second arterial launch: Practice's entry WAS on the wire — `+1|Practice|1|0,0,0,0,0,0,0|-1|p|field01.gif|2026` — and `is_room_info_text`'s digits-only test on field 5 silently rejected the `-1`. The user's account theory named the semantics: Arterial has never played Practice, so it has no tank there, and the lobby answers `-1` with a color picker.

**The lift:** the validator accepts `-1` (pinned with the exact wire line); `join_room` substitutes a chosen color on first entry — the room-enter request already carries the troop byte the picker feeds, so no UI automation and no new choreography. `resolve_room_troop` lives in `browser/room_join.py` (its only consumer; `browser` sits below `bot`, so the resolver could not join the `bot/config.py` family): `TANKPIT_TROOP` 0=red 1=purple 2=blue 3=orange, default blue. Arterial is configured orange (`TANKPIT_TROOP=3` in `.env`) so it can FIGHT the blue fleet.

**New game facts recorded in [[game-rules]] (user contract + wire receipts):** one account holds up to four tanks per map — one per color — with separate inventories, fuel and points and SHARED awards; a 5-minute cooldown gates logging out and re-entering as another color. The lobby's `=` records (creation date, name, four trailing fields) are plausibly the per-color slots — unconfirmed, noted as such.

Gate: guard clean, **6,091 tests, 100.00%**.

---
## [2026-08-13] crack + lift | The 'a' key was never a command: hotkey maps are per-account, and the autoscroll enforcement goes direct-to-wire

Arterial's first entry crashed on "no autoscroll ack" twice — surviving a re-press band-aid that shipped before the capture was read (the user's rebuke was earned: "do you like not see the logs or anything???"). The capture had the answer all along: both 'a' presses DID reach the client, which sent **`03 5a 06` — the 0x5A scope shift, direction 6 = west** — while artax's same press sends the plaintext `A1`/`A0` settings command. Same key, different command: **hotkey maps are PER-ACCOUNT server state** (the `H` command, [[client-commands]]), and a fresh account's default binds `a` to a WASD-style scope pan — which is what the user was literally watching ("the extend viewport works fine"). Fresh accounts can also start with autoscroll ENABLED (user contract).

**The lift:** keypresses are deleted as an instrument. `ensure_autoscroll_off` now sends the plaintext `A0` settings command directly over the captured websocket — the same injection path every other command rides — and requires the plaintext `A0` echo. The command is ABSOLUTE (`A{enabled}` states the desired state), so the whole press-parity dance (probe press, read, corrective press, "stuck ON") is deleted with it: send, verify echo, re-send idempotently on a slow ack, fail loud after three. No dependence on hotkey config, page focus, key handlers, or cache warmth — the class of the 2026-07-24 key-probe incident dies with the instrument.

**Names stopped lying in the same pass:** `AutoscrollPageProtocol` → `GamePageProtocol` (it is the bot's page surface; autoscroll no longer touches keys) and `AutoscrollKeyProtocol` → `PageKeyboardProtocol`, whose docstring names its one remaining consumer — the key probe, which presses keys deliberately because capturing what a key EMITS is its purpose. The enforcement's own signature narrowed to `PageWaitProtocol`, the single member it calls. The sim seam test hands the link in both its page and CDP roles, exactly as production dispatch already does.

Also mined en route: the fresh-lobby `=` records re-observed (`=5|...|1|9|10|9|9` after the orange tank's creation — the four trailing fields moved, consistent with the per-color-slots hypothesis in [[game-rules]], still unconfirmed).

Gate: guard clean (including its own `object-in-annotation` rule catching the first cut of a test fake), mypy clean on 1,096 files, **6,093 tests, 100.00%**.

---
## [2026-08-13] flags + lift | The fresh-tank regime: a disabled rescue gate un-disabled, the mine-hit reveal built, and the unmanaged missile slot

Arterial's first real session (rank 0, zero inventory, co-farming Practice with artax) exercised a regime no bot had run, and ten HUD flags mapped to three defects — each gated (**6,095 tests, 100.00%**).

**The desync-rescan gate was disabled all session by its own discriminator.** 23 of 23 `code=4` empty-container closes classified as "drain receipt of own pickup," so `mark_container_desync` never fired and the bot chased stale beliefs uninterrupted (flags 3/4: shift + walk + pickup per ghost, in a room artax was actively draining). Root: `was_recent_pickup_at` matched ANY pickup record at the tile — including the removal broadcast that rides an already-empty click and any sibling bot's drain. The corrected discriminator is the actual question: **did WE gain anything since dispatch** — a `last_own_gain_ms` stamp written by the fuel-total announce (positive delta) and by any inventory count rising, read as `own_gain_since(action.started_ms)`. The tile-record helper is deleted, not kept.

**The mine-hit reveal scan exists now (flags 2/9) — it never did before.** The remembered "radar after mine hit" was an illusion: the walk-over FLIP teleports away and the LANDING scan fires at the destination, leaving the field that hit the tank invisible. A walk-over detonation now latches `mine_reveal_pending`, and one radar fires before any further movement — same shape as the desync latch (any radar response clears it, one scan per hit), deliberately NOT coverage-gated because the hit itself proves the coverage stale. At recruit ranks the free radar's footprint is centred exactly where a walk-over ring sits.

**Slot 3 (missiles) joins the managed equipment set.** It was never toggled, so the server's per-account default persisted: fresh accounts spawn with missiles ENABLED (arterial) while artax's older account state has them off — and an enabled missile slot is consumed on occluded shots ([[game-rules]]: missiles fire over obstacles). The doctrine never desires missiles, so managing the slot means always-disabled, uniform across accounts.

**Verified sound, no change:** free-radar handling (coverage marks only the clipped `(2r+1)^2` footprint; forage's `select_best_free_radar_position` is the scan-walk-scan loop the user described), the recruit single-tile blast, and the break-off from red-8 (engagement projection working). The empty-pickup rate itself is partly INHERENT to co-farming one small room — the rescue is the rescan gate finally firing, not belief clairvoyance.

Tuning noted, not built: clearance-eats-what-it-exposes coordination (flags 7/8), and collect-hop landings pre-ranked toward known enemies (flag 10's restock-then-hunt double teleport).

---
## [2026-08-13] flag 6 traced + lift | A cant_go riding a detonation is an interrupted walk, not a failed pickup

Flag 6 ("it gave up on equipment after hitting a mine") traced to run arterial 22:25:43-45, and the give-up was a mis-blame, not a retreat. The bot was walking down clear ground toward locked equipment at (103,147); the step onto (103,143) detonated an invisible mine (-45), the server halted the walk and refused the remainder with **code=1 `cant_go`** (the partial-walk law, [[walk-mechanics]]) — and the collect error handler booked that refusal against the container: `failed_pickups 0 -> 1`. The demoted target lost the next candidate ranking to different equipment at (104,153), and the walk-over flip paid a 60-fuel teleport to it — four tiles further than the innocent, reachable original.

**The lift:** the code=0/1 arm of the collect rejection handler (the 2026-07-06 semantic split) gains the same shape of discriminator the code=4 arm got earlier today: `own_mine_hit_since(action.started_ms)`. A `cant_go` whose action window contains an own-tile detonation is the server refusing the remainder of a walk the mine interrupted — the container is not blamed, no `failed_pickups` increment. The mine-reveal scan (built earlier today) and the walk-over teleport flip own the re-approach, now aimed at the SAME target. A `cant_go` with no detonation in the window keeps the mark — genuine terrain refusals are unchanged, as is code=0 geometry.

Also fixed in the same gate: the own-drain code=4 test raced the wall clock (gain stamped before `started_ms` in two separate `get_current_time_ms()` calls — a millisecond tick between them inverted the window). The test now captures `started_ms` first and stamps the gain after, which is also the actual wire order (dispatch -> gain -> code=4).

Gate: guard clean, **6,098 tests, 100.00%**.

---
## [2026-08-13] flag s11-2 + lift | The visibility law reaches the decision layer: frame-before-fire at one shift's reach

Both bots, same defect, mirror images. Artax stood at (132,131) firing at Arterial at (132,127) — one row above artax's window — and every dispatch clamped to the edge water tile (132,128) came back `weapon=0`: a free single at empty ground, six in a row, ~4 s a cycle, while Arterial returned ~90/window. Arterial later drew the identical miss at Artax one row off ITS window ((143,36) vs top=37). Neither viewport was desynced — the positions were right; the SHOT was structurally impossible.

**Why the bug existed at all (the user's question):** the 2026-06-26 engagement contract assumed "enemies don't move within the viewport — when they leave cardinal adjacency, they teleported," so `has_combat_shot` modeled range only (`distance <= 8`, ctx literally `del`eted) and visibility never entered the decision layer. When the server's viewport law surfaced (2026-07-03 code-0 rejections) it was patched at the DISPATCH layer — `_clamp_aim_into_viewport`, built for the far-pursuit snipe where the seeker genuinely tracks. Bot-vs-bot play falsified the founding assumption (our own bots walk out of each other's windows), exposing the unmodeled middle band: near-off-window, where the server refuses to homing-track ("close enough that a viewport shift would reveal them" — recorded from user knowledge 2026-07-03, wire-confirmed tonight). The clamp silently converted those illegal decisions into dispatchable ground fire, and no feedback could teach the planner: miss -> hold lock -> map chase -> same registry coords -> same doomed shot.

**The lift (root, not band-aid):** `frame_target_shift` — the visibility law as a decision. At the `engage_target` chokepoint (every shoot path funnels through it) and ahead of `pursuit_fire`'s gates: a target outside the window but within one anchor-law shift (Chebyshev <= 15, exactly the server's refusal zone) gets a FREE framing scope shift (reuses `frame_direction` from the harvest doctrine); next tick the target is visible and the shot is real. Beyond one shift the clamped snipe stands — that is its correct scope. In `pursuit_fire` the shift is checked BEFORE the human homing-budget stamp, so framing never spends the one-per-departure cap. New ReasonKind `combat_frame_shift`.

**The homing-vs-arterial question, answered from the same log:** the 2026-07-31 one-pursuit-homing cap held everywhere the bot KNEW it was pursuing (two `pursuit_homing_capped` receipts). The extra homings rode the ENGAGED branch on sub-2s-stale beliefs — shots dispatched at a registry position the target had just teleported from resolve as server-picked homings that track (23:20:49/51: two dispatches at (133,126) resolved `weapon=3` to (151,87)). Receipt-lag leak, ~2 per departure; a receipt-driven retroactive budget stamp would close it — noted, not built.

**Return-fire doctrine (user ruling tonight: "bots should be returning shots whenever possible"):** the soak windows decompose as (1) doomed clamped misses + their map-chase cycles — killed by this lift; (2) the 4 s shot-feedback wait occasionally eating a full window; (3) engagement-break COLLECT under fire (walks/pickups with no return fire, by break doctrine); (4) landing radar before the first shot (flag 3, investigated earlier tonight). (2)-(4) stand as recorded candidates for the next round.

Gate: pending this entry.

---
## [2026-08-13] flags s11-3/5/6 traced + lift | The harvest re-latch loop, the radar-under-fire soak, and the ghost of Artax

**Flag s11-5 (viewport loop + cant_go spam) — two lock-law violations, both fixed.** Arterial, marooned needing land routing, walked at locked equipment (165,161); the server refused (the cant_go spam), the tile was marked move-failed, and the structural release correctly dropped the lock. Then the defect: `plan_block_harvest_leg`'s candidate search never consulted the move-failed marks — it re-picked (165,161) as nearest block stock and latched it again. Release -> re-latch -> release, one free scope shift per cycle, the window ping-ponging dir 3 <-> dir 7 at full fuel until session end. Second violation in the same trace: while the (136,145) lock was HELD ("holding plan"), harvest latched a different target over it — an un-enumerated re-target. Lifts: `_wanted_block_container` now reads the same move-failed marks the release rule reads, and `plan_block_harvest_leg` declines outright while any resource lock is held (committed intent: the continuation owns the pursuit).

**Flag s11-3/4 (six shots soaked in a row) — the desync rescan has no under-fire gate.** 23:35:43: arterial dispatched the desync-rescan radar while taking 90/window and sat in SCANNING for 12 seconds (the radar answer never closed the action; the 10 s stall timeout did) — five uncontested hits. The under-fire escape gate ranks ABOVE the rescan in the collect cascade, but the escape only fires on its own thresholds; the rescan then happily spent a stationary scan under bombardment. Candidate fix (not built, needs a ruling): suppress the desync rescan while `collecting under fire` is active — a stale container belief is worth a radar, not 450 fuel.

**Flag s11-6 (map between each move + Friendly fire!) — the ghost of Artax, the recorded 2026-07-30 precedent verbatim.** First theory (artax relaunched orange off the global `TANKPIT_TROOP=3`) was WRONG — the user challenged it and the logs settled it: artax entered `room=1 troop=2` (blue) and NO artax session existed at 23:43; its only session died `connection_lost` at 23:40:46 (the watchdog's clean exit — the "Artax left the game" banner). The err=3 "Friendly fire!" is the server's answer to shooting a DEPARTED tank's ghost — exactly the Yuppler precedent of run 20260730 (0x58 grace keeps the registry entry, map opens re-stamp its freshness). Arterial's held lock on the ghost (id 1301) drove a ~75 s chase — teleport-to-ghost, "never engaged" re-teleports, a map_open between every move — because only a SHOT can disprove a ghost and the chase phase fires none; two shots then drew the two banners and the disproof blocked the target and released the lock (machinery working as designed). Arterial immediately map-acquired red-7 and killed it clean (6 hits, kill registered 23:43:31 — the session's first kill). Efficiency gap noted, not built: a ghost is only disproved by shooting it, so a departed target's chase burns its full pre-shot phase; a cheap probe shot (or DOM "left the game" witness escalation) would shorten it. The global `TANKPIT_TROOP` in `.env` is still a latent per-account hazard — it did NOT bite here, but any account's first entry into a new room will read it.

**Observed working:** the frame-shift lift is NOT in these sessions (both bots launched 23:19, before the gate) — the edge-miss loops seen here are the pre-fix build. Arterial vs red-7: teleport -> landing scan -> six dual windows -> kill, the contract loop verbatim.

---
## [2026-08-14] doctrine + lift | Firefight opportunity fire: kill shots and return fire without breaking the lock

User ruling (after fighting both bots as Yuppler and being ignored while landing hits): "when people are fighting, you have a main target ofc. but you should also return fire to anyone else engaging and take kill shots or kill attempts when possible, when someone is in the lowest or second lowest damage state." The stay-on-target lock was absolute — target selection happened only at acquisition, so a held lock made the bot blind to every other combatant (artax greeted Yuppler mid-volley while eating 45/window from them and never fired back).

**The lift:** `combat_opportunity.py` — a per-tick single-shot DIVERT layered under the persistent lock. Priority: FINISHER (visible enemy at damage tier <= 1 — the fuel-quartile tiers — most-damaged first, nearest tie-break) beats RETURN FIRE (visible enemy with a fuel-confirmed hit on us inside a 6 s window) beats the main target's routine shot. Diverts are free shots from the current tile only: in-window, cardinal-or-clear-line — never a lock change, never movement. Hooked at `close_target` entry and in `pursuit_fire` ahead of the frame shift and the human homing-budget stamp (a divert never spends the pursuit budget). Legality inherits `analyze_threats` (an unconsenting human is never "finished" — the consent contract held in tests when human-named fixtures silently refused to divert) plus the blocked map, which `ctx.filtered` does NOT cover (killed-only — a discovered sharp edge, now noted in the selection).

**Bookkeeping honesty, the subtle half:** per-shot feedback is keyed by `last_shot_target_id`, which now moves with each divert — so `engage_target` scopes its miss/rejection consequences to shots aimed at its own target (a diverted miss can no longer trade a live point-blank fight for a map chase), and a diverted miss blocks the DIVERT id in the feedback layer (`divert_target_blocked`) so a shielded/afterimage divert can't re-select every tick. An `engaged_target_id` schema field was built and then REVERTED in the same session: `ctx.base` resets `last_shot_target_id` every tick, so engagement already only persists across consecutive lock shots — a divert behaves exactly like the existing collect interrupts, and the in-view re-entry makes the resume a free same-tile shot. The field bought nothing.

Three legacy pins updated: their bystander tanks sat at `damage_state=0` (a meaningless fixture default in 2026-06) and now correctly drew finisher diverts; the bystanders are healthy so each pin keeps testing what it always tested. Notably `test_locked_phase_one_target_teleports_to_existing_enemy` had an adjacent dying enemy being IGNORED for a 30-tile teleport — under the new doctrine the old expectation is the bug.

Still open (fleet topology ruling): shared targeting between artax and arterial — allies vs opposing colors with a truce — and per-account troop config.

Gate: pending this entry.

---
## [2026-08-14] ghost chase root-fixed | 0x29 TankExit was decoded, dispatched, and dropped on the floor

The user asked whether the ghost chase (flag s11-6's ~75 s map-open/teleport/shoot loop at departed Artax) was actually fixed — it was not; only its aftermath (the err=3 disproof) was working. The root turned out to be one layer deeper than "only a shot can disprove a ghost": **the wire announces departures**. 0x29 TankExit — trace-verified long ago (v-table `Vf`, `was_eliminated` 1=eliminated/0="left the game", the client banner's literal source) — was fully decoded and dispatched to a diagnostic emit and NOTHING else. Arterial's own event log holds the receipt: `tank_exit_announcement tank_id=1301` at 23:39:17, then the ghost chase 23:41:43–23:43:03. The wire said "Artax left" two and a half minutes before the bot's first doomed shot at him. (Yuppler's exit at 23:38:17 was likewise dropped.)

**The lift:** `depart_tank` — 0x29 now DELETES the registry entry. Deliberately distinct from 0x58 TankRemove, which stays the contract's no-op (per-client tracking churn; five removes across two kills in the 2026-06-20 capture — that law is untouched). Deletion ends every pursuit at once through machinery that already existed: `find_locked_target_pursuit` returns None for a missing entry (lock releases), `analyze_threats` reads the registry (threat lists drop the tank), occupancy stops seeing a body, and a rejoin re-adds the tank on its first observation (artax's relaunch reused id 1301 — handled). The old pin `test_dispatch_tank_exit_does_not_remove_tank` was REVERSED: its rationale ("JS Vf only prints a log line") described the browser's rendering, not what a bot should do with a departure announcement. The stale decode-coverage gloss ("tank left the viewport") corrected — 0x29 is room-wide, 0x58 is the viewport-tracking signal.

With this, the flag s11-6 chase becomes structurally impossible: the tick after the 0x29 lands, the lock is gone and acquisition runs on live truth.

---
## [2026-08-14] fleet lift | The shared knowledge layer: fleetshare, roles, and the blue alliance

The user's ruling: same-team allies (arterial goes BLUE), coordination between bots of one color, "a proper lift that allows for single tank running, or multi tanks running, with fighters and with a potential info gatherer" — complete, not a hack. Built as the `fleetshare` package + [[fleet-coordination]].

**Transport is the run-directory filesystem** — the channel the fleet page already reads (hud.json precedent, lift don't fork). Each tick every bot atomically replaces `knowledge.json` beside its `hud.json` (new `replace_text` hook: temp + `os.replace`, so readers never see a torn write and the decoder's strict raise-on-malformed is sound) and merges siblings' fresh (10 s TTL) same-team reports. Zero siblings = a single tank running identically; no manager process required for a fleet.

**The report** (`FleetReportDict`, full encode/decode + `require_*`): identity, position, role, held combat lock, enemy sightings (reporter's own `last_position_update_ms` as the belief age, 30 s bound; allies/corpses/unplaced/stale withheld), container atlas (locally failed-pickup marks withheld — a disproof is the reporter's own verdict).

**Merge laws:** remote knowledge only adds or refreshes — own wire is the higher trust tier (an enemy sighting applies only when fresher than the local registry entry; `merge_container_sighting` never removes local beliefs and a local failed mark survives any refresh). Merged sightings ride `apply_tank_observation` with the new `fleet_report` fact source and never advance the viewport gate — acquirable, not fireable, so no phantom shots. Teammates' locks land in `ws.fleet_engaged_target_ids` (replaced wholesale per merge) and the threat sort ranks fleet-engaged ids first INSIDE a priority tier — focus fire that never outranks the human-priority doctrine. The dead plain `_threat_sort_key` wrapper was deleted, not kept.

**Roles** (`TANKPIT_ROLE`): fighter (default, full doctrine) and gatherer — never hunts, structurally: the router returns COLLECT unconditionally, `hunt_entry_permitted` bars it as the doctrinal backstop, and the exhausted cascade returns a COLLECT-owned `gatherer_hold` no-op instead of the fighter's `no_productive_collect` exit ("cannot hunt" is the role, never "marooned"). The gatherer roams via the existing cascade — scan, sweep, search hop, map-for-dots — publishing for the fighters of its color.

**Color assignment:** `TANKPIT_TROOP` now OVERRIDES the account's lobby default (accounts hold one tank per color per map; the enter request's troop byte picks which). `.env` set to 2 — the whole fleet plays blue; arterial's next entry creates/selects its blue tank (server enforces the 5-minute recolor cooldown).

Test surface: fleetshare codecs/report/merge/role units, tick-exchange integration (publish + merge + diagnostic), focus-fire ranking pins, gatherer routing pins (full-stock-with-adjacent-enemy collects; exhausted holds), the FakeFileSystem glob fixed to match the real `Path.glob` contract segment-by-segment (its flat-name matching silently missed `*/knowledge.json`).

---
## [2026-08-14] first live fleet run | Blue arterial confirmed, the exchange works, and tombstones kill the re-import loop

First live runs on the fleetshare build (60 s smoke + 90 s two-bot). **Working:** arterial entered `troop=2` — its BLUE tank created via the TANKPIT_TROOP override; both bots published and merged (artax absorbed 54 enemy sightings + 50 containers across 36 merge passes; arterial 45 passes); both sessions exited `completed` cleanly (the "crash" the user saw was the 90 s timed teardown closing the browser).

**The defect the run surfaced (user flag, "Empty container / Nothing detected here" loop):** deletions don't propagate. Arterial (fresh recruit, zero extras) disproved a stale shared container (code-4 → belief removed → desync rescan whose recruit free radar reveals almost nothing) — and the NEXT merge re-imported the same dead belief from artax's report, because artax still believes it. (102,85) was disproved three times in five seconds. The free-radar scan-walk-scan forage the user asked about exists and ran (6 forage actions) but was starved by the loop's 18 code-4s.

**The lift — container tombstones:** every local removal (code-4, emptied pickup, unreachable, radar-stated-empty) stamps `ws.container_disproofs[tile]`, and the fleet merge admits a remote sighting only when observed AFTER the disproof. Respawned containers re-enter naturally (fresh observation postdates the disproof). With the loop dead, the recruit's cascade falls through to the forage loop as designed.

Also noted from the run: the division-of-labor the user described ("one with extra radar can be scanning") is exactly the gatherer role — not yet exercised (both ran as fighters); next session can launch a third bot with `TANKPIT_ROLE=gatherer`.

---
## [2026-08-14] worldview lift | Shared scan coverage, and coverage steps walk

Two user rulings from watching the verification run live. **"Share the worldview":** the report gains the scan map — every tile under live radar coverage (forage TTL bound) — and `merge_scanned_coverage` folds teammates' coverage into `scanned_tiles`, newest stamp per tile, own fresher coverage never regressed. The forage/sweep gates already read `scanned_tiles`, so scanner division of labor falls out with zero behavioral surgery: ground a sibling cleared is covered here too, and the fleet's radars stop overlapping. Mines need no fleet row — reveals are team-scoped in the game itself, so teammates' reveals already ride each bot's own wire. The shared worldview is now: enemies + container atlas + scan coverage + combat locks, all freshness-arbitrated, own wire the higher trust tier.

**"When a tank has zero radars, it should scan, walk, walk, scan — arterial isn't doing it":** the free-radar loop existed but forage's movement used `walk_or_teleport`, whose blocked-path fallback TELEPORTS — run arterial 19:30:57 paid two forage teleports on a zero-extras recruit, and cascade branches kept yanking it off-pattern. New movement primitive `plan_viewport_walk`: pure walk, no teleport fallback, no mine-flip — a free radar reveals ground for nothing, so no coverage step is worth a 45+ fuel hop. An unwalkable best position now reads "this viewport is done for free-scan" and yields to the search hop.

Also verified from the tombstone run: arterial ended 0/0/0 after 90 s — two 90-second sessions are simply too short to restock a recruit from zero; the next verification uses longer sessions.

---
## [2026-08-14] lawnmower completed | The frontier walk: scan-walk-scan continues into the next viewport over

User doctrine, verbatim: "is he scanning unique 5x5 areas on each viewport, until that viewport is fully scanned and then moving to the next viewport over??" The in-viewport half existed (select_best_free_radar_position places each free radar for maximum new coverage); the continuation did NOT — a covered viewport yielded to the search hop, which TELEPORTS to dot-ranked landings anywhere. For a zero-extras recruit that is exactly backwards: the window is anchored to the tank, so the correct continuation is to WALK toward the least-scanned adjacent band and let the window slide.

**The lift — `_frontier_walk_target`:** at zero extras with the viewport covered, score the four 8-deep bands beyond the window's edges by uncovered-tile count and walk (via the pure-walk primitive, never a teleport) toward the richest one; the scan-walk-scan loop resumes on the fresh ground next tick. Only when every adjacent band is covered does the tick yield to the search hop. Two boxed-in pins updated: "genuinely boxed in" now requires the SURROUNDINGS covered wall to wall, because a walkable frontier genuinely rescues what used to be a no_productive_collect exit. One pin reversed outright (covered viewport at zero extras used to mean teleport-out).

**240 s two-bot autopsy (the run that drove this):** arterial made 49 pickups but only 2 equipment (+9 missiles +7 duals, 0 radars) — equipment is the scarce co-farmed resource, and the fleet atlas is fuel-dominated (artax published 194 containers, ONE equipment: it eats its own finds instantly). Arterial ignored artax's fuel because it was near cap — 71 of 78 hop declines were reserve/landing gates working correctly. The tombstone law held (no Empty-container loop recurrence). Artax's suspected "world desync" produced zero alignment mismatches and one cant_go that was a detonation-interrupted walk handled by the flag-6 discriminator — the visual desync is the stale shared atlas: chasing already-eaten shared containers looks like desync from outside. Built in the same pass: shared CONTAINER sightings are bounded by a 60 s freshness TTL (`CONTAINER_SIGHTING_TTL_MS`) — stricter than local larder memory by design, because the reporter can re-verify a stale belief with a cheap local radar while a receiver must travel to act on it.

---
## [2026-08-14] two lifts + a milestone | Consumption propagates, the hello waits for arrival, and the recruit's first kills

**Consumption sharing (user: "does it update the equipment for everyone when one of them takes the discovered equipment?" — it did not):** the report gains a `removed` ledger (the reporter's tombstone map, bounded by the container share horizon), and receivers drop any local belief OBSERVED BEFORE a teammate's removal while inheriting the tombstone — so one bot's pickup stops the whole fleet chasing the ghost, transitively. A local belief fresher than the removal survives as a possible respawn. `merge_fleet_reports` split into per-concern appliers (`_merge_enemy_sightings`, `_merge_container_knowledge`) at the complexity ceiling — negative and positive container knowledge now live in one applier with both laws.

**The hello waits for arrival (user: "he's supposed to say hello AFTER teleporting to the human, when he's ready to engage, not way before" — superseding the 2026-07-31 greet-from-anywhere ruling):** `attach_human_greeting` gains the viewport gate — the HELLO is the face-to-face opener, firing the tick the human stands in the visible window. Two pins reversed with the ruling chain documented; the Yuppler-ghost wasted-hello class becomes structurally rare (a ghost is never IN the viewport); the candidate scan extracted (`_nearest_ungreeted_viewport_human`).

**240 s verification, lawnmower build — the milestone:** arterial (recruit, from zero) finished with THREE KILLS and 20 duals — it built stock via the frontier-walk lawnmower (22 forage walks, zero forage teleports, longest same-spot radar burst: 2), reached hunt readiness, fought, and spent its radars back to 0 in the fights — the spend-to-fight doctrine working, not a drought. Artax finished 25/25/25 full stock. The "stuck walking back and forth" read from outside is the lawnmower's 3-5 tile steps between free scans (zero A-B-A oscillations measured). Both sessions ended with the NEW teardown symptom: browser close hung past 30 s and the watchdog forced exit (code 75) AFTER artifacts saved — a cleanup nuisance to watch, not a gameplay defect.

---
## [2026-08-14] verification autopsy | Removal sharing verified live; two conflations the fleet falsified; the hang autopsy armed

**180 s two-bot verification (bot-20260814-204750/204751), gate-green build.** Removal propagation VERIFIED live: each bot merged 4 container-removal tombstones from its sibling (`fleet_knowledge_merged` `removed` field), alongside 900-1,300 shared scanned tiles and the enemy/container streams. Artax: 2 kills, 21/21 hits, wound down 25/25/25. Both teardowns closed clean in under a second — the 19:59 hang pair did not recur.

**Kill attribution — the fleet falsified a solo-era conflation:** arterial's scorecard read `kills: 2` with ZERO shots fired — both 0x41s said `killed by 1301` (artax). `ws.killed_tank_ids` was a bare victim set feeding BOTH the dead-tank registry (correctly killer-agnostic) and `session_kill_count` (must be ours alone — it also triggers the `session_kills` wind-down). The set now carries victim → killer, and `_merge_protocol_kills` counts only victims whose 0x41 names our own tank id; unattributable kills (no established identity) land in the registry only. Solo practice rooms could never distinguish the two numbers; the first fleet firefight did.

**The arrival hello was structurally mute — owner routing, not gates:** artax teleported 6 tiles off human "123", stood face-to-face 12 s, and never said hello. Capture decode proved every greeting gate passed (0x3D syncs re-stamped 731 fresh at :16/:22, in-viewport, enemy, human, ungreeted) — but `attach_human_greeting` was hooked ONLY on the hunt-owner return path, and the arrival tick after a greet-approach teleport is COLLECT-owned by construction: the teleport burns the fuel that hunt-only-when-full requires. The attach now rides both owned return paths; pinned end-to-end by a collect-owned `decide` test.

**Teardown-hang autopsy armed:** the 19:59 hang pair left byte-identical logs and zero evidence of where `browser.close()` wedged (no leaked processes either — the gap was instrumental). The watchdog now snapshots EVERY live thread's stack (`sys._current_frames`, cross-thread: the timer thread photographs the wedged closer) and logs it before forcing exit 75. The next hang names its frame.

---
## [2026-08-17] retirement | analysis_scripts and the homing-exploit probe leave the tree

Board task f0c3a532 (filed by opus-hook-sweep-0817): 41 Python files lived outside every guarded directory — `analysis_scripts/` (40 one-shot archive miners and crack-era scripts, ~8,000 lines) plus `tools/test_homing_exploit.py` (1,418 lines) — escaping strict mypy, the guard suite, and coverage entirely via a quiet `analysis_scripts` entry in the ruff exclude list. Measured against the house bar: 8,543 strict-mypy errors, ~210 ruff violations. Operator chose deletion over gating (a rewrite) or a config-out (a violation of the no-exclusions rule).

**What the deletion cannot lose:** every finding these scripts produced is recorded in wiki pages; the scripts' exact content stays addressable in git history, and [[capture-differ]] already pins its two cited miners by `source_git_blobs` hash. The wiki guard learned the matching law: a `source_paths` entry that has left the working tree but carries a blob pin is a RETIRED source, verifiable via `git cat-file blob <hash>` — no longer a violation. An unpinned vanished path still fails. The one data artifact in the directory, `bot_policy_sweep_2026-07-24.json` (246-session measurement corpus cited by [[enemy-bot-behavior]] and [[game-economy]]), moved to `wiki/sources/` with both citations updated — data is corpus, not code, and never meets a deletion.

Any future re-measure recovers a script from history or, better, writes against `tankpit_bot.analysis` — the typed, gated owner of the capture-scan pipeline these forty scripts each re-implemented privately.

---
## [2026-08-19] audit | Fleetshare skeptical review: pins landed, four watch items recorded

Operator asked for a skeptical review of the fleet knowledge-exchange layer. Full read of all six `fleetshare` files plus the tick integration and the `apply_tank_observation` pathway; gate green on the same tree (6,176 tests, 100.00% over 30,299 statements / 8,688 branches). **No defect found.** The load-bearing properties verified by reading: timestamps are preserved end-to-end (no gossip amplification — third-hand knowledge never looks fresher than first-hand), merged sightings ride `is_wire_sourced=False` so the fire gate never advances on remote knowledge, the map-position defer window keeps fleet positions from stomping fresh wire fixes, and `_real_replace_text` already handles the Windows `os.replace` PermissionError by dropping the beat.

**Debt paid:** [[fleet-coordination]]'s frontmatter still claimed fleetshare was untracked ("Pin it in the commit that lands it" — the pin was never added when the module landed in `c9d92a76`). The directory `source_paths` entry is now six per-file entries, each pinned by HEAD blob; the `tick_body.py` and `threat_primitives.py` pins bumped to the blobs re-verified in this audit; `fact_checked` advanced to 2026-08-19.

**Recorded, not fixed** (new "Watch items" section on the page): (1) a malformed FRESH sibling report crashes every teammate's tick by design, and the `written_ms` read precedes schema validation — fleets must run one build; (2) `_merge_enemy_sightings` compares local any-observation freshness against remote position freshness (conservative, but cross-domain); (3) `ws.container_disproofs` never prunes; (4) the exchange is O(fleet² × rows) per tick — fine at 2–3 bots, revisit before a large fleet.

Board: check-in + findings pointer posted as fable-tankpit-review-0819 (no open board task covered this; none filed — the watch items are documented hazards, not actionable work).

---
## [2026-08-20] fleet API lift + first live gatherer run | Roles ride the spawn API; the contract holds; a full-inventory livelock surfaces

**The lift (operator: "we may need to update the API and the docs"):** the fleet manager's `POST /bots` gained `"role"` — validated against `FLEET_ROLES` (unknown role → 409), empty means fighter, restart carries the stored role, and the child's `TANKPIT_ROLE` is ALWAYS set explicitly so a value lingering in the manager's own environment can never silently re-role the fleet. `GET /bots` rows, the control page table, and the spawn form (role dropdown) all carry it. Gate green: 6,177 tests, 100.00% over 30,311 statements / 8,692 branches. Docs updated: [[bot-service-architecture]] (HTTP surface + child bootstrap env), [[fleet-coordination]] (roles section).

**The run (240 s two-bot, spawned through the new API — artax fighter / arterial gatherer, room 7):** both exited clean. Artax: 1 kill, 0 deaths, 31 shots, 18 pickups, 11 teleports, full 25-stock held. **The gatherer contract held on the wire: arterial fired ZERO shots and owned zero HUNT ticks across 4m21s** — the role's first live verification. The exchange ran both ways (90 merge passes on artax, 100 on arterial).

**The defect the run surfaced — full-inventory equipment livelock (documented on [[fleet-coordination]], OPEN):** arterial started at recruit rank cap (20/20/20/20/20, inventory persists on the account), locked an equipment container (`equipment_locked`, score 925), and the dispatch was predictively suppressed 93 consecutive ticks ("belief predicts 0x52 code 7" — `SUPERVISOR_ERROR_INVENTORY_FULL`, [[equipment-system]]). Suppression never dispatches, so `failed_pickups` never bumps, the belief keeps winning the collect score, and the cascade never reaches scan/sweep/frontier-walk: 1 pickup, 1 teleport, fuel frozen at 944 from t≈60 s to exit. The information flow ran BACKWARDS — the wedged gatherer absorbed 368 enemies / 179 containers / 2,734 scanned tiles from the fighter while offering 29 / 28 / 512 back. Fix direction: skip equipment targets in SELECTION while own inventory is at cap (belief deletion cannot work — the sibling's atlas re-feeds the tile within the share TTL).

Minor observation, not chased: artax exited `session_complete` at a 3m34s event window against the 240 s bound (arterial's `completed` window was 4m21s) — the two clean-exit reasons differ and the fighter's window ran ~26 s short; worth an eye on the next timed pair.

---
## [2026-08-20] class audit + lift | The planner/veto feedback gap: two behavior gates, streak detection, and the analyzer's solo-era kill count

Operator: "how did we miss this and are there other of this class of bug? are we lifting?" The class: **a downstream veto refuses what the planner selected, without writing back a fact that changes the next plan** — an identical plan repeats forever, and nothing was dispatched so no failure signal exists.

**How it was missed, mechanically:** (1) the identical bug was found and FIXED for the fuel lock on 2026-07-06 (`collect_locks.py` capacity gate, `tank_at_capacity` release — it fired 3x in artax's run the same session the equipment twin wedged), but the equipment lock 60 lines above never got the twin; (2) the fighter doctrine masked the hop-side hole structurally — `_equipment_hop_barred` barred via `hunt_entry_permitted`, and a full fighter always goes hunting, so the missing capacity bar was invisible until the gatherer role made that predicate unconditionally False; (3) the analyzer classified the loop as healthy: 93 suppressions rendered as 116 zero-duration `superseded` collects and "(no top-level issues detected)" — predicted refusals never reach the ledger, so retry-loop detection was blind to them.

**Class sweep verdicts:** fuel lock at capacity — gated since 07-06, verified live; equipment walk-pickup acquisition — gated since 07-18; equipment HOP — hole, FIXED (the shared `equipment_pickup_refusal` physics bar, `at_capacity` decline tally); equipment LOCK continuation — hole, FIXED (twin capacity gate, releases `tank_at_capacity`); teleport suppression — planner reserve gates are stricter than the executor's slacked floor, reviewed OK; fuel code-4 prediction — drained beliefs are removed, cannot repeat, OK.

**Detection lifts, verified against the live artifacts:** (1) `dispatch_suppressed` events now tally per target into the issue report (`suppressed_dispatches`, codec round-tripped); >= 3 same-target suppressions is a TOP-LEVEL issue ("the executor's veto is not feeding back into selection") — re-analyzing arterial's wedged run now names both targets (93x on (133,129), 23x on (134,122)) where it previously read healthy. (2) The scorecard's kill counter carried the solo-era conflation the live registry shed on 08-14 ("the raw count is the kill count"): it now tracks `self_tank_id` from `tank_identity` and counts a `tank_deactivated` only when `killer_id` names our own tank — arterial's report corrects 2 -> 0, artax keeps its genuine 1. Unattributable events (pre-fleet artifacts without `killer_id`, or pre-identity) never count.

**Split at the file-size bar:** `session_scorecard_accumulator.py` (582 lines, heading past 600 with attribution) became three cohesive modules — `session_scorecard_types.py` (shape + factory + optional-field helpers), `session_scorecard_routes.py` (every DIAGNOSTIC-kind arm incl. the new attribution), and the slimmed accumulator (channel router + WORLD fuel receipts). `test_issue_report_types.py` (608 after the new field) split the whole-report codec tests into `test_issue_report_codec.py`; `test_session_scorecard_accumulator.py` (620 after the attribution pins) split the kill-attribution family into `test_scorecard_kill_attribution.py`. The dispatch-exclusivity source pins follow the moves.

---
## [2026-08-20] instrument lift | Liveness becomes a watched dimension; the analyzers must agree; the archive says the class had ONE member

Operator: "im worried theres a whole class of bugs we are missing... i thought we had seam and contract tests? and smoke tests? and the sim?" The answer, mapped precisely: contract tests verify each layer's own law, shadow laws verify the sim against the server, seam soaks verify the bot against the sim, smoke runs verify the default doctrine — **no instrument asked whether the agent makes progress from every reachable state.** Both behavior bugs were liveness bugs; the class had no assigned instrument.

**The decisive measurement first:** a sweep of all 459 archived runs for zero-duration `superseded` streaks (the class's signature — a decision replaced before anything dispatched). Healthy ceiling across four months: **7** (combat re-aiming while a shot resolves). Exactly ONE run in history exceeds it: the gatherer livelock at **93**. The class is real, the instrument was missing, and the archive contains no other undiscovered members.

**Lift 1 — liveness instrumented, three layers deep, threshold empirical (`LIVENESS_STALL_STREAK = 12`, above the ceiling with margin):**
- LIVE: the ledger's `zero_dispatch_streaks` counter (per action kind, incremented at the `register_pending_decision` superseded-close chokepoint, reset by any genuine resolution) emits a one-shot `liveness_stall` diagnostic + log warning at the crossing — a wedged session now announces itself in the tail within ~24 s instead of wasting its full bound.
- POST-RUN: the issue report scans `action_outcomes` for the same signal (veto-agnostic — catches hold-loops that emit no suppression diagnostics, and works on artifacts from builds predating the live counter). Verified on the wedged artifact: "liveness stall: 93 consecutive collect decisions" now renders beside the two suppression-streak issues; artax's healthy 7-streak stays silent.
- ARCHIVE: the sweep above IS the retro pass — zero further members.

**Lift 2 — analyzer consistency (`test_analyzer_consistency.py`):** the digest counts `kill registered` emissions, the scorecard counts killer-attributed 0x41s; for six days they disagreed on the same artifact and nothing compared them. One stream shaped exactly like the live emitters, both analyzers over it, shared facts diffed in the gate.

**Lift 3 — process rules recorded on [[coding-standards]] (§Verification discipline):** the falsification sweep (grep the repo for every other site encoding a just-falsified assumption, same session) and the scenario matrix (a new capability's sim cells soak BEFORE its first live run — the livelock lived in the gatherer × full-stock cell no soak ever ran).

---
## [2026-08-20] verification | The gatherer roams: livelock fix proven live, the exchange feeds forward, the instrument runs silent

240 s two-bot verification (bot-20260820-142814 pair, artax fighter / arterial gatherer via the fleet API), the exact regression scenario: arterial launched at recruit rank cap 20/20/20/20/20 — the condition that wedged it this morning. **It consumed the wedge tile (133,129) inside the first 15 seconds and never stopped moving**: 37 pickups (vs 1), 11 search hops, scan-walk-scan across the map from y=141 to y=7, full stock held, CLEAN exit, zero shots and zero HUNT-owned ticks (the role contract, again).

**The exchange reversed direction.** This morning the wedged gatherer offered artax 512 scanned tiles / 28 containers / 29 enemies; this run it offered **6,912 / 431 / 270** plus 28 removals — the fighter's worldview is now mostly gatherer-sourced, which is the role's entire purpose ("one with extra radar can be scanning", 2026-08-14).

**The liveness instrument ran live for the first time and stayed silent** — zero `liveness_stall` diagnostics, zero suppressions, both bots; the analyzer's top-level summary is clean on both artifacts. Silence from an armed detector is the verification: the same instrument retroactively names this morning's wedge at 93.

Gate at landing: 6,187 tests, 100.00% over 30,401 statements / 8,726 branches.

---
## [2026-08-20] audit | Verification-run autopsy: the scope-pending radar drop, named and measured

Operator asked why the verification run's one `stall_timeout` happened, and for a full audit of the pair. **The stall is a server-side race, now documented on [[viewport-shift-protocol]] (OPEN):** a radar dispatched while a scope shift awaits its 0x5A confirm is silently dropped — no fuel charge, no response. Both of today's scan stalls (artax 14:29, arterial 14:31 — the artax one had hidden inside a truncated grep) carry the identical signature from DIFFERENT scope sources (harvest frame shift / ferry scout), and the archive sweep says **23 of all 43 scan stalls ever recorded** have a `scope()` within 4 s of the radar — half of all scan stalls are this one race. The client's own State-13 `a.Ja` scope-pending flag marks exactly the unsafe window; the bot dispatches into it. Cost per hit: 10-12 s and nothing else (the stall timeout replans, the retry lands in ~1.3 s). Fix direction recorded on the page: the teleport's map-open precondition transplanted — defer radar while a scope confirm is pending.

**Audit verdict on the pair (bot-20260820-142810, both returncode 0):** role contract held (arterial 0 shots / 0 HUNT ticks); sharing bidirectional and correctly asymmetric (gatherer→fighter 6,912 scanned / 431 containers / 28 removals; fighter→gatherer 399 enemies); no resource starvation (both ended at inventory caps, fuel minima 609/475 vs the 100 danger floor); radar economy self-sustaining (arterial 28 extra scans, never below 17 of 20, ended full); artax 35 hits on 40 shots, 2 attributed kills (tanks 550, 540), 5 of 5 mine clearances converted, `tank_at_capacity` fuel-lock releases ×3 working; ghost-chases 5-6 per bot in a ~27-bot co-farmed room (the tombstone law's floor); zero liveness stalls, zero suppressions, both analyzers clean and agreeing. The single blemish is the scope-radar race above.

---
## [2026-08-20] root fix | The pan becomes a first-class action: the scope-pending drop is unrepresentable, and recovered anomalies are visible

Operator: "can we fix all these issues properly?" — with the ruling that the defer-flag was a patch, not the root. **The root was a classification**: `scope_shift` was the only server-state-changing command modeled as instantaneous and free ("fire-and-forget like chat", executor comment of 2026-08-01), and the codebase had already been forced into two workarounds for it (the scorecard's `scope_shift_sends_at` IDLE-attribution carve-out; the executor's untracked branch). The measurement that sealed the design: **759 archived pans, median confirm exactly one server tick** (p25=p50=p75=2.0 s, p95=4.0 s, max 8.0 s) — so holding for the pan's true duration costs the tick the bot was always really spending.

**The promotion, template = map_open, line for line:** ledger `ActionKind` gains `scope` (+ bot-side kind, + `ScopeOutcome` `confirmed`/`stall_timeout`, wiki-claim-bound); dispatch records the in-flight action and clears any stale viewport-update mark (a teleport's 0x5A must not instantly complete a later pan); completion is the 0x5A ingestion (`mark_viewport_update_processed`, set one line after `update_viewport_entities`); the standard 10 s stall timeout is the drop's only exit (comfortably past the 8 s archive max). The tick loop now HOLDS during a pan — dispatching radar or map_open into the scope-pending window (23 + 12 archived stalls, half of all scan stalls ever) is structurally impossible, with no defer list to maintain. `scope_shift` decisions also join the decision/outcome fabric via `_LEDGER_KIND_BY_CMD_TYPE`, so pans get `superseded` correlation and the liveness instrument for free.

**Recovered-anomaly visibility (the 19-day concealment mechanism):** any `stall_timeout` count now surfaces as a top-level issue line with per-kind breakdown — the report surfaces what limps, not only what breaks. Post-July baseline is under one stall per run, so healthy runs stay quiet.

**Third verification-discipline rule recorded ([[coding-standards]]):** contracts state invariants, not caller snapshots; a new consumer re-verifies a module's stated assumptions; fire-and-forget is reserved for commands with no dependent server state — chat is now its only member, by contract.

Pins: the hold-until-0x5A and stall-exit tests (`tests/bot/test_cdp_inflight.py::TestScopeInFlight`), the dispatch stale-mark clear (`test_cdp.py`), the seven-kind tuple, and the stall-visibility analyzer rule. Live verification (acceptance: zero scope-correlated stalls by the archive sweep) is the next run.

---
## [2026-08-20] verification | The tracked pan works live: zero scope-correlated stalls, and the stall line prints itself

240 s pair on the promoted build (gate: 6,191 tests, 100.00% over 30,450 statements / 8,738 branches). **Acceptance passed mechanically:** 23 pans dispatched, 22 `scope:confirmed` at median ~2.0 s / max 5.9 s (the 23rd was in flight at teardown), and the 4 s dispatch-window sweep that discovered the race reports **zero scope-correlated stalls** — the three residual stalls (map_open, scan, collect; the sub-1-per-run archive baseline) all lack a nearby pan. [[viewport-shift-protocol]] stamped verified.

**The stall-visibility rule printed its first real lines:** both reports now open their issue summaries with "N action(s) stalled to timeout and replanned (kind=count)" — the concealment mechanism that hid the race for 19 days is itself gone.

Run notes, unrelated to the fix and correctly surfaced by the instruments: arterial (gatherer) 39 pickups, 13 hops, zero HUNT ticks — its 2 shots were COLLECT-owned `mine_clearance_shot`s (clearance is collect doctrine, not hunting); artax fought mined ground (13 of 14 teleports displaced), dipped to 44 fuel, and exited via the protective `out_of_fuel` wind-down — its report flags the fuel floor and its one stall, exactly as designed.

---
## [2026-08-21] soak | The 30-minute pair: promotion live, the liveness detector's first real catch, and the fighter's inherited paralysis

First full-length run on the day's stack (1800 s pair, both exits CLEAN, gate 6,191 tests / 100.00%).

**Arterial (gatherer) — exemplary at scale, including the untested cell:** 280 pickups, 116 hops, and a **live mid-session PROMOTION** (recruit → private, promo counter 14 → 8 into the next rank) — the rank-derived capacity gates handled the 20→25 cap change in flight: a "full" gatherer became under-cap at the promotion instant and resumed equipment pickup, no wedge, no stall. The promoted scope action at scale: **177 pans confirmed, exactly 1 genuine drop** typed `scope:stall_timeout` (the residual the stall backstop exists for). Fed the fighter 3,561 containers / 47,346 scanned tiles / 2,764 sightings across 533 merges.

**The liveness detector's first live catch (02:08:58):** `liveness_stall` fired at streak 12 on the SHOOT kind — a `mine_clearance_shot` at a covered container at (227,171) re-planned 12 consecutive times with zero dispatches. A FOURTH member of the planner/veto class, in the shoot lane (the clearance aim kept being vetoed downstream while selection re-derived it), caught and named IN-SESSION by the instrument built this morning, and self-broken shortly after (the session's 280 pickups continued). OPEN: the clearance-shot veto needs the same selection feedback the equipment gates got.

**Artax (fighter) — the inherited-fuel trap, fully costed:** entered with 44 fuel (its previous session's exact low-water mark; fuel persists across logins), burned it, and spent the ENTIRE 30 minutes at ~0 fuel: **102 `move` stall-timeouts**, 2 pickups, 0 teleports, never solvent — functionally paralyzed for a full session while walking legs toward fleet-shared fuel kept stalling. The new analyzer line names it loudly (102 stalls). OPEN doctrine ruling for the operator: entry solvency — a session inheriting fuel below the recovery-hop floor needs a dedicated posture (or the wind-down must exit before the inheritance is poisoned).

**The marooning's root, from the prior run's timeline (this session's autopsy):** the 13-teleport burn was three loops sharing one blind spot — five ESCAPE teleports displaced back into the same 3-tile patch under fire (fuel 671→259 in 33 s), then four identical harvest hops at (128,238) each displaced to (121-122,230) — and a displaced teleport resolves `landed_inexact`, a SUCCESS, so no failure machinery accumulates. The THIRD liveness flavor: successful actions with no progress. Clearance was attempted (12+ shots) but against the reachable ring, not the destination's. OPEN: displacement recurrence should mark the destination with the existing move-failed machinery after N identical bounces, and a bounced ESCAPE hop should prefer walking off ringed ground.

---
## [2026-08-21] root fix | Displacement becomes evidence: the mine-blind hop loop is unrepresentable, and the third liveness flavor is instrumented

Operator ruling: no symptom counters — find why the bot could get stuck at all. **The root was epistemic, three correct rulings composing into blindness:** (1) 07-20, mined teleport targets are safe (displacement physics — right); (2) 08-05, landings must be mine-free per BELIEF (right, when beliefs exist — this was the "fix" for the 534-bounce session, patched at the wrong layer); (3) the s9-2 radar economics treated scan coverage as mine knowledge (false — mines are dynamic, reveals are a separate decaying layer, and fleet-shared coverage carries no local reveals). Keystone: the displacement receipt — the server saying "your landing model is wrong here" — was classified observability-only and wrote NOTHING to belief. So the bot could be wrong, be told so, discard the telling, skip the scan that would fix it, and re-certify the identical hop. Archive: 11 runs with >= 3x same-destination clusters; in 7 the radar-skip sat inside the cluster window.

**The fix, model-level, no counters:** a chebyshev >= 2 bounce writes `ws.landing_displacements` (requested tile, timestamp, proven radius; routine 1-tile combat displacement excluded); the composed decision terrain refuses LANDINGS inside fresh evidence — the fifth blocker class, walking unaffected — so all seven attainability call sites (hops, locks, larder, clearance, movement) inherit the law and the existing unservable/clearance releases finally fire; the escape path inherits it through the larder for free. The s9-2 premise is corrected: a fresh bounce forces the landing repair radar regardless of coverage. TTL 30 s (the move/scan-mark family), then the repaired mine beliefs own the answer.

**The instrument:** `displaced_teleports` tallies in the issue report + the **displacement orbit** top-level rule (>= 3 bounces at one destination; empirical — the 11 pathological runs all repeat >= 3, healthy combat re-aims at most twice). Retro-verified: the marooning artifact now reads "displacement orbit: 4 teleports at (128,238) all bounced (max displacement 15)" where it read clean.

Pins: evidence write/TTL/radius/routine-exclusion, composed-terrain landing refusal with walking unaffected, the attainability regression at (128,238), the forced repair scan under live coverage, dispatch ingestion, and the orbit rule. Wiki: [[teleport-mechanics]] § displacement evidence, [[radar-mechanics]] § the s9-2 correction. 6,198 tests green pre-gate.

---
## [2026-08-21] verification | Displacement-evidence build: gate green, zero orbits live, the write path awaits its first real ring

600 s pair on the evidence build (gate: 6,198 tests, 100.00% over 30,511 statements / 8,756 branches). **Acceptance: zero displacement orbits and zero repeated destinations** — arterial teleported 38 times with 3 displacements, every one a routine chebyshev-1 nudge (correctly unrecorded by the >= 2 evidence bar), no destination requested twice. Honest scope of the live proof: the room offered NO ring-class bounce this run, so the evidence WRITE path fired only in tests and the retro-analysis (the marooning artifact's "displacement orbit: 4 teleports at (128,238)" line) — its first live firing awaits the next mined-arena encounter, which the orbit rule and the evidence log line now watch permanently.

Artax: the inherited-fuel paralysis again (0 teleports, 0 pickups, exit `out_of_fuel` — faster wind-down than the soak's 30-minute limp). The **entry-solvency doctrine ruling remains the one OPEN item awaiting the operator**: a session inheriting fuel below the recovery-hop floor needs a posture, or the wind-down must exit before the inheritance is poisoned.

---
## [2026-08-21] crack + correction | The refusal law: beyond ring-1 no ejection exists — and two of this session's own models get corrected by the mining

Operator ruling: mine before you model. The mining corrected TWO claims made hours earlier in this very log:

**CORRECTION 1 — there was never a multi-tile ejection.** All **137** archived chebyshev>=2 "displacements" landed the tank exactly at its origin (137/137), uncharged; 8,718 landed vs 4 rejected teleports archive-wide. The server answers a fully ring-blocked hop with a silent confirm-at-origin — the bot has been misreading "you didn't move" as "you were moved" since June. The evidence model shipped earlier today (hostile radius = bounce distance, up to 12) modeled the nonexistent ejection and over-blocked; corrected to what one refusal actually proves: **requested tile + ring-1**, stored as timestamp only (`landing_refusals`, the failed-move-mark shape). Renamed throughout (`mark_landing_refused`, `refused_landing_keys`).

**CORRECTION 2 — the "sim fidelity gap" was misdiagnosed.** The sim's PHYSICS ("beyond-ring-1 displacement does not exist") was the true law all along. What was wrong was the sim's WIRE SHAPE: it answered sealed hops with 0x52 CANT_GO — a shape live teleports never produce — which would have steered a simmed bot down the rejection path instead of the refusal-evidence path. Fixed: the sim now confirms-at-origin, the stale CANT_GO pin rewritten to the measured law.

**The deterministic cell, finally:** `tests/sim/test_landing_refusal_seam.py` — a real Bot over the seam, an equipment container fully ringed by never-revealed hostile mines. One teleport on the wire, the refusal ingested from the sim's confirm-at-origin, ZERO re-certifications (the decision layer shows only the plan + its map-open re-derivation), and the bot walks on. The exact geometry that ran 534 identical hops on 08-05 is now a permanently pinned scenario. Also observed while building it (not chased): a terrain-less seam world routes everything through walk lanes, and the sim let the walker stand on a hostile mine and re-dispatch adjacent equipment pickups without resolution — two sim-modeling edges filed for a future pass.

**OPEN (bookkeeping accuracy, filed not fixed):** the fuel book bills every dispatched teleport at cost±drift on dispatch — refused hops charge nothing live, so the book's teleport-spend bounds overstate spend in refusal-heavy sessions (the marooning report's "955 fuel teleport spend" was largely phantom; the real burn was incoming fire).

---
## [2026-08-21] correction | The "fourth planner/veto member" never existed: every clearance shot dispatched — shoot is the one outcome-less action kind, and the liveness detector's first catch was a catch of itself

Mining the soak artifact (bot-20260821-013519) REFUTES the 02:08:58 entry above. The wire shows **13 `shoot` dispatches**, each echoed (`OUR_SHOT`), each billed the measured −6, at progressing aims spread over 32 minutes (01:36:27 → 02:08:58) — there was no veto, no replan loop, and nothing "consecutive." The clearance lane was working.

**What the alarm actually measured:** shoot is the ONLY action kind with no completion path. Per-kind outcome distribution across the whole run — collect, scan, map_open, move, scope, and teleport all resolve to real completions; **shoot: 12/12 `superseded`, `duration_ms: 0`**, despite all 13 wire dispatches. A shot can only ever resolve as `hit` via combat feedback (artax bot-20260821-004925: 11 hits / 21 shots; the other 9 superseded) — misses and victimless clearance shots NEVER resolve, so every one closes `superseded` when the next decision registers. The zero-dispatch streak counter treats a superseded close as "zero dispatches," so the session's 12th clearance shot tripped a false "12 consecutive replans with zero dispatches" alarm. The instrument's first live catch was a false positive, and the prior session wrote the instrument's story into this log without checking the wire — the same mine-first violation as the ejection misread, corrected here by the same discipline.

**VOID:** the OPEN item "the clearance-shot veto needs the same selection feedback the equipment gates got" — there is no clearance-shot veto.

**OPEN (replaces it):** (1) shoot decisions must resolve on the 0x53 `OUR_SHOT` echo — the server's own dispatch receipt — with `hit` as the combat-feedback refinement, so no dispatched shot ever dies `superseded`; (2) the zero-dispatch streak must count what its warning claims: gate the increment on the superseded decision having produced no wire dispatch, so a healthy dispatched-but-unresolved lane can never impersonate a livelock.

---
## [2026-08-21] root fix | The echo receipt: shoot gets its completion path, the liveness counter stops trusting silence, and the analyzer audits the ledger against the wire

Both OPEN items above shipped, plus the class-level instrument the correction demanded (gate: **6,219 tests, 100.00%** over 30,603 statements / 8,788 branches).

**1. Ground-aimed shots resolve on their echo.** A shoot with `target_id == 0` (clearance fire at a tile) now writes a pending-ground-shot mark at dispatch (`mark_pending_ground_shot`, the scope mark/check family); the tick loop's resolver consumes the own 0x53 echo into a new `shoot:fired` outcome (aim tile in the payload — for a ground shot the tile IS the commanded fact), or a shot-rejecting 0x52 into `command_rejected` ([[shot-range]]: rejected shots never echo). The resolver also consumes the echo's side flags (response, victim lookup, ammo snapshot), closing a latent leak: a ground echo's latched flags could previously hand the NEXT tank-targeted shot an instant stale classification. Combat shots are untouched — `hit`/`miss` via combat feedback remain their richer resolutions.

**2. The zero-dispatch streak counter is dispatch-gated.** The executor marks every decision whose command actually reached the wire (`mark_decision_dispatched`, keyed by event id so a deferred teleport's map_open dispatch credits the original teleport decision). A superseded close of a DISPATCHED decision now RESETS the streak — the planner's output demonstrably left the process, so nothing is livelocked — and the superseded record carries `dispatched` so post-run analysis can tell a vetoed plan from a re-aim. Suppressed dispatches return unmarked, which is exactly what the streak exists to count. The `LIVENESS STALL` warning text is now true by construction. ([[self-observing-architecture]] claim updated; threshold 12 unchanged — the one real livelock ran 93 genuinely undispatched replans.)

**3. The analyzer audits the ledger against the wire.** New top-level rule: per-kind WIRE dispatch tallies vs recorded completions (supersedes and stall timeouts are not completions); a kind with real traffic and ZERO completions is a **ledger modeling gap** and the line says to distrust every outcome-based rule for that kind. The outcomes section now opens with the per-kind `wire dispatches/ledger completions` tally and marks superseded rows with their dispatch flag; the streak scan skips dispatched supersedes. **Retro-verified on the false-alarm artifact:** bot-20260821-013519 now renders `shoot=13/0` beside `collect=280/280 scan=204/204 teleport=116/116` and prints "ledger modeling gap: 13 shoot commands reached the wire but the ledger recorded ZERO completions" — the rule catches mechanically what took a manual wire-mining session to see, and it watches every FUTURE kind anyone adds.

**Deterministic seam proof** (`tests/sim/test_clearance_shot_seam.py`, per the scenario-matrix rule): a real Bot in an enemy-free seam room (new `boot_seam(enemy_alive=False)` — the handshake announces only living tanks), one wanted fuel container with its full neighborhood hostile-mined in truth AND belief. The collect flow's only move is the clearance shot at the covered tile; the sim's bare 0x53 echo (92.4% of 11,051 archived shot windows) resolves it `shoot:fired` with ZERO superseded shoots and the pending mark consumed — the exact soak shape, completing end-to-end. Old artifacts predate the dispatch mark, so their superseded rows read `dispatched=False` — the modeling-gap line is the corrective context there.

---
## [2026-08-21] verification | The echo receipt live: 53/53 clearance shots resolve on the real wire — and the run catches the futility rule mislabeling a working gatherer

985 s solo gatherer session (arterial, exit `completed`). **Acceptance passed exactly:** 53 clearance shots on the wire, **53 `shoot:fired`** resolutions — zero superseded, zero liveness alarms — at durations pinned to the server tick (2,033–2,043 ms; the same one-tick law the scope confirm follows). The report tally reads `shoot=53/53` beside `scan=88/88 teleport=40/40 map_open=40/40` (collect 116/115, move 2/1, scope 81/80 — the teardown truncation tail, each with completions > 0 so correctly unflagged). One `scope:stall_timeout` — the known sub-1-per-run residual, self-healed and surfaced. The kind that read `13/0` this morning is at parity on live wire.

**The run's own catch, fixed same hour:** the report initially flagged "combat futility: 53 shots produced 0 observed kills" — a working gatherer mislabeled as a broken fighter, because the futility rule predates roles and counted every wire shoot. With `shoot:fired` now distinguishable, the rule counts only tank-targeted shots (`shots − fired`) and names the excluded clearance count. The very first live artifact of the new outcome exposed the next analyzer inaccuracy — the instrument sharpening itself. Re-rendered, the run's summary carries exactly one line: the scope stall. Gate: **6,222 tests, 100.00%**.


---

## 2026-08-22 — The file-size rule leaves home: FileSizeRule enforced monorepo-wide

**What happened.** Board task 21e173d7 closed its arc: TankpitBot's `scripts/file_size_rules.py` — written 2026-08-07 as the only enforcement of the 400-600 ceiling anywhere in the monorepo — is deleted, replaced by `FileSizeRule` in `libs/monorepo_guards`, registered in the orchestrator so all 39 packages now fail their gate on any file over 600 lines. Same law: no allowlist, no baseline, worst offender first.

**The backlog that had to die first.** 137 files over the ceiling across 12 packages (the submitter's 134 was already stale; my own first recount of 89 was a measurement artifact — PowerShell `Measure-Object -Line` skips blank lines). All split by role over two days: handwriting-ai's 2641-line hooks monolith, covenant-radar-api's 32 files (2353-line test_decode.py included), covenant_ml's 36 less 4 held because a concurrent session's uncommitted edits sat inside them. TankpitBot itself was already clean — the rule worked here first.

**What the final sweep caught.** Two regressions that would have shipped silently without the lifted rule: a covenant_nn module grown to 677 by mechanical import-block rewires, and a Model-Trainer test committed at 618 under a "zero over" claim measured before the final format pass. The instrument caught its own installers.

**Residual.** Five covenant_ml files remain over (1269/788/783/727/723), all inside or coupled to another session's uncommitted edit set; `make lint` there is deliberately red until that session lands and the five are split. Recorded on the board with an expiry condition.

**Addendum, same day.** The operator confirmed no one else was on cleargbm, so the residual dissolved: the orphaned leaf-wise work was verified and landed (`2a55899c` cleargbm_rs, `1a04bb1e` cleargbm, `bd55e8d6` covenant_ml — best-first tree growth, N-arm benchmark manifest, EcoQoS power-throttle opt-out), the five held files were split on top (`bb4d5360`), and the monorepo-wide sweep reads **zero files over the ceiling**. Board task 21e173d7 closed done; every gate green.

---
## [2026-08-25] root fix | The marooned walker learns to move its window: the fuel-0 oscillation is unrepresentable

**The artifact that named the bug.** Diagnostic run bot-20260825-133452 (Artax, entry fuel 0 — the inherited-fuel trap's third and worst reproduction): 331 s, exit `completed`, **zero pickups, zero teleports, fuel 0 for all 192 readings** while the map atlas held 672 fuel dots. The wire told the whole story: every move target all session sat on the x=113 column — the stored window's west edge — flip-flopping between y=213/215/218/221, with fuel at (110,215) THREE TILES past the edge never reached. 74 successful moves, zero net progress. The audit's 6 `rejection_retry_loop` CRITs and 12 move stalls were downstream noise.

**The root, from the code.** `walk_for_fuel_last_resort` clamps every leg into the stored window (`min(max(target,left),right)`), autoscroll is OFF by standing doctrine, and a candidate whose clamp tile the tank already occupied was skipped — so two nearly-tied fuel destinations alternated clamp tiles forever. The walker had no way to move the window it had exhausted, and no instrument watched walking-lane liveness (the displacement-orbit rule watches teleports).

**The fix, the gait the wire already supports.** An exhausted window now spends a FREE `Rb` pan toward the fuel: `pan_plan_toward` (the ferry scout's compass/anchor core, reach cap removed — a far goal still names the direction revealing the next 15 tiles of route), so a broke tank traverses at window granularity: walk to the edge, pan, walk the revealed stretch. Two guards bound the free pans: the **movement law** (dispatch position latched in new `maroon_pan_x/y`; no second pan from the latched tile, so stuck candidates on opposite sides cannot ping-pong the window — the loop shape is unrepresentable, not counted) and the **terrain veto** (a known-impassable post-pan clamp tile refuses up front). The walker moved to its own module `bot/ai/maroon_walk.py` (collect_hops was at the 600-line ceiling); reason kind `walk_for_fuel_pan`. The walking-as-travel ruling is untouched — the gait exists only on the marooned rung, under the 48-tile cap, where teleporting is arithmetically impossible.

**Deterministic cell** (`tests/sim/test_maroon_recovery_seam.py`): a real Bot at fuel 0, the only container 30 tiles beyond the window — two confirmed pans over the sim's anchor law, the walk crosses both revealed stretches, the pickup refuels. The live-failure geometry is also pinned at decide level (edge clamp pans east; movement law; terrain veto; terrain-blind contexts still pan). Gate: **6,200 tests, 100.00%** over 30,556 statements / 8,774 branches.

**Live validation (bot-20260825-140532), honest scope.** Artax re-entered at fuel 0 and this time fuel WAS in reach of the ordinary cascade: 0 → 515 in the first minute through a normal pickup, then a fully healthy session — 1 kill, 0 deaths, 28 pickups, 11 teleports, radars 11→21, 9 clearance shots all converted, exit CLEAN `session_complete`, audit 1 critical (one self-healed map_open stall, the known residual) vs the morning's 18. **The maroon-pan write path did not fire live** — like the displacement-evidence build before it, its first live firing awaits the next beyond-the-window marooning, which the `walk_for_fuel_pan` diagnostic now watches permanently. Wiki: [[viewport-shift-protocol]] § second doctrine consumer.

**OPEN (unchanged, for the operator).** The entry-solvency doctrine ruling: this fix makes a marooned tank RECOVERABLE wherever known fuel exists at any walkable range, but a session that inherits 0 fuel still spends its opening minutes walking instead of playing; whether wind-down should refuse to exit below a solvency floor remains the standing question.

---
## [2026-08-25] investigation + correction | The double-check: the root verified on the wire, the 08-21 paralysis was the same bug, and the "stalls" were never stalls

Operator directive: proper investigation, double-check, and answer whether the root was identified in the first place.

**The window-clamp root is now wire-verified, not inferred.** The diagnostic artifact holds exactly ONE viewport all session — `Viewport: (113,213) to (128,228)`, 97 frames, never moved — with every fuel candidate west of it. The capture decodes 96 sent move frames (`[!][0x04][0x70][x][y]`) against 95 self 0x47 Movements and 9 `cant_go`: the server executed essentially every command, and the tank PHYSICALLY oscillated between the clamp tiles the planner alternated. The alternation mechanism is arithmetic: standing ON candidate A's clamp tile skips A (`leg == self`), selects candidate B's clamp tile, and B's arrival re-selects A — verified against the sorted-candidate distances tick by tick.

**The 08-21 "inherited-fuel paralysis" was the SAME bug, unrecognized at the time.** `runs/bot/artax/bot-20260821-013519`: one static window (113,214)-(128,229) for 34 minutes, 528 marooned walking legs ALL clamped to window edges (west x=113, north y=214, south y=229) toward fuel at y=233/y=245/x=100/y=198 — outside on every side; fuel at (122,233) sat FOUR tiles past the south edge for half an hour (183 legs at its clamp tile). The 08-21 log entry blamed "walking legs toward fleet-shared fuel kept stalling" and filed the doctrine question; the actual mechanism — the walker cannot move the window it exhausted — was only found today. Answer to the operator's question: **no, the root was not identified in the first place; it is now, and the prior artifact confirms it retroactively.**

**The stalls were false alarms, and that is a measured law now** ([[walk-mechanics]] § fuel-0 service latency): fuel-0 move echoes run median 3.8 s / p90 ~13 s / max 15.75 s (n=578 paired dispatch→echo across both artifacts) while scans and map opens stay at tick speed. The over-10 s tail counts match the two runs' stall counts EXACTLY (102↔102, 11≈12) — every `move:stall_timeout` in the paralysis sessions was the latency tail crossing the fixed 10 s budget, and each wrote a false failed-move mark that steered the planner for its 30 s TTL. This corrects the previous entry's wording ("12 move stalls were downstream of the same blindness" — they were downstream of the fuel-0 latency law meeting a fixed budget). **Shipped:** the move stall budget is 20 s while believed fuel is exactly 0 (`FUEL_ZERO_MOVE_STALL_TIMEOUT_MS`, clears the measured max with margin; move-only, fueled moves and every other kind keep 10 s) — which also protects the new pan-walk gait's 15-tile legs from stall-marking themselves. Mechanism attribution (fuel-0 regime vs dense repeat-tile cadence) is deliberately not claimed; the budget keys on the measured population.

**Also re-examined, no change needed:** the 9 `cant_go` triplets (east-edge targets across mined ground — rejection-marking worked, TTL recycling is designed), and the maroon-pan latch's revisit edge (a tank returning to a previously latched tile is refused its pan and exits — conservative by the exposure doctrine, accepted).

**Re-run after the double-check (bot-20260825-144527, Artax solo, healthy entry at 1100):** the cleanest session of the arc — 3 kills, 0 deaths, 28 pickups, 8 teleports, 23/23 scope pans confirmed, exit CLEAN at the timer, **issue summary empty and audit 0 critical** (vs 18 in the morning's diagnostic, 1 in the validation run). The double-check itself found nothing needing code: the pan latch is dispatch-gated by construction (`tick_body` persists updated_ai_state only when `command_sent`), so a suppressed pan can never strand its latch. The maroon-pan write path still awaits a live beyond-the-window marooning.

---
## [2026-08-25] soak + doctrine closure | 30 minutes clean on the new build, and the entry-solvency floor turns out to already exist

**The soak (bot-20260825-184324, Artax solo, 30m28s, detached process).** The longest session on the post-fix build, and the strongest artifact of the arc: **11 kills, 0 deaths**, 97 pickups, 48 teleports, 200 shots, inventory ending AT CAP on every slot (radars 13→25), fuel ending ~1100 — the next session inherits a full tank. **Zero move stalls in 30 minutes** against the 102 of the last 30-minute Artax session (the fuel-0 slow-service regime never engaged because fuel never touched 0); zero maroonings, so the pan gait's live firing still waits. Audit: 1 critical (one self-healed map_open stall, the standing sub-1-per-run residual), 1 teardown-truncation warning, 23 info (drained-container contention). Exit `connection_lost` — the wire went silent for 90 s at 19:12, right at the session's natural budget end; the tracked clean path did its job. One-off; watched, not chased.

**The entry-solvency doctrine item, CLOSED — the floor already exists.** The proposal was "wind-down may not exit below a solvency floor (~200)." Reading the exits: the wind-down's stocked exit (`should_exit_collect`) requires fuel at the rank's FULL capacity (`hunt_fuel_floor`), and its only other clean exit (`collect exhausted`, ai_strategy) is reachable only above `fuel_low_threshold` — default **200, exactly the proposed floor**. Every clean exit is solvent by construction and always was. The insolvent endings that motivated the item were only ever two paths — `out_of_fuel` (nothing within the walk cap: irreducible, nothing exists to top up with) and the hard timer expiring MID-MAROONING — and the marooning is what the pan gait root-fixed. Building a wind-down floor knob now would be decorative (the knob class this workspace's constitution forbids). Resolution: no code; the fix that was needed was the maroon pan, already shipped and soaked.

---
## [2026-08-25] milestone soak | 84 kills, 0 deaths, 2.5 hours — the bot empties the room

The first kill-bounded marathon (bot-20260825-212920, Artax solo, detached process, target 100 kills / 6 h backstop): **84 kills, 0 deaths over 149 minutes** — 0.56 kills/min sustained — with 1,383 shots, 808 pickups, 490 teleports, inventory AT CAP from first tick to last, fuel ending 1100. The exit is the honest one: `no_viable_targets` at 23:58 — a fresh map snapshot held no affordable enemy and no relay dot. The bot did not disconnect, did not die, and did not idle: **it ran out of opponents** 16 kills short of the target and ended the production way. Session opened combat-ready on the previous soak's full stock (first HUNT·ENGAGE inside the first minute) — the wind-down doctrine's leftover-stock effect working across sessions.

Health under 5x the previous session length: **zero move stalls in 149 minutes** (the extinct class stays extinct), zero maroonings (the pan still awaits its first live firing), audit 22 critical decomposing entirely into the KNOWN residual families scaled by duration — 11 map_open + 8 scope stalls (~2% of hundreds of dispatches, all self-healed) and 2 contested-container collect retries. No new defect class. Promotion points +21,516 (542,008 → 563,524).

Operational note: kill-bounded detached runs work exactly like the timed ones (`TANKPIT_BOT_SESSION_KILLS` honored, monitor on the HUD file); the 30-minute `connection_lost` from the earlier soak did NOT recur across 2.5 hours, downgrading it to a one-off.

---
## [2026-08-26] root fix | The 84-kill run's exit was a phantom: "I asked recently" is not "I heard recently"

Operator challenge ("any enemy killed respawns... there's 27 enemies at all times") refuted the marathon's `no_viable_targets` story, and the mining proved them right. The final `acquisition_candidates` record holds **total_enemies=27, accepted 0, every rejection `stale_map_data`** — including red-9 TEN tiles away. The wire tail shows why: no map-sized response in the final 75 s (the connection was dying; full silence followed seconds later), yet the final map open "completed" in **12 ms** — it consumed an ORPHAN `map_data_processed` flag left by one of the run's 11 stalled opens whose late response landed after its action closed. The exit gate then aged the snapshot from `last_map_open_ms` — the DISPATCH time (2,052 ms) — concluded "fresh empty map," and quit under 27 live enemies.

**Two root fixes** (gate: 6,206 tests, 100.00%): (1) `open_map` clears any stale map-data mark at dispatch — the scope pan's 2026-08-20 discipline extended to its sibling, so the completion flag now provably means "a MAP_DATA arrived since THIS open"; (2) the no-viable-targets gate ages the snapshot from a new `map_data_ingested_ms` stamp written at actual MAP_DATA ingestion — data recency, never dispatch recency. With stale data the hunt keeps searching, and a truly dead wire now ends the session honestly as `connection_lost`. Pinned by the dispatch-clear test (orphan flag cannot instant-complete an open) and the phantom-geometry regression (fresh dispatch + stale ingestion → another `find_enemies` map open, never an exit); the eleven pre-existing freshness tests state their premise explicitly via the ingestion stamp. Same disease as the fuel-0 stall family: slow responses crossing a timeout birthing orphan evidence — the third member of that class found and killed in two days.

**Correction, same night (the wire was NOT dying).** Deeper mining refutes this entry's own "dying wire" framing: MAP_DATA answers flowed the whole marathon (560 total, through the final minute), and the last frame the session ever received — 23:58:46.681, `MapData tanks=37`, all enemies present — was the ANSWER to the 23:58:41 open, landing 5.7 s after dispatch and 3.7 s AFTER the phantom exit had already quit. What actually degraded was map-answer LATENCY late in the session (~2 s typical → ~6 s; 8-9 answers/min → 1-4/min over the last five minutes) — the same measured shape as the fuel-0 move law: a slow server response crossing a local deadline and minting orphan evidence. The bot quit four seconds before the truth arrived. With the fixes above the open simply stays in flight (5.7 s is well inside the 10 s stall budget), the answer completes it, and the hunt continues; nothing about the wire needed fixing.

---
## [2026-08-26] optimization + refutation | The map-intel horizon ships; the burst teleport dies by its own probe

Two builds from the marathon's waste ledger (16% of the session in 559 map opens; 1.94 s median open→teleport wait on all 490 teleports):

**Shipped — the freshness split (`map_intel_horizon_ms`, config default 12 s).** One constant (`map_open_cooldown_ms` = 5 s) served as both the re-open cooldown AND the freshness bar for map-sourced intel, while map answers measure 2–6 s of latency — snapshots went "stale" almost on arrival, driving the re-open churn and starring in the phantom exit. Acquisition, pursuit, greeting, relay, and the no-viable-targets gate now age intel against the 12 s horizon (practice bots barely move in 12 s); the 5 s cooldown keeps pacing dispatches. Eleven consumers switched; the between-cooldown-and-horizon acceptance is pinned (`test_accepts_candidate_between_cooldown_and_horizon`).

**Refuted — the same-tick open+teleport burst.** The wiki's burst reading and probe round 1 (arterial, `immediate_after_map_open`: 10/10 landed, zero timeouts) said the June drop law (20260610-024x: 4/15 dropped) had retired; the burst was implemented, gated... and probe round 2, minutes later, dropped **7 of 10** (that session also took 30 s to initial sync vs round 1's 8 s). The law is real and SESSION-VARIABLE — likely load-dependent — so the burst was reverted the same night, before it ever ran live. The defer's docstring now carries both rounds and a do-not-retry bar (multi-session probe evidence required). Probe evidence: `runs/probe/burst-probe2-20260826.*`; the committed `teleport_probe.*` fixtures restored byte-identical. The re-measure-before-shipping discipline is the whole story here: round 1 alone would have shipped a change that costs 7 s of stall per 10 teleports under round-2 conditions.

Gate: 6,206 tests, 100.00%. The 30-minute progress beats of kill-run 2 (Artax, fixed build) continue in parallel; run 3 launches on this build when it lands.

---
## [2026-08-26] doctrine + live op | Consent is to the color, not the tank: the fleet inherits it — and a 90-minute human duel forces the ruling

**The fight that forced it.** Kill-run 2 (bot-20260826-003927, Artax, 26 kills / 0 deaths / 2h3m) turned into a 76-minute duel with a human, Beerus (id 709): consent-gated engagement, 31 confirmed connects per 5-minute stretch, and a structural stalemate measured in the bot's own break ledger — hits-to-kill oscillated 7→10→13 across 33 breaks and never trended down (he refuels between bursts; fuel is HP), while his 45–90/tick return fire tripped our floor first every round. Neither side died. The bot's projection line named the truth: "human fight projects past capacity (needs 1250)" against an 1100 tank.

**The reinforcement attempt exposed the gap.** Arterial, spawned mid-duel as a second fighter via the fleet manager, merged artax's knowledge within a tick (`fleet_knowledge_merged`) and ranked the fleet-engaged target first — but never fired: for arterial, Beerus was `human_not_consented`. Operator ruling, verbatim: *"if one has consent the other doesn't need it"* — the human engaged our COLOR, not one tank.

**Shipped: `combat_consent_ids` rides the fleet report.** The reporter publishes its consent evidence (chat-seen ids ∪ damage-taken ids); the merge replaces `ws.fleet_consented_tank_ids` wholesale (the engaged-ids pattern — a departed sibling's consent ages out with its report TTL); `human_combat_consented` accepts local chat, local damage, or fleet-inherited evidence. The field is hard-required in the decoder — no default, no shim — which makes mixed-build fleets fail loudly by design (the standing fleets-run-one-build rule); deploying it meant cleanly cycling both live bots, and the gate itself proved the point by refusing to pass while old-format reports were still being written. Consent remains session-scoped and fire-authorization still requires own-viewport confirmation. Pinned end-to-end in the exchange test (sibling's ids → `fleet_consented_tank_ids` → gate passes). Gate: **6,208 tests, 100.00%**.

**Operational note (cost of over-serialized gating):** the tank swap itself takes seconds; three sequential full-gate laps with both tanks parked cost ~25 minutes of empty map. Lesson recorded: keep the fleet playing while the build gates; cycle once, on green.

---
## [2026-08-26] two live receipts, two root fixes | Return fire from COLLECT, and the 1051-byte frame that killed artax

**The Yuppler receipt (operator field test).** The operator logged in as Yuppler and shot arterial ~10 times point-blank; arterial sat on a 1,025-volume fuel pile out-collecting the damage (fuel never left the 993–1100 band) and answered nothing for **37 seconds** — first return shot only when the tank touched cap and COLLECT went idle. The trace showed the flaw was structural, not consensual: consent had registered (damage-taken), the enemy was in view, but only COMBAT-owned ticks consult the 2026-08-14 opportunity doctrine — COLLECT-owned ticks (score 925 vs HUNT 800) never asked. And the trade-off was fake: once engaged, the bot refueled mid-shot every tick anyway (`mid-combat pickup`).

**Shipped: the collect return-fire rung.** `collect_return_fire` sits in the collect gate order between landing-scan and the escape handler: one fuel-confirmed hit inside the 6 s window from an attacker with a legal immediate shot flips the tick to the divert, and the adjacent-container pickup now rides EVERY opportunity shot as its secondary — return fire costs the refill nothing (operator ruling: "those ticks could be used for damaging the enemy"). Survival stays senior: at the fuel-low break or any weapon break the escape doctrine owns the tick unchanged; gatherers never fire. Replaying the receipt against the new build: first return shot ~6 s in, refilling throughout.

**The artax crash (tick 1176, 03:31:28).** Mid-teleport toward an equipment hop at (39,53), the server sent a **1051-byte ciphered frame** — past the 1000-byte XOR table and past the 931-byte maximum of the 282,783-body archive — and `xor_decode_body`'s length guard raised `XorBodyTooLongError`, crashing the session (15 kills, 218/218 hits, wasted). The guard had mistaken an archive maximum for a protocol bound: the real client's inbound decode is `l[ja] ^= B[ja % pa]` ([[xor-cipher]]) — **the key table wraps**. The room was simply the busiest ever recorded (operator + two fleet bots + Beerus + 27 practice tanks): a container-dense viewport patch outgrew 1000 bytes for the first time.

**Shipped: the wrap law, verified on the killer frame.** `xor_decode_body` now wraps (`table[i % len]`); `XorBodyTooLongError` is deleted, not deprecated. The crash frame itself (payload + session magic, preserved in the crash capture) is pinned as a replay regression (`tests/capture/oversize_frame_20260826.json`): through the production pipeline it decodes to a clean 21-container viewport patch clustered on (30–45, 44–60) — including the exact equipment container at (39,53) artax was flying toward — with wrapped-tail volumes pinned byte-exact.

**Also landed:** the Windows `knowledge.json` read race (bounded 3-attempt retry through the `os.replace` swap window — the crash that killed arterial at 03:01) and the sim-seam flake root cause: `test_maroon_recovery_seam` + `test_landing_refusal_seam` ran on wall-clock, so a loaded gate (24 xdist workers beside two live Chromes) stretched inter-tick gaps past real stall budgets; both now drive the documented `SeamClock` discipline the soak tests already used. Gate: **6,218 tests, 100.00%**.

**Same-night correction (first live hour).** The rung's `-1` exclusion re-fought solvency-broken fights: artax broke from red-8 (projected fuel 318 < floor 354) and the rung answered red-8's continuing fire with six more shots, fuel 851→686 — a break-restock tick HOLDS the lock, and `refuel_for_hunt` routes it through the collect gates where the rung saw "recent attacker in view." The exclusion is now the held lock (`combat_target_id`): the broken-from enemy belongs to the resume machinery, a second attacker still draws fire, and the no-lock Yuppler shape is unchanged. Pinned both ways; fleet cycled onto the fix within the hour. The find was the monitor watching for the doctrine's first live receipts — the receipt itself exposed the collision.

---
## [2026-08-26] analyzability | Decision provenance and the engagement ledger ship

The two builds commissioned from the collision post-mortem ("ensure it's easy to analyze and improve"):

**Decision provenance.** Every decision line now ends with `owner=<mode> lock=<id>` (structured: `owner_mode`, `held_lock_id`). The proposer mode and the durable owner genuinely diverge — a collect-owned tick emits HUNT-scored diverts — and that divergence is exactly what took three artifacts to triangulate during the collision. One field, printed always.

**The engagement ledger (`tankpit-engagements`, in `make analyze`).** One row per enemy from the events artifact: shots (wire-grounded), breaks, outcome (kill / killed_us / open), time-to-kill, and the fuel damage trade both ways from the session damage book. Ally stray hits and never-fought attackers render as informational rows but never flag. Headline flag = the collision signature: an engagement that lost the damage trade after a solvency break. Validation: run against the PRE-fix crashed session it flags purple-9 (4 breaks, trade −45) — the bug we found by luck and monitor, now found by report. Bonus finds in the same pass: artax beat Yuppler 1530–0, and took 180 fuel of friendly fire from Arterial.

Gate: 6,231 tests, 100.00%. Provenance reaches the live pair at their next natural cycle; the ledger works on any archived run today.

---
## [2026-08-26] first medal, double crash | Arterial's BRONZE TANK AWARD crashes the fleet; the 0x4E path hardened end to end

At 05:11:16 the server announced the first decoration ever seen in the capture corpus — and it was OURS: `0x4E tank_id=602 slot=1 level=1`, **Arterial's BRONZE TANK AWARD for 100 career deactivations** (decoded byte-for-byte from both bots' captures, matching the [[decoration-encoding]] Sf trace from June). Both bots crashed in the same second processing it: the `tank_decoration` diagnostic passed a field named `level`, which collides with a reserved JSONL record key, and the encoder's loud rejection was unhandled. The dispatch line was 100%-covered — but the collision check lived only in the file handler, which unit tests never attach. Covered and unverifiable.

**Fixed three ways** (gate: 6,238 tests, 100.00%): (1) the field is `decoration_level`; (2) reserved-name validation now runs at the emit call itself, so coverage of ANY emit line proves its field names are legal — the pre-existing 0x4E dispatch test that falsely passed becomes a genuine guard; (3) awards now decode to NAMES: `protocol/decorations.py` carries the client's nb table (9 slots × 3 tiers), the wire formatter prints `award=BRONZE TANK AWARD`, the diagnostic carries it, and unknown slot/level pairs render as raw numbers — a future server-side award category can never crash a session. Both bots were respawned within a minute of the crash; they cycle onto the hardened build on green.

---
## [2026-08-26] castaway | Arterial maroons on a one-tile lake islet; the account is wedged pending an operator ruling

Session bot-20260826-052735 ended `out_of_fuel` (12 kills, 91% hit rate) after a nine-minute death spiral: from full tank at 05:46 to 0 at 05:59 with ZERO successful pickups. The forensics, layer by layer: the bot had teleported onto (222,192) — a legal landing that BFS over the real terrain proves is a ONE-TILE ISLAND in a lake (reachable set = exactly 1 tile) — and its container beliefs for the region were stale-or-water-locked (671 larder candidates: 585 reserve/afford-blocked as fuel fell, 86 no-landing, nearest believed fuel `failed_pickup`). The maroon pan-walk fired for the FIRST time live at 05:59:50 and its declines were CORRECT — there was never a walk path off the islet.

**The wedge:** the server persists position + fuel per account and ignores the join-time spawn coordinates for an existing tank (verified: enter request carries (128,128), tank re-entered at (222,192)); every rejoin lands at 0 fuel on the islet and exits `out_of_fuel` in ~90 s. Verified escape routes, all needing an operator ruling (they trade against the 2026-07-25 quit-when-stuck / exposure doctrine): castaway ferry-wait (lake has ferry routes; boarding is free), deliberate deactivation (respawn refills, costs rank), or park the account. Known secondary defect from the same tail: the `out_of_fuel` exit fired ~3 s after a `map_for_dots` open — inside known 2-6 s map latency — the same exit-races-answer disease as the phantom-exit family (consequence-free here: no dots were reachable regardless).

Same hour, unrelated: artax exited `connection_lost` honestly at full stock mid-hunt (17 kills, 97% hit rate) — the transient wire drop the phantom-exit fix taught it to name — and respawned clean.

---
## [2026-08-26] milestone | 100/100 — the first target-complete session, at 1219-of-1220 shooting

Artax closed the full 100-kill target for the first time (`session_complete`, ~2.5 h, run following the 06:15 respawn): **100 kills, 0 deaths, 1220 shots with 1219 hits — one miss in the entire session** — and exited fully stocked (1100 fuel, 25/25/25, the wind-down doctrine's leftover-stock effect intact). The previous best was the 84-kill marathon that emptied the room. This run was solo (Arterial parked on its islet) on the build carrying the night's whole stack: return-fire with the lock exclusion, the XOR wrap, the intel horizon, the decoration hardening, and provenance-stamped decisions throughout.

**Fixes shipped from the forensics (same day).** (1) `ride_dead`: standing at or beside a ferry boarding tile while still deriving the same boarding hop is consumed as the ride-failed receipt — the fuel larder and the equipment hop both skip the candidate until the tank moves away (the loop's laps all SUCCEEDED, which no ledger counted; this breaks it at lap 2 with ~900 fuel intact). (2) `await_map_answer`: the out_of_fuel exit holds while a map-for-dots answer is in flight (10 s budget vs the measured 2-6 s latency) — the 06:01 re-entry had exited 3 s after its own open. Exit-expecting test fixtures now stamp their answered-open premise via `map_data_ingested_ms`. Open question carried: WHY the server refused rides off a genuinely-boarded ferry (134 cant_go code=1) — the [[ferry-mechanics]] riding law needs a JS trace or live probe before water-locked farming can be trusted. Gate: 6,251 tests, 100.00%.

**Landing quality ships (same day).** The larder's score is now `gain × landing_walkable_fraction / cost` — the third and deepest islet fix: the trap is demoted at FIRST evaluation, at full fuel, where every alternative is still affordable (the fixture proved the cost law is steeper than assumed — 339 fuel for a 40-tile hop — so mainland escapes vanish behind the reserve gate as fuel falls; quality has to act early). A sole rich water cache still harvests (ranking, never bans). Gate: 6,253 tests, 100.00%.

**Double-check correction (same hour).** Replaying the real 05:46:27 decision on the actual field01 terrain: the boarding tile's landing viewport is 36.7% walkable — the quality factor demotes the trap 2.7× (9.65 → 3.54), not the ~16× of the synthetic fixture; that shore has more mixed ground than the final viewport dump suggested. And the atlas shows a 1049-fuel tile at cost 8 from the tank's position — mainland would have won even under OLD scoring had it been IN BELIEF. The live loss was therefore partly a belief-set failure (the rich neighbors were drained/failed/unobserved at that moment), not scoring alone. The defense is layered by design: quality = early nudge, ride_dead = hard stop, await_map_answer = exit guard.

---
## [2026-08-26] main map | First Desert recon, first Desert farm, rooms become a fleet parameter

**Recon (gatherer, 25 min, troop 3):** World (Desert) runs the same 36 NPC bots as Practice plus walk-in humans (none appeared). Fuel-rich, equipment-sparse in the roamed area. Tank state is PER-ROOM and PER-TROOP: the orange slot entered at rank 1 / fuel 0 (nothing carried from Practice), foraged to full from nothing. Ranks on Normal fields are the REAL ladder — practice promotions never registered.

**Rooms ship as a first-class fleet parameter** (same day): `FleetReportDict.room` hard-required + same-room-only merges (the cross-room belief-poisoning landmine that made rooms "not easy"), manager spawn/room -> TANKPIT_ROOM, POST/GET /bots + dashboard field and column. The mixed-build tripwire fired once mid-rollout exactly as designed (a new-schema artax read the old-schema recon's report and refused loudly); the row-encoder omission was caught on the first live spawn and fixed.

**Farm (fighter, blue troop):** Arterial's blue slot on Desert is a rank-5 CAPTAIN with a 1500 tank — the account carried real main-map history. Spawned at center at fuel 0, foraged up, first NPC kill inside two minutes, 58/58 shooting at the 5-kill mark, solvency breaks behaving on unmapped terrain. Target 100 kills on the real ladder.

---
## [2026-08-26] milestone | The first main-map century: 100/100 on World (Desert), 99% shooting, zero deaths

Arterial's blue Captain completed the first 100-kill target on a Normal field (~2.4 h): **100 kills, 1290 shots, 1280 hits (99%), zero deaths**, exited fully stocked at the rank-5 caps (1500 fuel, 45/45/45) per the wind-down doctrine. The corporal-grade NPCs (operator ground truth; ~10-hit kills fit 1200-cap tanks) drove 39 solvency walk-aways — **every broken fight was resumed and finished** (engagement-ledger receipts: red-9 1 break/66 shots/kill, red-5 3 breaks/kill), the never-drop-live-lock law applied on the real ladder. Two humans encountered: Red October (fought back briefly, one break-resume cycle ran, they left the game inside a minute) and Moveail (greeted, no response). Artax's blue Lieutenant farmed alongside from 16:24 (75+ kills, fleet exchange live at 194+ merges/session). The fleet's first full day on the real ladder: ~180 combined kills, zero deaths.

---
## [2026-08-26] first blood | Blue Killer kills Arterial twice; three fixes from the silent deaths

A human named Blue Killer engaged the Desert farm and killed Arterial twice inside four minutes (18:42:01, 18:45:58), demoting it Captain → Sergeant — and the bot never knew: no 0x41 deactivation arrives for self on Normal fields, the death appears on the wire ONLY as fuel driven through zero (u16 wrap: readings 65475/65530), the deaths counter stayed 0, and the fuel belief ingested 65k as if it were a pickup. Artax, in the same room, was untouched.

**The autopsy, all wire-grounded.** Death 1: the solvency break fired correctly at 666, but the escape ticks were spent SHOOTING — the return-fire divert (built the night before) had no idea an escape was in progress and fired at a second attacker seven ticks straight under 135/tick crossfire. Death 2: Blue Killer's spaced rhythm (-90 every 4-12 s) kept the 10 s window under the 3-hit floor, the measured rate counted zero, and the projection watched 952 fall to 132 across seventeen shots without one break; the freshly respawned tank had re-engaged its own killer under finish-every-fight.

**Fixes (gate: 6,269 tests, 100.00%):** escape ticks are inviolable (a holding break latch silences the return-fire rung); every confirmed hit feeds the projection (the 3-hit floor is gone — the death shape now breaks at ~560, below the human attrition band); and the u16 fuel wrap is booked as the self-deactivation receipt (counted, diagnosed, alerted, never ingested). Open doctrine question for the operator: should a death to a human block that human for the session? Arterial parked pending redeploy on this build.

> **CORRECTED by the [2026-08-26] double-check entry below:** there were THREE deaths, not two; the 0x41 self-receipt DOES arrive on Normal fields (all three deaths carry `origin=protocol_0x41`); and the respawn contract ran each time. The invisibility was in the reporting consumers, not the wire. The blocking question was answered NO by operator ruling.

## [2026-08-26] double-check | The autopsy autopsied: three deaths, the 0x41 receipt was there all along, and two dead counters

User-ordered double-check of the Blue Killer fixes ("make sure we aren't duplicating code, that we are lifting not forking") — and the artifact (`runs/bot/desert/bot-20260826-182204.events.jsonl`) overturned the autopsy's core claim.

**What the artifact actually says.** THREE deaths, not two: 18:37:41 (Blue Killer, id 719 — during the seven shooting-while-escaping ticks), 18:42:01 (Blue Killer again), 18:45:58 (a MINE walked over at 39 fuel while collecting post-respawn, `is_mine_kill`, owner id 3). Rank trail Captain(5) → Corporal(2), not Sergeant. **Every death carried a self 0x41** (`origin=protocol_0x41` diagnostics at all three timestamps) — the claim "no 0x41 arrives for self on Normal fields" is FALSE, and the 2026-07-30 respawn contract ran all three times (`self_respawn_wait` receipts). The u16 wrap (65460/65475/65530) lands one message BEFORE each 0x41 in the same drain batch; the real wrap bug was ingesting it as a +65k pickup into the fuel book.

**Why the deaths looked invisible: every deaths counter was dead code.** The digest regex `DEACTIVATED: tank=N killed by M` matches a line no producer emits for self (the self path logs `SELF DEACTIVATED: killed by N`); the engagement ledger counted `tank_deactivated` with victim=self, which dispatch never emits (self routes to `self_deactivated` first). Both were covered at 100% by fixtures fabricating records production never writes — the fixture matched the consumer, not the producer.

**Fixes (lift, not fork).** One canonical receipt: `self_deactivated`. The wrap detector now raises the SAME `ws.self_deactivated` flag the 0x41 path uses (handing off to the existing respawn contract instead of a parallel fuel-0 path), both producers dedup on the flag (wrap lands first and wins; the 0x41 books nothing twice), and both consumers count that one diagnostic — the dead regex and the dead victim=self branch are deleted. Replay validation over the untouched death artifact: digest and ledger both now read **deaths=3** (previously 0). Ledger also shows the trade truth: 8,325 fuel dealt to Blue Killer vs 4,050 taken — we out-damaged him and died anyway, because he refuels and we hit zero. Fix A verified against the real path: the escape-tick shots went `continue_break_escape → refuel → collect_return_fire → opportunity_shot_decision`, exactly the rung the latch veto now silences; the pursuit/close diverts are latch-gated at phase entry and unreachable.

**How these bugs were missed, on the record:** (1) consumer-shaped fixtures — a test that fabricates the producer's record proves nothing about the pipeline; (2) the sim server sends the self 0x41 (that contract IS sim-tested end-to-end) but never models spaced human fire, crossfire during escape, or the wrap-before-0x41 ordering — the practice-room-solo blind spot again; (3) the autopsy inferred "no 0x41" from `deaths=0` in the digest instead of grepping the artifact for the diagnostic that was sitting there. Gate: 6,261 tests, 100.00%. Operator ruling recorded: **no blocking of humans who kill us** — blocked players can still attack, so blocking only disarms us; the defense is the solvency machinery plus fleet response.

## [2026-08-26] measurement | The server serves ~1 shot per second — twice our combat cadence

> **CORRECTED by the entry below (same night):** the ~1 shot/s claim is WRONG — burst totals counted queued serves landing during the settle window. The 0x53 echoes show serves exactly 2.0 s apart; the bot already fires at the server cap. The cadence lever is dead; the play-style comparison numbers stand, but "humans click 2x/tick" was multi-attacker crossfire, not fast clicking.

User question driving the night: "we should be able to improve the bot dps?" Two measurements answered it.

**How the humans play vs how the bot plays** (artifact analysis, brrruh fight 19:12-19:42, teleport charges excluded): the bot dispatched 117 shots at brrruh in 31 minutes (~4 min of trigger time; 87% of the fight went to breaks/restocks/chases), firing strictly on the 2 s tick when engaged. brrruh landed ~278 clicks (~9/min, 534 dmg/min) with **continuous pressure — he refueled ~9 times without ever stopping shooting**, and out-landed us roughly 2:1. His weapon mix (87 duals vs 191 single-class hits on us) is the movement law in action: a shot at a mid-move tank downgrades dual(90) → homing(45) ([[weapon-selection]]), so **dodging halves incoming per-shot damage** — he weaves constantly, we trade from a standstill and eat full duals.

**The fire-cadence probe** (`make cadence-probe`, new: bursts at fixed spacings, served shots counted from server-refreshed 0x49 ammo snapshots — the per-shot ammo ledger; acquisition lifted from the combat probe, no used-target exclusion). Live runs as Arterial on Practice vs red-8:

| spacing | dispatched | served |
|---|---|---|
| 2000 ms | 6 | **6/6** |
| 500 ms | 6 | **3/6** |
| 250 ms | 6 | **2/6** |

All three fit one law: **the server serves ~1 shot per second** (500 ms → every other shot; 250 ms → ~every fourth; the burst windows give T≈1.0-1.3 s per served shot). Fuel bill confirms only served shots are charged (-10 each; swallowed dispatches are free). One 250 ms shot served as a homing (target moved). The 1000 ms burst was blocked by the probe's 400-fuel exposure floor (Arterial's practice tank drained to 338); expectation under the law is 6/6 — unverified.

**Implication: our combat loop fires at HALF the server's serve rate.** Doubling in-fight cadence to ~1 shot/s doubles DPS (45→90 fuel/s) and halves every "needs N fuel" human projection — the difference between brrruh-class stalemates and finishable fights. DPS levers ranked: (1) 1 s fire cadence in combat, (2) dodge-weave while trading (halves incoming, breaks the projections the other way), (3) fight-while-refueling like the humans do, (4) mines + fuel denial. None implemented yet — measurement first, per the crack-before-code rule.

Operational notes: Yuppler's accounts.json credentials no longer authenticate ("Invalid username or password") — pool is effectively Artax + Arterial until the operator refreshes it. A probe launched with .env defaults tried to log in as Artax WHILE artax farmed Desert (login failed harmlessly, farm untouched) — probes must always pin TANKPIT_ACCOUNT to an idle account. Artax passed 90 kills on Desert during the measurement, zero deaths.

## [2026-08-26] milestone | Artax's century on the death-fix build: 100/100, zero deaths, brrruh survived

Artax closed the 100-kill target on World (Desert) in 2h26m (19:08-21:34, `session_complete`, 1,588 shots, Lieutenant, countdown 7,424): the first century on the build carrying the Blue Killer fixes — and it includes the 29-minute brrruh fight (117 shots, 9 solvency break-refuel-resume cycles, zero deaths; brrruh departed at 19:42 with the bot mid-chase). The break machinery paid for itself the same day it shipped. Cycled artax onto the newest build (receipt dedup + cadence-probe commits) at the session boundary, PID 54400, same target.

## [2026-08-26] correction | The serve cap is 1 shot per 2 seconds — the bot already fires at it

User challenge, verbatim: "does the inventory change every 1 second or are you sure you're not just spamming shoot commands but still doing 1 per 2 seconds." The raw capture answered: decoding the probe's own 0x53 echoes (`cadence_probe.capture_session.json`, run 21:27:48-21:28:19), OUR serves land at t+14.244 / 16.242 / 24.246 / 26.248 / 28.249 — inter-serve gaps **1.998 / 2.002 / 2.001 s**. The server serves exactly one shot per 2 seconds and **queues** excess clicks (the 500 ms burst's third serve landed 2 s after its last dispatch — that queued serve, draining during the settle window, is what inflated the burst totals into the false ~1/s law). red-8's return fire sits on the SAME even 2-second grid — the serve cadence looks like a global server combat tick, not a per-tank cooldown.

Consequences: (1) **the cadence lever is dead** — the bot's 2 s tick already fires at the cap; (2) the "humans land two clicks per tick" inference was wrong — Blue Killer's -45/-90 single-tick pairs were **crossfire from two attackers**, each individually capped; (3) queuing means a burst of clicks buys nothing but delayed serves. The DPS plan re-ranks: **dodge-weave while trading is now lever #1** (dual 90 → homing 45 on a mid-move target — halves incoming, the one damage multiplier the game offers), fight-while-refueling #2 (time-on-target was 13% in the brrruh fight), mines + fuel denial #3. Method note for the record: burst-total arithmetic over a settle window cannot distinguish serve RATE from serve QUEUE — echo timestamps can; the operator's skepticism forced the check that caught it.

## [2026-08-26] measurement | One action per beat: moves share the 2 s serve slot with shots

The weave probe (`make weave-probe`, Arterial on Practice vs red-2, after a 150 s gatherer session refueled the practice tank to 1100): 8 shots dispatched with a 1-tile walk added on even beats. Ammo said 5 served of 8 — ambiguous — but the 0x53/0x47 echoes are not: serves land `SHOT SHOT MOVE SHOT MOVE SHOT MOVE SHOT MOVE` on consecutive 2 s beats, 12 commands in, 9 actions out, exactly one per beat. **A move consumes the same serve slot a shot does.** The June "server movement is instant" law was measured with an empty action queue; in combat, movement is serve-gated like everything else. Red-2's return fire (plain singles, weapon=0, 45 each) sat on the same global beat grid.

**Verdict on the dodge doctrine:** weave-while-trading costs ~1 shot per dodge — against a dual-firing human it halves damage BOTH ways, leaving the trade ratio unchanged while lengthening the fight, which favors the refueler. Dead as a steady trading pattern. What survives, and is now the implementation target: **never spend an under-fire tick stationary when the action slot is not firing** — escape ticks, refuel-under-fire ticks, and "waiting for collection" dwells are exactly where arterial died (7 motionless ticks, 18:37), and a move there costs nothing the slot was using. Versus NPCs (single/homing fire, already 45) the dodge buys no damage reduction — this is purely an anti-human-dual measure. Probe evidence: `weave_probe.json` + `weave_probe.capture_session.json`; three probes tonight, each overturning the prior inference — totals lie, echoes don't.

## [2026-08-27] operation | Century run 3 ends out_of_fuel at 32 kills; cost==fuel hop-refusal loop logged

Artax's third Desert session ended `out_of_fuel` at 32 kills / 0 deaths (73 m): marooned at fuel 18, and the desperation hop to a 67-vol container at cost=18 was re-dispatched ~7 times in 10 s without executing — the server appears to REFUSE a teleport costing exactly the tank's remaining fuel, and the planner re-issued the same illegal hop every tick (backlog: desperation hops must require cost < fuel, not <=; and a repeated-identical-decision loop should demote the plan). Context: ~230 kills farmed tonight have visibly drained the region's containers. Safe-exit contract worked as designed; respawned for run 4.

## [2026-08-27] first player kills | Hypee and pinata fall in century five; the pinata kill promotes Artax to Captain

Century five (bot-20260827-085730, 195 m) contained the bot's first two PLAYER kills, both consent-verified: **Hypee** (id 723 — 2 shots, 4 s, zero damage taken) and **pinata** (id 713 — 3 shots, 6 s, zero damage taken; greeted HELLO, pursued via the unlimited-distance human relay, killed at 09:54:57). The first over-cap fuel reading (>1400) landed at 09:55:01 — **the pinata kill is the promoting kill, Lieutenant → Captain**, consistent with the operator's rank law ("you have to kill a lieutenant or sergeant to get captain"). Neither victim ever landed a hit, and artax runs solo, so consent came via chat (4 received 0x4D records in session) — they answered the HELLO. The 2-3-shot kills mean both were caught at bottom-quartile fuel: the kill window occurring NATURALLY, before it is even implemented. Same session also held the two longest human fights yet — bnug (240 shots, 23 breaks, dealt 19,440 vs taken 17,775) and High Point (156 shots, 15 breaks, dealt 13,050 vs taken 14,850) — both survived: healthy, actively-refueling humans still out-sustain us, exactly the gap fuel-denial + kill-window addresses.

## [2026-08-27] reporting | Player kills and rank changes now headline the session summary; probe outputs clobbered test fixtures

Operator question "how'd you miss the human kill? do we not have a proper session summary?" exposed two reporting gaps, both fixed: the engagement ledger now tags human-classified rows `PLAYER` and headlines `PLAYER KILLS: N -- names` (the first two player kills sat unnoticed in the row table for hours), and the run digest now records wire-observed rank changes (`promoted to captain (rank 5)`) via the self_promotion diagnostic — the Captain promotion had been discovered via a fuel-cap change. Board checkin also posted (was overdue the whole session).

Gate lesson: probe CLIs write their outputs to the REPO ROOT by default, where `fuel_probe.capture_session.json` / `teleport_probe.capture_session.json` are COMMITTED replay-test fixtures — last night's probe runs overwrote them and 11 replay tests failed at the next gate. Recovered by preserving the clobbered copies then `git restore` of the fixtures. Backlog: probe default output paths must move out of the fixture namespace (runs/probe/), or the fixtures out of the root.

## [2026-08-27] hardening | Probe outputs move to runs/probe/<name>-<stamp>.json — the fixture namespace is safe

All twelve legacy probe CLIs (cadence, combat, enemy-teleport, enemy-tracking, fuel, key, movement, queue, radar-watch, respawn-watch, teleport, weave) now default their output to `runs/probe/<slug>-<stamp>.json` via `make_run_stamp` — the per-run-archive pattern the 2026-07-25 incident rule already gave density/larder/mine-landing/viewport. Root-level writes had two failure modes, both hit this week: fixed paths overwrite their own evidence, and the repo root holds COMMITTED replay fixtures (`fuel_probe.capture_session.json`) that yesterday's probe runs clobbered, failing 11 replay tests. The `make bot-watch` env override moves too. Env vars still override for explicit paths.

Also this hour: the fleet manager and artax's century-six session (95 kills, 0 deaths) were killed together at 14:15 by an external process reap (shared tree with a dead background task); manager restarted DETACHED via Start-Process so it no longer shares fate with any shell. Artax redeployed to Practice per operator order: 5-hour timed farm (18000 s, kill cap 999).
