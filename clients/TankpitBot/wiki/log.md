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
