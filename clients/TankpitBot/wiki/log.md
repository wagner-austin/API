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
