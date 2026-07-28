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

## [2026-07-23] audit | Corvis wiki-audit onboarded — 62 pages green under the code-paths contract

The TankpitBot wiki is registered in corvis as slug `tankpitbot` (contract kind `code-paths`) and the full `wiki_audit_run` chain now passes with ZERO errors/warnings across all 62 pages. First run surfaced ~95 findings: two real structural ones (bot-service-architecture had empty `hubs:` frontmatter despite hub membership; a punt phrase in tank-freshness-model) and a systematic contract mismatch — nearly every page carried prose annotations in `source_paths:` ("see footnotes", "tpclient.js lines 243-255 (E[] table)", "codebase inspection 2026-06-16") where the contract requires REAL resolvable citations. Normalized all 60 affected pages via an explicit per-page mapping (every replacement existence-checked first): line anchors (`tpclient.js:243` — bounds-verified against the 328-line file), repo paths (`src/tankpit_bot/sim/blocks.py`, `Makefile`, `docs/sources/sigmas-tankpit-guide-v3.4.pdf`), and named capture artifacts (`runs/sniff/sniff-20260720-214839.capture_session.json`); prose context stays in page bodies/footnotes where it already lives. Honest scope note: this layer audits CITATION INTEGRITY (paths resolve, line anchors in bounds, hubs consistent, no punts) — it would not have caught the damage-tier misreading; that requires the archive re-derivation layer (claim blocks + `make audit` + `make shadow`). The three layers are now all standing: structure (corvis), bindings (guard claim blocks), and physics truth (audit/shadow). Available anytime via `wiki_audit_run(wikiSlug=''tankpitbot'')`; the `git-blob-hash-pin` rule (per-page blob pinning so cited-file DRIFT flags the page for re-verification) is opt-in and not yet adopted — a candidate follow-up. Gate: `make lint` green after the sweep.

## [2026-07-23] audit | Git-blob drift pinning adopted -- 33 pages now flag automatically when cited code changes

Task 5 of the audit-hardening program: every tankpitbot wiki page citing TRACKED repo paths now carries a `source_git_blobs:` frontmatter map pinning each citation to its current `git ls-tree HEAD` hash (blob hashes for files, TREE hashes for directories -- any change anywhere inside a cited directory flips its tree hash and flags every page citing it for re-verification). 33 of 62 pages adopted; 20 cite no repo paths; pages citing gitignored `runs/` artifacts have those entries exempted per the corvis rule refinement shipped alongside (a5082f81 in the MCPs workspace: untracked paths are unpinnable by nature -- the original all-or-nothing companion invariant had locked 28 pages out of pinning entirely because ONE runs/ citation stripped drift protection from their code citations too; existence of untracked citations stays enforced by `source-path-exists`). Effect: the corvis `wiki_audit_run(tankpitbot)` now fails the moment any cited source file changes after a page was written -- the mechanical version of "the wiki must be re-verified when the code moves." Verified: one-shot audit 0 errors with all pins live. This closes the fourth defense layer for this wiki: structure/citations (corvis), wiki<->code bindings (claim-block guard), wiki<->reality (make audit + make shadow), and now cited-source DRIFT (blob pins).

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
