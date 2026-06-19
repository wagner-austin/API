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
