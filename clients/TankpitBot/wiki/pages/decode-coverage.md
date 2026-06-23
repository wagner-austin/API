---
title: Wire Decode Coverage Map
tags: [protocol, decode, coverage]
related: [[shoot-event-format]], [[tank-registry]], [[deactivation-format]], [[v-table-complete]], [[js-source-map]], [[tank-freshness-model]]
sources: [tpclient.js V table, runs/bot/bot-20260619-053210 capture, 2026-06-19 unification audit]
fact_checked: 2026-06-19
confidence: high
verified: 2026-06-19 (unified dispatcher + JS source + production captures)
---

# Wire Decode Coverage Map

Complete mapping of every message type in the game client (`tpclient.js` V table) against our decode pipeline.

## Architecture (post-2026-06-19)

Every wire byte has **exactly one decoder**, reachable from `protocol.decode_message(msg_type, body)`. The 0x2E container envelope is handled by `protocol.decoders.tank.decode_0x2e_message` — a subtype-first dispatcher that routes to protocol decoders for tunneled subtypes (0x21, 0x28, 0x2E, 0x3D, 0x3E, 0x3F, 0x41, 0x42, 0x44, 0x46, 0x47, 0x49, 0x4A, 0x4C, 0x4F, 0x52, 0x53, 0x54, 0x56, 0x58, 0x5A, 0x64, 0x67, 0x74). A length=9 shortcut routes any other 9-byte 0x2E body to Og.h short form. The remainder falls through to `container.decoders.decode_container_message` for the four container-only subtypes (0x43 ContainerPickup, 0x45 MineDetonation, 0x4B MinePlacement) plus 1-byte TeleportLanded. No more dual paths, no length-based "blob" fallbacks.

## Coverage Table

Status legend: **FULL** = all known fields decoded and dispatched. **PARTIAL** = some fields intentionally dropped. **NONE** = not decoded. **WRONG** = decoded with incorrect field semantics.

| Msg | Char | JS Handler | Description | Our Status | Gap |
|-----|------|-----------|-------------|-----------|-----|
| 0x21 | `!` | Tf | TankInfo: name, team, decorations, score | FULL | — |
| 0x28 | `(` | Uf | TankEntry: team, rank, position, score | FULL | — |
| 0x29 | `)` | Vf | TankExit: team, id, was_silent, was_eliminated | FULL | — |
| 0x2A | `*` | Xg | ActiveForces: team counts | FULL | — |
| 0x2B | `+` | Rf | Promotion: new_rank, was_promoted | FULL | Disambiguated from text WorldInfo by 3-byte body length |
| 0x2E | `.` | Og | TankStatusSync: team, id, rank, damage, lb_score, promo_state, fuel | FULL | — |
| 0x2F | `/` | Yg | ActivePlayers: id+rank list | FULL | — |
| 0x31 | `1` | Zg | Top10 leaderboard | FULL | — |
| 0x3C | `<` | wg | SupervisorText: free text from server | FULL | 0 production samples (practice room) -- decoded per JS spec |
| 0x3D | `=` | Mg | MovementResponse: team, id, pos, direction, damage, rank, lb_score, carrying | FULL | — |
| 0x3E | `>` | Qf | TankStatusFull: team, rank, decorations, lb_score, name | FULL | — |
| 0x3F | `?` | vg | Sync/heartbeat | FULL | — |
| 0x40 | `@` | ah | OverlayUpdate: mine tile patches | FULL | — |
| 0x41 | `A` | Pg | Deactivation: status, victim, promo_eligible, killer, is_mine_kill | FULL | — |
| 0x42 | `B` | Jg | BuildPickup: tank_id, src x/y, drop x/y, direction, obstacle_type, flag | FULL | — |
| 0x44 | `D` | Rg | FuelGain: absolute fuel total | FULL | — |
| 0x45 | `E` | dh | MineDetonate: positions | FULL | — |
| 0x46 | `F` | Fg | RadarAck: found flag | FULL | — |
| 0x47 | `G` | Lg | Movement: start pos, waypoints, direction | FULL | — |
| 0x48 | `H` | Tg | EnemyDetect: x, y, team, rank, tank_id | FULL | — |
| 0x49 | `I` | Xf | Inventory: counts + enabled flags | FULL | — |
| 0x4A | `J` | bh | TerrainUpdate: terrain tile patches | FULL | — |
| 0x4B | `K` | Dg | MinePlacement: type, tank, positions | FULL | — |
| 0x4C | `L` | Ig | MapData: all tank positions + fuel dots | FULL | — |
| 0x4D | `M` | Qg | Chat: sender, type, position | FULL | — |
| 0x4E | `N` | Sf | Decoration: tank_id, slot, level | FULL | — |
| 0x4F | `O` | ch | CombinedTileUpdate / RadarScanResult (structural disambiguation) | FULL | — |
| 0x52 | `R` | xg | CommandResult: reset_action, close_map, error_code | FULL | — |
| 0x53 | `S` | Gg | ShootEvent: team, shooter, source pos, target pos, weapon | FULL | — |
| 0x54 | `T` | Kg | ActionDone: bare completion ping | FULL | — |
| 0x56 | `V` | Wg | Statistics: playtime, destroyed, deactivated, score | FULL | — |
| 0x58 | `X` | Ug | TankRemove: server stopped per-tank updates (NOT a death — use 0x41). **Bot dispatch is a no-op as of 2026-06-22** — registry entry is kept so pursuit can keep firing homing at the cached coords. | FULL | Semantics clarified 2026-06-20, dispatch promoted to no-op 2026-06-22 -- see [[tank-freshness-model]] and [[bot-behavior-contract]] |
| 0x5A | `Z` | Vg | ViewportUpdate: position + entity tiles | FULL | — |
| 0x64 | `d` | Sg | FuelDeposit: absolute fuel total | FULL | — |
| 0x67 | `g` | Wf | EquipmentGain: counts per slot | FULL | — |
| 0x74 | `t` | Yf | EquipmentToggle: enabled flags | FULL | — |

## Container Subtypes (inside 0x2E envelope)

After 2026-06-19 unification, every 0x2E body goes through `decode_0x2e_message`. Subtype-first dispatch covers protocol-tunneled types in the table above; the subtypes below have no protocol counterpart and are dispatched by the container path.

| Subtype | Bytes | Type | Status | Notes |
|---------|-------|------|--------|-------|
| 0x43 | 5, 9, 13, ... (1 + 4N) | ContainerPickup (multi-record) | FULL | Each body carries N pickup records, each ``[x, y, remaining_lo, remaining_hi]``. Corpus 2026-06-20: 2653/80/2 samples at N=1/2/3. JS V.C = $g handler at ``tpclient.pretty.js:4743``. |
| 0x45 | 3+ | MineDetonation | FULL | — |
| 0x4B | 15 | MinePlacement | FULL | — |
| (any) | 1 | TeleportLanded | FULL | Always 0x54 subtype in production captures[^8] |

PositionUpdate (0x24 13-byte), DeactivationDeath (0x43 7-byte),
PlayerListShort (0x79 4-byte), PlayerListExtended (0x79 7-byte),
TankLeave (6-byte length-based) and TankRegistry (16-20 byte
length-based) were deleted 2026-06-20 after a corpus sweep of 150
sessions / 48,304 0x2E bodies proved zero production fires. The
corresponding protocol subtypes (0x3D MovementResponse, 0x41
Deactivation, 0x21 TankInfo, 0x58 TankRemove, 0x44 FuelGain, 0x47
Movement) cover every body that used to flow into those length-based
container fallbacks.

The 3-byte 0x24 (tasks #88, "room-join confirmation?") and 1-byte 0x41
(task #89, "action ack?") candidates were also closed 2026-06-20: 0
corpus samples for either across the same 156-session sweep. Both were
speculative -- the wire paths don't exist in production.

The 9-byte "TankStatusShort -> Og.h" shortcut (the prior fallback that
routed any 9-byte 0x2E body to ``decode_tank_status_sync``) was
removed 2026-06-20. The "74/74 sane samples" it was built for were
all 0x43-prefixed two-record ContainerPickups; the subtype-first
multi-record dispatch above now claims them at their real semantics.

## Critical Gaps (ordered by impact)

(none open at end of 2026-06-19 -- see ``analysis_scripts/crack_tank_update.py`` for the audit
that closed the last "TankUpdate*" gap by tracing the misclassified bodies back to tunneled
0x56 Statistics, 0x42 BuildPickup, and 0x47 Movement handlers.)

## Tunneling cross-check (2026-06-19 corpus, 150 sessions)

The length-based container fallback that used to label bodies as
``tank_update_compact/extended/full`` is mostly residual: when the
subtype-first dispatch is run on the same 150 capture sessions, the
populations classified as "TankUpdate*" collapse from 597 to 1.

- **0x56 / Wg Statistics**: 239/239 ex-``TankUpdateFull`` samples now
  route via tunneled Statistics. All 239 decode to sane minutes/seconds
  bounds and the playtime/destroyed/score series is monotonic across
  the session -- ground-truth via
  ``analysis_scripts/crack_tank_update.py``.
- **0x47 / Lg Movement**: every 14-byte 0x2E body in the corpus is a
  tunneled Movement carrying 1-2 waypoint chars. Long obstacle-rich
  paths can stretch this much further; ``inner >= 12`` is the only
  guard, so the tunneled path fires for every 0x47-prefixed length.
  ``TankUpdateExtendedDict`` was deleted as proven dead.
- **0x42 / Jg BuildPickup**: 2/2 ex-``TankUpdateCompact`` samples decode
  cleanly as own-tank obstacle drops (src/drop adjacent, byte 7 =
  ``obstacle_type``, never matched as 0/1 -- shipped W6 was treating
  it as ``is_bridge: bool``; that field has been corrected to
  ``obstacle_type: int``).
- **0x28 / Uf TankEntry**: 1 remaining 10-byte ``TankUpdateCompact``
  candidate routed by lowering the tunneled dispatch threshold from
  ``inner >= 10`` to ``inner >= 9`` -- matches JS Uf.h (which reads
  ``a[0..8]``) and our existing ``decode_tank_entry`` minimum. After
  the change, 0 length-10 bodies remain in the corpus's length-based
  fallback.
- **0x2E / Og TankStatusSync (short form)**: 74/74 ex-``TankStatusShort``
  samples now route via ``decode_tank_status_sync`` -- a hard fix:
  the container ``TankStatusShort`` layout was wrong on every byte
  position (74/74 produced rank > 8). The previous dispatch was
  silently feeding ``damage_state`` (and probably wiping it) from
  the wrong byte for every enemy tank update of this length. See
  ``analysis_scripts/crack_tank_status_short.py``.

## Proven Field Mappings

### 0x3D MovementResponse (V["="] / Mg.h)

Verified against production capture `runs/bot/bot-20260619-053210` and JS source.

```
[0]    flags (team in bits 0-1)
[1:3]  tank_id (LE u16)
[3]    x position
[4]    y position
[5]    direction (0-31 = alive facing, 32-33 = dead corpse)[^11]
[6]    damage_state (0-3)
[7]    rank (0-8)
[8:11] lb_score (24-bit BE: 256*(256*b[8]+b[9])+b[10])
[11]   carrying (obstacle-carry flag, per JS `a.la = 0 !== this.j`)
```

Tunneled inside 0x2E: outer subtype is `0x3D`, inner is 11 bytes (carrying optional — defaults to 0 when absent in trimmed test fixtures).

### 0x2E TankStatusSync (V["."] / Og.h)

Verified against production capture and JS source. Same decoder handles both the 9-byte short form and the 13-byte form with fuel at the tail.

```
[0]    subtype/team
[1:3]  tank_id (LE u16)
[3]    damage_state (0-3) — rank_category on init, overwritten with damage during gameplay
[4]    rank (0-8)
[5:8]  lb_score (24-bit BE: 256*(256*b[5]+b[6])+b[7])
[8]    promo_state — present if len >= 9
[9]    has_fuel_bar — present if len >= 10
[10:12] fuel (LE u16: b[10] + b[11]*256) — present if len >= 12; verified 98/152 exact match with FuelGain at same ms; 8/15 sessions start at 1100 (Private starting fuel)[^12]
```

### 0x53 ShootEvent (V.S / Gg.h)

Verified against production capture `runs/bot/bot-20260619-050303` msg t+25.47s and JS source.

```
[0]    team (flags byte, bits 0-1)
[1:3]  shooter_id (LE u16)
[3]    source_x (shooter position when fired)
[4]    source_y
[5]    target_x (impact tile)
[6]    target_y
[7]    aim_x (aim tile -- where the gun is pointed)
[8]    aim_y
[9]    weapon (0=single, 1=dual, 2=missile, 3=homing)
```

### 0x41 Deactivation (V.A / Pg.h)

Verified against JS source.

```
[0]    status
[1:3]  victim_id (LE u16)
[3]    promo_eligible (1=eligible)
[4:6]  killer_id_raw (LE u16)
```

Post-processing: if `killer_id_raw >= 65530`, the kill was a mine — `killer_id = killer_id_raw - 65530` is the mine team and `is_mine_kill = True`.

See [[deactivation-format]] for hit/kill semantics and [[shoot-event-format]] for shot-feedback behavior.

### Supervisor error codes (0x52 'R')

```
[0]  reset_action (1=reset to idle)
[1]  close_map (1=close map view)
[2]  error_code — index into:
     0 = "You can't do this"
     1 = "You can't go there!"
     2 = "Uncontrollable tank"
     3 = "Friendly fire!"
     4 = "Empty container"
     5 = "Tank full"
     6 = "You are already there!"
     7 = "Inventory full"
     8 = "Insufficient fuel"
     9 = "No enemies found"
     10 = "Congratulations!"
     128+ = custom text (remaining bytes)
```

[^8]: `runs/bot/bot-20260619-053210` capture: 7/7 single-byte 0x2E bodies had subtype 0x54. The unified dispatcher requires 0x54 ActionDone to have inner ≥ 1 byte so the bare 1-byte form falls through to length-based teleport_landed
[^11]: 42 corpse messages (direction>=32) found across 18 tanks in all captures; JS `Pg.prototype.h` sets `d.direction = (d.direction & 240) !== 0 ? 33 : 32` on deactivation
[^12]: fuel = byte[10] + byte[11]*256 (inner offsets, after subtype byte stripped); 98/152 exact match with FuelGain at same ms; mismatches from pre/post-update timing within same tick; 8/15 sessions start at 1100 (Private fuel)
