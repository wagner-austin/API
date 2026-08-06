---
title: Wire Decode Coverage Map
tags: [protocol, decode, coverage]
related:
  - "[[shoot-event-format]]"
  - "[[tank-registry]]"
  - "[[deactivation-format]]"
  - "[[v-table-complete]]"
  - "[[js-source-map]]"
  - "[[tank-freshness-model]]"
source_paths:
  - "tpclient.js"
  - "runs/bot/bot-20260619-053210.capture_session.json"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (unified dispatcher + JS source + production captures)
hubs: [protocol]
---

# Wire Decode Coverage Map

Complete mapping of every message type in the game client (`tpclient.js` V table) against our decode pipeline.[^1]

## Architecture (post-2026-06-19)

Every wire byte has **exactly one decoder**, reachable from `protocol.decode_message(msg_type, body)`.[^1] The 0x2E container envelope is handled by `protocol.decoders.tank.decode_0x2e_message` — a subtype-first dispatcher that routes to protocol decoders for tunneled subtypes (0x21, 0x28, 0x2E, 0x3D, 0x3E, 0x3F, 0x41, 0x42, 0x44, 0x46, 0x47, 0x49, 0x4A, 0x4C, 0x4F, 0x52, 0x53, 0x54, 0x56, 0x58, 0x5A, 0x64, 0x67, 0x74). A length=9 shortcut routes any other 9-byte 0x2E body to Og.h short form. The remainder falls through to `container.decoders.decode_container_message` for the four container-only subtypes (0x43 ContainerPickup, 0x45 MineDetonation, 0x4B MinePlacement) plus 1-byte TeleportLanded. No more dual paths, no length-based "blob" fallbacks.

## Coverage Table

Status legend: **FULL** = all known fields decoded and dispatched. **PARTIAL** = some fields intentionally dropped. **NONE** = not decoded. **WRONG** = decoded with incorrect field semantics.[^1]

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
| 0x4F | `O` | ch | RadarScanResult — a batch of per-tile cache + overlay writes (delta sync: reveals, volume corrections, removals via cache=0, mine clears via overlay>=8). **Single personality as of 2026-07-03**: the parallel "CombinedTileUpdate" decode was deleted after a 199-session corpus scan showed 0 top-level fires (all 1817 bodies arrive tunneled in 0x2E); count header fixed to LE u16 per JS `ch.h`. | FULL | — |
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

After 2026-06-19 unification, every 0x2E body goes through `decode_0x2e_message`. Subtype-first dispatch covers protocol-tunneled types in the table above; the subtypes below have no protocol counterpart and are dispatched by the container path.[^1]

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
container fallbacks.[^13]

The 3-byte 0x24 (tasks #88, "room-join confirmation?") and 1-byte 0x41
(task #89, "action ack?") candidates were also closed 2026-06-20: 0
corpus samples for either across the same 156-session sweep. Both were
speculative -- the wire paths don't exist in production.[^13]

The 9-byte "TankStatusShort -> Og.h" shortcut (the prior fallback that
routed any 9-byte 0x2E body to ``decode_tank_status_sync``) was
removed 2026-06-20. The "74/74 sane samples" it was built for were
all 0x43-prefixed two-record ContainerPickups; the subtype-first
multi-record dispatch above now claims them at their real semantics.[^13]

## Critical Gaps (ordered by impact)

(none open at end of 2026-06-19 -- see ``analysis_scripts/crack_tank_update.py`` for the audit
that closed the last "TankUpdate*" gap by tracing the misclassified bodies back to tunneled
0x56 Statistics, 0x42 BuildPickup, and 0x47 Movement handlers.)[^13]

## Tunneling cross-check (2026-06-19 corpus, 150 sessions)

The length-based container fallback that used to label bodies as
``tank_update_compact/extended/full`` is mostly residual: when the
subtype-first dispatch is run on the same 150 capture sessions, the
populations classified as "TankUpdate*" collapse from 597 to 1.[^13]

- **0x56 / Wg Statistics**: 239/239 ex-``TankUpdateFull`` samples now
  route via tunneled Statistics. All 239 decode to sane minutes/seconds
  bounds and the playtime/destroyed/score series is monotonic across
  the session -- ground-truth via
  ``analysis_scripts/crack_tank_update.py``.[^13]
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

Verified against production capture `runs/bot/bot-20260619-053210` and JS source.[^1]

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

Tunneled inside 0x2E: outer subtype is `0x3D`, inner is 11 bytes (carrying optional — defaults to 0 when absent in trimmed test fixtures).[^1]

### 0x2E TankStatusSync (V["."] / Og.h)

Verified against production capture and JS source. Same decoder handles both the 9-byte short form and the 13-byte form with fuel at the tail.[^1]

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

Verified against production capture `runs/bot/bot-20260619-050303` msg t+25.47s and JS source.[^1]

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

Verified against JS source.[^1]

```
[0]    status
[1:3]  victim_id (LE u16)
[3]    promo_eligible (1=eligible)
[4:6]  killer_id_raw (LE u16)
```

Post-processing: if `killer_id_raw >= 65530`, the kill was a mine — `killer_id = killer_id_raw - 65530` is the mine team and `is_mine_kill = True`.[^1]

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

## Machine-checked binding to `protocol/constants.py`

The coverage table above names wire message types; the block below binds
each one to its Python constant, and the `physics_claims` guard stage of
`make check` imports the symbol and compares. The table and the code
cannot drift apart without the gate going red.

This page is also the **canonical home for the 0x52 supervisor refusal
vocabulary**. Those eleven codes were previously stated only in passing
across seven pages with no single authority; `SUPERVISOR_ERROR_NAMES` is
bound here as a `members` claim, so the whole name table is verified as a
unit — an omitted code fails as loudly as an invented one.

Reverse coverage makes the binding total: every public symbol of
`tankpit_bot.protocol.constants` must be claimed exactly once, so a new
message type added to the module without a wiki claim fails the build.

```json claims
{
  "claims": [
    {
      "id": "msg-action-done",
      "code": "tankpit_bot.protocol.constants:MSG_ACTION_DONE",
      "value": 84,
      "means": "0x54 action completed"
    },
    {
      "id": "msg-active-forces",
      "code": "tankpit_bot.protocol.constants:MSG_ACTIVE_FORCES",
      "value": 42,
      "means": "0x2a active-forces listing"
    },
    {
      "id": "msg-active-players",
      "code": "tankpit_bot.protocol.constants:MSG_ACTIVE_PLAYERS",
      "value": 47,
      "means": "0x2f active-players listing"
    },
    {
      "id": "msg-build-pickup",
      "code": "tankpit_bot.protocol.constants:MSG_BUILD_PICKUP",
      "value": 66,
      "means": "0x42 movable-block pick up / drop"
    },
    {
      "id": "msg-cache-update",
      "code": "tankpit_bot.protocol.constants:MSG_CACHE_UPDATE",
      "value": 67,
      "means": "0x43 cache update"
    },
    {
      "id": "msg-chat",
      "code": "tankpit_bot.protocol.constants:MSG_CHAT",
      "value": 77,
      "means": "0x4d chat message"
    },
    {
      "id": "msg-deactivate",
      "code": "tankpit_bot.protocol.constants:MSG_DEACTIVATE",
      "value": 65,
      "means": "0x41 deactivation - fires for own kills too"
    },
    {
      "id": "msg-decoration",
      "code": "tankpit_bot.protocol.constants:MSG_DECORATION",
      "value": 78,
      "means": "0x4e decoration"
    },
    {
      "id": "msg-disconnect",
      "code": "tankpit_bot.protocol.constants:MSG_DISCONNECT",
      "value": 126,
      "means": "0x7e disconnect"
    },
    {
      "id": "msg-enemy-detect",
      "code": "tankpit_bot.protocol.constants:MSG_ENEMY_DETECT",
      "value": 72,
      "means": "0x48 nearest-enemy detection result"
    },
    {
      "id": "msg-equip-gain",
      "code": "tankpit_bot.protocol.constants:MSG_EQUIP_GAIN",
      "value": 103,
      "means": "0x67 equipment gained"
    },
    {
      "id": "msg-equip-toggle",
      "code": "tankpit_bot.protocol.constants:MSG_EQUIP_TOGGLE",
      "value": 116,
      "means": "0x74 equipment slot toggled"
    },
    {
      "id": "msg-fuel-deposit",
      "code": "tankpit_bot.protocol.constants:MSG_FUEL_DEPOSIT",
      "value": 100,
      "means": "0x64 fuel deposited into a container"
    },
    {
      "id": "msg-fuel-gain",
      "code": "tankpit_bot.protocol.constants:MSG_FUEL_GAIN",
      "value": 68,
      "means": "0x44 fuel gained"
    },
    {
      "id": "msg-inventory",
      "code": "tankpit_bot.protocol.constants:MSG_INVENTORY",
      "value": 73,
      "means": "0x49 inventory contents"
    },
    {
      "id": "msg-map-data",
      "code": "tankpit_bot.protocol.constants:MSG_MAP_DATA",
      "value": 76,
      "means": "0x4c full map blob, positions frozen at open time"
    },
    {
      "id": "msg-map-update",
      "code": "tankpit_bot.protocol.constants:MSG_MAP_UPDATE",
      "value": 90,
      "means": "0x5a map update (same byte as MSG_VIEWPORT)"
    },
    {
      "id": "msg-mine-detonate",
      "code": "tankpit_bot.protocol.constants:MSG_MINE_DETONATE",
      "value": 69,
      "means": "0x45 mine detonation"
    },
    {
      "id": "msg-mine-place",
      "code": "tankpit_bot.protocol.constants:MSG_MINE_PLACE",
      "value": 75,
      "means": "0x4b mine placement"
    },
    {
      "id": "msg-movement",
      "code": "tankpit_bot.protocol.constants:MSG_MOVEMENT",
      "value": 71,
      "means": "0x47 movement echo"
    },
    {
      "id": "msg-move-response",
      "code": "tankpit_bot.protocol.constants:MSG_MOVE_RESPONSE",
      "value": 61,
      "means": "0x3d movement response carrying the settled position"
    },
    {
      "id": "msg-overlay-update",
      "code": "tankpit_bot.protocol.constants:MSG_OVERLAY_UPDATE",
      "value": 64,
      "means": "0x40 overlay (mine ownership) update"
    },
    {
      "id": "msg-ping",
      "code": "tankpit_bot.protocol.constants:MSG_PING",
      "value": 96,
      "means": "0x60 ping reply"
    },
    {
      "id": "msg-promotion",
      "code": "tankpit_bot.protocol.constants:MSG_PROMOTION",
      "value": 43,
      "means": "0x2b rank promotion"
    },
    {
      "id": "msg-radar-result",
      "code": "tankpit_bot.protocol.constants:MSG_RADAR_RESULT",
      "value": 70,
      "means": "0x46 radar result - lists only newly revealed hidden entities"
    },
    {
      "id": "msg-radar-scan",
      "code": "tankpit_bot.protocol.constants:MSG_RADAR_SCAN",
      "value": 79,
      "means": "0x4f radar scan"
    },
    {
      "id": "msg-shoot",
      "code": "tankpit_bot.protocol.constants:MSG_SHOOT",
      "value": 83,
      "means": "0x53 shot fired - final byte is weapon type, not hit/miss"
    },
    {
      "id": "msg-statistics",
      "code": "tankpit_bot.protocol.constants:MSG_STATISTICS",
      "value": 86,
      "means": "0x56 statistics"
    },
    {
      "id": "msg-supervisor",
      "code": "tankpit_bot.protocol.constants:MSG_SUPERVISOR",
      "value": 82,
      "means": "0x52 supervisor refusal carrying one error code"
    },
    {
      "id": "msg-supervisor-text",
      "code": "tankpit_bot.protocol.constants:MSG_SUPERVISOR_TEXT",
      "value": 60,
      "means": "0x3c supervisor message in text form"
    },
    {
      "id": "msg-sync",
      "code": "tankpit_bot.protocol.constants:MSG_SYNC",
      "value": 63,
      "means": "0x3f session sync"
    },
    {
      "id": "msg-tank-entry",
      "code": "tankpit_bot.protocol.constants:MSG_TANK_ENTRY",
      "value": 40,
      "means": "0x28 tank entered the viewport"
    },
    {
      "id": "msg-tank-exit",
      "code": "tankpit_bot.protocol.constants:MSG_TANK_EXIT",
      "value": 41,
      "means": "0x29 tank left the viewport"
    },
    {
      "id": "msg-tank-info",
      "code": "tankpit_bot.protocol.constants:MSG_TANK_INFO",
      "value": 33,
      "means": "0x21 roster entry: name + team, no coordinates (the login dump)"
    },
    {
      "id": "msg-tank-pos",
      "code": "tankpit_bot.protocol.constants:MSG_TANK_POS",
      "value": 61,
      "means": "0x3d tank position (same byte as MSG_MOVE_RESPONSE)"
    },
    {
      "id": "msg-tank-remove",
      "code": "tankpit_bot.protocol.constants:MSG_TANK_REMOVE",
      "value": 88,
      "means": "0x58 tank removed from the viewport"
    },
    {
      "id": "msg-tank-stats",
      "code": "tankpit_bot.protocol.constants:MSG_TANK_STATS",
      "value": 46,
      "means": "0x2e per-tank stats sync"
    },
    {
      "id": "msg-tank-status",
      "code": "tankpit_bot.protocol.constants:MSG_TANK_STATUS",
      "value": 62,
      "means": "0x3e tank status"
    },
    {
      "id": "msg-tank-status-full",
      "code": "tankpit_bot.protocol.constants:MSG_TANK_STATUS_FULL",
      "value": 62,
      "means": "0x3e full tank status (same byte as MSG_TANK_STATUS)"
    },
    {
      "id": "msg-terrain-update",
      "code": "tankpit_bot.protocol.constants:MSG_TERRAIN_UPDATE",
      "value": 74,
      "means": "0x4a terrain patch - carries atomic ferry move pairs"
    },
    {
      "id": "msg-top10",
      "code": "tankpit_bot.protocol.constants:MSG_TOP10",
      "value": 49,
      "means": "0x31 leaderboard"
    },
    {
      "id": "msg-viewport",
      "code": "tankpit_bot.protocol.constants:MSG_VIEWPORT",
      "value": 90,
      "means": "0x5a viewport patch (same byte as MSG_MAP_UPDATE)"
    },
    {
      "id": "supervisor-error-already-there",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_ALREADY_THERE",
      "value": 6,
      "means": "code 6 - already at the requested tile"
    },
    {
      "id": "supervisor-error-cant-do",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_CANT_DO",
      "value": 0,
      "means": "code 0 - the action is not permitted at all"
    },
    {
      "id": "supervisor-error-cant-go",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_CANT_GO",
      "value": 1,
      "means": "code 1 - the destination is unreachable; a tank tile reads as impassable-occupied"
    },
    {
      "id": "supervisor-error-congratulations",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_CONGRATULATIONS",
      "value": 10,
      "means": "code 10 - congratulations (not a refusal)"
    },
    {
      "id": "supervisor-error-empty-container",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_EMPTY_CONTAINER",
      "value": 4,
      "means": "code 4 - the container is known-drained"
    },
    {
      "id": "supervisor-error-friendly-fire",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_FRIENDLY_FIRE",
      "value": 3,
      "means": "code 3 - refused: the target is on your own team"
    },
    {
      "id": "supervisor-error-insufficient-fuel",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_INSUFFICIENT_FUEL",
      "value": 8,
      "means": "code 8 - the teleport costs more fuel than the tank holds"
    },
    {
      "id": "supervisor-error-inventory-full",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_INVENTORY_FULL",
      "value": 7,
      "means": "code 7 - every equipment slot is at rank cap"
    },
    {
      "id": "supervisor-error-names",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_NAMES",
      "members": {
        "0": "cant_do",
        "1": "cant_go",
        "2": "uncontrollable",
        "3": "friendly_fire",
        "4": "empty_container",
        "5": "tank_full_clamp_receipt",
        "6": "already_there",
        "7": "inventory_full",
        "8": "insufficient_fuel",
        "9": "no_enemies",
        "10": "congratulations"
      },
      "means": "Canonical name for each 0x52 refusal code - the single home of this vocabulary."
    },
    {
      "id": "supervisor-error-no-enemies",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_NO_ENEMIES",
      "value": 9,
      "means": "code 9 - no enemies to detect"
    },
    {
      "id": "supervisor-error-tank-full",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_TANK_FULL",
      "value": 5,
      "means": "code 5 - at rank capacity; the clamp receipt"
    },
    {
      "id": "supervisor-error-uncontrollable",
      "code": "tankpit_bot.protocol.constants:SUPERVISOR_ERROR_UNCONTROLLABLE",
      "value": 2,
      "means": "code 2 - the tank is not controllable right now"
    },
    {
      "id": "text-msg-types",
      "code": "tankpit_bot.protocol.constants:TEXT_MSG_TYPES",
      "members": [
        96,
        36,
        37,
        42,
        43,
        45,
        82,
        61,
        126
      ],
      "means": "Message types delivered as text rather than XOR-encoded binary."
    },
    {
      "id": "is-text-message",
      "code": "tankpit_bot.protocol.constants:is_text_message",
      "law": "True when a message type is delivered as text rather than XOR-encoded binary - i.e. membership of TEXT_MSG_TYPES."
    }
  ]
}
```

[^1]: three-way cross-check, all on disk: `tpclient.js` (blob-pinned in frontmatter) is the JS side; the bot's decoder tree `src/tankpit_bot/protocol/` + `src/tankpit_bot/container/` is our side; production capture `runs/bot/bot-20260619-053210.capture_session.json` (frontmatter-pinned) plus `bot-20260619-050303` are the wire side. Standing receipt: every live session decodes through this exact pipeline — a wrong mapping surfaces as decode garbage immediately.
[^13]: corpus-sweep ground truth: `analysis_scripts/crack_tank_update.py` and `analysis_scripts/crack_tank_status_short.py` (both on disk, verified 2026-07-23) re-derive the 597→1 collapse and the per-type sample counts from the `runs/` capture corpus; the deletions themselves are 2026-06-20 commits in git history.
[^8]: `runs/bot/bot-20260619-053210` capture: 7/7 single-byte 0x2E bodies had subtype 0x54. The unified dispatcher requires 0x54 ActionDone to have inner ≥ 1 byte so the bare 1-byte form falls through to length-based teleport_landed
[^11]: 42 corpse messages (direction>=32) found across 18 tanks in all captures; JS `Pg.prototype.h` sets `d.direction = (d.direction & 240) !== 0 ? 33 : 32` on deactivation
[^12]: `src/tankpit_bot/protocol/decoders/tank.py:175` — `fuel: int | None = x16(data[10], data[11])`, i.e. `byte[10] + byte[11]*256` at inner offsets, after the subtype byte is stripped, matching this row exactly (re-verified 2026-08-06). The correlation figures behind it — 98/152 exact match with `FuelGain` at the same millisecond, the remainder explained by pre/post-update ordering inside one tick, and 8/15 sessions opening at 1100 (Private starting fuel, `RANK_FUEL[Rank.PRIVATE]`, machine-checked on [[client-constants]]) — come from the original 2026-06 sweep and are re-derivable by re-running it over the archive; the counts themselves are not stored.
