---
title: Client Commands
tags: [js-client, protocol, commands]
related: [[js-source-map]], [[v-table-complete]], [[decode-coverage]]
sources: [tpclient.js lines 25-31 (K subclasses), lines 6-10 (va subclasses)]
fact_checked: 2026-06-19
confidence: high
verified: 2026-06-19 (every command class traced through JS source)
---

# Client Commands

Every command the TankPit client can send to the server, extracted from tpclient.js K-subclass hierarchy (binary game commands) and va-subclass hierarchy (connection/settings commands).

## Binary Game Commands (K subclasses)

These are sent during active gameplay. Each starts with a length byte, then the command character byte, then payload.

### Movement & Positioning

| Code | Char | Bytes | Class | Description | Byte Layout |
|------|------|-------|-------|-------------|-------------|
| 0x70 | `p` | 4 | Pb | **Move to tile** | `[4, 'p', x, y]` |
| 0x74 | `t` | 4 | Ob | **Teleport** (via map) | `[4, 't', x, y]` |
| 0x7A | `z` | 4 | Sb | **Scope move** (set viewport offset) | `[4, 'z', abs_x, abs_y]` where abs = offset + viewport origin |
| 0x5A | `Z` | 3 | Rb | **Scope extend** (shift view direction) | `[3, 'Z', direction]` where direction=0-8 (N/NE/E/SE/S/SW/W/NW/center) |

### Combat

| Code | Char | Bytes | Class | Description | Byte Layout |
|------|------|-------|-------|-------------|-------------|
| 0x73 | `s` | 6 | Lb | **Shoot at tile** | `[6, 's', x, y, target_id_lo, target_id_hi]` — id from target tank at tile, 0 if empty |
| 0x6D | `m` | 4-6 | Hb | **Fire** (chat command action) | `[6, 'm', action_type, x, y, use_special]` or `[4, 'm', action_type, use_special]` |
| 0x68 | `h` | 2 | Xb | **Detect nearest enemy** | `[2, 'h']` |
| 0x6B | `k` | 2 | Qb | **Self-deactivate/exit** | `[2, 'k']` |

### Resources

| Code | Char | Bytes | Class | Description | Byte Layout |
|------|------|-------|-------|-------------|-------------|
| 0x6A | `j` | 4 | Ub | **Pick up item** (obstacle at tile) | `[4, 'j', x, y]` |
| 0x64 | `d` | 4 | Vb | **Drop obstacle** | `[4, 'd', x, y]` |
| 0x62 | `b` | 4 | Tb | **Build/pickup obstacle** | `[4, 'b', x, y]` |
| 0x44 | `D` | 6 | Wb | **Deposit fuel** | `[6, 'D', amount_lo, amount_hi, x, y]` — amount is LE u16 |

### Scanning & Info

| Code | Char | Bytes | Class | Description | Byte Layout |
|------|------|-------|-------|-------------|-------------|
| 0x6C | `l` | 2 | Nb | **Open radar scan** | `[2, 'l']` |
| 0x66 | `f` | 2 | Mb | **Open map** | `[2, 'f']` |
| 0x69 | `i` | 2 | Yb | **Request inventory** | `[2, 'i']` |
| 0x76 | `v` | 2 | bc | **Request statistics** | `[2, 'v']` |
| 0x2A | `*` | 2 | Zb | **Active forces** | `[2, '*']` |
| 0x2F | `/` | 2 | ac | **Active players** | `[2, '/']` |
| 0x31 | `1` | 3 | $b | **Top 10 request** | `[3, '1', team_filter]` — 255=all, 0-3=team |

### Misc

| Code | Char | Bytes | Class | Description | Byte Layout |
|------|------|-------|-------|-------------|-------------|
| 0x3F | `?` | 2 | Jb | **Heartbeat/ping** (sent on game join) | `[2, '?']` |
| 0x2E | `.` | 2 | Kb | **Ping** (latency check, sent via F6 key) | `[2, '.']` |
| 0x21 | `!` | 2 | dc | **Keep-alive** (sent every 30s idle) | `[2, '!']` |
| 0x72 | `r` | 3 | cc | **Hotkey action** (equipment toggle) | `[3, 'r', key_code]` — codes 49-53 for equip slots 1-5 |

## Connection/Settings Commands (va subclasses)

These are sent as text strings (not binary), during connection and lobby.

| Code | Char | Class | Description | Format |
|------|------|-------|-------------|--------|
| 0x25 | `%` | wa | **AUTH** | `%AUTH !{version} {user_id}|{fingerprint} {magic}` |
| 0x2A | `*` | xa | **Select game** | `*{game_id}` |
| 0x2B | `+` | ya | **Join game** | `+{game_id}|{team}|{x}|{y}|{encoded_urls}` |
| 0x21 | `!` | Aa | **Binary command wrapper** | Binary blob with XOR encoding |
| 0x2D | `-` | Ba | **Quit** | `-` |
| 0x26 | `&` | Ca | **Error report** | `&{error_code}{error_message_xor}` |
| 0x5E | `^` | Ea | **Fatal error** | `^{error_code}{error_message_xor}` |
| 0x56 | `V` | Ha | **Volume change** | `V{enabled}{volume}` |
| 0x41 | `A` | Ia | **Autoscroll toggle** | `A{enabled}` |
| 0x4F | `O` | Ja | **Overall series** | `O{enabled}` |
| 0x43 | `C` | Ka | **Chat toggle** | `C{enabled}` |
| 0x53 | `S` | La | **Scale change** | `S{scale_percent}` |
| 0x50 | `P` | Ma | **Sprites change** | `P{sprite_indices}` |
| 0x48 | `H` | Na | **Hotkey map** | `H{key1,action1,key2,action2,...}` |
| 0x4D | `M` | Oa | **Chat messages** (custom?) | `M{...}` |

## Command Encoding Pipeline

1. Create command object (K subclass)
2. Call `.h()` to serialize to Uint8Array (includes length prefix)
3. Apply XOR via `za(b, b[0])` — length byte used as XOR offset
4. Wrap in `Aa` (binary command wrapper): prepend `!` code byte
5. Apply framing: 2-byte LE length prefix + payload
6. Send via WebSocket binary

## Timing Constraints

- **Chat cooldown**: 2400ms between chat messages (Bb class, line 24)
- **Command queue**: Max 2 queued commands (line 24: `2 <= a.j.length`)
- **Keep-alive**: Every 30,000ms when idle (line 71)
- **Latency check**: Not automatic — only on F6 keypress
- **Action processing**: One command per tick, 200ms idle tick rate

## Action Types (Hb.i field)

From the toolbar click handler and Cb function:

| Value | Meaning | Target Check |
|-------|---------|--------------|
| 0 | Attack team (needs teammate in zone) | Team check: any ally in 17×17 |
| 1 | Normal fire | No target check |
| 2 | Fire at player (needs anyone in zone) | Any tank in 17×17 |
| 3 | Send chat message | Various |
| 8 | Fuel search (auto-target nearest fuel) | Uses Db() Manhattan search |
| 9 | Equipment search (auto-target nearest equipment) | Uses Db() Manhattan search |
| 14 | Ferry search (auto-target nearest ferry) | Uses Db() Manhattan search |

## Shoot Command Details

The shoot command (Lb, code 's') is only sent when:
1. Target tile has an enemy tank (`h.h !== this.i.h` — different team), OR
2. Target tile has a mine from another team (overlay 255 !== own team), OR
3. Forced fire from double-click

The last 2 bytes (target_id) are set to the tank's `.id` if a tank is at the target tile, or 0 for empty ground fire. This matters because the server uses this for homing shot targeting.
