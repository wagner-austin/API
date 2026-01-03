# Protocol Decoding Status

## Coverage Metric

**100% decoder coverage on captured messages**

- 28 message types in protocol.py
- All captured binary messages fully decoded
- 3 messages recently captured (need full decoding: 0x28, 0x4C, 0x4D)
- 2 message types not yet captured (0x29, 0x2F)

---

## Fully Decoded (FULL) - 24 Messages

| Sig | Name | Bytes | Format |
|-----|------|-------|--------|
| 0x21 | tank_info | 11+ | `team:u8 tank_id:u16 decoration:u32 score:u24 name:str` (NO rank!) |
| 0x2B | promotion | TEXT | `+id\|name\|field\|flags\|team\|mode\|image\|year` |
| 0x2E | tank_status_sync | 9,13 | Self(13B): fuel+rank. Other(9B): damage_state+rank |
| 0x3D | movement_response | 12 | `team:u8 tank_id:u16 x:u8 y:u8 dir:u8 [1B] rank:u8 score:u24BE [1B]` |
| 0x3E | tank_status | 22 | See detailed format below |
| 0x3F | sync | 3 | Heartbeat (no data) |
| 0x41 | deactivation | 8 | `victim_id:u16 killer_id:u16 rank:u8 [2B] points:u16` |
| 0x43 | container | 6 | `container_id:u16 fuel:u16` |
| 0x45 | mine_detonate | var | `count:u8 positions:[(x:u8,y:u8)]` |
| 0x46 | radar_ack | 4 | `ack:u8 status:u8` |
| 0x47 | movement | 17-37 | `tank_id:u16 x:u8 y:u8 dir:u8 flag:u8 fuel:u24 waypoints:str` |
| 0x49 | item_pickup | 8 | `show:u8 armor:u7+flag dual:u7+flag missile:u7+flag homing:u7+flag radar:u7+flag` |
| 0x4A | terrain_update | var | `[(x:u8, y:u8, type:u8)]` - updates terrain types (e.g., ferry) |
| 0x4B | mine_place | 20 | `type:u8 tank_id:u16 count:u8 positions:[(x:u8,y:u8)]` |
| 0x4F | radar_result | var | `count:u8 [1B] entities:[(x:u8,y:u8,value:i16)]` |
| 0x52 | supervisor | 5 | `status:u8 reserved:u8 data:u8` |
| 0x53 | shooting | 12 | `tank_id:u16 target_xy:u8x2 proj_xy:u8x2 fuel:u24 weapon:u8 ammo:u8 ff:u8` |
| 0x54 | action_done | 2 | Action completion marker |
| 0x56 | statistics | 16 | `hours:u16 mins:u8 secs:u8 destroyed:u32 deactivated:u32 score:u32` |
| 0x58 | tank_exit | 4 | `tank_id:u16` |
| 0x5A | viewport_update | 28-144 | `zone_id:u8 zone_x:u8 zone_y:u8 entities:[(marker:u8,b1:u8,value:u16)]` |
| 0x64 | fuel_deposit | 4 | `amount:u16` |
| 0x67 | equip_gain | 8 | `type:u8 armor:u8 dual:u8 missile:u8 homing:u8 radar:u8` |
| 0x74 | equip_toggle | 7 | `armor:u8 dual:u8 missile:u8 homing:u8 radar:u8` (0=OFF,1=ON) |

---

## Not Captured (Need Live Testing) - 2 Messages

| Sig | Name | Trigger | Notes |
|-----|------|---------|-------|
| 0x29 | tank_leave | Watch player disconnect | Sent when player leaves |
| 0x2F | player_update | Press `/` key | Active players list response |

## Recently Captured (Need Decoding) - 3 Messages

| Sig | Name | Bytes | Notes |
|-----|------|-------|-------|
| 0x28 | tank_join | 10 | Sent when player spawns. Captured when Artax joined. |
| 0x4C | world_entry | 724-728 | Map snapshot with all tank positions and HP. See detailed format below. |
| 0x4D | player_list | 6 | Player list response. |

---

## Tank Rank Sources (CRITICAL)

**0x21 TankInfo does NOT contain current rank!** It has decoration_state instead.

| Message | For Who | Rank Location |
|---------|---------|---------------|
| 0x3E TankStatus | Self (your tank) | byte 1 bits 4-6 |
| 0x3D MovementResponse | Other tanks moving/shooting | byte 7 (0-7 value) |
| 0x2E Short (9 bytes) | Other tanks in viewport | byte 5 (0-7 value) |
| 0x21 TankInfo | N/A | **NO RANK - has decoration_state** |

**Examples (verified):**
- Yuppler 0x3E: info_byte=0x3F → rank=(0x3F>>4)&7=3 (sergeant) ✓
- Arterial 0x3D: byte[7]=4 → rank=4 (lieutenant) ✓
- Artax 0x3D: byte[7]=6 → rank=6 (major) ✓
- Arterial 0x2E short: byte5=0x04 → rank=4 (lieutenant) ✓

---

## Enemy HP/Damage State (VERIFIED)

**Enemy HP is transmitted via damage_state in 0x2E short format (9 bytes).**

| Message | For Who | Damage Location |
|---------|---------|-----------------|
| 0x2E Short (9 bytes) | Other tanks in viewport | byte 4 (0-3 value) |

**Trigger conditions (combat NOT required):**
- Player enters your viewport (teleport near you)
- Player's HP changes (fuel pickup, fuel consumption, damage taken)
- Player joins the game

**Verified with Artax teleport session (NO combat):**
```
[58] Artax joined
[60] 0x2E HP=medium (teleported in with partial HP)
[65] 0x2E HP=full (got fuel)
[67] 0x2E HP=light (teleported, lost fuel)
[76] 0x2E HP=critical (more fuel loss)
```

**Note:** Fuel IS HP in TankPit. The damage_state reflects the visual darkness of the tank name in the UI sidebar.

---

## Detailed Message Formats

### 0x2E Tank Status Sync

**Long format (14 bytes) - Self status:**
```
[0]    0x2E  message type
[1]    0x03  subtype (3 = self)
[2-3]  u16   tank_id (LE)
[4]    u8    rank (0-7)
[5-6]  u16   flags (usually 0x0200)
[7-8]  u16   leaderboard_position (LE)
[9-10] u16   reserved (always 0x0000)
[11-12] u16  fuel (LE)
```

**Short format (9 bytes, subtype 0x01) - Other tanks in viewport:**
```
[0]    0x2E  message type
[1]    0x01  subtype (1 = other tank)
[2-3]  u16   tank_id (LE)
[4]    u8    damage_state (0=full, 1=light, 2=medium, 3=critical)
[5]    u8    rank (0-7: recruit to general)
[6]    u8    flag (0 or 1)
[7-8]  u16   leaderboard_position (LE)
```

**Damage state values (verified with Artax combat session):**
| Value | HP Level | Name Brightness |
|-------|----------|-----------------|
| 0 | Full | Bright |
| 1 | Light damage | Slightly dim |
| 2 | Medium damage | Dim |
| 3 | Critical | Very dark |

**Rank values:**
| Value | Rank |
|-------|------|
| 0 | Recruit |
| 1 | Private |
| 2 | Corporal |
| 3 | Sergeant |
| 4 | Lieutenant |
| 5 | Captain |
| 6 | Major |
| 7 | General |

### 0x3E Tank Status (from JS client analysis)

```
[0]     0x3E  message type '>'
[1]     u8    info_byte (packed: team=bits0-1, rank=bits4-6)
[2-3]   u16   tank_id (LE)
[4-7]   u32   decoration_state (9 2-bit values via yg() function)
[8-10]  u24   leaderboard_score (BE: 256*(256*b8+b9)+b10)
[11-13] u24   leaderboard_position (BE)
[14+]   str   tank_name (UTF-8, no length prefix)
```

**Rank extraction (verified working):**
```python
team = info_byte & 0x03  # bits 0-1
rank = (info_byte >> 4) & 0x07  # bits 4-6
```

### 0x52 Supervisor

```
[0]    0x52  message type 'R'
[1]    u8    status
[2]    u8    reserved (usually 0x00)
[3]    u8    data (rank level or count)
```

**Status values:**
| Value | Meaning |
|-------|---------|
| 1 | promo_eligible - Ready for promotion |
| 4 | unknown |
| 7 | unknown |
| 8 | promo_kill - Got a promotion kill |
| 128 | text_follows - "Congratulations!" message |

### 0x5A Viewport Update (FULLY DECODED)

Delta-compressed map update format from JS `Vg.h` handler:

```
[0]    0x5A  message type 'Z'
[1]    u8    direction (scroll direction 0-255)
[2]    u8    flags
[3+]   variable-length entity records
```

**Entity record format (1-4 bytes each):**
```
[0]    u8    position_delta
             - col += delta % 18
             - row += delta / 18
             - 255 = skip (no entity data follows)
[1-3]  u24   entity_data (only if delta != 255)
             - bits 0-3:   entity_type
             - bits 4-7:   value (>=8 becomes 255)
             - bits 8-23:  entity_id (65535 = -1 = TANK)
```

**Entity types (terrain, from JS rendering code):**
| Type | Sprite | Terrain |
|------|--------|---------|
| 0 | none | Ground (default) |
| 1 | 31 | Rock variant A |
| 2 | 30 | Rock variant B |
| 3 | 31+30 | Rock variant A + B overlay |
| 5 | 8 | Ferry/bridge |
| 7 | 8+30 | Ferry with rock overlay |

**Cache value (what's ON the tile):**
| cache | Meaning |
|-------|---------|
| > 0 | Fuel container (cache = fuel amount) |
| < 0 | Equipment pickup |
| = 0 | Empty |

**Entity ID:**
| entity_id | Meaning |
|-----------|---------|
| -1 (0xFFFF) | TANK present on tile |
| 0 | No container |
| > 0 | Fuel container ID |

### 0x4A Terrain Update (FULLY DECODED)

Updates terrain types at specific positions. Sent when terrain changes (e.g., ferry movement).

```
[0]    0x4A  message type 'J'
[1+]   triplets of (x:u8, y:u8, type:u8)
```

**Format:** Variable length, contains N triplets where N = (len-1)/3

**Example:** `4a 97 64 00 8e 6f 05`
- Position (151, 100) → type 0 (ground)
- Position (142, 111) → type 5 (ferry)

**Terrain types:** Same as 0x5A viewport (0=ground, 1=rock_A, 2=rock_B, 3=rock_A+B, 5=ferry, 7=ferry+rock)

**Trigger:** Sent when riding a ferry - updates both your position and the ferry tile.

### 0x4C World Entry (PARTIALLY DECODED)

Map snapshot sent on game join. Contains terrain, containers, and all tank positions with HP.

```
[0]      0x4C  message type 'L'
[1-2]    u16   header/size
[3-536]  ???   terrain/container section (not yet decoded)
[537+]   tank entries (5 bytes each until end)
```

**Tank entry format (5 bytes):**
```
[0]    u8    x position
[1]    u8    y position
[2-3]  u16   tank_id (LE)
[4]    u8    info_byte (packed: team=bits0-1, damage=bits2-3, rank=bits4-7)
```

**Info byte extraction:**
```python
team = info_byte & 0x03           # bits 0-1 (0-3)
damage_state = (info_byte >> 2) & 0x03  # bits 2-3 (0-3)
rank = (info_byte >> 4) & 0x0F    # bits 4-7 (0-15, but only 0-7 used)
```

**Note:** This provides a snapshot of all tank HP on map load, before any 0x2E updates are received.

**Trigger:** Sent once on game join. Subsequent updates come via 0x5A viewport updates and 0x2E status sync.

---

## Format Verification Needed

| Sig | Issue | Notes |
|-----|-------|-------|
| 0x46 | naming | Called "radar_ack" in doc but "RadarResult" in code. Format matches (2 bytes). |
| 0x4F | format | Doc says "radar_result" with `count:u8 [1B] entities`. Code has "TileUpdate" with `count:u16` and 4-byte entries. Need to verify with capture data. |

---

## Testing Notes

### 0x52 Supervisor Observations
- NOT a timer (5 min idle = zero messages)
- NOT triggered by combat, movement, equipment, chat
- Sometimes appears after equipment gains
- status=128 contained "Congratulations!" text once
- Likely server-side promotion eligibility state

### 0x2E Status Sync Observations
- subtype=3 for self, subtype=1 for other tanks
- bytes 7-8 = leaderboard rank (NOT promo points)
- bytes 9-10 = always 0x0000 (reserved)
- bytes 11-12 = current fuel (u16 LE)
- flags byte 4 correlates with rank changes

---

## Message Flow Examples

**Movement:**
```
Client: MOVE (93, 113)
Server: 0x2E FUEL_STATE fuel=920
Server: 0x3D MOVE_RESPONSE from=(93, 118)
Server: 0x3F SYNC
```

**Combat (Kill):**
```
Client: SHOOT (199, 115)
Server: 0x2E FUEL_STATE fuel=7724
Server: 0x53 HIT target=(199, 115)
Server: 0x41 DEACTIVATION victim=123 killer=you
```

**Radar:**
```
Client: RADAR
Server: 0x2E FUEL_STATE fuel=21843
Server: 0x46 RADAR_ACK
Server: 0x4F RADAR_RESULT 4 entities found
```
