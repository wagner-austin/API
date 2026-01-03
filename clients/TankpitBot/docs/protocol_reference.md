# Tankpit Protocol Reference

## Overview

Messages are either **TEXT** (pipe-delimited) or **BINARY** (XOR-encoded).

### Text Messages (Plain ASCII, `|` delimited)
| Char | Name | Format |
|------|------|--------|
| `=` | JOIN_CONFIRM | `=<team>\|<join_date>\|<name>\|<rank>\|<eq1>\|<eq2>\|<eq3>\|<eq4>` |
| `+` | WORLD_INFO | `+<id>\|<name>\|<field>\|<flags>\|<team>\|<mode>\|<image>\|<year>` |
| `%` | AUTH | `%AUTH !be <session>\|<token>\|<id> h` |
| `*` | FORCES | `*<count>` (text format during lobby) |
| `$` | UNKNOWN | `$<num>\|<num>` |
| `-` | HEARTBEAT | `-` (empty) |

**Example:** `=2|Sep. 25, 2012|Yuppler|2|9|9|9|9`

### Binary Messages (XOR-encoded)
XOR encoding with session-specific table:
```python
xor_table[i] = static_key[i] ^ magic[i % len(magic)]
```

- `static_key`: 1000-char key from `xor_static_key.txt`
- `magic`: 20-char session key captured at login
- **Message type byte (first byte) is NOT XOR encoded**
- **Data bytes (from byte 1 onwards) ARE XOR encoded**

---

## Client → Server (Prefix: `!` / 0x21)

### Decoded

| Type | ID | Name | Data | Status |
|------|----|------|------|--------|
| 4 | 112 | MOVE | `[x, y]` single bytes | ✅ VERIFIED |
| 4 | 106 | PICKUP | `[x, y]` single bytes | ✅ VERIFIED |
| 4 | 116 | TELEPORT | `[x, y]` single bytes | ✅ VERIFIED |
| 6 | ? | SHOOT | `[x, y]` target coords | ✅ VERIFIED |
| 2 | 63 | RADAR | (no data) | ✅ VERIFIED |

**XOR Decoding (offset0):**
```python
decrypted[0] = body[0]  # '!' prefix unchanged
decrypted[i] = body[i] ^ xor_table[i-1]  # for i >= 1
```

**Command structure:**
```
byte 0: 0x21 ('!')
byte 1: type (4=movement, 6=shoot, 2=radar)
byte 2: id (112=MOVE, 106=PICKUP, 116=TELEPORT)
byte 3+: data (varies by command)
```

### Not Decoded / Unknown

| Type | ID | Observed | Notes |
|------|----|----------|-------|
| ? | ? | Toggle commands | Armor, dual shots, etc. |
| ? | ? | Chat messages | If any |
| ? | ? | Deposit fuel | Command to deposit fuel |
| ? | ? | Buy items | Shop interactions |

---

## Server → Client Message Types (from client JS)

### Full Message Type Map

| Char | Hex | Format | Handler | Description | Status |
|------|-----|--------|---------|-------------|--------|
| `.` | 0x2E | BINARY | `Og` | Tank updates (subtypes by length) | ⚠️ PARTIAL |
| `!` | 0x21 | MIXED | `Tf` | Tank info (mostly text, some binary) | 🔍 TODO |
| `=` | 0x3D | TEXT | `Mg` | Join confirm (team, date, name, equipment) | ✅ VERIFIED |
| `G` | 0x47 | BINARY | `Lg` | Movement command (tank id, path waypoints) | ✅ VERIFIED |
| `S` | 0x53 | BINARY | `Gg` | Shooting/hit event (shooter id, target pos) | ✅ VERIFIED |
| `A` | 0x41 | BINARY | `Pg` | Deactivation/kill (victim, killer, points) | ✅ VERIFIED |
| `C` | 0x43 | BINARY | - | Container fuel update (id, fuel amount) | ✅ VERIFIED |
| `D` | 0x44 | BINARY | `Rg` | Fuel gain (tank id, auto-flag) | 🔍 TODO |
| `d` | 0x64 | BINARY | `Sg` | Fuel deposit (amount deposited) | ✅ VERIFIED |
| `H` | 0x48 | BINARY | `Tg` | Enemy detection (pos x/y, rank, team) | 🔍 TODO |
| `I` | 0x49 | BINARY | `Xf` | Item pickup confirmation (equipment gained) | ✅ VERIFIED |
| `g` | 0x67 | BINARY | `Wf` | Equipment gain (equipment flags) | ✅ VERIFIED |
| `t` | 0x74 | BINARY | `Yf` | Equipment toggle (5 enabled flags) | ✅ VERIFIED |
| `K` | 0x4B | BINARY | `Dg` | Mine placement (owner id, position x/y) | ✅ VERIFIED |
| `E` | 0x45 | BINARY | `dh` | Mine detonation (positions array) | ✅ VERIFIED |
| `F` | 0x46 | BINARY | `Fg` | Radar acknowledgement | ✅ VERIFIED |
| `M` | 0x4D | BINARY | `Qg` | Chat message (tank id, message type) | 🔍 TODO |
| `X` | 0x58 | BINARY | `Ug` | Tank exit/disconnect (tank id) | ✅ VERIFIED |
| `Z` | 0x5A | BINARY | `Vg` | Map update (zone x, zone y, tile data) | 🔍 TODO |
| `(` | 0x28 | BINARY | `Uf` | Tank entry (rank, position, name) | 🔍 TODO |
| `)` | 0x29 | BINARY | `Vf` | Tank exit (id, eliminated flag) | 🔍 TODO |
| `>` | 0x3E | BINARY | `Qf` | Tank status (rank, equipment, pos) | 🔍 TODO |
| `+` | 0x2B | TEXT | `Rf` | World info / promotion | ✅ VERIFIED |
| `N` | 0x4E | BINARY | `Sf` | Decoration/award (tank id, index) | 🔍 TODO |
| `V` | 0x56 | BINARY | `Wg` | Statistics (playtime, kills, points) | 🔍 TODO |
| `*` | 0x2A | TEXT | `Xg` | Active forces count | ✅ DETECTED |
| `/` | 0x2F | BINARY | `Yg` | Active players list (tank ids) | 🔍 TODO |
| `1` | 0x31 | BINARY | `Zg` | Top 10 leaderboard | 🔍 TODO |
| `~` | 0x7E | TEXT | `xe` | Connection lost signal | 🔍 TODO |
| `` ` `` | 0x60 | TEXT | `we` | Ping response | 🔍 TODO |
| `R` | 0x52 | BINARY | `xg` | Supervisor msg (status 1,4,7,8,128=text) | ⚠️ PARTIAL |
| `%` | 0x25 | TEXT | - | Auth response | ✅ DETECTED |
| `$` | 0x24 | TEXT | - | Unknown | 🔍 TODO |
| `-` | 0x2D | TEXT | - | Heartbeat (empty) | ✅ DETECTED |

### Currently Decoded & Verified

#### Binary Messages (`.` / 0x2E subtypes)
Note: Subtypes are XOR-encoded and vary by session. Use decoded signature to identify.

| Signature | Name | Data | Status |
|-----------|------|------|--------|
| 3 bytes | SYNC | Heartbeat | ✅ VERIFIED |
| 4 bytes, decoded 0x46 | RADAR_ACK | Radar acknowledgement | ✅ VERIFIED |
| 4 bytes, decoded 0x58 | TANK_EXIT | tank_id (player left/disconnected) | ✅ VERIFIED |
| 4 bytes, decoded 0x64 | FUEL_DEPOSIT | amount deposited (u16 LE) | ✅ VERIFIED |
| 5 bytes | BLOCKED | Path blocked response | ✅ VERIFIED |
| 6 bytes, decoded 0x43 | CONTAINER | container_id + fuel (u16 LE each) | ✅ VERIFIED |
| 7 bytes, decoded 0x74 | EQUIP_TOGGLE | 5 on/off flags | ✅ VERIFIED |
| 8 bytes, decoded 0x41 | DEACTIVATION | Kill/death event | ✅ VERIFIED |
| 8 bytes, decoded 0x67 | EQUIP_GAIN | Equipment gained flags | ✅ VERIFIED |
| 12 bytes | HIT | Hit confirmation | ✅ DETECTED |
| 14 bytes | FUEL_STATE | bytes 12-13 = fuel u16 LE | ✅ VERIFIED |
| 17-21 bytes | MOVE_RESPONSE | bytes 4-5 = FROM position | ✅ VERIFIED |
| variable, decoded 0x4F | RADAR_RESULT | Fuel/entity positions | ✅ VERIFIED |
| variable, decoded 0x4B | MINE_PLACED | owner_id + position | ✅ VERIFIED |
| variable, decoded 0x45 | MINE_EXPLODE | count + position array | ✅ VERIFIED |

#### Text Messages (pipe-delimited)
| Char | Hex | Name | Format | Status |
|------|-----|------|--------|--------|
| `=` | 0x3D | JOIN_CONFIRM | `=team\|date\|name\|rank\|eq...` | ✅ VERIFIED |
| `+` | 0x2B | WORLD_INFO | `+id\|name\|field\|flags\|...` | ✅ VERIFIED |
| `%` | 0x25 | AUTH | `%AUTH !be session\|token\|id h` | ✅ DETECTED |
| `*` | 0x2A | FORCES | `*count` | ✅ DETECTED |

### Protocol Decoder Module

A comprehensive decoder is available at `src/tankpit_bot/protocol.py` with:
- 24 message type dataclasses
- Automatic XOR decoding
- Helper functions (x16, x24 for byte combining)

**Usage:**
```python
from tankpit_bot.protocol import decode_message, TankPosition, FuelGain
result = decode_message(msg_type, xor_decoded_data)
```

### Game Constants (from client JS)

**Tank Ranks** (0-7): Recruit, Private, Corporal, Sergeant, Lieutenant, Captain, Major, General

**Equipment Types** (5 items):
1. Armor shields
2. Dual shots
3. Missile shots
4. Homing shots
5. Extra radars

**Team Colors** (0-3): Red, Purple, Blue, Orange

### Priority Messages to Decode

| Priority | Char | Why |
|----------|------|-----|
| HIGH | `=` | Tank position update - get current positions |
| HIGH | `H` | Enemy detection - radar results |
| HIGH | `S` | Shooting/hit - combat feedback |
| HIGH | `A` | Deactivation - kill confirmation |
| MEDIUM | `G` | Movement paths - predict enemy movement |
| MEDIUM | `I` | Inventory - equipment status |
| LOW | `M` | Chat messages |

---

## High Priority Message Byte Layouts (from client JS)

### `=` (0x3D) - Tank Position Update (Mg handler)

```
Byte 0-1: tank_id (X function: low + 256*high)
Byte 2:   x coordinate
Byte 3:   y coordinate
Byte 4:   direction
Byte 5:   rank (0-7)
Byte 6-10: fuel (256*(256*a[8]+a[9])+a[10])
Byte 11:  weapon type
```

### `H` (0x48) - Enemy Detection/Radar (Tg handler)

```
Byte 0-1: target_tank_id (X function)
Byte 2:   target x coordinate
Byte 3:   target y coordinate
Byte 4:   target rank (0-7)
Byte 5:   target team color (0-3)
```

Calculates bearing (N, S, E, W, NE, SE, SW, NW) from your position.

### `S` (0x53) - Shooting/Hit Event (Gg handler)

```
Byte 0-1: shooter_id (X function)
Byte 2:   target x
Byte 3:   target y
Byte 4:   projectile x
Byte 5:   projectile y
Byte 6-8: fuel (256 shift)
Byte 9:   weapon type
Byte 10:  ammo count
Byte 11:  friendly fire flag
```

### `A` (0x41) - Deactivation/Kill (Pg handler)

```
Byte 0-1: victim_id (X function)
Byte 2-3: killer_id (X function)
Byte 4:   rank
Byte 5-6: X function
Byte 7-8: points (65530 threshold for bonus)
```

Displays "has been deactivated by" message.

### `G` (0x47) - Movement Paths (Lg handler)

```
Byte 0-1: tank_id (X function)
Byte 2:   start x
Byte 3:   start y
Byte 4:   direction
Byte 5:   flag
Byte 6-8: fuel (256 shift)
Byte 9+:  waypoint array
```

### Helper Function: X(a, b)

```javascript
X(a, b) = (a & 255) + 256 * (b & 255)
```

Combines two bytes into 16-bit value (little-endian style).

---

## Server Tick Rate

**Tick interval: 2000 ms (2 seconds)**

Commands are processed on a 2-second tick cycle:
- HIT response intervals: 1995-2007 ms (avg 2002 ms)
- Shots/sec: 0.5
- Commands sent faster than tick rate are queued

---

## HIT Message Format (0x2E len=12)

```
Byte 0:    0x2E = message type (.)
Byte 1:    subtype (varies by session)
Bytes 2-5: XOR decoded to get first 4 data bytes
Bytes 6-7: target_x, target_y (XOR decoded)
Bytes 8-9: Repeat of target
Bytes 10-11: Additional data (flags?)
```

**Example:** Shot at (187, 138) produces decoded bytes 5-6 = `0xBB, 0x8A` = (187, 138)

---

## Deactivation Message Format (0x2E len=8)

Wrapped inside 0x2E message, decoded first byte = `0x41` ('A').
**Same format for kills and deaths** - check victim_id to determine if you died.

```
Raw: 0x2E + subtype + 6 data bytes
Decoded (from byte 1):
  [0] 0x41 = 'A' (deactivation marker)
  [1-2] victim_id (little-endian)
  [3-4] killer_id (little-endian)
  [5-6] rank/points data
```

**Examples:**
- You killed someone: `41001e02003c02` (victim=7680, killer=2)
- You got killed: `41033c02003002` (victim=15363, killer=2)

**Death Indicators:**
- 8-byte message with decoded `0x41` where victim_id = your tank
- Fuel spikes to ~65508 (overflow)
- Fuel resets ~20s later on respawn
- 4-byte notification follows: `2e41533d` → `2b0100`

---

## Item Pickup Message Format (0x2E len=8, subtype 0x49)

Wrapped inside 0x2E message with subtype byte 0x49 ('I').
Sent when equipment is picked up from the map.

```
Raw: 0x2E 0x49 + 6 data bytes
Decoded (from byte 1):
  [0] 0x67 = 'g' (item pickup marker)
  [1] 0x01 = constant
  [2] armor_count
  [3] unknown (always 0)
  [4] missile_count
  [5] homing_count
  [6] unknown (always 0)
```

**Examples:**
- 8 homing shots: decoded `67010000000800`
- 2 homing shots: decoded `67010000000200`
- 7 armor shields: decoded `67010700000000`
- 5 armor shields: decoded `67010500000000`
- 7 missile shots: decoded `67010000070000`

**Equipment Slots (from client JS `gc` array):**
| Byte | Index | Item Type |
|------|-------|-----------|
| 2 | 0 | Armor shields |
| 3 | 1 | Dual shots |
| 4 | 2 | Missile shots |
| 5 | 3 | Homing shots |
| 6 | 4 | Extra radars |

All 5 equipment types have distinct byte positions.

---

## Radar Result Message Format (0x2E, decoded 0x4F)

Radar scan results are sent as 0x2E messages that decode to start with 0x4F ('O').

```
Raw: 0x2E [subtype] + data bytes (variable length)
Decoded (from byte 1):
  [0] 0x4F = 'O' (radar result marker)
  [1] count = number of entities found
  [2] 0x00 = constant
  [3...] entity records (4 bytes each)

Entity Record (4 bytes):
  [0] x coordinate
  [1] y coordinate
  [2] value_lo
  [3] value_hi
  value = value_lo | (value_hi << 8) as SIGNED 16-bit

Value interpretation (from client JS cache field):
  - Positive (0x0000-0x7FFF): fuel container with amount
  - Negative (0x8000-0xFFFE): equipment (abs value = type?)
  - 0xFFFF (-1): tank/entity
```

**Example:** Radar at (151, 68) found 4 fuel containers:
```
decoded: 4f0400973dbb03974609009a3d44039a44e600
         O  4  00 [records...]

Records:
  (151, 61) fuel=955
  (151, 70) fuel=9
  (154, 61) fuel=836
  (154, 68) fuel=230
```

---

## Equipment Toggle Message Format (0x2E, decoded 0x74)

Equipment toggle state is sent as 7-byte messages that decode to start with 0x74 ('t').

```
Raw: 0x2E [subtype] + 5 data bytes
Decoded (from byte 1):
  [0] 0x74 = 't' (equipment toggle marker)
  [1] armor: 0=OFF, 1=ON
  [2] dual: 0=OFF, 1=ON
  [3] missile: 0=OFF, 1=ON
  [4] homing: 0=OFF, 1=ON
  [5] radar: 0=OFF, 1=ON
```

**Examples:**
```
decoded: 740000000001  # only radar ON
decoded: 740101010101  # all equipment ON
decoded: 740000000000  # all equipment OFF
decoded: 740100000001  # armor and radar ON
```

Server sends this message whenever equipment toggle state changes (pressing R key cycles through equipment).

---

## Mine Message Formats

### Mine Mechanics
- Placing mines creates a **3x3 grid** of mines centered on player position
- Shooting enemy mines triggers **chain reaction** detonations
- Mine drop command: type=4, id=98 or id=100

### Mine Drop Command (Client → Server)
```
Raw: 0x21 + XOR encoded data
Decoded:
  [1] type = 4 (movement-type command)
  [2] id = 98 or 100 (mine placement)
  [3] x coordinate (center of 3x3 grid)
  [4] y coordinate (center of 3x3 grid)
```

### Mine Placement Confirmation (Server → Client, decoded 0x4B)
```
Raw: 0x2E [subtype] + data bytes
Decoded (from byte 1):
  [0] 0x4B = 'K' (mine placement marker)
  [1-2] owner_id (little-endian)
  [3] x coordinate
  [4] y coordinate
```

### Mine Detonation/Chain Reaction (Server → Client, decoded 0x45)
```
Raw: 0x2E [subtype] + data bytes
Decoded (from byte 1):
  [0] 0x45 = 'E' (mine explosion marker)
  [1] count = number of mines detonated
  [2...] position pairs (2 bytes each: x, y)
```

**Example:** Chain reaction of 3 mines:
```
decoded: 4503 4e9f 4f9f 509f
         E  3  (78,159) (79,159) (80,159)
```

---

## Container Fuel Update (0x2E, decoded 0x43)

Fuel containers on the map have unique IDs separate from tank IDs. Server broadcasts container fuel levels.

```
Raw: 0x2E [subtype] + 4 data bytes
Decoded (from byte 1):
  [0] 0x43 = 'C' (container marker)
  [1-2] container_id (little-endian u16)
  [3-4] fuel_amount (little-endian u16)
```

**Notes:**
- Container IDs are distinct from tank IDs (no overlap observed)
- Fuel amount of 0 means container is depleted/empty
- Server may send updates when containers are picked up or respawn

**Examples:**
```
decoded: 43722d8901  # container 11634, fuel=393
decoded: 43722d0000  # container 11634, fuel=0 (depleted)
decoded: 4375336400  # container 13173, fuel=100
```

---

## Tank Exit/Disconnect (0x2E, decoded 0x58)

Indicates a player left the game or disconnected. Different from kill (0x41).

```
Raw: 0x2E [subtype] + 2 data bytes
Decoded (from byte 1):
  [0] 0x58 = 'X' (exit marker)
  [1-2] tank_id (little-endian u16)
```

**Notes:**
- Tank IDs match those seen in TANK_MOVE messages
- Same tank ID may appear multiple times if player reconnects/disconnects
- Global broadcast - you see all exits across the map

**Examples:**
```
decoded: 581902  # tank 537 left
decoded: 585f02  # tank 607 left
decoded: 588202  # tank 642 left
```

---

## Equipment Gain (0x2E, decoded 0x67)

Indicates equipment was gained (separate from 0x49 item pickup confirmation).

```
Raw: 0x2E [subtype] + 6 data bytes
Decoded (from byte 1):
  [0] 0x67 = 'g' (equipment gain marker)
  [1] type (usually 0x01)
  [2-4] zeros
  [5-6] equipment flags (bitfield)
```

**Equipment Flag Positions:**
- Bit 0: armor
- Bit 1: dual
- Bit 2: missile
- Bit 3: homing
- Bit 4: radar

**Examples:**
```
decoded: 67010000000003  # flags=3 (armor+dual?)
decoded: 67010000000002  # flags=2 (dual?)
decoded: 67010000000300  # flags=3 at position 5
```

---

## Fuel Deposit (0x2E, decoded 0x64)

Indicates fuel was deposited to base.

```
Raw: 0x2E [subtype] + 2 data bytes
Decoded (from byte 1):
  [0] 0x64 = 'd' (deposit marker)
  [1-2] amount (little-endian u16)
```

**Example:**
```
decoded: 64f102  # deposited 753 fuel
```

---

## Radar Acknowledgement (0x2E, decoded 0x46)

Server acknowledgement after using radar (S key).

```
Raw: 0x2E [subtype] + 2 data bytes
Decoded (from byte 1):
  [0] 0x46 = 'F' (radar ack marker)
  [1-2] data bytes (usually 0x00, 0x01)
```

**Notes:**
- Appears after radar command is sent
- Followed by RADAR_RESULT (0x4F) with actual scan results

**Example:**
```
decoded: 460001  # radar acknowledged
```

---

## Message Flow Examples

### Movement
```
Client: [SENT] MOVE: (93, 113)         # Request move to (93, 113)
Server: [FUEL:0x1c] 7184 (-13)         # Fuel decreased (movement cost)
Server: [POS:FROM] (93, 118)           # Confirms you moved FROM (93, 118)
Server: [RECEIVED] SYNC: 2e0d40        # Heartbeat
```

### Radar
```
Client: [SENT] CMD: ! type=2 id=63     # Radar command
Server: [FUEL] 21843 (-10)             # Fuel decreased by 10
Server: [RADAR] 4 found - fuel: (151,61)=955 (151,70)=9 (154,61)=836 (154,68)=230
```

### Combat (Kill)
```
Client: [SENT] SHOOT: (199, 115)       # Fire at target
Server: [FUEL] 7724 (-46)              # Fuel cost
Server: [HIT] len=12                   # Hit confirmation with target coords
Server: [DEACTIVATE] len=8             # Kill: decoded 0x41 + victim/killer IDs
```

### Combat (Death)
```
Server: [HIT] len=12                   # Enemy hit you (multiple times)
Server: [FUEL] 65508 (+50000)          # Fuel overflow = death indicator
Server: [DEACTIVATE] len=8             # Death: decoded 0x41, you are victim
Server: [4-byte] 2e41533d              # Death notification
... 20 seconds later ...
Server: [FUEL] 6494                    # Respawn with reset fuel
```

### Equipment Pickup
```
Client: [SENT] PICKUP: (122, 124)      # Long-press at equipment location
Server: [FUEL] 22126 (-12)             # Small fuel cost for pickup
Server: [ENTITY] sub=0x69 len=26       # Inventory state update
Server: [ITEM_PICKUP] 2e49073509...    # Equipment gained confirmation
         decoded: 67010000000800       # = 8 homing shots
Server: [GAME:EQUIPMENT] 8 homing shots gained  # DOM scraper confirms
```

---

## Blocked Movement (VERIFIED)

When movement is blocked, the server:
1. Sends a 5-byte response: `2e XX XX XX XX`
2. Game displays "You can't go there!"
3. **Partial movement**: Tank moves as far as possible before obstacle

**Test Cases (Session: hwvoiew1x26uiv6zlvas):**

| Request | Result | Notes |
|---------|--------|-------|
| MOVE (102,124) from (102,125) | Blocked | Terrain, stayed at 102,125 |
| MOVE (105,132) from (102,125) | Partial | Terrain, ended at 105,128 |
| MOVE (98,128) from (105,128) | Success | Moved to 98,128 |
| MOVE (97,119) from (98,125) | Partial | Enemy mine, ended at 97,121 |

**Blocking response hex**: `2e6347320d` (5 bytes)

---

## Testing Checklist

### Movement
- [x] Basic movement (up/down/left/right)
- [x] Diagonal movement
- [x] Movement blocked by wall/terrain
- [x] Movement blocked by enemy mine
- [x] Partial movement (moves as far as possible)
- [ ] Movement blocked by another tank
- [ ] Movement blocked by water

### Combat
- [x] Shooting (type=6) - command decoded
- [x] Hit detection response (HIT: 0x2E len=12, target coords at bytes 5-6)
- [x] Kill notification (0x2E len=8, decoded 0x41 + victim/killer IDs)
- [x] Death response (same as kill, but you are victim_id)
- [x] Death indicators (fuel spike to ~65508, 4-byte notification)
- [x] Respawn (~20s delay, fuel reset)

### Resources
- [x] Radar cost (-10 fuel)
- [x] Movement cost (-1 per tile)
- [x] Fuel deposit (-100)
- [x] Fuel pickup (+100)
- [ ] HP tracking
- [ ] Ammo tracking

### Items
- [ ] Toggle commands (armor, dual shots, etc.)
- [ ] Item usage
- [ ] Shop purchase

---

## File Structure

```
docs/
├── protocol_reference.md    # This file - overview
├── fuel_encoding.md         # Detailed fuel & position encoding
└── protocol.md              # Original protocol notes

src/tankpit_bot/
├── sniffer.py               # FuelTracker, PositionTracker
├── decoder.py               # Command decoder
└── commands.py              # Command constants
```

---

## Next Steps (Priority Order)

1. **Blocked movement detection** - What message indicates path blocked?
2. **Current position (TO)** - Is there a message with destination position?
3. **HP tracking** - Where is health stored?
4. **Other entity positions** - Decode 0x13 entity messages
5. **Combat messages** - Hit, kill, death, respawn
