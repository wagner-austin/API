# Tankpit Protocol Documentation

This document describes the WebSocket protocol used by Tankpit.com and the discovery process using the sniffer module.

## Overview

Tankpit uses WebSocket connections for real-time game communication. Since no public protocol documentation exists, we capture and analyze traffic to reverse-engineer the protocol.

## Shared Architecture

Both sniffer and probe inherit from `BrowserSession` (in `browser/session.py`) which provides:

- **CDP Setup**: WebSocket event handlers for frame capture
- **WebSocket Prototype Hook**: Captures game's WebSocket instance via `Page.addScriptToEvaluateOnNewDocument`
- **Intel Gathering**:
  - Console listener (filters for WS/Hook/WebSocket keywords)
  - WebSocket URL logging
  - JavaScript WebSocket debug check
  - Script URL logging
- **Magic Key Capture**: Reads `tankpit.magic` for XOR encoding
- **Login Integration**: Guest or account authentication

### Sniffer Architecture

The sniffer uses Playwright with Chrome DevTools Protocol (CDP) to intercept WebSocket traffic:

```
┌─────────────────┐
│  Playwright     │
│  sync_api       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BrowserSession │  Console listener
│  CDP Handlers   │  Intel gathering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CDP Session    │  Network.enable
│  Event Handlers │  webSocketCreated
└────────┬────────┘  webSocketFrameSent
         │           webSocketFrameReceived
         ▼
┌─────────────────┐
│  CaptureSession │
│  JSON output    │
└─────────────────┘
```

### Probe Architecture

The probe sends commands via WebSocket injection (not synthetic JS events):

```
┌─────────────────┐
│  BrowserSession │  WebSocket prototype hook
│  CDP Handlers   │  Console + intel gathering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  XOR Encoding   │  Static key + session magic
│  Command Build  │  encode_frame(XOR'd bytes)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  WebSocket      │  window.__capturedWS.send()
│  Injection      │  (or fallback to tankpit.ws)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Toggle Keys    │  First press: WS open
│  State Machine  │  Second press: JS close
└─────────────────┘
```

**Why WebSocket Injection?** Synthetic JavaScript KeyboardEvents don't work because browsers set `isTrusted: false` on programmatically created events. The game ignores untrusted events, so we must inject commands directly via the WebSocket.

## Running the Sniffer

```bash
make sniff
```

This will:
1. Launch a Chromium browser via Playwright
2. Navigate to tankpit.com
3. Capture all WebSocket frames for 60 seconds (configurable)
4. Save results to `runs/sniff/latest.capture_session.json` by default

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TANKPIT_URL` | `https://tankpit.com` | Target URL |
| `TANKPIT_OUTPUT` | `runs/sniff/latest.capture_session.json` | Output file |
| `TANKPIT_HEADLESS` | `false` | Run headlessly |
| `TANKPIT_DURATION_MS` | `60000` | Capture duration |
| `TANKPIT_LIVE_DECODE` | `true` | Show decoded messages in real-time |
| `TANKPIT_PREFER_ACCOUNT` | `false` | Skip guest login, use account directly |

### Live Decode Mode

When `TANKPIT_LIVE_DECODE=true` (default), messages are decoded and printed in real-time as you play:

```
[SENT] AUTH: %AUTH !be 62997|...
[RECV] ROOM_LIST: room=4 name=World (Meltdown)
[SENT] SELECT: room=4
[RECV] JOIN_CONFIRM: room=4 tank=Yuppler
[SENT] CMD: !1 2d43fe
[RECV] STATE: len=24 bytes
```

This makes protocol discovery much easier - play the game and see exactly what commands each action sends.

## Running the Probe

```bash
make probe
```

This will:
1. Join a game using account credentials
2. Install WebSocket prototype hook to capture game's WebSocket
3. Send known commands via WebSocket injection:
   - `f` - Map open (XOR command via WebSocket)
   - `f` - Map close (JavaScript keypress toggle)
   - `s` - Radar (XOR command)
   - `d` - Mine (XOR command)
   - `q` - Quit (plain command)
4. Record WebSocket responses from the server
5. Save results to `probe_session.json`

### Toggle Key Behavior

Some keys (like `f` for map) are toggle keys that open/close UI elements:

- **First press**: Sends XOR-encoded command via WebSocket to open UI
- **Second press**: Sends JavaScript keypress to close UI (the game handles closing locally)

This matches how the game works: opening the map requires a server command, but closing it is handled client-side.

### WebSocket Prototype Hook

The probe captures the game's WebSocket instance before the page loads:

```javascript
window.__capturedWS = null;
const origSend = WebSocket.prototype.send;
WebSocket.prototype.send = function(data) {
    if (!window.__capturedWS && this.readyState === 1) {
        window.__capturedWS = this;
    }
    return origSend.call(this, data);
};
```

This hook is injected via `Page.addScriptToEvaluateOnNewDocument` in CDP.

## Captured CDP Events

| Event | Description |
|-------|-------------|
| `Network.webSocketCreated` | New WebSocket connection established |
| `Network.webSocketFrameSent` | Client sent a message |
| `Network.webSocketFrameReceived` | Server sent a message |

## Capture Session Format

```json
{
  "session_id": "uuid",
  "start_timestamp_ms": 1234567890000,
  "end_timestamp_ms": 1234567950000,
  "base_url": "https://tankpit.com",
  "magic": "session-specific-xor-key",
  "messages": [
    {
      "timestamp_ms": 1234567890100,
      "direction": "sent",
      "payload": "{\"type\":\"ping\"}",
      "ws_url": "wss://tankpit.com/socket"
    }
  ]
}
```

The `magic` field contains the session-specific XOR key captured from `tankpit.magic` JavaScript variable after login. This key is used for encoding game commands.

## Protocol Analysis Workflow

1. **Run Sniffer**: `make sniff`
2. **Play the Game**: Log in and play in the browser window
3. **Analyze Capture**: Review `runs/sniff/latest.capture_session.json`
4. **Identify Patterns**: Look for:
   - Authentication handshake
   - Game join/leave messages
   - Movement commands (likely x/y coordinates)
   - Shooting/ability commands
   - State sync from server
5. **Document Findings**: Update this file with discovered message types

## Type Definitions

All protocol types are defined in `src/tankpit_bot/types/`:

- `CapturedMessage`: Single WebSocket frame (`types/message.py`)
- `CaptureSession`: Complete capture with metadata (`types/session.py`)
- `CDPWebSocketCreatedEvent`: CDP event for new connections (`types/cdp.py`)
- `CDPWebSocketFrameEvent`: CDP event for frame data (`types/cdp.py`)

## Discovered Protocol Format

WebSocket endpoint: `wss://dorothy.tankpit.com/ws/`

### Message Structure

Each message is base64-encoded binary with:
- **Header**: 2 bytes, little-endian **body length**
- **Body**: Text with pipe-delimited (`|`) fields, or binary game commands

Example: `02 00 2a 34` = length 2, body `*4` (SELECT room 4)

### Lobby Message Types

| Prefix | Type | Description |
|--------|------|-------------|
| `%AUTH` | AUTH | Authentication with session token |
| `+` | ROOM_LIST | Room/world information |
| `*` | SELECT | Room selection |
| `=` | JOIN_CONFIRM | Room join confirmation |
| `$` | RESPONSE | Server response/ack |

### Game Controls

The game uses **click-to-move** controls, not WASD. Keyboard keys are for actions:

| Key | Action | Description | Protocol |
|-----|--------|-------------|----------|
| Space | Shoot | Fire at mouse position | XOR cmd |
| S | Radar | Ping nearby entities | XOR type=2 |
| D | Mine | Place mine at current position | XOR type=2 |
| F | Open Map | Toggle full map view | XOR type=2 |
| E | Nearest Enemy | Target nearest enemy | XOR type=2 |
| 1-5 | Equipment | Toggle armor/dual/missile/homing/radar | XOR type=3 |
| Arrow Keys | Scope | Pan camera N/S/E/W | XOR type=3 |
| PageUp/Down | Scope | Pan camera NE/SE | XOR type=3 |
| Home/End | Scope | Pan camera NW/SW | XOR type=3 |
| I | Inventory | Show inventory | XOR type=2 |
| C | Statistics | Show game statistics | XOR type=2 |
| X | Active Forces | Show team force counts | XOR type=2 |
| / | Active Players | Show players in room | XOR type=2 |
| T/R/P/B/O | Top 10 | Leaderboard (all/red/purple/blue/orange) | XOR type=3 |
| F6 | Ping | Check server latency | XOR type=2 |
| L | Sound | Toggle sound on/off | Plain V140/V040 |
| A | Autoscroll | Toggle autoscroll | Plain A0/A1 |
| H | Help | Show help overlay | Client-only |
| M | Tips | Toggle tips display | Client-only |
| N | Next Tip | Show next tip | Client-only |
| Q | Quit | Exit current game | Plain `-` |

Mouse controls:
- **Single click**: Move to position
- **Double click**: Fire at position
- **Click and hold**: Pick up / Drop items

### Game Command Types

Commands start with `!` followed by XOR-encoded bytes. The encoding key changes per session.

#### Message Format

```
[2-byte length LE] + '!' + [type_byte] + [cmd_byte] + [data...]
```

- **type_byte**: Session-specific prefix (changes each login)
- **cmd_byte**: Command identifier (XOR'd with session key)
- **data**: Optional payload (coordinates, etc.)

#### Session XOR Encoding

The protocol uses per-session XOR encoding to prevent replay attacks. The encoding uses two keys:

1. **Static Key**: A 1000-character string embedded in `tpclient-*.js`, starting with `Y1DcZy...`
2. **Magic Key**: A session-specific string set in `tankpit.magic` after login

The XOR encoding formula (from decompiled client):
```javascript
qb[rb] = staticKey.charCodeAt(rb) ^ magic.charCodeAt(rb % magic.length)
```

The sniffer automatically captures the magic key via `page.evaluate("tankpit.magic")` after login and stores it in the session JSON.

The same action produces different wire bytes each session:

| Session | Type Byte | MAP cmd | RADAR cmd |
|---------|-----------|---------|-----------|
| 1 | `!` (0x21) | `?` (0x3f) | `5` (0x35) |
| 2 | `(` (0x28) | `.` (0x2e) | `$` (0x24) |
| 3 | `h` (0x68) | `h` (0x68) | `b` (0x62) |
| 4 | `"` (0x22) | `6` (0x36) | `&` (0x26) |
| 5 | `8` (0x38) | `'` (0x27) | `-` (0x2d) |

#### Discovered Commands

**Known Command IDs** (decoded after XOR):

| ID (dec) | ID (hex) | Key | Type | Action | Description |
|----------|----------|-----|------|--------|-------------|
| 42 | 0x2a | X | 2 | ACTIVE_FORCES | Show team force counts |
| 46 | 0x2e | F6 | 2 | PING | Server latency check |
| 47 | 0x2f | / | 2 | ACTIVE_PLAYERS | Show players in room |
| 49 | 0x31 | T/R/P/B/O | 3 | TOP10 | Leaderboard (+extra byte: ff=all, 00-03=team) |
| 63 | 0x3f | (click) | 2 | ENTER_GAME | Click to enter game |
| 90 | 0x5a | Arrows/Page | 3 | SCOPE | Pan camera (+extra byte: direction) |
| 102 | 0x66 | S | 2 | RADAR | Ping nearby entities |
| 104 | 0x68 | E | 2 | NEAREST_ENEMY | Target nearest enemy |
| 105 | 0x69 | I | 2 | INVENTORY | Show inventory |
| 107 | 0x6b | D | 2 | MINE | Deploy 3x3 mine grid at current position (no payload) |
| 108 | 0x6c | F | 2 | MAP_OPEN | Open full map view |
| 114 | 0x72 | 1-5 | 3 | TOGGLE_EQUIPMENT | Toggle equipment (+extra byte: slot) |
| 118 | 0x76 | C | 2 | STATISTICS | Show player statistics |
| 106 | 0x6a | Long press | 4 | PICKUP_MOVE | Move to pickup fuel/equipment (+2 byte payload: X, Y) |
| 112 | 0x70 | Click | 4 | MOVE | Move to coordinates (+2 byte payload: X, Y) |
| 116 | 0x74 | Map click | 4 | MAP_TELEPORT | Teleport via map (+2 byte payload: X, Y) |
| 115 | 0x73 | Space | 6 | SHOOT | Fire at coordinates (+4 byte payload: X, Y, target_id) |

**Movement Command** (type=4, cmd_id=0x70):

The movement command is sent when clicking on the game canvas to move the tank.

| Wire Format | Decoded | Description |
|-------------|---------|-------------|
| `! + type=4 + id=112 + X + Y` | Move to (X, Y) | 5 bytes total |

- X and Y are single bytes (0-255), meaning the **map is 256x256**
- Coordinates match the LOCATION displayed in-game

Example from captured session (ending at LOCATION 93,100):
```
Encoded: 05 00 21 2f 2e 5d 64  →  len=5, body=`! / . ] d`
Decoded: type=4, id=112, X=93, Y=100  →  Move to (93, 100)
```

**Shoot Command** (type=6, cmd_id=0x73):

The shoot command is sent when pressing Spacebar to fire at the mouse position.

| Wire Format | Decoded | Description |
|-------------|---------|-------------|
| `! + type=6 + id=115 + X + Y + id_lo + id_hi` | Fire at (X, Y) | 7 bytes total |

- X and Y are target coordinates (single bytes, 0-255)
- Bytes 3-4 are **target entity ID** (little-endian, 0x0000 if no specific target)
- Server calculates trajectory from tank position to target
- Shot type (regular, dual, missile, homing) determined by enabled equipment state

Tested shots at empty ground (no target):
```
Fire West  → payload=[102, 64, 0, 0]
Fire North → payload=[106, 60, 0, 0]
Fire East  → payload=[113, 64, 0, 0]
Fire SE    → payload=[108, 70, 0, 0]
```

Tested shots at multiple enemy tanks (entity IDs confirmed):
```
Red tank at 186,96     → payload=[186, 96, 31, 2]   entity_id=543 (0x021F)
Red tank at 209,123    → payload=[209, 123, 31, 2]  entity_id=543 (same tank)
Blue private at 230,165→ payload=[230, 165, 46, 2]  entity_id=558 (0x022E)
Red-4 private at 63,132→ payload=[63, 132, 27, 2]   entity_id=539 (0x021B)
```

Each tank has a unique entity ID. The same tank retains its ID even after moving.

**Map Teleport Command** (type=4, cmd_id=0x74):

Teleport to a location via the map view. Fuel cost is distance-dependent.

| Wire Format | Decoded | Description |
|-------------|---------|-------------|
| `! + type=4 + id=116 + X + Y` | Teleport to (X, Y) | 5 bytes total |

- Requires map to be open first (CMD_MAP_OPEN, 'f' key)
- X and Y are destination coordinates (single bytes, 0-255)

Tested teleports:
```
Teleport to 195,79 → payload=[195, 79]
Teleport to 209,90 → payload=[209, 90]
```

**Pickup Move Command** (type=4, cmd_id=0x6a):

Move to pick up fuel or equipment. Triggered by long press on fuel/equipment.

| Wire Format | Decoded | Description |
|-------------|---------|-------------|
| `! + type=4 + id=106 + X + Y` | Pickup at (X, Y) | 5 bytes total |

- Same payload format as regular move
- Tank moves to location and picks up item
- Inventory full prevents pickup

**Ping Command** (type=2, cmd_id=0x2e):

| Key | Action | Response |
|-----|--------|----------|
| F6 | PING | Latency in milliseconds (e.g., "60 ms") |

**Scope/View Commands** (type=3, cmd_id=0x5a):

| Key | Extra Byte | Direction | Description |
|-----|------------|-----------|-------------|
| ArrowUp | 0x00 | North | Pan camera north |
| ArrowRight | 0x02 | East | Pan camera east |
| PageDown | 0x03 | Southeast | Pan camera southeast |
| End | 0x05 | Southwest | Pan camera southwest |
| ArrowLeft | 0x06 | West | Pan camera west |
| Home | 0x07 | Northwest | Pan camera northwest |

Response: `Z` + viewport data with entity positions

**Plain Commands** (no XOR encoding):

| Wire | Key | Action | Description |
|------|-----|--------|-------------|
| `-` | Q | QUIT | Exit game and return to lobby |
| `A1` | A | AUTOSCROLL_ON | Enable autoscroll (JS: `"A" + Number(setting)`, true=1) |
| `A0` | A | AUTOSCROLL_OFF | Disable autoscroll |

**Equipment Toggle Commands** (type=3, cmd_id=0x72):

| Key | Extra Byte | Action | Description |
|-----|------------|--------|-------------|
| 1 | 0x31 | TOGGLE_ARMOR | Toggle armor shields |
| 2 | 0x32 | TOGGLE_DUAL | Toggle dual shots |
| 3 | 0x33 | TOGGLE_MISSILE | Toggle missile shots |
| 4 | 0x34 | TOGGLE_HOMING | Toggle homing shots |
| 5 | 0x35 | TOGGLE_RADAR | Toggle extra radars |

Response format: `t(1) + armor(1) + dual(1) + missile(1) + homing(1) + radar(1)` - each byte 0=off, 1=on

**Sound Toggle Commands** (plain, no XOR):

| Wire | Action | Description |
|------|--------|-------------|
| `V140` | SOUND_ON | Enable sound |
| `V040` | SOUND_OFF | Disable sound |

**Local-Only Commands** (no server message):

| Key | Action | Description |
|-----|--------|-------------|
| H | HELP | Show help overlay (client-side only) |

#### Response Formats

**Statistics (V prefix)** - 15 bytes:
```
V(1) + hours(2 LE) + mins(1) + secs(1) + destroyed(2 LE) + deactivated(2 LE) + pad(5) + promo_pts(1)
Example: 5600000b0b0000000000000000001a = 0h11m11s, 0 destroyed, 0 deactivated, 26 promo pts
```

**Inventory (I prefix)** - 8 bytes:
```
I(1) + version(1) + armor(1) + dual(1) + missile(1) + homing(1) + radar(1) + slot6(1)
Encoding: bit 7 (0x80) = disabled flag, bits 0-6 = count
- 0x94 = 0x80 | 0x14 = 20 items, disabled
- 0x14 = 20 items, enabled
- 0x00 = empty
Example: 49029494949494 = 20 of each, all disabled
```

**Active Forces (* prefix)** - 5 bytes:
```
*(1) + red(1) + purple(1) + blue(1) + orange(1)
Example: 2a0909090a = red=9, purple=9, blue=9, orange=10
```

**Active Players (/ prefix)** - 4+ bytes:
```
/(1) + capacity(1) + count(2 LE) + player_data...
Example: 2f3c0200 = capacity=60, count=2
```

**Nearest Enemy (H prefix)** - 7 bytes:
```
H(1) + x(1) + y(1) + team(1) + player_num(1) + rank(1) + ?(1)
Team: 0=red, 1=purple, 2=blue, 3=orange
Rank: 0x1b=private (others TBD)
Example: 483f8400011b02 = coords=[63,132], red, private
```

**Leaderboard (1 prefix)** - variable:
```
1(1) + team(1) + pad(4) + entries...
Entry: rank(1) + mystery(1) + score(2 LE) + team(1) + 0x08 + namelen(1) + name
Team filter: 0x00=red, 0x01=purple, 0x02=blue, 0x03=orange, 0xff=all
```

**Toggle State (t prefix)** - 6 bytes:
```
t(1) + armor(1) + dual(1) + missile(1) + homing(1) + radar(1)
Each byte: 0=off, 1=on
Example: 740101010101 = all 5 equipment types active
```

**Map Data (L prefix)** - ~673 bytes:
```
L(1) + map_data...
Binary blob containing map tile/entity data
```

**Query Commands** (3 bytes: `!` + type + cmd):

| Base Action | Description | Response Size |
|-------------|-------------|---------------|
| SPAWN | Initialize/respawn | Many STATE updates |
| MAP | Request full map | ~565-580 bytes |
| RADAR | Ping nearby entities | ~14-28 bytes |
| FUEL/EQUIP | Equipment panel | ~6-24 bytes |

**Action Commands** (variable length):

| Prefix Pattern | Action | Data | Description |
|----------------|--------|------|-------------|
| `!` + type + `#`-like | **SHOOT** | 2 bytes | Direction/angle vector |
| `!` + type + `$`-like | **MINE** | 3 bytes | X, Y coordinates |
| `!` + type + `i`-like | **MOVE** | 2 bytes | Direction + velocity |

#### Example Wire Formats

**RADAR request** (session with type=`"`):
```
03 00 21 22 26  →  len=3, body="!\"&"
```

**MINE placement** (session with type=`$`):
```
05 00 21 24 30 0c ab  →  len=5, body="!$" + coords(0x30, 0x0c, 0xab)
```

**SHOOT** (session with type=`#`):
```
04 00 21 23 1a 75  →  len=4, body="!#" + angle(0x1a, 0x75)
```

### Game State Updates

Server sends binary state updates prefixed with `.` containing:
- Player positions
- Projectile data
- Game objects

### Initial State on Join

When joining a game, the server automatically pushes several messages:

1. **Join Confirmation** with spawn coordinates:
   ```
   +2|3|128|128|<encoded_data>...
   ```
   Fields: `+<room_id>|<team>|<x>|<y>|<session_data>`

   The coordinates (128,128) represent your spawn LOCATION displayed in-game.

2. **Inventory State** (pushed without request):
   ```
   I(1) + version(1) + armor(1) + dual(1) + missile(1) + homing(1) + radar(1) + slot6(1)
   ```
   The server sends this automatically - no need to press 'I' key.

3. **Equipment Toggle State** (pushed without request):
   ```
   t(1) + armor(1) + dual(1) + missile(1) + homing(1) + radar(1)
   ```
   Shows which equipment is currently enabled (each byte: 0=off, 1=on).

4. **Viewport Data** (Z prefix):
   ```
   Z + x(1) + y(1) + <entity_data>...
   ```
   Contains visible entities in current viewport.

### Recurring State Messages

These state update subtypes appear continuously during gameplay:

| Subtype | Char | Description |
|---------|------|-------------|
| 0x21 | `!` | Entity position updates |
| 0x2e | `.` | Tank status sync (rank position, fuel) |
| 0x3d | `=` | Position confirmation |
| 0x3f | `?` | Heartbeat/sync |
| 0x46 | `F` | Fuel/energy update |
| 0x4b | `K` | Kill/event notification |
| 0x4f | `O` | Object state |
| 0x5a | `Z` | Viewport entity data |

### HIT Event Message (12 bytes)

Hit events are 12-byte state messages that notify of combat hits:

```
.(1) + 0x6c(1) + data(8) + hit_type(1) + 0x60(1)
```

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 1 | subtype | 0x2e (`.`) |
| 1 | 1 | type | 0x6c (108) |
| 2-8 | 7 | data | Hit event data (coords, damage, etc.) |
| 9 | 1 | hit_type | 1=you were hit, 2=you hit enemy |
| 10 | 1 | unknown | Variable |
| 11 | 1 | footer | 0x60 (96) |

Examples from combat:
```
You hit enemy: 2e6c450b106b4b6f23024760 (byte 9 = 0x02)
Enemy hit you: 2e6c440710684b6c23014760 (byte 9 = 0x01)
```

### Tank Status Sync Message (0x2E subtype 0x03, 13 bytes)

Periodic state update containing leaderboard position and fuel. Different from the 14-byte fuel message below which uses other subtypes.

**Format (13 bytes, about self, subtype 0x03):**
```
.(1) + subtype(1) + tank_id(2 LE) + flags(3) + rank_pos(2 BE) + unknown(2) + fuel(2 LE)
```

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 1 | sig | 0x2E (`.`) |
| 1 | 1 | subtype | 0x03 = about self, 0x01 = about others |
| 2-3 | 2 | tank_id | Tank ID (little-endian) |
| 4-6 | 3 | flags | Status flags (observed: 030200, 020200) |
| 7-8 | 2 | rank_pos | Leaderboard position (big-endian, 1 = top) |
| 9-10 | 2 | unknown | Always 0x0000 in captures (purpose TBD) |
| 11-12 | 2 | fuel | Fuel/HP value (little-endian) |

**Note:** Bytes 7-8 are leaderboard position, NOT promotion points. Promotion points come from the 0x56 Statistics message.

**Format (9 bytes, about others, subtype 0x01):**
```
.(1) + 0x01(1) + tank_id(2 LE) + data(5)
```

### Supervisor Message (0x52, 4+ bytes)

Server supervisor notification. Trigger conditions not fully understood.

**Format:**
```
R(1) + 0x01(1) + 0x00(1) + status(1) [+ text...]
```

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 1 | sig | 0x52 (`R`) |
| 1 | 1 | byte1 | Always 0x01 |
| 2 | 1 | byte2 | Always 0x00 |
| 3 | 1 | status | Status value |
| 4+ | var | text | Optional ASCII text (when status=128) |

**Observed status values:**
- `1` - Post-event state
- `4` - Normal state
- `7` - Seen after equipment gains
- `8` - Seen occasionally
- `128` (0x80) - Contains text message (e.g., "Congratulations!")

**Testing observations:**
- NOT a timer/heartbeat (5 min idle = zero messages)
- NOT consistently triggered by combat, movement, or equipment
- Sometimes appears after equipment gains
- status=128 with "Congratulations!" appeared in one session (context unclear)
- Trigger appears to be server-side state, not player actions

### Tank Fuel Message (0x2E, 14 bytes, XOR-encoded subtypes)

14-byte state message containing player tank fuel (which equals HP). Uses subtypes other than 0x03 (e.g., 0x06, 0x0f, 0x15):

```
.(1) + subtype(1) + data(10) + fuel(2 LE)
```

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 1 | prefix | 0x2e (`.`) |
| 1 | 1 | subtype | Varies per session (XOR encoded, e.g., 0x06, 0x0f, 0x15) |
| 2-11 | 10 | data | Entity/status fields |
| 12-13 | 2 | fuel | XOR-encoded fuel value (u16 little-endian) |

**Fuel Decoding Formula (XOR required!):**

```python
decoded_fuel = (body[12] ^ xor_table[12]) | ((body[13] ^ xor_table[13]) << 8)
```

Where:
- `xor_table[i] = static_key[i] ^ magic[i % len(magic)]`
- Track delta changes between updates to monitor fuel consumption

**Verified Fuel Costs:**

| Action | Fuel Cost |
|--------|-----------|
| Radar (S key) | -10 |
| Movement | -1 per tile |
| Fuel deposit | -100 |
| Teleport | -50 to -500 (distance) |
| Combat damage | -30 to -50 per hit |
| Fuel pickup | +100 to +200 |

**Example timeline (XOR decoded):**
```
Time    Event      decoded   delta
----------------------------------
41.4s   RADAR
42.3s   STATE      151       -10   <- radar cost
54.3s   STATE       17      -100   <- fuel deposit
64.3s   STATE      123      +116   <- fuel pickup
```

**Note:** The subtype byte (body[1]) varies per session because it's also XOR encoded.
Common subtypes observed: 0x13, 0x1b, 0x40, 0x45

### Authentication Flow

1. Browser navigates to `/play`
2. Server redirects to `/before-playing` if not authenticated
3. User logs in (guest or account)
4. Browser connects to `wss://dorothy.tankpit.com/ws/`
5. Client sends AUTH message with session token
6. Server responds with ROOM_LIST
7. Client sends SELECT to join a room
8. Server responds with JOIN_CONFIRM

### Example Messages

**AUTH (sent)** - Guest:
```
%AUTH !be 104832|#guest-8h8y24lsv|<session_token>
```

**AUTH (sent)** - Registered Account:
```
%AUTH !be 62997|52f48ab480e74ccd84e6221e6c0183e0|3674672830 <token>
```
Fields: `%AUTH !be <user_id>|<session_token>|<auth_id> <extra_token>`

**ROOM_LIST (received)**:
```
+4|World (Meltdown)|42|1,1,1,0,1,0,0|2|n|field42.gif|2025
+3|Practice|1|0,0,0,0,0,0,0|1|p|field01.gif|2025
```
Fields: `+<room_id>|<name>|<player_count>|<flags>|<unknown>|<mode>|<background>|<year>`

**SELECT (sent)**:
```
*4
```
Selects room 4 (World Meltdown)

**JOIN_CONFIRM (received)**:
```
=4|Sep. 25, 2012|Yuppler|4|9|9|9|9
```
Fields: `=<room_id>|<join_date>|<tank_name>|<stat1>|<stat2>|<stat3>|<stat4>|<stat5>`

### Rate Limiting

Guest accounts are limited per IP address. After ~10 guest tanks, the server returns:
> "There are too many tanks associated with your IP. Please log in to or register a TankPit account."

Solution: Use a registered account with `TANKPIT_USERNAME` and `TANKPIT_PASSWORD` environment variables.

## Next Steps (Phase 3)

1. ~~Add live decode to sniffer~~ ✓
2. ~~Capture XOR magic key from tankpit.magic~~ ✓
3. ~~Implement protocol encoder/decoder module~~ ✓
   - `protocol/codec.py` - XOR encode/decode with static + session keys
   - `protocol/framing.py` - 2-byte length framing
   - `decoder.py` - Session decoder for captured data
   - `parser.py` - Lobby message parser
   - `protocol/commands.py` - Command type definitions
4. ~~Implement container message decoder~~ ✓
   - `container/` - Length-based 0x2E subtype identification
   - `protocol/types.py` - TypedDict message structures with msg_type literals
   - 13 container subtypes decoded (tank_registry, position_update, combat_hit, deactivation_kill, deactivation_death, etc.)
5. Complete command discovery using live decode mode
6. Build WebSocket client with connection management (`client.py`)
7. Implement bot entry point with game loop
8. Add AI strategy for movement and combat
