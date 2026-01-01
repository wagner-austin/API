# Tankpit Protocol Documentation

This document describes the WebSocket protocol used by Tankpit.com and the discovery process using the sniffer module.

## Overview

Tankpit uses WebSocket connections for real-time game communication. Since no public protocol documentation exists, we capture and analyze traffic to reverse-engineer the protocol.

## Shared Architecture

Both sniffer and probe inherit from `BrowserSession` (in `browser.py`) which provides:

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
4. Save results to `capture_session.json`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TANKPIT_URL` | `https://tankpit.com` | Target URL |
| `TANKPIT_OUTPUT` | `capture_session.json` | Output file |
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
3. **Analyze Capture**: Review `capture_session.json`
4. **Identify Patterns**: Look for:
   - Authentication handshake
   - Game join/leave messages
   - Movement commands (likely x/y coordinates)
   - Shooting/ability commands
   - State sync from server
5. **Document Findings**: Update this file with discovered message types

## Type Definitions

All protocol types are defined in `src/tankpit_bot/types.py`:

- `CapturedMessage`: Single WebSocket frame
- `CaptureSession`: Complete capture with metadata
- `CDPWebSocketCreatedEvent`: CDP event for new connections
- `CDPWebSocketFrameEvent`: CDP event for frame data

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

| Key | Action | Description |
|-----|--------|-------------|
| Space | Shoot | Fire at mouse position |
| S | Radar | Ping nearby entities |
| D | Mine | Place mine at current position |
| F | Open Map | Toggle full map view |
| E | Nearest Enemy | Target nearest enemy |
| 1 | Armor Shields | Toggle armor shields |
| 2 | Dual Shots | Toggle dual shot mode |
| 3 | Missile Shots | Toggle missile mode |
| 4 | Homing Shots | Toggle homing mode |
| 5 | Extra Radars | Toggle extra radar range |
| Arrow Keys | Scope | Pan camera N/S/E/W |
| I | Inventory | Open inventory |
| C | Statistics | Show game statistics |
| X | Active Forces | Show active forces |
| Q | Quit | Exit current game |

Mouse controls:
- **Single click**: Move to position
- **Double click**: Fire at position

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

| ID (dec) | ID (hex) | Key | Action | Description |
|----------|----------|-----|--------|-------------|
| 63 | 0x3f | (click) | ENTER_GAME | Click to enter game |
| 102 | 0x66 | S | RADAR | Ping nearby entities |
| 107 | 0x6b | D | MINE | Drop mine at position |
| 108 | 0x6c | F | MAP_OPEN | Open full map view |

**Plain Commands** (no XOR encoding):

| Wire | Key | Action | Description |
|------|-----|--------|-------------|
| `-` | Q | QUIT | Exit game and return to lobby |

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
   - `codec.py` - XOR encode/decode with static + session keys
   - `framing.py` - 2-byte length framing
   - `decoder.py` - Session decoder for captured data
   - `parser.py` - Lobby message parser
   - `commands.py` - Command type definitions
4. Complete command discovery using live decode mode
5. Build WebSocket client with connection management (`client.py`)
6. Implement high-level protocol layer (`protocol.py`)
7. Implement bot entry point with game loop
8. Add AI strategy for movement and combat
