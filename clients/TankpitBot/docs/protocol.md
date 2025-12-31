# Tankpit Protocol Documentation

This document describes the WebSocket protocol used by Tankpit.com and the discovery process using the sniffer module.

## Overview

Tankpit uses WebSocket connections for real-time game communication. Since no public protocol documentation exists, we capture and analyze traffic to reverse-engineer the protocol.

## Sniffer Architecture

The sniffer uses Playwright with Chrome DevTools Protocol (CDP) to intercept WebSocket traffic:

```
┌─────────────────┐
│  Playwright     │
│  sync_api       │
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
- **Header**: 2 bytes (message type indicator)
- **Body**: Pipe-delimited (`|`) fields

### Message Types

| Direction | Header (hex) | Type | Description |
|-----------|--------------|------|-------------|
| sent | `50 00` | AUTH | Authentication with session token |
| received | `39 00` | ROOM_LIST | Room/world information |
| sent | `02 00` | SELECT | Room selection |
| received | `22 00` | JOIN_CONFIRM | Room join confirmation |

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

## Next Steps

1. Create TypedDicts for each discovered message type
2. Implement WebSocket client that speaks the protocol
3. Add game logic and AI strategy
