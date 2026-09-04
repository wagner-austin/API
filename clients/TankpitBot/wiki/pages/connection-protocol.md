---
title: Connection Protocol
tags: [js-client, protocol, connection]
related:
  - "[[js-source-map]]"
  - "[[xor-cipher]]"
  - "[[client-commands]]"
source_paths:
  - "tpclient.js:214"
  - "tpclient.js:11"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (full connection flow traced through JS)
hubs: [js-client]
---

# Connection Protocol

The complete connection handshake from page load to active gameplay, extracted from the bootstrap IIFE (lines 214-224) and WebSocket transport (lines 11-14).[^1]

## Connection URL

```javascript
var fb = "https:" === window.location.protocol;
var hb = tankpit.connect_server || "dorothy.tankpit.com";
var ib = (fb ? "wss://" : "ws://") + hb +
         ("undefined" === typeof tankpit.connect_url ? "/ws/" : tankpit.connect_url);
```

Default: `wss://dorothy.tankpit.com/ws/` (HTTPS) or `ws://dorothy.tankpit.com/ws/` (HTTP).[^1]

The `tankpit` global object is injected server-side in the HTML and contains:[^1]
- `connect_server` — WebSocket host
- `connect_url` — WebSocket path
- `user_id` — authenticated user ID
- `magic` — session XOR key
- `images_dir` — sprite image path
- `sound` — sound enabled
- `volume` — volume level
- `chat` — chat enabled
- `autoscroll` — autoscroll setting
- `force_overall_series` — leaderboard preference
- `game_scale` — zoom level (0=auto)
- `hotkey_map` — custom key bindings
- `map_colors` — custom map colors
- `sprites` — custom sprite theme
- `series_id` — current leaderboard year
- `playback` / `playback_url` — replay mode

## WebSocket Framing

All messages (both directions) use a 2-byte LE length prefix:[^1]

```
[length_lo, length_hi, ...payload]
```

The `Va` handler (line 12) splits received blobs:[^1]
```javascript
function Va(a) {
  var b = new Uint8Array(a.result);
  for (a = []; 0 < b.length;) {
    var c = (b[0] & 255) + 256 * (b[1] & 255);  // LE u16 length
    b = b.subarray(2);
    if (c <= b.length)
      a.push(b.subarray(0, c)),
      b = b.subarray(c);
    else break;  // incomplete message → error
  }
  for (b = 0; b < a.length; b++)
    a[b].length && this.m(a[b]);  // dispatch each message
}
```

Multiple messages can arrive in a single WebSocket frame (batched by server).[^1]

## Handshake Sequence

### Phase 1: Connect

```
Client: Opens WebSocket to wss://dorothy.tankpit.com/ws/
Server: Sends connection acknowledgment
```

On successful WebSocket open, the `Pa` function fires → sets status=1 → calls `this.l()` (onConnect callback).[^1]

### Phase 2: Authenticate

```
Client → Server: %AUTH !be {user_id}|{fingerprint} {magic}
```

The AUTH message (wa class) is sent as plain text (no XOR):[^1]
- Version: `"be"` (hardcoded)
- user_id: from `tankpit.user_id`
- fingerprint: MurmurHash3 of browser properties (sb class)
- magic: from `tankpit.magic`

If there's a pending error report (Ng), it's sent immediately after AUTH.[^1]

### Phase 3: Game List

Server responds with game list messages:[^1]

```
Server → Client: code 43 (+) for each available game
  Format: +{id}|{name}|{field}|{flags}|{team}|{mode}|{image}|{year}
  Parsed into Jf(id, name, field_id, flags, team, mode, year) objects
```

Game modes (parsed from the mode string):[^1]
- `"e"` or `"t"` → mode 5 (tournament)
- `"p"` → mode 6 (practice)
- `"n"` → mode 7 (normal)

Games are added to the lobby select list. Server can also:[^1]
- code 47 (`/`): remove a game from the list
- code 37 (`%`): rename a game

### Phase 4: Select Game

```
Client → Server: *{game_id}
```

Sent when user clicks a game in the lobby. Plain text, no XOR.[^1]

### Phase 5: Game Info

```
Server → Client: code 61 (=)
  Format: ={game_id}|{start_date}|{name}|{rank}|{orange_count}|{purple_count}|{blue_count}|{red_count}|{max_rank}
```

Updates the lobby statistics panel with game details.[^1]

### Phase 6: Join Game

```
Client → Server: +{game_id}|{team}|{x}|{y}|{encoded_urls}
```

- team: 0=orange, 1=purple, 2=blue, 3=red (from troop picker click)
- x, y: click position on field preview image (999,999 if troop button clicked directly)
- encoded_urls: pipe-delimited list of current page URL + script URLs + "j2lk" marker (anti-cheat fingerprint, max 255 chars)

### Phase 7: Join Confirmation

```
Server → Client: code 36 ($)
  Format: ${game_id}|{??? }|{team}
```

On receiving this:[^1]
1. Set `Fa = true` (in-game flag)
2. Create `$d` game session instance
3. If immediate start (`l` flag): start after 0ms, else wait 2000ms
4. Begin asset loading (sprites, map image, audio)

### Phase 8: Game Start

After assets load:[^1]
1. Send heartbeat: `?` (Jb command)
2. Wait for server messages in queue
3. Call `start()` → set state to 0, begin tick loop

### Phase 9: Active Gameplay

All binary messages now flow through:[^1]
- **Inbound**: code 46 (0x2E container) → XOR decode → V table dispatch
- **Outbound**: K subclass → `.h()` serialize → `za()` XOR encode → `Aa` wrapper → `Xa()` send

### Phase 10: Leave Game

```
Server → Client: code 45 (-)
  Triggers: Fa=false, session cleanup, return to lobby
```

Or client-initiated:[^1]
```
Client → Server: - (Ba command, plain text)
```

## Error Handling

- **Connection lost**: code 126 (`~`) or WebSocket close → show reconnect dialog
- **Connection error**: WebSocket error → show "Try Again" dialog
- **Game error**: Ca (code `&`) or Ea (code `^`) → send error report to server with XOR-encoded stack trace

Error report format (Ca class):[^1]
```
&{length}{69}{error_code}{xor_encoded_error_string_with_stack}
```

## Reconnection

On disconnect:[^1]
1. Show "Connection lost" message
2. User clicks "Reconnect" button
3. New WebSocket created after 1000ms delay
4. Full handshake repeats (AUTH → game list → ...)
5. Previous game state is lost (no session resume)

## Timing Notes

- WebSocket creation: 1000ms delay after trigger (line 214)
- Asset loading timeout: 50 retries × 200ms = 10 seconds max
- Join delay: 2000ms after server confirms (splash screen), or 1000ms for fast join
- Keep-alive interval: 30,000ms (dc command)


[^1]: JS truth: `tpclient.js` on disk (frontmatter-pinned lines 214/11) — bootstrap IIFE (lines 214-224), transport `Pa`/`ab`/`Va` (lines 11-14), AUTH `wa` (line 6), quoted verbatim in the fences above; full flow traced 2026-06-19 (frontmatter `verified:` field). Standing receipt: the bot performs this exact handshake at the start of every run in `runs/` — a wrong sequence would fail login.
