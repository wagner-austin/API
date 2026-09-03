---
title: Client Commands
tags: [js-client, protocol, commands]
related:
  - "[[js-source-map]]"
  - "[[v-table-complete]]"
  - "[[decode-coverage]]"
source_paths:
  - "tpclient.js:25"
  - "tpclient.js:6"
fact_checked: "2026-07-27"
confidence: high
verified: 2026-06-19 (every command class traced through JS source)
hubs: [js-client]
---

# Client Commands

Every command the TankPit client can send to the server, extracted from tpclient.js K-subclass hierarchy (binary game commands) and va-subclass hierarchy (connection/settings commands).[^1]

## Binary Game Commands (K subclasses)

These are sent during active gameplay. Each starts with a length byte, then the command character byte, then payload.[^1]

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
| 0x6C | `l` | 2 | Nb | **Open map** (CORRECTED 2026-07-24 — was listed as radar) | `[2, 'l']` |
| 0x66 | `f` | 2 | Mb | **Radar scan** (CORRECTED 2026-07-24 — was listed as map) | `[2, 'f']` |
| 0x69 | `i` | 2 | Yb | **Request inventory**. Answered with a 0x49 in all 4 archived sends (2026-09-03). Unmapped in the sim's decoder until then, so it decoded to `other` and crashed the server. | `[2, 'i']` |
| 0x76 | `v` | 2 | bc | **Request statistics** | `[2, 'v']` |
| 0x2A | `*` | 2 | Zb | **Active forces** | `[2, '*']` |
| 0x2F | `/` | 2 | ac | **Active players** | `[2, '/']` |
| 0x31 | `1` | 3 | $b | **Top 10 request** | `[3, '1', team_filter]` — 255=all, 0-3=team |

**Correction (2026-07-24) — the `l`/`f` rows above were swapped in
the original JS trace** (minified name reuse across scopes is the
likely culprit). The live wire is unambiguous and re-proven every
session: the bot's `CMD_MAP_OPEN = 0x6C ('l')` is followed by 0x4C
MapData on every `map_open → map_data_processed` completion in the
entire archive, and `CMD_RADAR = 0x66 ('f')` is followed by 0x4F
radar results in every collect-cascade scan. RE-TRACE COMPLETE
(2026-07-24): [[js-source-map]] §Default keymap — `Mb.code="f"` /
`Nb.code="l"` in the source, and the June swap came from assuming
the keyboard key equals the wire char (the real keymap sends 'f'
from the S key and 'l' from the F key; KeyL is the sound toggle).
Wire, JS source, and the bot's `protocol/commands.py` comments now
agree three ways.

**Map open/close semantics** (user contract, 2026-07-24, verbatim):
*"the programmatic map open command doesnt close the map. so you can
programmatically open the map, but closing the map usually
necessitates an 'm' key press to close the map. teleporting of
course closes the map as well."* — i.e. the open command is NOT a
toggle; the close paths are the client-side keypress or a teleport.
First surfaced by the 2026-07-24 bot-watch run (the probe opened the
map programmatically and never key-closed it).[^3]

### Misc

| Code | Char | Bytes | Class | Description | Byte Layout |
|------|------|-------|-------|-------------|-------------|
| 0x3F | `?` | 2 | Jb | **Enter game** (`CMD_ENTER_GAME`). 343 archived sends, **every one answered**, self-caused tokens `49 49 5A 3Dself` per send — the join burst. **The burst is an ANSWER, not a push**: the sim emitted it unprompted at connect because our bot never sends this command (it joins through the lobby's `join_room`). Unmapped until 2026-09-03, so it crashed the server. | `[2, '?']` |
| 0x2E | `.` | 2 | Kb | **Ping** (latency check, sent via F6 key) | `[2, '.']` |
| 0x21 | `!` | 2 | dc | **Keep-alive**. Cadence REFINED 2026-09-03 from 11,871 archived sends: p10 1,999 ms, **median 2,006 ms**, p90 30,070 ms — the "every 30s idle" figure is the idle TAIL; the common case is one per 2-second tick. The server never answers it: 9,746 windows wholly silent, and every self-caused token in the other 2,125 belongs to another command whose answer arrived late. **Our bot never sends one**, which is why the sim had no law for it until 2026-09-03 and a real client's first keep-alive crashed the server ([[capture-differ]]). | `[2, '!']` |
| 0x72 | `r` | 3 | cc | **Hotkey action** (equipment toggle) | `[3, 'r', key_code]` — codes 49-53 (ASCII '1'-'5') toggle slots in inventory order: 1 armor, 2 dual, 3 missile, 4 homing, 5 radar (user contract + JS trace 2026-07-24; the server holds the enabled state — a scan with extras disabled consumes nothing) |

### Observed but NOT modelled

| Code | Bytes | Sends | Why not |
|------|-------|------:|---------|
| 0x44 | 6, type 6 | 7 | Payloads differ every send (`06446400aaae`, `06442003ae2f`, `06442601ae31`, `06449001b236`). Type 6 is combat and the shape is shoot-like — coordinates plus a two-byte tail — but seven samples with four distinct payloads support no law. Every one IS answered (tokens `pickup`, `64`, `47self`), so silence would be as wrong as a guess. `CMD_UNMODELLED_COMBAT` names the byte; the sim REFUSES it by name rather than inventing a response.

**Constants defined and never once sent** in 342 archived sessions:
`CMD_PING` (0x2E), `CMD_TOP10` (0x31), `CMD_ACTIVE_FORCES` (0x2A),
`CMD_ACTIVE_PLAYERS` (0x2F), `CMD_NEAREST_ENEMY` (0x68). They are
real client capabilities; nobody in the corpus used them.

### The crash class, and why the archive found it

Three commands a real browser sends decoded to `other`, the one kind
`queue_command` refuses — so each would kill a hosted server on
arrival. All three were found in ONE sweep of the sniff archive
(2026-09-03), not by probing: `runs/sniff/` is real-client traffic,
so "what does a real client send" was already answered on disk.

The shared cause is that **our bot is the only client that does not
send them.** A sim validated against our own bot cannot see a command
our bot never emits, however long it soaks. That is a property of the
corpus, not of the sim.

## Connection/Settings Commands (va subclasses)

These are sent as text strings (not binary), during connection and lobby.[^1]

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
| 0x41 | `A` | Ia | **Autoscroll toggle** | `A{enabled}` — the server ACKS by echoing the two-byte command back **plaintext, un-XORed** (raw `4130`/`4131`; overloaded with the XOR-encoded Deactivation, discriminated PRE-XOR by `try_decode_plaintext_ack`; key probe 2026-07-24, corrected 2026-07-25 after the first decoder read the flag post-XOR-corruption) |
| 0x4F | `O` | Ja | **Overall series** | `O{enabled}` |
| 0x43 | `C` | Ka | **Chat toggle** | `C{enabled}` — the server ACKS by echoing the two-byte command back **plaintext, un-XORed** (raw `4330`/`4331`; overloaded with the XOR-encoded CacheUpdate, discriminated PRE-XOR by `try_decode_plaintext_ack` — this frame CRASHED the first key-probe run and the official client mis-parses it silently; corrected 2026-07-25) |
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

From the toolbar click handler and Cb function:[^2]

| Value | Meaning | Target Check |
|-------|---------|--------------|
| 0 | Attack team (needs teammate in zone) | Team check: any ally in 17×17 |
| 1 | Normal fire | No target check |
| 2 | Fire at player (needs anyone in zone) | Any tank in 17×17 |
| 3 | Send chat message | Various |
| 8 | Fuel search (auto-target nearest fuel) | Uses Db() Manhattan search |
| 9 | Equipment search (auto-target nearest equipment) | Uses Db() Manhattan search |
| 14 | Ferry search (auto-target nearest ferry) | Uses Db() Manhattan search |

## Long-press pickup gesture (decoded 2026-07-27)

The human pickup gesture is a **press-and-hold >300 ms released on the
same tile** (`bb` release handler, tpclient.js): the client checks the
held tile and dispatches the matching action — obstacle → "PICK UP
OBST." (action 7), tile ``cache > 0`` → "GET FUEL" (action 5), ``cache
< 0`` → "GET EQUIPMENT" (action 6), deposit context → "DEPOSIT FUEL"
(action 10). The long-press is UI disambiguation from movement only:
the wire command is the same action code the bot already sends
programmatically — there is no separate long-press command. Probe
answer (2026-07-27, `make larder-probe`): the server DOES honor an
equipment pickup targeting the tank's OWN tile — 3/3 credits while
standing on the container (run `larder-20260727-230858`), so the
2026-06-21 silent sample is superseded; a pickup with every slot at
cap is instead rejected with the 0x52 code-7 "Inventory full" receipt
([[equipment-system]]). Fuel still needs no command at all on/adjacent
to a landing — see [[fuel-system]].

## Shoot Command Details

The shoot command (Lb, code 's') is only sent when:[^2]
1. Target tile has an enemy tank (`h.h !== this.i.h` — different team), OR
2. Target tile has a mine from another team (overlay 255 !== own team), OR
3. Forced fire from double-click

The last 2 bytes (target_id) are set to the tank's `.id` if a tank is at the target tile, or 0 for empty ground fire. This matters because the server uses this for homing shot targeting ([[weapon-selection]], [[shoot-event-format]]).[^2]

## Machine-checked binding to `protocol/commands.py`

Every opcode, direction byte, type byte and plain-text payload above is
bound to its Python constant by the claim block below, and the
`physics_claims` guard stage of `make check` imports each `code`
address and compares the value. The wiki table and the code cannot
drift apart without the gate going red.[^guard]

The binding is **total**: reverse coverage requires every public symbol
of `tankpit_bot.protocol.commands` to be claimed exactly once, so a
constant added to the module without a wiki claim is also a build
failure. Callables and the two TypedDicts carry `law` claims — prose
plus an existence check — because their behaviour is not an int
comparison; the 34 integer opcodes and 5 byte payloads are verified
computationally.[^guard]

What this does and does not prove: it proves the wiki table and the
Python constants agree. It does **not** independently re-derive either
from `tpclient.js` — that trace is [^1], dated 2026-06-19. The binding
catches drift from here on; it does not re-audit the original
reverse-engineering.

```json claims
{
  "claims": [
    {
      "id": "cmd-active-forces",
      "code": "tankpit_bot.protocol.commands:CMD_ACTIVE_FORCES",
      "value": 42,
      "means": "'x' key - show active forces"
    },
    {
      "id": "cmd-active-players",
      "code": "tankpit_bot.protocol.commands:CMD_ACTIVE_PLAYERS",
      "value": 47,
      "means": "'/' key - show active players"
    },
    {
      "id": "cmd-block",
      "code": "tankpit_bot.protocol.commands:CMD_BLOCK",
      "value": 98,
      "means": "'b' long press - pick up / drop a movable block"
    },
    {
      "id": "cmd-enter-game",
      "code": "tankpit_bot.protocol.commands:CMD_ENTER_GAME",
      "value": 63,
      "means": "click to enter the game"
    },
    {
      "id": "cmd-inventory",
      "code": "tankpit_bot.protocol.commands:CMD_INVENTORY",
      "value": 105,
      "means": "'i' key - show inventory"
    },
    {
      "id": "cmd-map-open",
      "code": "tankpit_bot.protocol.commands:CMD_MAP_OPEN",
      "value": 108,
      "means": "'f' key - open the map view"
    },
    {
      "id": "cmd-map-teleport",
      "code": "tankpit_bot.protocol.commands:CMD_MAP_TELEPORT",
      "value": 116,
      "means": "map click - teleport (fuel cost varies by distance)"
    },
    {
      "id": "cmd-mine",
      "code": "tankpit_bot.protocol.commands:CMD_MINE",
      "value": 107,
      "means": "'d' key - drop a mine"
    },
    {
      "id": "cmd-move",
      "code": "tankpit_bot.protocol.commands:CMD_MOVE",
      "value": 112,
      "means": "mouse click - walk the tank to a tile"
    },
    {
      "id": "cmd-nearest-enemy",
      "code": "tankpit_bot.protocol.commands:CMD_NEAREST_ENEMY",
      "value": 104,
      "means": "'e' key - target the nearest enemy"
    },
    {
      "id": "cmd-pickup-equipment",
      "code": "tankpit_bot.protocol.commands:CMD_PICKUP_EQUIPMENT",
      "value": 106,
      "means": "'j' long press - pick up equipment"
    },
    {
      "id": "cmd-pickup-fuel",
      "code": "tankpit_bot.protocol.commands:CMD_PICKUP_FUEL",
      "value": 100,
      "means": "'d' long press - pick up fuel"
    },
    {
      "id": "cmd-ping",
      "code": "tankpit_bot.protocol.commands:CMD_PING",
      "value": 46,
      "means": "'F6' key - ping the server, returns latency in ms"
    },
    {
      "id": "cmd-radar",
      "code": "tankpit_bot.protocol.commands:CMD_RADAR",
      "value": 102,
      "means": "'s' key - toggle the radar display"
    },
    {
      "id": "cmd-scope",
      "code": "tankpit_bot.protocol.commands:CMD_SCOPE",
      "value": 90,
      "means": "arrow / page keys - pan the camera view"
    },
    {
      "id": "cmd-shoot",
      "code": "tankpit_bot.protocol.commands:CMD_SHOOT",
      "value": 115,
      "means": "spacebar - fire at a target position"
    },
    {
      "id": "cmd-unmodelled-combat",
      "code": "tankpit_bot.protocol.commands:CMD_UNMODELLED_COMBAT",
      "value": 68,
      "means": "0x44 -- observed live and deliberately NOT modelled. Seven archived sends, type 6 (combat), four DISTINCT payloads (06446400aaae, 06442003ae2f, 06442601ae31, 06449001b236) whose shape is shoot-like: coordinates plus a two-byte tail. Every send IS answered (self-caused tokens pickup, 64, 47self), so silence would be as wrong as a guess, and seven samples across four payloads support no law. The constant exists so the sim can REFUSE the byte by name instead of by build phase; modelling it needs either more archive or a live probe."
    },
    {
      "id": "cmd-keepalive",
      "code": "tankpit_bot.protocol.commands:CMD_KEEPALIVE",
      "value": 33,
      "means": "the client keep-alive, JS class dc. Cadence measured 2026-09-03 over 11,871 archived sends: p10 1,999 ms, median 2,006 ms, p90 30,070 ms -- the table row's 'every 30s idle' is the idle TAIL, the common case is one per 2-second tick. The server NEVER answers it: 9,746 windows wholly silent, and every self-caused token in the other 2,125 belongs to another command whose answer arrived late. Our bot never sends one, so the sim had no law for it and a real client's first keep-alive raised SimError out of queue_command -- a hosted server dying seconds after a browser connected."
    },
    {
      "id": "cmd-statistics",
      "code": "tankpit_bot.protocol.commands:CMD_STATISTICS",
      "value": 118,
      "means": "'c' key - show statistics; COSTS A TICK (operator, 2026-08-31), so it belongs at the session boundary, never on a poll -- the executor dispatches one command per tick, so a press displaces the shot or teleport that tick would have spent. The reply (0x56, decode_statistics) carries destroyed/deactivated/promo_points, which are CUMULATIVE totals rather than a rate, so polling buys nothing a single exit press does not."
    },
    {
      "id": "cmd-toggle-equipment",
      "code": "tankpit_bot.protocol.commands:CMD_TOGGLE_EQUIPMENT",
      "value": 114,
      "means": "'1'-'5' keys - toggle an equipment slot"
    },
    {
      "id": "cmd-top10",
      "code": "tankpit_bot.protocol.commands:CMD_TOP10",
      "value": 49,
      "means": "leaderboard; extra byte ff=all, 00-03=team"
    },
    {
      "id": "command-prefix",
      "code": "tankpit_bot.protocol.commands:COMMAND_PREFIX",
      "value": 33,
      "means": "every binary command frame opens with the '!' prefix byte"
    },
    {
      "id": "plain-autoscroll-off",
      "code": "tankpit_bot.protocol.commands:PLAIN_AUTOSCROLL_OFF",
      "bytes": "A0",
      "means": "'a' key - autoscroll off (JS emits \"A\" + Number(false))"
    },
    {
      "id": "plain-autoscroll-on",
      "code": "tankpit_bot.protocol.commands:PLAIN_AUTOSCROLL_ON",
      "bytes": "A1",
      "means": "'a' key - autoscroll on (JS emits \"A\" + Number(true))"
    },
    {
      "id": "plain-quit",
      "code": "tankpit_bot.protocol.commands:PLAIN_QUIT",
      "bytes": "-",
      "means": "'q' key - quit the game and return to the lobby"
    },
    {
      "id": "plain-sound-off",
      "code": "tankpit_bot.protocol.commands:PLAIN_SOUND_OFF",
      "bytes": "V040",
      "means": "'l' key - sound off"
    },
    {
      "id": "plain-sound-on",
      "code": "tankpit_bot.protocol.commands:PLAIN_SOUND_ON",
      "bytes": "V140",
      "means": "'l' key - sound on"
    },
    {
      "id": "scope-center",
      "code": "tankpit_bot.protocol.commands:SCOPE_CENTER",
      "value": 8,
      "means": "recenter the window on the tank (user-confirmed 2026-08-01)"
    },
    {
      "id": "scope-east",
      "code": "tankpit_bot.protocol.commands:SCOPE_EAST",
      "value": 2,
      "means": "ArrowRight; measured three times - window left = tank_x"
    },
    {
      "id": "scope-north",
      "code": "tankpit_bot.protocol.commands:SCOPE_NORTH",
      "value": 0,
      "means": "ArrowUp; measured - extend view N puts window top at tank_y-15"
    },
    {
      "id": "scope-northeast",
      "code": "tankpit_bot.protocol.commands:SCOPE_NORTHEAST",
      "value": 1,
      "means": "measured - extend view NE puts the window at (tank_x, tank_y-15)"
    },
    {
      "id": "scope-northwest",
      "code": "tankpit_bot.protocol.commands:SCOPE_NORTHWEST",
      "value": 7,
      "means": "Home key; clockwise-table completion"
    },
    {
      "id": "scope-south",
      "code": "tankpit_bot.protocol.commands:SCOPE_SOUTH",
      "value": 4,
      "means": "clockwise-table completion; UNOBSERVED on the wire"
    },
    {
      "id": "scope-southeast",
      "code": "tankpit_bot.protocol.commands:SCOPE_SOUTHEAST",
      "value": 3,
      "means": "PageDown; measured twice - window sits on the tank tile exactly"
    },
    {
      "id": "scope-southwest",
      "code": "tankpit_bot.protocol.commands:SCOPE_SOUTHWEST",
      "value": 5,
      "means": "End key; clockwise-table completion"
    },
    {
      "id": "scope-west",
      "code": "tankpit_bot.protocol.commands:SCOPE_WEST",
      "value": 6,
      "means": "ArrowLeft; measured - window left = tank_x-15"
    },
    {
      "id": "tick-rate-ms",
      "code": "tankpit_bot.protocol.commands:TICK_RATE_MS",
      "value": 2000,
      "means": "the server processes queued commands on a fixed tick (verified by fire-spam testing)"
    },
    {
      "id": "type-combat",
      "code": "tankpit_bot.protocol.commands:TYPE_COMBAT",
      "value": 6,
      "means": "type byte for combat commands (shoot)"
    },
    {
      "id": "type-movement",
      "code": "tankpit_bot.protocol.commands:TYPE_MOVEMENT",
      "value": 4,
      "means": "type byte for movement commands (move, pickup, teleport)"
    },
    {
      "id": "type-query",
      "code": "tankpit_bot.protocol.commands:TYPE_QUERY",
      "value": 2,
      "means": "type byte for query commands (radar, mine, inventory)"
    },
    {
      "id": "type-ui",
      "code": "tankpit_bot.protocol.commands:TYPE_UI",
      "value": 3,
      "means": "type byte for UI commands (scope, leaderboard, equipment toggle)"
    },
    {
      "id": "actioncommand",
      "code": "tankpit_bot.protocol.command_frames:ActionCommand",
      "law": "Variable-length action frame: '!' + type byte + cmd byte + data."
    },
    {
      "id": "querycommand",
      "code": "tankpit_bot.protocol.command_frames:QueryCommand",
      "law": "Three-byte query frame: '!' + type byte + cmd byte, no payload."
    },
    {
      "id": "build-block-command",
      "code": "tankpit_bot.protocol.command_builders:build_block_command",
      "law": "Length-prefixed block pick-up / drop frame for a cardinally adjacent tile."
    },
    {
      "id": "build-move-command",
      "code": "tankpit_bot.protocol.command_builders:build_move_command",
      "law": "Length-prefixed MOVE frame for a destination tile."
    },
    {
      "id": "build-pickup-equipment-command",
      "code": "tankpit_bot.protocol.command_builders:build_pickup_equipment_command",
      "law": "Length-prefixed equipment-pickup frame for a container tile."
    },
    {
      "id": "build-pickup-fuel-command",
      "code": "tankpit_bot.protocol.command_builders:build_pickup_fuel_command",
      "law": "Length-prefixed fuel-pickup frame for a container tile."
    },
    {
      "id": "build-query-command",
      "code": "tankpit_bot.protocol.command_builders:build_query_command",
      "law": "Length-prefixed payload-free query frame for any query cmd id."
    },
    {
      "id": "build-quit-command",
      "code": "tankpit_bot.protocol.command_builders:build_quit_command",
      "law": "Length-prefixed graceful-quit frame."
    },
    {
      "id": "build-scope-command",
      "code": "tankpit_bot.protocol.command_builders:build_scope_command",
      "law": "Length-prefixed SCOPE frame carrying one of the nine direction bytes."
    },
    {
      "id": "build-shoot-command",
      "code": "tankpit_bot.protocol.command_builders:build_shoot_command",
      "law": "Length-prefixed SHOOT frame carrying tile and target id (0 when the tile holds no tank)."
    },
    {
      "id": "build-teleport-command",
      "code": "tankpit_bot.protocol.command_builders:build_teleport_command",
      "law": "Length-prefixed MAP_TELEPORT frame for a map-click destination."
    },
    {
      "id": "build-toggle-equipment-command",
      "code": "tankpit_bot.protocol.command_builders:build_toggle_equipment_command",
      "law": "Length-prefixed frame toggling one equipment slot."
    },
    {
      "id": "decode-action-command",
      "code": "tankpit_bot.protocol.command_frames:decode_action_command",
      "law": "Rebuild an ActionCommand from a dict, validating every field."
    },
    {
      "id": "decode-query-command",
      "code": "tankpit_bot.protocol.command_frames:decode_query_command",
      "law": "Rebuild a QueryCommand from a dict, validating every field."
    },
    {
      "id": "deserialize-command",
      "code": "tankpit_bot.protocol.command_frames:deserialize_command",
      "law": "Parse wire bytes back into a command AFTER XOR decoding, dispatching on the type byte."
    },
    {
      "id": "encode-action-command",
      "code": "tankpit_bot.protocol.command_frames:encode_action_command",
      "law": "Project an ActionCommand to a JSON-serializable dict."
    },
    {
      "id": "encode-query-command",
      "code": "tankpit_bot.protocol.command_frames:encode_query_command",
      "law": "Project a QueryCommand to a JSON-serializable dict."
    },
    {
      "id": "make-action-command",
      "code": "tankpit_bot.protocol.command_frames:make_action_command",
      "law": "Construct an ActionCommand from a cmd id and its payload bytes."
    },
    {
      "id": "make-query-command",
      "code": "tankpit_bot.protocol.command_frames:make_query_command",
      "law": "Construct a payload-free QueryCommand from a cmd id."
    },
    {
      "id": "serialize-action-command",
      "code": "tankpit_bot.protocol.command_frames:serialize_action_command",
      "law": "Render an ActionCommand to wire bytes BEFORE XOR encoding."
    },
    {
      "id": "serialize-query-command",
      "code": "tankpit_bot.protocol.command_frames:serialize_query_command",
      "law": "Render a QueryCommand to wire bytes BEFORE XOR encoding."
    }
  ]
}
```

[^1]: tpclient.js lines 25-31 (K subclasses) and 6-10 (va subclasses) — every command class traced 2026-06-19; file pinned via `source_paths` line anchors
[^2]: tpclient.js Lb class + Cb dispatch + toolbar click handler — same 2026-06-19 trace; target_id semantics wire-confirmed via the id-targeted reroute law ([[shoot-event-format]])
[^3]: user (Austin), 2026-07-24 — quoted verbatim above; corroborating captures on disk: `bot_watch_probe.capture_session.json` and companions at repo root (the three designed watch runs), recorded in the wiki-log entries of 2026-07-24 ("bot-watch run 1" through "push-on-activity stream discovered")
[^guard]: `scripts/physics_claims.py:305` — `run_physics_claim_rules`, the `physics_claims` guard stage wired into `scripts/guard.py:16`; it imports each claim's `code` address and compares the value. Reverse coverage is `_reverse_coverage_violations` at `:281`, called at `:331`, which is what makes an unclaimed public symbol a build failure rather than a silent gap. Verified present 2026-08-07.
