"""The client command VOCABULARY: what a command is called, not how it is built.

Pure data — the prefix, the tick rate, every command byte, the scope
compass, the plain (un-XORed) literals, and the four type numbers. No
behaviour, so nothing here can fail and nothing imports anything.

Split from a 598-line module 2026-09-03 (the 400-600 rule). The two
halves that left had different jobs and different readers: the typed
command model and its serialization went to
:mod:`tankpit_bot.protocol.command_frames`, the wire builders to
:mod:`tankpit_bot.protocol.command_builders`. The vocabulary stayed
here because it is what every other module actually imports, and
because the wiki claim addresses in [[client-commands]] bind to it.
"""

from __future__ import annotations

# Command prefix byte (this is known from protocol analysis)
COMMAND_PREFIX = ord("!")

# Server tick rate (verified by fire spam testing)
TICK_RATE_MS = 2000  # Commands processed every 2 seconds

# =============================================================================
# Known Command IDs (discovered from protocol analysis)
# =============================================================================

# XOR-encoded commands (type=2, start with '!')
CMD_ENTER_GAME = 63  # 0x3f - Click to enter game
CMD_ACTIVE_FORCES = 42  # 0x2a - 'x' key - Show active forces
CMD_ACTIVE_PLAYERS = 47  # 0x2f - '/' key - Show active players
CMD_RADAR = 102  # 0x66 - 's' key - Toggle radar display
CMD_NEAREST_ENEMY = 104  # 0x68 - 'e' key - Target nearest enemy
CMD_INVENTORY = 105  # 0x69 - 'i' key - Show inventory
CMD_MINE = 107  # 0x6b - 'd' key - Drop mine
CMD_MAP_OPEN = 108  # 0x6c - 'f' key - Open map view
CMD_STATISTICS = 118  # 0x76 - 'c' key - Show statistics

# XOR-encoded commands (type=2, start with '!')
CMD_PING = 46  # 0x2e - 'F6' key - Ping server, returns latency in ms

CMD_KEEPALIVE = 33  # 0x21 - client keep-alive, JS class dc ([[client-commands]])
CMD_UNMODELLED_COMBAT = 68  # 0x44 - observed live, type 6, NO law ([[client-commands]])

# XOR-encoded commands (type=4, start with '!') - Movement
CMD_MOVE = 112  # 0x70 - Mouse click - Tank movement (click to move)
# Movement payload: X (1 byte) + Y (1 byte) = target coordinates

CMD_PICKUP_FUEL = 100  # 0x64 / 'd' - Long press - Get fuel
# Pickup payload: X (1 byte) + Y (1 byte) = target coordinates

CMD_PICKUP_EQUIPMENT = 106  # 0x6a / 'j' - Long press - Get equipment
# Pickup payload: X (1 byte) + Y (1 byte) = target coordinates

CMD_MAP_TELEPORT = 116  # 0x74 - Map click - Teleport via map (fuel cost varies by distance)
# Teleport payload: X (1 byte) + Y (1 byte) = destination coordinates
# Requires map to be open first (CMD_MAP_OPEN)

# Movable concrete blocks (wiki [[movable-blocks]], wire-cracked
# 2026-07-20): ONE command for both pickup and drop — the server
# decides from carry state. Client binds it as "Click and Hold:
# Pick Up / Drop".
CMD_BLOCK = 98  # 0x62 / 'b' - Long press - Pick up / drop a movable block

# XOR-encoded commands (type=6, start with '!') - Combat
CMD_SHOOT = 115  # 0x73 - Spacebar - Fire at target position
# Shoot payload: X (1 byte) + Y (1 byte) + target_id_lo (1 byte) + target_id_hi (1 byte)
# - Bytes 0-1: Target coordinates (where mouse clicked)
# - Bytes 2-3: Target entity ID (little-endian, 0x0000 if no specific target)
# Shot type (regular/dual/missile/homing) determined by enabled equipment state

# XOR-encoded commands (type=3, start with '!')
CMD_TOP10 = 49  # 0x31 - 't/r/p/b/o' keys - Leaderboard (extra byte: ff=all, 00-03=team)
# Leaderboard response: rank(1) + mystery(1) + score(2) + team(1) + 0x08 + namelen(1) + name

CMD_SCOPE = 90  # 0x5a - Arrow/Page keys - Pan camera view
# Direction byte is compass CLOCKWISE FROM NORTH: 0=N 1=NE 2=E 3=SE
# 4=S 5=SW 6=W 7=NW. Wire-measured 2026-08-01 against the 2026-07-10
# human capture (sniff-20260710-202821): all 8 sent Rb frames decode
# as [3,'Z',dir] with 0=N, 1=NE, 2=E, 3=SE, 6=W paired to the game
# log's "Extend view {dir}" lines; 5=SW and 7=NW corroborated by the
# JS key handlers (End/Home). Response: 0x5A with the shifted window
# — the ANCHOR law, not a fixed stride ([[viewport-shift-protocol]]).

CMD_TOGGLE_EQUIPMENT = 114  # 0x72 - '1-5' keys - Toggle equipment
# Extra byte: 0x31='1'=armor, 0x32='2'=dual, 0x33='3'=missile, 0x34='4'=homing, 0x35='5'=radar
# Response: t(1) + armor(1) + dual(1) + missile(1) + homing(1) + radar(1) - each 0=off, 1=on

# Scope direction codes (extra byte for CMD_SCOPE) — the full
# clockwise-from-north compass, wire-measured (see CMD_SCOPE above).
SCOPE_NORTH = 0x00  # ArrowUp; measured (Extend view N -> window top = tank_y-15)
SCOPE_NORTHEAST = 0x01  # measured (Extend view NE -> window = (tank_x, tank_y-15))
SCOPE_EAST = 0x02  # ArrowRight; measured x3 (window left = tank_x)
SCOPE_SOUTHEAST = 0x03  # PageDown; measured x2 (window = tank tile exactly)
SCOPE_SOUTH = 0x04  # clockwise-table completion (unobserved on the wire)
SCOPE_SOUTHWEST = 0x05  # End
SCOPE_WEST = 0x06  # ArrowLeft; measured (window left = tank_x-15)
SCOPE_NORTHWEST = 0x07  # Home
SCOPE_CENTER = 0x08  # recenter on the tank (user-confirmed option 2026-08-01;
# byte inferred — the measured compass occupies 0-7, center is the one
# remaining value of the client's 0-8 direction range)

# Plain commands (no XOR encoding, just length header + body)
PLAIN_QUIT = b"-"  # 'q' key - Quit game and return to lobby
PLAIN_SOUND_ON = b"V140"  # 'l' key - Sound on
PLAIN_SOUND_OFF = b"V040"  # 'l' key - Sound off
PLAIN_AUTOSCROLL_ON = b"A1"  # 'a' key - Autoscroll on (JS: "A" + Number(true))
PLAIN_AUTOSCROLL_OFF = b"A0"  # 'a' key - Autoscroll off (JS: "A" + Number(false))


# =============================================================================

# =============================================================================
# Wire type numbers
# =============================================================================
#
# Based on protocol analysis of captured traffic, sent commands use:
#   Type byte = 0x20 | type_number (NOT XOR encoded)
#   Format: ! + type_byte + cmd_id + [payload]
#
# Type numbers:
#   2 = Query (hotkeys like radar, inventory)
#   3 = UI (scope, leaderboard, equipment toggle)
#   4 = Movement (move, pickup, teleport)
#   6 = Combat (shoot)

TYPE_QUERY = 2  # Query commands (radar, mine, inventory, etc.)
TYPE_UI = 3  # UI commands (scope, leaderboard, equipment toggle)
TYPE_MOVEMENT = 4  # Movement commands (move, pickup, teleport)
TYPE_COMBAT = 6  # Combat commands (shoot)


__all__ = [
    "CMD_ACTIVE_FORCES",
    "CMD_ACTIVE_PLAYERS",
    "CMD_BLOCK",
    "CMD_ENTER_GAME",
    "CMD_INVENTORY",
    "CMD_KEEPALIVE",
    "CMD_MAP_OPEN",
    "CMD_MAP_TELEPORT",
    "CMD_MINE",
    "CMD_MOVE",
    "CMD_NEAREST_ENEMY",
    "CMD_PICKUP_EQUIPMENT",
    "CMD_PICKUP_FUEL",
    "CMD_PING",
    "CMD_RADAR",
    "CMD_SCOPE",
    "CMD_SHOOT",
    "CMD_STATISTICS",
    "CMD_TOGGLE_EQUIPMENT",
    "CMD_TOP10",
    "CMD_UNMODELLED_COMBAT",
    "COMMAND_PREFIX",
    "PLAIN_AUTOSCROLL_OFF",
    "PLAIN_AUTOSCROLL_ON",
    "PLAIN_QUIT",
    "PLAIN_SOUND_OFF",
    "PLAIN_SOUND_ON",
    "SCOPE_CENTER",
    "SCOPE_EAST",
    "SCOPE_NORTH",
    "SCOPE_NORTHEAST",
    "SCOPE_NORTHWEST",
    "SCOPE_SOUTH",
    "SCOPE_SOUTHEAST",
    "SCOPE_SOUTHWEST",
    "SCOPE_WEST",
    "TICK_RATE_MS",
    "TYPE_COMBAT",
    "TYPE_MOVEMENT",
    "TYPE_QUERY",
    "TYPE_UI",
]
