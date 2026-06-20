"""Test data constants for container decoder tests.

All test data uses real patterns from captured sessions.
"""

from __future__ import annotations

# =============================================================================
# Combat Hit Test Data
# =============================================================================

# Combat hit: 11 bytes
# From "5953cd07998493ce9c8051" - session capture
# 0x53 ShootEvent fixtures moved to tests/protocol/ -- container path
# was deleted 2026-06-19. Real wire bytes used as protocol-layer test
# data via tankpit_bot.protocol.decode_shoot_event.

# =============================================================================
# Tank Registry Test Data
# =============================================================================

# Tank registry: 16 bytes (minimum)
# From session capture - 16 bytes
TANK_REGISTRY_16 = bytes.fromhex("7c0980530b0f41094aedcf0f326e6576")

# Tank registry: 20 bytes (maximum)
# Extended data pattern for name length variation
TANK_REGISTRY_20 = bytes.fromhex("7c0980530b0f41094aedcf0f326e657600112233")

# Tank registry: bot (17 bytes)
# Structure: [subtype:1][flags:1][tank_id:2 LE][info:13]
# info for bot: [zeros:6][bot_num:1][name:5+null] - bot has first 6 info bytes as zeros
# flags=0x01 (red team), tank_id=0x023A, zeros(6), bot_num=5, name="red-3\0"
TANK_REGISTRY_BOT = bytes.fromhex("7c013a02000000000000057265642d3300")

# Tank registry: container with wasd name (18 bytes)
# flags=0x7e (extended), tank_id=0x1E82, info has x=17,y=9, name="sse"
# Extended format: name at offset 10
TANK_REGISTRY_CONTAINER_WASD = bytes.fromhex("7c7e821e11090200030000007373650000")

# Tank registry: container with short garbage name (16 bytes)
# flags=0x35 (extended), tank_id=0x081D, info has x=3,y=146, name=non-printable
# Extended format: name at offset 10, info[10:12] = 00 82 (non-printable)
TANK_REGISTRY_CONTAINER_GARBAGE = bytes.fromhex("7c351d08039280000000000000000082")

# =============================================================================
# Movement Test Data
# =============================================================================

# Movement messages: 16-20 bytes but ending with waypoint directions (w/s/n/e)
# These should NOT match is_tank_registry_structure because tail4 is all direction chars
# Movement 18 bytes: subtype=0x47('G'), ends with "ennnw" (0x65 0x6e 0x6e 0x6e 0x77)
# From session capture: tank moving east, north, north, north, west
MOVEMENT_18_ENNNW = bytes.fromhex("477e026e5c0c03002e87030000656e6e6e77")

# Movement 19 bytes: subtype=0x47('G'), ends with "wwwwww" (6x 0x77)
# From session capture: tank moving 6 tiles west
MOVEMENT_19_WWWWWW = bytes.fromhex("477e02745c0803002e87030000777777777777")

# Movement 16 bytes: minimal length with 4 direction chars at end "ssss"
# Constructed to test exact boundary - 12 header bytes + 4 waypoints (s=0x73)
MOVEMENT_16_SSSS = bytes.fromhex("470102030405060708091011" + "73737373")

# Movement 20 bytes: maximal TankRegistry length with 4 directions "nesw"
# Constructed to test exact boundary - 16 header bytes + 4 waypoints (n=0x6e,e=0x65,s=0x73,w=0x77)
MOVEMENT_20_NESW = bytes.fromhex("47010203040506070809101112131415" + "6e657377")

# =============================================================================
# Position Update Test Data
# =============================================================================

# Position update: exactly 13 bytes
# From "2453cd0715121d67b315515506" capture
POSITION_UPDATE_13 = bytes.fromhex("2453cd0715121d67b315515506")

# =============================================================================
# Tank Status Test Data
# =============================================================================

# Tank status short: 9 bytes (enemy status with rank/damage)
# Structure: [subtype:1] [tank_id:2 LE] [damage:1] [rank:1] [flag:1] [lb_pos:2 LE] [extra:1]
# Example: tank_id=0x5782, damage=2 (medium), rank=4 (lieutenant), flag=0, lb_pos=0x0015, extra=0
# TANK_STATUS_SHORT_9 deleted 2026-06-19: the container "tank_status_short"
# was wrong on every byte position (crack confirmed 0/74 corpus samples
# produced a valid container rank). 9-byte 0x2E bodies now route to
# Og.h TankStatusSync. See analysis_scripts/crack_tank_status_short.py.

# Tank status sync: 2 bytes
TANK_STATUS_SYNC_2 = bytes.fromhex("0100")

# Tank status sync: 3 bytes
TANK_STATUS_SYNC_3 = bytes.fromhex("030102")

# Tank update compact / full / extended fixtures deleted 2026-06-19.
# After the tunneled-subtype dispatch fix (analysis_scripts/
# crack_tank_update.py), zero production bodies fell through to the
# length-based "tank_update_*" container fallback across 150 sessions,
# so the types, decoders, and these fixtures were removed together.

# Tunneled mine placement: 15 bytes
# From capture after mine command by Artax at (131,126)
# Structure: [0x4B][mine_type:1][tank_id:2 LE][count:1][positions: count*2]
MINE_PLACEMENT_15 = bytes.fromhex("4b02150505837e837d847d847e847f")

# Tunneled mine detonation: solitary impact (3 bytes)
# From visible mine shot at (44,59)
MINE_DETONATION_3 = bytes.fromhex("452c3b")

# Tunneled mine detonation: chain reaction (15 bytes)
# From visible mine cluster shot around (38,53)
MINE_DETONATION_15 = bytes.fromhex("452634273526362535273627342536")

# Tank leave: 6 bytes with tank_id pattern (byte[3] == 0 for tank IDs < 256)
# From capture: "7f138b004213" - Arterial (tank 139) left the game
TANK_LEAVE_6 = bytes.fromhex("7f138b004213")

# Tank leave with large tank ID (> 256): 6 bytes
# From capture: "204a845d5201" - tank 23940 left the game
TANK_LEAVE_LARGE_ID = bytes.fromhex("204a845d5201")

# =============================================================================
# Unknown/Invalid Test Data
# =============================================================================

# Unknown: 8 bytes (doesn't match any pattern - gap between 7 and 9)
UNKNOWN_8_BYTES = bytes.fromhex("7e51460516112233")

# Unknown: 12 bytes (doesn't match any known container pattern)
UNKNOWN_12_BYTES = bytes.fromhex("010203040506070809101112")

# =============================================================================
# Teleport Test Data
# =============================================================================

# Teleport landed: 1 byte (0x0C subtype)
# From capture: single byte confirmation after teleport completes
TELEPORT_LANDED_1 = bytes.fromhex("0c")

# =============================================================================
# Container Pickup Test Data
# =============================================================================

# Container pickup: 5 bytes [subtype:1][x:1][y:1][volume:2 LE]
# Equipment pickup (volume=0)
CONTAINER_PICKUP_EQUIPMENT = bytes.fromhex("43" + "88" + "5e" + "0000")  # x=136, y=94, vol=0
# Fuel pickup (volume=618 = 0x026a)
CONTAINER_PICKUP_FUEL = bytes.fromhex("43" + "89" + "5f" + "6a02")  # x=137, y=95, vol=618

# =============================================================================
# Radar Response Test Data
# =============================================================================

# Radar response: [subtype:1][count:2 LE][entries: count*4]
# Each entry: [x:1][y:1][volume:2 LE] (volume=0xFFFF for equipment)
# 1 equipment container at (123, 105)
RADAR_RESPONSE_1 = bytes.fromhex("4f" + "0100" + "7b69ffff")  # count=1, (123,105):equip
# 2 containers: 1 equipment + 1 fuel
RADAR_RESPONSE_2 = bytes.fromhex("4f" + "0200" + "7b69ffff" + "895fea02")  # count=2
# 5 containers (4 equipment + 1 fuel) - realistic radar response
RADAR_RESPONSE_5 = bytes.fromhex(
    "4f"
    + "0500"  # subtype + count=5
    + "7b69ffff"  # (123,105):equip
    + "7d68ffff"  # (125,104):equip
    + "8469ffff"  # (132,105):equip
    + "885effff"  # (136,94):equip
    + "895fea02"  # (137,95):fuel=746
)

# =============================================================================
# Tip Notification Test Data
# =============================================================================

# Tip notification: 29 bytes (minimum of range 29-79)
# From session capture - game tips and notifications
TIP_NOTIFICATION_29 = bytes.fromhex("68" + "00" * 28)

# Tip notification: 79 bytes (maximum of range 29-79)
TIP_NOTIFICATION_79 = bytes.fromhex("68" + "01" * 78)

# Tip notification: 55 bytes (middle of range)
TIP_NOTIFICATION_55 = bytes.fromhex("68" + "02" * 54)

# =============================================================================
# Chunk Data Test Data
# =============================================================================

# Chunk data: 80 bytes (minimum of range 80-130)
# From session capture - terrain/map chunk data
CHUNK_DATA_80 = bytes.fromhex("14" + "00" * 79)

# Chunk data: 130 bytes (maximum of range 80-130)
CHUNK_DATA_130 = bytes.fromhex("14" + "01" * 129)

# Chunk data: 95 bytes (middle of range - from session summary)
CHUNK_DATA_95 = bytes.fromhex("14" + "02" * 94)

# =============================================================================
# World State Test Data
# =============================================================================

# World state: 500 bytes (minimum)
# From session capture - full world/map state
WORLD_STATE_500 = bytes.fromhex("14" + "00" * 499)

# World state: 650 bytes (common size from session summary)
WORLD_STATE_650 = bytes.fromhex("14" + "01" * 649)

# =============================================================================
# Player List Test Data
# =============================================================================

# Player list short: 4 bytes response to '/' key
# From capture: "79990507" - single player response
PLAYER_LIST_SHORT_4 = bytes.fromhex("79990507")

# Player list extended: 7 bytes response with multiple players
# From capture: "79990507ce1144" - multi-player response
PLAYER_LIST_EXTENDED_7 = bytes.fromhex("79990507ce1144")

# =============================================================================
# Deactivation Test Data
# =============================================================================

# 0x41 Deactivation moved to the protocol-layer test fixture set
# (tests/protocol/test_combat.py) -- container path was deleted
# 2026-06-19 along with the dual-path collapse.

# Deactivation death: 7 bytes [0x43, flags, killer_lo, killer_hi, extra...]
# From capture: "430786160c7f1f" - you were killed by tank 5766
DEACTIVATION_DEATH_7 = bytes.fromhex("430786160c7f1f")
