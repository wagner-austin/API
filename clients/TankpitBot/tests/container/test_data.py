"""Test data constants for container decoder tests.

All test data uses real patterns from captured sessions.
"""

from __future__ import annotations

# =============================================================================
# Combat Hit Test Data
# =============================================================================
# 0x53 ShootEvent fixtures moved to tests/protocol/ -- container path
# was deleted 2026-06-19. Real wire bytes used as protocol-layer test
# data via tankpit_bot.protocol.decode_shoot_event.


# =============================================================================
# Tank Registry / TankLeave / PositionUpdate / PlayerList / DeactivationDeath
# =============================================================================
# All container fixtures for the above types were deleted 2026-06-20
# along with their decoders, after a corpus sweep of 150 sessions /
# 48,304 0x2E bodies confirmed zero production fires. The unified
# `decode_0x2e_message` dispatches the corresponding protocol subtypes
# (0x21 TankInfo, 0x3D MovementResponse, etc.) via subtype-first
# matching.


# =============================================================================
# Tank Status Test Data
# =============================================================================

# Tank status sync: 2 bytes
TANK_STATUS_SYNC_2 = bytes.fromhex("0100")

# Tank status sync: 3 bytes
TANK_STATUS_SYNC_3 = bytes.fromhex("030102")


# Tank update compact / full / extended fixtures deleted 2026-06-19.
# After the tunneled-subtype dispatch fix, zero production bodies fell
# through to the length-based "tank_update_*" container fallback
# across 150 sessions, so the types, decoders, and these fixtures
# were removed together.


# Tunneled mine placement: 15 bytes (count=5 positions)
# From capture after mine command by Artax at (131,126)
# Structure: [0x4B][mine_type:1][tank_id:2 LE][count:1][positions: count*2]
MINE_PLACEMENT_15 = bytes.fromhex("4b02150505837e837d847d847e847f")

# Tunneled mine placement: 19 bytes (count=7 positions)
# Real-combat capture 2026-06-20 15:02:56,
# practice-vs-real-20260620-150138.capture_session.json:
# Artax placed a 7-position mine cluster around (133,124). Prior decoder
# required exactly 15 bytes and silently dropped this body into
# UnknownContainer, which is how the bug was first detected.
MINE_PLACEMENT_19 = bytes.fromhex("4b02150507857c847c857b867b867c857d847d")

# Tunneled mine detonation: solitary impact (3 bytes)
# From visible mine shot at (44,59)
MINE_DETONATION_3 = bytes.fromhex("452c3b")

# Tunneled mine detonation: chain reaction (15 bytes)
# From visible mine cluster shot around (38,53)
MINE_DETONATION_15 = bytes.fromhex("452634273526362535273627342536")


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

# Container pickup: 5 bytes [subtype:1=0x43][x:1][y:1][remaining_volume:2 LE]
# remaining_volume is the fuel LEFT IN the container after pickup, not
# the fuel transferred to the picker (verified 2026-06-20 against an
# annotated multi-pickup capture). 0 = container emptied (equipment OR
# fuel fully consumed); >0 = partial fuel pickup that left the rest
# behind because the picker's tank was near the 1100 cap.
CONTAINER_PICKUP_EQUIPMENT = bytes.fromhex("43" + "88" + "5e" + "0000")  # x=136, y=94, remaining=0
CONTAINER_PICKUP_FUEL = bytes.fromhex("43" + "89" + "5f" + "6a02")  # x=137, y=95, remaining=618


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
# TipNotification / ChunkData / WorldState / Deactivation fixtures all
# deleted: zero production traffic across the 150-session corpus after
# the relevant protocol decoders were tunneled correctly inside 0x2E.
# =============================================================================
