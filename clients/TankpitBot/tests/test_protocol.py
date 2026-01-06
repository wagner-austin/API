"""Tests for protocol module - XOR-decoded message parsing.

All tests use realistic data patterns.
No mocks, no weak assertions, 100% coverage target.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    MSG_ACTION_DONE,
    MSG_ACTIVE_FORCES,
    MSG_CHAT,
    MSG_CONTAINER,
    MSG_DEACTIVATE,
    MSG_ENEMY_DETECT,
    MSG_EQUIP_GAIN,
    MSG_EQUIP_TOGGLE,
    MSG_FUEL_DEPOSIT,
    MSG_FUEL_GAIN,
    MSG_INVENTORY,
    MSG_MINE_DETONATE,
    MSG_MINE_PLACE,
    MSG_MOVE_RESPONSE,
    MSG_MOVEMENT,
    MSG_PROMOTION,
    MSG_RADAR_RESULT,
    MSG_SHOOT,
    MSG_STATISTICS,
    MSG_SUPERVISOR,
    MSG_SYNC,
    MSG_TANK_ENTRY,
    MSG_TANK_EXIT,
    MSG_TANK_INFO,
    MSG_TANK_POS,
    MSG_TANK_STATS,
    MSG_TANK_STATUS_FULL,
    MSG_TERRAIN_UPDATE,
    MSG_TILE_UPDATE,
    MSG_VIEWPORT,
    RANK_FUEL,
    SUPERVISOR_STATUS_PROMO_ELIGIBLE,
    SUPERVISOR_STATUS_PROMO_KILL,
    TEXT_MSG_TYPES,
    DecodeError,
    Equipment,
    Rank,
    SupervisorDict,
    Team,
    TerrainType,
    ViewportEntityDict,
    _require_exact_length,
    _require_min_length,
    _require_parts,
    _require_prefix,
    _x16,
    _x24,
    decode_0x2e_message,
    decode_action_done,
    decode_active_forces,
    decode_chat_message,
    decode_container,
    decode_deactivation,
    decode_enemy_detection,
    decode_equipment_gain,
    decode_equipment_toggle,
    decode_fuel_deposit,
    decode_fuel_gain,
    decode_hit_confirmation,
    decode_inventory,
    decode_join_confirm,
    decode_message,
    decode_mine_detonation,
    decode_mine_placement,
    decode_movement,
    decode_movement_response,
    decode_radar_result,
    decode_radar_scan_result,
    decode_shoot_event,
    decode_statistics,
    decode_supervisor,
    decode_sync,
    decode_tank_entry,
    decode_tank_exit,
    decode_tank_info,
    decode_tank_status,
    decode_tank_status_sync,
    decode_terrain_update,
    decode_text_message,
    decode_viewport_update,
    decode_world_info,
    is_text_message,
    supervisor_has_promo_kill,
    supervisor_is_promo_eligible,
    try_decode_binary_message,
    try_decode_message,
    viewport_entity_is_container,
    viewport_entity_is_empty,
    viewport_entity_is_tank,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestRankEnum:
    """Tests for Rank enumeration."""

    def test_rank_values(self) -> None:
        """Rank values are 0-7."""
        assert int(Rank.RECRUIT) == 0
        assert int(Rank.PRIVATE) == 1
        assert int(Rank.CORPORAL) == 2
        assert int(Rank.SERGEANT) == 3
        assert int(Rank.LIEUTENANT) == 4
        assert int(Rank.CAPTAIN) == 5
        assert int(Rank.MAJOR) == 6
        assert int(Rank.GENERAL) == 7

    def test_rank_fuel_mapping(self) -> None:
        """RANK_FUEL maps ranks to starting fuel."""
        assert RANK_FUEL[Rank.RECRUIT] == 1000
        assert RANK_FUEL[Rank.PRIVATE] == 1100
        assert RANK_FUEL[Rank.CORPORAL] == 1200
        assert RANK_FUEL[Rank.SERGEANT] == 1300
        assert RANK_FUEL[Rank.LIEUTENANT] == 1400
        assert RANK_FUEL[Rank.CAPTAIN] == 1500
        assert RANK_FUEL[Rank.MAJOR] == 1600
        assert RANK_FUEL[Rank.GENERAL] == 1700


class TestTeamEnum:
    """Tests for Team enumeration."""

    def test_team_values(self) -> None:
        """Team values are 0-3."""
        assert int(Team.RED) == 0
        assert int(Team.PURPLE) == 1
        assert int(Team.BLUE) == 2
        assert int(Team.ORANGE) == 3


class TestEquipmentEnum:
    """Tests for Equipment enumeration."""

    def test_equipment_values(self) -> None:
        """Equipment values are 0-4."""
        assert int(Equipment.ARMOR_SHIELD) == 0
        assert int(Equipment.DUAL_SHOT) == 1
        assert int(Equipment.MISSILE_SHOT) == 2
        assert int(Equipment.HOMING_SHOT) == 3
        assert int(Equipment.EXTRA_RADAR) == 4


class TestTerrainTypeEnum:
    """Tests for TerrainType enumeration."""

    def test_terrain_values(self) -> None:
        """TerrainType values match protocol."""
        assert int(TerrainType.GROUND) == 0
        assert int(TerrainType.ROCK_A) == 1
        assert int(TerrainType.ROCK_B) == 2
        assert int(TerrainType.ROCK_AB) == 3
        assert int(TerrainType.FERRY) == 5
        assert int(TerrainType.FERRY_ROCK) == 7


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestX16:
    """Tests for _x16 helper function."""

    def test_combines_bytes_little_endian(self) -> None:
        """Combines two bytes into uint16 little-endian."""
        assert _x16(0x34, 0x12) == 0x1234
        assert _x16(0x00, 0x00) == 0x0000
        assert _x16(0xFF, 0xFF) == 0xFFFF
        assert _x16(0x01, 0x00) == 0x0001
        assert _x16(0x00, 0x01) == 0x0100

    def test_masks_to_byte_range(self) -> None:
        """Masks input values to byte range."""
        assert _x16(0x134, 0x112) == _x16(0x34, 0x12)


class TestX24:
    """Tests for _x24 helper function."""

    def test_combines_bytes_big_endian(self) -> None:
        """Combines three bytes into uint24 big-endian."""
        assert _x24(0x12, 0x34, 0x56) == 0x123456
        assert _x24(0x00, 0x00, 0x00) == 0x000000
        assert _x24(0xFF, 0xFF, 0xFF) == 0xFFFFFF
        assert _x24(0x01, 0x00, 0x00) == 0x010000
        assert _x24(0x00, 0x01, 0x00) == 0x000100


# =============================================================================
# Validation Function Tests
# =============================================================================


class TestRequireMinLength:
    """Tests for _require_min_length validation."""

    def test_passes_when_sufficient(self) -> None:
        """Validation passes when length is sufficient."""
        _require_min_length(bytes([1, 2, 3]), 3, "Test")  # Should not raise
        _require_min_length(bytes([1, 2, 3, 4]), 3, "Test")  # Should not raise

    def test_raises_when_insufficient(self) -> None:
        """Validation raises DecodeError when length is insufficient."""
        with pytest.raises(DecodeError) as exc:
            _require_min_length(bytes([1, 2]), 5, "TestContext")
        assert "TestContext" in str(exc.value)
        assert ">= 5 bytes" in str(exc.value)
        assert "got 2" in str(exc.value)


class TestRequireExactLength:
    """Tests for _require_exact_length validation."""

    def test_passes_when_exact(self) -> None:
        """Validation passes when length matches exactly."""
        _require_exact_length(bytes([1, 2, 3]), 3, "Test")  # Should not raise

    def test_raises_when_wrong_length(self) -> None:
        """Validation raises DecodeError when length is wrong."""
        with pytest.raises(DecodeError) as exc:
            _require_exact_length(bytes([1, 2]), 5, "TestContext")
        assert "TestContext" in str(exc.value)
        assert "expected 5 bytes" in str(exc.value)


class TestRequirePrefix:
    """Tests for _require_prefix validation."""

    def test_passes_with_correct_prefix(self) -> None:
        """Validation passes with correct prefix."""
        _require_prefix("=team|data", "=", "Test")  # Should not raise
        _require_prefix("+info|data", "+", "Test")  # Should not raise

    def test_raises_without_prefix(self) -> None:
        """Validation raises DecodeError without expected prefix."""
        with pytest.raises(DecodeError) as exc:
            _require_prefix("team|data", "=", "TestContext")
        assert "TestContext" in str(exc.value)
        assert "expected prefix '='" in str(exc.value)


class TestRequireParts:
    """Tests for _require_parts validation."""

    def test_passes_with_enough_parts(self) -> None:
        """Validation passes with sufficient parts."""
        _require_parts(["a", "b", "c"], 3, "Test")  # Should not raise
        _require_parts(["a", "b", "c", "d"], 3, "Test")  # Should not raise

    def test_raises_with_insufficient_parts(self) -> None:
        """Validation raises DecodeError with too few parts."""
        with pytest.raises(DecodeError) as exc:
            _require_parts(["a", "b"], 5, "TestContext")
        assert "TestContext" in str(exc.value)
        assert ">= 5 parts" in str(exc.value)


# =============================================================================
# Text Message Decoder Tests
# =============================================================================


class TestDecodeJoinConfirm:
    """Tests for decode_join_confirm function."""

    def test_decodes_valid_join_confirm(self) -> None:
        """Decodes valid join confirmation message."""
        data = b"=2|2024-01-15|PlayerName|4|1|0|1|0"
        result = decode_join_confirm(data)
        assert result["msg_type"] == 0x3D
        assert result["team"] == 2
        assert result["join_date"] == "2024-01-15"
        assert result["name"] == "PlayerName"
        assert result["rank"] == 4
        assert result["equipment"] == [1, 0, 1, 0]

    def test_decodes_with_missing_equipment(self) -> None:
        """Decodes join confirmation with missing equipment fields."""
        data = b"=1|2024-01-15|Tank|3"
        result = decode_join_confirm(data)
        assert result["equipment"] == []

    def test_raises_on_wrong_prefix(self) -> None:
        """Raises DecodeError when prefix is wrong."""
        with pytest.raises(DecodeError):
            decode_join_confirm(b"+wrong|prefix")

    def test_raises_on_too_few_parts(self) -> None:
        """Raises DecodeError when too few parts."""
        with pytest.raises(DecodeError):
            decode_join_confirm(b"=1|2|3")


class TestDecodeWorldInfo:
    """Tests for decode_world_info function."""

    def test_decodes_valid_world_info(self) -> None:
        """Decodes valid world info message."""
        data = b"+123|WorldName|456|1,2,3|2|mode|image.png|2024"
        result = decode_world_info(data)
        assert result["msg_type"] == 0x2B
        assert result["world_id"] == 123
        assert result["name"] == "WorldName"
        assert result["field_id"] == 456
        assert result["flags"] == [1, 2, 3]
        assert result["team"] == 2
        assert result["mode"] == "mode"
        assert result["image"] == "image.png"
        assert result["year"] == 2024

    def test_handles_non_numeric_year(self) -> None:
        """Handles non-numeric year field."""
        data = b"+1|Name|2|1|0|mode|img|invalid"
        result = decode_world_info(data)
        assert result["year"] == 0

    def test_raises_on_wrong_prefix(self) -> None:
        """Raises DecodeError when prefix is wrong."""
        with pytest.raises(DecodeError):
            decode_world_info(b"=wrong|prefix")

    def test_raises_on_too_few_parts(self) -> None:
        """Raises DecodeError when too few parts."""
        with pytest.raises(DecodeError):
            decode_world_info(b"+1|2|3|4|5")


class TestDecodeTextMessage:
    """Tests for decode_text_message function."""

    def test_dispatches_join_confirm(self) -> None:
        """Dispatches to join confirm decoder."""
        data = b"=2|2024-01-15|Player|4|1|0|1|0"
        result = decode_text_message(data)
        assert result["msg_type"] == 0x3D

    def test_dispatches_world_info(self) -> None:
        """Dispatches to world info decoder."""
        data = b"+123|Name|456|1|2|mode|img|2024"
        result = decode_text_message(data)
        assert result["msg_type"] == 0x2B

    def test_raises_on_empty_body(self) -> None:
        """Raises DecodeError on empty body."""
        with pytest.raises(DecodeError) as exc:
            decode_text_message(b"")
        assert "empty body" in str(exc.value)

    def test_raises_on_unknown_type(self) -> None:
        """Raises DecodeError on unknown message type."""
        with pytest.raises(DecodeError) as exc:
            decode_text_message(b"X unknown")
        assert "unknown type" in str(exc.value)


# =============================================================================
# Binary Message Decoder Tests
# =============================================================================


class TestDecodeShootEvent:
    """Tests for decode_shoot_event function."""

    def test_decodes_valid_shoot_event(self) -> None:
        """Decodes valid shooting event."""
        # shooter_id=0x0102, target=(10,20), proj=(15,25), fuel=0x030405, weapon=1, ammo=5, ff=0
        data = bytes([0x02, 0x01, 10, 20, 15, 25, 0x03, 0x04, 0x05, 1, 5, 0])
        result = decode_shoot_event(data)
        assert result["msg_type"] == 0x53
        assert result["shooter_id"] == 0x0102
        assert result["target_x"] == 10
        assert result["target_y"] == 20
        assert result["projectile_x"] == 15
        assert result["projectile_y"] == 25
        assert result["fuel"] == _x24(0x03, 0x04, 0x05)
        assert result["weapon"] == 1
        assert result["ammo"] == 5
        assert result["friendly_fire"] is False

    def test_decodes_friendly_fire(self) -> None:
        """Decodes friendly fire flag correctly."""
        data = bytes([0x02, 0x01, 10, 20, 15, 25, 0x03, 0x04, 0x05, 1, 5, 1])
        result = decode_shoot_event(data)
        assert result["friendly_fire"] is True

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_shoot_event(bytes([1, 2, 3, 4, 5]))


class TestDecodeHitConfirmation:
    """Tests for decode_hit_confirmation function."""

    def test_decodes_valid_hit_confirmation(self) -> None:
        """Decodes valid hit confirmation."""
        # 12 bytes starting with 0x2E
        # After XOR decode: decoded[5]=target_y, decoded[6]=target_x
        # data[6] -> decoded[5], data[7] -> decoded[6]
        data = bytes([0x2E, 0x01, 0x02, 0x03, 0x04, 0x05, 0x35, 0x50, 0x08, 0x09, 0x0A, 0x0B])
        xor_table = bytes([0x00] * 11)  # No-op XOR
        result = decode_hit_confirmation(data, xor_table)
        assert result["msg_type"] == 0x2E
        assert result["target_y"] == 0x35  # decoded[5] = data[6]
        assert result["target_x"] == 0x50  # decoded[6] = data[7]

    def test_raises_on_wrong_length(self) -> None:
        """Raises DecodeError on wrong length."""
        data = bytes([0x2E, 0x01, 0x02])
        xor_table = bytes([0x00] * 3)
        with pytest.raises(DecodeError):
            decode_hit_confirmation(data, xor_table)

    def test_raises_on_wrong_prefix(self) -> None:
        """Raises DecodeError on wrong prefix."""
        data = bytes([0x3E] + [0x00] * 11)
        xor_table = bytes([0x00] * 11)
        with pytest.raises(DecodeError) as exc:
            decode_hit_confirmation(data, xor_table)
        assert "expected 0x2E prefix" in str(exc.value)


class TestDecodeDeactivation:
    """Tests for decode_deactivation function."""

    def test_decodes_with_points(self) -> None:
        """Decodes deactivation with points field."""
        # victim=0x0102, killer=0x0304, rank=5, points=0x0607
        data = bytes([0x02, 0x01, 0x04, 0x03, 5, 0x07, 0x06])
        result = decode_deactivation(data)
        assert result["msg_type"] == 0x41
        assert result["victim_id"] == 0x0102
        assert result["killer_id"] == 0x0304
        assert result["rank"] == 5
        assert result["points"] == 0x0607

    def test_decodes_without_points(self) -> None:
        """Decodes deactivation without points field."""
        data = bytes([0x02, 0x01, 0x04, 0x03, 5])
        result = decode_deactivation(data)
        assert result["points"] == 0

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_deactivation(bytes([1, 2, 3]))


class TestDecodeFuelGain:
    """Tests for decode_fuel_gain function."""

    def test_decodes_paid_fuel(self) -> None:
        """Decodes paid fuel gain."""
        # amount=0x1234, is_free=False (data[2] != 0)
        data = bytes([0x34, 0x12, 1])
        result = decode_fuel_gain(data)
        assert result["msg_type"] == 0x44
        assert result["amount"] == 0x1234
        assert result["is_free"] is False

    def test_decodes_free_fuel(self) -> None:
        """Decodes free fuel gain."""
        data = bytes([0x34, 0x12, 0])
        result = decode_fuel_gain(data)
        assert result["is_free"] is True

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_fuel_gain(bytes([1, 2]))


class TestDecodeFuelDeposit:
    """Tests for decode_fuel_deposit function."""

    def test_decodes_fuel_deposit(self) -> None:
        """Decodes fuel deposit amount."""
        data = bytes([0x64, 0x00])  # amount=100
        result = decode_fuel_deposit(data)
        assert result["msg_type"] == 0x64
        assert result["amount"] == 100

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_fuel_deposit(bytes([1]))


class TestDecodeRadarResult:
    """Tests for decode_radar_result function."""

    def test_decodes_radar_found(self) -> None:
        """Decodes radar result with entity found."""
        data = bytes([3, 1])  # detection_type=3, found=True
        result = decode_radar_result(data)
        assert result["msg_type"] == 0x46
        assert result["detection_type"] == 3
        assert result["found"] is True

    def test_decodes_radar_not_found(self) -> None:
        """Decodes radar result with no entity found."""
        data = bytes([3, 0])
        result = decode_radar_result(data)
        assert result["found"] is False

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_radar_result(bytes([1]))


class TestDecodeEnemyDetection:
    """Tests for decode_enemy_detection function."""

    def test_decodes_enemy_detection(self) -> None:
        """Decodes enemy detection message."""
        # tank_id=0x0102, x=50, y=60, rank=4, team=2
        data = bytes([0x02, 0x01, 50, 60, 4, 2])
        result = decode_enemy_detection(data)
        assert result["msg_type"] == 0x48
        assert result["tank_id"] == 0x0102
        assert result["x"] == 50
        assert result["y"] == 60
        assert result["rank"] == 4
        assert result["team"] == 2

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_enemy_detection(bytes([1, 2, 3]))


class TestDecodeInventory:
    """Tests for decode_inventory function."""

    def test_decodes_inventory_show(self) -> None:
        """Decodes inventory with show flag."""
        # show=1, counts with enabled flags
        data = bytes([1, 5, 10 | 128, 3, 7, 0])  # armor enabled, dual disabled, others enabled
        result = decode_inventory(data)
        assert result["msg_type"] == 0x49
        assert result["show"] is True
        assert result["alternate"] is False
        assert result["counts"] == [5, 10, 3, 7, 0]
        assert result["enabled"] == [True, False, True, True, True]

    def test_decodes_inventory_alternate(self) -> None:
        """Decodes inventory with alternate flag."""
        data = bytes([2, 0, 0, 0, 0, 0])
        result = decode_inventory(data)
        assert result["alternate"] is True

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_inventory(bytes([1, 2, 3]))


class TestDecodeEquipmentGain:
    """Tests for decode_equipment_gain function."""

    def test_decodes_equipment_gain(self) -> None:
        """Decodes equipment gain message."""
        # show_message=1, gained=[1,2,0,1,0]
        data = bytes([1, 1, 2, 0, 1, 0])
        result = decode_equipment_gain(data)
        assert result["msg_type"] == 0x67
        assert result["show_message"] is True
        assert result["gained"] == [1, 2, 0, 1, 0]

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_equipment_gain(bytes([1, 2, 3]))


class TestDecodeEquipmentToggle:
    """Tests for decode_equipment_toggle function."""

    def test_decodes_equipment_toggle(self) -> None:
        """Decodes equipment toggle message."""
        data = bytes([1, 0, 1, 1, 0])  # armor=on, dual=off, missile=on, homing=on, radar=off
        result = decode_equipment_toggle(data)
        assert result["msg_type"] == 0x74
        assert result["enabled"] == [True, False, True, True, False]

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_equipment_toggle(bytes([1, 2]))


class TestDecodeMinePlacement:
    """Tests for decode_mine_placement function."""

    def test_decodes_mine_placement(self) -> None:
        """Decodes mine placement message."""
        # type=1, tank_id=0x0102, count=2, positions=[(10,20), (30,40)]
        data = bytes([1, 0x02, 0x01, 2, 10, 20, 30, 40])
        result = decode_mine_placement(data)
        assert result["msg_type"] == 0x4B
        assert result["mine_type"] == 1
        assert result["tank_id"] == 0x0102
        assert result["positions"] == [(10, 20), (30, 40)]

    def test_handles_truncated_positions(self) -> None:
        """Handles truncated position data."""
        data = bytes([1, 0x02, 0x01, 3, 10, 20])  # Claims 3 positions but only has 1
        result = decode_mine_placement(data)
        assert result["positions"] == [(10, 20)]

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_mine_placement(bytes([1, 2]))


class TestDecodeMineDetonation:
    """Tests for decode_mine_detonation function."""

    def test_decodes_mine_detonation(self) -> None:
        """Decodes mine detonation message."""
        data = bytes([10, 20, 30, 40])  # Two positions
        result = decode_mine_detonation(data)
        assert result["msg_type"] == 0x45
        assert result["positions"] == [(10, 20), (30, 40)]

    def test_handles_empty_data(self) -> None:
        """Handles empty position data."""
        result = decode_mine_detonation(b"")
        assert result["positions"] == []


class TestDecodeRadarScanResult:
    """Tests for decode_radar_scan_result function."""

    def test_decodes_radar_scan(self) -> None:
        """Decodes radar scan with entities."""
        # count=2, skip 1 byte, then 4-byte entities (x, y, value_lo, value_hi)
        data = bytes([2, 0, 10, 20, 0x34, 0x12, 30, 40, 0xFF, 0x7F])
        result = decode_radar_scan_result(data)
        assert result["msg_type"] == 0x4F
        assert len(result["entities"]) == 2
        assert result["entities"][0] == (10, 20, 0x1234)
        assert result["entities"][1] == (30, 40, 0x7FFF)

    def test_handles_negative_values(self) -> None:
        """Handles negative signed values correctly."""
        # value with high bit set becomes negative
        data = bytes([1, 0, 10, 20, 0x00, 0x80])  # 0x8000 -> -32768
        result = decode_radar_scan_result(data)
        assert result["entities"][0][2] == -32768

    def test_handles_truncated_entities(self) -> None:
        """Handles truncated entity data."""
        data = bytes([2, 0, 10, 20, 0x34])  # Claims 2 but only partial first
        result = decode_radar_scan_result(data)
        assert result["entities"] == []

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_radar_scan_result(bytes([1]))


class TestDecodeMovement:
    """Tests for decode_movement function."""

    def test_decodes_movement(self) -> None:
        """Decodes movement message."""
        # tank_id=0x0102, start=(50, 60), dir=3, flag=1, fuel=0x030405
        data = bytes([0x02, 0x01, 50, 60, 3, 1, 0x03, 0x04, 0x05])
        result = decode_movement(data)
        assert result["msg_type"] == 0x47
        assert result["tank_id"] == 0x0102
        assert result["start_x"] == 50
        assert result["start_y"] == 60
        assert result["direction"] == 3
        assert result["flag"] == 1
        assert result["fuel"] == _x24(0x03, 0x04, 0x05)
        assert result["waypoints"] == []

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_movement(bytes([1, 2, 3, 4]))


class TestDecodeTankInfo:
    """Tests for decode_tank_info function."""

    def test_decodes_tank_info_with_name(self) -> None:
        """Decodes tank info with name."""
        # team=2, tank_id=0x0102, decoration=4 bytes, score=0x030405, name="Test"
        data = bytes([2, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF, 0x03, 0x04, 0x05]) + b"Test"
        result = decode_tank_info(data)
        assert result["msg_type"] == 0x21
        assert result["team"] == 2
        assert result["tank_id"] == 0x0102
        assert result["decoration_state"] == bytes([0xDE, 0xAD, 0xBE, 0xEF])
        assert result["score"] == _x24(0x03, 0x04, 0x05)
        assert result["name"] == "Test"

    def test_decodes_tank_info_without_name(self) -> None:
        """Decodes tank info without name."""
        data = bytes([2, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF, 0x03, 0x04, 0x05])
        result = decode_tank_info(data)
        assert result["name"] == ""

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_tank_info(bytes([1, 2, 3]))


class TestDecodeMovementResponse:
    """Tests for decode_movement_response function."""

    def test_decodes_movement_response(self) -> None:
        """Decodes movement response message."""
        # team=1, tank_id=0x0102, x=50, y=60, dir=3, skip 1, rank=4, lb_pos=0x050607
        data = bytes([1, 0x02, 0x01, 50, 60, 3, 0x00, 4, 0x05, 0x06, 0x07])
        result = decode_movement_response(data)
        assert result["msg_type"] == 0x3D
        assert result["team"] == 1
        assert result["tank_id"] == 0x0102
        assert result["x"] == 50
        assert result["y"] == 60
        assert result["direction"] == 3
        assert result["rank"] == 4
        assert result["leaderboard_position"] == _x24(0x05, 0x06, 0x07)

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_movement_response(bytes([1, 2, 3, 4, 5]))


class TestDecodeSync:
    """Tests for decode_sync function."""

    def test_decodes_sync(self) -> None:
        """Decodes sync message (always succeeds)."""
        result = decode_sync(b"")
        assert result["msg_type"] == 0x3F

    def test_ignores_extra_data(self) -> None:
        """Ignores any extra data in sync message."""
        result = decode_sync(bytes([1, 2, 3, 4, 5]))
        assert result["msg_type"] == 0x3F


class TestDecodeContainer:
    """Tests for decode_container function."""

    def test_decodes_container(self) -> None:
        """Decodes container fuel message."""
        # container_id=0x0102, fuel=0x0304
        data = bytes([0x02, 0x01, 0x04, 0x03])
        result = decode_container(data)
        assert result["msg_type"] == 0x43
        assert result["container_id"] == 0x0102
        assert result["fuel"] == 0x0304

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_container(bytes([1, 2]))


class TestDecodeTankEntry:
    """Tests for decode_tank_entry function."""

    def test_decodes_tank_entry_with_name(self) -> None:
        """Decodes tank entry with name."""
        # tank_id=5, x=0x0102, y=60, padding to 10 bytes, then name
        data = bytes([5, 0x02, 0x01, 60, 0, 0, 0, 0, 0, 0]) + b"Tank"
        result = decode_tank_entry(data)
        assert result["msg_type"] == 0x28
        assert result["tank_id"] == 5
        assert result["x"] == 0x0102
        assert result["y"] == 60
        assert result["name"] == "Tank"

    def test_decodes_tank_entry_without_name(self) -> None:
        """Decodes tank entry without name."""
        data = bytes([5, 0x02, 0x01, 60, 0, 0, 0, 0, 0, 0])
        result = decode_tank_entry(data)
        assert result["name"] == ""

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_tank_entry(bytes([1, 2, 3]))


class TestDecodeTankExit:
    """Tests for decode_tank_exit function."""

    def test_decodes_tank_exit(self) -> None:
        """Decodes tank exit message."""
        data = bytes([0x02, 0x01])  # tank_id=0x0102
        result = decode_tank_exit(data)
        assert result["msg_type"] == 0x58
        assert result["tank_id"] == 0x0102

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_tank_exit(bytes([1]))


class TestDecodeActionDone:
    """Tests for decode_action_done function."""

    def test_decodes_action_done(self) -> None:
        """Decodes action done message (always succeeds)."""
        result = decode_action_done(b"")
        assert result["msg_type"] == 0x54


class TestDecodeChatMessage:
    """Tests for decode_chat_message function."""

    def test_decodes_chat_with_coords(self) -> None:
        """Decodes chat message with coordinates."""
        # sender_id=0x0102, message_type=1, x=50, y=60
        data = bytes([0x02, 0x01, 1, 50, 60])
        result = decode_chat_message(data)
        assert result["msg_type"] == 0x4D
        assert result["sender_id"] == 0x0102
        assert result["message_type"] == 1
        assert result["x"] == 50
        assert result["y"] == 60

    def test_decodes_chat_without_coords(self) -> None:
        """Decodes chat message without coordinates."""
        data = bytes([0x02, 0x01, 1])
        result = decode_chat_message(data)
        assert result["x"] is None
        assert result["y"] is None

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_chat_message(bytes([1, 2]))


class TestDecodeStatistics:
    """Tests for decode_statistics function."""

    def test_decodes_statistics(self) -> None:
        """Decodes statistics message."""
        data = (
            bytes([0x10, 0x00, 30, 45])  # hours=16, mins=30, secs=45
            + (100).to_bytes(4, "little")  # destroyed
            + (50).to_bytes(4, "little")  # deactivated
            + (5000).to_bytes(4, "little")  # score
        )
        result = decode_statistics(data)
        assert result["msg_type"] == 0x56
        assert result["playtime_hours"] == 16
        assert result["playtime_minutes"] == 30
        assert result["playtime_seconds"] == 45
        assert result["destroyed"] == 100
        assert result["deactivated"] == 50
        assert result["score"] == 5000

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_statistics(bytes([1, 2, 3, 4]))


class TestDecodeActiveForces:
    """Tests for decode_active_forces function."""

    def test_decodes_active_forces(self) -> None:
        """Decodes active forces message."""
        data = bytes([10, 15, 8, 12])  # Team counts
        result = decode_active_forces(data)
        assert result["msg_type"] == 0x2A
        assert result["team_counts"] == [10, 15, 8, 12]

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_active_forces(bytes([1, 2]))


class TestDecodeTankStatusSync:
    """Tests for decode_tank_status_sync function."""

    def test_decodes_short_format(self) -> None:
        """Decodes 8-byte tank status sync."""
        # subtype=1, tank_id=0x0102, damage=2, rank=4, flags, lb_pos
        data = bytes([1, 0x02, 0x01, 2, 4, 0, 0x10, 0x00])
        result = decode_tank_status_sync(data)
        assert result["msg_type"] == 0x2E
        assert result["subtype"] == 1
        assert result["tank_id"] == 0x0102
        assert result["damage_state"] == 2
        assert result["rank"] == 4
        assert result["fuel"] is None

    def test_decodes_long_format(self) -> None:
        """Decodes 12+ byte tank status sync with fuel."""
        data = bytes([3, 0x02, 0x01, 0, 5, 0, 0x10, 0x00, 0, 0, 0xE8, 0x03])  # fuel=1000
        result = decode_tank_status_sync(data)
        assert result["subtype"] == 3
        assert result["fuel"] == 1000

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_tank_status_sync(bytes([1, 2, 3]))


class TestDecodeTankStatus:
    """Tests for decode_tank_status function."""

    def test_decodes_tank_status_with_name(self) -> None:
        """Decodes full tank status with name."""
        # info_byte: team=2, rank=4 -> (4<<4)|2 = 0x42
        # tank_id, decoration(4), lb_score(3), lb_pos(3), name
        header = bytes([0x42, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF])
        lb_bytes = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        data = header + lb_bytes + b"Tank"
        result = decode_tank_status(data)
        assert result["msg_type"] == 0x3E
        assert result["team"] == 2
        assert result["rank"] == 4
        assert result["tank_id"] == 0x0102
        assert result["name"] == "Tank"

    def test_decodes_tank_status_without_name(self) -> None:
        """Decodes tank status without name."""
        data = bytes([0x42, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        result = decode_tank_status(data)
        assert result["name"] == ""

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_tank_status(bytes([1, 2, 3, 4, 5]))


class TestDecodeSupervisor:
    """Tests for decode_supervisor function."""

    def test_decodes_supervisor(self) -> None:
        """Decodes supervisor message."""
        data = bytes([1, 0, 3])  # status=1, reserved=0, data=3
        result = decode_supervisor(data)
        assert result["msg_type"] == 0x52
        assert result["status"] == 1
        assert result["reserved"] == 0
        assert result["data"] == 3

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_supervisor(bytes([1, 2]))


class TestSupervisorHelpers:
    """Tests for supervisor helper functions."""

    def test_supervisor_is_promo_eligible(self) -> None:
        """Checks promo eligibility correctly."""
        eligible: SupervisorDict = {"msg_type": 0x52, "status": 1, "reserved": 0, "data": 0}
        not_eligible: SupervisorDict = {"msg_type": 0x52, "status": 8, "reserved": 0, "data": 0}
        assert supervisor_is_promo_eligible(eligible) is True
        assert supervisor_is_promo_eligible(not_eligible) is False

    def test_supervisor_has_promo_kill(self) -> None:
        """Checks promo kill correctly."""
        has_kill: SupervisorDict = {"msg_type": 0x52, "status": 8, "reserved": 0, "data": 0}
        no_kill: SupervisorDict = {"msg_type": 0x52, "status": 1, "reserved": 0, "data": 0}
        assert supervisor_has_promo_kill(has_kill) is True
        assert supervisor_has_promo_kill(no_kill) is False


class TestDecodeTerrainUpdate:
    """Tests for decode_terrain_update function."""

    def test_decodes_terrain_updates(self) -> None:
        """Decodes terrain update triplets."""
        # Two updates: (10, 20, 5) and (30, 40, 0)
        data = bytes([10, 20, 5, 30, 40, 0])
        result = decode_terrain_update(data)
        assert result["msg_type"] == 0x4A
        assert result["updates"] == [(10, 20, 5), (30, 40, 0)]

    def test_handles_empty_data(self) -> None:
        """Handles empty terrain data."""
        result = decode_terrain_update(b"")
        assert result["updates"] == []


class TestDecodeViewportUpdate:
    """Tests for decode_viewport_update function."""

    def test_decodes_viewport_header(self) -> None:
        """Decodes viewport update header."""
        data = bytes([3, 0x0F])  # direction=3, flags=0x0F
        result = decode_viewport_update(data)
        assert result["msg_type"] == 0x5A
        assert result["direction"] == 3
        assert result["flags"] == 0x0F
        assert result["entities"] == []

    def test_decodes_viewport_with_entities(self) -> None:
        """Decodes viewport with entity data."""
        # direction=0, flags=0, delta=1 (col=1, row=0), entity data (3 bytes)
        # z = (entity_id << 8) | (value << 4) | terrain_type
        # Let's encode: terrain=5, value=2, entity_id=100
        # z = (100 << 8) | (2 << 4) | 5 = 0x6425
        # Big endian 3 bytes: 0x00, 0x64, 0x25
        data = bytes([0, 0, 1, 0x00, 0x64, 0x25])
        result = decode_viewport_update(data)
        assert len(result["entities"]) == 1
        entity = result["entities"][0]
        assert entity["col"] == 1
        assert entity["row"] == 0
        assert entity["terrain_type"] == 5
        assert entity["entity_id"] == 100

    def test_handles_skip_marker(self) -> None:
        """Handles delta 255 as skip marker."""
        data = bytes([0, 0, 255])  # Skip marker, no entity data follows
        result = decode_viewport_update(data)
        assert result["entities"] == []

    def test_handles_column_wrap(self) -> None:
        """Handles column wraparound to next row."""
        # Delta of 20 should wrap: col += 20 -> col=20, then col>=18 so col-=18=2, row+=1
        data = bytes([0, 0, 20, 0x00, 0x00, 0x00])
        result = decode_viewport_update(data)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["col"] == 2
        assert result["entities"][0]["row"] == 1

    def test_handles_column_wrap_multiple_entities(self) -> None:
        """Handles column accumulation across entities requiring normalization."""
        # First entity: delta=10 -> col=10, row=0
        # Second entity: delta=10 -> col=20, triggers while loop: col=2, row=1
        data = bytes([0, 0, 10, 0x00, 0x00, 0x00, 10, 0x00, 0x00, 0x00])
        result = decode_viewport_update(data)
        assert len(result["entities"]) == 2
        assert result["entities"][0]["col"] == 10
        assert result["entities"][0]["row"] == 0
        assert result["entities"][1]["col"] == 2
        assert result["entities"][1]["row"] == 1

    def test_handles_truncated_entity_data(self) -> None:
        """Handles truncated data gracefully by breaking early."""
        # Non-255 delta but only 1 byte of entity data (need 3)
        data = bytes([0, 0, 5, 0x00])
        result = decode_viewport_update(data)
        # Should break early, no entities parsed
        assert result["entities"] == []

    def test_handles_tank_entity_id(self) -> None:
        """Handles special tank entity ID (65535 -> -1)."""
        # entity_id=65535 (0xFFFF) means tank
        # z = (0xFFFF << 8) | 0 | 0 = 0xFFFF00
        data = bytes([0, 0, 1, 0xFF, 0xFF, 0x00])
        result = decode_viewport_update(data)
        assert result["entities"][0]["entity_id"] == -1

    def test_handles_high_value(self) -> None:
        """Handles value >= 8 becoming 255."""
        # value = 8 -> becomes 255
        # z = (0 << 8) | (8 << 4) | 0 = 0x80
        data = bytes([0, 0, 1, 0x00, 0x00, 0x80])
        result = decode_viewport_update(data)
        assert result["entities"][0]["value"] == 255

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_viewport_update(bytes([1]))


class TestViewportEntityHelpers:
    """Tests for viewport entity helper functions."""

    def test_viewport_entity_is_tank(self) -> None:
        """Checks if entity is a tank."""
        tank: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": -1,
            "value": 0,
            "terrain_type": 0,
        }
        not_tank: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": 100,
            "value": 0,
            "terrain_type": 0,
        }
        assert viewport_entity_is_tank(tank) is True
        assert viewport_entity_is_tank(not_tank) is False

    def test_viewport_entity_is_container(self) -> None:
        """Checks if entity is a container."""
        container: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": 100,
            "value": 0,
            "terrain_type": 0,
        }
        not_container: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": 0,
            "value": 0,
            "terrain_type": 0,
        }
        assert viewport_entity_is_container(container) is True
        assert viewport_entity_is_container(not_container) is False

    def test_viewport_entity_is_empty(self) -> None:
        """Checks if tile is empty."""
        empty: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": 0,
            "value": 0,
            "terrain_type": 0,
        }
        not_empty: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": 100,
            "value": 0,
            "terrain_type": 0,
        }
        assert viewport_entity_is_empty(empty) is True
        assert viewport_entity_is_empty(not_empty) is False


class TestDecode0x2eMessage:
    """Tests for decode_0x2e_message function."""

    def test_dispatches_to_container_decoder(self) -> None:
        """Dispatches to container decoder module."""
        # 11 bytes = combat hit
        data = bytes([0x59, 0x09, 0xCD, 0x07, 0x99, 0x84, 0x93, 0xCE, 0x9C, 0x80, 0x51])
        result = decode_0x2e_message(data)
        assert result["msg_type"] == "combat_hit"


# =============================================================================
# Message Type Detection Tests
# =============================================================================


class TestIsTextMessage:
    """Tests for is_text_message function."""

    def test_text_message_types(self) -> None:
        """Returns True for text message types."""
        assert is_text_message(MSG_TANK_POS) is True  # '='
        assert is_text_message(MSG_PROMOTION) is True  # '+'

    def test_binary_message_types(self) -> None:
        """Returns False for binary message types."""
        assert is_text_message(MSG_SHOOT) is False
        assert is_text_message(MSG_MOVEMENT) is False
        assert is_text_message(MSG_CONTAINER) is False

    def test_text_msg_types_constant(self) -> None:
        """TEXT_MSG_TYPES contains expected values."""
        assert MSG_TANK_POS in TEXT_MSG_TYPES
        assert MSG_PROMOTION in TEXT_MSG_TYPES


# =============================================================================
# Main Dispatcher Tests
# =============================================================================


class TestDecodeMessage:
    """Tests for decode_message dispatcher function."""

    def test_dispatches_combat_messages(self) -> None:
        """Dispatches to combat message decoders."""
        # Shoot
        shoot_data = bytes([0x02, 0x01, 10, 20, 15, 25, 0x03, 0x04, 0x05, 1, 5, 0])
        result = decode_message(MSG_SHOOT, shoot_data)
        assert result["msg_type"] == 0x53

        # Deactivation
        deactivation_data = bytes([0x02, 0x01, 0x04, 0x03, 5, 0x07, 0x06])
        result = decode_message(MSG_DEACTIVATE, deactivation_data)
        assert result["msg_type"] == 0x41

        # Mine placement
        mine_place_data = bytes([1, 0x02, 0x01, 1, 10, 20])
        result = decode_message(MSG_MINE_PLACE, mine_place_data)
        assert result["msg_type"] == 0x4B

        # Mine detonation
        mine_det_data = bytes([10, 20])
        result = decode_message(MSG_MINE_DETONATE, mine_det_data)
        assert result["msg_type"] == 0x45

    def test_dispatches_resource_messages(self) -> None:
        """Dispatches to resource message decoders."""
        # Fuel gain
        fuel_gain_data = bytes([0x34, 0x12, 1])
        result = decode_message(MSG_FUEL_GAIN, fuel_gain_data)
        assert result["msg_type"] == 0x44

        # Fuel deposit
        fuel_deposit_data = bytes([0x64, 0x00])
        result = decode_message(MSG_FUEL_DEPOSIT, fuel_deposit_data)
        assert result["msg_type"] == 0x64

        # Inventory
        inventory_data = bytes([1, 5, 10, 3, 7, 0])
        result = decode_message(MSG_INVENTORY, inventory_data)
        assert result["msg_type"] == 0x49

        # Equipment gain
        equip_gain_data = bytes([1, 1, 2, 0, 1, 0])
        result = decode_message(MSG_EQUIP_GAIN, equip_gain_data)
        assert result["msg_type"] == 0x67

        # Equipment toggle
        equip_toggle_data = bytes([1, 0, 1, 1, 0])
        result = decode_message(MSG_EQUIP_TOGGLE, equip_toggle_data)
        assert result["msg_type"] == 0x74

    def test_dispatches_radar_messages(self) -> None:
        """Dispatches to radar message decoders."""
        # Radar result
        radar_data = bytes([3, 1])
        result = decode_message(MSG_RADAR_RESULT, radar_data)
        assert result["msg_type"] == 0x46

        # Enemy detection
        enemy_data = bytes([0x02, 0x01, 50, 60, 4, 2])
        result = decode_message(MSG_ENEMY_DETECT, enemy_data)
        assert result["msg_type"] == 0x48

        # Tile update (radar scan result)
        tile_data = bytes([1, 0, 10, 20, 0x34, 0x12])
        result = decode_message(MSG_TILE_UPDATE, tile_data)
        assert result["msg_type"] == 0x4F

    def test_dispatches_tank_messages(self) -> None:
        """Dispatches to tank message decoders."""
        # Tank entry
        entry_data = bytes([5, 0x02, 0x01, 60, 0, 0, 0, 0, 0, 0])
        result = decode_message(MSG_TANK_ENTRY, entry_data)
        assert result["msg_type"] == 0x28

        # Tank exit
        exit_data = bytes([0x02, 0x01])
        result = decode_message(MSG_TANK_EXIT, exit_data)
        assert result["msg_type"] == 0x58

        # Tank stats (0x2E) - uses container decoder
        stats_data = bytes([0x59, 0x09, 0xCD, 0x07, 0x99, 0x84, 0x93, 0xCE, 0x9C, 0x80, 0x51])
        result = decode_message(MSG_TANK_STATS, stats_data)
        assert result["msg_type"] == "combat_hit"

        # Tank status full
        status_header = bytes([0x42, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF])
        status_lb = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        result = decode_message(MSG_TANK_STATUS_FULL, status_header + status_lb)
        assert result["msg_type"] == 0x3E

        # Tank info
        info_data = bytes([2, 0x02, 0x01, 0xDE, 0xAD, 0xBE, 0xEF, 0x03, 0x04, 0x05])
        result = decode_message(MSG_TANK_INFO, info_data)
        assert result["msg_type"] == 0x21

    def test_dispatches_movement_messages(self) -> None:
        """Dispatches to movement message decoders."""
        # Movement
        movement_data = bytes([0x02, 0x01, 50, 60, 3, 1, 0x03, 0x04, 0x05])
        result = decode_message(MSG_MOVEMENT, movement_data)
        assert result["msg_type"] == 0x47

        # Move response
        response_data = bytes([1, 0x02, 0x01, 50, 60, 3, 0x00, 4, 0x05, 0x06, 0x07])
        result = decode_message(MSG_MOVE_RESPONSE, response_data)
        assert result["msg_type"] == 0x3D

    def test_dispatches_world_messages(self) -> None:
        """Dispatches to world message decoders."""
        # Viewport
        viewport_data = bytes([0, 0])
        result = decode_message(MSG_VIEWPORT, viewport_data)
        assert result["msg_type"] == 0x5A

        # Terrain update
        terrain_data = bytes([10, 20, 5])
        result = decode_message(MSG_TERRAIN_UPDATE, terrain_data)
        assert result["msg_type"] == 0x4A

        # Sync
        sync_data = b""
        result = decode_message(MSG_SYNC, sync_data)
        assert result["msg_type"] == 0x3F

        # Container
        container_data = bytes([0x02, 0x01, 0x04, 0x03])
        result = decode_message(MSG_CONTAINER, container_data)
        assert result["msg_type"] == 0x43

    def test_dispatches_misc_messages(self) -> None:
        """Dispatches to misc message decoders."""
        # Chat
        chat_data = bytes([0x02, 0x01, 1])
        result = decode_message(MSG_CHAT, chat_data)
        assert result["msg_type"] == 0x4D

        # Statistics
        stats_data = bytes([0x10, 0x00, 30, 45]) + bytes(12)
        result = decode_message(MSG_STATISTICS, stats_data)
        assert result["msg_type"] == 0x56

        # Active forces
        forces_data = bytes([10, 15, 8, 12])
        result = decode_message(MSG_ACTIVE_FORCES, forces_data)
        assert result["msg_type"] == 0x2A

        # Supervisor
        supervisor_data = bytes([1, 0, 3])
        result = decode_message(MSG_SUPERVISOR, supervisor_data)
        assert result["msg_type"] == 0x52

        # Action done
        action_data = b""
        result = decode_message(MSG_ACTION_DONE, action_data)
        assert result["msg_type"] == 0x54

    def test_raises_on_unknown_type(self) -> None:
        """Raises DecodeError on unknown message type."""
        with pytest.raises(DecodeError) as exc:
            decode_message(0xFF, b"data")
        assert "unknown type 0xFF" in str(exc.value)


class TestTryDecodeMessage:
    """Tests for try_decode_message function."""

    def test_returns_decoded_for_known_types(self) -> None:
        """Returns decoded message for known types."""
        sync_data = b""
        result = try_decode_message(MSG_SYNC, sync_data)
        # Use decode_message which never returns None for known types
        # to verify the dispatch works correctly
        expected = decode_message(MSG_SYNC, sync_data)
        assert result == expected

    def test_returns_none_for_unknown_types(self) -> None:
        """Returns None for unknown message types."""
        result = try_decode_message(0xFF, b"data")
        assert result is None


class TestTryDecodeBinaryMessage:
    """Tests for try_decode_binary_message function."""

    def test_returns_decoded_for_known_types(self) -> None:
        """Returns decoded message for known types."""
        sync_data = b""
        result = try_decode_binary_message(MSG_SYNC, sync_data)
        # Use decode_message which never returns None for known types
        # to verify the dispatch works correctly
        expected = decode_message(MSG_SYNC, sync_data)
        assert result == expected

    def test_returns_none_for_unknown_types(self) -> None:
        """Returns None for unknown message types."""
        result = try_decode_binary_message(0xFF, b"data")
        assert result is None

    def test_returns_combat_message(self) -> None:
        """Returns decoded combat message (MSG_SHOOT)."""
        shoot_data = bytes([0x02, 0x01, 10, 20, 15, 25, 0x03, 0x04, 0x05, 1, 5, 0])
        result = try_decode_binary_message(MSG_SHOOT, shoot_data)
        expected = decode_message(MSG_SHOOT, shoot_data)
        assert result == expected

    def test_returns_resource_message(self) -> None:
        """Returns decoded resource message (MSG_FUEL_GAIN)."""
        fuel_data = bytes([0x34, 0x12, 1])
        result = try_decode_binary_message(MSG_FUEL_GAIN, fuel_data)
        expected = decode_message(MSG_FUEL_GAIN, fuel_data)
        assert result == expected

    def test_returns_radar_message(self) -> None:
        """Returns decoded radar message (MSG_RADAR_RESULT)."""
        radar_data = bytes([3, 1])
        result = try_decode_binary_message(MSG_RADAR_RESULT, radar_data)
        expected = decode_message(MSG_RADAR_RESULT, radar_data)
        assert result == expected

    def test_returns_tank_message(self) -> None:
        """Returns decoded tank message (MSG_TANK_ENTRY)."""
        entry_data = bytes([5, 0x02, 0x01, 60, 0, 0, 0, 0, 0, 0])
        result = try_decode_binary_message(MSG_TANK_ENTRY, entry_data)
        expected = decode_message(MSG_TANK_ENTRY, entry_data)
        assert result == expected

    def test_returns_movement_message(self) -> None:
        """Returns decoded movement message (MSG_MOVEMENT)."""
        movement_data = bytes([0x02, 0x01, 50, 60, 3, 1, 0x03, 0x04, 0x05])
        result = try_decode_binary_message(MSG_MOVEMENT, movement_data)
        expected = decode_message(MSG_MOVEMENT, movement_data)
        assert result == expected

    def test_returns_misc_message(self) -> None:
        """Returns decoded misc message (MSG_CHAT)."""
        chat_data = bytes([0x02, 0x01, 1])
        result = try_decode_binary_message(MSG_CHAT, chat_data)
        expected = decode_message(MSG_CHAT, chat_data)
        assert result == expected


# =============================================================================
# Message Constants Tests
# =============================================================================


class TestMessageConstants:
    """Tests for message type constants."""

    def test_msg_constants_are_ascii_ord(self) -> None:
        """Message constants match ASCII ordinal values."""
        assert ord("S") == MSG_SHOOT
        assert ord("G") == MSG_MOVEMENT
        assert ord("A") == MSG_DEACTIVATE
        assert ord("?") == MSG_SYNC
        assert ord("!") == MSG_TANK_INFO
        assert ord("C") == MSG_CONTAINER
        assert ord("Z") == MSG_VIEWPORT

    def test_supervisor_status_constants(self) -> None:
        """Supervisor status constants have expected values."""
        assert SUPERVISOR_STATUS_PROMO_ELIGIBLE == 1
        assert SUPERVISOR_STATUS_PROMO_KILL == 8
