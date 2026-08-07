"""Tests for radar message decoders.

Tests for radar result, enemy detection, radar scan result decoders and validators.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.protocol import (
    RadarContainerDict,
    RadarMineClearDict,
    RadarMineDict,
    RadarScanResultDict,
    decode_enemy_detection,
    decode_radar_result,
    decode_radar_scan_result,
    encode_radar_scan_result,
    require_radar_container,
    require_radar_mine,
    require_radar_mine_clear,
    require_radar_scan_result,
)
from tankpit_bot.wire.helpers import DecodeError


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
        """Decodes enemy detection per trace-verified Tg.h: x,y,team,rank,tank_id."""
        # x=50, y=60, team=2, rank=4, tank_id=0x0102
        data = bytes([50, 60, 2, 4, 0x02, 0x01])
        result = decode_enemy_detection(data)
        assert result["msg_type"] == 0x48
        assert result["x"] == 50
        assert result["y"] == 60
        assert result["team"] == 2
        assert result["rank"] == 4
        assert result["tank_id"] == 0x0102

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_enemy_detection(bytes([1, 2, 3]))


class TestDecodeRadarScanResult:
    """Tests for decode_radar_scan_result function."""

    def test_decodes_radar_scan_containers(self) -> None:
        """Decodes radar scan with container entries."""
        # count=2, flags=0, then 4-byte containers (x, y, value_lo, value_hi)
        data = bytes([2, 0, 10, 20, 0x34, 0x12, 30, 40, 0xFF, 0x7F])
        result = decode_radar_scan_result(data)
        assert result["msg_type"] == 0x4F
        assert len(result["containers"]) == 2
        assert result["containers"][0] == {"x": 10, "y": 20, "volume": 0x1234}
        assert result["containers"][1] == {"x": 30, "y": 40, "volume": 0x7FFF}
        assert result["mines"] == []

    def test_decodes_equipment_container(self) -> None:
        """Decodes equipment container (0xFFFF value)."""
        data = bytes([1, 0, 10, 20, 0xFF, 0xFF])  # 0xFFFF -> equipment (-1)
        result = decode_radar_scan_result(data)
        assert result["containers"][0]["volume"] == -1

    def test_decodes_radar_scan_with_mines(self) -> None:
        """Decodes radar scan with containers and mines."""
        # 1 container, then 2 mines (3 bytes each: x, y, team)
        data = bytes([1, 0, 10, 20, 0x00, 0x00, 45, 203, 0, 46, 203, 0])
        result = decode_radar_scan_result(data)
        assert result["msg_type"] == 0x4F
        assert len(result["containers"]) == 1
        assert result["containers"][0] == {"x": 10, "y": 20, "volume": 0}
        assert len(result["mines"]) == 2
        assert result["mines"][0] == {"x": 45, "y": 203, "team": 0}
        assert result["mines"][1] == {"x": 46, "y": 203, "team": 0}

    def test_decodes_mines_all_teams(self) -> None:
        """Decodes mines from all teams."""
        # 0 containers, 4 mines (one of each team)
        data = bytes([0, 0, 10, 10, 0, 20, 20, 1, 30, 30, 2, 40, 40, 3])
        result = decode_radar_scan_result(data)
        assert result["containers"] == []
        assert len(result["mines"]) == 4
        assert result["mines"][0] == {"x": 10, "y": 10, "team": 0}  # red
        assert result["mines"][1] == {"x": 20, "y": 20, "team": 1}  # purple
        assert result["mines"][2] == {"x": 30, "y": 30, "team": 2}  # blue
        assert result["mines"][3] == {"x": 40, "y": 40, "team": 3}  # orange
        assert result["mine_clears"] == []

    def test_decodes_container_removal_entry(self) -> None:
        """A cache entry with value 0 decodes as a volume-0 removal.

        Corpus scan 2026-07-03 (199 sessions): 247 of 2093 cache
        entries carried value 0 -- the server's "tile now empty"
        statement, applied downstream as authoritative removal.
        """
        data = bytes([1, 0, 12, 34, 0, 0])
        result = decode_radar_scan_result(data)
        assert result["containers"] == [{"x": 12, "y": 34, "volume": 0}]

    def test_decodes_overlay_clear_as_mine_clear(self) -> None:
        """Overlay values >= 8 decode as mine clears, not phantom mines.

        JS ch writes the overlay byte into tile.m raw and 255 is the
        canonical no-mine sentinel (dh detonation handler); the old
        decoder misread a 255 entry as a mine with team=255.
        """
        data = bytes([0, 0, 50, 60, 255, 70, 80, 8])
        result = decode_radar_scan_result(data)
        assert result["mines"] == []
        assert result["mine_clears"] == [{"x": 50, "y": 60}, {"x": 70, "y": 80}]

    def test_decodes_mine_variant_team_from_low_bits(self) -> None:
        """Overlay values 4-7 decode as mines with team = value & 3."""
        data = bytes([0, 0, 15, 25, 5])
        result = decode_radar_scan_result(data)
        assert result["mines"] == [{"x": 15, "y": 25, "team": 1}]
        assert result["mine_clears"] == []

    def test_decodes_count_high_byte(self) -> None:
        """The container count is a LE u16, not count-byte plus flags.

        JS ch.h reads ``X(a[0], a[1])``; the old decoder treated byte 1
        as an always-zero flags byte.
        """
        data = bytes([1, 0, 12, 34, 100, 0])
        result = decode_radar_scan_result(data)
        assert result["containers"] == [{"x": 12, "y": 34, "volume": 100}]

    def test_raises_on_truncated_container(self) -> None:
        """Raises DecodeError on truncated container data."""
        data = bytes([2, 0, 10, 20, 0x34])  # Claims 2 but only partial first
        with pytest.raises(DecodeError):
            decode_radar_scan_result(data)

    def test_raises_on_invalid_mine_bytes(self) -> None:
        """Raises DecodeError when remaining bytes not divisible by 3."""
        # 1 container (correct), then 2 bytes (not divisible by 3)
        data = bytes([1, 0, 10, 20, 0x00, 0x00, 45, 203])
        with pytest.raises(DecodeError):
            decode_radar_scan_result(data)

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_radar_scan_result(bytes([1]))

    def test_encode_decode_roundtrip(self) -> None:
        """Encode and decode produces equivalent result."""
        original = RadarScanResultDict(
            msg_type=0x4F,
            containers=[
                RadarContainerDict(x=44, y=208, volume=-1),  # equipment
                RadarContainerDict(x=53, y=215, volume=501),  # fuel
            ],
            mines=[
                RadarMineDict(x=45, y=203, team=0),  # red
                RadarMineDict(x=46, y=203, team=0),  # red
                RadarMineDict(x=47, y=203, team=0),  # red
            ],
            mine_clears=[
                RadarMineClearDict(x=48, y=204),
            ],
        )
        encoded = encode_radar_scan_result(original)
        decoded = decode_radar_scan_result(encoded)

        assert decoded["msg_type"] == original["msg_type"]
        assert len(decoded["containers"]) == 2
        assert decoded["containers"][0]["x"] == 44
        assert decoded["containers"][0]["y"] == 208
        assert decoded["containers"][0]["volume"] == -1
        assert decoded["containers"][1]["volume"] == 501
        assert len(decoded["mines"]) == 3
        assert decoded["mines"][0] == {"x": 45, "y": 203, "team": 0}
        assert decoded["mines"][1] == {"x": 46, "y": 203, "team": 0}
        assert decoded["mines"][2] == {"x": 47, "y": 203, "team": 0}
        assert decoded["mine_clears"] == [{"x": 48, "y": 204}]

    def test_encode_empty_result(self) -> None:
        """Encodes empty radar result."""
        result = RadarScanResultDict(msg_type=0x4F, containers=[], mines=[], mine_clears=[])
        encoded = encode_radar_scan_result(result)
        assert encoded == bytes([0, 0])


class TestRequireRadarContainer:
    """Tests for require_radar_container validation."""

    def test_validates_valid_container(self) -> None:
        """Validates valid container."""
        container: JSONObject = {"x": 100, "y": 150, "volume": 500}
        result = require_radar_container(container)
        assert result["x"] == 100
        assert result["y"] == 150
        assert result["volume"] == 500

    def test_validates_equipment_container(self) -> None:
        """Validates equipment container with -1 volume."""
        container: JSONObject = {"x": 44, "y": 208, "volume": -1}
        result = require_radar_container(container)
        assert result["volume"] == -1

    def test_raises_on_invalid_x(self) -> None:
        """Raises ValueError for x out of range."""
        container: JSONObject = {"x": 256, "y": 100, "volume": 0}
        with pytest.raises(ValueError, match="x out of range"):
            require_radar_container(container)

    def test_raises_on_invalid_volume(self) -> None:
        """Raises ValueError for volume out of range."""
        container: JSONObject = {"x": 0, "y": 0, "volume": -2}
        with pytest.raises(ValueError, match="volume out of range"):
            require_radar_container(container)

    def test_raises_on_y_out_of_range(self) -> None:
        """Raises ValueError for y out of range."""
        container: JSONObject = {"x": 0, "y": 256, "volume": 0}
        with pytest.raises(ValueError, match="y out of range"):
            require_radar_container(container)

    def test_raises_on_missing_x(self) -> None:
        """Raises ValueError when x is missing."""
        container: JSONObject = {"y": 0, "volume": 0}
        with pytest.raises(ValueError, match="x must be int"):
            require_radar_container(container)

    def test_raises_on_missing_y(self) -> None:
        """Raises ValueError when y is missing."""
        container: JSONObject = {"x": 0, "volume": 0}
        with pytest.raises(ValueError, match="y must be int"):
            require_radar_container(container)

    def test_raises_on_missing_volume(self) -> None:
        """Raises ValueError when volume is missing."""
        container: JSONObject = {"x": 0, "y": 0}
        with pytest.raises(ValueError, match="volume must be int"):
            require_radar_container(container)


class TestRequireRadarMine:
    """Tests for require_radar_mine validation."""

    def test_validates_valid_mine(self) -> None:
        """Validates valid mine."""
        mine: JSONObject = {"x": 45, "y": 203, "team": 0}
        result = require_radar_mine(mine)
        assert result["x"] == 45
        assert result["y"] == 203
        assert result["team"] == 0

    def test_validates_all_teams(self) -> None:
        """Validates mines from all teams."""
        for team in range(4):
            mine: JSONObject = {"x": 0, "y": 0, "team": team}
            result = require_radar_mine(mine)
            assert result["team"] == team

    def test_raises_on_invalid_team(self) -> None:
        """Raises ValueError for team out of range."""
        mine: JSONObject = {"x": 0, "y": 0, "team": 4}
        with pytest.raises(ValueError, match="team out of range"):
            require_radar_mine(mine)

    def test_raises_on_invalid_coordinates(self) -> None:
        """Raises ValueError for coordinates out of range."""
        mine: JSONObject = {"x": 256, "y": 0, "team": 0}
        with pytest.raises(ValueError, match="x out of range"):
            require_radar_mine(mine)

    def test_raises_on_y_out_of_range(self) -> None:
        """Raises ValueError for y out of range."""
        mine: JSONObject = {"x": 0, "y": 256, "team": 0}
        with pytest.raises(ValueError, match="y out of range"):
            require_radar_mine(mine)

    def test_raises_on_missing_x(self) -> None:
        """Raises ValueError when x is missing."""
        mine: JSONObject = {"y": 0, "team": 0}
        with pytest.raises(ValueError, match="x must be int"):
            require_radar_mine(mine)

    def test_raises_on_missing_y(self) -> None:
        """Raises ValueError when y is missing."""
        mine: JSONObject = {"x": 0, "team": 0}
        with pytest.raises(ValueError, match="y must be int"):
            require_radar_mine(mine)

    def test_raises_on_missing_team(self) -> None:
        """Raises ValueError when team is missing."""
        mine: JSONObject = {"x": 0, "y": 0}
        with pytest.raises(ValueError, match="team must be int"):
            require_radar_mine(mine)


class TestRequireRadarMineClear:
    """Tests for require_radar_mine_clear validation."""

    def test_validates_valid_clear(self) -> None:
        """Validates valid mine-clear entry."""
        clear: JSONObject = {"x": 48, "y": 204}
        result = require_radar_mine_clear(clear)
        assert result["x"] == 48
        assert result["y"] == 204

    def test_raises_on_missing_x(self) -> None:
        """Raises ValueError when x is missing."""
        clear: JSONObject = {"y": 0}
        with pytest.raises(ValueError, match="x must be int"):
            require_radar_mine_clear(clear)

    def test_raises_on_missing_y(self) -> None:
        """Raises ValueError when y is missing."""
        clear: JSONObject = {"x": 0}
        with pytest.raises(ValueError, match="y must be int"):
            require_radar_mine_clear(clear)

    def test_raises_on_x_out_of_range(self) -> None:
        """Raises ValueError for x out of range."""
        clear: JSONObject = {"x": 256, "y": 0}
        with pytest.raises(ValueError, match="x out of range"):
            require_radar_mine_clear(clear)

    def test_raises_on_y_out_of_range(self) -> None:
        """Raises ValueError for y out of range."""
        clear: JSONObject = {"x": 0, "y": 256}
        with pytest.raises(ValueError, match="y out of range"):
            require_radar_mine_clear(clear)


class TestRequireRadarScanResult:
    """Tests for require_radar_scan_result validation."""

    def test_validates_valid_result(self) -> None:
        """Validates valid radar scan result."""
        result: JSONObject = {
            "msg_type": 0x4F,
            "containers": [{"x": 10, "y": 20, "volume": 100}],
            "mines": [{"x": 45, "y": 203, "team": 0}],
            "mine_clears": [{"x": 48, "y": 204}],
        }
        validated = require_radar_scan_result(result)
        assert validated["msg_type"] == 0x4F
        assert len(validated["containers"]) == 1
        assert len(validated["mines"]) == 1
        assert len(validated["mine_clears"]) == 1

    def test_raises_on_invalid_container(self) -> None:
        """Raises ValueError for invalid container in result."""
        result: JSONObject = {
            "msg_type": 0x4F,
            "containers": [{"x": 256, "y": 0, "volume": 0}],
            "mines": [],
            "mine_clears": [],
        }
        with pytest.raises(ValueError, match="container"):
            require_radar_scan_result(result)

    def test_raises_on_invalid_mine(self) -> None:
        """Raises ValueError for invalid mine in result."""
        result: JSONObject = {
            "msg_type": 0x4F,
            "containers": [],
            "mines": [{"x": 0, "y": 0, "team": 5}],
            "mine_clears": [],
        }
        with pytest.raises(ValueError, match="mine"):
            require_radar_scan_result(result)

    def test_raises_on_invalid_mine_clear(self) -> None:
        """Raises ValueError for invalid mine-clear in result."""
        result: JSONObject = {
            "msg_type": 0x4F,
            "containers": [],
            "mines": [],
            "mine_clears": [{"x": 256, "y": 0}],
        }
        with pytest.raises(ValueError, match=r"mine_clear\[0\]"):
            require_radar_scan_result(result)

    def test_raises_on_wrong_msg_type(self) -> None:
        """Raises ValueError for wrong msg_type."""
        result: JSONObject = {
            "msg_type": 0x00,
            "containers": [],
            "mines": [],
            "mine_clears": [],
        }
        with pytest.raises(ValueError, match="msg_type must be 0x4F"):
            require_radar_scan_result(result)

    def test_raises_on_containers_not_list(self) -> None:
        """Raises ValueError when containers is not a list."""
        result: JSONObject = {
            "msg_type": 0x4F,
            "containers": None,
            "mines": [],
            "mine_clears": [],
        }
        with pytest.raises(ValueError, match="containers must be list"):
            require_radar_scan_result(result)

    def test_raises_on_mines_not_list(self) -> None:
        """Raises ValueError when mines is not a list."""
        result: JSONObject = {
            "msg_type": 0x4F,
            "containers": [],
            "mines": "not a list",
            "mine_clears": [],
        }
        with pytest.raises(ValueError, match="mines must be list"):
            require_radar_scan_result(result)

    def test_raises_on_mine_clears_not_list(self) -> None:
        """Raises ValueError when mine_clears is not a list."""
        result: JSONObject = {
            "msg_type": 0x4F,
            "containers": [],
            "mines": [],
            "mine_clears": "not a list",
        }
        with pytest.raises(ValueError, match="mine_clears must be list"):
            require_radar_scan_result(result)

    def test_raises_on_container_not_dict(self) -> None:
        """Raises ValueError when container item is not a dict."""
        result: JSONObject = {
            "msg_type": 0x4F,
            "containers": ["not a dict"],
            "mines": [],
            "mine_clears": [],
        }
        with pytest.raises(ValueError, match=r"container\[0\] must be dict"):
            require_radar_scan_result(result)

    def test_raises_on_mine_not_dict(self) -> None:
        """Raises ValueError when mine item is not a dict."""
        result: JSONObject = {
            "msg_type": 0x4F,
            "containers": [],
            "mines": [123],
            "mine_clears": [],
        }
        with pytest.raises(ValueError, match=r"mine\[0\] must be dict"):
            require_radar_scan_result(result)

    def test_raises_on_mine_clear_not_dict(self) -> None:
        """Raises ValueError when mine-clear item is not a dict."""
        result: JSONObject = {
            "msg_type": 0x4F,
            "containers": [],
            "mines": [],
            "mine_clears": [123],
        }
        with pytest.raises(ValueError, match=r"mine_clear\[0\] must be dict"):
            require_radar_scan_result(result)


class TestDecodeRadarContainerNotEnoughBytes:
    """Tests for decode_radar_container error handling."""

    def test_raises_on_not_enough_bytes(self) -> None:
        """Raises DecodeError when not enough bytes."""
        from tankpit_bot.protocol import decode_radar_container
        from tankpit_bot.wire.helpers import DecodeError

        data = bytes([0x00, 0x00, 0x00])  # Only 3 bytes, need 4
        with pytest.raises(DecodeError, match="not enough bytes"):
            decode_radar_container(data, 0)
