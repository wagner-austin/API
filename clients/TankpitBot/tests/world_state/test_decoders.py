"""Tests for state decode functions."""

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.state import (
    add_mine,
    decode_container_state,
    decode_mine_state,
    decode_self_state,
    decode_tank_state,
    decode_terrain_tile,
    decode_viewport_state,
    decode_world_state,
    encode_world_state,
    make_empty_world_state,
    update_container_from_radar,
)


class TestDecodeTankState:
    """Tests for decode_tank_state."""

    def test_decodes_valid_data(self) -> None:
        """Decodes valid tank state data."""
        data: JSONObject = {
            "tank_id": 42,
            "x": 100,
            "y": 150,
            "team": 2,
            "rank": 3,
            "damage_state": 1,
            "name": "Test",
            "is_bot": True,
            "is_self": False,
            "timestamp_ms": 5000,
        }
        tank = decode_tank_state(data)

        assert tank["tank_id"] == 42
        assert tank["x"] == 100
        assert tank["y"] == 150
        assert tank["team"] == 2
        assert tank["rank"] == 3
        assert tank["damage_state"] == 1
        assert tank["timestamp_ms"] == 5000
        assert tank["name"] == "Test"
        assert tank["is_bot"] is True
        assert tank["is_self"] is False

    def test_missing_field_raises(self) -> None:
        """Raises JSONTypeError for missing field."""
        data: JSONObject = {"tank_id": 42}  # Missing other fields
        with pytest.raises(JSONTypeError):
            decode_tank_state(data)


class TestDecodeContainerState:
    """Tests for decode_container_state."""

    def test_decodes_valid_data(self) -> None:
        """Decodes valid container state data."""
        data: JSONObject = {
            "x": 50,
            "y": 75,
            "is_fuel": True,
            "volume": 300,
            "timestamp_ms": 5000,
            "failed_pickups": 0,
        }
        container = decode_container_state(data)

        assert container["x"] == 50
        assert container["y"] == 75
        assert container["is_fuel"] is True
        assert container["volume"] == 300
        assert container["timestamp_ms"] == 5000
        assert container["failed_pickups"] == 0


class TestDecodeMineState:
    """Tests for decode_mine_state."""

    def test_decodes_valid_data(self) -> None:
        """Decodes valid mine state data."""
        data: JSONObject = {"x": 25, "y": 35, "mine_type": 2, "tank_id": 99, "team": 1}
        mine = decode_mine_state(data)

        assert mine["x"] == 25
        assert mine["y"] == 35
        assert mine["mine_type"] == 2
        assert mine["tank_id"] == 99
        assert mine["team"] == 1


class TestDecodeTerrainTile:
    """Tests for decode_terrain_tile."""

    def test_decodes_valid_data(self) -> None:
        """Decodes valid terrain tile data."""
        data: JSONObject = {
            "x": 10,
            "y": 20,
            "terrain_type": 3,
            "cache_value": 5,
            "overlay_value": 255,
        }
        tile = decode_terrain_tile(data)

        assert tile["x"] == 10
        assert tile["y"] == 20
        assert tile["terrain_type"] == 3
        assert tile["cache_value"] == 5
        assert tile["overlay_value"] == 255


class TestDecodeViewportState:
    """Tests for decode_viewport_state."""

    def test_decodes_valid_data(self) -> None:
        """Decodes valid viewport state data."""
        data: JSONObject = {"left": 100, "top": 50, "width": 18, "height": 18}
        viewport = decode_viewport_state(data)

        assert viewport["left"] == 100
        assert viewport["top"] == 50
        assert viewport["width"] == 18
        assert viewport["height"] == 18


class TestDecodeSelfState:
    """Tests for decode_self_state."""

    def test_decodes_valid_data(self) -> None:
        """Decodes valid self state data."""
        data: JSONObject = {
            "tank_id": 1,
            "x": 128,
            "y": 128,
            "team": 0,
            "rank": 4,
            "fuel": 500,
            "leaderboard_position": 5,
        }
        self_state = decode_self_state(data)

        assert self_state["tank_id"] == 1
        assert self_state["x"] == 128
        assert self_state["y"] == 128
        assert self_state["team"] == 0
        assert self_state["rank"] == 4
        assert self_state["fuel"] == 500
        assert self_state["leaderboard_position"] == 5


class TestDecodeWorldState:
    """Tests for decode_world_state."""

    def test_decodes_empty_state(self) -> None:
        """Decodes empty world state."""
        data: JSONObject = {
            "self_state": None,
            "tanks": {},
            "containers": {},
            "mines": {},
            "terrain": {},
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "scanned_viewports": {},
            "timestamp_ms": 0,
        }
        state = decode_world_state(data)

        assert state["self_state"] is None
        assert state["tanks"] == {}
        assert state["containers"] == {}
        assert state["mines"] == {}
        assert state["terrain"] == {}
        assert state["timestamp_ms"] == 0

    def test_round_trip_encoding(self) -> None:
        """Encode then decode returns equivalent state."""
        original = make_empty_world_state()
        original = update_container_from_radar(original, 100, 100, 500, 1000)
        original = add_mine(original, 50, 50, 1, 42, team=0, timestamp_ms=2000)

        encoded = encode_world_state(original)
        decoded = decode_world_state(encoded)

        assert decoded["containers"]["100,100"]["volume"] == 500
        assert decoded["mines"]["50,50"]["mine_type"] == 1
        assert decoded["timestamp_ms"] == 2000

    def test_decodes_with_valid_self_state(self) -> None:
        """Decodes world state with valid self_state dict."""
        data: JSONObject = {
            "self_state": {
                "tank_id": 1,
                "x": 100,
                "y": 150,
                "team": 2,
                "rank": 3,
                "fuel": 750,
                "leaderboard_position": 5,
            },
            "tanks": {},
            "containers": {},
            "mines": {},
            "terrain": {},
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "scanned_viewports": {},
            "timestamp_ms": 1000,
        }
        state = decode_world_state(data)

        self_state = state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["tank_id"] == 1
        assert self_state["x"] == 100
        assert self_state["y"] == 150
        assert self_state["team"] == 2
        assert self_state["rank"] == 3
        assert self_state["fuel"] == 750
        assert self_state["leaderboard_position"] == 5

    def test_raises_on_invalid_viewport(self) -> None:
        """Raises JSONTypeError when viewport is not a dict."""
        data: JSONObject = {
            "self_state": None,
            "tanks": {},
            "containers": {},
            "mines": {},
            "terrain": {},
            "viewport": "invalid",
            "scanned_viewports": {},
            "timestamp_ms": 0,
        }
        with pytest.raises(JSONTypeError, match="viewport must be an object"):
            decode_world_state(data)

    def test_handles_non_dict_tanks_value(self) -> None:
        """Skips non-dict values in tanks field."""
        data: JSONObject = {
            "self_state": None,
            "tanks": {"42": "not_a_dict", "43": None},
            "containers": {},
            "mines": {},
            "terrain": {},
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "scanned_viewports": {},
            "timestamp_ms": 0,
        }
        state = decode_world_state(data)
        assert state["tanks"] == {}

    def test_handles_non_dict_containers_value(self) -> None:
        """Skips non-dict values in containers field."""
        data: JSONObject = {
            "self_state": None,
            "tanks": {},
            "containers": {"100,100": "not_a_dict", "50,50": 123},
            "mines": {},
            "terrain": {},
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "scanned_viewports": {},
            "timestamp_ms": 0,
        }
        state = decode_world_state(data)
        assert state["containers"] == {}

    def test_handles_non_dict_mines_value(self) -> None:
        """Skips non-dict values in mines field."""
        data: JSONObject = {
            "self_state": None,
            "tanks": {},
            "containers": {},
            "mines": {"75,125": ["not", "a", "dict"], "25,35": True},
            "terrain": {},
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "scanned_viewports": {},
            "timestamp_ms": 0,
        }
        state = decode_world_state(data)
        assert state["mines"] == {}

    def test_handles_non_dict_terrain_value(self) -> None:
        """Skips non-dict values in terrain field."""
        data: JSONObject = {
            "self_state": None,
            "tanks": {},
            "containers": {},
            "mines": {},
            "terrain": {"10,20": 12345, "15,25": False},
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "scanned_viewports": {},
            "timestamp_ms": 0,
        }
        state = decode_world_state(data)
        assert state["terrain"] == {}

    def test_handles_non_dict_field_values(self) -> None:
        """Handles when field values themselves are not dicts."""
        data: JSONObject = {
            "self_state": None,
            "tanks": "not_a_dict",
            "containers": 123,
            "mines": ["list", "not", "dict"],
            "terrain": None,
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "scanned_viewports": {},
            "timestamp_ms": 0,
        }
        state = decode_world_state(data)
        assert state["tanks"] == {}
        assert state["containers"] == {}
        assert state["mines"] == {}
        assert state["terrain"] == {}

    def test_decodes_with_tanks(self) -> None:
        """Decodes world state with tanks."""
        data: JSONObject = {
            "self_state": None,
            "tanks": {
                "42": {
                    "tank_id": 42,
                    "x": 100,
                    "y": 150,
                    "team": 0,
                    "rank": 2,
                    "damage_state": 0,
                    "name": "TestTank",
                    "is_bot": False,
                    "is_self": False,
                    "timestamp_ms": 500,
                },
            },
            "containers": {},
            "mines": {},
            "terrain": {},
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "scanned_viewports": {},
            "timestamp_ms": 1000,
        }
        state = decode_world_state(data)
        assert "42" in state["tanks"]
        tank = state["tanks"]["42"]
        assert tank["tank_id"] == 42
        assert tank["x"] == 100
        assert tank["name"] == "TestTank"
        assert tank["timestamp_ms"] == 500

    def test_decodes_with_terrain(self) -> None:
        """Decodes world state with terrain tiles."""
        data: JSONObject = {
            "self_state": None,
            "tanks": {},
            "containers": {},
            "mines": {},
            "terrain": {
                "10,20": {
                    "x": 10,
                    "y": 20,
                    "terrain_type": 1,
                    "cache_value": 0,
                    "overlay_value": 255,
                },
            },
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "scanned_viewports": {"0,0": 1234},
            "timestamp_ms": 1000,
        }
        state = decode_world_state(data)
        assert "10,20" in state["terrain"]
        tile = state["terrain"]["10,20"]
        assert tile["x"] == 10
        assert tile["y"] == 20
        assert tile["terrain_type"] == 1
        assert tile["cache_value"] == 0
        assert tile["overlay_value"] == 255
        assert state["scanned_viewports"] == {"0,0": 1234}

    def test_raises_on_non_integer_scanned_viewport_timestamp(self) -> None:
        """Raises JSONTypeError for non-integer scanned viewport timestamps."""
        data: JSONObject = {
            "self_state": None,
            "tanks": {},
            "containers": {},
            "mines": {},
            "terrain": {},
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "scanned_viewports": {"0,0": True},
            "timestamp_ms": 0,
        }

        with pytest.raises(JSONTypeError, match=r"scanned_viewports\.0,0 must be an integer"):
            decode_world_state(data)
