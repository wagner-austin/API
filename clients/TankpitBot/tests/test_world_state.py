"""Tests for state module."""

from pathlib import Path

import pytest
from PIL import Image
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.state import (
    ASCII_ALLY,
    ASCII_ENEMY,
    ASCII_EQUIPMENT,
    ASCII_FERRY,
    ASCII_FUEL,
    ASCII_GROUND,
    ASCII_MINE,
    ASCII_ROCK,
    ASCII_SELF,
    ASCII_UNKNOWN,
    ASCII_WATER,
    DAMAGE_CRITICAL,
    DAMAGE_FULL,
    DAMAGE_LIGHT,
    DAMAGE_MEDIUM,
    TEAM_BLUE,
    TEAM_ORANGE,
    TEAM_PURPLE,
    TEAM_RED,
    TERRAIN_FERRY,
    TERRAIN_FERRY_ROCK,
    TERRAIN_GROUND,
    TERRAIN_ROCK_A,
    TERRAIN_ROCK_AB,
    TERRAIN_ROCK_B,
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    add_mine,
    coord_key,
    decode_container_state,
    decode_mine_state,
    decode_self_state,
    decode_tank_state,
    decode_terrain_tile,
    decode_viewport_state,
    decode_world_state,
    encode_container_state,
    encode_mine_state,
    encode_self_state,
    encode_tank_state,
    encode_terrain_tile,
    encode_viewport_state,
    encode_world_state,
    make_container_state,
    make_empty_world_state,
    make_mine_state,
    make_self_state,
    make_tank_state,
    make_terrain_tile,
    parse_coord_key,
    remove_container,
    remove_mine,
    remove_tank,
    render_world_ascii,
    terrain_to_ascii,
    update_container_from_radar,
    update_self_from_movement_response,
    update_tank_damage,
    update_tank_from_registry,
    update_terrain_from_viewport,
)
from tankpit_bot.terrain import TerrainMap


def get_self_state(state: WorldStateDict) -> SelfStateDict:
    """Extract self_state from world state, raising if None.

    Test helper for type narrowing.
    """
    result = state["self_state"]
    if result is None:
        raise AssertionError("self_state is None")
    return result


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_terrain_constants(self) -> None:
        """Verify terrain type constants."""
        assert TERRAIN_GROUND == 0
        assert TERRAIN_ROCK_A == 1
        assert TERRAIN_ROCK_B == 2
        assert TERRAIN_ROCK_AB == 3
        assert TERRAIN_FERRY == 5
        assert TERRAIN_FERRY_ROCK == 7

    def test_team_constants(self) -> None:
        """Verify team ID constants."""
        assert TEAM_RED == 0
        assert TEAM_PURPLE == 1
        assert TEAM_BLUE == 2
        assert TEAM_ORANGE == 3

    def test_damage_constants(self) -> None:
        """Verify damage state constants."""
        assert DAMAGE_FULL == 0
        assert DAMAGE_LIGHT == 1
        assert DAMAGE_MEDIUM == 2
        assert DAMAGE_CRITICAL == 3

    def test_ascii_constants(self) -> None:
        """Verify ASCII character constants."""
        assert ASCII_GROUND == "."
        assert ASCII_ROCK == "#"
        assert ASCII_FERRY == "~"
        assert ASCII_WATER == "W"
        assert ASCII_FUEL == "F"
        assert ASCII_EQUIPMENT == "E"
        assert ASCII_MINE == "*"
        assert ASCII_SELF == "@"
        assert ASCII_ENEMY == "T"
        assert ASCII_ALLY == "A"
        assert ASCII_UNKNOWN == "?"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestMakeEmptyWorldState:
    """Tests for make_empty_world_state."""

    def test_creates_empty_state(self) -> None:
        """Creates valid empty world state."""
        state = make_empty_world_state()

        assert state["self_state"] is None
        assert state["tanks"] == {}
        assert state["containers"] == {}
        assert state["mines"] == {}
        assert state["terrain"] == {}
        assert state["timestamp_ms"] == 0

    def test_default_viewport(self) -> None:
        """Default viewport is 18x18 at origin."""
        state = make_empty_world_state()
        vp = state["viewport"]

        assert vp["left"] == 0
        assert vp["top"] == 0
        assert vp["width"] == 18
        assert vp["height"] == 18


class TestMakeTankState:
    """Tests for make_tank_state."""

    def test_creates_tank_state(self) -> None:
        """Creates tank state with all fields."""
        tank = make_tank_state(
            tank_id=42,
            x=100,
            y=150,
            team=TEAM_BLUE,
            rank=3,
            damage_state=DAMAGE_LIGHT,
            name="TestPlayer",
            is_bot=False,
            is_self=True,
        )

        assert tank["tank_id"] == 42
        assert tank["x"] == 100
        assert tank["y"] == 150
        assert tank["team"] == TEAM_BLUE
        assert tank["rank"] == 3
        assert tank["damage_state"] == DAMAGE_LIGHT
        assert tank["name"] == "TestPlayer"
        assert tank["is_bot"] is False
        assert tank["is_self"] is True

    def test_bot_tank(self) -> None:
        """Creates bot tank state."""
        tank = make_tank_state(
            tank_id=999,
            x=50,
            y=50,
            team=TEAM_RED,
            rank=0,
            damage_state=DAMAGE_FULL,
            name="Bot001",
            is_bot=True,
            is_self=False,
        )

        assert tank["is_bot"] is True
        assert tank["is_self"] is False


class TestMakeContainerState:
    """Tests for make_container_state."""

    def test_creates_fuel_container(self) -> None:
        """Creates fuel container state."""
        container = make_container_state(x=120, y=80, is_fuel=True, volume=500)

        assert container["x"] == 120
        assert container["y"] == 80
        assert container["is_fuel"] is True
        assert container["volume"] == 500

    def test_creates_equipment_container(self) -> None:
        """Creates equipment container state."""
        container = make_container_state(x=60, y=200, is_fuel=False, volume=0)

        assert container["is_fuel"] is False
        assert container["volume"] == 0


class TestMakeMineState:
    """Tests for make_mine_state."""

    def test_creates_mine_state(self) -> None:
        """Creates mine state with all fields."""
        mine = make_mine_state(x=75, y=125, mine_type=1, tank_id=42, team=0)

        assert mine["x"] == 75
        assert mine["y"] == 125
        assert mine["mine_type"] == 1
        assert mine["tank_id"] == 42
        assert mine["team"] == 0


class TestMakeTerrainTile:
    """Tests for make_terrain_tile."""

    def test_creates_terrain_tile(self) -> None:
        """Creates terrain tile with all fields."""
        tile = make_terrain_tile(x=10, y=20, terrain_type=TERRAIN_ROCK_A, entity_id=0)

        assert tile["x"] == 10
        assert tile["y"] == 20
        assert tile["terrain_type"] == TERRAIN_ROCK_A
        assert tile["entity_id"] == 0

    def test_tile_with_tank_entity(self) -> None:
        """Creates tile with tank entity marker."""
        tile = make_terrain_tile(x=15, y=25, terrain_type=TERRAIN_GROUND, entity_id=-1)

        assert tile["entity_id"] == -1


class TestMakeSelfState:
    """Tests for make_self_state."""

    def test_creates_self_state(self) -> None:
        """Creates self state with all fields."""
        self_state = make_self_state(
            tank_id=1,
            x=128,
            y=128,
            team=TEAM_PURPLE,
            rank=5,
            fuel=750,
            leaderboard_position=10,
        )

        assert self_state["tank_id"] == 1
        assert self_state["x"] == 128
        assert self_state["y"] == 128
        assert self_state["team"] == TEAM_PURPLE
        assert self_state["rank"] == 5
        assert self_state["fuel"] == 750
        assert self_state["leaderboard_position"] == 10


# =============================================================================
# Coordinate Key Tests
# =============================================================================


class TestCoordKey:
    """Tests for coord_key function."""

    def test_creates_key(self) -> None:
        """Creates comma-separated key."""
        assert coord_key(100, 200) == "100,200"

    def test_zero_coords(self) -> None:
        """Handles zero coordinates."""
        assert coord_key(0, 0) == "0,0"

    def test_max_coords(self) -> None:
        """Handles maximum coordinates."""
        assert coord_key(255, 255) == "255,255"


class TestParseCoordKey:
    """Tests for parse_coord_key function."""

    def test_parses_key(self) -> None:
        """Parses comma-separated key."""
        x, y = parse_coord_key("100,200")
        assert x == 100
        assert y == 200

    def test_zero_coords(self) -> None:
        """Parses zero coordinates."""
        x, y = parse_coord_key("0,0")
        assert x == 0
        assert y == 0

    def test_invalid_format_raises(self) -> None:
        """Raises ValueError for invalid format."""
        with pytest.raises(ValueError, match="Invalid coord key format"):
            parse_coord_key("invalid")

    def test_too_many_parts_raises(self) -> None:
        """Raises ValueError for too many parts."""
        with pytest.raises(ValueError):
            parse_coord_key("1,2,3")


# =============================================================================
# Encode Tests
# =============================================================================


class TestEncodeTankState:
    """Tests for encode_tank_state."""

    def test_encodes_all_fields(self) -> None:
        """Encodes all tank state fields."""
        tank = make_tank_state(
            tank_id=42,
            x=100,
            y=150,
            team=2,
            rank=3,
            damage_state=1,
            name="Test",
            is_bot=True,
            is_self=False,
        )
        encoded = encode_tank_state(tank)

        assert encoded["tank_id"] == 42
        assert encoded["x"] == 100
        assert encoded["y"] == 150
        assert encoded["team"] == 2
        assert encoded["rank"] == 3
        assert encoded["damage_state"] == 1
        assert encoded["name"] == "Test"
        assert encoded["is_bot"] is True
        assert encoded["is_self"] is False


class TestEncodeContainerState:
    """Tests for encode_container_state."""

    def test_encodes_all_fields(self) -> None:
        """Encodes all container state fields."""
        container = make_container_state(x=50, y=75, is_fuel=True, volume=300)
        encoded = encode_container_state(container)

        assert encoded["x"] == 50
        assert encoded["y"] == 75
        assert encoded["is_fuel"] is True
        assert encoded["volume"] == 300


class TestEncodeMineState:
    """Tests for encode_mine_state."""

    def test_encodes_all_fields(self) -> None:
        """Encodes all mine state fields."""
        mine = make_mine_state(x=25, y=35, mine_type=2, tank_id=99, team=1)
        encoded = encode_mine_state(mine)

        assert encoded["x"] == 25
        assert encoded["y"] == 35
        assert encoded["mine_type"] == 2
        assert encoded["tank_id"] == 99
        assert encoded["team"] == 1


class TestEncodeTerrainTile:
    """Tests for encode_terrain_tile."""

    def test_encodes_all_fields(self) -> None:
        """Encodes all terrain tile fields."""
        tile = make_terrain_tile(x=10, y=20, terrain_type=3, entity_id=5)
        encoded = encode_terrain_tile(tile)

        assert encoded["x"] == 10
        assert encoded["y"] == 20
        assert encoded["terrain_type"] == 3
        assert encoded["entity_id"] == 5


class TestEncodeViewportState:
    """Tests for encode_viewport_state."""

    def test_encodes_all_fields(self) -> None:
        """Encodes all viewport state fields."""
        viewport = ViewportStateDict(left=100, top=50, width=18, height=18)
        encoded = encode_viewport_state(viewport)

        assert encoded["left"] == 100
        assert encoded["top"] == 50
        assert encoded["width"] == 18
        assert encoded["height"] == 18


class TestEncodeSelfState:
    """Tests for encode_self_state."""

    def test_encodes_all_fields(self) -> None:
        """Encodes all self state fields."""
        self_state = make_self_state(
            tank_id=1, x=128, y=128, team=0, rank=4, fuel=500, leaderboard_position=5
        )
        encoded = encode_self_state(self_state)

        assert encoded["tank_id"] == 1
        assert encoded["x"] == 128
        assert encoded["y"] == 128
        assert encoded["team"] == 0
        assert encoded["rank"] == 4
        assert encoded["fuel"] == 500
        assert encoded["leaderboard_position"] == 5


class TestEncodeWorldState:
    """Tests for encode_world_state."""

    def test_encodes_empty_state(self) -> None:
        """Encodes empty world state."""
        state = make_empty_world_state()
        encoded = encode_world_state(state)

        assert encoded["self_state"] is None
        assert encoded["tanks"] == {}
        assert encoded["containers"] == {}
        assert encoded["mines"] == {}
        assert encoded["terrain"] == {}
        assert encoded["timestamp_ms"] == 0
        # Viewport has default values, not empty
        viewport = encoded["viewport"]
        if not isinstance(viewport, dict):
            raise AssertionError("viewport should be a dict")
        assert viewport["left"] == 0
        assert viewport["top"] == 0
        assert viewport["width"] == 18
        assert viewport["height"] == 18

    def test_encodes_populated_state(self) -> None:
        """Encodes world state with data."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, 100, 100, 500, 1000)
        state = add_mine(state, 50, 50, 1, 42, team=0, timestamp_ms=2000)

        encoded = encode_world_state(state)

        containers = encoded["containers"]
        mines = encoded["mines"]
        if not isinstance(containers, dict):
            raise AssertionError("containers should be a dict")
        if not isinstance(mines, dict):
            raise AssertionError("mines should be a dict")
        assert "100,100" in containers
        assert "50,50" in mines
        assert encoded["timestamp_ms"] == 2000


# =============================================================================
# Decode Tests
# =============================================================================


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
        }
        tank = decode_tank_state(data)

        assert tank["tank_id"] == 42
        assert tank["x"] == 100
        assert tank["y"] == 150
        assert tank["team"] == 2
        assert tank["rank"] == 3
        assert tank["damage_state"] == 1
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
        data: JSONObject = {"x": 50, "y": 75, "is_fuel": True, "volume": 300}
        container = decode_container_state(data)

        assert container["x"] == 50
        assert container["y"] == 75
        assert container["is_fuel"] is True
        assert container["volume"] == 300


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
        data: JSONObject = {"x": 10, "y": 20, "terrain_type": 3, "entity_id": 5}
        tile = decode_terrain_tile(data)

        assert tile["x"] == 10
        assert tile["y"] == 20
        assert tile["terrain_type"] == 3
        assert tile["entity_id"] == 5


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
                },
            },
            "containers": {},
            "mines": {},
            "terrain": {},
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "timestamp_ms": 1000,
        }
        state = decode_world_state(data)
        assert "42" in state["tanks"]
        tank = state["tanks"]["42"]
        assert tank["tank_id"] == 42
        assert tank["x"] == 100
        assert tank["name"] == "TestTank"

    def test_decodes_with_terrain(self) -> None:
        """Decodes world state with terrain tiles."""
        data: JSONObject = {
            "self_state": None,
            "tanks": {},
            "containers": {},
            "mines": {},
            "terrain": {
                "10,20": {"x": 10, "y": 20, "terrain_type": 1, "entity_id": 0},
            },
            "viewport": {"left": 0, "top": 0, "width": 18, "height": 18},
            "timestamp_ms": 1000,
        }
        state = decode_world_state(data)
        assert "10,20" in state["terrain"]
        tile = state["terrain"]["10,20"]
        assert tile["x"] == 10
        assert tile["y"] == 20
        assert tile["terrain_type"] == 1


# =============================================================================
# Update Function Tests
# =============================================================================


class TestUpdateSelfFromMovementResponse:
    """Tests for update_self_from_movement_response."""

    def test_creates_self_state(self) -> None:
        """Creates self state from movement response."""
        state = make_empty_world_state()
        updated = update_self_from_movement_response(
            state,
            tank_id=1,
            x=100,
            y=150,
            team=TEAM_BLUE,
            rank=3,
            leaderboard_position=5,
            timestamp_ms=1000,
        )

        self_state = get_self_state(updated)
        assert self_state["tank_id"] == 1
        assert self_state["x"] == 100
        assert self_state["y"] == 150
        assert self_state["team"] == TEAM_BLUE
        assert self_state["rank"] == 3
        assert self_state["leaderboard_position"] == 5
        assert updated["timestamp_ms"] == 1000

    def test_preserves_fuel(self) -> None:
        """Preserves existing fuel value."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=100, y=100, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        # Manually set fuel
        state = WorldStateDict(
            self_state=SelfStateDict(
                tank_id=1, x=100, y=100, team=0, rank=0, fuel=750, leaderboard_position=1
            ),
            tanks=state["tanks"],
            containers=state["containers"],
            mines=state["mines"],
            terrain=state["terrain"],
            viewport=state["viewport"],
            timestamp_ms=state["timestamp_ms"],
        )

        updated = update_self_from_movement_response(
            state,
            tank_id=1,
            x=110,
            y=110,
            team=0,
            rank=0,
            leaderboard_position=2,
            timestamp_ms=1000,
        )

        self_state = get_self_state(updated)
        assert self_state["fuel"] == 750

    def test_default_fuel_for_new_self(self) -> None:
        """Uses default fuel of 1000 for new self state."""
        state = make_empty_world_state()
        updated = update_self_from_movement_response(
            state,
            tank_id=1,
            x=100,
            y=100,
            team=0,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=1000,
        )

        self_state = get_self_state(updated)
        assert self_state["fuel"] == 1000


class TestUpdateTankFromRegistry:
    """Tests for update_tank_from_registry."""

    def test_adds_new_tank(self) -> None:
        """Adds new tank to state."""
        state = make_empty_world_state()
        updated = update_tank_from_registry(
            state,
            tank_id=42,
            team=TEAM_RED,
            name="Enemy",
            rank=2,
            is_bot=False,
            x=50,
            y=75,
            timestamp_ms=1000,
        )

        assert "42" in updated["tanks"]
        tank = updated["tanks"]["42"]
        assert tank["tank_id"] == 42
        assert tank["team"] == TEAM_RED
        assert tank["name"] == "Enemy"
        assert tank["rank"] == 2
        assert tank["is_bot"] is False
        assert tank["x"] == 50
        assert tank["y"] == 75

    def test_updates_existing_tank(self) -> None:
        """Updates existing tank position."""
        state = make_empty_world_state()
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=0,
            name="Test",
            rank=1,
            is_bot=False,
            x=50,
            y=50,
            timestamp_ms=500,
        )
        updated = update_tank_from_registry(
            state,
            tank_id=42,
            team=0,
            name="Test",
            rank=2,
            is_bot=False,
            x=60,
            y=70,
            timestamp_ms=1000,
        )

        tank = updated["tanks"]["42"]
        assert tank["x"] == 60
        assert tank["y"] == 70
        assert tank["rank"] == 2

    def test_preserves_damage_state(self) -> None:
        """Preserves existing damage state when updating."""
        state = make_empty_world_state()
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=0,
            name="Test",
            rank=1,
            is_bot=False,
            x=50,
            y=50,
            timestamp_ms=500,
        )
        state = update_tank_damage(state, tank_id=42, damage_state=DAMAGE_MEDIUM, timestamp_ms=750)
        updated = update_tank_from_registry(
            state,
            tank_id=42,
            team=0,
            name="Test",
            rank=2,
            is_bot=False,
            x=60,
            y=70,
            timestamp_ms=1000,
        )

        assert updated["tanks"]["42"]["damage_state"] == DAMAGE_MEDIUM

    def test_marks_self_tank(self) -> None:
        """Marks tank as is_self when matching self_state tank_id."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state,
            tank_id=42,
            x=100,
            y=100,
            team=0,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=500,
        )
        updated = update_tank_from_registry(
            state,
            tank_id=42,
            team=0,
            name="Self",
            rank=1,
            is_bot=False,
            x=100,
            y=100,
            timestamp_ms=1000,
        )

        assert updated["tanks"]["42"]["is_self"] is True


class TestUpdateTankDamage:
    """Tests for update_tank_damage."""

    def test_updates_damage_state(self) -> None:
        """Updates tank damage state."""
        state = make_empty_world_state()
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=0,
            name="Test",
            rank=1,
            is_bot=False,
            x=50,
            y=50,
            timestamp_ms=500,
        )
        updated = update_tank_damage(
            state, tank_id=42, damage_state=DAMAGE_CRITICAL, timestamp_ms=1000
        )

        assert updated["tanks"]["42"]["damage_state"] == DAMAGE_CRITICAL

    def test_returns_unchanged_for_unknown_tank(self) -> None:
        """Returns unchanged state for unknown tank ID."""
        state = make_empty_world_state()
        updated = update_tank_damage(
            state, tank_id=999, damage_state=DAMAGE_LIGHT, timestamp_ms=1000
        )

        assert updated is state  # Same reference


class TestUpdateContainerFromRadar:
    """Tests for update_container_from_radar."""

    def test_adds_fuel_container(self) -> None:
        """Adds fuel container from radar."""
        state = make_empty_world_state()
        updated = update_container_from_radar(state, x=100, y=150, volume=500, timestamp_ms=1000)

        key = "100,150"
        assert key in updated["containers"]
        container = updated["containers"][key]
        assert container["x"] == 100
        assert container["y"] == 150
        assert container["is_fuel"] is True
        assert container["volume"] == 500

    def test_adds_equipment_container(self) -> None:
        """Treats -1 volume as equipment."""
        state = make_empty_world_state()
        updated = update_container_from_radar(state, x=50, y=75, volume=-1, timestamp_ms=1000)

        container = updated["containers"]["50,75"]
        assert container["is_fuel"] is False
        assert container["volume"] == 0

    def test_skips_empty_fuel_container(self) -> None:
        """Skips empty fuel containers (volume=0) since they have no contents."""
        state = make_empty_world_state()
        updated = update_container_from_radar(state, x=50, y=75, volume=0, timestamp_ms=1000)

        # Empty fuel containers are skipped, not added
        assert "50,75" not in updated["containers"]


class TestRemoveContainer:
    """Tests for remove_container."""

    def test_removes_existing_container(self) -> None:
        """Removes container from state."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=100, y=100, volume=500, timestamp_ms=500)
        updated = remove_container(state, x=100, y=100, timestamp_ms=1000)

        assert "100,100" not in updated["containers"]

    def test_returns_unchanged_for_nonexistent(self) -> None:
        """Returns unchanged state if container doesn't exist."""
        state = make_empty_world_state()
        updated = remove_container(state, x=100, y=100, timestamp_ms=1000)

        assert updated is state  # Same reference


class TestAddMine:
    """Tests for add_mine."""

    def test_adds_mine(self) -> None:
        """Adds mine to state."""
        state = make_empty_world_state()
        updated = add_mine(state, x=75, y=125, mine_type=1, tank_id=42, team=0, timestamp_ms=1000)

        key = "75,125"
        assert key in updated["mines"]
        mine = updated["mines"][key]
        assert mine["x"] == 75
        assert mine["y"] == 125
        assert mine["mine_type"] == 1
        assert mine["tank_id"] == 42
        assert mine["team"] == 0


class TestAddMineFromRadar:
    """Tests for add_mine_from_radar."""

    def test_adds_radar_mine(self) -> None:
        """Adds mine discovered via radar."""
        from tankpit_bot.state import add_mine_from_radar

        state = make_empty_world_state()
        updated = add_mine_from_radar(state, x=45, y=203, team=0, timestamp_ms=1000)

        key = "45,203"
        assert key in updated["mines"]
        mine = updated["mines"][key]
        assert mine["x"] == 45
        assert mine["y"] == 203
        assert mine["team"] == 0
        assert mine["mine_type"] == 0  # Unknown from radar
        assert mine["tank_id"] == -1  # Unknown from radar

    def test_adds_multiple_radar_mines(self) -> None:
        """Adds multiple radar-discovered mines."""
        from tankpit_bot.state import add_mine_from_radar

        state = make_empty_world_state()
        state = add_mine_from_radar(state, x=45, y=203, team=0, timestamp_ms=1000)
        state = add_mine_from_radar(state, x=46, y=203, team=0, timestamp_ms=1000)
        state = add_mine_from_radar(state, x=47, y=203, team=0, timestamp_ms=1000)

        assert len(state["mines"]) == 3
        assert "45,203" in state["mines"]
        assert "46,203" in state["mines"]
        assert "47,203" in state["mines"]

    def test_adds_mines_from_different_teams(self) -> None:
        """Adds mines from different teams."""
        from tankpit_bot.state import add_mine_from_radar

        state = make_empty_world_state()
        state = add_mine_from_radar(state, x=10, y=10, team=0, timestamp_ms=1000)  # red
        state = add_mine_from_radar(state, x=20, y=20, team=1, timestamp_ms=1000)  # purple
        state = add_mine_from_radar(state, x=30, y=30, team=2, timestamp_ms=1000)  # blue
        state = add_mine_from_radar(state, x=40, y=40, team=3, timestamp_ms=1000)  # orange

        assert state["mines"]["10,10"]["team"] == 0
        assert state["mines"]["20,20"]["team"] == 1
        assert state["mines"]["30,30"]["team"] == 2
        assert state["mines"]["40,40"]["team"] == 3


class TestRemoveMine:
    """Tests for remove_mine."""

    def test_removes_existing_mine(self) -> None:
        """Removes mine from state."""
        state = make_empty_world_state()
        state = add_mine(state, x=75, y=125, mine_type=1, tank_id=42, team=0, timestamp_ms=500)
        updated = remove_mine(state, x=75, y=125, timestamp_ms=1000)

        assert "75,125" not in updated["mines"]

    def test_returns_unchanged_for_nonexistent(self) -> None:
        """Returns unchanged state if mine doesn't exist."""
        state = make_empty_world_state()
        updated = remove_mine(state, x=75, y=125, timestamp_ms=1000)

        assert updated is state  # Same reference


class TestUpdateTerrainFromViewport:
    """Tests for update_terrain_from_viewport."""

    def test_updates_terrain_tiles(self) -> None:
        """Updates terrain from viewport entities."""
        state = make_empty_world_state()
        entities = [
            (0, 0, TERRAIN_GROUND, 0),
            (1, 0, TERRAIN_ROCK_A, 0),
            (2, 0, TERRAIN_FERRY, 0),
        ]
        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=entities, timestamp_ms=1000
        )

        # Entities at viewport-relative coords are converted to world coords
        assert "100,50" in updated["terrain"]
        assert "101,50" in updated["terrain"]
        assert "102,50" in updated["terrain"]

        assert updated["terrain"]["100,50"]["terrain_type"] == TERRAIN_GROUND
        assert updated["terrain"]["101,50"]["terrain_type"] == TERRAIN_ROCK_A
        assert updated["terrain"]["102,50"]["terrain_type"] == TERRAIN_FERRY

    def test_updates_viewport_position(self) -> None:
        """Updates viewport position."""
        state = make_empty_world_state()
        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=[], timestamp_ms=1000
        )

        assert updated["viewport"]["left"] == 100
        assert updated["viewport"]["top"] == 50


class TestRemoveTank:
    """Tests for remove_tank."""

    def test_removes_existing_tank(self) -> None:
        """Removes tank from state."""
        state = make_empty_world_state()
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=0,
            name="Test",
            rank=1,
            is_bot=False,
            x=50,
            y=50,
            timestamp_ms=500,
        )
        updated = remove_tank(state, tank_id=42, timestamp_ms=1000)

        assert "42" not in updated["tanks"]

    def test_returns_unchanged_for_nonexistent(self) -> None:
        """Returns unchanged state if tank doesn't exist."""
        state = make_empty_world_state()
        updated = remove_tank(state, tank_id=999, timestamp_ms=1000)

        assert updated is state  # Same reference


# =============================================================================
# ASCII Rendering Tests
# =============================================================================


class TestTerrainToAscii:
    """Tests for terrain_to_ascii."""

    def test_ground(self) -> None:
        """Ground terrain returns dot."""
        assert terrain_to_ascii(TERRAIN_GROUND) == ASCII_GROUND

    def test_rock_types(self) -> None:
        """Rock terrain types return hash."""
        assert terrain_to_ascii(TERRAIN_ROCK_A) == ASCII_ROCK
        assert terrain_to_ascii(TERRAIN_ROCK_B) == ASCII_ROCK
        assert terrain_to_ascii(TERRAIN_ROCK_AB) == ASCII_ROCK

    def test_ferry(self) -> None:
        """Ferry terrain returns tilde."""
        assert terrain_to_ascii(TERRAIN_FERRY) == ASCII_FERRY

    def test_ferry_rock(self) -> None:
        """Ferry + rock returns hash."""
        assert terrain_to_ascii(TERRAIN_FERRY_ROCK) == ASCII_ROCK

    def test_unknown(self) -> None:
        """Unknown terrain returns question mark."""
        assert terrain_to_ascii(99) == ASCII_UNKNOWN


@pytest.fixture()
def terrain_map(tmp_path: Path) -> TerrainMap:
    """Create a test TerrainMap with uniform ground."""
    gif_path = tmp_path / "test.gif"
    img = Image.new("RGB", (256, 256), (60, 129, 85))  # Dark green = ground
    img.save(gif_path)
    return TerrainMap(gif_path)


class TestRenderWorldAscii:
    """Tests for render_world_ascii."""

    def test_renders_empty_state(self, terrain_map: TerrainMap) -> None:
        """Renders empty state with ground tiles."""
        state = make_empty_world_state()
        output = render_world_ascii(state, terrain_map)

        assert "Viewport:" in output
        assert "Legend:" in output
        assert ASCII_GROUND in output

    def test_renders_self(self, terrain_map: TerrainMap) -> None:
        """Renders self position."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=5, y=5, team=0, rank=0, leaderboard_position=1, timestamp_ms=1000
        )
        output = render_world_ascii(state, terrain_map)

        assert ASCII_SELF in output
        assert "Self:" in output

    def test_renders_fuel_container(self, terrain_map: TerrainMap) -> None:
        """Renders fuel container."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=5, y=5, volume=500, timestamp_ms=1000)
        output = render_world_ascii(state, terrain_map)

        assert ASCII_FUEL in output

    def test_renders_equipment_container(self, terrain_map: TerrainMap) -> None:
        """Renders equipment container."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=5, y=5, volume=-1, timestamp_ms=1000)
        output = render_world_ascii(state, terrain_map)

        assert ASCII_EQUIPMENT in output

    def test_renders_mine(self, terrain_map: TerrainMap) -> None:
        """Renders mine."""
        state = make_empty_world_state()
        state = add_mine(state, x=5, y=5, mine_type=1, tank_id=42, team=0, timestamp_ms=1000)
        output = render_world_ascii(state, terrain_map)

        assert ASCII_MINE in output

    def test_renders_enemy_tank(self, terrain_map: TerrainMap) -> None:
        """Renders enemy tank."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state,
            tank_id=1,
            x=0,
            y=0,
            team=TEAM_BLUE,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=500,
        )
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=TEAM_RED,
            name="Enemy",
            rank=1,
            is_bot=False,
            x=5,
            y=5,
            timestamp_ms=1000,
        )
        output = render_world_ascii(state, terrain_map)

        assert ASCII_ENEMY in output

    def test_renders_ally_tank(self, terrain_map: TerrainMap) -> None:
        """Renders ally tank."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state,
            tank_id=1,
            x=0,
            y=0,
            team=TEAM_BLUE,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=500,
        )
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=TEAM_BLUE,
            name="Ally",
            rank=1,
            is_bot=False,
            x=5,
            y=5,
            timestamp_ms=1000,
        )
        output = render_world_ascii(state, terrain_map)

        assert ASCII_ALLY in output

    def test_renders_terrain_from_map(self, terrain_map: TerrainMap) -> None:
        """Renders terrain from TerrainMap."""
        state = make_empty_world_state()
        output = render_world_ascii(state, terrain_map)

        assert ASCII_GROUND in output

    def test_self_takes_priority(self, terrain_map: TerrainMap) -> None:
        """Self position takes priority over other entities."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state,
            tank_id=1,
            x=5,
            y=5,
            team=0,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=500,
        )
        # Add container at same position
        state = update_container_from_radar(state, x=5, y=5, volume=500, timestamp_ms=1000)
        output = render_world_ascii(state, terrain_map)

        # Should show @ not F at position 5,5
        assert ASCII_SELF in output

    def test_shows_tank_counts(self, terrain_map: TerrainMap) -> None:
        """Shows tank count summary."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state,
            tank_id=1,
            x=0,
            y=0,
            team=TEAM_BLUE,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=500,
        )
        state = update_tank_from_registry(
            state,
            tank_id=10,
            team=TEAM_BLUE,
            name="Ally",
            rank=1,
            is_bot=False,
            x=5,
            y=5,
            timestamp_ms=600,
        )
        state = update_tank_from_registry(
            state,
            tank_id=20,
            team=TEAM_RED,
            name="Enemy",
            rank=1,
            is_bot=False,
            x=6,
            y=6,
            timestamp_ms=700,
        )
        output = render_world_ascii(state, terrain_map)

        assert "Tanks:" in output
        assert "allies=" in output
        assert "enemies=" in output
