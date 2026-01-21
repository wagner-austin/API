"""Tests for state encode functions."""

from tankpit_bot.state import (
    ViewportStateDict,
    add_mine,
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
    update_container_from_radar,
)


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
