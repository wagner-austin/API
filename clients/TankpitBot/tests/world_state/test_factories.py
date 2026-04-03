"""Tests for state factory functions."""

from tankpit_bot.state import (
    DAMAGE_FULL,
    DAMAGE_LIGHT,
    TEAM_BLUE,
    TEAM_PURPLE,
    TEAM_RED,
    TERRAIN_GROUND,
    TERRAIN_ROCK_A,
    make_container_state,
    make_empty_world_state,
    make_mine_state,
    make_self_state,
    make_tank_state,
    make_terrain_tile,
)


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
        """Default viewport is 16x16 at origin."""
        state = make_empty_world_state()
        vp = state["viewport"]

        assert vp["left"] == 0
        assert vp["top"] == 0
        assert vp["width"] == 16
        assert vp["height"] == 16


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
        tile = make_terrain_tile(
            x=10,
            y=20,
            terrain_type=TERRAIN_ROCK_A,
            cache_value=0,
            overlay_value=255,
        )

        assert tile["x"] == 10
        assert tile["y"] == 20
        assert tile["terrain_type"] == TERRAIN_ROCK_A
        assert tile["cache_value"] == 0
        assert tile["overlay_value"] == 255

    def test_tile_with_equipment_cache(self) -> None:
        """Creates tile with equipment cache marker."""
        tile = make_terrain_tile(
            x=15,
            y=25,
            terrain_type=TERRAIN_GROUND,
            cache_value=-1,
            overlay_value=4,
        )

        assert tile["cache_value"] == -1
        assert tile["overlay_value"] == 4


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
