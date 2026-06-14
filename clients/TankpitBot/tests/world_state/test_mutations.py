"""Tests for state mutation functions."""

from tankpit_bot.state import (
    DAMAGE_CRITICAL,
    DAMAGE_LIGHT,
    DAMAGE_MEDIUM,
    TEAM_BLUE,
    TEAM_RED,
    TERRAIN_FERRY,
    TERRAIN_GROUND,
    TERRAIN_ROCK_A,
    SelfStateDict,
    WorldStateDict,
    add_mine,
    add_mine_from_radar,
    coord_key,
    make_container_state,
    make_empty_world_state,
    pickup_container,
    remove_container,
    remove_mine,
    remove_tank,
    set_self_fuel,
    update_container_from_radar,
    update_self_from_movement_response,
    update_self_fuel,
    update_tank_damage,
    update_tank_from_registry,
    update_terrain_from_viewport,
)
from tests.world_state.helpers import get_self_state


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
            scanned_viewports=state["scanned_viewports"],
            map_fuel_dots=state["map_fuel_dots"],
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
        """Uses default fuel of 0 for new self state (real fuel comes from TankStatusSync)."""
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
        assert self_state["fuel"] == 0


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
            source="viewport",
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
            source="viewport",
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
            source="viewport",
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
            source="viewport",
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
            source="viewport",
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
            source="viewport",
            timestamp_ms=1000,
        )

        assert updated["tanks"]["42"]["is_self"] is True


class TestWirePresenceFunnel:
    """Tests for the ``wire_present`` rule in update_tank_from_registry."""

    def test_wire_present_advances_last_wire_seen_ms(self) -> None:
        """A wire-present update stamps last_wire_seen_ms to the timestamp."""
        state = make_empty_world_state()
        updated = update_tank_from_registry(
            state,
            tank_id=42,
            team=1,
            name="Enemy",
            rank=1,
            is_bot=False,
            x=50,
            y=50,
            source="viewport",
            timestamp_ms=1000,
            wire_present=True,
        )

        assert updated["tanks"]["42"]["last_wire_seen_ms"] == 1000

    def test_new_non_wire_tank_has_zero_last_wire_seen_ms(self) -> None:
        """A first sighting from a non-wire source never vouches presence."""
        state = make_empty_world_state()
        updated = update_tank_from_registry(
            state,
            tank_id=42,
            team=1,
            name="Ghost",
            rank=1,
            is_bot=False,
            x=34,
            y=96,
            source="world_state",
            timestamp_ms=1000,
            wire_present=False,
        )

        assert updated["tanks"]["42"]["last_wire_seen_ms"] == 0
        assert updated["tanks"]["42"]["timestamp_ms"] == 1000

    def test_map_refresh_preserves_wire_stamp_while_advancing_timestamp(self) -> None:
        """A map (non-wire) refresh advances timestamp but freezes the wire stamp.

        This is the exact ghost mechanism: a tank confirmed on the wire
        leaves, the map keeps re-listing it, and only ``timestamp_ms``
        advances. ``last_wire_seen_ms`` must stay frozen at the last real
        sighting so the kill gate can tell the afterimage apart.
        """
        state = make_empty_world_state()
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=1,
            name="Enemy",
            rank=1,
            is_bot=False,
            x=50,
            y=50,
            source="viewport",
            timestamp_ms=1000,
            wire_present=True,
        )

        refreshed = update_tank_from_registry(
            state,
            tank_id=42,
            team=1,
            name="Enemy",
            rank=1,
            is_bot=False,
            x=50,
            y=50,
            source="world_state",
            timestamp_ms=9000,
            wire_present=False,
        )

        assert refreshed["tanks"]["42"]["timestamp_ms"] == 9000
        assert refreshed["tanks"]["42"]["last_wire_seen_ms"] == 1000

    def test_returning_wire_update_readvances_wire_stamp(self) -> None:
        """A fresh wire sighting after a map-only gap re-stamps presence."""
        state = make_empty_world_state()
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=1,
            name="Enemy",
            rank=1,
            is_bot=False,
            x=50,
            y=50,
            source="viewport",
            timestamp_ms=1000,
            wire_present=True,
        )
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=1,
            name="Enemy",
            rank=1,
            is_bot=False,
            x=50,
            y=50,
            source="world_state",
            timestamp_ms=9000,
            wire_present=False,
        )

        returned = update_tank_from_registry(
            state,
            tank_id=42,
            team=1,
            name="Enemy",
            rank=1,
            is_bot=False,
            x=51,
            y=50,
            source="viewport",
            timestamp_ms=12000,
            wire_present=True,
        )

        assert returned["tanks"]["42"]["last_wire_seen_ms"] == 12000


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
            source="viewport",
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

    def test_removes_existing_fuel_container_when_radar_reports_empty(self) -> None:
        """Radar volume=0 clears an already-known fuel container."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=50, y=75, volume=500, timestamp_ms=500)

        updated = update_container_from_radar(state, x=50, y=75, volume=0, timestamp_ms=1000)

        assert "50,75" not in updated["containers"]

    def test_preserves_failed_pickups_when_refreshing_existing_container(self) -> None:
        """Refreshing a known container keeps failed pickup memory intact."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=100, y=150, volume=500, timestamp_ms=500)
        container = state["containers"]["100,150"]
        state["containers"]["100,150"] = make_container_state(
            x=container["x"],
            y=container["y"],
            is_fuel=container["is_fuel"],
            volume=container["volume"],
            timestamp_ms=container["timestamp_ms"],
            failed_pickups=2,
        )

        updated = update_container_from_radar(state, x=100, y=150, volume=500, timestamp_ms=1000)

        assert updated["containers"]["100,150"]["failed_pickups"] == 2


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
            (0, 0, TERRAIN_GROUND, 0, 255),
            (1, 0, TERRAIN_ROCK_A, -1, 3),
            (2, 0, TERRAIN_FERRY, 25, 255),
        ]
        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=entities, timestamp_ms=1000
        )

        assert "99,49" in updated["terrain"]
        assert "100,49" in updated["terrain"]
        assert "101,49" in updated["terrain"]

        assert updated["terrain"]["99,49"]["terrain_type"] == TERRAIN_GROUND
        assert updated["terrain"]["100,49"]["terrain_type"] == TERRAIN_ROCK_A
        assert updated["terrain"]["101,49"]["terrain_type"] == TERRAIN_FERRY
        assert updated["terrain"]["100,49"]["cache_value"] == -1
        assert updated["terrain"]["100,49"]["overlay_value"] == 3

    def test_updates_viewport_position(self) -> None:
        """Updates viewport position."""
        state = make_empty_world_state()
        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=[], timestamp_ms=1000
        )

        assert updated["viewport"]["left"] == 100
        assert updated["viewport"]["top"] == 50

    def test_marks_viewport_as_confirmed(self) -> None:
        """Visible viewport updates confirm local resource coverage."""
        state = make_empty_world_state()

        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=[], timestamp_ms=1000
        )

        assert updated["scanned_viewports"]["100,50"] == 1000


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
            source="viewport",
            timestamp_ms=500,
        )
        updated = remove_tank(state, tank_id=42, timestamp_ms=1000)

        assert "42" not in updated["tanks"]

    def test_returns_unchanged_for_nonexistent(self) -> None:
        """Returns unchanged state if tank doesn't exist."""
        state = make_empty_world_state()
        updated = remove_tank(state, tank_id=999, timestamp_ms=1000)

        assert updated is state  # Same reference


class TestUpdateSelfFuel:
    """Tests for update_self_fuel mutation."""

    def test_update_self_fuel_no_self_state(self) -> None:
        """Returns unchanged state when self_state is None."""
        state = make_empty_world_state()
        result = update_self_fuel(state, 50, 1000)
        assert result["self_state"] is None
        assert result is state

    def test_update_self_fuel_adds_fuel(self) -> None:
        """Adds fuel to existing self_state."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        initial_fuel = get_self_state(state)["fuel"]

        result = update_self_fuel(state, 50, 1000)
        assert get_self_state(result)["fuel"] == initial_fuel + 50

    def test_update_self_fuel_subtracts_fuel(self) -> None:
        """Subtracts fuel (damage) from self_state, clamped to zero."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        current_fuel = get_self_state(state)["fuel"]

        # Subtract more than available - should clamp to 0
        result = update_self_fuel(state, -(current_fuel + 200), 700)
        assert get_self_state(result)["fuel"] == 0


class TestSetSelfFuel:
    """Tests for set_self_fuel mutation."""

    def test_set_self_fuel_no_self_state(self) -> None:
        """Returns unchanged state when self_state is None."""
        state = make_empty_world_state()
        result = set_self_fuel(state, 50, 1000)
        assert result["self_state"] is None
        assert result is state

    def test_set_self_fuel_sets_absolute_value(self) -> None:
        """Sets fuel to absolute value."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )

        result = set_self_fuel(state, 250, 1000)
        assert get_self_state(result)["fuel"] == 250

    def test_set_self_fuel_clamps_negative(self) -> None:
        """Clamps negative values to zero."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )

        result = set_self_fuel(state, -10, 1000)
        assert get_self_state(result)["fuel"] == 0


class TestPickupContainer:
    """Tests for pickup_container mutation."""

    def test_pickup_container_removes_container(self) -> None:
        """Removes container from world state."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=50, y=60, volume=100, timestamp_ms=500)
        assert coord_key(50, 60) in state["containers"]

        result = pickup_container(state, 50, 60, 1000)
        assert coord_key(50, 60) not in result["containers"]

    def test_pickup_fuel_container_adds_fuel(self) -> None:
        """Picking up fuel container adds fuel to self."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        initial_fuel = get_self_state(state)["fuel"]

        # Add a fuel container with volume 100
        state = update_container_from_radar(state, x=50, y=60, volume=100, timestamp_ms=600)

        result = pickup_container(state, 50, 60, 700)
        assert get_self_state(result)["fuel"] == initial_fuel + 100
        assert coord_key(50, 60) not in result["containers"]

    def test_pickup_equipment_container_no_fuel_change(self) -> None:
        """Picking up equipment container does not add fuel."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        initial_fuel = get_self_state(state)["fuel"]

        # Add an equipment container (volume=-1)
        state = update_container_from_radar(state, x=50, y=60, volume=-1, timestamp_ms=600)

        result = pickup_container(state, 50, 60, 700)
        # Fuel unchanged since equipment containers have is_fuel=False
        assert get_self_state(result)["fuel"] == initial_fuel
        assert coord_key(50, 60) not in result["containers"]

    def test_pickup_nonexistent_container(self) -> None:
        """Picking up nonexistent container just returns state with no change."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        initial_fuel = get_self_state(state)["fuel"]

        result = pickup_container(state, 99, 99, 700)
        assert get_self_state(result)["fuel"] == initial_fuel

    def test_pickup_container_no_self_state(self) -> None:
        """Picking up container without self_state still removes container."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=50, y=60, volume=100, timestamp_ms=500)

        result = pickup_container(state, 50, 60, 700)
        assert coord_key(50, 60) not in result["containers"]
        assert result["self_state"] is None
