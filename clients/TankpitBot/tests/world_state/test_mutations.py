"""Tests for state mutation functions."""

from tankpit_bot.state import (
    TEAM_BLUE,
    TERRAIN_FERRY,
    TERRAIN_GROUND,
    TERRAIN_ROCK_A,
    SelfStateDict,
    WorldStateDict,
    add_mine,
    add_mine_from_radar,
    coord_key,
    deactivate_tank,
    make_container_state,
    make_empty_world_state,
    pickup_container,
    remove_container,
    remove_mine,
    remove_tank,
    set_self_fuel,
    set_self_rank,
    update_container_from_radar,
    update_self_from_movement_response,
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


# TestUpdateTankFromRegistry, TestWirePresenceFunnel, and
# TestUpdateTankDamage were deleted 2026-06-19 with the freshness-model
# refactor. The deleted mutators are replaced by apply_tank_observation;
# its contract -- including the freshness-funnel rules these classes
# pinned -- lives in tests/world_state/test_tank_observation.py.


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


class TestIncrementContainerFailedPickups:
    """Tests for increment_container_failed_pickups (central state-layer mutator)."""

    def test_advances_failed_pickups_by_one(self) -> None:
        """An existing container's failed_pickups counter advances by 1."""
        from tankpit_bot.state import increment_container_failed_pickups

        state = make_empty_world_state()
        state = update_container_from_radar(state, x=100, y=100, volume=500, timestamp_ms=500)
        before = state["containers"]["100,100"]["failed_pickups"]

        updated = increment_container_failed_pickups(state, x=100, y=100)

        assert updated["containers"]["100,100"]["failed_pickups"] == before + 1

    def test_preserves_container_timestamp(self) -> None:
        """The bump must NOT advance the container's freshness timestamp.

        ``failed_pickups`` is a planner-deprioritization counter, not a
        freshness signal. If this test ever flips, planning logic that
        relies on container freshness will silently regress.
        """
        from tankpit_bot.state import increment_container_failed_pickups

        state = make_empty_world_state()
        state = update_container_from_radar(state, x=100, y=100, volume=500, timestamp_ms=500)
        container_ts_before = state["containers"]["100,100"]["timestamp_ms"]

        updated = increment_container_failed_pickups(state, x=100, y=100)

        assert updated["containers"]["100,100"]["timestamp_ms"] == container_ts_before

    def test_returns_unchanged_for_missing_container(self) -> None:
        """No container at ``(x, y)`` -> returned state IS the input state."""
        from tankpit_bot.state import increment_container_failed_pickups

        state = make_empty_world_state()
        updated = increment_container_failed_pickups(state, x=200, y=300)

        assert updated is state


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


class TestAddMineFromRadarPreservesWireFields:
    """Radar refresh must NOT clobber wire-known mine_type / tank_id.

    Radar 3-byte mine entries (per V.O / 0x4F tunneled, see
    wiki/pages/v-table-complete.md) carry only x, y, team. Wire
    MinePlacement (V.K / 0x4B per Dg.h) carries mine_type and tank_id.
    A radar refresh of a wire-placed mine MUST preserve the
    wire-richer fields. Locked here as a contract.
    """

    def test_radar_refresh_preserves_wire_mine_type(self) -> None:
        """A radar refresh keeps the wire-placed mine_type intact."""
        state = make_empty_world_state()
        state = add_mine(state, x=50, y=50, mine_type=2, tank_id=42, team=1, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["mine_type"] == 2

    def test_radar_refresh_preserves_wire_tank_id(self) -> None:
        """A radar refresh keeps the wire-placed placer tank_id intact."""
        state = make_empty_world_state()
        state = add_mine(state, x=50, y=50, mine_type=2, tank_id=42, team=1, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["tank_id"] == 42

    def test_radar_refresh_preserves_wire_source_label(self) -> None:
        """A wire-known mine refreshed by radar stays marked as viewport-sourced.

        The mine_type and tank_id remain wire-richer values, so the
        source label that flags them as wire-richer must stick.
        """
        state = make_empty_world_state()
        state = add_mine(state, x=50, y=50, mine_type=2, tank_id=42, team=1, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["source"] == "viewport"

    def test_radar_refresh_advances_timestamp(self) -> None:
        """A radar refresh still advances the mine's timestamp."""
        state = make_empty_world_state()
        state = add_mine(state, x=50, y=50, mine_type=2, tank_id=42, team=1, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["timestamp_ms"] == 1500

    def test_radar_refresh_updates_team_when_radar_disagrees(self) -> None:
        """When radar reports a different team for an existing mine,
        the radar value wins. Team cannot legally change for an
        undetonated mine, so a discrepancy indicates the wire team
        field went stale and is re-synced from radar.
        """
        state = make_empty_world_state()
        state = add_mine(state, x=50, y=50, mine_type=2, tank_id=42, team=0, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=3, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["team"] == 3

    def test_radar_refresh_of_radar_mine_preserves_radar_source(self) -> None:
        """Radar-then-radar refresh keeps source='radar' (no wire history)."""
        state = make_empty_world_state()
        state = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=500)

        refreshed = add_mine_from_radar(state, x=50, y=50, team=1, timestamp_ms=1500)

        assert refreshed["mines"]["50,50"]["source"] == "radar"
        assert refreshed["mines"]["50,50"]["mine_type"] == 0
        assert refreshed["mines"]["50,50"]["tank_id"] == -1


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
        """Deletes the tank from the registry.

        ``remove_tank`` is the 0x58 TankRemove handler. 0x58 doesn't
        mean "tank died" -- it means the server stopped broadcasting
        per-tank updates to this client (verified 2026-06-20
        ghost_observe capture: orange-5 got 5 TankRemove events across
        2 actual kills). Simpler correct behaviour: drop the tank from
        the registry. The next MapData or per-tank wire re-adds it at
        its current position with ``liveness="alive"``.
        """
        from tankpit_bot.state import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation

        state = make_empty_world_state()
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=42,
                timestamp_ms=500,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(50, 50),
                team=0,
                rank=1,
                name="Test",
                is_bot=False,
            ),
        )
        assert state["tanks"]["42"]["liveness"] == "alive"

        updated = remove_tank(state, tank_id=42, timestamp_ms=1000)

        assert "42" not in updated["tanks"]
        assert updated["timestamp_ms"] == 1000

    def test_returns_unchanged_for_nonexistent(self) -> None:
        """Returns unchanged state if tank doesn't exist."""
        state = make_empty_world_state()
        updated = remove_tank(state, tank_id=999, timestamp_ms=1000)

        assert updated is state  # Same reference


class TestDeactivateTank:
    """Tests for deactivate_tank (0x41 corpse-window handler)."""

    def test_marks_tank_deactivated(self) -> None:
        """Existing tank flips to ``liveness="deactivated"`` and keeps tile.

        Replays the 2026-06-20 ghost_visual kill cycle at the
        world-state layer: TankEntry establishes orange-8 at
        (170, 174); Deactivation marks it ``deactivated`` while
        preserving the death tile (the bot still reasons about that
        tile for mines, fuel deposits, etc.).
        """
        from tankpit_bot.state import apply_tank_observation
        from tankpit_bot.state.types import make_tank_observation

        state = make_empty_world_state()
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=534,
                timestamp_ms=500,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(170, 174),
                team=3,
                rank=2,
                name="orange-8",
                is_bot=True,
            ),
        )
        assert state["tanks"]["534"]["liveness"] == "alive"

        updated = deactivate_tank(state, tank_id=534, timestamp_ms=1000)

        tank = updated["tanks"]["534"]
        assert tank["liveness"] == "deactivated"
        assert tank["x"] == 170
        assert tank["y"] == 174
        assert tank["timestamp_ms"] == 1000

    def test_returns_unchanged_for_nonexistent(self) -> None:
        """Deactivating an unknown tank id is a no-op."""
        state = make_empty_world_state()
        updated = deactivate_tank(state, tank_id=999, timestamp_ms=1000)
        assert updated is state


# TestUpdateSelfFuel was deleted 2026-06-19 with the additive-delta
# update_self_fuel mutator. Production wire fuel messages all funnel
# through set_self_fuel (absolute value); the delta variant had no
# callers in src/.


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


class TestSetSelfRank:
    """Tests for set_self_rank mutation (driven by 0x2B Rf Promotion)."""

    def test_returns_unchanged_when_no_self_state(self) -> None:
        """No-op until self has joined: rank can't precede a self_state."""
        state = make_empty_world_state()
        result = set_self_rank(state, 5, 1000)
        assert result["self_state"] is None
        assert result is state

    def test_sets_absolute_rank_and_preserves_other_fields(self) -> None:
        """Promotion lifts rank only; team/position/fuel/lb stay intact."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=42, x=10, y=20, team=1, rank=2, leaderboard_position=7, timestamp_ms=500
        )
        state = set_self_fuel(state, 175, 600)

        result = set_self_rank(state, 5, 1000)

        promoted = get_self_state(result)
        assert promoted["rank"] == 5
        assert promoted["tank_id"] == 42
        assert promoted["x"] == 10
        assert promoted["y"] == 20
        assert promoted["team"] == 1
        assert promoted["fuel"] == 175
        assert promoted["leaderboard_position"] == 7

    def test_timestamp_advances(self) -> None:
        """Mutation stamps the new world-state timestamp."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=0, y=0, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        result = set_self_rank(state, 3, 1234)
        assert result["timestamp_ms"] == 1234


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
