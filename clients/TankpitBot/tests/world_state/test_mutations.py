"""Tests for container, mine, and terrain mutations.

``test_mutations.py`` was 828 lines; self and tank mutations are now
siblings, mirroring the ``state/*_mutations.py`` split.
"""

from tankpit_bot.state import (
    add_mine,
    add_mine_from_radar,
    coord_key,
    make_container_state,
    make_empty_world_state,
    pickup_container,
    remove_container,
    remove_mine,
    update_container_from_radar,
    update_self_from_movement_response,
    update_terrain_from_viewport,
)
from tankpit_bot.types.constants import (
    TERRAIN_BLOCK_BRIDGE,
    TERRAIN_FERRY,
    TERRAIN_GROUND,
)
from tests.world_state.helpers import get_self_state


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


class TestPickupContainer:
    """Tests for pickup_container mutation."""

    def test_pickup_container_removes_container(self) -> None:
        """Removes container from world state."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=50, y=60, volume=100, timestamp_ms=500)
        assert coord_key(50, 60) in state["containers"]

        result = pickup_container(state, 50, 60, 1000)
        assert coord_key(50, 60) not in result["containers"]

    def test_pickup_container_does_not_modify_self_fuel(self) -> None:
        """``pickup_container`` only touches the container registry, not fuel.

        Fuel updates flow through the wire's absolute-fuel messages
        (0x44 FuelGain / 0x2E TankStatusSync / 0x64 FuelDeposit), which
        call :func:`set_self_fuel` separately. Adding ``transferred``
        fuel here on top of the wire's absolute total double-counts the
        pickup -- a 438-volume container produced a +438 ghost beyond
        the wire's already-correct 633 fuel in live run 20260623-0035
        before this branch was removed.
        """
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        initial_fuel = get_self_state(state)["fuel"]
        state = update_container_from_radar(state, x=50, y=60, volume=100, timestamp_ms=600)

        result = pickup_container(state, 50, 60, 700)

        assert get_self_state(result)["fuel"] == initial_fuel
        assert coord_key(50, 60) not in result["containers"]

    def test_pickup_equipment_container_no_fuel_change(self) -> None:
        """Equipment-container pickup also leaves fuel untouched."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=10, y=10, team=0, rank=0, leaderboard_position=1, timestamp_ms=500
        )
        initial_fuel = get_self_state(state)["fuel"]
        state = update_container_from_radar(state, x=50, y=60, volume=-1, timestamp_ms=600)

        result = pickup_container(state, 50, 60, 700)

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
            (1, 0, TERRAIN_BLOCK_BRIDGE, -1, 3),
            (2, 0, TERRAIN_FERRY, 25, 255),
        ]
        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=entities, timestamp_ms=1000
        )

        assert "99,49" in updated["terrain"]
        assert "100,49" in updated["terrain"]
        assert "101,49" in updated["terrain"]

        assert updated["terrain"]["99,49"]["terrain_type"] == TERRAIN_GROUND
        assert updated["terrain"]["100,49"]["terrain_type"] == TERRAIN_BLOCK_BRIDGE
        assert updated["terrain"]["101,49"]["terrain_type"] == TERRAIN_FERRY

    def test_updates_viewport_position(self) -> None:
        """Updates viewport position."""
        state = make_empty_world_state()
        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=[], timestamp_ms=1000
        )

        assert updated["viewport"]["left"] == 100
        assert updated["viewport"]["top"] == 50

    def test_does_not_touch_scan_coverage(self) -> None:
        """0x5A viewport patches carry terrain only -- never scan coverage.

        Only the wire-side radar handler writes ``scanned_tiles`` -- a
        viewport patch confirms terrain but says nothing about whether
        containers / mines on those tiles were revealed.
        """
        state = make_empty_world_state()

        updated = update_terrain_from_viewport(
            state, viewport_left=100, viewport_top=50, entities=[], timestamp_ms=1000
        )

        assert updated["scanned_tiles"] == {}
